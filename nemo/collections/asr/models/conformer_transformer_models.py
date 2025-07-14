"""
Conformer-Transformer model for streaming ASR.
This model combines Conformer encoder with Transformer decoder for sequence-to-sequence ASR.
"""

import copy
import json
import os
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset, AudioToBPEDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.modules.transformer.transformer_decoder_asr import TransformerDecoderASR
from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin, ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogitsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['EncDecConformerTransformerModel']


class EncDecConformerTransformerModel(ASRModel, ASRBPEMixin, ASRModuleMixin, ExportableEncDecModel):
    """
    Encoder-Decoder Conformer-Transformer model for streaming ASR.
    
    This model combines a Conformer encoder with cache-aware streaming support
    and a Transformer decoder, suitable for sequence-to-sequence ASR tasks.
    
    Args:
        cfg: Model configuration from Hydra/OmegaConf
        trainer: PyTorch Lightning trainer instance
    """
    
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to OmegaConf DictConfig if needed
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
            
        # Setup tokenizer first (from ASRBPEMixin)
        self.setup_tokenizer(cfg.tokenizer)
        
        # Initialize vocabulary size
        vocab_size = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else self.tokenizer.vocab_size
        
        # Setup special tokens
        self.blank_id = vocab_size  # For potential CTC use
        if hasattr(self.tokenizer, 'pad_id'):
            self.pad_id = self.tokenizer.pad_id
            self.bos_id = self.tokenizer.bos_id
            self.eos_id = self.tokenizer.eos_id
        else:
            # Add special tokens if not in tokenizer
            self.pad_id = vocab_size
            self.bos_id = vocab_size + 1
            self.eos_id = vocab_size + 2
            vocab_size += 3
            
        # Call parent init
        super().__init__(cfg=cfg, trainer=trainer)
        
        # Setup preprocessor
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        
        # Setup data augmentation
        self.spec_augmentation = self.from_config_dict(cfg.get('spec_augment'))
        
        # Setup Conformer encoder with cache-aware streaming
        encoder_cfg = cfg.encoder
        with open_dict(encoder_cfg):
            # Ensure streaming configuration
            encoder_cfg.self_attention_model = encoder_cfg.get('self_attention_model', 'rel_pos')
            encoder_cfg.att_context_style = encoder_cfg.get('att_context_style', 'chunked_limited')
            
        self.encoder = ConformerEncoder(
            feat_in=cfg.preprocessor.features,
            n_layers=encoder_cfg.n_layers,
            d_model=encoder_cfg.d_model,
            feat_out=encoder_cfg.get('feat_out', -1),
            subsampling=encoder_cfg.subsampling,
            subsampling_factor=encoder_cfg.subsampling_factor,
            subsampling_conv_channels=encoder_cfg.get('subsampling_conv_channels', -1),
            causal_downsampling=encoder_cfg.get('causal_downsampling', True),
            ff_expansion_factor=encoder_cfg.ff_expansion_factor,
            self_attention_model=encoder_cfg.self_attention_model,
            n_heads=encoder_cfg.n_heads,
            att_context_size=encoder_cfg.att_context_size,
            att_context_style=encoder_cfg.att_context_style,
            xscaling=encoder_cfg.get('xscaling', True),
            untie_biases=encoder_cfg.get('untie_biases', True),
            pos_emb_max_len=encoder_cfg.get('pos_emb_max_len', 5000),
            conv_kernel_size=encoder_cfg.conv_kernel_size,
            conv_norm_type=encoder_cfg.get('conv_norm_type', 'batch_norm'),
            conv_context_size=encoder_cfg.get('conv_context_size', None),
            dropout=encoder_cfg.dropout,
            dropout_pre_encoder=encoder_cfg.get('dropout_pre_encoder', 0.1),
            dropout_emb=encoder_cfg.get('dropout_emb', 0.0),
            dropout_att=encoder_cfg.get('dropout_att', 0.1),
        )
        
        # Get encoder output dimension
        encoder_output_dim = self.encoder._feat_out
            
        # Setup Transformer decoder
        decoder_cfg = cfg.decoder
        self.decoder = TransformerDecoderASR(
            vocab_size=vocab_size,
            d_model=decoder_cfg.get('d_model', encoder_output_dim),
            n_layers=decoder_cfg.n_layers,
            n_heads=decoder_cfg.n_heads,
            d_ff=decoder_cfg.get('d_ff', 2048),
            dropout=decoder_cfg.get('dropout', 0.1),
            max_seq_length=decoder_cfg.get('max_seq_length', 5000),
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        
        # Loss function
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction='mean')
        
        # Metrics
        self.wer = WER(
            vocabulary=self.tokenizer.vocab if hasattr(self.tokenizer, 'vocab') else None,
            batch_dim_index=0,
            use_cer=False,
            ctc_decode=False,
            dist_sync_on_step=True,
            log_prediction=True,
        )
        
        # Setup streaming configuration
        self.streaming_cfg = cfg.get('streaming', DictConfig({}))
        self.local_agreement_threshold = self.streaming_cfg.get('local_agreement_threshold', 0.5)
        self.chunk_size = self.streaming_cfg.get('chunk_size', 0.5)
        self.max_decode_length = self.streaming_cfg.get('max_decode_length', 100)
        
        # For training/validation metrics
        self._loss_metric = GlobalAverageLossMetric(dist_sync_on_step=False)
        
    def setup_tokenizer(self, cfg: DictConfig):
        """Setup tokenizer from config (uses ASRBPEMixin)"""
        self._setup_tokenizer(cfg)
        
    @property
    def input_types(self):
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal()),
            "input_signal_length": NeuralType(('B',), LengthsType()),
        }
        
    @property
    def output_types(self):
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "encoded_lengths": NeuralType(('B',), LengthsType()),
        }
        
    @typecheck()
    def forward(
        self,
        input_signal,
        input_signal_length,
        transcript=None,
        transcript_length=None,
    ):
        """
        Forward pass through encoder and decoder
        
        Args:
            input_signal: Raw audio signal [B, T]
            input_signal_length: Length of each audio signal [B]
            transcript: Target transcript tokens [B, U] (optional, for training)
            transcript_length: Length of each transcript [B] (optional)
            
        Returns:
            Dictionary containing logits and encoded lengths
        """
        # Preprocessor
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )
        
        # Spec augmentation during training
        if self.training and self.spec_augmentation is not None:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)
            
        # Encoder forward
        encoder_output, encoder_length = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )
        
        # Decoder forward
        if transcript is not None:
            # Training or teacher forcing mode
            decoder_output = self.decoder(
                encoder_output=encoder_output,
                encoder_lengths=encoder_length,
                targets=transcript,
                target_lengths=transcript_length,
            )
        else:
            # Inference mode - return encoder outputs for decoding
            return encoder_output, encoder_length
            
        return decoder_output['logits'], encoder_length
        
    def training_step(self, batch, batch_idx):
        """Training step"""
        input_signal, input_signal_length, transcript, transcript_length = batch
        
        # Forward pass
        logits, _ = self.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            transcript=transcript,
            transcript_length=transcript_length,
        )
        
        # Calculate loss
        loss = self.loss(
            logits.transpose(1, 2),  # [B, V, U]
            transcript  # [B, U]
        )
        
        # Update metrics
        self._loss_metric.update(loss.detach(), transcript.size(0))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], on_step=True, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step"""
        input_signal, input_signal_length, transcript, transcript_length = batch
        
        # Forward pass
        logits, _ = self.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            transcript=transcript,
            transcript_length=transcript_length,
        )
        
        # Calculate loss
        loss = self.loss(
            logits.transpose(1, 2),  # [B, V, U]
            transcript  # [B, U]
        )
        
        # Update metrics
        self._loss_metric.update(loss.detach(), transcript.size(0))
        
        # Decode for WER calculation
        predictions = logits.argmax(dim=-1)
        predictions_text = self.tokenizer.ids_to_text(predictions[0].cpu().tolist())
        transcript_text = self.tokenizer.ids_to_text(transcript[0].cpu().tolist())
        
        self.wer.update(predictions_text, transcript_text)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss
        
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Compute and log WER
        wer_value = self.wer.compute()
        self.log('val_wer', wer_value, on_epoch=True, logger=True, prog_bar=True)
        self.wer.reset()
        
        # Reset loss metric
        avg_loss = self._loss_metric.compute()
        self.log('val_loss_epoch', avg_loss, on_epoch=True, logger=True)
        self._loss_metric.reset()
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step"""
        return self.validation_step(batch, batch_idx, dataloader_idx)
        
    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        # Compute and log WER
        wer_value = self.wer.compute()
        self.log('test_wer', wer_value, on_epoch=True, logger=True)
        self.wer.reset()
        
        # Reset loss metric
        avg_loss = self._loss_metric.compute()
        self.log('test_loss', avg_loss, on_epoch=True, logger=True)
        self._loss_metric.reset()
        
    def _setup_dataloader_from_config(self, config: Optional[DictConfig]):
        """Setup dataloader from config"""
        if config is None:
            return None
            
        shuffle = config.get('shuffle', False)
        
        # Instantiate dataset
        dataset_class = AudioToBPEDataset if hasattr(self.tokenizer, 'vocab') else AudioToCharDataset
        dataset = dataset_class(
            manifest_filepath=config.manifest_filepath,
            tokenizer=self.tokenizer,
            sample_rate=config.sample_rate,
            int_values=config.get('int_values', False),
            augmentor=None,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', False),
            use_start_end_token=config.get('use_start_end_token', True),
            parser=config.get('parser', 'en'),
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
        
        return dataloader
        
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Setup training data loader"""
        if isinstance(train_data_config, dict):
            train_data_config = DictConfig(train_data_config)
        self._train_dl = self._setup_dataloader_from_config(train_data_config)
        
    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data loader"""
        if isinstance(val_data_config, dict):
            val_data_config = DictConfig(val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(val_data_config)
        
    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data loader"""
        if isinstance(test_data_config, dict):
            test_data_config = DictConfig(test_data_config)
        self._test_dl = self._setup_dataloader_from_config(test_data_config)
        
    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 1,
        return_hypotheses: bool = False,
        num_workers: int = 0,
    ) -> List[str]:
        """
        Transcribe audio files (non-streaming)
        
        Args:
            paths2audio_files: List of audio file paths
            batch_size: Batch size for processing
            return_hypotheses: Whether to return detailed hypotheses
            num_workers: Number of workers for data loading
            
        Returns:
            List of transcriptions
        """
        # Create temporary manifest
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            for audio_file in paths2audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                f.write(json.dumps(entry) + '\n')
            manifest_path = f.name
            
        try:
            # Create config for temporary dataset
            config = {
                'manifest_filepath': manifest_path,
                'sample_rate': self.cfg.preprocessor.sample_rate,
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': num_workers,
                'pin_memory': False,
                'use_start_end_token': False,
            }
            
            # Setup temporary dataloader
            temp_dataloader = self._setup_dataloader_from_config(DictConfig(config))
            
            # Transcribe
            transcriptions = []
            for batch in temp_dataloader:
                input_signal, input_signal_length = batch[:2]
                
                # Get encoder output
                encoder_output, encoder_length = self.forward(
                    input_signal=input_signal.to(self.device),
                    input_signal_length=input_signal_length.to(self.device),
                )
                
                # Decode
                for i in range(encoder_output.size(0)):
                    text = self._greedy_decode(
                        encoder_output[i:i+1],
                        encoder_length[i:i+1],
                    )
                    transcriptions.append(text)
                    
        finally:
            # Clean up temporary manifest
            os.unlink(manifest_path)
            
        return transcriptions
        
    @torch.no_grad()
    def transcribe_with_cache_aware_streaming(
        self,
        paths2audio_files: List[str],
        batch_size: int = 1,
        return_hypotheses: bool = False,
        online_normalization: bool = True,
        use_local_agreement: bool = True,
        chunk_size: Optional[float] = None,
        overlap_size: float = 0.0,
    ) -> List[str]:
        """
        Transcribe audio files using cache-aware streaming
        
        Args:
            paths2audio_files: List of audio file paths
            batch_size: Batch size for processing
            return_hypotheses: Whether to return detailed hypotheses
            online_normalization: Whether to use online normalization
            use_local_agreement: Whether to use local agreement decoding
            chunk_size: Size of each chunk in seconds (uses self.chunk_size if None)
            overlap_size: Overlap between chunks in seconds
            
        Returns:
            List of transcriptions
        """
        if batch_size > 1:
            logging.warning("Batch streaming not fully implemented. Using batch_size=1")
            batch_size = 1
            
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        transcriptions = []
        
        for audio_path in paths2audio_files:
            # Setup streaming buffer
            streaming_buffer = CacheAwareStreamingAudioBuffer(
                model=self,
                online_normalization=online_normalization,
            )
            
            # Add audio to buffer
            streaming_buffer.append_audio_file(audio_path, stream_id=-1)
            
            # Initialize encoder cache
            cache_last_channel, cache_last_time, cache_last_channel_len = self.encoder.get_initial_cache_state(
                batch_size=1
            )
            
            # Enable decoder caching
            self.decoder.set_cache_enabled(True)
            
            # Process chunks
            chunk_outputs = []
            previous_output = None
            committed_text = []
            
            for chunk_idx, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
                # Encoder forward with caching
                (
                    encoded,
                    encoded_len,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                ) = self.encoder.cache_aware_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                )
                
                # Decoder forward - greedy decoding
                chunk_text = self._greedy_decode_chunk(
                    encoded, encoded_len, cache_id=0
                )
                
                # Apply local agreement if enabled
                if use_local_agreement and previous_output is not None:
                    agreed_text = self._apply_local_agreement(
                        previous_output, chunk_text
                    )
                    if agreed_text:
                        committed_text.append(agreed_text)
                        logging.debug(f"Chunk {chunk_idx}: Committed '{agreed_text}'")
                else:
                    committed_text.append(chunk_text)
                    
                chunk_outputs.append(chunk_text)
                previous_output = chunk_text
                
            # Clear decoder cache
            self.decoder.clear_cache()
            self.decoder.set_cache_enabled(False)
            
            # Merge committed outputs
            if use_local_agreement:
                final_transcription = ' '.join(committed_text)
            else:
                final_transcription = ' '.join(chunk_outputs)
                
            transcriptions.append(final_transcription)
            
            # Reset buffer for next audio
            streaming_buffer.reset_buffer()
            
        return transcriptions
        
    def _greedy_decode(
        self, 
        encoder_output: torch.Tensor, 
        encoder_length: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> str:
        """Greedy decoding for full utterance"""
        if max_length is None:
            max_length = self.max_decode_length
            
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with BOS token
        current_tokens = torch.full(
            (batch_size, 1), 
            self.bos_id, 
            dtype=torch.long, 
            device=device
        )
        
        for _ in range(max_length):
            # Decoder forward
            decoder_output = self.decoder(
                encoder_output=encoder_output,
                encoder_lengths=encoder_length,
                targets=current_tokens,
            )
            
            # Get next token
            logits = decoder_output['logits']
            next_token = logits[:, -1, :].argmax(dim=-1)
            
            # Check for EOS
            if next_token.item() == self.eos_id:
                break
                
            # Append token
            current_tokens = torch.cat(
                [current_tokens, next_token.unsqueeze(1)], 
                dim=1
            )
            
        # Convert to text (exclude BOS/EOS)
        token_ids = current_tokens[0, 1:].cpu().tolist()
        if token_ids and token_ids[-1] == self.eos_id:
            token_ids = token_ids[:-1]
            
        text = self.tokenizer.ids_to_text(token_ids)
        
        return text
        
    def _greedy_decode_chunk(
        self, 
        encoder_output: torch.Tensor, 
        encoder_length: torch.Tensor,
        cache_id: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """Greedy decoding for a single chunk"""
        if max_length is None:
            max_length = self.max_decode_length
            
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with BOS token
        current_tokens = torch.full(
            (batch_size, 1), 
            self.bos_id, 
            dtype=torch.long, 
            device=device
        )
        
        for _ in range(max_length):
            # Decoder forward with caching
            decoder_output = self.decoder(
                encoder_output=encoder_output,
                encoder_lengths=encoder_length,
                targets=current_tokens,
                cache_id=cache_id,
            )
            
            # Get next token
            logits = decoder_output['logits']
            next_token = logits[:, -1, :].argmax(dim=-1)
            
            # Check for EOS
            if next_token.item() == self.eos_id:
                break
                
            # Append token
            current_tokens = torch.cat(
                [current_tokens, next_token.unsqueeze(1)], 
                dim=1
            )
            
        # Convert to text (exclude BOS/EOS)
        token_ids = current_tokens[0, 1:].cpu().tolist()
        if token_ids and token_ids[-1] == self.eos_id:
            token_ids = token_ids[:-1]
            
        text = self.tokenizer.ids_to_text(token_ids)
        
        return text
        
    def _apply_local_agreement(
        self,
        prev_output: str,
        current_output: str,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Apply local agreement strategy for partial hypothesis selection
        
        Args:
            prev_output: Previous chunk output
            current_output: Current chunk output
            threshold: Agreement threshold (uses self.local_agreement_threshold if None)
            
        Returns:
            Agreed prefix string
        """
        if threshold is None:
            threshold = self.local_agreement_threshold
            
        # Tokenize both outputs for comparison
        prev_tokens = self.tokenizer.text_to_ids(prev_output)
        curr_tokens = self.tokenizer.text_to_ids(current_output)
        
        # Find longest common prefix
        common_prefix_len = 0
        min_len = min(len(prev_tokens), len(curr_tokens))
        
        for i in range(min_len):
            if prev_tokens[i] == curr_tokens[i]:
                common_prefix_len = i + 1
            else:
                break
                
        # Calculate agreement ratio
        if len(prev_tokens) > 0:
            agreement_ratio = common_prefix_len / len(prev_tokens)
        else:
            agreement_ratio = 0
            
        # Decide whether to commit based on agreement
        if agreement_ratio >= threshold and common_prefix_len > 0:
            # Return the agreed prefix
            agreed_tokens = curr_tokens[:common_prefix_len]
            return self.tokenizer.ids_to_text(agreed_tokens)
        else:
            return ""
            
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """List available pretrained models"""
        # This would be populated with pretrained models once they are trained
        return []
        
    def _prepare_for_export(self, **kwargs):
        """Prepare model for export"""
        # Override this method if needed for ONNX export
        pass
        
    def export(self, **kwargs):
        """Export model"""
        # Override this method if needed for ONNX export
        pass
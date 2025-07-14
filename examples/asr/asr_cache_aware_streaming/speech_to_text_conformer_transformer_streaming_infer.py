"""
This script performs cache-aware streaming inference for Conformer-Transformer ASR models.
It implements local agreement decoding strategy for partial hypothesis selection.

Usage:
    python speech_to_text_conformer_transformer_streaming_infer.py \
        --asr_model=conformer_transformer_model.nemo \
        --audio_file=audio_file.wav \
        --chunk_size=0.5 \
        --use_local_agreement \
        --debug_mode

    python speech_to_text_conformer_transformer_streaming_infer.py \
        --asr_model=conformer_transformer_model.nemo \
        --manifest_file=manifest.json \
        --batch_size=16 \
        --chunk_size=0.5 \
        --use_local_agreement
"""

import json
import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models.conformer_transformer_models import EncDecConformerTransformerModel
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.utils import logging


@dataclass
class StreamingHypothesis:
    """Hypothesis for streaming decoding"""
    text: str
    tokens: List[int]
    score: float
    timestamp: float
    chunk_index: int
    is_partial: bool = True


class LocalAgreementDecoder:
    """Implements local agreement strategy for partial hypothesis selection"""
    
    def __init__(self, agreement_threshold: float = 0.5):
        self.agreement_threshold = agreement_threshold
        self.previous_outputs = []
        self.committed_text = []
        
    def apply_local_agreement(
        self,
        current_hypothesis: StreamingHypothesis,
        previous_hypothesis: Optional[StreamingHypothesis] = None
    ) -> Tuple[str, bool]:
        """
        Apply local agreement between consecutive chunks
        
        Args:
            current_hypothesis: Current chunk hypothesis
            previous_hypothesis: Previous chunk hypothesis
            
        Returns:
            Tuple of (agreed_text, should_commit)
        """
        if previous_hypothesis is None:
            # First chunk - don't commit anything yet
            return "", False
            
        # Find longest common prefix between previous and current
        prev_tokens = previous_hypothesis.tokens
        curr_tokens = current_hypothesis.tokens
        
        common_prefix_len = 0
        for i in range(min(len(prev_tokens), len(curr_tokens))):
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
        should_commit = agreement_ratio >= self.agreement_threshold
        
        if should_commit and common_prefix_len > 0:
            # Return the agreed prefix
            agreed_tokens = curr_tokens[:common_prefix_len]
            return self._tokens_to_text(agreed_tokens), True
        else:
            return "", False
            
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text (placeholder - should use actual tokenizer)"""
        # This should use the actual tokenizer from the model
        return " ".join([str(t) for t in tokens])


class ConformerTransformerStreamingInference:
    """Main class for streaming inference with Conformer-Transformer models"""
    
    def __init__(
        self,
        model: EncDecConformerTransformerModel,
        chunk_size: float = 0.5,
        use_local_agreement: bool = True,
        local_agreement_threshold: float = 0.5,
        overlap_size: float = 0.0,
        debug_mode: bool = False,
        decode_strategy: str = 'greedy',
        beam_size: int = 5,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.debug_mode = debug_mode
        
        # Decooding strategy (greedy or beam search)
        self.decode_strategy = decode_strategy
        self.beam_size = beam_size

        # Local agreement decoder
        self.use_local_agreement = use_local_agreement
        if use_local_agreement:
            self.local_agreement_decoder = LocalAgreementDecoder(local_agreement_threshold)
            
        # Enable caching in decoder
        self.model.decoder.set_cache_enabled(True)
        
        # Setup streaming parameters
        self.model.setup_streaming_params(
            chunk_size=int(chunk_size * 1000),  # Convert to ms
            shift_size=int((chunk_size - overlap_size) * 1000),
        )
        
    @torch.no_grad()
    def transcribe_file(self, audio_file: str) -> Tuple[str, List[StreamingHypothesis]]:
        """
        Transcribe a single audio file with streaming
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (final_transcription, chunk_hypotheses)
        """
        # Create streaming buffer
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=True,
        )
        
        # Add audio file to buffer
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            audio_file, stream_id=-1
        )
        
        # Process chunks
        chunk_hypotheses = []
        committed_texts = []
        previous_hypothesis = None
        
        # Get initial cache state
        batch_size = 1
        cache_last_channel, cache_last_time, cache_last_channel_len = self.model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )
        
        chunk_index = 0
        start_time = time.time()
        
        for chunk_audio, chunk_lengths in streaming_buffer:
            chunk_start_time = time.time()
            
            # Encoder forward with caching
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.model.encoder.cache_aware_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
            )
            
            # Decoder forward with beam search
            hypothesis = self._decode_chunk(
                encoded, encoded_len, chunk_index, 
                cache_id=stream_id,
                decode_strategy=self.decode_strategy,
                beam_size=self.beam_size
            )
            
            # Apply local agreement if enabled
            if self.use_local_agreement and previous_hypothesis is not None:
                agreed_text, should_commit = self.local_agreement_decoder.apply_local_agreement(
                    hypothesis, previous_hypothesis
                )
                
                if should_commit:
                    committed_texts.append(agreed_text)
                    if self.debug_mode:
                        logging.info(f"Chunk {chunk_index}: Committed '{agreed_text}'")
            else:
                # Without local agreement, commit everything
                committed_texts.append(hypothesis.text)
                
            chunk_hypotheses.append(hypothesis)
            previous_hypothesis = hypothesis
            
            chunk_end_time = time.time()
            if self.debug_mode:
                logging.info(
                    f"Chunk {chunk_index}: {hypothesis.text} "
                    f"(processing time: {chunk_end_time - chunk_start_time:.3f}s)"
                )
                
            chunk_index += 1
            
        # Final transcription
        final_transcription = " ".join(committed_texts)
        
        total_time = time.time() - start_time
        if self.debug_mode:
            logging.info(f"Total processing time: {total_time:.3f}s")
            logging.info(f"Final transcription: {final_transcription}")
            
        return final_transcription, chunk_hypotheses

    def _decode_chunk(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        chunk_index: int,
        cache_id: Optional[int] = None,
        decode_strategy: str = "greedy",  # "greedy" or "beam"
        beam_size: int = 5,
    ) -> StreamingHypothesis:
        """Decode a single chunk using greedy or beam search decoding"""
        device = encoded.device
        batch_size = encoded.size(0)
        
        if decode_strategy == "greedy":
            # Current greedy implementation
            current_tokens = torch.full((batch_size, 1), self.model.decoder.bos_id, device=device)
            token_list = []
            
            max_decode_length = self.model.streaming_cfg.get('max_decode_length', 50)
            
            for step in range(max_decode_length):
                # Decoder forward
                decoder_output = self.model.decoder(
                    encoder_output=encoded,
                    encoder_lengths=encoded_len,
                    targets=current_tokens,
                    cache_id=cache_id,
                )
                
                # Get next token (greedy)
                logits = decoder_output['logits']
                next_token = logits[:, -1, :].argmax(dim=-1)
                
                # Check for EOS
                if next_token.item() == self.model.decoder.eos_id:
                    break
                    
                # Append token
                token_list.append(next_token.item())
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
                
            # Convert tokens to text
            text = self.model.tokenizer.ids_to_text(token_list)
            score = 0.0  # Greedy doesn't have a meaningful score
            
        elif decode_strategy == "beam":
            # Beam search implementation
            max_decode_length = self.model.streaming_cfg.get('max_decode_length', 50)
            vocab_size = self.model.decoder.output_projection.out_features
            
            # Initialize beam
            # Each beam entry: (tokens, score, finished)
            beam = [(
                torch.full((batch_size, 1), self.model.decoder.bos_id, device=device),
                0.0,
                False
            )]
            
            for step in range(max_decode_length):
                # Candidates for next step
                candidates = []
                
                for tokens, score, finished in beam:
                    if finished:
                        candidates.append((tokens, score, finished))
                        continue
                    
                    # Decoder forward
                    decoder_output = self.model.decoder(
                        encoder_output=encoded,
                        encoder_lengths=encoded_len,
                        targets=tokens,
                        cache_id=cache_id,
                    )
                    
                    # Get log probabilities for next token
                    logits = decoder_output['logits'][:, -1, :]  # [B, V]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top-k candidates
                    topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size, dim=-1)
                    
                    for k in range(beam_size):
                        next_token = topk_indices[0, k]
                        next_score = score + topk_log_probs[0, k].item()
                        
                        if next_token.item() == self.model.decoder.eos_id:
                            # End of sequence
                            candidates.append((tokens, next_score, True))
                        else:
                            # Continue decoding
                            new_tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(1)], dim=1)
                            candidates.append((new_tokens, next_score, False))
                
                # Sort candidates by score and keep top beam_size
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:beam_size]
                
                # Check if all beams are finished
                if all(finished for _, _, finished in beam):
                    break
            
            # Select best hypothesis
            best_tokens, best_score, _ = beam[0]
            token_list = best_tokens[0, 1:].cpu().tolist()  # Exclude BOS
            
            # Remove EOS if present
            if token_list and token_list[-1] == self.model.decoder.eos_id:
                token_list = token_list[:-1]
                
            text = self.model.tokenizer.ids_to_text(token_list)
            score = best_score
            
        else:
            raise ValueError(f"Unknown decode strategy: {decode_strategy}")
        
        # Create hypothesis
        hypothesis = StreamingHypothesis(
            text=text,
            tokens=token_list,
            score=score,
            timestamp=time.time(),
            chunk_index=chunk_index,
            is_partial=True,
        )
        
        return hypothesis
    
    def transcribe_manifest(
        self,
        manifest_file: str,
        output_file: Optional[str] = None,
        batch_size: int = 1,
    ) -> List[Tuple[str, List[StreamingHypothesis]]]:
        """
        Transcribe multiple files from a manifest
        
        Args:
            manifest_file: Path to manifest JSON file
            output_file: Optional output file for results
            batch_size: Batch size (currently only 1 is supported)
            
        Returns:
            List of (transcription, hypotheses) tuples
        """
        results = []
        
        # Load manifest
        with open(manifest_file, 'r') as f:
            manifest_data = [json.loads(line) for line in f]
            
        logging.info(f"Processing {len(manifest_data)} files from manifest")
        
        # Process each file
        for idx, sample in enumerate(manifest_data):
            audio_file = sample['audio_filepath']
            reference_text = sample.get('text', '')
            
            logging.info(f"Processing {idx+1}/{len(manifest_data)}: {audio_file}")
            
            # Transcribe
            transcription, hypotheses = self.transcribe_file(audio_file)
            
            # Calculate WER if reference is available
            if reference_text:
                wer = word_error_rate([transcription], [reference_text])
                logging.info(f"WER: {wer:.2%}")
                
            results.append((transcription, hypotheses))
            
        # Save results if output file specified
        if output_file:
            self._save_results(manifest_data, results, output_file)
            
        return results
        
    def _save_results(
        self,
        manifest_data: List[dict],
        results: List[Tuple[str, List[StreamingHypothesis]]],
        output_file: str,
    ):
        """Save transcription results to file"""
        output_data = []
        
        for sample, (transcription, hypotheses) in zip(manifest_data, results):
            output_entry = {
                'audio_filepath': sample['audio_filepath'],
                'reference': sample.get('text', ''),
                'hypothesis': transcription,
                'chunk_hypotheses': [
                    {
                        'text': h.text,
                        'chunk_index': h.chunk_index,
                        'timestamp': h.timestamp,
                    }
                    for h in hypotheses
                ],
            }
            
            if 'text' in sample:
                output_entry['wer'] = word_error_rate([transcription], [sample['text']])
                
            output_data.append(output_entry)
            
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logging.info(f"Results saved to {output_file}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        required=True,
        help="Path to Conformer-Transformer ASR model .nemo file",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        help="Path to a single audio file to transcribe",
    )
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to manifest file for batch transcription",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output file for results",
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=0.5,
        help="Chunk size in seconds",
    )
    parser.add_argument(
        "--overlap_size",
        type=float,
        default=0.0,
        help="Overlap size between chunks in seconds",
    )
    parser.add_argument(
    "--decode_strategy",
    type=str,
    default="greedy",
    choices=["greedy", "beam"],
    help="Decoding strategy to use",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search",
    )
    parser.add_argument(
        "--use_local_agreement",
        action="store_true",
        help="Use local agreement decoding strategy",
    )
    parser.add_argument(
        "--local_agreement_threshold",
        type=float,
        default=0.5,
        help="Threshold for local agreement (0-1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for manifest processing",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.audio_file and not args.manifest_file:
        raise ValueError("Either --audio_file or --manifest_file must be specified")
        
    if args.audio_file and args.manifest_file:
        raise ValueError("Only one of --audio_file or --manifest_file can be specified")
        
    # Load model
    logging.info(f"Loading model from {args.asr_model}")
    model = EncDecConformerTransformerModel.restore_from(args.asr_model)
    model = model.to(args.device)
    model.eval()
    
    # Create streaming inference object
    streaming_inference = ConformerTransformerStreamingInference(
        model=model,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        use_local_agreement=args.use_local_agreement,
        local_agreement_threshold=args.local_agreement_threshold,
        debug_mode=args.debug_mode,
    )
    
    # Perform transcription
    if args.audio_file:
        # Single file transcription
        transcription, hypotheses = streaming_inference.transcribe_file(args.audio_file)
        
        print(f"\nFinal transcription: {transcription}")
        
        if args.debug_mode:
            print("\nChunk-by-chunk hypotheses:")
            for h in hypotheses:
                print(f"  Chunk {h.chunk_index}: {h.text}")
                
    else:
        # Manifest transcription
        results = streaming_inference.transcribe_manifest(
            args.manifest_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
        )
        
        # Calculate overall statistics
        if args.output_file:
            total_wer = []
            for idx, (transcription, _) in enumerate(results):
                if 'text' in manifest_data[idx]:
                    wer = word_error_rate([transcription], [manifest_data[idx]['text']])
                    total_wer.append(wer)
                    
            if total_wer:
                avg_wer = np.mean(total_wer)
                print(f"\nAverage WER: {avg_wer:.2%}")


if __name__ == "__main__":
    main()

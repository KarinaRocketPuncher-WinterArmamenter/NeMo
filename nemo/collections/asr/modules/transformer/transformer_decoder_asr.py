"""
ASR-specific wrapper for Transformer decoder with streaming support.
This module wraps the existing TransformerDecoder for ASR tasks.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from nemo.collections.asr.modules.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.common.parts.utils import mask_from_lens
from nemo.core.classes import NeuralModule
from nemo.core.neural_types import (
    ChannelType,
    LengthsType,
    LogitsType,
    NeuralType,
)


class TransformerDecoderASR(NeuralModule):
    """
    ASR-specific Transformer decoder wrapper with streaming support.
    
    This class wraps the existing TransformerDecoder module and adds:
    - Token embeddings and output projection
    - ASR-specific forward methods for training and inference
    - Support for streaming with caching
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Dimension of feedforward layers
        dropout: Dropout rate
        max_seq_length: Maximum sequence length for positional encoding
        pad_id: Padding token ID
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
    """
    
    @property
    def input_types(self):
        return {
            "encoder_output": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_lengths": NeuralType(tuple('B'), LengthsType()),
            "targets": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }
    
    @property
    def output_types(self):
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
        }
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding(max_seq_length, d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder (using existing implementation)
        self.transformer = TransformerDecoder(
            num_layers=n_layers,
            hidden_size=d_model,
            inner_size=d_ff,
            num_attention_heads=n_heads,
            attn_score_dropout=dropout,
            attn_layer_dropout=dropout,
            ffn_dropout=dropout,
            hidden_act="relu",
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Caching for streaming inference
        self.use_cache = False
        self.cache = {}
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        cache_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of transformer decoder
        
        Args:
            encoder_output: Encoder outputs [B, T, D]
            encoder_lengths: Encoder output lengths [B]
            targets: Target token IDs [B, U] (training) or [B, 1] (inference)
            target_lengths: Target lengths [B]
            cache_id: Cache ID for streaming inference
            
        Returns:
            Dictionary containing:
                - logits: Output logits [B, U, V]
                - hidden_states: Decoder hidden states [B, U, D]
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Create encoder mask
        encoder_mask = mask_from_lens(encoder_lengths, max_len=encoder_output.size(1))
        
        if self.training and targets is not None:
            # Training mode with teacher forcing
            return self._forward_train(encoder_output, encoder_mask, targets, target_lengths)
        else:
            # Inference mode
            return self._forward_inference(encoder_output, encoder_mask, targets, cache_id)
            
    def _forward_train(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training with teacher forcing"""
        batch_size, max_len = targets.size()
        device = targets.device
        
        # Shift targets for teacher forcing (add BOS, remove last token)
        bos_tokens = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([bos_tokens, targets[:, :-1]], dim=1)
        
        # Token embeddings with scaling
        token_embeddings = self.token_embedding(decoder_input) * math.sqrt(self.d_model)
        
        # Add positional encoding
        positions = self.positional_encoding[:, :max_len, :].to(device)
        decoder_states = self.dropout(token_embeddings + positions)
        
        # Create decoder mask
        if target_lengths is not None:
            decoder_mask = mask_from_lens(target_lengths, max_len=max_len)
        else:
            decoder_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
            
        # Transformer decoder forward
        decoder_output = self.transformer(
            decoder_states=decoder_states,
            decoder_mask=decoder_mask,
            encoder_states=encoder_output,
            encoder_mask=encoder_mask,
            return_mems=False,
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return {
            'logits': logits,
            'hidden_states': decoder_output,
        }
        
    def _forward_inference(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        partial_targets: Optional[torch.Tensor] = None,
        cache_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass during inference with optional caching
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        if partial_targets is None:
            # Start decoding with BOS token
            partial_targets = torch.full(
                (batch_size, 1), 
                self.bos_id, 
                dtype=torch.long, 
                device=device
            )
            
        current_len = partial_targets.size(1)
        
        # Token embeddings with scaling
        token_embeddings = self.token_embedding(partial_targets) * math.sqrt(self.d_model)
        
        # Add positional encoding
        positions = self.positional_encoding[:, :current_len, :].to(device)
        decoder_states = self.dropout(token_embeddings + positions)
        
        # Create decoder mask (all ones for inference)
        decoder_mask = torch.ones(batch_size, current_len, dtype=torch.bool, device=device)
        
        # Get cached states if available
        decoder_mems_list = None
        if self.use_cache and cache_id is not None:
            decoder_mems_list = self.cache.get(cache_id, None)
            
        # Transformer decoder forward
        if self.use_cache:
            # Forward with caching
            output_mems = self.transformer(
                decoder_states=decoder_states,
                decoder_mask=decoder_mask,
                encoder_states=encoder_output,
                encoder_mask=encoder_mask,
                decoder_mems_list=decoder_mems_list,
                return_mems=True,
                return_mems_as_list=True,
            )
            
            # Extract decoder output (last memory state)
            decoder_output = output_mems[-1]
            
            # Update cache
            if cache_id is not None:
                self.cache[cache_id] = output_mems
        else:
            # Standard forward without caching
            decoder_output = self.transformer(
                decoder_states=decoder_states,
                decoder_mask=decoder_mask,
                encoder_states=encoder_output,
                encoder_mask=encoder_mask,
                return_mems=False,
            )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return {
            'logits': logits,
            'hidden_states': decoder_output,
        }
        
    def decode_step(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        prev_tokens: torch.Tensor,
        cache_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step decoding for autoregressive generation
        
        Args:
            encoder_output: Encoder outputs [B, T, D]
            encoder_mask: Encoder mask [B, T]
            prev_tokens: Previously generated tokens [B, U]
            cache_id: Cache ID for streaming
            
        Returns:
            Tuple of (next_token_logits [B, V], hidden_states [B, 1, D])
        """
        output = self._forward_inference(
            encoder_output, 
            encoder_mask, 
            prev_tokens, 
            cache_id
        )
        
        # Return only the last token's logits
        next_token_logits = output['logits'][:, -1, :]
        last_hidden = output['hidden_states'][:, -1:, :]
        
        return next_token_logits, last_hidden
        
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching for streaming inference"""
        self.use_cache = enabled
        if not enabled:
            self.cache.clear()
            
    def clear_cache(self, cache_id: Optional[int] = None):
        """Clear cache for given ID or all caches"""
        if cache_id is not None:
            self.cache.pop(cache_id, None)
        else:
            self.cache.clear()

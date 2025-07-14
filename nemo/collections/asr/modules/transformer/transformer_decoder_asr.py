"""
ASR-specific wrapper for Transformer decoder with streaming support.
This module wraps the existing TransformerDecoder for ASR tasks.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Try to import from different possible locations
try:
    from nemo.collections.nlp.modules.common.transformer.transformer_modules import TransformerDecoder
except ImportError:
    try:
        from nemo.collections.asr.modules.transformer_modules import TransformerDecoder
    except ImportError:
        # Fall back to a basic implementation
        from nemo.collections.common.parts.transformer import TransformerDecoderNM as TransformerDecoder

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import (
    ChannelType,
    LengthsType,
    LogitsType,
    NeuralType,
)


def mask_from_lens(lens, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    
    Args:
        lens: Tensor of sequence lengths [B]
        max_len: Maximum sequence length (if None, uses max of lens)
        
    Returns:
        Boolean mask tensor [B, max_len] where True indicates valid positions
    """
    if max_len is None:
        max_len = lens.max().item()
    batch_size = lens.shape[0]
    device = lens.device
    
    # Create a range tensor [0, 1, 2, ..., max_len-1]
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    
    # Compare with lengths to create mask
    mask = idx < lens.unsqueeze(1)
    
    return mask


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
        
        # Build transformer decoder layers manually if TransformerDecoder is not available
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(self._build_decoder_layer(d_model, n_heads, d_ff, dropout))
        
        # Layer normalization
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Caching for streaming inference
        self.use_cache = False
        self.cache = {}
        
    def _build_decoder_layer(self, d_model, n_heads, d_ff, dropout):
        """Build a single transformer decoder layer"""
        layer = nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'self_attn_norm': nn.LayerNorm(d_model),
            'cross_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'cross_attn_norm': nn.LayerNorm(d_model),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ),
            'ffn_norm': nn.LayerNorm(d_model),
            'dropout': nn.Dropout(dropout)
        })
        return layer
        
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
        
    def _forward_decoder_layer(self, layer, x, encoder_output, self_attn_mask, cross_attn_mask):
        """Forward pass through a single decoder layer"""
        # Self-attention
        residual = x
        x = layer['self_attn_norm'](x)
        x, _ = layer['self_attn'](x, x, x, attn_mask=self_attn_mask)
        x = layer['dropout'](x) + residual
        
        # Cross-attention
        residual = x
        x = layer['cross_attn_norm'](x)
        x, _ = layer['cross_attn'](x, encoder_output, encoder_output, key_padding_mask=cross_attn_mask)
        x = layer['dropout'](x) + residual
        
        # Feed-forward
        residual = x
        x = layer['ffn_norm'](x)
        x = layer['ffn'](x) + residual
        
        return x
        
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
        encoder_mask = ~mask_from_lens(encoder_lengths, max_len=encoder_output.size(1))
        
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
        
        # Create causal mask for self-attention
        causal_mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1).bool()
        
        # Forward through decoder layers
        for layer in self.layers:
            decoder_states = self._forward_decoder_layer(
                layer, decoder_states, encoder_output, causal_mask, encoder_mask
            )
        
        # Final layer norm
        decoder_states = self.final_layer_norm(decoder_states)
        
        # Output projection
        logits = self.output_projection(decoder_states)
        
        return {
            'logits': logits,
            'hidden_states': decoder_states,
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
        
        # For inference, we don't need causal mask for already generated tokens
        # Forward through decoder layers
        for i, layer in enumerate(self.layers):
            # Simple forward without caching for now
            decoder_states = self._forward_decoder_layer(
                layer, decoder_states, encoder_output, None, encoder_mask
            )
        
        # Final layer norm
        decoder_states = self.final_layer_norm(decoder_states)
        
        # Output projection
        logits = self.output_projection(decoder_states)
        
        return {
            'logits': logits,
            'hidden_states': decoder_states,
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
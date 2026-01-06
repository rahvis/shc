"""
Embedding Layers

Token and positional embeddings for transformer input.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token embedding with optional scaling.
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Embedding dimension
        padding_idx: Index for padding token (no gradient)
        scale: Scale embeddings by sqrt(hidden_dim)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_idx: Optional[int] = None,
        scale: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        self.embedding = nn.Embedding(
            vocab_size, 
            hidden_dim, 
            padding_idx=padding_idx,
        )
        
        # Initialize
        nn.init.normal_(self.embedding.weight, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Embed input tokens.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            
        Returns:
            Embeddings of shape (batch, seq_len, hidden_dim)
        """
        embeddings = self.embedding(input_ids)
        
        if self.scale:
            embeddings = embeddings * math.sqrt(self.hidden_dim)
        
        return embeddings


class PositionalEmbedding(nn.Module):
    """
    Learnable absolute positional embeddings.
    
    Note: Modern LLMs typically use RoPE instead, which is applied
    in the attention layer. This is provided for compatibility.
    
    Args:
        max_seq_len: Maximum sequence length
        hidden_dim: Embedding dimension
    """
    
    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Register position indices as buffer
        self.register_buffer(
            'positions',
            torch.arange(max_seq_len).unsqueeze(0),
        )
        
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional embeddings to input.
        
        Args:
            x: Input of shape (batch, seq_len, hidden_dim)
            
        Returns:
            Input with positional embeddings added
        """
        seq_len = x.size(1)
        positions = self.positions[:, :seq_len].expand(x.size(0), -1)
        return x + self.embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings (Vaswani et al.).
    
    Non-learnable, can extrapolate to longer sequences.
    
    Args:
        hidden_dim: Embedding dimension
        max_seq_len: Maximum sequence length for precomputation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Precompute positional encodings
        pe = self._create_sinusoidal_embeddings(max_seq_len, hidden_dim)
        self.register_buffer('pe', pe)
    
    def _create_sinusoidal_embeddings(
        self, 
        max_len: int, 
        dim: int,
    ) -> Tensor:
        """Create sinusoidal position embeddings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: Tensor) -> Tensor:
        """Add sinusoidal positional embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, used in LLaMA.
    
    Args:
        hidden_dim: Dimension to normalize
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

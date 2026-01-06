"""
Factorized KV Cache Layer

Implements learned low-rank compression for multi-stream hidden states:
    x_bar ≈ U · V^T

This reduces KV cache from 4× to 1.2× baseline by caching only the
compressed representation and reconstructing on-the-fly.

Reference: Section 3.2 of the SHC paper
"""

from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FactorizedKVCache(nn.Module):
    """
    Learned low-rank compression for multi-stream KV cache.
    
    Instead of caching full n×d tensors per token, we cache:
        1. Shared projection matrix V ∈ R^{d×r}
        2. Per-token coefficients u_t ∈ R^r
    
    Reconstruction: x_bar_t = u_t · V^T
    
    For r=1 (rank-1), storage per token becomes 1 scalar + shared projection,
    achieving ~1.2× overhead vs 4× for full caching.
    
    Args:
        n: Number of streams
        d: Hidden dimension
        r: Rank of factorization (default: 1 for maximum compression)
        init_method: Initialization method ('svd' or 'orthogonal')
    
    Example:
        >>> cache = FactorizedKVCache(n=4, d=768, r=1)
        >>> x_bar = torch.randn(32, 4, 768)  # [batch, streams, hidden]
        >>> u = cache.compress(x_bar)  # [batch, 1]
        >>> x_reconstructed = cache.decompress(u)  # [batch, 4, 768]
    """
    
    def __init__(
        self,
        n: int,
        d: int,
        r: int = 1,
        init_method: str = 'orthogonal',
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.r = r
        
        # Learned projection matrix V ∈ R^{d×r}
        # This is shared across the sequence (stored once per layer)
        self.V = nn.Parameter(torch.empty(d, r))
        
        # Stream projection U ∈ R^{n×r}
        # Projects from n streams to r compressed dimensions
        self.U = nn.Parameter(torch.empty(n, r))
        
        # Initialize projections
        self._init_projections(init_method)
        
    def _init_projections(self, method: str):
        """Initialize projection matrices."""
        if method == 'orthogonal':
            # Orthogonal initialization for better conditioning
            nn.init.orthogonal_(self.V)
            nn.init.orthogonal_(self.U)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.V)
            nn.init.xavier_uniform_(self.U)
        else:
            # Default: scaled random
            nn.init.normal_(self.V, std=1.0 / math.sqrt(self.d))
            nn.init.normal_(self.U, std=1.0 / math.sqrt(self.n))
    
    def compress(self, x_bar: Tensor) -> Tensor:
        """
        Compress multi-stream hidden state for KV cache storage.
        
        x_bar: [batch, n, d] → u: [batch, r]
        
        Compression formula: u = (U^T @ x_bar) @ V
        This projects from (n, d) space to r-dimensional coefficients.
        
        Args:
            x_bar: Multi-stream hidden state of shape (batch, n, d)
            
        Returns:
            u: Compressed coefficients of shape (batch, r)
        """
        # Project streams: (batch, n, d) → (batch, r, d) via U^T
        projected_streams = torch.einsum('bnd,nr->brd', x_bar, self.U)  # (batch, r, d)
        
        # Project features: (batch, r, d) @ (d, r) → (batch, r, r)
        # For r=1, this reduces to (batch, 1)
        u = torch.einsum('brd,dr->br', projected_streams, self.V)  # (batch, r)
        
        return u
    
    def decompress(self, u: Tensor) -> Tensor:
        """
        Reconstruct multi-stream hidden state from compressed coefficients.
        
        u: [batch, r] → x_bar: [batch, n, d]
        
        Reconstruction formula: x_bar = U @ (u @ V^T)
        
        Args:
            u: Compressed coefficients of shape (batch, r)
            
        Returns:
            x_bar: Reconstructed hidden state of shape (batch, n, d)
        """
        # Expand features: (batch, r) @ (r, d) → (batch, d)
        # But we need (batch, r, d) for intermediate
        # u: (batch, r) → (batch, r, 1) @ (1, d) = (batch, r, d)
        expanded_features = torch.einsum('br,rd->brd', u, self.V.T)  # (batch, r, d)
        
        # Expand streams: (n, r) @ (batch, r, d) → (batch, n, d)
        x_bar = torch.einsum('nr,brd->bnd', self.U, expanded_features)  # (batch, n, d)
        
        return x_bar
    
    def forward(
        self, 
        x_bar: Tensor, 
        mode: str = 'compress'
    ) -> Tensor:
        """
        Forward pass in either compress or decompress mode.
        
        Args:
            x_bar: Input tensor
            mode: 'compress' or 'decompress'
            
        Returns:
            Compressed or decompressed tensor
        """
        if mode == 'compress':
            return self.compress(x_bar)
        elif mode == 'decompress':
            return self.decompress(x_bar)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'compress' or 'decompress'.")
    
    def compute_reconstruction_error(self, x_bar: Tensor) -> Tensor:
        """
        Compute reconstruction error: ||x_bar - decompress(compress(x_bar))||_F.
        
        Used to validate compression quality (target: 99% reconstruction).
        
        Args:
            x_bar: Original hidden state
            
        Returns:
            Relative reconstruction error (lower is better)
        """
        u = self.compress(x_bar)
        x_reconstructed = self.decompress(u)
        
        # Relative Frobenius norm error
        error = torch.norm(x_bar - x_reconstructed, p='fro') / torch.norm(x_bar, p='fro')
        return error
    
    def compute_reconstruction_accuracy(self, x_bar: Tensor) -> Tensor:
        """
        Compute reconstruction accuracy: 1 - relative_error.
        
        Target: ≥99% for r=1.
        
        Args:
            x_bar: Original hidden state
            
        Returns:
            Reconstruction accuracy (0 to 1, higher is better)
        """
        return 1.0 - self.compute_reconstruction_error(x_bar)
    
    def get_compression_ratio(self) -> float:
        """
        Compute theoretical compression ratio.
        
        Original: n × d per token
        Compressed: r (coefficients) + d × r / L (amortized projection)
        
        For large L, ratio ≈ (n × d) / r
        
        Returns:
            Compression ratio (higher = more compression)
        """
        original_size = self.n * self.d
        compressed_size = self.r  # Per-token storage (ignoring shared V)
        return original_size / compressed_size
    
    def get_memory_stats(self, seq_len: int) -> Dict[str, float]:
        """
        Compute memory usage statistics.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Dict with memory stats in number of elements
        """
        # Original KV cache: n × d × seq_len per layer
        original = self.n * self.d * seq_len
        
        # Compressed: r × seq_len (coefficients) + d × r (shared projection)
        compressed = self.r * seq_len + self.d * self.r
        
        return {
            'original': original,
            'compressed': compressed,
            'ratio': original / compressed,
            'savings_percent': (1 - compressed / original) * 100,
        }
    
    def extra_repr(self) -> str:
        return (
            f'n={self.n}, d={self.d}, r={self.r}, '
            f'compression_ratio={self.get_compression_ratio():.1f}x'
        )


class AdaptiveFactorizedCache(nn.Module):
    """
    Factorized cache with adaptive rank selection.
    
    Allows different ranks for different layers based on learned importance.
    Early layers may use r=1, while later layers may use r=2.
    
    Args:
        n: Number of streams
        d: Hidden dimension
        max_r: Maximum rank
        n_layers: Number of layers
    """
    
    def __init__(
        self,
        n: int,
        d: int,
        max_r: int = 2,
        n_layers: int = 32,
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.max_r = max_r
        self.n_layers = n_layers
        
        # Per-layer factorization modules
        self.layer_caches = nn.ModuleList([
            FactorizedKVCache(n, d, r=max_r)
            for _ in range(n_layers)
        ])
        
        # Learned rank importance per layer
        self.rank_importance = nn.Parameter(torch.ones(n_layers, max_r))
        
    def get_effective_rank(self, layer_idx: int, threshold: float = 0.5) -> int:
        """
        Get effective rank for a layer based on learned importance.
        
        Args:
            layer_idx: Layer index
            threshold: Importance threshold
            
        Returns:
            Effective rank (1 to max_r)
        """
        importance = torch.sigmoid(self.rank_importance[layer_idx])
        effective_r = (importance > threshold).sum().item()
        return max(1, int(effective_r))
    
    def compress_layer(self, x_bar: Tensor, layer_idx: int) -> Tensor:
        """Compress for a specific layer."""
        return self.layer_caches[layer_idx].compress(x_bar)
    
    def decompress_layer(self, u: Tensor, layer_idx: int) -> Tensor:
        """Decompress for a specific layer."""
        return self.layer_caches[layer_idx].decompress(u)


class StreamingKVCache:
    """
    Streaming KV cache with factorized storage.
    
    Manages incremental caching during autoregressive generation.
    
    Args:
        factorizer: FactorizedKVCache module
        max_seq_len: Maximum sequence length to cache
    """
    
    def __init__(
        self,
        factorizer: FactorizedKVCache,
        max_seq_len: int = 8192,
    ):
        self.factorizer = factorizer
        self.max_seq_len = max_seq_len
        
        # Cache storage
        self._cache: Optional[Tensor] = None
        self._len: int = 0
        
    def reset(self):
        """Clear the cache."""
        self._cache = None
        self._len = 0
        
    def append(self, x_bar: Tensor):
        """
        Append new token(s) to cache.
        
        Args:
            x_bar: New hidden state(s) of shape (batch, num_new_tokens, n, d)
                   or (batch, n, d) for single token
        """
        if x_bar.dim() == 3:
            x_bar = x_bar.unsqueeze(1)  # Add seq dim
            
        batch, num_new, n, d = x_bar.shape
        
        # Compress new tokens: (batch, num_new, n, d) → (batch, num_new, r)
        x_flat = x_bar.view(batch * num_new, n, d)
        u_flat = self.factorizer.compress(x_flat)  # (batch * num_new, r)
        u_new = u_flat.view(batch, num_new, -1)  # (batch, num_new, r)
        
        if self._cache is None:
            self._cache = u_new
        else:
            self._cache = torch.cat([self._cache, u_new], dim=1)
            
        self._len += num_new
        
        # Truncate if exceeding max length
        if self._len > self.max_seq_len:
            excess = self._len - self.max_seq_len
            self._cache = self._cache[:, excess:]
            self._len = self.max_seq_len
    
    def get_all(self) -> Tensor:
        """
        Get all cached hidden states (decompressed).
        
        Returns:
            All cached states of shape (batch, cached_len, n, d)
        """
        if self._cache is None:
            return None
            
        batch, seq_len, r = self._cache.shape
        
        # Decompress all cached tokens
        u_flat = self._cache.view(batch * seq_len, r)
        x_flat = self.factorizer.decompress(u_flat)  # (batch * seq_len, n, d)
        
        return x_flat.view(batch, seq_len, self.factorizer.n, self.factorizer.d)
    
    def __len__(self) -> int:
        return self._len
    
    @property
    def memory_usage(self) -> int:
        """Current memory usage in number of elements."""
        if self._cache is None:
            return 0
        return self._cache.numel()

"""
Multi-Head Attention Layer

Standard multi-head attention with optional optimizations:
- Flash Attention support
- KV cache for autoregressive generation
- Rotary position embeddings (RoPE)

Compatible with SHC's multi-stream routing.
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for position-aware attention.
    
    Applies rotation to query and key vectors based on position,
    enabling relative position awareness without explicit position embeddings.
    
    Args:
        dim: Dimension of the embedding (typically head_dim)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 8192, 
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        """Build position embedding cache."""
        positions = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        
        # Cache cos and sin
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
        
    def forward(
        self, 
        x: Tensor, 
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get cos and sin embeddings for given sequence length.
        
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Sequence length (defaults to x.shape[1])
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with SHC compatibility.
    
    Features:
        - Standard scaled dot-product attention
        - Optional flash attention (when available)
        - KV cache for autoregressive generation
        - Rotary position embeddings
        - Dropout and residual support
    
    Args:
        hidden_dim: Model hidden dimension
        n_heads: Number of attention heads
        head_dim: Dimension per head (defaults to hidden_dim // n_heads)
        dropout: Attention dropout probability
        use_flash: Use flash attention if available
        use_rope: Use rotary position embeddings
        max_seq_len: Maximum sequence length (for RoPE cache)
        
    Example:
        >>> attn = MultiHeadAttention(hidden_dim=768, n_heads=12)
        >>> x = torch.randn(32, 128, 768)
        >>> output = attn(x)  # Self-attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim or hidden_dim // n_heads
        self.dropout = dropout
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.use_rope = use_rope
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)
        
        # Rotary position embeddings
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            kv_cache: Optional tuple of (cached_k, cached_v) for generation
            attention_mask: Optional attention mask
            is_causal: Apply causal masking (for autoregressive)
            
        Returns:
            output: Attention output of shape (batch, seq_len, hidden_dim)
            new_kv_cache: Updated KV cache (if input cache provided)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        if self.use_rope:
            cos, sin = self.rope(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        new_kv_cache = (k, v) if kv_cache is not None else None
        
        # Compute attention
        if self.use_flash:
            # Use PyTorch's scaled_dot_product_attention (flash attention)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and kv_cache is None,
            )
        else:
            # Manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if is_causal and kv_cache is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
                
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_kv_cache
    
    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.hidden_dim}, n_heads={self.n_heads}, '
            f'head_dim={self.head_dim}, use_flash={self.use_flash}'
        )


class GroupedQueryAttention(MultiHeadAttention):
    """
    Grouped-Query Attention (GQA) for efficient inference.
    
    Uses fewer KV heads than query heads to reduce memory and computation.
    Common in modern LLMs (e.g., Llama 2, Mistral).
    
    Args:
        hidden_dim: Model hidden dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (< n_heads)
        **kwargs: Additional arguments for MultiHeadAttention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int = 8,
        **kwargs,
    ):
        super().__init__(hidden_dim, n_heads, **kwargs)
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        
        # Smaller KV projections
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        
    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward with grouped-query attention."""
        batch, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rope(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos[:, :self.n_kv_heads], sin[:, :self.n_kv_heads])
        
        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        new_kv_cache = (k, v) if kv_cache is not None else None
        
        # Expand KV for grouped attention
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)
        
        # Compute attention (using flash if available)
        if self.use_flash:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and kv_cache is None,
            )
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if is_causal and kv_cache is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_kv_cache

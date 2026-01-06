"""
Feed-Forward Network Layer

Standard transformer FFN with SwiGLU activation (used in LLaMA, Mistral).
Compatible with SHC's multi-stream architecture.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.
    
    Standard architecture:
        FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
    
    SwiGLU provides better performance than standard ReLU/GELU FFN.
    
    Args:
        hidden_dim: Model hidden dimension
        ffn_dim: Intermediate FFN dimension (typically 4 * hidden_dim or 8/3 * hidden_dim)
        dropout: Dropout probability
        bias: Use bias in linear layers
        
    Example:
        >>> ffn = FeedForward(hidden_dim=768, ffn_dim=3072)
        >>> x = torch.randn(32, 128, 768)
        >>> output = ffn(x)  # Same shape as input
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim or int(8 / 3 * hidden_dim)  # LLaMA-style
        
        # SwiGLU projections
        self.w_gate = nn.Linear(hidden_dim, self.ffn_dim, bias=bias)
        self.w_up = nn.Linear(hidden_dim, self.ffn_dim, bias=bias)
        self.w_down = nn.Linear(self.ffn_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            
        Returns:
            Output tensor of same shape
        """
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        
        # Project down
        output = self.w_down(hidden)
        output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f'hidden_dim={self.hidden_dim}, ffn_dim={self.ffn_dim}'


class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts Feed-Forward with sparse expert selection.
    
    Each token is routed to top-k experts, enabling larger capacity
    without proportional compute increase.
    
    Args:
        hidden_dim: Model hidden dimension
        ffn_dim: FFN dimension per expert
        n_experts: Total number of experts
        top_k: Number of experts per token
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim or int(8 / 3 * hidden_dim)
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Expert router
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            FeedForward(hidden_dim, self.ffn_dim, dropout)
            for _ in range(n_experts)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward with sparse expert routing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            
        Returns:
            Output tensor of same shape
        """
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch * seq_len, dim)
        
        # Compute routing weights
        router_logits = self.router(x_flat)  # (batch * seq_len, n_experts)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Apply experts (can be parallelized with expert parallelism)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get routing weight for this expert
            expert_indices = (selected_experts == i).float()
            expert_weights = (routing_weights * expert_indices).sum(dim=-1, keepdim=True)
            
            # Apply expert and accumulate
            expert_output = expert(x_flat[expert_mask])
            output[expert_mask] += expert_weights[expert_mask] * expert_output
        
        return output.view(batch, seq_len, dim)
    
    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.hidden_dim}, ffn_dim={self.ffn_dim}, '
            f'n_experts={self.n_experts}, top_k={self.top_k}'
        )


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward with adaptive computation.
    
    Uses learned gating to potentially skip FFN for "easy" tokens.
    Useful for adaptive compute in SHC.
    
    Args:
        hidden_dim: Model hidden dimension
        ffn_dim: FFN dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward with gated FFN.
        
        Args:
            x: Input tensor
            
        Returns:
            Gated FFN output
        """
        # Compute gate
        g = self.gate(x)  # (batch, seq_len, 1)
        
        # Apply FFN scaled by gate
        ffn_out = self.ffn(x)
        return g * ffn_out

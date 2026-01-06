"""
Adaptive Rank Selection Layer

Implements input-dependent rank selection via Gumbel-Softmax:
    n_eff(x, l) = Σ_j j · π_j
    
where π = Gumbel-Softmax(W_l · x).

This allows different inputs and layers to use different effective
stream counts, bypassing multi-stream overhead when not needed.

Reference: Section 3.3 of the SHC paper
"""

from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveRankSelector(nn.Module):
    """
    Input-dependent rank selection for adaptive stream expansion.
    
    Uses Gumbel-Softmax during training for differentiable discrete selection,
    and argmax during inference for deterministic selection.
    
    This allows the model to learn:
        - Early layers: n_eff ≈ 1 (simple feature extraction)
        - Middle layers: n_eff ≈ 2-4 (complex reasoning)
        - Late layers: n_eff ≈ 1-2 (output projection)
    
    Args:
        hidden_dim: Input hidden dimension
        max_n: Maximum number of streams (default: 4)
        temperature: Initial Gumbel-Softmax temperature
        min_temperature: Minimum temperature during annealing
    
    Example:
        >>> selector = AdaptiveRankSelector(hidden_dim=768, max_n=4)
        >>> x = torch.randn(32, 128, 768)  # [batch, seq, hidden]
        >>> n_eff = selector(x)  # [batch] effective ranks
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_n: int = 4,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.temperature = temperature
        self.min_temperature = min_temperature
        
        # Rank logits predictor
        self.rank_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, max_n),
        )
        
        # Register rank values as buffer (not trainable)
        self.register_buffer(
            'rank_values', 
            torch.arange(1, max_n + 1, dtype=torch.float)
        )
        
        # Optional: learnable layer-specific bias
        self.layer_bias = nn.Parameter(torch.zeros(max_n))
        
    def get_rank_logits(self, x: Tensor) -> Tensor:
        """
        Compute rank selection logits from input.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim) or (batch, hidden_dim)
            
        Returns:
            Logits of shape (batch, max_n)
        """
        # Pool over sequence if present
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        else:
            x_pooled = x
            
        logits = self.rank_proj(x_pooled)  # (batch, max_n)
        logits = logits + self.layer_bias  # Add layer-specific bias
        
        return logits
    
    def forward(
        self, 
        x: Tensor, 
        temperature: Optional[float] = None,
        hard: bool = True,
    ) -> Tensor:
        """
        Compute effective rank for each input.
        
        Args:
            x: Input tensor
            temperature: Override temperature (uses self.temperature if None)
            hard: If True, use straight-through estimator for hard selection
            
        Returns:
            n_eff: Effective ranks of shape (batch,)
        """
        temp = temperature if temperature is not None else self.temperature
        logits = self.get_rank_logits(x)  # (batch, max_n)
        
        if self.training:
            # Gumbel-Softmax for differentiable selection
            probs = F.gumbel_softmax(logits, tau=temp, hard=hard)
        else:
            # Hard selection at inference
            idx = logits.argmax(dim=-1)
            probs = F.one_hot(idx, num_classes=self.max_n).float()
            
        # Compute expected rank: Σ_j j · π_j
        n_eff = (probs * self.rank_values.unsqueeze(0)).sum(dim=-1)
        
        return n_eff
    
    def get_rank_distribution(self, x: Tensor) -> Tensor:
        """
        Get probability distribution over ranks (for analysis).
        
        Args:
            x: Input tensor
            
        Returns:
            Probability distribution of shape (batch, max_n)
        """
        logits = self.get_rank_logits(x)
        return F.softmax(logits, dim=-1)
    
    def get_hard_rank(self, x: Tensor) -> Tensor:
        """
        Get hard rank selection (integer) for inference.
        
        Args:
            x: Input tensor
            
        Returns:
            Integer ranks of shape (batch,) in range [1, max_n]
        """
        logits = self.get_rank_logits(x)
        return logits.argmax(dim=-1) + 1  # +1 because ranks are 1-indexed
    
    def set_temperature(self, temperature: float):
        """
        Set Gumbel-Softmax temperature.
        
        Lower temperature → more peaked distribution (harder selection).
        Typically annealed during training.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = max(temperature, self.min_temperature)
    
    def anneal_temperature(self, factor: float = 0.99):
        """
        Anneal temperature by multiplicative factor.
        
        Called periodically during training to gradually
        transition from soft to hard selection.
        
        Args:
            factor: Multiplicative annealing factor
        """
        self.set_temperature(self.temperature * factor)
    
    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.hidden_dim}, max_n={self.max_n}, '
            f'temperature={self.temperature:.3f}'
        )


class LayerWiseRankSelector(nn.Module):
    """
    Rank selector with per-layer learned biases.
    
    Different layers may prefer different rank distributions:
        - Early layers: lower ranks (simple operations)
        - Middle layers: higher ranks (complex reasoning)
        - Late layers: medium ranks (output formatting)
    
    Args:
        hidden_dim: Input hidden dimension
        max_n: Maximum number of streams
        n_layers: Number of layers in the model
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_n: int = 4,
        n_layers: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        self.n_layers = n_layers
        self.temperature = temperature
        
        # Shared feature extractor
        self.shared_proj = nn.Linear(hidden_dim, hidden_dim // 4)
        
        # Per-layer rank predictors
        self.layer_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, max_n)
            for _ in range(n_layers)
        ])
        
        # Per-layer learned biases (encourage layer-specific patterns)
        self.layer_biases = nn.Parameter(torch.zeros(n_layers, max_n))
        
        # Initialize biases to encourage expected pattern
        self._init_layer_biases()
        
        # Register rank values
        self.register_buffer(
            'rank_values',
            torch.arange(1, max_n + 1, dtype=torch.float)
        )
        
    def _init_layer_biases(self):
        """Initialize layer biases to encourage expected rank patterns."""
        with torch.no_grad():
            for l in range(self.n_layers):
                # Normalized layer position [0, 1]
                pos = l / (self.n_layers - 1) if self.n_layers > 1 else 0.5
                
                # Early layers: prefer low ranks
                # Middle layers: prefer high ranks
                # Late layers: prefer medium ranks
                if pos < 0.25:
                    # Early: favor rank 1-2
                    self.layer_biases[l] = torch.tensor([1.0, 0.5, -0.5, -1.0])
                elif pos < 0.75:
                    # Middle: favor rank 3-4
                    self.layer_biases[l] = torch.tensor([-0.5, 0.0, 0.5, 1.0])
                else:
                    # Late: favor rank 2
                    self.layer_biases[l] = torch.tensor([0.0, 1.0, 0.0, -0.5])
    
    def forward(
        self,
        x: Tensor,
        layer_idx: int,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Compute effective rank for a specific layer.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim) or (batch, hidden_dim)
            layer_idx: Layer index
            temperature: Override temperature
            
        Returns:
            n_eff: Effective ranks of shape (batch,)
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Pool if needed
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)
        else:
            x_pooled = x
            
        # Shared features
        features = F.gelu(self.shared_proj(x_pooled))  # (batch, hidden_dim // 4)
        
        # Layer-specific logits
        logits = self.layer_heads[layer_idx](features)  # (batch, max_n)
        logits = logits + self.layer_biases[layer_idx]  # Add layer bias
        
        if self.training:
            probs = F.gumbel_softmax(logits, tau=temp, hard=True)
        else:
            idx = logits.argmax(dim=-1)
            probs = F.one_hot(idx, num_classes=self.max_n).float()
            
        n_eff = (probs * self.rank_values.unsqueeze(0)).sum(dim=-1)
        
        return n_eff
    
    def get_layer_rank_stats(self, x: Tensor) -> Tensor:
        """
        Get expected rank for all layers (for analysis).
        
        Args:
            x: Input tensor
            
        Returns:
            Expected ranks for each layer of shape (n_layers,)
        """
        ranks = []
        for l in range(self.n_layers):
            with torch.no_grad():
                n_eff = self.forward(x, l)
                ranks.append(n_eff.mean().item())
        return torch.tensor(ranks)


class StreamExpander(nn.Module):
    """
    Expands input from single stream to n streams based on effective rank.
    
    When n_eff=1: bypass (no expansion overhead)
    When n_eff>1: expand via learned projection
    
    Args:
        hidden_dim: Input hidden dimension
        max_n: Maximum number of streams
    """
    
    def __init__(self, hidden_dim: int, max_n: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        
        # Expansion projection: d → n×d
        self.expand_proj = nn.Linear(hidden_dim, max_n * hidden_dim)
        
        # Initialize to near-identity for stability
        with torch.no_grad():
            # Main diagonal blocks ≈ identity
            weight = self.expand_proj.weight.view(max_n, hidden_dim, hidden_dim)
            for i in range(max_n):
                nn.init.eye_(weight[i])
                weight[i] *= 0.9  # Slightly shrink
            self.expand_proj.weight.copy_(weight.view(-1, hidden_dim))
            nn.init.zeros_(self.expand_proj.bias)
    
    def forward(
        self,
        x: Tensor,
        n_eff: Union[int, Tensor],
    ) -> Tensor:
        """
        Expand input to multi-stream representation.
        
        Args:
            x: Input of shape (batch, seq, hidden_dim)
            n_eff: Effective rank (int or tensor)
            
        Returns:
            x_bar: Expanded of shape (batch, seq, n_eff, hidden_dim)
        """
        batch, seq, d = x.shape
        
        # Get n_eff as int if tensor
        if isinstance(n_eff, Tensor):
            n_eff = int(n_eff.mean().item())
            n_eff = max(1, min(n_eff, self.max_n))
        
        if n_eff == 1:
            # Bypass: just add stream dimension
            return x.unsqueeze(2)  # (batch, seq, 1, d)
        
        # Full expansion
        x_expanded = self.expand_proj(x)  # (batch, seq, max_n * d)
        x_bar = x_expanded.view(batch, seq, self.max_n, d)  # (batch, seq, max_n, d)
        
        # Truncate to effective rank
        if n_eff < self.max_n:
            x_bar = x_bar[:, :, :n_eff, :]
            
        return x_bar
    
    def extra_repr(self) -> str:
        return f'hidden_dim={self.hidden_dim}, max_n={self.max_n}'

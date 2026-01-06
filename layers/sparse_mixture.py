"""
Sparse Orthogonal Mixture Layer

Implements the core SHC routing mechanism (Equation 7):
    H^res = Σ_i α_i(x) · Q_i

where:
    - α_i(x) = softmax(W_α · x) are input-dependent mixing weights
    - Q_i are orthogonal matrices generated via Cayley transform

This is the key innovation that replaces Sinkhorn normalization with
closed-form orthogonal routing.

Reference: Section 3.1 of the SHC paper
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shc.layers.cayley import BatchedCayleyTransform


class SparseOrthogonalMixture(nn.Module):
    """
    Sparse mixture of orthogonal matrices for SHC routing.
    
    Computes H^res = Σ_i α_i(x) · Q_i where:
        - k orthogonal matrices Q_i are parameterized via Cayley transform
        - Mixing weights α_i are input-dependent via learned projection
    
    Properties:
        - ρ(H^res) ≤ 1 by construction (Proposition 1)
        - Closed-form computation (no iteration)
        - Input-dependent routing for adaptive computation
    
    Args:
        n: Number of streams (size of routing matrix n x n)
        k: Number of mixture components (typically 2-3)
        hidden_dim: Dimension of input for computing mixing weights
        init_scale: Initialization scale for Cayley parameters
    
    Example:
        >>> mixer = SparseOrthogonalMixture(n=4, k=2, hidden_dim=768)
        >>> x = torch.randn(32, 128, 768)  # [batch, seq, hidden]
        >>> H_res = mixer(x)  # [batch, 4, 4] routing matrices
    """
    
    def __init__(
        self,
        n: int,
        k: int = 2,
        hidden_dim: int = 768,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.hidden_dim = hidden_dim
        
        # k orthogonal matrices via batched Cayley transform
        self.cayley = BatchedCayleyTransform(n=n, k=k, init_scale=init_scale)
        
        # Mixing weight predictor: maps hidden state → k weights
        self.mixing_proj = nn.Linear(hidden_dim, k)
        
        # Initialize mixing projection for near-uniform weights initially
        nn.init.xavier_uniform_(self.mixing_proj.weight, gain=0.1)
        nn.init.zeros_(self.mixing_proj.bias)
        
    def compute_mixing_weights(
        self, 
        x: Tensor, 
        temperature: float = 1.0
    ) -> Tensor:
        """
        Compute input-dependent mixing weights α(x).
        
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim) or (batch, hidden_dim)
            temperature: Softmax temperature (lower = more peaked)
            
        Returns:
            alpha: Mixing weights of shape (batch, k) summing to 1
        """
        # Pool over sequence if present
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        else:
            x_pooled = x  # (batch, hidden_dim)
            
        # Project to k logits
        logits = self.mixing_proj(x_pooled)  # (batch, k)
        
        # Softmax with temperature
        alpha = F.softmax(logits / temperature, dim=-1)  # (batch, k)
        
        return alpha
    
    def forward(
        self,
        x: Tensor,
        temperature: float = 1.0,
        return_components: bool = False,
    ) -> Tensor:
        """
        Compute sparse orthogonal mixture H^res = Σ_i α_i · Q_i.
        
        Args:
            x: Input tensor for computing mixing weights
               Shape: (batch, seq, hidden_dim) or (batch, hidden_dim)
            temperature: Softmax temperature for mixing weights
            return_components: If True, also return Q matrices and alpha weights
            
        Returns:
            H_res: Routing matrix of shape (batch, n, n)
            If return_components=True, also returns (Q_matrices, alpha)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Get k orthogonal matrices: (k, n, n)
        Q_matrices = self.cayley()
        
        # Ensure Q is on same device/dtype as input
        Q_matrices = Q_matrices.to(device=device, dtype=dtype)
        
        # Get input-dependent mixing weights: (batch, k)
        alpha = self.compute_mixing_weights(x, temperature)
        
        # Compute mixture: H_res = Σ_i α_i · Q_i
        # alpha: (batch, k) → (batch, k, 1, 1)
        # Q_matrices: (k, n, n) → (1, k, n, n)
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # (batch, k, 1, 1)
        Q_expanded = Q_matrices.unsqueeze(0)  # (1, k, n, n)
        
        # Weighted sum over k components
        H_res = (alpha_expanded * Q_expanded).sum(dim=1)  # (batch, n, n)
        
        if return_components:
            return H_res, Q_matrices, alpha
        return H_res
    
    def get_spectral_norm(self, x: Tensor) -> Tensor:
        """
        Compute spectral norm ρ(H^res) for given input.
        
        By Proposition 1, this should always be ≤ 1.
        
        Args:
            x: Input tensor for computing mixing weights
            
        Returns:
            Spectral norms of shape (batch,)
        """
        H_res = self.forward(x)  # (batch, n, n)
        # Compute max singular value for each batch
        return torch.linalg.svdvals(H_res)[:, 0]
    
    def get_mixing_entropy(self, x: Tensor) -> Tensor:
        """
        Compute entropy of mixing weights (measure of sparsity).
        
        Lower entropy = more concentrated on single component.
        
        Args:
            x: Input tensor
            
        Returns:
            Entropy values of shape (batch,)
        """
        alpha = self.compute_mixing_weights(x)  # (batch, k)
        # Entropy: -Σ α_i log(α_i)
        entropy = -torch.sum(alpha * torch.log(alpha + 1e-10), dim=-1)
        return entropy
    
    def get_max_alpha(self, x: Tensor) -> Tensor:
        """
        Get maximum mixing weight (measure of concentration).
        
        When max_alpha ≈ 1, H^res ≈ Q_j for some j (orthogonal).
        This explains why ρ ≈ 1 in practice (Section 4.3).
        
        Args:
            x: Input tensor
            
        Returns:
            Max alpha values of shape (batch,)
        """
        alpha = self.compute_mixing_weights(x)  # (batch, k)
        return alpha.max(dim=-1).values
    
    def apply_routing(self, x_bar: Tensor, H_res: Tensor) -> Tensor:
        """
        Apply routing matrix to multi-stream hidden state.
        
        Computes H^res @ x_bar for each batch.
        
        Args:
            x_bar: Multi-stream hidden state of shape (batch, n, d)
            H_res: Routing matrices of shape (batch, n, n)
            
        Returns:
            Routed hidden state of shape (batch, n, d)
        """
        # Batched matrix multiply: (batch, n, n) @ (batch, n, d) → (batch, n, d)
        return torch.bmm(H_res, x_bar)
    
    def extra_repr(self) -> str:
        return f'n={self.n}, k={self.k}, hidden_dim={self.hidden_dim}'


class TripleRoutingMatrices(nn.Module):
    """
    Three routing matrices (H^pre, H^res, H^post) for complete SHC block.
    
    As described in Algorithm 1:
        x_in = H^pre @ x_bar
        x_out = H^res @ x_bar + H^post @ f(x_in)
    
    Args:
        n: Number of streams
        k: Number of mixture components per routing matrix
        hidden_dim: Dimension for computing mixing weights
    """
    
    def __init__(
        self,
        n: int,
        k: int = 2,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.n = n
        self.k = k
        
        # Three routing matrices
        self.H_pre = SparseOrthogonalMixture(n, k, hidden_dim)
        self.H_res = SparseOrthogonalMixture(n, k, hidden_dim)
        self.H_post = SparseOrthogonalMixture(n, k, hidden_dim)
        
    def forward(
        self, 
        x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute all three routing matrices.
        
        Args:
            x: Input tensor for computing mixing weights
            
        Returns:
            Tuple of (H_pre, H_res, H_post), each of shape (batch, n, n)
        """
        return (
            self.H_pre(x),
            self.H_res(x),
            self.H_post(x),
        )
    
    def get_combined_spectral_norm(self, x: Tensor) -> Tensor:
        """
        Get spectral norms for all three routing matrices.
        
        Returns:
            Dict with 'pre', 'res', 'post' spectral norms
        """
        return {
            'pre': self.H_pre.get_spectral_norm(x),
            'res': self.H_res.get_spectral_norm(x),
            'post': self.H_post.get_spectral_norm(x),
        }

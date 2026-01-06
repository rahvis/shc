"""
Cayley Transform Layer

Implements closed-form orthogonal matrix generation via the Cayley transform:
    Q(A) = (I - A)(I + A)^{-1}
where A is a skew-symmetric matrix.

This replaces iterative Sinkhorn normalization with a single matrix inversion,
providing 16× speedup in routing computation.

Reference: Section 2.2 of the SHC paper
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CayleyTransform(nn.Module):
    """
    Generates an orthogonal matrix via the Cayley transform.
    
    The Cayley transform maps skew-symmetric matrices to orthogonal matrices:
        Q = (I - A)(I + A)^{-1}
    
    where A = -A^T is skew-symmetric, parameterized by n(n-1)/2 free parameters.
    
    Properties:
        - Q^T Q = I (orthogonal)
        - ρ(Q) = 1 (spectral norm exactly 1)
        - Closed-form (no iteration needed)
        - Differentiable (standard backprop through matrix inverse)
    
    Args:
        n: Size of the orthogonal matrix (n x n)
        init_scale: Scale for parameter initialization (default: 0.01)
    
    Example:
        >>> cayley = CayleyTransform(n=4)
        >>> Q = cayley()  # Returns 4x4 orthogonal matrix
        >>> torch.allclose(Q @ Q.T, torch.eye(4), atol=1e-5)
        True
    """
    
    def __init__(self, n: int, init_scale: float = 0.01):
        super().__init__()
        self.n = n
        self.num_params = n * (n - 1) // 2
        
        # Learnable parameters for the upper triangle of skew-symmetric matrix
        # Initialize near identity (small skew-symmetric → Q ≈ I)
        self.params = nn.Parameter(torch.randn(self.num_params) * init_scale)
        
        # Register buffer for indices (not trainable)
        triu_indices = torch.triu_indices(n, n, offset=1)
        self.register_buffer('triu_row', triu_indices[0])
        self.register_buffer('triu_col', triu_indices[1])
        
    def _build_skew_symmetric(self) -> Tensor:
        """
        Build skew-symmetric matrix A from learnable parameters.
        
        A skew-symmetric matrix satisfies A = -A^T.
        We parameterize only the upper triangle and mirror to lower.
        
        Returns:
            A: Skew-symmetric matrix of shape (n, n)
        """
        A = torch.zeros(self.n, self.n, device=self.params.device, dtype=self.params.dtype)
        
        # Fill upper triangle
        A[self.triu_row, self.triu_col] = self.params
        
        # Make skew-symmetric: A = A - A^T
        A = A - A.T
        
        return A
    
    def forward(self) -> Tensor:
        """
        Compute orthogonal matrix via Cayley transform.
        
        Q = (I - A)(I + A)^{-1}
        
        Returns:
            Q: Orthogonal matrix of shape (n, n) with ρ(Q) = 1
        """
        A = self._build_skew_symmetric()
        I = torch.eye(self.n, device=A.device, dtype=A.dtype)
        
        # Q = (I - A)(I + A)^{-1}
        # Use solve for numerical stability instead of explicit inverse
        Q = torch.linalg.solve(I + A, I - A)
        
        return Q
    
    def get_spectral_norm(self) -> Tensor:
        """
        Compute spectral norm ρ(Q) = max singular value.
        
        For orthogonal matrices, this should always be 1.0.
        
        Returns:
            Spectral norm (should be ≈ 1.0)
        """
        Q = self.forward()
        return torch.linalg.svdvals(Q)[0]
    
    def verify_orthogonality(self, atol: float = 1e-5) -> bool:
        """
        Verify that Q^T Q ≈ I.
        
        Args:
            atol: Absolute tolerance for comparison
            
        Returns:
            True if Q is orthogonal within tolerance
        """
        Q = self.forward()
        I = torch.eye(self.n, device=Q.device, dtype=Q.dtype)
        return torch.allclose(Q @ Q.T, I, atol=atol)
    
    def extra_repr(self) -> str:
        return f'n={self.n}, num_params={self.num_params}'


class BatchedCayleyTransform(nn.Module):
    """
    Batched Cayley transform for generating k orthogonal matrices.
    
    Efficiently generates k orthogonal matrices in parallel, used for
    the sparse mixture of orthogonal matrices in SHC.
    
    Args:
        n: Size of each orthogonal matrix (n x n)
        k: Number of orthogonal matrices to generate
        init_scale: Scale for parameter initialization
    
    Example:
        >>> batched_cayley = BatchedCayleyTransform(n=4, k=2)
        >>> Q_list = batched_cayley()  # Returns list of 2 orthogonal 4x4 matrices
    """
    
    def __init__(self, n: int, k: int, init_scale: float = 0.01):
        super().__init__()
        self.n = n
        self.k = k
        self.num_params_per_matrix = n * (n - 1) // 2
        
        # Learnable parameters for k skew-symmetric matrices
        self.params = nn.Parameter(
            torch.randn(k, self.num_params_per_matrix) * init_scale
        )
        
        # Register buffer for indices
        triu_indices = torch.triu_indices(n, n, offset=1)
        self.register_buffer('triu_row', triu_indices[0])
        self.register_buffer('triu_col', triu_indices[1])
        
    def _build_skew_symmetric_batch(self) -> Tensor:
        """
        Build k skew-symmetric matrices from parameters.
        
        Returns:
            A: Tensor of shape (k, n, n) containing k skew-symmetric matrices
        """
        A = torch.zeros(self.k, self.n, self.n, 
                       device=self.params.device, dtype=self.params.dtype)
        
        # Fill upper triangle for all k matrices
        A[:, self.triu_row, self.triu_col] = self.params
        
        # Make skew-symmetric
        A = A - A.transpose(-2, -1)
        
        return A
    
    def forward(self) -> Tensor:
        """
        Compute k orthogonal matrices via batched Cayley transform.
        
        Returns:
            Q: Tensor of shape (k, n, n) containing k orthogonal matrices
        """
        A = self._build_skew_symmetric_batch()
        I = torch.eye(self.n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(self.k, -1, -1)
        
        # Batched solve: Q = (I - A)(I + A)^{-1}
        Q = torch.linalg.solve(I + A, I - A)
        
        return Q
    
    def get_spectral_norms(self) -> Tensor:
        """
        Compute spectral norms for all k matrices.
        
        Returns:
            Tensor of shape (k,) containing spectral norms (all ≈ 1.0)
        """
        Q = self.forward()  # (k, n, n)
        # Compute max singular value for each matrix
        return torch.linalg.svdvals(Q)[:, 0]
    
    def extra_repr(self) -> str:
        return f'n={self.n}, k={self.k}, params_per_matrix={self.num_params_per_matrix}'

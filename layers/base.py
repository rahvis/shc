"""
Base Layer Module

Abstract base classes for SHC layers providing consistent interfaces
and enforcing implementation of key methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class BaseSHCLayer(nn.Module, ABC):
    """
    Abstract base class for all SHC layers.
    
    All SHC layers should inherit from this class to ensure consistent
    interfaces and enable type checking.
    
    Attributes:
        n: Dimension of the layer (typically number of streams).
    
    Example:
        >>> class CustomLayer(BaseSHCLayer):
        ...     def forward(self, x):
        ...         return x
        ...     def get_spectral_norm(self):
        ...         return 1.0
    """
    
    def __init__(self, n: int) -> None:
        """
        Initialize base SHC layer.
        
        Args:
            n: Dimension of the layer.
        """
        super().__init__()
        self.n = n
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward pass - must be implemented by subclasses.
        
        Returns:
            Output tensor(s) from the layer.
        """
        pass
    
    @abstractmethod
    def get_spectral_norm(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute the spectral norm of the layer's matrices.
        
        For SHC layers, this should be bounded by 1.0 to ensure
        training stability per Proposition 1 of the SHC paper.
        
        Returns:
            Spectral norm as a scalar tensor.
        """
        pass
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """
        Get diagnostic statistics for the layer.
        
        Returns:
            Dictionary containing layer statistics.
        """
        return {
            "n": self.n,
            "num_params": sum(p.numel() for p in self.parameters()),
        }
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"n={self.n}"


class BaseRoutingLayer(BaseSHCLayer):
    """
    Base class for routing layers (H^pre, H^res, H^post).
    
    Routing layers implement the orthogonal mixture routing mechanism
    described in the SHC paper.
    """
    
    def __init__(self, n: int, k: int, hidden_dim: int) -> None:
        """
        Initialize routing layer.
        
        Args:
            n: Number of streams.
            k: Number of orthogonal matrices in the mixture.
            hidden_dim: Hidden dimension for computing mixing weights.
        """
        super().__init__(n)
        self.k = k
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def compute_mixing_weights(
        self, 
        x: Tensor, 
        temperature: float = 1.0
    ) -> Tensor:
        """
        Compute input-dependent mixing weights α(x).
        
        Args:
            x: Input tensor.
            temperature: Softmax temperature.
            
        Returns:
            Mixing weights summing to 1.
        """
        pass
    
    def extra_repr(self) -> str:
        return f"n={self.n}, k={self.k}, hidden_dim={self.hidden_dim}"


class BaseCacheLayer(BaseSHCLayer):
    """
    Base class for KV cache compression layers.
    
    Cache layers implement the low-rank factorization for efficient
    KV cache storage during inference.
    """
    
    def __init__(self, n: int, d: int, r: int) -> None:
        """
        Initialize cache layer.
        
        Args:
            n: Number of streams.
            d: Hidden dimension.
            r: Factorization rank.
        """
        super().__init__(n)
        self.d = d
        self.r = r
    
    @abstractmethod
    def compress(self, x: Tensor) -> Tensor:
        """
        Compress multi-stream tensor to low-rank representation.
        
        Args:
            x: Input tensor of shape (batch, n, d).
            
        Returns:
            Compressed representation.
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed: Tensor) -> Tensor:
        """
        Decompress low-rank representation back to full tensor.
        
        Args:
            compressed: Compressed representation.
            
        Returns:
            Reconstructed tensor of shape (batch, n, d).
        """
        pass
    
    def get_compression_ratio(self) -> float:
        """
        Get the theoretical compression ratio.
        
        Returns:
            Compression ratio (original_size / compressed_size).
        """
        original = self.n * self.d
        compressed = self.r  # Simplified; actual may vary
        return original / max(compressed, 1)
    
    def extra_repr(self) -> str:
        return f"n={self.n}, d={self.d}, r={self.r}"

"""
SHC Layers Module

Core building blocks for Sparse Selective Hyper-Connections.

This module provides the fundamental layers used in SHC:
    - CayleyTransform: Closed-form orthogonal matrix generation
    - SparseOrthogonalMixture: Input-dependent routing with ρ ≤ 1
    - FactorizedKVCache: Low-rank KV cache compression
    - AdaptiveRankSelector: Gumbel-Softmax rank selection

Example:
    >>> from shc.layers import CayleyTransform, SparseOrthogonalMixture
    >>> 
    >>> # Create orthogonal matrix
    >>> cayley = CayleyTransform(n=4)
    >>> Q = cayley()  # 4x4 orthogonal matrix
    >>>
    >>> # Create routing layer
    >>> routing = SparseOrthogonalMixture(n=4, k=2, hidden_dim=768)
    >>> H_res = routing(x)  # Routing matrix with ρ ≤ 1
"""

from shc.layers.base import BaseSHCLayer, BaseRoutingLayer, BaseCacheLayer
from shc.layers.cayley import CayleyTransform, BatchedCayleyTransform
from shc.layers.sparse_mixture import SparseOrthogonalMixture, TripleRoutingMatrices
from shc.layers.factorized_cache import FactorizedKVCache
from shc.layers.adaptive_rank import AdaptiveRankSelector

__all__ = [
    # Base classes
    "BaseSHCLayer",
    "BaseRoutingLayer",
    "BaseCacheLayer",
    # Core layers
    "CayleyTransform",
    "BatchedCayleyTransform",
    "SparseOrthogonalMixture",
    "TripleRoutingMatrices",
    "FactorizedKVCache",
    "AdaptiveRankSelector",
]

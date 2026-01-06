"""
SHC Layers Module

Core building blocks for Sparse Selective Hyper-Connections.
"""

from shc.layers.cayley import CayleyTransform
from shc.layers.sparse_mixture import SparseOrthogonalMixture
from shc.layers.factorized_cache import FactorizedKVCache
from shc.layers.adaptive_rank import AdaptiveRankSelector

__all__ = [
    "CayleyTransform",
    "SparseOrthogonalMixture",
    "FactorizedKVCache",
    "AdaptiveRankSelector",
]

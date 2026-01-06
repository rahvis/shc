"""
Sparse Selective Hyper-Connections (SHC) Implementation

A PyTorch implementation of the SHC paper for efficient multi-stream
residual architectures with closed-form orthogonal routing.

Paper: "Sparse Selective Hyper-Connections: A Unified Framework for 
        Stable and Efficient Deep Residual Learning"

Key Components:
    - CayleyTransform: Closed-form orthogonal matrix generation
    - SparseOrthogonalMixture: Input-dependent routing matrices
    - FactorizedKVCache: Low-rank KV cache compression
    - SHCTransformer: Complete transformer model
    - SSMStudent: State-space model for O(L) inference
"""

__version__ = "0.1.0"
__author__ = "SHC Research Team"

# Core layers
from shc.layers import (
    CayleyTransform,
    SparseOrthogonalMixture, 
    FactorizedKVCache,
    AdaptiveRankSelector,
)

# Models (lazy import to avoid circular dependencies)
def get_shc_transformer():
    """Get SHCTransformer class."""
    from shc.models.transformer import SHCTransformer
    return SHCTransformer

def get_ssm_student():
    """Get SSMStudent class."""
    from shc.models.ssm_student import SSMStudent
    return SSMStudent

__all__ = [
    # Layers
    "CayleyTransform",
    "SparseOrthogonalMixture", 
    "FactorizedKVCache",
    "AdaptiveRankSelector",
    # Model getters
    "get_shc_transformer",
    "get_ssm_student",
]

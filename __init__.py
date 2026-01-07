"""
Sparse Selective Hyper-Connections (SHC) Implementation

A PyTorch implementation of the SHC paper for efficient multi-stream
residual architectures with closed-form orthogonal routing.

Paper: "Sparse Selective Hyper-Connections: A Unified Framework for 
        Stable and Efficient Deep Residual Learning"

Installation:
    pip install sparse-hyper-connections

Quick Start:
    >>> from shc.models import SHCTransformer, get_config
    >>> 
    >>> # Create model
    >>> config = get_config('500m')
    >>> model = SHCTransformer(config)
    >>> 
    >>> # Generate text
    >>> output = model.generate(input_ids, max_new_tokens=100)

Key Components:
    - CayleyTransform: Closed-form orthogonal matrix generation
    - SparseOrthogonalMixture: Input-dependent routing matrices
    - FactorizedKVCache: Low-rank KV cache compression
    - SHCTransformer: Complete transformer model
    - SSMStudent: State-space model for O(L) inference

Features:
    - Bounded spectral norm (ρ ≤ 1) for training stability
    - 16× faster routing via Cayley transform
    - 3.3× KV cache reduction
    - O(L) inference via SSM distillation
"""

__version__ = "0.1.2"
__author__ = "SHC Research Team"
__license__ = "MIT"

# Core layers
from shc.layers import (
    CayleyTransform,
    BatchedCayleyTransform,
    SparseOrthogonalMixture,
    TripleRoutingMatrices,
    FactorizedKVCache,
    AdaptiveRankSelector,
    BaseSHCLayer,
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

# Version info
def get_version():
    """Return package version string."""
    return __version__

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    "get_version",
    # Base classes
    "BaseSHCLayer",
    # Layers
    "CayleyTransform",
    "BatchedCayleyTransform",
    "SparseOrthogonalMixture",
    "TripleRoutingMatrices",
    "FactorizedKVCache",
    "AdaptiveRankSelector",
    # Model getters
    "get_shc_transformer",
    "get_ssm_student",
]

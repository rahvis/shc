"""
SHC Models Module

Complete model architectures built with SHC components.

This module provides ready-to-use models:
    - SHCTransformer: Main transformer with orthogonal routing
    - SSMStudent: State-space model for O(L) inference

Example:
    >>> from shc.models import SHCTransformer, get_config
    >>> 
    >>> # Create 500M parameter model
    >>> config = get_config('500m')
    >>> model = SHCTransformer(config)
    >>> 
    >>> # Generate text
    >>> output = model.generate(input_ids, max_new_tokens=100)
"""

from shc.models.base import BaseSHCModel
from shc.models.transformer import SHCTransformer, SHCTransformerConfig, get_config, CONFIGS
from shc.models.embeddings import TokenEmbedding, PositionalEmbedding, RMSNorm
from shc.models.ssm_student import SSMStudent, SSMConfig

__all__ = [
    # Base class
    "BaseSHCModel",
    # Main models
    "SHCTransformer",
    "SHCTransformerConfig",
    "SSMStudent",
    "SSMConfig",
    # Utilities
    "get_config",
    "CONFIGS",
    # Embeddings
    "TokenEmbedding",
    "PositionalEmbedding",
    "RMSNorm",
]

"""
SHC Configuration Classes

Dataclasses for model, training, and data configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SHCLayerConfig:
    """Configuration for SHC layers."""
    
    # Stream configuration
    n_streams: int = 4
    k_mixture: int = 2
    
    # Factorization
    factorization_rank: int = 1
    
    # Cayley transform
    cayley_init_scale: float = 0.01
    
    # Adaptive rank
    use_adaptive_rank: bool = True
    adaptive_temperature: float = 1.0
    min_temperature: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for SHC Transformer model."""
    
    # Architecture
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # SHC-specific
    shc: SHCLayerConfig = field(default_factory=SHCLayerConfig)
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Optimization
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    use_torch_compile: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Batch configuration
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    
    # Precision
    mixed_precision: str = "bf16"  # "fp32", "fp16", "bf16"
    
    # Distributed
    use_ddp: bool = True
    use_fsdp: bool = False
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Data source
    dataset_name: str = "cerebras/SlimPajama-627B"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    
    # Processing
    max_seq_len: int = 2048
    
    # Loading
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


# Predefined model configurations
MODEL_CONFIGS = {
    "500m": ModelConfig(
        hidden_dim=1024,
        n_layers=24,
        n_heads=16,
        ffn_dim=4096,
    ),
    "3b": ModelConfig(
        hidden_dim=2560,
        n_layers=32,
        n_heads=32,
        ffn_dim=10240,
    ),
    "7b": ModelConfig(
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        ffn_dim=11008,
    ),
}


def get_config(model_size: str = "500m") -> ModelConfig:
    """Get predefined model configuration."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size]

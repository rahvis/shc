"""
Base Model Module

Abstract base classes for SHC models with common functionality
for loading, saving, and inference.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import torch
import torch.nn as nn
from torch import Tensor


class BaseSHCModel(nn.Module, ABC):
    """
    Abstract base class for all SHC models.
    
    Provides common functionality for:
    - Model checkpointing (save/load)
    - Parameter counting
    - Memory estimation
    - Inference utilities
    
    Subclasses must implement forward() and generate() methods.
    
    Attributes:
        config: Model configuration dataclass.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize base SHC model.
        
        Args:
            config: Model configuration dataclass.
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask.
            labels: Optional labels for loss computation.
            **kwargs: Additional model-specific arguments.
            
        Returns:
            Logits or (loss, logits) if labels provided.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            **kwargs: Additional generation arguments.
            
        Returns:
            Generated token IDs.
        """
        pass
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            non_embedding: Exclude embedding parameters if True.
            
        Returns:
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'embedding'):
            n_params -= self.embedding.weight.numel()
        return n_params
    
    def get_memory_footprint(self, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """
        Estimate memory usage in GB.
        
        Args:
            dtype: Data type for estimation.
            
        Returns:
            Dictionary with memory breakdown.
        """
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(dtype, 4)
        
        total_params = sum(p.numel() for p in self.parameters())
        param_memory = total_params * bytes_per_param / (1024 ** 3)
        
        return {
            "parameters_gb": param_memory,
            "optimizer_gb": param_memory * 2,  # Adam states
            "total_estimated_gb": param_memory * 3,
        }
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "BaseSHCModel":
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint directory or file.
            **kwargs: Additional arguments passed to model constructor.
            
        Returns:
            Loaded model instance.
        """
        path = Path(path)
        
        # Handle both directory and file paths
        if path.is_dir():
            config_path = path / "config.json"
            weights_path = path / "model.pt"
        else:
            config_path = path.parent / "config.json"
            weights_path = path
        
        # Load config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        # Create model (subclass must handle config instantiation)
        # This is a template - subclasses should override
        raise NotImplementedError(
            "Subclasses must implement from_pretrained with proper config handling"
        )
    
    def save_pretrained(self, path: str) -> None:
        """
        Save model to checkpoint.
        
        Args:
            path: Path to save directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}
        with open(path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), path / "model.pt")
        
        # Save metadata
        metadata = {
            "num_params": self.get_num_params(),
            "model_class": self.__class__.__name__,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using standard scheme.
        
        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

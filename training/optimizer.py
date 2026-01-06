"""
Optimizer and Scheduler Utilities

Creates optimizers and learning rate schedulers with proper weight decay
handling and warmup support.
"""

from typing import Optional, List, Dict, Any, Iterable
import math

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    OneCycleLR,
    _LRScheduler,
)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with weight decay handling.
    
    Certain parameters (biases, LayerNorm) should not have weight decay
    applied to avoid training instability.
    
    Args:
        model: Model to get parameters from
        weight_decay: Weight decay value
        no_decay_keywords: Keywords to identify no-decay parameters
        
    Returns:
        List of parameter group dicts
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['bias', 'layernorm', 'layer_norm', 'ln_']
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should have no decay
        name_lower = name.lower()
        if any(nd in name_lower for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with proper parameter groups.
    
    Args:
        model: Model to optimize
        optimizer_type: 'adamw', 'adam', 'sgd'
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
        
    Example:
        >>> optimizer = create_optimizer(model, lr=3e-4, weight_decay=0.1)
    """
    param_groups = get_parameter_groups(model, weight_decay)
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adamw':
        return AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs,
        )
    elif optimizer_type == 'sgd':
        return SGD(
            param_groups,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    LR schedule:
        - Warmup: linear increase from 0 to peak_lr
        - Decay: cosine annealing to min_lr
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of peak LR (default: 0.1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_ratio = step / max(1, self.warmup_steps)
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]
        else:
            # Cosine decay
            decay_steps = self.max_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Scale between min_lr and peak_lr
            lr_range = 1.0 - self.min_lr_ratio
            multiplier = self.min_lr_ratio + lr_range * cosine_decay
            
            return [base_lr * multiplier for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum LR ratio at end of training
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_ratio = step / max(1, self.warmup_steps)
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]
        else:
            # Linear decay
            decay_steps = self.max_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, decay_steps)
            
            # Linear interpolation from 1.0 to min_lr_ratio
            multiplier = 1.0 - progress * (1.0 - self.min_lr_ratio)
            
            return [base_lr * multiplier for base_lr in self.base_lrs]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    warmup_steps: int = 2000,
    max_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs,
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: 'cosine', 'linear', 'constant', 'one_cycle'
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr_ratio: Minimum LR ratio
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == 'linear':
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == 'constant':
        # Warmup then constant
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],
            total_steps=max_steps,
            pct_start=warmup_steps / max_steps,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class GradientAccumulator:
    """
    Helper for gradient accumulation across multiple micro-batches.
    
    Enables training with large effective batch sizes on limited GPU memory.
    
    Args:
        accumulation_steps: Number of steps to accumulate
        
    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=8)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     accumulator.backward(loss)
        ...     if accumulator.should_step():
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def backward(self, loss: torch.Tensor):
        """Scale loss and backward."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.current_step += 1
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.current_step >= self.accumulation_steps
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0
        
    @property
    def is_accumulating(self) -> bool:
        """Check if currently accumulating (not ready to step)."""
        return self.current_step < self.accumulation_steps


class GradientClipper:
    """
    Gradient clipping utility with logging.
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2 for L2)
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self._last_norm = 0.0
        
    def clip(self, parameters: Iterable[torch.Tensor]) -> float:
        """
        Clip gradients and return the norm.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Total gradient norm before clipping
        """
        self._last_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_norm,
            norm_type=self.norm_type,
        )
        return self._last_norm
    
    @property
    def last_norm(self) -> float:
        """Get last computed gradient norm."""
        return self._last_norm

"""
SHC Training Module

Training infrastructure with multi-GPU support via DDP and FSDP.
"""

from shc.training.trainer import SHCTrainer, TrainingArgs
from shc.training.optimizer import create_optimizer, create_scheduler
from shc.training.distributed import setup_distributed, cleanup_distributed
from shc.training.distillation import DistillationTrainer, DistillationConfig

__all__ = [
    "SHCTrainer",
    "TrainingArgs",
    "create_optimizer",
    "create_scheduler",
    "setup_distributed",
    "cleanup_distributed",
    "DistillationTrainer",
    "DistillationConfig",
]

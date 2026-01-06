"""
Distributed Training Utilities

Setup and utilities for multi-GPU training via DDP and FSDP.
Supports NVIDIA GPUs with NCCL backend for optimal performance.
"""

from typing import Optional, Tuple
import os
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_distributed(
    backend: str = 'nccl',
    init_method: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Initialize distributed training environment.
    
    Supports:
        - torchrun / torch.distributed.launch
        - SLURM
        - Single-node multi-GPU
    
    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Optional initialization method URL
        
    Returns:
        Tuple of (rank, local_rank, world_size)
        
    Example:
        >>> rank, local_rank, world_size = setup_distributed()
        >>> torch.cuda.set_device(local_rank)
    """
    # Check if already initialized
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get('LOCAL_RANK', 0)), dist.get_world_size()
    
    # Get distributed environment variables
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Single GPU case - no distributed needed
    if world_size == 1:
        return rank, local_rank, world_size
    
    # Set CUDA device before init
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Determine init method
    if init_method is None:
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', str(get_free_port()))
        init_method = f'tcp://{master_addr}:{master_port}'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    
    # Synchronize
    dist.barrier()
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce tensor and compute mean across all processes.
    
    Args:
        tensor: Tensor to reduce
        
    Returns:
        Reduced tensor (averaged)
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def broadcast_object(obj, src: int = 0):
    """
    Broadcast Python object from source rank to all processes.
    
    Args:
        obj: Object to broadcast (only used on source rank)
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def sync_gradients(model: torch.nn.Module):
    """
    Synchronize gradients across all processes.
    
    Note: DDP handles this automatically, but useful for manual sync.
    """
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= get_world_size()


class DistributedContext:
    """
    Context manager for distributed training setup/cleanup.
    
    Example:
        >>> with DistributedContext() as ctx:
        ...     rank, local_rank, world_size = ctx.info
        ...     # Training code here
    """
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.info = None
        
    def __enter__(self):
        self.info = setup_distributed(self.backend)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
        return False


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        device_ids: GPU device IDs (defaults to local_rank)
        find_unused_parameters: Find unused parameters (slower but handles dynamic graphs)
        gradient_as_bucket_view: Memory optimization
        
    Returns:
        DDP-wrapped model
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if device_ids is None:
        device_ids = [local_rank]
    
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def wrap_model_fsdp(
    model: torch.nn.Module,
    mixed_precision: bool = True,
    sharding_strategy: str = 'FULL_SHARD',
    cpu_offload: bool = False,
):
    """
    Wrap model with Fully Sharded Data Parallel (FSDP).
    
    FSDP shards model parameters, gradients, and optimizer states
    across GPUs for memory-efficient training of large models.
    
    Args:
        model: Model to wrap
        mixed_precision: Use BF16/FP16 mixed precision
        sharding_strategy: 'FULL_SHARD', 'SHARD_GRAD_OP', or 'NO_SHARD'
        cpu_offload: Offload parameters to CPU when not in use
        
    Returns:
        FSDP-wrapped model
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from shc.blocks.shc_block import SHCBlock
    
    # Sharding strategy
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
        'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # CPU offload
    offload_policy = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Auto-wrap policy for SHC blocks
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={SHCBlock},
    )
    
    return FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        cpu_offload=offload_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

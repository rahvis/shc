"""
SHC DataLoader

High-performance data loading with:
- Distributed sampling
- Multi-worker parallelism
- Memory pinning for fast GPU transfer
- Prefetching for CPU-GPU overlap
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import os

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler


@dataclass
class DataLoaderConfig:
    """Configuration for data loader."""
    
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    shuffle: bool = True


class SHCDataLoader:
    """
    Wrapper for PyTorch DataLoader with distributed support.
    
    Features:
        - Automatic DistributedSampler for multi-GPU
        - Optimal worker configuration
        - Memory pinning for fast GPU transfer
        - Prefetching for CPU-GPU overlap
    
    Args:
        dataset: Dataset to load from
        config: DataLoader configuration
        is_distributed: Enable distributed sampling
        
    Example:
        >>> dataset = TokenizedDataset(data)
        >>> loader = SHCDataLoader(dataset, config)
        >>> for batch in loader:
        ...     pass
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: DataLoaderConfig,
        is_distributed: bool = False,
    ):
        self.dataset = dataset
        self.config = config
        self.is_distributed = is_distributed
        self.epoch = 0
        
        # Create sampler
        if is_distributed:
            self.sampler = DistributedSampler(
                dataset,
                shuffle=config.shuffle,
                drop_last=config.drop_last,
            )
        elif config.shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = None
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=self.sampler,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            drop_last=config.drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into tensors."""
        # Assuming batch is list of dicts with 'input_ids' and 'labels'
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler (for proper shuffling)."""
        self.epoch = epoch
        if self.is_distributed and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    is_distributed: bool = False,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> SHCDataLoader:
    """
    Create a dataloader with optimal settings.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of worker processes (auto if None)
        is_distributed: Enable distributed sampling
        shuffle: Shuffle data
        pin_memory: Pin memory for fast GPU transfer
        **kwargs: Additional config options
        
    Returns:
        Configured SHCDataLoader
    """
    # Auto-detect optimal worker count
    if num_workers is None:
        if torch.cuda.is_available():
            # Use ~4 workers per GPU, cap at available CPUs
            num_gpus = torch.cuda.device_count() or 1
            num_workers = min(4 * num_gpus, os.cpu_count() or 4)
        else:
            num_workers = min(4, os.cpu_count() or 4)
    
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        shuffle=shuffle,
        **kwargs,
    )
    
    return SHCDataLoader(dataset, config, is_distributed)


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> SHCDataLoader:
    """
    Create a dataloader for distributed training.
    
    Automatically creates DistributedSampler for multi-GPU training.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        **kwargs: Additional config options
        
    Returns:
        Distributed SHCDataLoader
    """
    import torch.distributed as dist
    
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=is_distributed,
        **kwargs,
    )


class InfiniteDataLoader:
    """
    Infinite data loader that cycles through dataset.
    
    Useful for step-based training where epochs are not meaningful.
    
    Args:
        dataloader: Base dataloader to cycle
        
    Example:
        >>> loader = InfiniteDataLoader(base_loader)
        >>> for step in range(max_steps):
        ...     batch = next(loader)
    """
    
    def __init__(self, dataloader: SHCDataLoader):
        self.dataloader = dataloader
        self._iterator = None
        self._epoch = 0
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        
        try:
            batch = next(self._iterator)
        except StopIteration:
            # Start new epoch
            self._epoch += 1
            self.dataloader.set_epoch(self._epoch)
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
        
        return batch
    
    @property
    def epoch(self) -> int:
        return self._epoch


class PrefetchLoader:
    """
    Prefetch data to GPU asynchronously.
    
    Overlaps data transfer with GPU computation for better throughput.
    
    Args:
        dataloader: Base dataloader
        device: Target device for prefetching
        
    Example:
        >>> loader = PrefetchLoader(base_loader, device='cuda:0')
        >>> for batch in loader:
        ...     # Batch is already on GPU
        ...     output = model(batch['input_ids'])
    """
    
    def __init__(
        self, 
        dataloader: SHCDataLoader,
        device: torch.device,
    ):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def _prefetch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prefetch batch to device asynchronously."""
        if self.stream is None:
            return {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.cuda.stream(self.stream):
            return {
                k: v.to(self.device, non_blocking=True) 
                for k, v in batch.items()
            }
    
    def __iter__(self):
        first = True
        next_batch = None
        
        for batch in self.dataloader:
            if first:
                first = False
                next_batch = self._prefetch(batch)
            else:
                # Wait for previous prefetch
                if self.stream:
                    torch.cuda.current_stream().wait_stream(self.stream)
                current_batch = next_batch
                next_batch = self._prefetch(batch)
                yield current_batch
        
        # Yield last batch
        if next_batch is not None:
            if self.stream:
                torch.cuda.current_stream().wait_stream(self.stream)
            yield next_batch
    
    def __len__(self) -> int:
        return len(self.dataloader)

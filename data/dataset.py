"""
Dataset Classes

Tokenized datasets for language model training with efficient memory usage.
"""

from typing import Optional, Dict, List, Any, Iterator
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, IterableDataset


class TokenizedDataset(Dataset):
    """
    Pre-tokenized dataset for language modeling.
    
    Stores tokenized sequences in memory for fast access.
    Best for smaller datasets that fit in RAM.
    
    Args:
        data: List of tokenized sequences (list of int lists)
        max_seq_len: Maximum sequence length (truncates longer)
        pad_token_id: Token ID for padding
        
    Example:
        >>> data = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]
        >>> dataset = TokenizedDataset(data, max_seq_len=6)
        >>> batch = dataset[0]  # {'input_ids': tensor([1,2,3,4,0,0]), 'labels': tensor([2,3,4,0,0,-100])}
    """
    
    def __init__(
        self,
        data: List[List[int]],
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
    ):
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Pad if needed
        padding_length = self.max_seq_len - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.pad_token_id] * padding_length
            
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels: shifted input_ids with padding masked
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.
    
    Lazily loads and tokenizes data from files/URLs.
    Supports distributed training with proper sharding.
    
    Args:
        data_paths: List of paths to data files
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
        shuffle: Shuffle data files/chunks
        seed: Random seed for shuffling
        
    Example:
        >>> dataset = StreamingDataset(
        ...     data_paths=['data/shard_00.txt', 'data/shard_01.txt'],
        ...     tokenizer=tokenizer,
        ... )
        >>> for batch in DataLoader(dataset, batch_size=32):
        ...     pass
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any,
        max_seq_len: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_paths = [Path(p) for p in data_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        
        # Distributed info (set when iterating)
        self._rank = 0
        self._world_size = 1
        
    def _get_worker_info(self):
        """Get worker info for proper data sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers
    
    def _shard_paths(self, paths: List[Path]) -> List[Path]:
        """Shard data paths across workers and ranks."""
        worker_id, num_workers = self._get_worker_info()
        
        # Shard across distributed ranks
        paths = paths[self._rank::self._world_size]
        
        # Shard across dataloader workers
        paths = paths[worker_id::num_workers]
        
        return paths
    
    def _read_and_tokenize(self, path: Path) -> Iterator[List[int]]:
        """Read file and yield tokenized chunks."""
        with open(path, 'r', encoding='utf-8') as f:
            buffer = []
            for line in f:
                tokens = self.tokenizer.encode(line.strip())
                buffer.extend(tokens)
                
                # Yield complete chunks
                while len(buffer) >= self.max_seq_len:
                    yield buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len:]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        import torch.distributed as dist
        
        # Update distributed info
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        
        # Get sharded paths
        paths = list(self.data_paths)
        if self.shuffle:
            # Deterministic shuffle based on epoch
            random.Random(self.seed).shuffle(paths)
        paths = self._shard_paths(paths)
        
        # Iterate over files
        for path in paths:
            for tokens in self._read_and_tokenize(path):
                input_ids = torch.tensor(tokens, dtype=torch.long)
                labels = input_ids.clone()
                
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                }


class PackedDataset(Dataset):
    """
    Packed dataset for efficient training.
    
    Concatenates multiple sequences into single tensors with document
    boundaries marked, avoiding padding waste.
    
    Args:
        sequences: List of tokenized sequences
        max_seq_len: Target sequence length
        eos_token_id: End-of-sequence token
        
    Example:
        >>> # Short sequences [1,2], [3,4,5], [6] packed into [1,2,EOS,3,4,5,EOS,6,PAD,PAD]
        >>> dataset = PackedDataset(sequences, max_seq_len=10, eos_token_id=2)
    """
    
    def __init__(
        self,
        sequences: List[List[int]],
        max_seq_len: int = 2048,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # Pack sequences
        self.packed_data = self._pack_sequences(sequences)
        
    def _pack_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pack sequences into fixed-length chunks."""
        packed = []
        current_chunk = []
        
        for seq in sequences:
            # Add EOS after each sequence
            seq_with_eos = seq + [self.eos_token_id]
            
            # Check if it fits in current chunk
            if len(current_chunk) + len(seq_with_eos) <= self.max_seq_len:
                current_chunk.extend(seq_with_eos)
            else:
                # Save current chunk if non-empty
                if current_chunk:
                    packed.append(current_chunk)
                
                # Start new chunk (handle long sequences)
                if len(seq_with_eos) > self.max_seq_len:
                    # Truncate long sequences
                    current_chunk = seq_with_eos[:self.max_seq_len]
                else:
                    current_chunk = seq_with_eos
        
        # Don't forget last chunk
        if current_chunk:
            packed.append(current_chunk)
        
        return packed
    
    def __len__(self) -> int:
        return len(self.packed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.packed_data[idx]
        
        # Pad if needed
        padding_length = self.max_seq_len - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.pad_token_id] * padding_length
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels with padding masked
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and debugging.
    
    Generates random token sequences for validation of training pipeline.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        vocab_size: Vocabulary size
        seed: Random seed
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 512,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate all data upfront for reproducibility
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (num_samples, seq_len), generator=rng
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

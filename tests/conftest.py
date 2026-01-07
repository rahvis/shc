"""
Pytest Configuration and Fixtures

Shared fixtures and configuration for all SHC tests.
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple


# Device selection
def get_device() -> torch.device:
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def device() -> torch.device:
    """Fixture providing test device."""
    return get_device()


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_len() -> int:
    """Standard sequence length for tests."""
    return 32


@pytest.fixture
def hidden_dim() -> int:
    """Standard hidden dimension for tests."""
    return 64


@pytest.fixture
def n_streams() -> int:
    """Standard number of streams for SHC."""
    return 4


@pytest.fixture
def n_heads() -> int:
    """Standard number of attention heads."""
    return 4


@pytest.fixture
def sample_input(batch_size: int, seq_len: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
    """Generate sample input tensor."""
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


@pytest.fixture
def sample_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate sample token IDs."""
    return torch.randint(0, 1000, (batch_size, seq_len), device=device)


@pytest.fixture
def small_vocab_size() -> int:
    """Small vocabulary for fast testing."""
    return 1000


# Test configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

# Installation

## Requirements

- **Python**: 3.9 or higher
- **PyTorch**: 2.0 or higher
- **CUDA**: 11.8+ (optional, for GPU training)

## Install from PyPI

The easiest way to install SHC is via pip:

```bash
pip install sparse-hyper-connections
```

## Install from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/your-org/shc.git
cd shc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Optional Dependencies

### Training Dependencies

For distributed training with DeepSpeed and TensorBoard:

```bash
pip install sparse-hyper-connections[training]
```

### Documentation Dependencies

To build the documentation locally:

```bash
pip install sparse-hyper-connections[docs]
```

### All Dependencies

Install everything:

```bash
pip install sparse-hyper-connections[all]
```

## Verifying Installation

After installation, verify everything works:

```python
import shc
print(f"SHC version: {shc.__version__}")

from shc.layers import CayleyTransform
import torch

# Test Cayley transform
cayley = CayleyTransform(n=4)
Q = cayley()
print(f"Orthogonal matrix shape: {Q.shape}")
print(f"Q^T @ Q ≈ I: {torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5)}")
```

Expected output:

```
SHC version: 0.1.1
Orthogonal matrix shape: torch.Size([4, 4])
Q^T @ Q ≈ I: True
```

## GPU Support

SHC automatically uses GPU if available:

```python
import torch
from shc.models import SHCTransformer, get_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SHCTransformer(get_config('500m')).to(device)
```

## Troubleshooting

### CUDA Out of Memory

For large models (7B+), use gradient checkpointing:

```python
config = get_config('7b')
config.use_gradient_checkpointing = True
model = SHCTransformer(config)
```

### Import Errors

Ensure PyTorch is installed correctly:

```bash
python -c "import torch; print(torch.__version__)"
```

If you encounter `ModuleNotFoundError`, reinstall:

```bash
pip uninstall sparse-hyper-connections
pip install sparse-hyper-connections --no-cache-dir
```

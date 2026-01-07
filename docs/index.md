# Sparse Hyper-Connections Documentation

```{image} https://img.shields.io/pypi/v/sparse-hyper-connections.svg
:target: https://pypi.org/project/sparse-hyper-connections/
:alt: PyPI version
```
```{image} https://img.shields.io/badge/python-3.9+-blue.svg
:target: https://www.python.org/downloads/
:alt: Python 3.9+
```
```{image} https://img.shields.io/badge/pytorch-2.0+-orange.svg
:target: https://pytorch.org/
:alt: PyTorch 2.0+
```
```{image} https://img.shields.io/badge/License-MIT-green.svg
:target: https://github.com/your-org/shc/blob/main/LICENSE
:alt: License: MIT
```

**Sparse Selective Hyper-Connections (SHC)** is a practical efficiency framework for multi-stream residual architectures that achieves substantial computational and memory improvements while maintaining equivalent accuracy.

## Key Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🎯 Guaranteed Stability
Bounded spectral norm **ρ ≤ 1** by construction via closed-form Cayley transform, ensuring stable training at any depth.
:::

:::{grid-item-card} ⚡ 16× Faster Routing
Replace iterative Sinkhorn normalization with closed-form orthogonal matrix generation via the Cayley transform.
:::

:::{grid-item-card} 💾 3.3× Less Memory
Factorized KV cache compression reduces memory from 4× to ~1.2× baseline through learned low-rank projections.
:::

:::{grid-item-card} 📈 O(L) Inference
Optional SSM distillation enables linear-time generation without KV cache, trading ~1% accuracy for 4.4× memory reduction.
:::

::::

## Quick Installation

```bash
pip install sparse-hyper-connections
```

## Quick Start

```python
from shc.models import SHCTransformer, get_config

# Create model with predefined configuration
config = get_config('500m')  # Options: '500m', '1b', '3b', '7b'
model = SHCTransformer(config)

# Forward pass
import torch
input_ids = torch.randint(0, 32000, (2, 512))
logits = model(input_ids)

# Generate text
output = model.generate(
    input_ids[:, :10],  # prompt
    max_new_tokens=100,
    temperature=0.7,
)
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

theory
architecture
training
inference
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/layers
api/models
api/training
api/evaluation
```

```{toctree}
:maxdepth: 1
:caption: Development

changelog
contributing
```

## Citation

If you use SHC in your research, please cite:

```bibtex
@article{shc2026,
  title={Sparse Selective Hyper-Connections: A Unified Framework for 
         Stable and Efficient Deep Residual Learning},
  author={SHC Research Team},
  journal={IEEE Conference},
  year={2026}
}
```

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

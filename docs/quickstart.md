# Quick Start Guide

This guide will help you get started with SHC in just a few minutes.

## Creating Your First Model

```python
from shc.models import SHCTransformer, get_config

# Load a predefined configuration
config = get_config('500m')  # Options: '500m', '1b', '3b', '7b'

# Create the model
model = SHCTransformer(config)

# Print model info
print(f"Parameters: {model.get_num_params():,}")
```

## Forward Pass

```python
import torch

# Create sample input
batch_size, seq_len = 2, 512
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # (2, 512, 32000)
```

## Text Generation

```python
# Generate text from a prompt
prompt = torch.randint(0, config.vocab_size, (1, 10))

output = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

print(f"Generated {output.shape[1] - prompt.shape[1]} new tokens")
```

## Using Individual Components

### Cayley Transform

The Cayley transform generates orthogonal matrices with spectral norm exactly 1:

```python
from shc.layers import CayleyTransform

# Create a 4x4 orthogonal matrix generator
cayley = CayleyTransform(n=4)

# Generate orthogonal matrix
Q = cayley()
print(f"Q^T @ Q ≈ I: {torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5)}")
print(f"Spectral norm: {cayley.get_spectral_norm():.4f}")  # Always 1.0
```

### Sparse Orthogonal Mixture

The routing layer that guarantees ρ ≤ 1:

```python
from shc.layers import SparseOrthogonalMixture

# Create routing layer with k=2 orthogonal matrices
routing = SparseOrthogonalMixture(n=4, k=2, hidden_dim=768)

# Compute routing matrix for input
x = torch.randn(2, 768)  # batch_size=2, hidden_dim=768
H_res = routing(x)       # (2, 4, 4)

# Verify spectral norm bound
norms = routing.get_spectral_norm(x)
print(f"Max spectral norm: {norms.max():.4f}")  # Always ≤ 1.0
```

### Factorized KV Cache

Compress multi-stream representations:

```python
from shc.layers import FactorizedKVCache

# Create factorized cache with rank-1 compression
cache = FactorizedKVCache(n=4, d=768, r=1)

# Compress multi-stream hidden state
x_bar = torch.randn(2, 4, 768)  # (batch, n_streams, hidden_dim)
compressed = cache.compress(x_bar)

# Decompress when needed
reconstructed = cache.decompress(compressed)
print(f"Compression ratio: {cache.get_compression_ratio():.1f}x")
```

## Custom Model Configuration

```python
from shc.models import SHCTransformer, SHCTransformerConfig

# Create custom configuration
config = SHCTransformerConfig(
    vocab_size=32000,
    hidden_dim=1024,
    n_layers=24,
    n_heads=16,
    max_seq_len=4096,
    n_streams=4,         # Number of parallel residual streams
    k_mixture=2,         # Number of orthogonal matrices in mixture
    factorization_rank=1, # Rank for KV cache compression
)

model = SHCTransformer(config)
```

## GPU Training

```python
import torch

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create data on GPU
input_ids = torch.randint(0, 32000, (2, 512), device=device)
labels = torch.randint(0, 32000, (2, 512), device=device)

# Forward pass with loss
outputs = model(input_ids, labels=labels)
```

## Next Steps

- Read the [Theory](theory.md) section to understand the mathematics
- Learn about [Training](training.md) for large-scale training
- Explore the [API Reference](api/layers.md) for detailed documentation

# Architecture

The SHC Transformer architecture extends the standard transformer with multi-stream residual connections using sparse orthogonal routing.

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                    SHC Transformer                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Input: token_ids (batch, seq_len)                     │
│              ↓                                           │
│   ┌─────────────────────┐                               │
│   │   Token Embedding   │  vocab_size → hidden_dim      │
│   └─────────────────────┘                               │
│              ↓                                           │
│   ┌─────────────────────┐                               │
│   │    N × SHC Block    │  With orthogonal routing      │
│   └─────────────────────┘                               │
│              ↓                                           │
│   ┌─────────────────────┐                               │
│   │      RMS Norm       │                               │
│   └─────────────────────┘                               │
│              ↓                                           │
│   ┌─────────────────────┐                               │
│   │     LM Head         │  hidden_dim → vocab_size      │
│   └─────────────────────┘                               │
│              ↓                                           │
│   Output: logits (batch, seq_len, vocab_size)           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## SHC Block

Each SHC Block implements Algorithm 1 from the paper:

```
SHC Block Forward Pass
──────────────────────
Input: x ∈ ℝ^d, layer index l

1. n_eff ← AdaptiveRank(x, l)         # Adaptive stream expansion

2. IF n_eff > 1:
   a. x̄ ← StreamExpand(x, n_eff)     # Expand to n streams
   b. α ← softmax(W_α · x̄)           # Compute mixing weights
   c. H^res ← Σ αᵢ · Q(Aᵢ)           # Cayley routing matrix
   
   d. x̄_out ← H^res · x̄ + H^post · f(H^pre · x̄)
              ↑           ↑         ↑
           residual    output    input
           routing     routing   routing
   
   e. x_out ← Compress(x̄_out, r=1)   # Factorized cache

3. ELSE:
   x_out ← x + f(x)                   # Standard residual

Return: x_out
```

## Model Configurations

| Size | Hidden Dim | Layers | Heads | FFN Dim | Parameters |
|------|------------|--------|-------|---------|------------|
| 500M | 1024 | 24 | 16 | 4096 | ~500M |
| 1B | 2048 | 24 | 16 | 8192 | ~1B |
| 3B | 2560 | 32 | 32 | 10240 | ~3B |
| 7B | 4096 | 32 | 32 | 11008 | ~7B |

```python
from shc.models import get_config, SHCTransformer

# Load predefined configuration
config = get_config('3b')
model = SHCTransformer(config)
```

## Core Components

### CayleyTransform

Generates orthogonal matrices with exactly $\rho = 1$:

```python
from shc.layers import CayleyTransform

cayley = CayleyTransform(n=4, init_scale=0.01)
Q = cayley()  # 4×4 orthogonal matrix
```

### SparseOrthogonalMixture

Input-dependent mixture of $k$ orthogonal matrices:

```python
from shc.layers import SparseOrthogonalMixture

routing = SparseOrthogonalMixture(
    n=4,           # Number of streams
    k=2,           # Number of orthogonal matrices
    hidden_dim=768 # Dimension for computing mixing weights
)

H_res = routing(x)  # (batch, n, n) routing matrix
```

### FactorizedKVCache

Low-rank compression for efficient caching:

```python
from shc.layers import FactorizedKVCache

cache = FactorizedKVCache(
    n=4,    # Number of streams
    d=768,  # Hidden dimension
    r=1     # Factorization rank
)
```

### AdaptiveRankSelector

Layer-wise and input-dependent effective rank:

```python
from shc.layers import AdaptiveRankSelector

selector = AdaptiveRankSelector(n=4, hidden_dim=768)
n_eff = selector(x)  # Effective number of streams
```

## Multi-Head Attention

Standard multi-head attention with RoPE positional encoding:

```python
from shc.blocks import MultiHeadAttention

attention = MultiHeadAttention(
    hidden_dim=768,
    n_heads=12,
    max_seq_len=4096,
    use_rope=True
)
```

## Feed-Forward Network

SwiGLU activation with learnable gating:

```python
from shc.blocks import FeedForward

ffn = FeedForward(
    hidden_dim=768,
    ffn_dim=3072  # Typically 4× hidden_dim
)
```

## Generation

Autoregressive generation with KV caching:

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
```

## SSM Distillation

For O(L) inference, distill into a State Space Model:

```python
from shc.models import SSMStudent
from shc.training import DistillationTrainer

# Create student matching teacher dimensions
student = SSMStudent.from_teacher_config(teacher.config)

# Distill
trainer = DistillationTrainer(teacher, student, config, data)
trainer.train()

# Student generates without KV cache
output = student.generate(input_ids, max_new_tokens=100)
```

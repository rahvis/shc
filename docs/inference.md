# Inference Guide

This guide covers efficient inference with SHC models.

## Basic Inference

```python
from shc.models import SHCTransformer

# Load model
model = SHCTransformer.from_pretrained('path/to/model')
model.eval()

# Move to GPU
device = torch.device('cuda')
model = model.to(device)

# Generate
import torch
prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)
output = model.generate(prompt, max_new_tokens=100)
```

## Generation Parameters

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,     # Maximum tokens to generate
    temperature=0.7,         # Higher = more random
    top_k=50,               # Top-k sampling
    top_p=0.9,              # Nucleus sampling
    do_sample=True,         # Enable sampling (vs greedy)
    eos_token_id=2,         # Stop token
    pad_token_id=0,         # Padding token
)
```

## KV Cache Efficiency

SHC uses factorized KV caching by default:

| Configuration | Cache Size | Memory @ 32K |
|---------------|------------|--------------|
| Baseline Transformer | 1× | 24.8 GB |
| mHC (4 streams) | 4× | 99.2 GB |
| **SHC (factorized)** | **1.2×** | **29.8 GB** |

## Batch Inference

```python
# Batch generation
prompts = torch.tensor([
    [1, 2, 3, 4, 5, 0, 0],  # padded
    [1, 2, 3, 0, 0, 0, 0],
], device=device)

attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
], device=device)

outputs = model.generate(
    prompts,
    attention_mask=attention_mask,
    max_new_tokens=50,
)
```

## SSM Inference (O(L))

For linear-time inference without KV cache:

```python
from shc.models import SSMStudent

# Load distilled model
student = SSMStudent.from_pretrained('path/to/student')
student.eval()

# Generate (no KV cache needed!)
output = student.generate(prompt, max_new_tokens=100)
```

### Memory Comparison

| Mode | BBH | MMLU | Memory |
|------|-----|------|--------|
| Full Attention | 51.3% | 63.6% | 18.4 GB |
| SSM Distilled | 50.8% | 63.1% | 4.2 GB |

## Routing Analysis

Analyze SHC routing behavior:

```python
# Get routing statistics
stats = model.get_routing_stats(input_ids)

for layer_idx, layer_stats in stats.items():
    print(f"Layer {layer_idx}:")
    print(f"  Spectral norm: {layer_stats['spectral_norm']:.4f}")
    print(f"  Max alpha: {layer_stats['max_alpha']:.4f}")
```

## Profiling

```python
from shc.evaluation import EfficiencyProfiler

profiler = EfficiencyProfiler(model)
results = profiler.profile_inference(
    batch_size=1,
    seq_len=2048,
    num_warmup=5,
    num_runs=20,
)

print(f"Latency: {results['latency_ms']:.2f} ms")
print(f"Throughput: {results['tokens_per_second']:.0f} tok/s")
print(f"Memory: {results['peak_memory_gb']:.2f} GB")
```

## Long-Context Inference

SHC excels at long contexts due to factorized caching:

```python
# 32K context
config = get_config('3b')
config.max_seq_len = 32768
model = SHCTransformer(config)

# Generate with long context
long_prompt = torch.randint(0, 32000, (1, 16000), device=device)
output = model.generate(long_prompt, max_new_tokens=1000)
```

## Deployment Recommendations

1. **Standard Deployment**: Use factorized cache (default)
2. **Memory-Constrained**: Use SSM distilled model
3. **Long-Context**: SHC shines vs 4× cache of mHC
4. **Batch Processing**: Use batch inference for throughput

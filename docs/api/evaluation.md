# Evaluation API Reference

Benchmark suite and efficiency profiling.

## BenchmarkSuite

Run standard LLM benchmarks.

```{eval-rst}
.. autoclass:: shc.evaluation.BenchmarkSuite
   :members:
   :undoc-members:
   :show-inheritance:
```

## BenchmarkResult

Result container for benchmarks.

```{eval-rst}
.. autoclass:: shc.evaluation.BenchmarkResult
   :members:
   :undoc-members:
```

## EfficiencyProfiler

Profile inference efficiency.

```{eval-rst}
.. autoclass:: shc.evaluation.EfficiencyProfiler
   :members:
   :undoc-members:
```

## MemoryProfiler

Profile memory usage.

```{eval-rst}
.. autoclass:: shc.evaluation.MemoryProfiler
   :members:
   :undoc-members:
```

## Metric Functions

### compute_perplexity

```{eval-rst}
.. autofunction:: shc.evaluation.compute_perplexity
```

### compute_accuracy

```{eval-rst}
.. autofunction:: shc.evaluation.compute_accuracy
```

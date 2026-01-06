"""
SHC Evaluation Module

Benchmark suite and efficiency profiling for SHC models.
"""

from shc.evaluation.benchmarks import BenchmarkSuite, BenchmarkResult
from shc.evaluation.profiler import EfficiencyProfiler, MemoryProfiler
from shc.evaluation.metrics import compute_perplexity, compute_accuracy

__all__ = [
    "BenchmarkSuite",
    "BenchmarkResult",
    "EfficiencyProfiler",
    "MemoryProfiler",
    "compute_perplexity",
    "compute_accuracy",
]

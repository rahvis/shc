# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Core SHC layers (CayleyTransform, SparseOrthogonalMixture, FactorizedKVCache)
- SHCTransformer model with configurable scales (500M, 1B, 3B, 7B)
- SSM student model for O(L) inference via distillation
- Multi-GPU training support (DDP, FSDP)
- Benchmark evaluation suite (BBH, GSM8K, MMLU)
- Comprehensive documentation and examples

## [0.1.0] - 2026-01-07

### Added
- Initial implementation of Sparse Selective Hyper-Connections
- Closed-form Cayley transform for orthogonal matrix generation
- Sparse mixture of orthogonal matrices with bounded spectral norm
- Factorized KV cache with adaptive rank selection
- Training infrastructure with gradient checkpointing
- Evaluation metrics and profiling tools

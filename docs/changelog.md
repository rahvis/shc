# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-01-07

### Added
- Comprehensive Sphinx documentation
- Theory documentation with mathematical foundations
- Architecture documentation with diagrams
- Training and inference guides
- API reference for all modules

### Changed
- Updated package structure for better organization
- Enhanced docstrings throughout codebase

## [0.1.0] - 2026-01-07

### Added
- Initial public release on PyPI
- Core SHC layers (CayleyTransform, SparseOrthogonalMixture, FactorizedKVCache)
- SHCTransformer model with configurable scales (500M, 1B, 3B, 7B)
- SSM student model for O(L) inference via distillation
- Multi-GPU training support (DDP, FSDP)
- Benchmark evaluation suite (BBH, GSM8K, MMLU)
- Comprehensive test suite
- GitHub Actions CI/CD

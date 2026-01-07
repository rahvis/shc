# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-01-07

### Added
- Comprehensive Sphinx documentation hosted on ReadTheDocs
- Theory documentation with mathematical foundations from the paper
- Architecture, training, and inference guides
- Full API reference for all modules
- GitHub Actions CI/CD workflows for testing and publishing

### Changed
- Updated project URLs to point to official GitHub repository
- Enhanced `__init__.py` files with better exports and docstrings

### Fixed
- ReadTheDocs build configuration

## [0.1.0] - 2026-01-07

### Added
- Initial public release on PyPI
- Core SHC layers (CayleyTransform, SparseOrthogonalMixture, FactorizedKVCache)
- SHCTransformer model with configurable scales (500M, 1B, 3B, 7B)
- SSM student model for O(L) inference via distillation
- Multi-GPU training support (DDP, FSDP)
- Benchmark evaluation suite (BBH, GSM8K, MMLU)
- Abstract base classes for layers and models
- Test suite with pytest fixtures
- Example scripts for basic usage, training, and distillation

[Unreleased]: https://github.com/rahvis/shc/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/rahvis/shc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rahvis/shc/releases/tag/v0.1.0

# Contributing to SHC

Thank you for your interest in contributing to SHC! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/shc.git
   cd shc
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing

Run all checks:
```bash
# Format code
black shc tests
isort shc tests

# Type check
mypy shc

# Run tests
pytest tests -v
```

## Testing

All new features should include tests. Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=shc --cov-report=html

# Run specific test file
pytest tests/test_layers/test_cayley.py -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with appropriate tests
3. Ensure all tests pass and code is formatted
4. Update documentation if needed
5. Submit a pull request with a clear description

## Reporting Issues

When reporting issues, please include:

- Python and PyTorch versions
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

## Questions?

Open a GitHub issue or discussion for questions about contributing.

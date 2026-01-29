# Contributing Guide

Thank you for your interest in the Markitai project! This document will help you understand how to participate in project development.

## Development Environment Setup

### Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Clone the Repository

```bash
git clone https://github.com/Ynewtime/markitai.git
cd markitai
```

### Install Dependencies

```bash
# Install all dependencies (including dev dependencies)
uv sync

# Install optional LLM provider SDKs
uv sync --all-extras
```

### Install Pre-commit Hooks

```bash
uv run pre-commit install
```

### Verify Installation

```bash
# Run tests
uv run pytest

# Run lint
uv run ruff check

# Run type checking
uv run pyright
```

---

## Code Style Guidelines

### Python Version

- Target version: Python 3.13
- Compatible versions: Python 3.11+

### Formatting and Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting:

```bash
# Check
uv run ruff check

# Auto-fix
uv run ruff check --fix

# Format
uv run ruff format
```

### Type Annotations

- All functions must have type annotations
- Use modern syntax: `str | None` instead of `Optional[str]`
- Add `from __future__ import annotations` at the beginning of files

```python
from __future__ import annotations

def process(text: str | None = None) -> dict[str, Any]:
    ...
```

### Docstrings

Use Google style:

```python
def convert(path: Path, options: ConvertOptions) -> ConversionResult:
    """Convert a file to Markdown.

    Args:
        path: Path to the input file.
        options: Conversion options.

    Returns:
        The conversion result containing markdown content and metadata.

    Raises:
        FileNotFoundError: If the input file does not exist.
        UnsupportedFormatError: If the file format is not supported.
    """
```

### Import Order

Managed automatically by Ruff, following isort rules:

```python
# Standard library
import asyncio
from pathlib import Path

# Third-party libraries
import click
from pydantic import BaseModel

# Local modules
from markitai.config import get_config
```

---

## Commit Guidelines

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation update |
| `style` | Code formatting (no functional changes) |
| `refactor` | Refactoring (no new features or bug fixes) |
| `perf` | Performance optimization |
| `test` | Test-related changes |
| `chore` | Build/toolchain/CI |

### Examples

```
feat: add support for EPUB format conversion

- Add EpubConverter class
- Register .epub extension
- Add unit tests

Closes #123
```

```
fix: handle empty PDF pages correctly

Previously, empty pages would cause an IndexError.
Now they are skipped with a warning.
```

---

## Pull Request Workflow

### 1. Create a Branch

```bash
# Feature branch
git checkout -b feat/your-feature

# Fix branch
git checkout -b fix/your-fix
```

### 2. Development

- Write code
- Add tests
- Update documentation (if needed)

### 3. Local Verification

```bash
# Run all checks
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

### 4. Commit

```bash
git add .
git commit -m "feat: your feature description"
```

### 5. Push and Create PR

```bash
git push -u origin feat/your-feature
```

Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Code passes all CI checks
- [ ] New features have corresponding tests
- [ ] Documentation is updated (if needed)
- [ ] Commit messages follow conventions
- [ ] PR description clearly explains the changes

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run with coverage
uv run pytest --cov=markitai

# Run in parallel
uv run pytest -n auto
```

### Quick Tests (Skip Slow Tests)

Some tests are slower due to OCR processing or network requests. Use pytest markers to skip them:

```bash
# Skip slow tests (e.g., OCR processing ~40s)
uv run pytest -m "not slow"

# Skip network-dependent tests
uv run pytest -m "not network"

# Skip both slow and network tests
uv run pytest -m "not slow and not network"

# Run only CLI tests (fast)
uv run pytest packages/markitai/tests/integration/test_cli_full.py
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests that take >10s (e.g., OCR) |
| `@pytest.mark.network` | Tests requiring network access |

### Test Structure

```
packages/markitai/tests/
├── unit/              # Unit tests
│   ├── test_config.py
│   ├── test_llm.py
│   └── ...
├── integration/       # Integration tests
│   ├── test_cli.py
│   └── ...
└── conftest.py        # Shared fixtures
```

### Writing Tests

```python
import pytest
from markitai.config import resolve_env_value

class TestResolveEnvValue:
    def test_plain_value(self):
        """Plain values should be returned as-is."""
        assert resolve_env_value("hello") == "hello"

    def test_env_value(self, monkeypatch):
        """env: prefix should resolve environment variables."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert resolve_env_value("env:TEST_VAR") == "test_value"

    def test_missing_env_strict(self):
        """Missing env var should raise in strict mode."""
        with pytest.raises(EnvVarNotFoundError):
            resolve_env_value("env:NONEXISTENT", strict=True)
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

---

## Documentation Requirements

### Code Documentation

- All public functions must have docstrings
- Complex logic requires inline comments
- Use type annotations to enhance readability

### User Documentation

User documentation is located in the `website/` directory:

```
website/
├── guide/
│   ├── getting-started.md
│   ├── configuration.md
│   └── cli.md
└── zh/                # Chinese version
    └── guide/
```

### Building Documentation

```bash
cd website
pnpm install
pnpm docs:dev    # Development mode
pnpm docs:build  # Build
```

---

## Project Structure

```
markitai/
├── packages/markitai/     # Main package
│   ├── src/markitai/      # Source code
│   │   ├── cli/           # CLI package
│   │   ├── llm/           # LLM integration package
│   │   ├── providers/     # Custom LLM providers
│   │   ├── converter/     # Format converters
│   │   ├── workflow/      # Processing workflows
│   │   └── utils/         # Utilities
│   └── tests/             # Tests
├── scripts/               # Setup scripts
├── docs/                  # Internal documentation
├── website/               # User documentation
├── pyproject.toml         # Workspace configuration
└── CONTRIBUTING.md        # This file
```

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Ynewtime/markitai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ynewtime/markitai/discussions)

---

## License

By submitting a Pull Request, you agree that your contributions will be licensed under the project's MIT license.

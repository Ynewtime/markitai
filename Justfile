# Justfile - Cross-platform task runner
# Install: cargo install just / brew install just / choco install just / scoop install just
# Usage: just <recipe> or just --list

set shell := ["bash", "-c"]

# Default recipe: show available commands
default:
    @just --list

# Run all CI checks (same as GitHub Actions)
ci: lint typecheck test
    @echo "✅ All CI checks passed!"

# Lint with ruff
lint:
    uv run ruff check .

# Format with ruff
format:
    uv run ruff format .

# Type check with pyright
typecheck:
    uv run pyright src/markit

# Run tests
test:
    uv run pytest

# Run tests with coverage
test-cov:
    uv run pytest --cov=src/markit --cov-report=html

# Build package
build:
    uv build

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage coverage.xml

# Install pre-commit hooks
install-hooks:
    pre-commit install
    pre-commit install --hook-type pre-push

# Update pre-commit hooks
update-hooks:
    pre-commit autoupdate

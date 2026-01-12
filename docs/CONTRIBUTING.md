# Contributing Guide

This guide covers development setup, architecture overview, and contribution guidelines for MarkIt.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Development Setup

```bash
# Clone the repository
git clone https://github.com/user/markit.git
cd markit

# Create virtual environment and install dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

## Development Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_pipeline.py

# Run specific test with verbose output
pytest tests/unit/test_pipeline.py::test_function_name -v

# Code linting and formatting
ruff check .
ruff format .

# Type checking
pyright src/markit

# Pre-commit hooks (recommended)
pre-commit install                    # Install pre-commit hook
pre-commit install --hook-type pre-push  # Install pre-push hook
pre-commit run --all-files            # Run all checks manually

# Justfile commands (cross-platform task runner)
# Install: cargo install just / brew install just / choco install just / scoop install just
just --list      # List all available commands
just ci          # Run all CI checks (lint + typecheck + test)
just test-cov    # Run tests with coverage report
just clean       # Clean build artifacts
just build       # Build package
```

## Architecture Overview

MarkIt uses a modular, service-oriented architecture with clear separation of concerns.

### Core Pipeline Flow

```
Input File → FormatRouter → Preprocessor → Converter → ImageProcessingService → LLMOrchestrator → OutputManager → Output
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ConversionPipeline                                │
│                                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ FormatRouter    │  │ ImageProcessing  │  │ LLMOrchestrator            │  │
│  │                 │  │ Service          │  │                            │  │
│  │ - Route files   │  │ - Compression    │  │ - ProviderManager          │  │
│  │ - Select        │  │ - Deduplication  │  │ - MarkdownEnhancer         │  │
│  │   converter     │  │ - Format convert │  │ - ImageAnalyzer            │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OutputManager                               │    │
│  │                                                                     │    │
│  │  - Conflict resolution (rename/overwrite/skip)                      │    │
│  │  - Write markdown + assets                                          │    │
│  │  - Generate image description .md files                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **FormatRouter** (`markit/core/router.py`): Routes files to appropriate converters
   - PDF: Routes to pymupdf4llm/pymupdf/pdfplumber based on config
   - Legacy formats (.doc, .ppt, .xls): Adds OfficePreprocessor for LibreOffice conversion
   - Modern Office/HTML: Uses MarkItDown converter

2. **ConversionPipeline** (`markit/core/pipeline.py`): Main orchestrator
   - Document conversion with fallback support
   - Delegates image processing to ImageProcessingService
   - Delegates LLM operations to LLMOrchestrator
   - Writes output via OutputManager

### Service Layer

- **ImageProcessingService** (`markit/services/image_processor.py`): Handles image format conversion, compression (via oxipng/Pillow), deduplication, and prepares images for LLM analysis

- **LLMOrchestrator** (`markit/services/llm_orchestrator.py`): Centralizes all LLM operations:
  - Manages ProviderManager for multi-provider support
  - Creates MarkdownEnhancer for text cleanup
  - Creates ImageAnalyzer for vision tasks
  - Implements capability-based routing (text models vs vision models)

- **OutputManager** (`markit/services/output_manager.py`): Handles file writing, conflict resolution, generates image description markdown files

### LLM Provider System

**ProviderManager** (`markit/llm/manager.py`): Manages multiple LLM providers:
- Lazy initialization (validates providers on demand)
- Capability-based routing (text vs vision tasks)
- Automatic fallback on failure
- Concurrent fallback (starts backup model when primary times out)
- Round-robin load balancing
- Per-model cost tracking

Supported providers: OpenAI, Anthropic, Gemini, Ollama, OpenRouter (all in `markit/llm/`)

### Configuration System

Settings defined in `markit/config/settings.py` using pydantic-settings:
- **LLMConfig**: Supports both legacy single-provider and new credential/model separation
- **LLMCredentialConfig**: Provider credentials (can reference environment variables)
- **LLMModelConfig**: Model instances referencing credentials, with capability declarations

Configuration loaded from `markit.yaml` in current or parent directories.

## Key Design Patterns

### 1. Phased Pipeline

For batch processing, the pipeline is divided into phases:
- **Phase 1**: `convert_document_only()` - CPU-intensive conversion, releases file semaphore early
- **Phase 2**: `create_llm_tasks()` - Creates coroutines for LLM queue
- **Phase 3**: `finalize_output()` - Merges results and writes output

### 2. Capability-Based Routing

Models declare capabilities (`["text"]` or `["text", "vision"]`). Pure text tasks route to cheaper text models; vision tasks only route to vision-capable models.

### 3. AIMD Adaptive Concurrency

Implements Additive Increase Multiplicative Decrease algorithm:
- After N consecutive successes, concurrency increases by 1
- On 429 rate limit, concurrency multiplies by 0.5
- Cooldown period prevents oscillation

### 4. Dead Letter Queue (DLQ)

Tracks failures per file:
- Records failure count and last error
- Marks files as permanently failed after max retries
- Prevents "poison files" from blocking resume operations

### 5. LibreOffice Profile Pool

`markit/converters/libreoffice_pool.py`: Uses isolated LibreOffice profile directories to avoid conflicts during parallel .doc/.ppt/.xls conversion.

### 6. Process Pool for Images

Heavy image compression uses process pool to bypass Python GIL.

## Project Structure

```
.
├── src/markit/        # Source code (src layout)
│   ├── cli/           # Typer CLI commands (convert, batch, config, provider, model)
│   ├── config/        # Settings, constants
│   ├── converters/    # Format converters (markitdown, pandoc, pdf/)
│   ├── core/          # Pipeline, router, state management
│   ├── image/         # Image analysis, compression, extraction
│   ├── llm/           # LLM providers (openai, anthropic, gemini, ollama, openrouter)
│   ├── markdown/      # Markdown processing (chunker, formatter, frontmatter)
│   ├── services/      # Service layer (image_processor, llm_orchestrator, output_manager)
│   └── utils/         # Utilities (concurrency, fs, logging, stats)
├── tests/
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   ├── e2e/           # End-to-end tests (require external services)
│   └── fixtures/      # Test fixtures
├── docs/              # Documentation
└── .github/workflows/ # CI/CD
```

## Testing

### Unit Tests

```bash
# Run all unit tests (excludes e2e by default)
pytest

# Run with coverage
pytest --cov=src/markit --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/
```

### E2E Tests (End-to-End)

```bash
# Run e2e tests (requires API keys or local services)
pytest -m e2e
```

## Documentation Sync Rules

When updating documentation, keep these files in sync:
- `README.md` ↔ `docs/README_ZH.md`: Content must stay synchronized
- `CLAUDE.md` ↔ `AGENTS.md`: Development guidelines must be consistent
- `docs/ROADMAP.md`: Update task progress promptly
- `docs/CONTRIBUTING.md` ↔ `docs/CONTRIBUTING_ZH.md`: Keep in sync

## Code Style

- Follow existing code patterns
- Use `ruff` for linting and formatting
- Use `mypy` for type checking
- All async I/O should use `anyio`
- Use `structlog` for logging with context

# MarkIt - Product Requirements Document

**Version**: 1.0.2 | **Status**: Active

## Overview

MarkIt is a CLI tool for batch converting office documents to Markdown with optional LLM-powered enhancement.

**Target Users**: Technical writers, knowledge base admins, content migration engineers

**Core Value**:
- Multi-format support (Office, PDF, HTML, images)
- LLM-powered format cleanup and image analysis
- Concurrent processing with resume capability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (Typer)                        │
├─────────────────────────────────────────────────────────────┤
│               Configuration (pydantic-settings)             │
├─────────────────────────────────────────────────────────────┤
│                   Processing Pipeline                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Converter │→ │  Image   │→ │   LLM    │→ │ Markdown │     │
│  │  Engine  │  │Processor │  │ Enhancer │  │  Writer  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
├──────────────────┬──────────────────────────────────────────┤
│ Converters       │ LLM Providers                            │
│ • MarkItDown     │ • OpenAI      • Anthropic                │
│ • PyMuPDF4LLM    │ • Gemini      • Ollama                   │
│ • pdfplumber     │ • OpenRouter                             │
│ • LibreOffice    │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

## Project Structure

```
markit/
├── cli/
│   ├── main.py              # Typer app
│   └── commands/
│       ├── convert.py       # Single file conversion
│       ├── batch.py         # Batch conversion
│       ├── config.py        # Config management
│       └── provider.py      # LLM provider commands
├── config/
│   ├── settings.py          # pydantic-settings config
│   └── constants.py         # Constants
├── core/
│   ├── pipeline.py          # Main processing pipeline
│   ├── router.py            # Format router
│   └── state.py             # Batch state (resume)
├── converters/
│   ├── base.py              # Converter base class
│   ├── markitdown.py        # MarkItDown wrapper
│   ├── office.py            # LibreOffice conversion
│   └── pdf/
│       ├── pymupdf4llm.py   # Default PDF engine
│       ├── pymupdf.py
│       └── pdfplumber.py
├── image/
│   ├── extractor.py         # Image extraction
│   ├── compressor.py        # PNG/JPEG compression
│   └── analyzer.py          # LLM image analysis
├── llm/
│   ├── base.py              # Provider base class
│   ├── manager.py           # Provider manager + fallback
│   ├── enhancer.py          # Markdown enhancement
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   ├── ollama.py
│   └── openrouter.py
├── markdown/
│   ├── formatter.py
│   ├── frontmatter.py
│   └── chunker.py           # Large file chunking
└── utils/
    ├── logging.py           # structlog setup
    ├── concurrency.py
    └── fs.py
```

## Supported Formats

| Format | Extensions | Primary Engine | Fallback |
|--------|------------|----------------|----------|
| Word | .docx | MarkItDown | - |
| Word (legacy) | .doc | LibreOffice → MarkItDown | - |
| PowerPoint | .pptx | MarkItDown | - |
| PowerPoint (legacy) | .ppt | LibreOffice → MarkItDown | - |
| Excel | .xlsx | MarkItDown | - |
| Excel (legacy) | .xls | LibreOffice → MarkItDown | - |
| PDF | .pdf | PyMuPDF4LLM | PyMuPDF, pdfplumber |
| HTML | .html, .htm | MarkItDown | - |
| CSV | .csv | MarkItDown | - |
| Images | .png, .jpg, .gif, .webp, .bmp | LLM analysis | - |

## CLI Commands

### convert

```bash
markit convert [OPTIONS] INPUT_FILE
```

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory | `./output` |
| `--llm` | Enable LLM enhancement | disabled |
| `--analyze-image` | Generate alt text | disabled |
| `--analyze-image-with-md` | Generate .md description files | disabled |
| `--no-compress` | Disable image compression | enabled |
| `--pdf-engine` | pymupdf4llm, pymupdf, pdfplumber, markitdown | pymupdf4llm |
| `--llm-provider` | openai, anthropic, gemini, ollama, openrouter | first available |
| `--llm-model` | Model name | provider default |
| `-v, --verbose` | Verbose output | disabled |
| `--dry-run` | Show plan only | disabled |

### batch

```bash
markit batch [OPTIONS] INPUT_DIR
```

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --recursive` | Process subdirectories | disabled |
| `--include` | Include glob pattern | all supported |
| `--exclude` | Exclude glob pattern | none |
| `--file-concurrency` | Concurrent files | 4 |
| `--image-concurrency` | Concurrent images | 8 |
| `--llm-concurrency` | Concurrent LLM requests | 5 |
| `--on-conflict` | skip, overwrite, rename | rename |
| `--resume` | Resume interrupted batch | disabled |
| `--state-file` | State file path | `.markit-state.json` |

### provider

```bash
markit provider add       # Add LLM provider credential
markit provider test      # Test LLM connectivity
markit provider list      # List configured credentials
markit provider fetch     # Fetch available models from providers
```

### model

```bash
markit model add          # Interactive wizard to add a model
markit model list         # List configured models
```

## Configuration

**Priority**: CLI args > Environment vars > markit.yaml > Defaults

### markit.yaml

```yaml
log_level: "INFO"
log_dir: ".logs"

output:
  default_dir: "output"
  on_conflict: "rename"

image:
  enable_compression: true
  filter_small_images: true

concurrency:
  file_workers: 4
  image_workers: 8
  llm_workers: 5

pdf:
  engine: "pymupdf4llm"

# New Schema (recommended for multiple models)
# Define credentials separately from models
llm:
  credentials:
    - id: "openai-main"
      provider: "openai"
      # api_key: "sk-..."  # Or use OPENAI_API_KEY env var

  models:
    - name: "GPT-4o"
      model: "gpt-4o"
      credential_id: "openai-main"
      capabilities: ["text", "vision"]
      timeout: 120
```

### Environment Variables

```bash
# API Keys (no MARKIT_ prefix)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...

# Settings (MARKIT_ prefix)
MARKIT_LOG_LEVEL=DEBUG
MARKIT_LOG_FILE=logs/markit.log
MARKIT_CONCURRENCY__FILE_WORKERS=8
```

## LLM Features

### Markdown Enhancement (`--llm`)

- Insert YAML frontmatter (title, source, processed date)
- Remove headers/footers and junk content
- Fix heading levels (start from h2)
- Normalize blank lines
- Follow GFM specification

### Image Analysis (`--analyze-image`)

- Generate alt text for each extracted image
- Single LLM call per image

### Image Description Files (`--analyze-image-with-md`)

Generates `<image>.md` alongside each image:

```markdown
---
source_image: diagram_001.png
image_type: diagram
generated_at: 2026-01-08T12:00:00Z
---

## Alt Text
A flowchart showing the data processing pipeline.

## Detailed Description
This diagram illustrates a three-stage pipeline...

## Detected Text
"Input" -> "Process" -> "Output"
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| CLI | Typer |
| Config | pydantic-settings |
| Conversion | MarkItDown, PyMuPDF4LLM, LibreOffice |
| Image | Pillow, oxipng |
| Async | anyio, httpx |
| LLM SDKs | openai, anthropic, google-genai, ollama |
| Text chunking | langchain-text-splitters, tiktoken |
| Logging | structlog |

## Dependencies

- Python 3.12+
- LibreOffice (for .doc, .ppt, .xls)
- oxipng (optional, for PNG compression)

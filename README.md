# Markitai

English | [简体中文](./README_ZH.md)

Opinionated Markdown converter with native LLM enhancement support.

## Features

- **Multi-format Support** - DOCX/DOC, PPTX/PPT, XLSX/XLS, PDF, TXT, MD, JPG/PNG/WebP, URLs
- **LLM Enhancement** - Format cleaning, metadata generation, image analysis
- **Batch Processing** - Concurrent conversion, resume capability, progress display
- **OCR Recognition** - Text extraction from scanned PDFs and images
- **URL Conversion** - Direct webpage conversion with SPA browser rendering support
- **Smart Caching** - LLM result caching, SPA domain learning, auto-proxy detection

## Installation

### One-Click Setup (Recommended)

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```

### Manual Installation

```bash
# Requires Python 3.11-3.13 (3.14 not yet supported)
uv tool install markitai

# Or using uv pip (for virtual environment)
uv pip install markitai
```

## Quick Start

### First Run

```bash
# Interactive mode (recommended for new users)
markitai -I

# Or convert a file directly
markitai document.pdf

# With LLM enhancement
markitai document.pdf --llm
```

### Check Setup

```bash
# Verify all dependencies
markitai doctor

# Auto-fix missing components
markitai doctor --fix
```

### Common Tasks

```bash
# Basic conversion
markitai document.docx

# URL conversion
markitai https://example.com/article

# LLM enhancement
markitai document.docx --llm

# Using presets
markitai document.pdf --preset rich      # LLM + alt + desc + screenshot
markitai document.pdf --preset standard  # LLM + alt + desc
markitai document.pdf --preset minimal   # Basic conversion only

# Batch processing
markitai ./docs -o ./output

# Resume interrupted job
markitai ./docs -o ./output --resume

# Batch URL processing (auto-detect .urls files)
markitai urls.urls -o ./output
```

## Output Structure

```
output/
├── document.docx.md        # Basic Markdown
├── document.docx.llm.md    # LLM-enhanced version
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # Image descriptions
├── screenshots/            # Page screenshots (with --screenshot)
│   └── example_com.full.jpg
```

## Configuration

Priority: CLI arguments > Environment variables > Config file > Defaults

```bash
# View configuration
markitai config list

# Initialize config file
markitai config init -o .

# View cache status
markitai cache stats

# Clear cache
markitai cache clear

# Check system health and dependencies
markitai doctor
```

Config file location: `./markitai.json` or `~/.markitai/config.json`

### Local Providers (Subscription-based)

Use your existing Claude Code or GitHub Copilot subscription:

```bash
# Claude Agent (requires Claude Code CLI)
markitai document.pdf --llm  # Configure claude-agent/sonnet in config

# GitHub Copilot (requires Copilot CLI)
markitai document.pdf --llm  # Configure copilot/gpt-5.2 in config
```

Install CLI tools:
```bash
# Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# GitHub Copilot CLI
curl -fsSL https://gh.io/copilot-install | bash
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `JINA_API_KEY` | Jina Reader API Key (URL conversion) |

## Dependencies

- [pymupdf4llm](https://github.com/pymupdf/RAG) - PDF conversion
- [markitdown](https://github.com/microsoft/markitdown) - Office documents and URL conversion
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM gateway
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR recognition

## Documentation

- [Getting Started](https://markitai.ynewtime.com/guide/getting-started)
- [Configuration](https://markitai.ynewtime.com/guide/configuration)
- [CLI Reference](https://markitai.ynewtime.com/guide/cli)

## License

MIT

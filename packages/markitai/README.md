# Markitai

English | [简体中文](./README_ZH.md)

Opinionated Markdown converter with native LLM enhancement support.

## Features

- **Multi-format Support** - DOCX/DOC, PPTX/PPT, XLSX/XLS, PDF, HTML, EPUB, CSV, TXT, MD, JPG/PNG/WebP/GIF/BMP/TIFF, URLs, and 10+ more via optional converters
- **LLM Enhancement** - Format cleaning, metadata generation, image analysis
- **Local Providers** - Use existing Claude Code, GitHub Copilot, ChatGPT, or Gemini CLI subscriptions — no API keys needed
- **Batch Processing** - Concurrent conversion, resume capability, progress display
- **OCR Recognition** - Text extraction from scanned PDFs and images
- **URL Conversion** - Smart strategy chain (Defuddle → Jina → Static → Playwright → Cloudflare) with SPA auto-detection
- **Cloudflare Integration** - Cloud-based URL rendering (Browser Rendering) and file conversion (Workers AI toMarkdown) via `--cloudflare`
- **Smart Caching** - LLM result caching, SPA domain learning, auto-proxy detection
- **Fetch Security** - Configurable strategy priority, domain/IP exemption with NO_PROXY support for information security compliance

## Installation

### One-Click Setup (Recommended)

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex"
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

# Cloudflare cloud rendering
markitai https://example.com --cloudflare

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
├── document.docx.md            # Basic Markdown (skipped in --llm mode unless --keep-base)
├── document.docx.llm.md        # LLM-enhanced version (when --llm is used)
├── .markitai/                   # Metadata namespace (isolated from user content)
│   ├── assets/
│   │   ├── document.docx.0001.jpg
│   │   └── images.json         # Image descriptions
│   ├── screenshots/            # Page screenshots (with --screenshot)
│   │   └── example_com.full.jpg
│   ├── reports/                # Conversion reports (JSON)
│   └── states/                 # Batch state files (for --resume)
```

> In `--llm` mode, only `.llm.md` is written by default. Use `--keep-base` to also write the base `.md`.

## Configuration

Priority: CLI arguments > Environment variables > Config file > Defaults

```bash
# View configuration
markitai config list

# Initialize config file
markitai init

# View cache status
markitai cache stats

# Clear cache
markitai cache clear

# Check system health and dependencies
markitai doctor
```

Config file location: `./markitai.json` or `~/.markitai/config.json`

### Local Providers (Subscription-based)

Use your existing subscriptions — no API keys needed:

```bash
# Claude Agent (requires Claude Code CLI)
markitai document.pdf --llm  # Configure claude-agent/sonnet in config

# GitHub Copilot (requires Copilot CLI)
markitai document.pdf --llm  # Configure copilot/gpt-5.2 in config

# ChatGPT (OAuth Device Code — no SDK needed)
markitai auth login chatgpt  # One-time browser login
markitai document.pdf --llm  # Configure chatgpt/gpt-5.2 in config

# Gemini CLI (reuses ~/.gemini/oauth_creds.json)
markitai document.pdf --llm  # Configure gemini-cli/gemini-2.5-pro in config
```

Install CLI tools (for claude-agent / copilot):
```bash
# Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# GitHub Copilot CLI
curl -fsSL https://gh.io/copilot-install | bash
```

Check provider authentication status:
```bash
markitai auth status
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `JINA_API_KEY` | Jina Reader API Key (URL conversion) |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API Token (Browser Rendering / Workers AI) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare Account ID |

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

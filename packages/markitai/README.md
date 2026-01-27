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

# Or using pip
pip install --user markitai
```

## Quick Start

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
```

Config file location: `./markitai.json` or `~/.markitai/config.json`

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

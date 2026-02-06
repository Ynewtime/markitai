# Getting Started

## Prerequisites

### Required

- **Python 3.11-3.13** - Required runtime (3.14 not yet supported due to onnxruntime)
- **[uv](https://docs.astral.sh/uv/)** - Package manager (recommended)

### Optional Dependencies

These are required for specific features:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| **Playwright** | `--playwright` (SPA rendering) | Package auto-installed, browser requires `uv run playwright install chromium` |
| **FFmpeg** | Audio/video processing | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| **Jina API Key** | `--jina` (URL conversion) | Set `JINA_API_KEY` env var |
| **LLM API Key** | `--llm` (AI enhancement) | Set `OPENAI_API_KEY` or provider-specific key |

::: tip Browser Automation
For SPA websites (Twitter, React apps, etc.), Playwright is used automatically. Before first use, install the browser:
```bash
uv run playwright install chromium
# Linux also requires system dependencies:
uv run playwright install-deps chromium
```
Then use `--playwright` flag to force browser rendering.
:::

## Installation

### One-Click Setup (Recommended)

Run the setup script to automatically install Python, UV, and markitai:

::: code-group
```bash [Linux/macOS]
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh
```

```powershell [Windows]
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```
:::

::: warning Security Notice
- The script will warn you if running as root/Administrator
- All installations require explicit confirmation (default: No)
- Remote script execution requires two-step confirmation
:::

The script will:
- Check for Python 3.11-3.13
- Install [uv](https://docs.astral.sh/uv/) package manager (requires confirmation)
- Install markitai with all optional dependencies

#### Version Pinning

Pin specific versions using environment variables:

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.5.0"
export UV_VERSION="0.9.27"
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.5.0"
$env:UV_VERSION = "0.9.27"
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```
:::

### Manual Installation

```bash
# Using uv (recommended)
uv tool install markitai

# Or using uv pip (for virtual environment)
uv pip install markitai
```

## Quick Start

### First Run

For new users, the interactive mode guides you through initial setup:

```bash
markitai -I
```

Or initialize configuration with the setup wizard:

```bash
# Interactive setup wizard
markitai init

# Quick mode (generate default config)
markitai init --yes
```

### Basic Conversion

Convert a single document to Markdown:

```bash
markitai document.docx
```

Without `-o`, this prints the output to stdout. With `-o`, it saves to the specified directory:

```bash
markitai document.docx -o output/
```

### URL Conversion

Convert web pages directly:

```bash
markitai https://example.com/article -o output/
```

### LLM Enhancement

Enable AI-powered format cleaning and optimization:

```bash
markitai document.docx --llm
```

This requires setting up an API key (see [Configuration](/guide/configuration)).

### Using Presets

Markitai provides three presets for common use cases:

```bash
# Rich: LLM + alt text + descriptions + screenshots
markitai document.pdf --preset rich

# Standard: LLM + alt text + descriptions
markitai document.pdf --preset standard

# Minimal: Basic conversion only
markitai document.pdf --preset minimal
```

### Batch Processing

Convert multiple files in a directory:

```bash
markitai ./docs -o ./output
```

Resume interrupted batch processing:

```bash
markitai ./docs -o ./output --resume
```

### System Check

Verify all dependencies and auto-fix missing components:

```bash
# Check system health
markitai doctor

# Auto-fix missing components
markitai doctor --fix
```

## Output Structure

```
output/
├── document.docx.md        # Basic Markdown output
├── document.docx.llm.md    # LLM-enhanced version (when --llm is used)
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # Image descriptions
├── screenshots/             # Page screenshots (when --screenshot is used)
│   └── document.docx.0001.png
```

## Supported Formats

| Format | Extensions |
|--------|------------|
| Word | `.docx`, `.doc` |
| PowerPoint | `.pptx`, `.ppt` |
| Excel | `.xlsx`, `.xls` |
| PDF | `.pdf` |
| Text | `.txt`, `.md`, `.markdown` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp` |
| URLs | `http://`, `https://` |

## Platform-Specific Features

Some features have platform-specific behavior or limitations:

### Windows

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ✅ Full | Uses COM automation |
| PPTX Slide Rendering | ✅ Full | MS Office preferred, LibreOffice fallback |
| EMF/WMF Images | ✅ Full | Native support |
| Browser Automation | ✅ Full | Hidden window mode |

### Linux

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ✅ Full | Requires LibreOffice (Windows uses COM, LibreOffice as fallback) |
| PPTX Slide Rendering | ✅ Full | Requires LibreOffice |
| EMF/WMF Images | ❌ No | Windows-only format |
| Browser Automation | ✅ Full | Requires system dependencies |

**Install LibreOffice:**
```bash
# Ubuntu/Debian
sudo apt-get install libreoffice

# Fedora/RHEL
sudo dnf install libreoffice
```

**Install Playwright browsers:**
```bash
uv run playwright install chromium
uv run playwright install-deps chromium  # Install system dependencies
```

### macOS

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ✅ Full | Requires LibreOffice |
| PPTX Slide Rendering | ✅ Full | Requires LibreOffice |
| EMF/WMF Images | ❌ No | Windows-only format |
| Browser Automation | ✅ Full | - |

**Install LibreOffice:**
```bash
brew install --cask libreoffice
```

## Next Steps

- [Configuration](/guide/configuration) - Configure LLM providers and other settings
- [CLI Reference](/guide/cli) - Full command reference

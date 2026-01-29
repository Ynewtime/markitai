# Getting Started

## Prerequisites

### Required

- **Python 3.11-3.13** - Required runtime (3.14 not yet supported due to onnxruntime)
- **[uv](https://docs.astral.sh/uv/)** - Package manager (recommended) or pip

### Optional Dependencies

These are required for specific features:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| **[Node.js 22+](https://nodejs.org/)** | `--agent-browser` (SPA rendering) | See [nodejs.org](https://nodejs.org/) |
| **[agent-browser](https://www.npmjs.com/package/agent-browser)** | `--agent-browser` | `pnpm add -g agent-browser && agent-browser install` |
| **Jina API Key** | `--jina` (URL conversion) | Set `JINA_API_KEY` env var |
| **LLM API Key** | `--llm` (AI enhancement) | Set `OPENAI_API_KEY` or provider-specific key |

::: tip Browser Automation
For SPA websites (Twitter, React apps, etc.), install `agent-browser`:
```bash
pnpm add -g agent-browser
agent-browser install              # Download Chromium browser
agent-browser install --with-deps  # Linux: also install system dependencies
```
Then use `--agent-browser` flag to enable browser rendering.
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
- Optionally install `agent-browser` for browser automation

#### Version Pinning

Pin specific versions using environment variables:

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.4.0"
export UV_VERSION="0.9.27"
export AGENT_BROWSER_VERSION="0.5.0"
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.4.0"
$env:UV_VERSION = "0.9.27"
$env:AGENT_BROWSER_VERSION = "0.5.0"
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```
:::

### Manual Installation

```bash
# Using uv (recommended)
uv tool install markitai

# Or using pip
pip install --user markitai
```

## Quick Start

### Basic Conversion

Convert a single document to Markdown:

```bash
markitai document.docx
```

This saves the output to the current directory and prints it to stdout. Use `-o` to specify a different output directory:

```bash
markitai document.docx -o output/
```

### URL Conversion

Convert web pages directly:

```bash
markitai https://example.com/article
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

## Output Structure

```
output/
├── document.docx.md        # Basic Markdown output
├── document.docx.llm.md    # LLM-enhanced version (when --llm is used)
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # Image descriptions
```

## Supported Formats

| Format | Extensions |
|--------|------------|
| Word | `.docx`, `.doc` |
| PowerPoint | `.pptx`, `.ppt` |
| Excel | `.xlsx`, `.xls` |
| PDF | `.pdf` |
| Text | `.txt`, `.md` |
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

::: tip Windows Performance
On Windows, markitai automatically limits concurrency to 4 threads due to higher thread-switching overhead.
:::

### Linux

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ❌ No | Requires Windows COM |
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

**Install browser dependencies:**
```bash
agent-browser install --with-deps
```

### macOS

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ❌ No | Requires Windows COM |
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

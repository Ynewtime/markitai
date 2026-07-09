# Getting Started

## Prerequisites

### Required

- **Python 3.11-3.14** - Required runtime
- **[uv](https://docs.astral.sh/uv/)** - Package manager (recommended)

### Optional Dependencies

These are required for specific features:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| **Playwright** | `-s playwright` (SPA rendering) | `uv pip install markitai[browser]`, browser requires `uv run playwright install chromium` |
| **FFmpeg** | Checked by `doctor`/`init` (transitive `markitdown[all]` dependency); markitai does not currently register any audio/video conversion format | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| **Jina API Key** | `-s jina` (URL conversion) | Set `JINA_API_KEY` env var |
| **LLM API Key** | `--llm` (AI enhancement) | Set `OPENAI_API_KEY` or provider-specific key. Subscription providers (`chatgpt/`, `claude-agent/`, `copilot/`) use CLI/OAuth auth instead |
| **Cloudflare** | `-s cloudflare` (cloud rendering & conversion) | Set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` env vars |
| **CairoSVG** | High-quality SVG rendering | `uv pip install markitai[svg]` |
| **pillow-heif** | HEIC/HEIF/AVIF image input | `uv pip install markitai[heif]` |
| **kreuzberg** | `.xml`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex`, `.odt`, `.ods` conversion | `uv pip install markitai[kreuzberg]` (included in `[all]`) |

::: tip Browser Automation
For SPA websites (Twitter, React apps, etc.), Playwright is used automatically. Before first use, install the browser:
```bash
uv run playwright install chromium
# Linux also requires system dependencies:
uv run playwright install-deps chromium
```
Then use `-s playwright` to force browser rendering.
:::

## Installation

### One-Click Setup (Recommended)

Run the setup script to automatically install Python, UV, and markitai:

::: code-group
```bash [Linux/macOS]
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

::: warning Security Notice
- The script checks for root/Administrator and asks before continuing
- Each component prompts before installing — uv and the Playwright browser default to Yes; LibreOffice, FFmpeg, and the Claude/Copilot CLIs default to No
:::

The script will:
- Check for / auto-install Python 3.11-3.14 (no prompt)
- Install [uv](https://docs.astral.sh/uv/) package manager (confirmation, defaults to Yes)
- Install markitai itself, plus its pip-only extras (browser automation, `extra-fetch`, `kreuzberg`, `svg`, `heif`), with no further prompts; LibreOffice, FFmpeg, and the Claude/Copilot CLIs are offered afterward with their own confirmation (defaults to No)

#### Version Pinning

Pin specific versions using environment variables:

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.14.0"
export UV_VERSION="0.9.27"
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.14.0"
$env:UV_VERSION = "0.9.27"
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

### Manual Installation

If you already have Python 3.11–3.14 and prefer a minimal install:

```bash
# Using uv (recommended, isolated environment)
uv tool install "markitai[all]"

# Or using uv pip (into a virtual environment)
uv pip install "markitai[all]"
```

Unlike the one-click setup, a manual install does **not** set up optional
components or config for you. Do the remaining steps yourself:

```bash
markitai doctor           # see what's installed / missing
markitai doctor --fix     # install the Playwright Chromium browser (for --playwright)
markitai init             # create a config and set up an LLM provider
```

::: tip Both `markitai` and `mkai` work
Every install provides the `markitai` command **and the shorter `mkai` alias** —
they are the exact same command (`mkai --help` == `markitai --help`). If a
different `mkai` already exists on your `PATH`, use the full `markitai` to avoid
ambiguity.
:::

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
├── document.pdf.md          # Basic Markdown (skipped in --llm mode unless --keep-base)
├── document.pdf.llm.md      # LLM-enhanced version (when --llm is used)
├── .markitai/                 # Metadata namespace
│   ├── assets/
│   │   ├── document.docx.0001.jpg   # Images embedded in the source document
│   │   └── images.json      # Image descriptions
│   ├── screenshots/          # Page/slide screenshots (PDF/PPTX only; full-page for URLs; --screenshot)
│   │   └── document.pdf.page0001.jpg
│   ├── reports/               # Conversion reports (JSON) — batch/URL-batch runs by default, or when output.report = true
│   └── states/                # Batch state files (for --resume)
```

::: tip
The output filename appends `.md` to the full input filename: `document.docx` → `document.docx.md` (`document.docx.llm.md` with `--llm`). The source format stays visible and distinct inputs (e.g. `report.pdf` and `report.docx`) never collide.
:::

::: tip
In `--llm` mode, only `.llm.md` is written by default. Use `--keep-base` to also write the base `.md` file.
:::

## Supported Formats

| Format | Extensions |
|--------|------------|
| Office | `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.odt`, `.ods`, `.numbers` |
| PDF | `.pdf` |
| Text / Markup / Structured Data | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xhtml`, `.xml`, `.csv`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp`, `.svg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.heic`, `.heif`, `.avif` (last three need `markitai[heif]`) |
| Other Documents | `.epub`, `.eml`, `.msg`, `.ipynb` |
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

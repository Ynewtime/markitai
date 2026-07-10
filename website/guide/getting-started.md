# Getting Started

## Your First Conversion in 60 Seconds

Install the core package. Optional browser rendering, extra format support, and LLM providers can wait until you need them:

```bash
uv tool install markitai
```

Convert a real page — this very guide:

```bash
mkai https://markitai.dev/guide/getting-started --pure
```

`mkai` is the short alias installed alongside `markitai`. With `--pure`, the Markdown body goes to stdout without frontmatter:

```text
# Getting Started
...
```

Add `-o output/` when you want a file instead. Continue below for the guided installer, documents and URLs, LLM enhancement, and optional format support.

## Prerequisites

### Required

- **Python 3.11-3.14** - Required runtime
- **[uv](https://docs.astral.sh/uv/)** - Package manager (recommended)

### Optional Dependencies

These are required for specific features:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| **Playwright** | `-s playwright` (SPA rendering) | Recommended uv-tool path: `uv tool install 'markitai[browser]' --force`, then `markitai doctor --fix`. See [Manual Installation](#manual-installation) for pipx and virtual environments. |
| **FFmpeg** | Checked by `doctor`/`init` (transitive `markitdown[all]` dependency); markitai does not currently register any audio/video conversion format | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| **Jina API Key** | `-s jina` (URL conversion) | Set `JINA_API_KEY` env var |
| **LLM authentication** | `--llm` (AI enhancement) | Use a provider API key, or sign in through a subscription provider (`chatgpt/`, `claude-agent/`, `copilot/`) with OAuth or its CLI |
| **Cloudflare** | `-s cloudflare` (cloud rendering & conversion) | Set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` env vars |
| **CairoSVG** | High-quality SVG rendering | `uv pip install markitai[svg]` |
| **pillow-heif** | HEIC/HEIF/AVIF image input | `uv pip install markitai[heif]` |
| **kreuzberg** | `.xml`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex`, `.odt`, `.ods` conversion | `uv pip install markitai[kreuzberg]` (included in `[all]`) |

::: tip Browser Automation
For SPA websites (Twitter, React apps, etc.), Playwright is used automatically. The commands below assume the recommended uv-tool installation. If you installed Markitai with pipx or into a virtual environment, use the matching browser-extra command under [Manual Installation](#manual-installation).
```bash
uv tool install 'markitai[browser]' --force
markitai doctor --fix
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
- In an interactive terminal, optional components prompt before installing. The Playwright browser defaults to Yes; LibreOffice, FFmpeg, and the Claude/Copilot CLIs default to No
- Without a usable terminal, only uv, Python, and Markitai are installed. Set `MARKITAI_INSTALL_OPTIONAL=1` to explicitly enable the optional steps in automation
:::

The script will:
- Check for / auto-install Python 3.11-3.14 (no prompt)
- Install [uv](https://docs.astral.sh/uv/) package manager (confirmation, defaults to Yes)
- Install markitai itself, plus its pip-only extras (browser automation, `extra-fetch`, `kreuzberg`, `svg`, `heif`), with no further prompts; LibreOffice, FFmpeg, and the Claude/Copilot CLIs are offered afterward with their own confirmation (defaults to No)

#### Version Pinning

Pin specific versions using environment variables:

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.19.0"
export UV_VERSION="0.9.27"
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.19.0"
$env:UV_VERSION = "0.9.27"
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

This example pins the release documented here. Omit `MARKITAI_VERSION` to install the latest stable release; if you are reading an older copy of the docs, check the current release before reusing the pin.

### Manual Installation

If you already have Python 3.11-3.14 and prefer a minimal install:

```bash
# Using uv (recommended, isolated environment)
uv tool install markitai

# Or using uv pip (into a virtual environment)
uv pip install markitai
```

Add only the extras required by your workflow later, for example `markitai[browser]` for Playwright or `markitai[heif]` for HEIC/HEIF/AVIF input. See [Optional Dependencies](#optional-dependencies) for the full list.

Unlike the one-click setup, a manual install does **not** set up optional
components or config for you. Do the remaining steps yourself:

```bash
markitai doctor           # see core and optional capabilities
markitai init             # create a config and set up an LLM provider
```

For Playwright browser rendering, choose the browser-extra command that matches how you installed Markitai:

::: code-group
```bash [uv tool]
uv tool install 'markitai[browser]' --force
```

```bash [pipx]
pipx install 'markitai[browser]' --force
```

```bash [Active virtual environment]
uv pip install 'markitai[browser]'
```
:::

Then let `doctor` install Chromium into that environment:

```bash
markitai doctor --fix
```

`doctor --fix` installs Chromium only when the Playwright package is already present. With a core-only install, it exits safely and tells you which extra to add.

::: tip Both `markitai` and `mkai` work
Every install provides the `markitai` command **and the shorter `mkai` alias**.
They are the exact same command (`mkai --help` == `markitai --help`). If a
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

For public URLs, local methods run first on standard sites, then `auto` may try Defuddle, Jina, or Cloudflare without asking. The first remote attempt in a process is disclosed on stderr. Private, local, intranet, and credential-bearing URLs remain local-only. Set `MARKITAI_NO_REMOTE_FETCH=1` if every URL must stay local.

### LLM Enhancement

Enable AI-powered format cleaning and optimization:

```bash
markitai document.docx --llm
```

Configure either a provider API key or a subscription provider. ChatGPT uses OAuth; Claude Agent and GitHub Copilot use their CLI authentication. See [Configuration](/guide/configuration#supported-providers).

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

Check the core requirement and see which optional capabilities are available. Missing optional tools do not fail the core health check:

```bash
# Check system health
markitai doctor

# If the Playwright package is present, install and re-check Chromium
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
Use the browser-extra command matching your environment under [Manual Installation](#manual-installation), then run:

```bash
markitai doctor --fix
```

### macOS

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.xls`, `.ppt`) | ✅ Full | LibreOffice preferred; falls back to installed MS Office |
| PPTX Slide Rendering | ✅ Full | LibreOffice preferred; falls back to installed MS PowerPoint |
| EMF/WMF Images | ❌ No | Windows-only format |
| Browser Automation | ✅ Full | - |

**Install LibreOffice:**
```bash
brew install --cask libreoffice
```

**MS Office fallback (no LibreOffice):** if Word/PowerPoint/Excel are
installed, markitai drives them via AppleScript instead. The first
conversion triggers a one-time macOS consent dialog ("Terminal wants to
control Microsoft Word"). Approve it once per app. This fallback opens
the app window briefly and needs a GUI session; disable it with
`"office": { "macos_fallback": false }` in config for headless use.

## Next Steps

- [Configuration](/guide/configuration) - Configure LLM providers and other settings
- [CLI Reference](/guide/cli) - Full command reference

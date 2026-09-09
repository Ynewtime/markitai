# Getting Started

## One-Click Setup (Recommended)

The setup script installs Python (if needed), uv, and markitai in one step:

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
- In an interactive terminal, optional components prompt before installing: the Playwright browser, Web UI, and OCR default to Yes; the Claude/Copilot CLIs default to No. LibreOffice is not an installer step — PPTX slide rendering warns at conversion time and points to the install command when no renderer exists
- Without a usable terminal, only uv, Python, and markitai are installed. Set `MARKITAI_INSTALL_OPTIONAL=1` to enable the optional steps in automation
- The default package index is used unless it measures as slow or unreachable from your machine. `MARKITAI_USE_MIRROR=1` always offers a mirror; `=0` never asks
:::

Pin exact versions via environment variables (omit both for the latest stable release, the recommended default):

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="X.Y.Z"   # https://pypi.org/project/markitai/#history
export UV_VERSION="X.Y.Z"         # https://github.com/astral-sh/uv/releases
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "X.Y.Z"   # https://pypi.org/project/markitai/#history
$env:UV_VERSION = "X.Y.Z"         # https://github.com/astral-sh/uv/releases
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

## Your First Conversion

Convert a real page — this very guide:

```bash
mkai https://markitai.dev/guide/getting-started --pure
```

Every install provides both the `markitai` command and the shorter `mkai` alias (identical; use the full name if another `mkai` exists on your `PATH`). With `--pure`, the Markdown body goes to stdout without frontmatter. Add `-o output/` to write a file instead:

```bash
markitai document.docx -o output/          # document
markitai https://example.com/article -o output/   # web page
markitai ./docs -o ./output                # whole directory
```

### Turning on LLM enhancement

Export a key for any supported provider and `--llm` works immediately — markitai
reads the model from your environment, so no config file is needed to start:

```bash
export GEMINI_API_KEY=...    # or OPENAI_/ANTHROPIC_/DEEPSEEK_/OPENROUTER_API_KEY
markitai report.pdf -o output/ --llm
```

To pin one model instead, set `MODEL=gemini/gemini-flash-lite-latest`. For a
config file, a subscription provider, or several models with fallback:

```bash
markitai init                # guided setup (or: markitai -I interactive mode)
markitai doctor              # check core and optional capabilities
```

## Optional Capabilities

Add extras only when you need them (`uv tool install 'markitai[<extra>]' --force`):

| Extra / Dependency | Enables |
|--------------------|---------|
| `markitai[browser]` (Playwright) | `-s playwright` browser rendering for SPA/JS-heavy pages |
| `markitai[ocr]` (RapidOCR) | `--ocr` local OCR for scanned PDFs/images |
| `markitai[legacy]` (anydoc) | Legacy Office `.doc`/`.ppt` conversion |
| `markitai[heif]` | HEIC/HEIF/AVIF image input |
| `markitai[svg]` | High-quality SVG rendering |
| `markitai[kreuzberg]` | `.rtf` conversion (`.xml`, `.tsv`, `.rst`, `.org`, `.tex`, `.odt`, `.ods` convert natively since 1.0.0) |
| `markitai[serve]` | Local web workspace and REST API |
| Jina API key | `-s jina` remote reader (`JINA_API_KEY` env var) |
| Cloudflare | `-s cloudflare` cloud rendering (`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`) |

After adding the browser extra, install Chromium once:

```bash
markitai doctor --fix
```

(`doctor --fix` installs Chromium only when the Playwright package is present; with a core-only install it exits safely and names the extra to add.)

## Manual Installation

If you already have Python 3.11–3.13 and prefer a minimal install:

```bash
uv tool install markitai        # recommended: isolated tool environment
uv pip install markitai         # or into the active virtual environment
pipx install markitai           # or pipx
```

A manual install sets up nothing optional: run `markitai doctor` to see what's available and `markitai init` for config and LLM provider setup. For browser rendering, add the browser extra matching your install method (`uv tool install 'markitai[browser]' --force`, `pipx install 'markitai[browser]' --force`, or `uv pip install 'markitai[browser]'`), then `markitai doctor --fix`.

## Feature Notes

**URLs**: for public URLs, local methods run first, then `auto` may try Defuddle, Jina, or Cloudflare without asking (the first remote attempt in a process is disclosed on stderr). Private, local, intranet, and credential-bearing URLs stay local-only. `MARKITAI_NO_REMOTE_FETCH=1` forces everything local.

**LLM enhancement** (`--llm`): clean formatting and generate frontmatter. Configure a provider API key or a subscription provider (`chatgpt/` OAuth; `claude-agent/`, `copilot/` CLI sign-in) — see [Configuration](/guide/configuration#supported-providers).

**Presets** bundle common flags: `rich` (LLM + alt + desc + screenshot), `standard` (LLM + alt + desc), `minimal` (plain conversion). Any preset flag can be overridden with `--no-*`, e.g. `--preset rich --no-desc`.

**Batch runs** write a JSON report and support `--resume` after interruption. See [CLI Reference](/guide/cli) for `--llm-batch` (Batch API, half price) and more.

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

The output filename appends `.md` to the full input filename: `document.docx` → `document.docx.md` (`document.docx.llm.md` with `--llm`), so distinct inputs (`report.pdf`, `report.docx`) never collide.

## Supported Formats

| Format | Extensions |
|--------|------------|
| Office | `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.odt`, `.ods`, `.numbers` |
| PDF | `.pdf` |
| Text / Markup / Structured Data | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xhtml`, `.xml`, `.csv`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex` (`.rtf` needs `markitai[kreuzberg]`) |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp`, `.svg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.heic`, `.heif`, `.avif` (last three need `markitai[heif]`) |
| Other Documents | `.epub`, `.eml`, `.msg`, `.ipynb` |
| URLs | `http://`, `https://` |

## Platform-Specific Features

### Windows

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.ppt`) | ✅ Full | Needs `markitai[legacy]` (anydoc Rust backend, no Office install; PPT tables flatten to text) |
| Legacy Excel (`.xls`) | ✅ Full | Built-in (pure Python) |
| PPTX Slide Rendering | ✅ Full | MS Office preferred, LibreOffice fallback |
| EMF/WMF Images | ✅ Full | Native support |
| Browser Automation | ✅ Full | Hidden window mode |

### Linux

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.ppt`) | ✅ Full | Needs `markitai[legacy]` (no LibreOffice needed) |
| Legacy Excel (`.xls`) | ✅ Full | Built-in (pure Python) |
| PPTX Slide Rendering | ✅ Full | Requires LibreOffice (`apt-get install libreoffice` / `dnf install libreoffice`) |
| EMF/WMF Images | ❌ No | Windows-only format |
| Browser Automation | ✅ Full | Requires system dependencies |

### macOS

| Feature | Support | Notes |
|---------|---------|-------|
| Legacy Office (`.doc`, `.ppt`) | ✅ Full | Needs `markitai[legacy]` (no Office install) |
| Legacy Excel (`.xls`) | ✅ Full | Built-in (pure Python) |
| PPTX Slide Rendering | ✅ Full | LibreOffice preferred (`brew install --cask libreoffice`); falls back to installed MS PowerPoint |
| EMF/WMF Images | ❌ No | Windows-only format |
| Browser Automation | ✅ Full | - |

The macOS PowerPoint fallback drives PowerPoint via AppleScript: the first render triggers a one-time consent dialog ("Terminal wants to control Microsoft PowerPoint"), opens the app briefly, and needs a GUI session. Disable it with `"office": { "macos_fallback": false }` in config for headless use.

## Next Steps

- [Configuration](/guide/configuration) - LLM providers and all settings
- [CLI Reference](/guide/cli) - Full command reference

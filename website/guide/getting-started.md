# Getting Started

## One-Click Setup (Recommended)

One command installs Python (if missing), uv and markitai:

::: code-group
```bash [Linux/macOS]
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

The script asks before installing optional pieces. The Playwright browser, the web workspace and OCR default to yes; the Claude and Copilot CLIs default to no.

Without an interactive terminal it installs only the core. Set `MARKITAI_INSTALL_OPTIONAL=1` to add the optional pieces in automation, and `MARKITAI_VERSION=X.Y.Z` to pin a release.

## Your First Conversion

Convert a real page, this very guide:

```bash
mkai https://markitai.dev/guide/getting-started --pure
```

`mkai` is a short alias for `markitai`. `--pure` prints plain Markdown to the terminal. To save files instead, give an output directory:

```bash
markitai document.docx -o output/
markitai https://example.com/article -o output/
markitai ./docs -o output/
```

### Turning on LLM enhancement

Export an API key for any supported provider and add `--llm`:

```bash
export GEMINI_API_KEY=...    # or OPENAI_ / ANTHROPIC_ / DEEPSEEK_ / OPENROUTER_API_KEY
markitai report.pdf -o output/ --llm
```

markitai picks a model from the key it finds. To choose one yourself, set `MODEL=provider/model`.

For a config file, a subscription provider (ChatGPT, Claude Code, Copilot) or several models with fallback, run the guided setup:

```bash
markitai init
markitai doctor     # shows what is installed and configured
```

## Optional Capabilities

The core install converts documents on its own. Add an extra only when you need it:

```bash
uv tool install 'markitai[browser]' --force
```

| Extra | Enables |
|-------|---------|
| `markitai[browser]` | Browser rendering (`-s playwright`) for JavaScript-heavy pages and URL screenshots |
| `markitai[ocr]` | Local OCR (`--ocr`) for scanned PDFs and images |
| `markitai[serve]` | The [web workspace](/guide/serve) and its REST API |
| `markitai[mcp]` | The [MCP server](/guide/mcp) for AI agents |
| `markitai[claude-agent]` | Claude Code subscription as an LLM provider |
| `markitai[copilot]` | GitHub Copilot subscription as an LLM provider |
| `markitai[legacy]` | Legacy Office `.doc` and `.ppt` files |
| `markitai[heif]` | HEIC, HEIF and AVIF images |
| `markitai[svg]` | High-quality SVG rendering |
| `markitai[extra-fetch]` | curl-cffi client for sites with TLS fingerprint checks |
| `markitai[all]` | Everything above |

Two remote fetch strategies need credentials instead of an extra: `-s jina` reads `JINA_API_KEY`, and `-s cloudflare` reads `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.

After adding the browser extra, install Chromium once:

```bash
markitai doctor --fix
```

## Manual Installation

If Python 3.11 to 3.13 is already installed:

```bash
uv tool install markitai     # recommended
pipx install markitai
uv pip install markitai      # into the active virtual environment
```

A manual install has no optional pieces. Run `markitai doctor` to see what is available, and `markitai init` to set up an LLM provider.

## Feature Notes

**Presets** bundle the common flags. `minimal` is plain conversion, `standard` adds LLM cleanup and image analysis, `rich` adds page screenshots. Turn any part off with `--no-*`, for example `--preset rich --no-desc`.

**URLs** are fetched locally first. When that fails on a public page, markitai may fall back to a remote reader (Defuddle, Jina or Cloudflare) and says so once on stderr. Private, intranet and credential-bearing URLs never leave your machine. `MARKITAI_NO_REMOTE_FETCH=1` keeps everything local.

**Directories** convert as a batch with a progress display and a JSON report. If a run is interrupted, add `--resume` to pick it up.

## Output Structure

```text
output/
├── document.pdf.md          # Markdown (with --llm, only .llm.md is written unless --keep-base)
├── document.pdf.llm.md      # LLM-enhanced version
└── .markitai/
    ├── assets/              # images from the source, plus images.json descriptions
    ├── screenshots/         # page, slide or full-page screenshots (--screenshot)
    ├── reports/             # JSON reports for batch runs
    └── states/              # batch state for --resume
```

The output name is the full input name plus `.md`, so `report.pdf` and `report.docx` never collide.

## Supported Formats

| Format | Extensions |
|--------|------------|
| Office | `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.odt`, `.ods`, `.numbers` |
| PDF | `.pdf` |
| Text and markup | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xhtml`, `.xml`, `.csv`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp`, `.svg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.heic`, `.heif`, `.avif` (last three need `markitai[heif]`) |
| Other documents | `.epub`, `.eml`, `.msg`, `.ipynb` |
| URLs | `http://`, `https://` |

## Platform-Specific Features

Everything works on Windows, Linux and macOS, with two things to know:

- **EMF/WMF images** are a Windows-only format and only convert on Windows.
- **PPTX slide screenshots** need a renderer. Windows uses Microsoft Office or LibreOffice. Linux needs LibreOffice (`apt-get install libreoffice`). macOS prefers LibreOffice (`brew install --cask libreoffice`) and otherwise drives an installed PowerPoint, which pops a one-time permission dialog and needs a desktop session. Set `"office": { "macos_fallback": false }` in the config to disable that on headless Macs.

Legacy `.doc` and `.ppt` files need `markitai[legacy]` and no Office install on any platform.

## Next Steps

- [Configuration](/guide/configuration) for LLM providers and every setting
- [CLI Reference](/guide/cli) for every command and flag

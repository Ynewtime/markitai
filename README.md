# Markitai

[![PyPI](https://img.shields.io/pypi/v/markitai)](https://pypi.org/project/markitai/)
[![Python](https://img.shields.io/pypi/pyversions/markitai)](https://pypi.org/project/markitai/)
[![CI](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml/badge.svg)](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Ynewtime/markitai/blob/main/LICENSE)

Opinionated Markdown converter with native LLM enhancement support.

- **Multi-format**: DOCX, PPTX, XLSX, PDF, TXT, MD, images (JPG/PNG/WebP), and URLs → clean Markdown
- **LLM enhancement**: AI-powered format cleaning, frontmatter metadata, and vision analysis of embedded images via [litellm](https://github.com/BerriAI/litellm), so any provider works (OpenAI, Anthropic, Gemini, local CLIs, and more)
- **Batch processing**: concurrent conversion with progress display and `--resume` for interrupted jobs
- **OCR**: scanned PDFs and images via RapidOCR
- **Web fetching**: static HTTP with cache revalidation, or Playwright rendering for JS-heavy pages

Docs: <https://markitai.dev>

## Install

**Recommended: guided installer.** Checks/installs Python and uv, lets you
pick extras, installs optional components (Playwright browser, LibreOffice,
FFmpeg), offers China-mainland mirror acceleration, and is bilingual (EN/中文):

```bash
# Linux/macOS
curl -fsSL https://markitai.dev/setup.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

**Minimal (uv / pip)**, if you already have Python 3.11-3.14 and just want the package:

```bash
uv tool install markitai     # isolated environment (recommended)
pipx install markitai        # or pipx
```

After a uv/pip install, do the setup steps the guided installer would have done for you:

```bash
markitai doctor           # check core and optional capabilities
markitai init             # create a config and set up an LLM provider
```

If you need Playwright browser rendering, add its package before asking `doctor` to install Chromium:

```bash
uv tool install "markitai[browser]" --force     # uv tool install
# pipx install "markitai[browser]" --force      # pipx alternative
markitai doctor --fix                           # install Chromium
```

Both installs provide the `markitai` command **and the shorter `mkai` alias**.
They are the same command (`mkai --help` == `markitai --help`). If you already
have a different `mkai` on your PATH, use the full `markitai` to avoid ambiguity.

### Extras

| Extra | Enables |
| --- | --- |
| `browser` | Playwright rendering for JS-heavy pages |
| `claude-agent` | Claude Agent SDK as an LLM provider |
| `copilot` | GitHub Copilot SDK as an LLM provider |
| `extra-fetch` | curl-cffi HTTP client (better anti-bot compatibility) |
| `kreuzberg` | Kreuzberg extraction backend |
| `svg` | SVG rasterization via cairosvg |
| `all` | Everything above |

## Quick start

```bash
markitai document.pdf -o out/            # convert a file
markitai https://example.com -o out/     # convert a URL
markitai ./docs -o out/ --llm            # batch convert with LLM enhancement
markitai document.pdf --preset rich      # LLM + alt text + descriptions + screenshots
markitai doctor                          # check dependencies and configuration
markitai init                            # interactive configuration setup
```

See the [Getting Started guide](https://markitai.dev/guide/getting-started) for LLM configuration, presets, caching, and batch options.

## License

[MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE)

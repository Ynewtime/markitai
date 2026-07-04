# Markitai

[![PyPI](https://img.shields.io/pypi/v/markitai)](https://pypi.org/project/markitai/)
[![Python](https://img.shields.io/pypi/pyversions/markitai)](https://pypi.org/project/markitai/)
[![CI](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml/badge.svg)](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Ynewtime/markitai/blob/main/LICENSE)

Opinionated Markdown converter with native LLM enhancement support.

- **Multi-format**: DOCX, PPTX, XLSX, PDF, TXT, MD, images (JPG/PNG/WebP), and URLs → clean Markdown
- **LLM enhancement**: AI-powered format cleaning, frontmatter metadata, and vision analysis of embedded images — via [litellm](https://github.com/BerriAI/litellm), so any provider works (OpenAI, Anthropic, Gemini, local CLIs, ...)
- **Batch processing**: concurrent conversion with progress display and `--resume` for interrupted jobs
- **OCR**: scanned PDFs and images via RapidOCR
- **Web fetching**: static HTTP with cache revalidation, or Playwright rendering for JS-heavy pages

Docs: <https://markitai.ynewtime.com>

## Install

```bash
# Recommended: uv tool (isolated environment)
uv tool install "markitai[all]"

# Or with pipx
pipx install "markitai[all]"

# Or the guided installer (checks Python, installs uv, picks extras)
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh          # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex"  # Windows
```

Requires Python 3.11–3.14.

### Extras

| Extra | Enables |
| --- | --- |
| `browser` | Playwright rendering for JS-heavy pages |
| `claude-agent` | Claude Agent SDK as an LLM provider |
| `copilot` | GitHub Copilot SDK as an LLM provider |
| `gemini-cli` | Gemini CLI OAuth as an LLM provider |
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

See the [Getting Started guide](https://markitai.ynewtime.com/guide/getting-started) for LLM configuration, presets, caching, and batch options.

## License

[MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE)

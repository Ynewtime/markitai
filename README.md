# Markitai

[![PyPI](https://img.shields.io/pypi/v/markitai)](https://pypi.org/project/markitai/)
[![Python](https://img.shields.io/pypi/pyversions/markitai)](https://pypi.org/project/markitai/)
[![CI](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml/badge.svg)](https://github.com/Ynewtime/markitai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Ynewtime/markitai/blob/main/LICENSE)

Opinionated Markdown converter with native LLM enhancement support.

- **Multi-format**: DOCX, PPTX, XLSX, PDF, EPUB, EML, TXT, MD, images (JPG/PNG/WebP), and URLs → clean Markdown; legacy `.doc`/`.ppt` via the `legacy` extra
- **LLM enhancement**: AI-powered format cleaning, frontmatter metadata, and vision analysis of embedded images via [litellm](https://github.com/BerriAI/litellm), so any provider works (OpenAI, Anthropic, Gemini, local CLIs, and more)
- **Batch processing**: concurrent conversion with progress display and `--resume` for interrupted jobs
- **OCR**: scanned PDFs and images via local RapidOCR (optional extra, see below), or `--ocr --llm` to have the vision model read the page images directly (VLM-OCR)
- **Web fetching**: static HTTP with cache revalidation, or Playwright rendering for JS-heavy pages
- **Local web workspace**: upload files or folders, submit URLs, configure LLM providers, compare results, retry failures, and revisit conversion history — CLI runs can opt in too, via `--record-history`

Docs: <https://markitai.dev>

## Install

**Guided installer** (recommended). Installs Python and uv if needed, lets you
pick extras and the Playwright browser, and offers a mirror when the default
index is unreachable. Bilingual (EN/中文).

```bash
# Linux/macOS
curl -fsSL https://markitai.dev/setup.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

**Already have Python 3.11–3.13?** Install the package alone, then run the two
setup steps yourself:

```bash
uv tool install markitai     # or: pipx install markitai
markitai doctor              # check core and optional capabilities
markitai init                # config and LLM provider
```

Browser rendering needs the `browser` extra, then Chromium:

```bash
uv tool install "markitai[browser]" --force
markitai doctor --fix
```

Both routes install `markitai` and the shorter `mkai` alias.

### Extras

| Extra | Enables |
| --- | --- |
| `browser` | Playwright rendering for JS-heavy pages |
| `claude-agent` | Claude Agent SDK as an LLM provider |
| `copilot` | GitHub Copilot SDK as an LLM provider |
| `extra-fetch` | curl-cffi HTTP client (better anti-bot compatibility) |
| `heif` | HEIC/HEIF/AVIF image input |
| `legacy` | Legacy Office conversion (`.doc`/`.ppt`) via the anydoc Rust backend |
| `mcp` | Bundled `markitai-mcp` server for AI agents (Model Context Protocol) |
| `ocr` | Local OCR for scanned PDFs and images (`--ocr`) |
| `serve` | Local web workspace and REST API |
| `svg` | SVG rasterization via cairosvg |
| `all` | Everything above |

`ocr` is opt-in because it adds ~160MB of models. The guided installer asks
about it, and `markitai doctor` prints the command when it is missing:

```bash
uv tool install "markitai[ocr]" --force
```

Launch the local web workspace with:

```bash
uv tool install "markitai[serve]" --force
markitai serve
```

## Quick start

```bash
markitai document.pdf -o out/            # convert a file
markitai https://example.com -o out/     # convert a URL
markitai ./docs -o out/                  # batch convert a directory
markitai ./docs -o out/ --json           # machine-readable results for automation
markitai https://example.com --no-remote-fetch -o out/  # local URL extraction only
markitai doctor                          # check dependencies and configuration
```

For LLM enhancement, export any supported provider key — markitai picks the
model up from the environment, no config file needed:

```bash
export GEMINI_API_KEY=...                # or OPENAI_/ANTHROPIC_/DEEPSEEK_/OPENROUTER_API_KEY
markitai document.pdf -o out/ --llm      # clean formatting + generated frontmatter
markitai document.pdf --preset rich      # LLM + alt text + descriptions + screenshots
markitai init                            # or configure it interactively, once
```

## MCP server

`markitai-mcp` exposes conversion to AI agents over the Model Context Protocol
with four tools: `convert_document`, `convert_url`, `batch_convert`,
`job_status`. Nothing to install — `uvx` runs it on demand, and large outputs
land on disk instead of in the model context. For Claude Code:

```bash
claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp
```

Other clients, LLM enhancement and batch jobs are covered in the
[MCP guide](https://markitai.dev/guide/mcp). `markitai mcp` starts the same
server through the CLI itself, which is how the
[MCP Registry](https://registry.modelcontextprotocol.io) lists it.

<!-- mcp-name: io.github.Ynewtime/markitai -->

## Documentation

- [Getting started](https://markitai.dev/guide/getting-started) — install, first conversion, output layout, supported formats
- [CLI reference](https://markitai.dev/guide/cli) — every command and flag
- [Configuration](https://markitai.dev/guide/configuration) — config file, environment variables, LLM providers, every setting
- [Fetch policy](https://markitai.dev/guide/fetch-policy) — the URL strategy cascade, domain profiles, what stays local
- [Output profiles](https://markitai.dev/guide/output-profiles) — `rag`, `obsidian` and `okf` output shaping
- [Web workspace](https://markitai.dev/guide/serve) — `markitai serve`, its history and REST API
- [Python API](https://markitai.dev/guide/python-api) — `convert()` and `aconvert()` as a library
- [Conversion performance](https://markitai.dev/guide/performance) — measured local-conversion results and how to reproduce them
- [Why Markitai](https://markitai.dev/guide/comparison) — how it compares with markitdown, docling and anydoc

Contributors start at [CONTRIBUTING.md](https://github.com/Ynewtime/markitai/blob/main/CONTRIBUTING.md).

## How markitai compares

Two of the tools markitai is usually compared with are also its dependencies:
markitdown converts the Office formats, and anydoc handles legacy `.doc`/`.ppt`
behind `markitai[legacy]`. Against them and docling, markitai trades ecosystem
reach, ML document-structure models and dependency-free speed for a built-in
LLM pipeline, live web fetching and a local workspace. The feature-by-feature
table is in [Why Markitai](https://markitai.dev/guide/comparison).

## License

markitai's own source code is [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE).

The default installation is not uniformly MIT, because the PDF engine is not.
The PyMuPDF packages `pymupdf`, `pymupdf-layout` and `pymupdf4llm` come from
Artifex Software and are dual-licensed under **AGPL-3.0 or a commercial licence
from Artifex**. They are core dependencies — PDF conversion does not work
without them. Running the CLI on your own machine, or a `markitai serve`
instance only you talk to, carries no AGPL obligation; redistributing the
combined work or offering it to other people over a network does.

[NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE) carries the
full terms, the rest of the dependency licensing, and the attribution for the
code markitai ports from [defuddle](https://github.com/kepano/defuddle) (MIT)
and [marker](https://github.com/VikParuchuri/marker) (Apache-2.0).

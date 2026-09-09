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

**Recommended: guided installer.** Checks/installs Python and uv, lets you
pick extras, installs the optional Playwright browser,
falls back to a mirror when it measures the default index as unreachable, and
is bilingual (EN/中文):

```bash
# Linux/macOS
curl -fsSL https://markitai.dev/setup.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

**Minimal (uv / pip)**, if you already have Python 3.11-3.13 and just want the package:

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
| `heif` | HEIC/HEIF/AVIF image input |
| `legacy` | Legacy Office conversion (`.doc`/`.ppt`) via the anydoc Rust backend |
| `mcp` | Bundled `markitai-mcp` server for AI agents (Model Context Protocol) |
| `ocr` | Local OCR for scanned PDFs and images (`--ocr`) |
| `serve` | Local web workspace and REST API |
| `svg` | SVG rasterization via cairosvg |
| `all` | Everything above |

The base install is deliberately lean (~475MB). `ocr` is the one extra that
adds real weight (~160MB of models and OpenCV), so it is opt-in:

```bash
uv tool install "markitai[ocr]" --force
```

The guided installer offers it as a yes/no question (defaulting to yes, and
remembering a "no" for the rest of the run); `markitai doctor` reports OCR as
an optional capability and prints this command when it is not installed.

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

See the [Getting Started guide](https://markitai.dev/guide/getting-started) for LLM configuration, presets, caching, and batch options.

## MCP server

The MCP server `markitai-mcp` (bundled with markitai, enabled by the `mcp` extra) exposes conversion to AI agents over the Model Context Protocol: `convert_document`, `convert_url`, `batch_convert`, `job_status`. Zero install via `uvx`; large outputs land on disk instead of in the model context. For Claude Code, `claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp`; for other clients:

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["--from", "markitai[mcp]", "markitai-mcp"] }
  }
}
```

`markitai mcp` starts the same server through the CLI itself (`uvx --from "markitai[mcp]" markitai mcp`), which is how the [MCP Registry](https://registry.modelcontextprotocol.io) lists it. See the [MCP guide](https://markitai.dev/guide/mcp) for LLM enhancement and batch jobs.

<!-- mcp-name: io.github.Ynewtime/markitai -->

## Comparison

How markitai compares to three tools people mention in the same breath. No star or download counts — those go stale immediately.

| | **markitai** | markitdown | docling | anydoc |
| --- | --- | --- | --- | --- |
| Engine | Python; rule-based conversion + optional LLM pipeline | Python; lightweight rule-based converters + plugins | Python; ML layout/table/VLM document-structure models | Rust; zero-ML parsers |
| LLM enhancement | Built-in: format cleaning, frontmatter, vision analysis, per-run JSON cost/usage reports | Optional: image captions, transcription, an OCR plugin | VLM for structure (DocTags), not prose cleanup | None |
| Web pages | 5-strategy fetch cascade, local-first; static runs a from-scratch port of [defuddle](https://github.com/kepano/defuddle)'s readability algorithm before falling back to a browser or 3 remote APIs | Whole-DOM HTML→Markdown, no main-content pass | Downloads a document URL into the same file pipeline | No URL input — local files/bytes only |
| Scanned docs | Optional local OCR (`markitai[ocr]`, RapidOCR), or `--ocr --llm` to have the vision model read the pages | Optional plugin (LLM-vision or Azure OCR) | Built-in OCR for scanned PDFs/images | None in the OSS library |
| Positioning | Independent project; CLI + local bilingual (EN/中文) web workspace | Microsoft (AutoGen team); widest ecosystem/plugin adoption | IBM Research origin, now governed by the LF AI & Data Foundation; enterprise RAG building block | Firecrawl open-source; dependency-free, millisecond-scale, 14 formats, Node/Python/WASM bindings |

Each optimizes for a different job: anydoc for dependency-free speed, docling for ML-driven document structure in RAG pipelines, markitdown for ecosystem reach — markitai trades those for a built-in LLM pipeline, live web fetching, and a local UI. Two of them are also dependencies rather than only alternatives: markitdown converts the Office formats, and anydoc handles legacy `.doc`/`.ppt` behind `markitai[legacy]`.

## License

markitai's own source code is [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE).

The default installation is not uniformly MIT, because the PDF engine is not.
The PyMuPDF packages `pymupdf`, `pymupdf-layout`, and `pymupdf4llm` come from
Artifex Software and are
dual-licensed under **AGPL-3.0 or a commercial licence from Artifex**. They are
core dependencies — PDF conversion does not work without them.

For local use — running the CLI on your own machine, or a `markitai serve`
instance only you talk to — this changes nothing. AGPL obligations attach when
you *redistribute* the combined work or offer it to other people over a network:
in that case AGPL-3.0 asks you to make the corresponding source available on the
same terms, or to buy a [commercial licence from Artifex](https://artifex.com/licensing/)
instead.

Everything else in the default install is MIT, Apache-2.0, BSD, or MIT-CMU. CI
enforces this: `scripts/check_licenses.py` fails the build on any
non-commercial or proprietary dependency, and on any AGPL/GPL package outside an
explicit allowlist.

Full details, plus attribution for the code markitai ports from
[defuddle](https://github.com/kepano/defuddle) (MIT) and
[marker](https://github.com/VikParuchuri/marker) (Apache-2.0), are in
[NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE).

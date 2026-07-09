# CLI Reference

## Basic Usage

```bash
markitai <input> [options]
```

The `<input>` can be:
- A file path (`document.docx`)
- A directory path (`./docs`)
- A URL (`https://example.com`)

## Conversion Options

### `--llm`

Enable LLM-powered format cleaning and optimization. By default, only `.llm.md` is written (base `.md` is skipped). Use `--keep-base` to write both.

```bash
markitai document.docx --llm
```

Social posts (extractor-curated content marked `content_profile: social_post`, e.g. X/Twitter posts) keep their body verbatim — the LLM only generates frontmatter metadata, so post structure and wording are never altered.

::: tip
`--llm`, `--alt`, `--desc`, `--ocr`, and `--screenshot` all have `--no-*` counterparts (`--no-llm`, `--no-alt`, `--no-desc`, `--no-ocr`, `--no-screenshot`) to explicitly disable a feature a preset would otherwise enable — e.g. `--preset rich --no-desc`.
:::

### `-p, --preset <name>`

Use a predefined configuration preset.

| Preset | Description |
|--------|-------------|
| `rich` | LLM + alt + desc + screenshot |
| `standard` | LLM + alt + desc |
| `minimal` | Basic conversion only |

```bash
markitai document.pdf --preset rich
markitai document.pdf --preset rich --no-desc   # Rich without desc — any preset feature can be toggled off with --no-*
```

### `--alt`

Generate alt text for images using AI. Requires `--llm` — without it, image analysis is skipped with a warning.

```bash
markitai document.pdf --llm --alt
```

### `--desc`

Generate detailed descriptions for images. Requires `--llm` — without it, image analysis is skipped with a warning.

```bash
markitai document.pdf --llm --desc
```

### `--screenshot`

Enable screenshot capture:
- **PDF/PPTX**: Renders pages/slides as JPEG images
- **URLs**: Captures full-page screenshots using Playwright

```bash
# Document screenshots
markitai document.pdf --screenshot
markitai presentation.pptx --screenshot

# URL screenshots
markitai https://example.com --screenshot
```

::: tip
For URLs, `--screenshot` automatically upgrades the fetch strategy to `playwright` if needed. The screenshot is saved as `{domain}_path.full.jpg` in the `.markitai/screenshots/` subdirectory.
:::

### `--screenshot-only`

Capture screenshots only without extracting content. Behavior depends on `--llm`, **for URL input**:

| Command | Output |
|---------|--------|
| `--screenshot-only` | Screenshots only (no .md files) |
| `--llm --screenshot-only` | `.llm.md` + screenshots (LLM extracts from screenshots); add `--keep-base` to also get `.md` |

```bash
# Just capture screenshots
markitai https://example.com --screenshot-only

# LLM extracts content purely from screenshots
markitai https://example.com --llm --screenshot-only
```

::: tip
Use `--llm --screenshot-only` for pages where traditional content extraction fails (e.g., heavy JavaScript sites, social media).
:::

::: warning
For **file input** (PDF/PPTX), `--screenshot-only` without `--llm` does **not** skip `.md` — it still writes the normal extracted-text markdown alongside the screenshots. The "no .md files" guarantee above only applies to URL input.
:::

### `--ocr`

Enable OCR for scanned documents.

```bash
markitai scanned.pdf --ocr
```

### `--pure`

Transparent pass-through mode: LLM only does text cleaning, no frontmatter generation or post-processing.

```bash
# Without --llm: writes raw markdown without frontmatter
markitai document.docx --pure

# With --llm: sends content through LLM for text cleaning only
markitai document.docx --llm --pure

# With --preset: preset controls features, --pure controls output format
markitai document.pdf --preset rich --pure
```

::: tip
`--pure` and `--llm` are independent flags. `--pure` alone skips frontmatter generation; `--pure --llm` sends content to LLM for cleaning but returns raw output without generated metadata (description, tags, etc.).
:::

::: warning
`--pure` silently overrides `--alt`, `--desc`, and `--screenshot`. A warning is displayed when these flags are used together.
:::

### `--keep-base`

Write base `.md` file even in LLM mode. By default, `--llm` only outputs `.llm.md` to avoid redundant files.

```bash
# Default: only .llm.md is written
markitai document.docx --llm

# Keep both .md and .llm.md
markitai document.docx --llm --keep-base
```

### `--no-compress`

Disable image compression.

```bash
markitai document.pdf --no-compress
```

## Output Options

### `-o, --output <path>`

Specify the output location. For a single file/URL input, `-o` can also be an exact file target (e.g. `-o result.md`) rather than a directory. If omitted, single file/URL conversions print to stdout instead; directory-batch and `.urls`-list input require `-o`.

```bash
markitai document.docx -o ./output
markitai document.docx -o ./result.md
```

### `--resume`

Resume interrupted batch processing. Completed files are skipped, `FAILED`/interrupted (`IN_PROGRESS` at crash time) files are retried, and newly-added files are picked up — reporting `Resuming batch: N completed, M remaining`. Only applies to batch (directory/`.urls`) input; ignored for a single file/URL.

```bash
markitai ./docs -o ./output --resume
```

## Concurrency Options

### `--llm-concurrency <n>`

Number of concurrent LLM requests (default: 10).

```bash
markitai ./docs --llm --llm-concurrency 10
```

### `-j, --batch-concurrency <n>`

Number of concurrent file processing tasks (default: 10).

```bash
markitai ./docs -o ./output -j 4
```

::: tip
For mixed file and URL batches, use `--url-concurrency` to control URL fetching separately. This prevents slow URLs from blocking file processing.
:::

## Cache Options

### `--no-cache`

Disable LLM result caching (force fresh API calls).

```bash
markitai document.docx --llm --no-cache
```

### `--no-cache-for <patterns>`

Disable cache for specific files or patterns (comma-separated).

```bash
# Single file
markitai ./docs --no-cache-for file1.pdf

# Glob pattern
markitai ./docs --no-cache-for "*.pdf"

# Multiple patterns
markitai ./docs --no-cache-for "*.pdf,reports/**"
```

## URL Options

### `.urls` File Support

When the input is a `.urls` file, Markitai automatically processes it as a URL batch. Directory batch input also auto-discovers and processes any `.urls` files found within the scan tree (subject to the same `--glob`/`--max-depth` rules), merging their URLs into the same batch run alongside regular files.

```bash
markitai urls.urls -o ./output
```

The `.urls` file supports three formats:

Plain text — one URL per line, with an optional custom output name after whitespace:
```
# Comments start with #
https://example.com/page1
https://example.com/page2 custom_name
```

JSON array of URL strings:
```json
["https://example1.com", "https://example2.com"]
```

JSON array of objects, with an optional `output_name`:
```json
[
  {"url": "https://example1.com"},
  {"url": "https://example2.com", "output_name": "custom"}
]
```

### `--glob, -g <pattern>`

Restrict directory batch discovery to matching relative paths. Can be repeated to add multiple patterns. Prefix with `!` to exclude.

```bash
# Only process PDF files
markitai ./docs -o ./output -g "*.pdf"

# Process PDFs and DOCX files
markitai ./docs -o ./output -g "*.pdf" -g "*.docx"

# Exclude a subdirectory
markitai ./docs -o ./output -g '!drafts/**'
```

::: tip
Only applies to directory input. Use single quotes in shells with history expansion (e.g., zsh) when using `!` prefix.
:::

### `--max-depth <n>`

Override recursive directory scan depth for batch discovery (default: 5). `0` means only scan the input directory itself (no recursion).

```bash
markitai ./docs -o ./output --max-depth 2
```

### `--url-concurrency <n>`

Number of concurrent URL fetch operations (default: 5). This is separate from `--batch-concurrency` to prevent slow URLs from blocking file processing.

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `-s, --strategy <name>`

Select the URL fetch strategy. This is the primary flag for URL fetching, orthogonal to `-b/--backend` below.

| Value | Description |
|-------|-------------|
| `auto` (default) | Tries strategies in policy order, falling back on failure |
| `static` | Static HTTP fetch with native webextract — fast, no JS, no external API |
| `playwright` | Browser rendering via Playwright — for JS-heavy SPA sites (e.g. x.com) |
| `defuddle` | Defuddle API — free, no authentication, excellent content cleaning |
| `jina` | Jina Reader API — cloud-based alternative when browser rendering is unavailable |
| `cloudflare` | Cloudflare Browser Rendering `/content` API. Also enables Workers AI `toMarkdown` for file conversion (see `-b/--backend`) |

```bash
markitai https://example.com -s defuddle
markitai https://x.com/user/status/123 -s playwright
```

::: tip
`-s cloudflare` requires `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (environment variables or `markitai.json`). Create a token at [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens) with *Browser Rendering: Edit* and *Workers AI: Read* permissions. See [Configuration → Cloudflare Settings](/guide/configuration#cloudflare-settings).
:::

::: tip
To pre-install Playwright browsers:
```bash
uv run playwright install chromium
# Linux also requires system dependencies:
uv run playwright install-deps chromium
```
:::

### `-b, --backend <name>`

Select the file conversion backend, orthogonal to `-s/--strategy` (which only affects URL fetching).

| Value | Description |
|-------|-------------|
| `native` (default) | Built-in converters (DOCX, PDF, images, etc.) |
| `kreuzberg` | Force the kreuzberg converter for all file formats — requires `uv pip install markitai[kreuzberg]` |
| `cloudflare` | Cloudflare Workers AI `toMarkdown` — requires CF credentials |

```bash
markitai document.pdf -b kreuzberg
markitai document.pdf -b cloudflare
markitai https://example.com -s playwright -b kreuzberg   # -s and -b combine freely
```

`-b kreuzberg` and `-s cloudflare` are mutually exclusive — both override file conversion.

::: tip
Cloudflare Browser Rendering is available on the Free plan. Workers AI `toMarkdown` is free for PDF/Office/CSV/XML; image conversion uses Neurons quota. For formats with a local converter, native/kreuzberg generally produce higher-quality output — `-b cloudflare` warns when a better local converter is available.
:::

### Deprecated per-backend flags

`--playwright`, `--defuddle`, `--static`, `--jina`, `--cloudflare`, and `--kreuzberg` still work as deprecated aliases. Each prints a one-line deprecation notice to stderr and resolves to the flag above:

| Deprecated flag | Equivalent |
|------------------|------------|
| `--playwright` | `-s playwright` |
| `--defuddle` | `-s defuddle` |
| `--static` | `-s static` |
| `--jina` | `-s jina` |
| `--cloudflare` | `-s cloudflare` (also enables CF file conversion, same as `-b cloudflare`) |
| `--kreuzberg` | `-b kreuzberg` |

```bash
markitai https://example.com --defuddle   # deprecated, same as: markitai https://example.com -s defuddle
```

::: warning
`--playwright`, `--defuddle`, `--static`, `--jina`, and `--cloudflare` are mutually exclusive with each other and with `-s/--strategy`. `--kreuzberg` is mutually exclusive with `-b/--backend`.
:::

## Setup Commands

### `markitai init`

Interactive setup wizard that checks dependencies, detects LLM providers (including ChatGPT OAuth, Claude/Copilot CLIs, and a `GEMINI_API_KEY` env var), and generates a configuration file.

```bash
# Interactive setup wizard
markitai init

# Quick mode (generate default config without prompts)
markitai init --yes
markitai init -y

# Generate local project config (./markitai.json)
markitai init --local

# Specify output path
markitai init -o ./markitai.json
```

### `-I, --interactive`

Enter interactive mode for guided file conversion setup.

```bash
markitai -I
```

## Configuration Commands

### `markitai config list`

Display all configuration settings.

```bash
markitai config list                        # Default format: json
markitai config list --format table         # Compact table view
markitai config list -f yaml                # Requires: uv add pyyaml
```

### `markitai config get <key>`

Get a specific configuration value.

```bash
markitai config get llm.enabled
markitai config get cache.enabled
```

### `markitai config set <key> <value>`

Set a configuration value.

```bash
markitai config set llm.enabled true
markitai config set cache.enabled false
```

### `markitai config path`

Show configuration file paths.

```bash
markitai config path
```

### `markitai config edit`

Interactively edit configuration settings with a guided menu.

```bash
markitai config edit
```

### `markitai config validate`

Validate a configuration file.

```bash
markitai config validate
markitai config validate ./markitai.json    # Validate a specific file
```

## Cache Commands

### `markitai cache stats`

Display cache statistics.

```bash
markitai cache stats
markitai cache stats -v           # Verbose mode (same as --verbose)
markitai cache stats --json       # JSON output
markitai cache stats --verbose --limit 50   # Limit entries shown (default: 20)
```

### `markitai cache clear`

Clear cached data.

```bash
markitai cache clear
markitai cache clear -y                       # Skip confirmation
markitai cache clear --include-spa-domains    # Also clear learned SPA domains
```

### `markitai cache spa-domains`

View or manage learned SPA domains. These are domains automatically detected as requiring browser rendering.

```bash
markitai cache spa-domains             # List learned domains
markitai cache spa-domains --json      # JSON output
markitai cache spa-domains --clear     # Clear all learned domains
```

::: tip
SPA domains are learned automatically when static fetch detects JavaScript requirement. This speeds up subsequent requests by skipping wasted static fetch attempts.
:::

## Diagnostic Commands

### `markitai doctor`

Check system health, dependencies, and authentication status. This is the primary diagnostic command. Exits non-zero when a required dependency is missing, so it can be used in scripts and CI.

```bash
markitai doctor
markitai doctor --fix     # Currently only auto-installs the Playwright browser; other missing components just print an install hint
markitai doctor --json    # JSON output
markitai doctor --suggest-extras   # Comma-separated pip extras for `uv tool install "markitai[...]"` — always includes browser/extra-fetch/kreuzberg/svg/heif, plus claude-agent/copilot if those CLIs are detected
```

This command verifies:
- **Playwright**: For dynamic URL fetching (SPA rendering)
- **LibreOffice**: For Office document conversion (doc, docx, xls, xlsx, ppt, pptx)
- **FFmpeg**: For audio/video file processing (mp3, mp4, wav, etc.)
- **RapidOCR**: For scanned document OCR (built-in, no external dependencies)
- **LLM API**: Configuration and model status
- **Vision Model**: For image analysis (auto-detected from litellm)
- **Local Provider Auth**: Authentication status for Claude Agent, GitHub Copilot, and ChatGPT (if configured)

Example output:
```
◆ System Check

  • Config: ~/.markitai/config.json

Required Dependencies
  ✓ Playwright: Chromium installed
  ✓ LibreOffice: installed
  ✓ RapidOCR: v1.4.0, lang: en (English)

Optional Dependencies
  ✓ FFmpeg: v6.0

LLM
  ✓ LLM API: 1 model(s) configured
  ✓ Vision Model: 1 detected: copilot/claude-haiku-4.5
  ✓ GitHub Copilot SDK: SDK + CLI installed

Authentication
  ✓ Copilot Auth: Authenticated

✓ All dependencies configured correctly
```

::: tip
`API provider(s): ...` in the `LLM API` line only appears when a genuine remote-API model (not `claude-agent/`/`copilot/`) is configured.
:::

::: tip
When using local providers (`claude-agent/` or `copilot/`), the doctor command also checks authentication status and provides resolution hints if authentication fails.
:::

## Authentication Commands

### `markitai auth`

Authentication helpers for local providers (Copilot, Claude, ChatGPT). Gemini
access is via a direct API key or OpenRouter (see [Configuration](/guide/configuration#model-naming)) — not through this command. Run with no subcommand for a one-line login-status overview of all three providers.

```bash
markitai auth                   # Overview of all providers
```

### `markitai auth copilot status`

Show GitHub Copilot CLI authentication status.

```bash
markitai auth copilot status
markitai auth copilot status --json    # JSON output
```

### `markitai auth copilot login`

Run GitHub Copilot CLI authentication.

```bash
markitai auth copilot login
```

### `markitai auth claude status`

Show Claude Code CLI authentication status.

```bash
markitai auth claude status
markitai auth claude status --json    # JSON output
```

### `markitai auth claude login`

Run Claude Code CLI authentication.

```bash
markitai auth claude login
```

### `markitai auth chatgpt status`

Show ChatGPT OAuth authentication status.

```bash
markitai auth chatgpt status
markitai auth chatgpt status --json    # JSON output
```

### `markitai auth chatgpt login`

Run ChatGPT OAuth Device Code Flow authentication.

```bash
markitai auth chatgpt login
```

::: tip
You can also use `markitai doctor` to check authentication status for all configured providers at once.
:::

## Other Options

### `--quiet, -q`

Suppress non-essential output.

```bash
markitai document.docx --quiet
```

### `-v, --verbose`

Enable verbose output.

```bash
markitai document.docx --verbose
```

### `--dry-run`

Preview conversion without writing files.

```bash
markitai document.docx --dry-run
```

### `-c, --config <path>`

Specify configuration file path.

```bash
markitai document.docx --config ./my-config.json
```

### `--config-json <json>`

Inline JSON config overrides, deep-merged over the config file (explicit CLI flags still win). Useful for agents/CI.

```bash
markitai document.docx --config-json '{"llm": {"concurrency": 4}}'
```

### `-V, --version`

Show version information.

```bash
markitai -V
```

### `-h, --help`

Show help message.

```bash
markitai -h
markitai config -h
markitai cache -h
```

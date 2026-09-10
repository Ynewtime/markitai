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

Social posts (extractor-curated content marked `content_profile: social_post`, e.g. X/Twitter posts) keep their body verbatim. The LLM only generates frontmatter metadata, so post structure and wording are never altered.

For **directory batches**, `--llm-batch` runs the enhancement through the provider's Batch API at half the list price:

```bash
markitai docs/ --llm --llm-batch -o out/       # waits up to --llm-batch-timeout (1h), then hands off
markitai --llm-batch-collect <batch-id> -o out/  # finish a handed-off batch later
```

Only files successfully converted by the current run are enhanced; unrelated files already in the output directory are excluded. Requires a single-model OpenAI or Anthropic pool. Cache hits are served instantly; failed batch requests are retried live at standard rates, and base output remains available if enhancement fails. On OpenAI, reasoning models run with reasoning off inside a batch — the batch deployments require that of function-tool calls.

`--alt`/`--desc` and `--screenshot` ride the same job, so image analysis and page-image enhancement are batched at the same discount; a document with more pages than fit one call is enhanced live instead, since each extra batch round is a separate wait. Not yet combinable with `--ocr`, whose pages are never rendered — the batch converts with the LLM off first, and that is the branch which reads scanned pages with local OCR.

::: tip
`--llm`, `--alt`, `--desc`, `--ocr`, and `--screenshot` all have `--no-*` counterparts (`--no-llm`, `--no-alt`, `--no-desc`, `--no-ocr`, `--no-screenshot`) to explicitly disable a feature a preset would otherwise enable, for example `--preset rich --no-desc`.
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
markitai document.pdf --preset rich --no-desc   # Rich without desc; any preset feature can be toggled off with --no-*
```

### `--profile <name>`

Shape the output for a downstream consumer. Orthogonal to `--preset`: presets pick which features run, a profile picks what the written output looks like. Without `--profile` the output is unchanged.

| Profile | Effect |
|---------|--------|
| `rag` | Visible `assets/` directory (instead of hidden `.markitai/assets/`), `<!-- page: N -->` markers for PDFs, pipe-table column checks |
| `obsidian` | Visible `assets/` directory; optional wikilink image refs via `output.wikilinks` |
| `okf` | Frontmatter aligned with the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog) spec |

```bash
markitai document.pdf --profile rag -o out/
markitai document.pdf --preset rich --profile rag -o out/   # features + shape
```

See [Output Profiles](./output-profiles.md) for details.

### `--alt`

Generate alt text for images using AI. Requires `--llm`. Without it, image analysis is skipped with a warning.

```bash
markitai document.pdf --llm --alt
```

### `--desc`

Generate detailed descriptions for images. Requires `--llm`. Without it, image analysis is skipped with a warning.

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
| `--screenshot-only` | Screenshots only (no `.md` files) |
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
For **file input** (PDF/PPTX), `--screenshot-only` without `--llm` does **not** skip `.md`. It still writes the normal extracted-text markdown alongside the screenshots. The "no `.md` files" guarantee above only applies to URL input.
:::

Use `--no-screenshot-only` to turn the mode back off when a config file enables it.

### `--ocr`

Enable OCR for scanned documents.

```bash
markitai scanned.pdf --ocr
```

- Without `--llm`: `--ocr` uses local RapidOCR.
- With `--llm`: the vision model reads page images directly (VLM OCR) instead of RapidOCR. No OCR backend needed, but page images go to the remote model; `MARKITAI_NO_VLM_OCR=1` forces the local RapidOCR path.

For a single image input, enable `--ocr` to extract text or `--llm` to analyze the image. If neither feature is enabled, Markitai exits with status 1 instead of reporting a successful conversion with no output.

#### Mathematics in PDFs

PDF text extraction has no notion of a formula, so how much of one survives depends on how much of the page a model gets to look at:

| Run | What happens to a formula |
|-----|---------------------------|
| `--ocr --llm` | Inline math is written as `$...$` LaTeX, and any prose the broken extraction had swallowed comes back |
| `--alt` / `--desc` | A display equation reaches markitai as an image; its LaTeX is recovered into `images.json` (`text`), with the alt text summarising it |
| Neither | A display equation is kept as an image reference — nothing is lost, but nothing is text either. Inline math stays as extraction noise |

Web pages are different: MathJax and MathML are converted to `$...$` / `$$...$$` with no model involved.

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

Use `--no-pure` to restore frontmatter and post-processing when a config file enables pure mode.

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

Use `--compress` to force compression back on when a config file disables it.

## Output Options

### `-o, --output <path>`

Specify the output location. A single file or URL may instead name a `.md` file here (e.g. `-o result.md`). A batch (directory or `.urls` input) requires a directory: a `.md` value is rejected with a usage error rather than turned into a directory. If omitted, single file/URL conversions print to stdout instead; directory-batch and `.urls`-list input require `-o`.

```bash
markitai document.docx -o ./output
markitai document.docx -o ./result.md
```

### `--json`

Print one machine-readable JSON result on stdout and suppress progress output. Needs `-o`, because stdout carries the JSON, and it is not compatible with `--llm-batch-collect` or `--dry-run` (a dry run reports no item the envelope could carry).

The document is `{version, ok, error, items[], totals}`; `version` is the envelope schema version, not the markitai release.

- `items[]` — one entry per work item with `kind`, `source`, `status` (`completed` / `failed` / `skipped`), `output`, `error`, `skip_reason`, `images`, `screenshots`, `cost_usd`, `duration_s`, cache flags, `fetch_strategy` and `llm_usage`.
- `error` — a run-level failure that produced no item (a missing input path, a rejected URL scheme, an unreadable or invalid config file, an interrupt), otherwise `null`.
- `totals` — `total`, `completed`, `failed`, `skipped`, `cost_usd`, and `duration_s` (the sum of the per-item durations; a concurrent batch finishes sooner than that sum).
- `ok` — `false` when any item failed or a run-level `error` is present.

```bash
markitai ./docs -o ./output --json
markitai document.pdf -o ./output --json | jq '.items[] | select(.status == "failed")'
```

The exit code keeps its usual meaning (see [Exit codes](#exit-codes)); scripts should read `ok` as well, since a partial batch exits `10` while still emitting a JSON document. Argument/usage errors (an unknown flag, `--json` without `-o`, or a nonexistent path passed to `-c`) exit on stderr without JSON. Runtime configuration errors, including malformed JSON, a non-object root or invalid UTF-8, appear in the JSON `error` field. Interactive mode (`-I`) re-runs the gathered command in a subprocess without `--json`, so combine the two only for human sessions.

With `--llm-batch`, the envelope includes final enhanced paths and combined usage/costs. On timeout, unfinished enhancements are marked failed with a pending explanation and the process exits `2`; use the printed collect command to finish them. Collection itself does not support `--json`.

### `--resume`

Resume interrupted batch processing. Completed files are skipped, `FAILED`/interrupted (`IN_PROGRESS` at crash time) files are retried, and newly-added files are picked up. The command reports `Resuming batch: N completed, M remaining`. This option applies only to batch (directory/`.urls`) input and is ignored for a single file/URL.

```bash
markitai ./docs -o ./output --resume
```

### `--record-history` {#record-history}

Record the completed run as a job in the `markitai serve` history (stored under `~/.markitai/serve/jobs/`, with outputs and referenced assets copied in), so it shows up in the web UI history page live, without restarting the server. CLI-recorded entries are marked with a "CLI" badge and share the same seven-day cleanup, deletion, and archive download as web-created jobs.

```bash
markitai document.docx -o ./output --record-history
```

Precedence: `--record-history` / `--no-record-history` > `MARKITAI_RECORD_HISTORY` (truthy: `1`/`true`/`yes`/`on`; a set-but-falsy value is an explicit opt-out) > `history.record` in the config > default (off). Recording is skipped in stdout/pipe mode and never fails a conversion; a one-line confirmation goes to stderr (suppressed by `--quiet`). See [Configuration → Environment Variables](/guide/configuration#markitai-settings).

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

Use `--cache` to allow cache reads again when a config file disables them.

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

Plain text: one URL per line, with an optional custom output name after whitespace:
```text
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

Successful URLs are kept even when another URL in the batch fails. A partially successful `.urls` run exits with status 10, so scripts and CI can distinguish it from complete success.

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
| `static` | Static HTTP fetch with native webextract; fast, no JS, no external API |
| `playwright` | Browser rendering via Playwright for JS-heavy SPA sites (e.g. x.com) |
| `defuddle` | Defuddle API: free, no authentication, excellent content cleaning |
| `jina` | Jina Reader API, a cloud-based alternative when browser rendering is unavailable |
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
| `cloudflare` | Cloudflare Workers AI `toMarkdown`; requires CF credentials |

```bash
markitai document.pdf -b cloudflare
markitai https://example.com -s playwright -b cloudflare   # -s and -b combine freely
```

::: tip
Cloudflare Browser Rendering is available on the Free plan. Workers AI `toMarkdown` is free for PDF/Office/CSV/XML; image conversion uses Neurons quota. For formats with a local converter, the native converters generally produce higher-quality output. `-b cloudflare` warns when a better local converter is available.
:::

### Removed per-backend flags

The six per-backend aliases below were **removed** in 1.0.0. Passing one is now a usage error that names its replacement, so a stale script fails loudly instead of quietly converting with the wrong engine:

| Removed flag | Use instead |
|------------------|------------|
| `--playwright` | `-s playwright` |
| `--defuddle` | `-s defuddle` |
| `--static` | `-s static` |
| `--jina` | `-s jina` |
| `--cloudflare` | `-s cloudflare` (add `-b cloudflare` for CF file conversion) |
| `--kreuzberg` | *(nothing — `.rtf` converts natively since 1.0.0)* |

```bash
markitai https://example.com -s defuddle   # replaces the old --defuddle alias
```

The mutual-exclusion rules those aliases needed are gone with them: `-s/--strategy` and `-b/--backend` are orthogonal and combine freely.

### `--no-remote-fetch`

Never send URLs to remote extraction services (Defuddle, Jina, Cloudflare). Same effect as `MARKITAI_NO_REMOTE_FETCH=1`, but stated on the command line, so the privacy choice is explicit instead of inherited from `--quiet`.

```bash
markitai https://example.com -o ./output --no-remote-fetch
```

`--quiet` suppresses the remote-fetch consent prompt, so with `fetch.remote_consent=ask` a quiet run skips every remote strategy. The CLI now prints a note on stderr in that case and points at `--no-remote-fetch`. See [Fetch Policy → Remote fallback and local-only URLs](/guide/fetch-policy#remote-fallback-and-local-only-urls).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success, including `--dry-run` |
| `1` | A single item failed, or a runtime error (including `config validate` failing, or `cache stats --json` reading a corrupt cache) |
| `2` | Argument/usage error, or a Batch API wait timed out and requires `--llm-batch-collect` |
| `10` | Batch run finished with partial failures (directory batch or URL batch); successful items are kept |

`--json` does not change these codes: a partial batch still exits `10` while printing a JSON document, so scripts should check the `ok` field too.

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

Display the effective configuration. Secret values are redacted recursively by default, including values nested inside provider, fetch, or authentication settings.

```bash
markitai config list                        # Default format: json
markitai config list --format table         # Compact table view
markitai config list -f yaml                # Requires: uv add pyyaml
markitai config list --show-secrets         # Explicitly reveal original values
```

::: warning
Use `--show-secrets` only for local inspection. Do not paste its complete output into an issue, chat, CI log, or other shared channel.
:::

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

Check core health, optional capabilities, and authentication status. Missing unused optional tools are warnings, not a failed installation. The command exits non-zero when a configured Playwright workflow cannot launch; an active API model references a missing environment variable; an actively configured local LLM provider cannot load or authenticate; or a requested automatic repair does not succeed. Scripts and CI can therefore rely on the result.

```bash
markitai doctor
markitai doctor --fix     # Safely install and re-check Chromium when the Playwright package is already present
markitai doctor --json    # JSON output
markitai doctor --suggest-extras   # Comma-separated pip extras for `uv tool install "markitai[...]"`; includes browser/extra-fetch/svg/heif/ocr, plus detected provider extras
```

Every check is a capability report, and a capability you have not enabled never fails the run:

- **Optional: RapidOCR** for `--ocr` on scanned PDFs and images. It ships in the `ocr` extra rather than the core install, so "not installed" here means "OCR is off", not "something is broken"
- **Optional: VLM OCR** for `--ocr --llm` on scanned documents: the vision model reads page images instead of RapidOCR. Available when a vision-capable model is configured; `MARKITAI_NO_VLM_OCR=1` forces the local RapidOCR path
- **Optional until configured: Playwright** for dynamic URL fetching (SPA rendering). It becomes a blocking check when `fetch.strategy` is `playwright` or `screenshot.enabled` is true
- **Optional: LibreOffice** for PPTX slide rendering (on macOS, an installed Microsoft PowerPoint is used as a fallback)
- **LLM API**: Configuration and model status
- **Vision Model**: For image analysis (auto-detected from litellm)
- **Local Provider Auth**: Authentication status for Claude Agent, GitHub Copilot, and ChatGPT (if configured)

Every normal doctor run performs an isolated, timeout-bounded Chromium launch smoke test when the browser files exist, so a stale marker or missing Linux system library cannot produce a green result. `doctor --fix` never adds a Python package to the current project.

If Playwright is installed but Chromium is missing or unusable, it installs Chromium with Markitai's own interpreter and repeats the runtime check. A failed launch stays non-zero; on Linux the message includes the `playwright install-deps chromium` recovery command. If the Playwright package itself is missing, the command exits safely and tells you to replace the isolated tool install with `uv tool install 'markitai[browser]' --force` or the pipx equivalent.

`--json` and `--fix` are mutually exclusive: JSON is a read-only health snapshot, while repair is an interactive human-facing operation.

Example output:
```text
◆ System Check

  • Config: ~/.markitai/config.json

Optional Capabilities
  ⚠ RapidOCR: not installed — --ocr unavailable (everything else works)
  ⚠ Playwright: Playwright not installed
  ⚠ LibreOffice: Not installed

LLM
  ✓ LLM API: 1 active model(s) configured
  ✓ Vision Model: 1 detected: copilot/claude-haiku-4.5
  ✓ GitHub Copilot SDK: SDK + CLI installed

Authentication
  ✓ Copilot Auth: Authenticated

⚠ Core check passed (3 required/configured checks passed, 3 non-blocking warnings)
```

::: tip
`API provider(s): ...` in the `LLM API` line only appears when a genuine remote-API model (not `claude-agent/`, `copilot/`, or `chatgpt/`) is active. Explicit `env:` references on active API models are resolved by doctor; a missing variable is a configured failure rather than a green model count.
:::

::: tip
When using local providers (`claude-agent/` or `copilot/`), the doctor command also checks authentication status and provides resolution hints if authentication fails.
:::

## Authentication Commands

### `markitai auth`

Authentication helpers for local providers (Copilot, Claude, ChatGPT). Gemini
access is via a direct API key or OpenRouter (see [Configuration](/guide/configuration#model-naming)), not through this command. Run with no subcommand for a one-line login-status overview of all three providers.

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

## Server & Agent Commands

### `markitai serve`

Run the local web workspace and its REST + SSE API. It needs the `serve` extra (`fastapi`, `uvicorn`, `python-multipart`):

```bash
uv tool install "markitai[serve]" --force
markitai serve                    # http://127.0.0.1:3600, opens the browser
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host <interface>` | `127.0.0.1` | Interface to bind. The default is reachable only from this machine; any other value publishes the API to every host that can reach it, and those requests then need the startup access token (unless `--no-auth`) |
| `--port <n>` | `3600` | Port to listen on |
| `--no-open` | off | Do not open the browser after startup |
| `--no-auth` | off | Disable the access token. Other machines then need no credential, URL targets are restricted to public addresses and LLM settings are blocked, but file uploads and history access, downloads and deletion remain available; loopback keeps full access either way |
| `--allowed-host <hostname>` | — | Additional hostname accepted in the `Host`/`Origin` headers (repeatable). `localhost` and IP literals are always accepted; other names are rejected to block DNS rebinding. This is name filtering, not authentication |

An access token is generated at startup and printed as a ready-to-open URL (`http://host:port/#token=…`), or pinned with `MARKITAI_SERVE_TOKEN`. Requests from this machine never need it; requests from any other machine must carry it — the web UI reads it from the URL fragment and stores it in `sessionStorage` (the browser strips it from the address bar), while scripts send `Authorization: Bearer <token>` (or `?token=` on download/SSE URLs). The token is a credential: anyone holding it gets conversions, history, downloads, deletion and LLM settings. See the [Web Workspace guide](/guide/serve) for the workspace itself and the full API table.

### `markitai mcp`

Run the bundled MCP server over stdio for AI agents:

```bash
markitai mcp
```

It is the same server as the `markitai-mcp` console script, so registries and clients can start it through the host CLI package without knowing a second executable name (`uvx --from "markitai[mcp]" markitai mcp`). It exposes `convert_document`, `convert_url`, `batch_convert` and `job_status`; see the [MCP guide](/guide/mcp) for client setup, LLM configuration and the extras each capability needs.

## Other Options

### `--quiet, -q`

Suppress progress and informational messages. Errors still go to stderr. If a single conversion normally writes Markdown to stdout, `--quiet` preserves that payload; it does not turn the result into an empty stream. The one-time remote privacy disclosure is also intentionally kept on stderr when a remote service may receive a public URL.

```bash
markitai document.docx --quiet
```

### `-v, --verbose`

Enable verbose output.

```bash
markitai document.docx --verbose
```

### `--log-level <level>`

Minimum level for the **log file** (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`), overriding `log.level` from the config. File logging is off until `log.dir` is set, so this flag has no effect without one; it applies to conversion runs (subcommands print their own output). Console output stays governed by `--verbose` / `--quiet`; this flag never makes the terminal louder.

```bash
markitai ./docs -o out --log-level WARNING
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

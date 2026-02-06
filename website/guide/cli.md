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

Enable LLM-powered format cleaning and optimization.

```bash
markitai document.docx --llm
```

### `--preset <name>`

Use a predefined configuration preset.

| Preset | Description |
|--------|-------------|
| `rich` | LLM + alt + desc + screenshot |
| `standard` | LLM + alt + desc |
| `minimal` | Basic conversion only |

```bash
markitai document.pdf --preset rich
```

### `--alt`

Generate alt text for images using AI.

```bash
markitai document.pdf --alt
```

### `--desc`

Generate detailed descriptions for images.

```bash
markitai document.pdf --desc
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
For URLs, `--screenshot` automatically upgrades the fetch strategy to `playwright` if needed. The screenshot is saved as `{domain}_path.full.jpg` in the `screenshots/` subdirectory.
:::

### `--screenshot-only`

Capture screenshots only without extracting content. Behavior depends on `--llm`:

| Command | Output |
|---------|--------|
| `--screenshot-only` | Screenshots only (no .md files) |
| `--llm --screenshot-only` | .md + .llm.md + screenshots (LLM extracts from screenshots) |

```bash
# Just capture screenshots
markitai https://example.com --screenshot-only

# LLM extracts content purely from screenshots
markitai https://example.com --llm --screenshot-only
```

::: tip
Use `--llm --screenshot-only` for pages where traditional content extraction fails (e.g., heavy JavaScript sites, social media).
:::

### `--ocr`

Enable OCR for scanned documents.

```bash
markitai scanned.pdf --ocr
```

### `--no-compress`

Disable image compression.

```bash
markitai document.pdf --no-compress
```

## Output Options

### `-o, --output <path>`

Specify output directory.

```bash
markitai document.docx -o ./output
```

### `--resume`

Resume interrupted batch processing.

```bash
markitai ./docs -o ./output --resume
```

## Concurrency Options

### `--llm-concurrency <n>`

Number of concurrent LLM requests.

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

When the input is a `.urls` file, Markitai automatically processes it as a URL batch.

```bash
markitai urls.urls -o ./output
```

The `.urls` file format:
```
# Comments start with #
https://example.com/page1
https://example.com/page2
```

### `--url-concurrency <n>`

Number of concurrent URL fetch operations (default: 5). This is separate from `--batch-concurrency` to prevent slow URLs from blocking file processing.

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `--playwright`

Force browser rendering for URL fetching using Playwright. Useful for JavaScript-heavy SPA websites (e.g., x.com, dynamic web apps).

```bash
markitai https://x.com/user/status/123 --playwright
```

::: tip
To pre-install Playwright browsers:
```bash
uv run playwright install chromium
# Linux also requires system dependencies:
uv run playwright install-deps chromium
```
:::

### `--jina`

Force Jina Reader API for URL fetching. A cloud-based alternative when browser rendering is not available.

```bash
markitai https://example.com --jina
```

::: warning
`--playwright` and `--jina` are mutually exclusive. You can only use one at a time.
:::

## Setup Commands

### `markitai init`

Interactive setup wizard that checks dependencies, detects LLM providers, and generates a configuration file.

```bash
# Interactive setup wizard
markitai init

# Quick mode (generate default config without prompts)
markitai init --yes

# Generate global config
markitai init --global

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
markitai config list
markitai config list --format json
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

### `markitai config validate`

Validate a configuration file.

```bash
markitai config validate
```

## Cache Commands

### `markitai cache stats`

Display cache statistics.

```bash
markitai cache stats
markitai cache stats --verbose    # Verbose mode
markitai cache stats --json       # JSON output
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

Check system health, dependencies, and authentication status. This is the primary diagnostic command.

```bash
markitai doctor
markitai doctor --fix     # Auto-fix missing components
markitai doctor --json    # JSON output
```

This command verifies:
- **Playwright**: For dynamic URL fetching (SPA rendering)
- **LibreOffice**: For Office document conversion (doc, docx, xls, xlsx, ppt, pptx)
- **FFmpeg**: For audio/video file processing (mp3, mp4, wav, etc.)
- **RapidOCR**: For scanned document OCR (built-in, no external dependencies)
- **LLM API**: Configuration and model status
- **Vision Model**: For image analysis (auto-detected from litellm)
- **Local Provider Auth**: Authentication status for Claude Agent and GitHub Copilot (if configured)

Example output:
```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Dependency Status                               │
├─────────────────────┬────────┬──────────────────────────────┬────────────┤
│ Component           │ Status │ Description                  │ Details    │
├─────────────────────┼────────┼──────────────────────────────┼────────────┤
│ Playwright          │   ✓    │ Browser automation           │ Installed  │
│ LibreOffice         │   ✓    │ Office document conversion   │ v7.6.4     │
│ FFmpeg              │   ✓    │ Audio/video processing       │ v6.0       │
│ RapidOCR            │   ✓    │ OCR for scanned documents    │ v1.4.0     │
│ LLM API (copilot)   │   ✓    │ Content enhancement          │ 1 model(s) │
│ Copilot Auth        │   ✓    │ GitHub Copilot auth status   │ Authenticated │
│ Vision Model        │   ✓    │ Image analysis               │ 1 detected │
└─────────────────────┴────────┴──────────────────────────────┴────────────┘
```

::: tip
When using local providers (`claude-agent/` or `copilot/`), the doctor command also checks authentication status and provides resolution hints if authentication fails.
:::

## Other Options

### `--quiet, -q`

Suppress non-essential output.

```bash
markitai document.docx --quiet
```

### `--verbose`

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

### `-v, --version`

Show version information.

```bash
markitai -v
```

### `-h, --help`

Show help message.

```bash
markitai -h
markitai config -h
markitai cache -h
```

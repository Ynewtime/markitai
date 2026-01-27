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
- **URLs**: Captures full-page screenshots using agent-browser

```bash
# Document screenshots
markitai document.pdf --screenshot
markitai presentation.pptx --screenshot

# URL screenshots
markitai https://example.com --screenshot
```

::: tip
For URLs, `--screenshot` automatically upgrades the fetch strategy to `browser` if needed. The screenshot is saved as `{domain}_path.full.jpg` in the `screenshots/` subdirectory.
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

### `--agent-browser`

Force browser rendering for URL fetching. Useful for JavaScript-heavy SPA websites (e.g., x.com, dynamic web apps).

```bash
markitai https://x.com/user/status/123 --agent-browser
```

::: tip
Requires `agent-browser` to be installed:
```bash
pnpm add -g agent-browser
agent-browser install           # Download Chromium
agent-browser install --with-deps  # Linux: also install system deps
```
See [agent-browser documentation](https://github.com/vercel-labs/agent-browser) for details.
:::

### `--jina`

Force Jina Reader API for URL fetching. A cloud-based alternative when browser rendering is not available.

```bash
markitai https://example.com --jina
```

::: warning
`--agent-browser` and `--jina` are mutually exclusive. You can only use one at a time.
:::

## Configuration Commands

### `markitai config list`

Display all configuration settings.

```bash
markitai config list
markitai config list --json
```

### `markitai config init`

Create a new configuration file.

```bash
markitai config init
markitai config init -o ~/.markitai/
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
markitai cache stats -v           # Verbose mode
markitai cache stats --json       # JSON output
markitai cache stats --scope project  # Project cache only
```

### `markitai cache clear`

Clear cached data.

```bash
markitai cache clear
markitai cache clear --scope project  # Clear project cache only
markitai cache clear --scope global   # Clear global cache only
markitai cache clear --include-spa-domains  # Also clear learned SPA domains
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

### `markitai check-deps`

Check all optional dependencies and their status. Useful for diagnosing setup issues.

```bash
markitai check-deps
markitai check-deps --json    # JSON output
```

This command verifies:
- **agent-browser**: For dynamic URL fetching (SPA rendering)
- **LibreOffice**: For Office document conversion (doc, docx, xls, xlsx, ppt, pptx)
- **Tesseract OCR**: For scanned document processing (optional, RapidOCR is built-in)
- **LLM API**: Configuration and connectivity status

## Other Options

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

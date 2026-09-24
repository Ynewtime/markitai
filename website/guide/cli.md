# CLI Reference

## Basic Usage

```bash
markitai <input> [options]
```

The input is a file (`document.docx`), a directory (`./docs`) or a URL (`https://example.com`). `mkai` is a shorter alias for the same command.

## Conversion Options

### `--llm`

Clean up formatting and generate frontmatter with an LLM. By default markitai writes only `.llm.md`; add `--keep-base` to keep the plain `.md` as well.

A failed enhancement fails the item. That covers an invalid key, a provider error, a timeout, a refusal, and a result that fell back to unenhanced text. markitai still writes the plain `.md` so no content is lost, but the item is reported `failed`: a single input exits with `1`, a batch with `10`, and `--json` reports `ok: false` with the error. A document too long for one LLM call is enhanced in chunks, so the whole text is cleaned.

```bash
markitai document.docx --llm
```

Social posts (X/Twitter and similar) keep their body verbatim. The LLM only writes the frontmatter.

`--llm`, `--alt`, `--desc`, `--ocr` and `--screenshot` each have a `--no-*` twin (`--no-llm`, `--no-alt`, `--no-desc`, `--no-ocr`, `--no-screenshot`) to switch off something a preset or config file turned on, for example `--preset rich --no-desc`.

#### Batch API

For directory batches, `--llm-batch` sends the enhancement through the provider's Batch API at half the list price:

```bash
markitai docs/ --llm --llm-batch -o out/         # waits up to --llm-batch-timeout (1h), then hands off
markitai --llm-batch-collect <batch-id> -o out/  # finish a handed-off batch later
```

It needs a single OpenAI or Anthropic model, checked before anything is converted. The batch uses that model's `api_key` and `api_base` from `llm.model_list` (including `env:` references), like a real-time run; without them it falls back to the provider's environment variables. Pass the same `-c` config to `--llm-batch-collect`; image answers are applied the way the submitting run's `--alt`/`--desc` asked, even when that config does not repeat them. Image analysis and screenshots ride the same batch job, including the images of a document too long for one request (that document's text is enhanced live, in chunks). An image the cache already answers is not sent again, and its alt text and `images.json` entry are still written. `--ocr` does not work in batch mode yet. A `--screenshot-only` URL capture has no Markdown to enhance in batch mode: it is kept as is, and the item says so under `warnings`.

If enhancement fails, markitai still writes the base output and fails the item. A submission that fails (network, credentials, provider error) ends with a one-line error and exit `1`, and nothing is left behind to collect. If the wait times out or loses contact with the API, the batch keeps running: markitai prints the collect command, even under `--quiet` or `--json`, and exits `2`. Ctrl-C during the wait prints the same collect command. If some items also failed, the exit code is still `2`, because the batch has to be collected; the failures show in `--json` (`ok: false`, `totals.failed`) and are counted on stderr. `--resume` leaves out documents that are still in a batch submitted but never collected, so they are not paid for twice, and prints that batch's collect command. A batch that ended without results (failed, expired or cancelled) no longer holds its documents back. An image the batch could not analyze keeps its original alt text, and the item reports it under `warnings`. Collecting a batch a second time prints that it was already collected and leaves the outputs alone.

### `-p, --preset <name>`

Use a bundle of common flags:

| Preset | LLM | alt | desc | screenshot | OCR |
|--------|:---:|:---:|:----:|:----------:|:---:|
| `minimal` | – | – | – | – | – |
| `standard` | ✓ | ✓ | ✓ | – | – |
| `rich` | ✓ | ✓ | ✓ | ✓ | – |

No preset turns on OCR; `--ocr` is always explicit. Define your own presets in the [config file](/guide/configuration#presets).

```bash
markitai document.pdf --preset rich
markitai document.pdf --preset rich --no-desc
```

### `--profile <name>`

Shape the written output for a downstream consumer. Presets pick which features run, a profile picks what the files look like. Without `--profile` the output is unchanged.

| Profile | Effect |
|---------|--------|
| `rag` | Visible `assets/` directory, `<!-- page: N -->` markers for PDFs, pipe-table column checks |
| `obsidian` | Visible `assets/` directory, optional wikilink image references |
| `okf` | Frontmatter aligned with the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog) |

```bash
markitai document.pdf --profile rag -o out/
markitai document.pdf --preset rich --profile rag -o out/
```

See [Output Profiles](./output-profiles.md) for details.

### `--alt`

Generate alt text for images. Needs `--llm`.

```bash
markitai document.pdf --llm --alt
```

### `--desc`

Generate detailed image descriptions into `images.json`. Needs `--llm`.

```bash
markitai document.pdf --llm --desc
```

### `--screenshot`

Render PDF pages and PPTX slides as JPEG images, or capture a full-page screenshot of a URL with Playwright. Screenshots go to `.markitai/screenshots/`.

```bash
markitai document.pdf --screenshot
markitai https://example.com --screenshot
```

For URLs, `--screenshot` switches the fetch strategy to `playwright` when needed. The capture shows the page as rendered, before markitai strips navigation and styling for text extraction; a long page is split into tiles (`name.full.jpg`, `name.full--1.jpg`, ...). URLs that differ only in their query string get separate files. When no screenshot can be taken (Playwright or Chromium missing, the site refuses the headless browser), the Markdown is still written and markitai prints a warning, which `--json` also reports in the item's `warnings`.

### `--screenshot-only`

Capture screenshots and skip content extraction. For URLs:

| Command | Output |
|---------|--------|
| `--screenshot-only` | Screenshots only, no `.md` |
| `--llm --screenshot-only` | `.llm.md` written by the LLM from the screenshots, plus the screenshots |

```bash
markitai https://example.com --screenshot-only
markitai https://example.com --llm --screenshot-only
```

`--llm --screenshot-only` is the fallback for pages where text extraction fails, such as heavy JavaScript sites or canvas apps: when every text strategy fails, the page is still captured and the run completes from the screenshot. Without a screenshot there is nothing to deliver, so a URL whose capture fails is a failed item (exit code `1`, or a failed entry in a batch). For PDF and PPTX files, `--screenshot-only` without `--llm` still writes the normal extracted `.md` next to the screenshots. `--no-screenshot-only` turns the mode off when a config file enables it.

### `--ocr`

Read scanned PDFs and images.

```bash
markitai scanned.pdf --ocr
```

Without `--llm`, OCR runs locally with RapidOCR (`markitai[ocr]`). With `--llm`, the vision model reads the page images instead, so you need no OCR extra, but the pages go to the model. `MARKITAI_NO_VLM_OCR=1` forces the local path.

PDF pages that already have a text layer keep it (headings, lists, tables and images included); OCR reads the scanned pages and the pictures on the others. For PPTX, the slide text comes from the file and OCR reads the pictures on the slides. A file OCR cannot read fails instead of producing an empty success; a blank image converts with a notice that no text was found.

A single image input needs `--ocr` or `--llm`; with neither, markitai exits with status 1 rather than reporting an empty success.

#### Mathematics in PDFs

| Run | What happens to a formula |
|-----|---------------------------|
| `--ocr --llm` | Inline math becomes `$...$` LaTeX |
| `--alt` or `--desc` | A display equation is an image; its LaTeX lands in `images.json` |
| Neither | Display equations stay image references, inline math stays extraction noise |

Web pages need no model: MathJax and MathML are converted to `$...$` and `$$...$$` directly.

### `--pure`

Output plain Markdown with no frontmatter. With `--llm`, the model only cleans the text and adds no metadata.

```bash
markitai document.docx --pure
markitai document.docx --llm --pure
```

::: warning
`--pure` overrides `--alt`, `--desc` and `--screenshot`. Combining them prints a warning.
:::

`--no-pure` restores frontmatter when a config file enables pure mode.

### `--keep-base`

Write the plain `.md` alongside `.llm.md` in LLM mode.

```bash
markitai document.docx --llm --keep-base
```

### `--no-compress`

Keep images at their original size and format. `--compress` forces compression back on when a config file disables it.

```bash
markitai document.pdf --no-compress
```

## Output Options

### `-o, --output <path>`

Where to write. A directory works for any input; a single file or URL may also name a `.md` file (`-o result.md`). Without `-o`, a single file or URL prints to stdout. Batches (directories and `.urls` files) require a directory.

```bash
markitai document.docx -o ./output
markitai document.docx -o ./result.md
```

### `--json`

Print one machine-readable JSON result on stdout and suppress progress output. Requires `-o`.

```bash
markitai ./docs -o ./output --json
markitai document.pdf -o ./output --json | jq '.items[] | select(.status == "failed")'
```

The document is `{version, ok, error, batch, items[], totals}`:

- `items[]`: one entry per input with `source`, `status` (`completed`, `failed`, `skipped`, `pending`), `output`, `error`, `warnings`, `cost_usd`, `duration_s`, `fetch_strategy`, `screenshots` and `llm_usage`. `pending` only comes from an `--llm-batch` run that stopped waiting: the base `.md` is written and the enhancement is still running in the batch. `warnings` lists problems that did not fail the item, such as an image whose analysis failed or a requested screenshot that could not be captured.
- `error`: a run-level failure that produced no item, otherwise `null`.
- `batch`: after an `--llm-batch` handoff, `{id, status, collect_command}` for the batch still running; otherwise `null`.
- `totals`: counts by status plus `cost_usd` and `duration_s`.
- `ok`: `false` when any item failed or is pending, or `error` is set.

The exit code keeps its usual meaning (see [Exit codes](#exit-codes)); a partial batch exits `10` while still printing JSON, so scripts should check `ok` too. Usage errors go to stderr without JSON. You cannot combine `--json` with `--dry-run` or `--llm-batch-collect`.

### `--resume`

Resume an interrupted batch. It skips completed files, retries failed and interrupted ones, and picks up new files. Retried items overwrite their own earlier output instead of adding a `.v2` copy; an item with no recorded output name (it failed before claiming one, or the state comes from an older version) follows `output.on_conflict` like a new item, so it never overwrites a file the batch did not write. Works for a directory and for a `.urls` list; progress is saved on Ctrl-C too. The state records absolute output paths, so you can resume from another working directory (`markitai ../docs -o ../output --resume`).

```bash
markitai ./docs -o ./output --resume
markitai links.urls -o ./output --resume
```

### `--record-history` {#record-history}

Add this run to the [web workspace](/guide/serve#history) history, with a CLI badge and seven-day retention.

```bash
markitai document.docx -o ./output --record-history
```

Precedence: `--record-history` / `--no-record-history`, then `MARKITAI_RECORD_HISTORY`, then `history.record` in the config, then off. Stdout mode skips recording, and a failed recording never fails the conversion.

## Concurrency Options

### `--llm-concurrency <n>`

Concurrent LLM requests (default 10).

```bash
markitai ./docs --llm --llm-concurrency 10
```

### `-j, --batch-concurrency <n>`

Concurrent file conversions (default 10). URLs have their own pool, see `--url-concurrency`.

```bash
markitai ./docs -o ./output -j 4
```

## Cache Options

### `--no-cache`

Skip cached LLM results and fetched pages, and call the API or fetch the URL again. Results are still written to the cache. `--cache` re-enables reads when a config file disables them.

```bash
markitai document.docx --llm --no-cache
```

### `--no-cache-for <patterns>`

Skip the cache for specific files, URLs or globs, comma-separated. A pattern is matched against a URL in full, without its scheme, against its host and against its last path segment, so `"https://example.com/*"`, `"example.com/docs/*"`, `"*.example.com"` and `"*.pdf"` all work.

```bash
markitai ./docs --no-cache-for "*.pdf,reports/**"
markitai urls.urls --no-cache-for "news.example.com"
```

## URL Options

### `.urls` File Support

markitai treats a `.urls` file as a URL batch. Directory batches also pick up any `.urls` files inside the tree.

```bash
markitai urls.urls -o ./output
```

The file is plain text with one URL per line and an optional output name after a space, or a JSON array of strings or `{"url", "output_name"}` objects. Lines starting with `#` are comments.

```text
https://example.com/page1
https://example.com/page2 custom_name
```

Each line is its own item, also for `--resume`: the same URL may appear twice under different output names and both files are written. A line that repeats an earlier URL *and* name is skipped with a warning.

When one URL fails, the successful ones stay; a partially successful run exits with status 10.

### `--glob, -g <pattern>`

Restrict a directory batch to matching relative paths. Repeat for several patterns; prefix with `!` to exclude.

```bash
markitai ./docs -o ./output -g "*.pdf" -g "*.docx"
markitai ./docs -o ./output -g '!drafts/**'
```

Use single quotes around `!` patterns in shells with history expansion.

### `--max-depth <n>`

How deep to scan a directory (default 5). `0` scans only the directory itself.

```bash
markitai ./docs -o ./output --max-depth 2
```

### `--url-concurrency <n>`

Concurrent URL fetches (default 5), separate from file conversions so slow pages never block local files.

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `-s, --strategy <name>`

How to fetch URLs:

| Value | Description |
|-------|-------------|
| `auto` (default) | Try strategies in policy order, local first |
| `static` | Plain HTTP fetch with the built-in extractor. Fast, no JavaScript, nothing leaves your machine |
| `playwright` | Browser rendering for JavaScript-heavy sites. Needs `markitai[browser]` |
| `defuddle` | Defuddle API, free, no key |
| `jina` | Jina Reader API, needs `JINA_API_KEY` |
| `cloudflare` | Cloudflare Browser Rendering, needs `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` |

```bash
markitai https://example.com -s defuddle
markitai https://x.com/user/status/123 -s playwright
```

See [Fetch Policy](/guide/fetch-policy) for the order `auto` uses and [Cloudflare Settings](/guide/configuration#cloudflare-settings) for the token.

### `-b, --backend <name>`

How to convert files. Independent of `-s`, which only affects URLs.

| Value | Description |
|-------|-------------|
| `native` (default) | Built-in converters |
| `cloudflare` | Cloudflare Workers AI `toMarkdown`. Needs Cloudflare credentials |

```bash
markitai document.pdf -b cloudflare
```

The native converters usually produce better output for formats they support; `-b cloudflare` warns when that is the case.

### Removed per-backend flags

1.0.0 removed six aliases. Passing one is a usage error, and the error names the replacement:

| Removed flag | Use instead |
|--------------|-------------|
| `--playwright` | `-s playwright` |
| `--defuddle` | `-s defuddle` |
| `--static` | `-s static` |
| `--jina` | `-s jina` |
| `--cloudflare` | `-s cloudflare` (plus `-b cloudflare` for file conversion) |
| `--kreuzberg` | nothing; `.rtf` converts natively |

### `--no-remote-fetch`

Never send URLs to remote services (Defuddle, Jina, Cloudflare). Same as `MARKITAI_NO_REMOTE_FETCH=1`.

```bash
markitai https://example.com -o ./output --no-remote-fetch
```

`--quiet` suppresses the consent prompt, so with `fetch.remote_consent=ask` a quiet run skips every remote service and says so on stderr. See [Fetch Policy](/guide/fetch-policy#remote-fallback-and-local-only-urls).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success, including `--dry-run` |
| `1` | A single item failed, or a runtime error |
| `2` | Usage error, or a Batch API wait timed out (or lost contact) and needs `--llm-batch-collect`. This wins over `10`: items that failed alongside it are reported in `--json` and on stderr |
| `10` | Batch finished with partial failures; successful items are kept |

## Setup Commands

### `markitai init`

Guided setup: checks dependencies, detects LLM providers and writes a config file.

```bash
markitai init              # interactive
markitai init --yes        # defaults, no prompts (-y)
markitai init --local      # write ./markitai.json instead of ~/.markitai/config.json
markitai init -o ./markitai.json
```

### `-I, --interactive`

Guided conversion: markitai asks for input, output and options, then runs.

```bash
markitai -I
```

## Configuration Commands

### `markitai config list`

Show the effective configuration, with secrets redacted.

```bash
markitai config list                    # JSON
markitai config list --format table
markitai config list -f yaml            # needs pyyaml
markitai config list --show-secrets
```

::: warning
Use `--show-secrets` only for local inspection. Never paste its output into an issue, chat or CI log.
:::

### `markitai config get <key>`

```bash
markitai config get llm.enabled
```

### `markitai config set <key> <value>`

```bash
markitai config set llm.enabled true
```

### `markitai config path`

Show where the config files are.

### `markitai config edit`

Edit settings through a guided menu.

### `markitai config validate`

Validate the config file, optionally a specific one.

```bash
markitai config validate
markitai config validate ./markitai.json
```

## Cache Commands

### `markitai cache stats`

Entries and size of the LLM cache (`cache.db`) and the URL fetch cache (`fetch_cache.db`).

```bash
markitai cache stats
markitai cache stats --verbose --limit 50   # entries by model (-v; default limit 20)
markitai cache stats --json
```

### `markitai cache clear`

Empties both the LLM cache and the URL fetch cache.

```bash
markitai cache clear
markitai cache clear -y                       # skip confirmation
markitai cache clear --include-spa-domains    # also forget learned SPA domains
```

### `markitai cache spa-domains`

Domains markitai has learned to render in a browser (see [SPA learning](/guide/fetch-policy#spa-learning)).

```bash
markitai cache spa-domains
markitai cache spa-domains --json
markitai cache spa-domains --clear
```

## Diagnostic Commands

### `markitai doctor`

Check the installation, optional capabilities, LLM configuration and provider sign-in. Missing optional tools are warnings, not failures.

```bash
markitai doctor
markitai doctor --fix              # install Chromium when the Playwright package is present
markitai doctor --json
markitai doctor --suggest-extras   # extras to add for the capabilities this machine could use
```

The command exits non-zero when something you configured cannot work: a Playwright strategy that cannot launch, a model whose API key variable is missing, or a local provider that is not signed in. Scripts and CI can rely on that.

`--fix` never adds a Python package. If Playwright itself is missing, it tells you to reinstall with `markitai[browser]`. `--json` and `--fix` cannot be combined.

## Authentication Commands

### `markitai auth`

Sign-in helpers for the subscription providers. Run without a subcommand for a status overview.

```bash
markitai auth
```

### `markitai auth copilot status`

```bash
markitai auth copilot status          # add --json for machine-readable output
```

### `markitai auth copilot login`

```bash
markitai auth copilot login
```

### `markitai auth claude status`

```bash
markitai auth claude status
```

### `markitai auth claude login`

```bash
markitai auth claude login
```

### `markitai auth chatgpt status`

```bash
markitai auth chatgpt status
```

### `markitai auth chatgpt login`

Device-code OAuth sign-in for a ChatGPT subscription.

```bash
markitai auth chatgpt login
```

Gemini has no sign-in flow; use an API key or OpenRouter (see [Model Naming](/guide/configuration#model-naming)).

## Server & Agent Commands

### `markitai serve`

Start the [web workspace](/guide/serve). Needs `markitai[serve]`.

```bash
markitai serve                    # http://127.0.0.1:3600, opens the browser
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host <interface>` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` for other devices; they then need the printed access token |
| `--port <n>` | `3600` | Port |
| `--no-open` | off | Do not open the browser |
| `--no-auth` | off | Disable the access token. Remote clients are then limited to public URL targets and cannot change LLM settings |
| `--allowed-host <hostname>` | — | Extra hostname to accept when you browse by DNS name (repeatable) |

### `markitai mcp`

Start the [MCP server](/guide/mcp) over stdio. Needs `markitai[mcp]`.

```bash
markitai mcp
```

## Other Options

### `--quiet, -q`

Hide progress and informational messages. Errors, stdout Markdown and the one-time remote-fetch notice still appear.

### `-v, --verbose`

Show more detail.

### `--log-level <level>`

Minimum level for the log file (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Takes effect only when `log.dir` is set; `--verbose` and `--quiet` control terminal output.

```bash
markitai ./docs -o out --log-level WARNING
```

### `--dry-run`

Show what would be converted without writing anything.

```bash
markitai ./docs --dry-run
```

### `-c, --config <path>`

Use a specific config file. The file must exist; a missing path is a usage error rather than a silent fall back to defaults.

```bash
markitai document.docx --config ./my-config.json
```

Before a subcommand it applies to that subcommand too, so `config set` writes to that file. `config set` and `config edit` also accept a path that does not exist yet and create the file:

```bash
markitai -c ./my-config.json config set llm.enabled true
```

### `--config-json <json>`

Inline config overrides, merged over the config file. Explicit flags still win. Handy for agents and CI.

```bash
markitai document.docx --config-json '{"llm": {"concurrency": 4}}'
```

Subcommands that read the config (`config list`/`get`/`path`, `cache`, `doctor`) see the overrides too. `config set` and `config edit` refuse them, because inline JSON has no file to save to.

### `-V, --version`

```bash
markitai -V
```

### `-h, --help`

```bash
markitai -h
markitai config -h
```

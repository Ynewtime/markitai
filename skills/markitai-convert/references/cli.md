# markitai CLI reference (condensed)

Authoritative long-form docs: <https://markitai.dev/guide/cli>. This file keeps the facts an agent needs while composing commands.

## Conversion flags

| Flag | Effect |
|---|---|
| `--llm` | LLM cleanup + frontmatter; writes only `.llm.md` unless `--keep-base` |
| `--preset rich\|standard\|minimal` | Feature bundles: rich = LLM+alt+desc+screenshot; standard = LLM+alt+desc; minimal = plain |
| `--alt` / `--desc` | Image alt text / detailed descriptions; both require `--llm`, otherwise skipped with a warning |
| `--ocr` | OCR for scanned documents (RapidOCR) |
| `--screenshot` | PDF/PPTX: render pages/slides as JPEG; URLs: full-page screenshot (auto-upgrades fetch to Playwright) |
| `--screenshot-only` | URLs: screenshots without extraction (no `.md`); with `--llm`: LLM extracts content from the screenshots into `.llm.md`. File inputs (PDF/PPTX) still write normal `.md` |
| `--pure` | Body-only output, no frontmatter; silently overrides `--alt`/`--desc`/`--screenshot`; independent of `--llm` |
| `--keep-base` | Also write base `.md` in `--llm` mode |
| `--no-compress` | Keep original image bytes |
| `--no-llm` `--no-alt` `--no-desc` `--no-ocr` `--no-screenshot` | Explicitly disable a feature a preset would enable, e.g. `--preset rich --no-desc` |

## Output and batch

| Flag | Effect |
|---|---|
| `-o, --output <path>` | Directory, or exact file target for single input. Without it, single input prints to stdout; directory/`.urls` input requires `-o` |
| `--resume` | Batch only: skip completed, retry failed/interrupted, pick up new files; prints `Resuming batch: N completed, M remaining` |
| `-g, --glob <pat>` | Restrict directory discovery; repeatable; `!` prefix excludes (`-g '!drafts/**'`, single-quote in zsh) |
| `--max-depth <n>` | Directory scan depth (default 5; 0 = no recursion) |
| `-j, --batch-concurrency <n>` | Concurrent file tasks (default 10) |
| `--url-concurrency <n>` | Concurrent URL fetches (default 5), separate so slow URLs don't block files |
| `--llm-concurrency <n>` | Concurrent LLM requests (default 10) |

## `.urls` list files

A `.urls` input file is processed as a URL batch; directory batches also auto-discover `.urls` files in the scan tree and merge them. Three formats:

```
# plain text: URL per line, optional output name after whitespace
https://example.com/page1
https://example.com/page2 custom_name
```

```json
["https://example1.com", "https://example2.com"]
```

```json
[{"url": "https://example1.com"}, {"url": "https://example2.com", "output_name": "custom"}]
```

Partial success exits with status 10 (successful URLs are still written).

## URL strategy and file backend

| Flag | Values |
|---|---|
| `-s, --strategy` | `auto` (default) `static` `playwright` `defuddle` `jina` `cloudflare` — URL fetching only |
| `-b, --backend` | `native` (default) `kreuzberg` `cloudflare` — file conversion only; `-s`/`-b` combine freely, but `-b kreuzberg` and `-s cloudflare` are mutually exclusive |

Credentials: `-s jina` needs `JINA_API_KEY`; `-s cloudflare` (and `-b cloudflare`) need `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`; `-b kreuzberg` needs the `kreuzberg` extra. Deprecated aliases (`--playwright`, `--defuddle`, `--static`, `--jina`, `--cloudflare`, `--kreuzberg`) still resolve to the flags above with a stderr notice.

Strategy ordering, per-domain tuning, SPA cache, and privacy rules: [url-fetching.md](url-fetching.md).

## Cache

| Command | Effect |
|---|---|
| `--no-cache` | Skip LLM result cache for this run |
| `--no-cache-for "*.pdf,reports/**"` | Skip cache for matching inputs only |
| `markitai cache stats [-v] [--json]` | Cache statistics |
| `markitai cache clear [-y] [--include-spa-domains]` | Clear cache (optionally learned SPA domains too) |
| `markitai cache spa-domains [--json] [--clear]` | Inspect/clear domains learned to need browser rendering |

## Config on the command line

| Flag | Effect |
|---|---|
| `-c, --config <path>` | Explicit config file |
| `--config-json '<json>'` | Inline deep-merge overrides (agent/CI friendly); explicit CLI flags still win |
| `markitai config list\|get\|set\|path\|edit\|validate` | Inspect and edit persisted config; `config list` redacts secrets unless `--show-secrets` |

Config resolution order: CLI args > env vars > config file (`--config` > `MARKITAI_CONFIG` > `./markitai.json` > `~/.markitai/config.json`) > defaults.

## Misc

| Flag | Effect |
|---|---|
| `-q, --quiet` | Suppress progress; stdout Markdown payload of a single conversion is preserved |
| `-v, --verbose` | Verbose logging |
| `--dry-run` | Preview without writing |
| `-I, --interactive` | Guided conversion setup |
| `-V, --version` / `-h, --help` | Version / help |

## Environment variables that change conversion behavior

| Variable | Effect |
|---|---|
| `MARKITAI_NO_REMOTE_FETCH=1` | Hard-disable remote extraction, even explicit remote `-s` flags |
| `MARKITAI_STATIC_HTTP=curl_cffi` | TLS-impersonating static fetch (needs `extra-fetch` extra; silently falls back to httpx if absent) |
| `MARKITAI_PURE=1` | Same as `--pure` |
| `MARKITAI_LANG=en\|zh` | CLI language |
| `MODEL` | Single-model override when no `model_list` is configured |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Failure — including a single-image input with neither `--ocr` nor `--llm` |
| 10 | `.urls` batch partial success |

# Single URL Stdout Fix & External Image Inline Display

## Problem

Two issues with single URL stdout mode:

1. **Single URL mode uses config `output.dir` as fallback, bypassing stdout.** When `output.dir` is set in `~/.markitai/config.json`, `markitai URL` silently writes to a file instead of printing to stdout. Single file mode already correctly ignores config fallback (`main.py:800-808`), but single URL mode passes `effective_output` which includes the config fallback. This is inconsistent.

2. **External image URLs are not rendered inline.** Stdout mode only handles locally extracted `.markitai/assets/` images. External URLs (e.g., `![](https://example.com/img.jpg)`) from web pages are preserved as-is, missing the opportunity to render them inline in supported terminals.

## Design

### Feature A: Single URL Mode Stdout Fix

**Root cause:** `main.py:764` passes `effective_output` (which reads `config.output.dir` as fallback) to `process_url()`. Single file mode (`main.py:805`) correctly passes `output` (CLI arg only).

**Fix:** Single URL mode should pass `output` (the raw CLI argument) instead of `effective_output`, matching single file mode behavior. When the user doesn't pass `-o`, `output` is `None`, and `process_url` enters stdout mode.

This is a one-line change in `cli/main.py:766`:

```python
# Before:
await process_url(input_path_str, effective_output, ...)

# After:
await process_url(input_path_str, output, ...)
```

No changes to `--quiet` semantics (stays as "complete silence"), no changes to `OutputManager`, no changes to `on_conflict` handling. These all remain as-is.

### Feature B: External Image Terminal Inline Display

**Scope:** URL mode only. Local files (PDF/DOCX) extract images as local assets — no external URLs to handle. `download_url_images()` requires a `base_url` for resolving relative image paths, which is naturally available in URL mode but not in file mode.

**Trigger:** Config `image.stdout_fetch_external: bool = false` (default disabled) AND terminal supports image protocol (Kitty/iTerm2). Both conditions must be true.

**Data flow:**

```
markdown with external image URLs (![](https://example.com/img.jpg))
  → stdout_fetch_external enabled + terminal protocol detected
  → download_url_images(markdown, temp_dir, base_url=url, config)
      (reuses existing httpx-based downloader with concurrency, timeout, proxy)
  → external URLs replaced with local .markitai/assets/ paths in markdown
  → resolve_asset_references() handles them via existing three-tier cascade
  → terminal inline display
```

When `stdout_fetch_external` is disabled or terminal doesn't support image protocol, external URLs are preserved as-is (current behavior).

Download failures are non-fatal: the original URL is kept, a warning is logged.

**Configuration:**

```python
class ImageConfig(BaseModel):
    # ... existing fields ...
    stdout_fetch_external: bool = False  # NEW
```

### Files Changed

| File | Change |
|------|--------|
| `cli/main.py:766` | Pass `output` instead of `effective_output` for single URL mode. |
| `cli/processors/url.py` | In stdout branch, add `download_url_images` call before `resolve_asset_references` when `stdout_fetch_external` is enabled and protocol is detected. |
| `config.py` | Add `stdout_fetch_external: bool = False` to `ImageConfig`. |
| `config.schema.json` | Add `stdout_fetch_external` to `ImageConfig` definition. |
| `tests/unit/test_schema_sync.py` | Existing tests verify new config field sync (no manual changes needed). |

### Out of Scope

- Single file mode changes (already correct — ignores config fallback)
- Batch mode changes (already writes to files)
- `--quiet` semantics changes (stays as complete silence)
- `OutputManager` changes (stays bound to `is_stdout_mode`)
- External image download in file mode (no `base_url`, images already local)
- External image download caching across runs (future: could use AssetStore)
- "Save + print simultaneously" UX enhancement

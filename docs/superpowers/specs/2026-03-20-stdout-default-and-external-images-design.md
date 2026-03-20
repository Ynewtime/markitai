# Stdout Default Behavior & External Image Inline Display

## Problem

Two related issues with the current stdout mode:

1. **Single file/URL mode doesn't always print to terminal.** When `output.dir` is set in config (e.g., `~/.markitai/config.json`), single file/URL mode silently writes to a file instead of printing to stdout. Users expect `markitai file.pdf` to always show output in the terminal.

2. **External image URLs are not rendered inline.** Stdout mode only handles locally extracted `.markitai/assets/` images. External URLs (e.g., `![](https://example.com/img.jpg)`) from web pages are preserved as-is, missing the opportunity to render them inline in supported terminals.

## Design

### Feature A: Single File Mode Always Prints to Terminal

**Current behavior:**

```
is_stdout_mode = is_single_mode and output is None
```

When `output.dir` is configured (even in global config), `get_effective_output()` returns a path, `output` is not None, and the CLI writes to a file without printing.

**New behavior:**

| Context | Has output target | Action |
|---------|------------------|--------|
| TTY | Yes (from `-o` or config) | Save file + print full content to terminal. Print `Saved to <path>` hint to stderr in gray before content. |
| TTY | No | Print full content to terminal (current stdout mode). |
| Non-TTY (pipe) | Yes, from explicit `-o` | Save file + pipe content to stdout. |
| Non-TTY (pipe) | Yes, from config only | Pipe content to stdout only. Do NOT save file (backward compat). |
| Non-TTY (pipe) | No | Pipe content to stdout (current behavior). |

**Key design decisions:**

- The save path hint goes to **stderr** (not stdout), so it doesn't pollute piped output or markdown content.
- Distinguishing "explicit `-o`" from "config `output.dir`" is needed for the non-TTY case. The CLI already tracks whether `output` was passed as a CLI argument vs resolved from config.
- `--quiet` suppresses the stderr hint but does not suppress the content output. It controls log/progress noise, not results.
- Terminal image rendering (Kitty/iTerm2) applies to the printed content. The saved file keeps relative `.markitai/assets/` paths.

**Implementation notes:**

The core change is in `cli/main.py` — `is_stdout_mode` becomes `always_print = is_single_mode` (always true for single file/URL). A new flag `also_save = effective_output is not None and (explicit_output or is_tty)` controls whether to also write a file.

In `cli/processors/file.py` and `cli/processors/url.py`, the stdout output branch runs unconditionally in single mode. If `also_save` is true, the file is written first (as today), then the content is printed with terminal image rendering applied.

### Feature B: External Image Terminal Inline Display

**Trigger:** Config `image.stdout_fetch_external: bool = false` (default disabled) AND terminal supports image protocol (Kitty/iTerm2). Both conditions must be true.

**Data flow:**

```
markdown with external image URLs (![](https://example.com/img.jpg))
  → stdout_fetch_external enabled + terminal protocol detected
  → download_url_images(markdown, temp_dir, base_url, config)
      (reuses existing httpx-based downloader with concurrency, timeout, proxy)
  → external URLs replaced with local .markitai/assets/ paths in markdown
  → resolve_asset_references() handles them via existing three-tier cascade
  → terminal inline display
```

When `stdout_fetch_external` is disabled or terminal doesn't support image protocol, external URLs are preserved as-is (current behavior, no change).

Download failures are non-fatal: the original URL is kept, a warning is logged, and rendering continues.

**Configuration:**

```python
class ImageConfig(BaseModel):
    # ... existing fields ...
    stdout_persist: bool = False
    stdout_persist_dir: str = "~/.markitai/assets"
    stdout_fetch_external: bool = False  # NEW
```

```json
{
  "image": {
    "stdout_fetch_external": true
  }
}
```

**Scope:** Only applies to stdout mode (single file/URL). File mode and batch mode are not affected — they already have `download_url_images` integration in their own pipelines.

### Files Changed

| File | Change |
|------|--------|
| `cli/main.py` | Rework `is_stdout_mode` → `always_print` + `also_save` logic for single mode. |
| `cli/processors/file.py` | Stdout branch: unconditional print in single mode. Add external image download before `resolve_asset_references` when configured. |
| `cli/processors/url.py` | Same as file.py. |
| `config.py` | Add `stdout_fetch_external: bool = False` to `ImageConfig`. |
| `config.schema.json` | Add `stdout_fetch_external` to `ImageConfig` definition. |
| `tests/unit/cli/test_file_processor.py` | Tests for always-print behavior and external image download integration. |
| `tests/unit/test_schema_sync.py` | Existing tests verify new config field sync (no manual changes needed). |

### Out of Scope

- Batch mode changes (already writes to files)
- External image download caching across runs (future: could use AssetStore)
- `--quiet` behavior changes beyond suppressing the save hint
- External image download in file mode (already handled separately)

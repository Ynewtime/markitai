# Stdout Terminal Image Display

## Problem

When markitai outputs to stdout (no `-o` flag), embedded images are replaced with `[image: filename]` text placeholders because the temporary directory is deleted after printing. This loses image information and produces non-standard markdown.

**Breaking change:** The placeholder format changes from `[image: filename]` (plain text) to `![image: filename]()` (valid markdown image syntax with empty src). Users who grep or parse the old `[image:` pattern will need to update. This is a deliberate improvement — the new format is valid markdown and consistent with the image syntax used throughout the output. The change ships in the same minor version bump as this feature.

## Design

### Priority Cascade

Stdout mode resolves image references in three tiers, evaluated top-down:

1. **Terminal protocol inline display** — if stdout is a TTY (`sys.stdout.isatty()`) **and** the terminal supports Kitty or iTerm2 graphics protocol, render images inline at their position in the markdown flow. When stdout is not a TTY (pipe, redirect, CI), protocol detection is skipped entirely to prevent escape sequences from leaking into files or downstream processes.
2. **Persistent asset store** — if `image.stdout_persist` is enabled, save images to a content-addressed store and output absolute-path markdown references.
3. **Placeholder fallback** — output `![image: filename]()` (valid markdown image syntax with empty src).

### Terminal Protocol Detection — `utils/terminal_image.py`

New module. Detects terminal image capabilities and renders inline images.

**Detection logic (non-exhaustive, extensible):**

| Protocol | Detection signals |
|----------|------------------|
| Kitty | `$KITTY_PID` set, or `$TERM` contains `xterm-kitty` |
| iTerm2 | `$TERM_PROGRAM` is `iTerm.app`, `WezTerm`, `mintty`, `Hyper`, or `Tabby` |

Terminals that support both (e.g., WezTerm) prefer Kitty protocol.

**Public API:**

```python
class Protocol(Enum):
    KITTY = "kitty"
    ITERM2 = "iterm2"

def detect_protocol() -> Protocol | None:
    """Detect the best supported terminal image protocol.

    Returns None if:
    - sys.stdout.isatty() is False (pipe, redirect, CI)
    - No supported terminal protocol is detected
    """

def render_inline_image(image_path: Path, protocol: Protocol) -> str:
    """Read image file, return the terminal escape sequence for inline display."""
```

**Protocol encoding:**

- Kitty: Direct data transmission (`a=T,t=d`). Image data is base64-encoded and sent in chunks — intermediate chunks use `m=1`, the final chunk uses `m=0` to terminate the transfer. See [Kitty graphics protocol spec](https://sw.kovidgoyal.net/kitty/graphics-protocol/) for details.
- iTerm2: `\033]1337;File=inline=1;size=<bytes>:<base64>\a`

Both protocols read the image file, base64-encode it, and wrap it in the appropriate escape sequence. The escape sequence is returned as a string to be printed to stdout in place of the markdown image reference.

**stdout interaction with Rich:** The current stdout path uses `console.print(final_content, markup=False, highlight=False)` from Rich. When terminal protocol escape sequences are present, Rich may interfere with or escape them. The implementation must bypass Rich for the final output when protocol inline images are used, writing directly to `sys.stdout.buffer` or `sys.stdout` instead.

### Asset Store — `utils/asset_store.py`

New module. Content-addressed image storage with symlink-based indexing.

**Storage structure:**

```
<persist_dir>/
  blobs/
    a1b2c3d4e5f6g7h8.jpg       # actual file, SHA-256 first 16 hex chars
  refs/
    sample.pdf/
      sample.pdf-0001-10.jpg → ../../blobs/a1b2c3d4e5f6g7h8.jpg
    report.docx/
      report.docx-0003-02.jpg → ../../blobs/a1b2c3d4e5f6g7h8.jpg  # dedup
```

- `blobs/` stores the actual image data, keyed by content hash. Identical images are stored once.
- `refs/` provides human-navigable symlinks grouped by source file name.

**Hash collision mitigation:** SHA-256 truncated to 16 hex chars (64 bits) has negligible collision probability for typical asset stores. As a safeguard, when a blob file already exists, compare file sizes before assuming dedup. If sizes differ (collision), append an incrementing suffix (e.g., `a1b2c3d4e5f6g7h8_1.jpg`).

**Public API:**

```python
class AssetStore:
    def __init__(self, persist_dir: Path):
        """Initialize store at the given directory.

        Calls Path.expanduser() on persist_dir to resolve ~.
        Creates blobs/ and refs/ subdirectories if needed.
        """

    def save(self, image_path: Path, source_name: str) -> Path:
        """Persist an image. Returns the absolute path to the ref symlink.

        - Computes SHA-256 of the file content (first 16 hex chars for filename).
        - Writes to blobs/ if not already present (dedup with size check).
        - Creates symlink in refs/<source_name>/ (overwrites if exists, for re-processing).
        - Returns the absolute symlink path for use in markdown output.
        - The returned path is safe for markdown embedding: use Path.as_uri()
          or percent-encode special characters (spaces, parentheses, etc.) to
          produce valid markdown image references like ![alt](file:///path/to/image.jpg).
        """
```

**Error handling:** On any `AssetStore` error (permission denied, disk full, symlink unsupported), log a warning and fall through to the placeholder tier. The asset store must never cause the overall conversion to fail.

### Resolving Asset References — `cli/processors/file.py`

Rename `strip_asset_references()` to `resolve_asset_references()` with expanded signature:

```python
def resolve_asset_references(
    markdown: str,
    temp_dir: Path,
    protocol: Protocol | None = None,
    asset_store: AssetStore | None = None,
) -> str:
    """Resolve image references in stdout-mode markdown.

    For each match of the asset reference pattern:
    1. If protocol is set: replace with terminal inline image escape sequence.
    2. Else if asset_store is set: persist image, replace with absolute-path reference.
    3. Else: replace with placeholder ![image: filename]().
    """
```

The existing regex pattern `_ASSET_REF_PATTERN` must be updated to handle both forward slash and backslash path separators for Windows compatibility. The current pattern only matches `/`, but other parts of the codebase (e.g., `image.py:508`) already use `[/\\]`. Updated pattern:

```python
_ASSET_REF_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(\.markitai[/\\](?:assets|screenshots)[/\\]([^)]+)\)"
)
```

The replacement callback inspects the arguments to decide which tier to apply. The image file is resolved via `temp_dir / ".markitai" / ("assets" or "screenshots") / filename`.

**Important:** `resolve_asset_references()` must be called **before** the temp directory is cleaned up (`shutil.rmtree` in the `finally` block). For the `AssetStore.save()` path this is load-bearing — the image file must be copied to the blob store while it still exists in the temp directory.

### Configuration — `config.py`

Two new fields on `ImageConfig`:

```python
class ImageConfig(BaseModel):
    # ... existing fields ...
    stdout_persist: bool = False
    stdout_persist_dir: str = "~/.markitai/assets"
```

- `stdout_persist` — enable persistent image storage in stdout mode. Default `false`.
- `stdout_persist_dir` — base directory for the asset store. Default `~/.markitai/assets`. Tilde expansion is performed by `AssetStore.__init__()`.

Configurable via `~/.markitai/config.json` or `./markitai.json`:

```json
{
  "image": {
    "stdout_persist": true,
    "stdout_persist_dir": "~/.markitai/assets"
  }
}
```

No new CLI flags. This is a configuration-file-only setting to keep the CLI surface minimal.

### Data Flow

```
Input file
  → converter (PDF/DOCX/etc.)
  → markdown + images in temp_dir/.markitai/assets/
  → detect_protocol()
  ┌─ protocol found ──→ render_inline_image() for each reference
  ├─ no protocol + stdout_persist ──→ AssetStore.save() → absolute path reference
  └─ neither ──→ ![image: filename]() placeholder
  → print resolved markdown to stdout  (bypass Rich if protocol images present)
  → delete temp_dir
```

### Files Changed

| File | Change |
|------|--------|
| `utils/terminal_image.py` | **New.** Protocol detection and inline rendering. |
| `utils/asset_store.py` | **New.** Content-addressed storage with symlink refs. |
| `cli/processors/file.py` | Rename `strip_asset_references` → `resolve_asset_references`. Update stdout output section to pass protocol/store. |
| `cli/processors/url.py` | Update import and call site from `strip_asset_references` → `resolve_asset_references`. |
| `config.py` | Add `stdout_persist` and `stdout_persist_dir` to `ImageConfig`. |
| `config.schema.json` | Add `stdout_persist` and `stdout_persist_dir` to `ImageConfig` definition. Required to pass `test_schema_sync.py` field-sync assertions. |
| `tests/unit/cli/test_file_processor.py` | Update existing tests for new function signature, placeholder format, and backslash path patterns. |
| `tests/unit/utils/test_terminal_image.py` | **New.** Protocol detection tests (mocked env vars). |
| `tests/unit/utils/test_asset_store.py` | **New.** Store save/dedup/symlink tests. |

### Out of Scope

- Sixel protocol support (complex encoding, low adoption benefit)
- `AssetStore.gc()` for cleaning orphaned blobs (future enhancement)
- Batch mode image handling (already writes to output directory)
- New CLI flags for these settings

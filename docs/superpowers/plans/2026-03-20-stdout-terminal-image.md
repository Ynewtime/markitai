# Stdout Terminal Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable terminal inline image display and persistent image storage for markitai's stdout mode.

**Architecture:** Three-tier priority cascade: (1) Kitty/iTerm2 terminal protocol inline display when stdout is a TTY, (2) content-addressed asset store with symlink refs when persistence is configured, (3) valid-markdown placeholder fallback. Two new utility modules, config extension, and integration into the existing stdout output path.

**Tech Stack:** Python 3.10+, base64 (stdlib), hashlib (stdlib), pathlib, Pydantic config, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-stdout-terminal-image-design.md`

---

### Task 1: Terminal Protocol Detection — `utils/terminal_image.py`

**Files:**
- Create: `packages/markitai/src/markitai/utils/terminal_image.py`
- Create: `packages/markitai/tests/unit/utils/__init__.py`
- Create: `packages/markitai/tests/unit/utils/test_terminal_image.py`

- [ ] **Step 1: Write failing tests for protocol detection**

Create `packages/markitai/tests/unit/utils/__init__.py` (empty file).

Create `packages/markitai/tests/unit/utils/test_terminal_image.py`:

```python
"""Tests for terminal image protocol detection and rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.utils.terminal_image import Protocol, detect_protocol, render_inline_image


class TestDetectProtocol:
    """detect_protocol() returns the best available terminal image protocol."""

    def test_returns_none_when_not_tty(self) -> None:
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert detect_protocol() is None

    def test_returns_kitty_when_kitty_pid_set(self) -> None:
        env = {"KITTY_PID": "12345", "TERM": "", "TERM_PROGRAM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY

    def test_returns_kitty_when_term_is_xterm_kitty(self) -> None:
        env = {"TERM": "xterm-kitty", "TERM_PROGRAM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY

    def test_returns_iterm2_when_term_program_is_iterm(self) -> None:
        env = {"TERM_PROGRAM": "iTerm.app", "TERM": "xterm-256color"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.ITERM2

    def test_returns_iterm2_for_wezterm(self) -> None:
        env = {"TERM_PROGRAM": "WezTerm", "TERM": "xterm-256color"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.ITERM2

    def test_returns_none_for_unknown_terminal(self) -> None:
        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "unknown"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() is None

    def test_kitty_takes_priority_over_iterm2(self) -> None:
        """When both Kitty and iTerm2 signals are present, prefer Kitty."""
        env = {"KITTY_PID": "12345", "TERM_PROGRAM": "WezTerm", "TERM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY


class TestRenderInlineImage:
    """render_inline_image() produces terminal escape sequences."""

    def test_kitty_output_starts_with_apc(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = render_inline_image(img, Protocol.KITTY)
        assert result.startswith("\033_G")

    def test_kitty_output_ends_with_st(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = render_inline_image(img, Protocol.KITTY)
        assert result.endswith("\033\\")

    def test_iterm2_output_contains_protocol_header(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

        result = render_inline_image(img, Protocol.ITERM2)
        assert "\033]1337;File=" in result
        assert "inline=1" in result

    def test_kitty_multi_chunk_for_large_image(self, tmp_path: Path) -> None:
        """Images larger than KITTY_CHUNK_SIZE should produce multiple chunks."""
        img = tmp_path / "large.png"
        # Create image larger than 4096 bytes of base64 (~3072 raw bytes)
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4000)

        result = render_inline_image(img, Protocol.KITTY)
        # Should have intermediate chunks with m=1 and final with m=0
        assert "m=1;" in result
        assert "m=0;" in result

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError):
            render_inline_image(missing, Protocol.KITTY)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/utils/test_terminal_image.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'markitai.utils.terminal_image'`

- [ ] **Step 3: Implement terminal_image.py**

Create `packages/markitai/src/markitai/utils/terminal_image.py`:

```python
"""Terminal image protocol detection and inline rendering.

Supports Kitty graphics protocol and iTerm2 inline images protocol.
Detection requires stdout to be a TTY — pipes, redirects, and CI
environments always return None to prevent escape sequence leakage.
"""

from __future__ import annotations

import base64
import os
import sys
from enum import Enum
from pathlib import Path

KITTY_CHUNK_SIZE = 4096  # bytes of base64 data per chunk


class Protocol(Enum):
    """Supported terminal image protocols."""

    KITTY = "kitty"
    ITERM2 = "iterm2"


# iTerm2-compatible TERM_PROGRAM values (non-exhaustive, extensible)
_ITERM2_TERM_PROGRAMS = frozenset(
    {"iTerm.app", "WezTerm", "mintty", "Hyper", "Tabby"}
)


def detect_protocol() -> Protocol | None:
    """Detect the best supported terminal image protocol.

    Returns None if stdout is not a TTY or no supported protocol is detected.
    When both Kitty and iTerm2 signals are present, Kitty takes priority.
    """
    if not sys.stdout.isatty():
        return None

    # Kitty detection (highest priority)
    if os.environ.get("KITTY_PID"):
        return Protocol.KITTY
    term = os.environ.get("TERM", "")
    if "xterm-kitty" in term:
        return Protocol.KITTY

    # iTerm2 detection
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in _ITERM2_TERM_PROGRAMS:
        return Protocol.ITERM2

    return None


def render_inline_image(image_path: Path, protocol: Protocol) -> str:
    """Read an image file and return the terminal escape sequence for inline display.

    Args:
        image_path: Path to the image file. Must exist.
        protocol: Which terminal protocol to use.

    Returns:
        String containing the terminal escape sequence.

    Raises:
        FileNotFoundError: If image_path does not exist.
    """
    data = image_path.read_bytes()

    if protocol == Protocol.KITTY:
        return _render_kitty(data)
    elif protocol == Protocol.ITERM2:
        return _render_iterm2(data)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


def _render_kitty(data: bytes) -> str:
    """Render image using Kitty graphics protocol (direct data transmission)."""
    encoded = base64.standard_b64encode(data).decode("ascii")
    chunks: list[str] = []

    for i in range(0, len(encoded), KITTY_CHUNK_SIZE):
        chunk = encoded[i : i + KITTY_CHUNK_SIZE]
        is_last = i + KITTY_CHUNK_SIZE >= len(encoded)
        m = 0 if is_last else 1

        if i == 0:
            # First chunk includes transmission parameters
            chunks.append(f"\033_Ga=T,t=d,m={m};{chunk}\033\\")
        else:
            chunks.append(f"\033_Gm={m};{chunk}\033\\")

    return "".join(chunks)


def _render_iterm2(data: bytes) -> str:
    """Render image using iTerm2 inline images protocol."""
    encoded = base64.standard_b64encode(data).decode("ascii")
    size = len(data)
    return f"\033]1337;File=inline=1;size={size}:{encoded}\a"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/utils/test_terminal_image.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/utils/terminal_image.py packages/markitai/tests/unit/utils/
git commit -m "feat: add terminal image protocol detection (Kitty/iTerm2)"
```

---

### Task 2: Asset Store — `utils/asset_store.py`

**Files:**
- Create: `packages/markitai/src/markitai/utils/asset_store.py`
- Create: `packages/markitai/tests/unit/utils/test_asset_store.py`

- [ ] **Step 1: Write failing tests for AssetStore**

Create `packages/markitai/tests/unit/utils/test_asset_store.py`:

```python
"""Tests for content-addressed asset store with symlink refs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from markitai.utils.asset_store import AssetStore


class TestAssetStoreInit:
    """AssetStore.__init__ creates directory structure."""

    def test_creates_blobs_and_refs_dirs(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "assets"
        AssetStore(store_dir)
        assert (store_dir / "blobs").is_dir()
        assert (store_dir / "refs").is_dir()

    def test_expands_tilde(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        store = AssetStore(Path("~/.markitai/assets"))
        assert store.persist_dir == tmp_path / ".markitai" / "assets"

    def test_idempotent_on_existing_dirs(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "assets"
        AssetStore(store_dir)
        AssetStore(store_dir)  # should not raise


class TestAssetStoreSave:
    """AssetStore.save() persists images with content-hash dedup."""

    def _make_image(self, tmp_path: Path, name: str, content: bytes) -> Path:
        img = tmp_path / name
        img.write_bytes(content)
        return img

    def test_saves_blob_and_creates_symlink(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "test.jpg", b"image-data-123")

        ref_path = store.save(img, "sample.pdf")

        assert ref_path.is_symlink()
        assert ref_path.resolve().exists()
        assert ref_path.read_bytes() == b"image-data-123"
        assert "refs" in str(ref_path)
        assert "sample.pdf" in str(ref_path)

    def test_dedup_same_content(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img1 = self._make_image(tmp_path, "a.jpg", b"same-content")
        img2 = self._make_image(tmp_path, "b.jpg", b"same-content")

        ref1 = store.save(img1, "doc1.pdf")
        ref2 = store.save(img2, "doc2.pdf")

        # Both symlinks should point to the same blob
        assert ref1.resolve() == ref2.resolve()
        # Only one blob file should exist
        blobs = list((tmp_path / "store" / "blobs").iterdir())
        assert len(blobs) == 1

    def test_different_content_different_blobs(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img1 = self._make_image(tmp_path, "a.jpg", b"content-A")
        img2 = self._make_image(tmp_path, "b.jpg", b"content-B")

        ref1 = store.save(img1, "doc.pdf")
        ref2 = store.save(img2, "doc.pdf")

        assert ref1.resolve() != ref2.resolve()

    def test_overwrites_existing_symlink(self, tmp_path: Path) -> None:
        """Re-processing same file should update the symlink."""
        store = AssetStore(tmp_path / "store")
        img1 = self._make_image(tmp_path, "test.jpg", b"version-1")
        img2 = self._make_image(tmp_path, "test.jpg", b"version-2")

        store.save(img1, "doc.pdf")
        img2 = self._make_image(tmp_path, "test.jpg", b"version-2")
        ref = store.save(img2, "doc.pdf")

        assert ref.read_bytes() == b"version-2"

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "test.jpg", b"data")

        ref_path = store.save(img, "source.pdf")
        assert ref_path.is_absolute()

    def test_ref_path_under_refs_not_blobs(self, tmp_path: Path) -> None:
        """Returned path must be under refs/, not blobs/."""
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "chart.png", b"chart-data")

        ref_path = store.save(img, "report.docx")
        assert "refs" in str(ref_path)
        assert "blobs" not in str(ref_path)
        assert "report.docx" in str(ref_path)
        assert ref_path.name == "chart.png"


class TestAssetStoreHashCollision:
    """Hash collision mitigation: appends suffix when sizes differ."""

    def test_hash_collision_appends_suffix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If blob exists but has different size, save with suffix."""
        from markitai.utils import asset_store as mod

        # Force all hashes to be the same
        monkeypatch.setattr(mod, "_content_hash", lambda _data: "aaaaaaaaaaaaaaaa")

        store = AssetStore(tmp_path / "store")

        img1 = tmp_path / "img1.jpg"
        img1.write_bytes(b"short")
        img2 = tmp_path / "img2.jpg"
        img2.write_bytes(b"longer-content-here")

        ref1 = store.save(img1, "doc.pdf")
        ref2 = store.save(img2, "doc.pdf")

        # Both should exist and have different content
        assert ref1.read_bytes() == b"short"
        assert ref2.read_bytes() == b"longer-content-here"
        assert ref1.resolve() != ref2.resolve()


class TestAssetStoreErrorHandling:
    """AssetStore errors should not propagate — fall through to placeholder."""

    def test_save_nonexistent_image_raises(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        missing = tmp_path / "missing.jpg"
        with pytest.raises(FileNotFoundError):
            store.save(missing, "doc.pdf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/utils/test_asset_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'markitai.utils.asset_store'`

- [ ] **Step 3: Implement asset_store.py**

Create `packages/markitai/src/markitai/utils/asset_store.py`:

```python
"""Content-addressed image storage with symlink-based indexing.

Stores image blobs keyed by content hash (SHA-256, first 16 hex chars).
Provides human-navigable symlink refs grouped by source file name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger


def _content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data, return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


class AssetStore:
    """Content-addressed image store with symlink refs.

    Storage layout::

        <persist_dir>/
          blobs/<hash>.<ext>          # actual image data, deduped
          refs/<source>/<filename>    # symlinks to blobs
    """

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir).expanduser().resolve()
        self._blobs_dir = self.persist_dir / "blobs"
        self._refs_dir = self.persist_dir / "refs"
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._refs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image_path: Path, source_name: str) -> Path:
        """Persist an image and return the absolute ref symlink path.

        Args:
            image_path: Path to the image file. Must exist.
            source_name: Source document name (e.g., "sample.pdf").

        Returns:
            Absolute path to the symlink in refs/<source_name>/.

        Raises:
            FileNotFoundError: If image_path does not exist.
        """
        data = image_path.read_bytes()
        ext = image_path.suffix  # e.g., ".jpg"

        # Compute content hash and resolve blob path
        h = _content_hash(data)
        blob_name = f"{h}{ext}"
        blob_path = self._blobs_dir / blob_name

        # Dedup: if blob exists, check for hash collision (different sizes)
        if blob_path.exists():
            if blob_path.stat().st_size != len(data):
                # Hash collision — append incrementing suffix
                for i in range(1, 1000):
                    blob_name = f"{h}_{i}{ext}"
                    blob_path = self._blobs_dir / blob_name
                    if not blob_path.exists() or blob_path.stat().st_size == len(data):
                        break
        # Write blob if not already present
        if not blob_path.exists():
            blob_path.write_bytes(data)

        # Create ref symlink
        ref_dir = self._refs_dir / source_name
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = ref_dir / image_path.name

        # Compute relative symlink target
        rel_target = Path(
            "..",
            "..",
            "blobs",
            blob_name,
        )

        # Overwrite if exists (re-processing same document)
        if ref_path.is_symlink() or ref_path.exists():
            ref_path.unlink()
        ref_path.symlink_to(rel_target)

        # Return absolute path of the symlink itself (not following it)
        return ref_dir.resolve() / image_path.name

    def ref_path_to_markdown_uri(self, ref_path: Path) -> str:
        """Convert a ref path to a safe markdown-embeddable URI.

        Handles spaces and special characters via percent-encoding.
        """
        return ref_path.as_uri()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/utils/test_asset_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/utils/asset_store.py packages/markitai/tests/unit/utils/test_asset_store.py
git commit -m "feat: add content-addressed asset store with symlink refs"
```

---

### Task 3: Config — Add `stdout_persist` fields + schema sync

**Files:**
- Modify: `packages/markitai/src/markitai/config.py:267-278`
- Modify: `packages/markitai/src/markitai/config.schema.json:410-461`
- Verify: `packages/markitai/tests/unit/test_schema_sync.py` (existing tests should pass)

- [ ] **Step 1: Run existing schema sync tests to confirm green baseline**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/test_schema_sync.py -v`
Expected: All PASS

- [ ] **Step 2: Add fields to ImageConfig in config.py**

In `packages/markitai/src/markitai/config.py`, add two fields to `ImageConfig` after `filter`:

```python
class ImageConfig(BaseModel):
    """Image processing configuration."""

    alt_enabled: bool = False
    desc_enabled: bool = False
    compress: bool = True
    quality: int = Field(default=DEFAULT_IMAGE_QUALITY, ge=1, le=100)
    format: Literal["jpeg", "png", "webp"] = DEFAULT_IMAGE_FORMAT
    max_width: int = DEFAULT_IMAGE_MAX_WIDTH
    max_height: int = DEFAULT_IMAGE_MAX_HEIGHT
    filter: ImageFilterConfig = Field(default_factory=ImageFilterConfig)
    stdout_persist: bool = False
    stdout_persist_dir: str = "~/.markitai/assets"
```

- [ ] **Step 3: Add fields to config.schema.json**

In `packages/markitai/src/markitai/config.schema.json`, add to the `ImageConfig.properties` object (after `"filter"`):

```json
"stdout_persist": {
  "default": false,
  "title": "Stdout Persist",
  "type": "boolean"
},
"stdout_persist_dir": {
  "default": "~/.markitai/assets",
  "title": "Stdout Persist Dir",
  "type": "string"
}
```

- [ ] **Step 4: Run schema sync tests**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/test_schema_sync.py -v`
Expected: All PASS (including `test_image_config_fields_match` and `test_image_config_defaults`)

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/config.py packages/markitai/src/markitai/config.schema.json
git commit -m "feat: add stdout_persist config fields to ImageConfig"
```

---

### Task 4: Update regex pattern and rename `strip_asset_references` → `resolve_asset_references`

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py:33-58`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py:522`
- Modify: `packages/markitai/tests/unit/cli/test_file_processor.py:1-82`

- [ ] **Step 1: Update tests for new function name, placeholder format, and backslash support**

Replace the entire `TestStripAssetReferences` class in `packages/markitai/tests/unit/cli/test_file_processor.py`:

```python
from pathlib import Path
from unittest.mock import patch

from markitai.cli.processors.file import resolve_asset_references


class TestResolveAssetReferences:
    """resolve_asset_references() replaces image refs based on available tiers."""

    def test_placeholder_fallback_format(self) -> None:
        """Without protocol or store, produces ![image: filename]() placeholder."""
        markdown = "# Title\n\n![diagram](.markitai/assets/image1.png)\n\nText."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/assets/" not in result
        assert "![image: image1.png]()" in result

    def test_strips_screenshot_reference(self) -> None:
        markdown = "![page1](.markitai/screenshots/doc.pdf.page0001.jpg)\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/screenshots/" not in result
        assert "![image: doc.pdf.page0001.jpg]()" in result

    def test_preserves_non_asset_content(self) -> None:
        markdown = "# Title\n\nSome text with [a link](https://example.com).\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_strips_multiple_references(self) -> None:
        markdown = (
            "![img1](.markitai/assets/a.png)\n"
            "Text between.\n"
            "![img2](.markitai/assets/b.jpg)\n"
            "![page](.markitai/screenshots/c.png)\n"
        )
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert "![image: a.png]()" in result
        assert "![image: b.jpg]()" in result
        assert "![image: c.png]()" in result

    def test_handles_empty_alt_text(self) -> None:
        markdown = "![](.markitai/assets/image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_preserves_external_urls(self) -> None:
        markdown = "![photo](https://example.com/photo.jpg)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_no_images(self) -> None:
        markdown = "# Hello\n\nJust text."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_backslash_paths(self) -> None:
        """Windows-style backslash separators should also be matched."""
        markdown = "![img](.markitai\\assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_handles_mixed_separators(self) -> None:
        """Mixed forward/backslash separators should be matched."""
        markdown = "![img](.markitai/assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/cli/test_file_processor.py::TestResolveAssetReferences -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_asset_references'`

- [ ] **Step 3: Update file.py — rename function, update regex, new placeholder format**

In `packages/markitai/src/markitai/cli/processors/file.py`, replace lines 33-58:

```python
# Pattern matches markdown image references to .markitai/assets/ or .markitai/screenshots/
# Supports both forward slash and backslash for Windows compatibility
_ASSET_REF_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(\.markitai[/\\](?:assets|screenshots)[/\\]([^)]+)\)"
)


def resolve_asset_references(
    markdown: str,
    temp_dir: Path,
    protocol: "Protocol | None" = None,
    asset_store: "AssetStore | None" = None,
    source_name: str = "unknown",
) -> str:
    """Resolve .markitai/assets/ and .markitai/screenshots/ image references.

    Priority cascade:
    1. If protocol is set: replace with terminal inline image escape sequence.
    2. If asset_store is set: persist image, replace with absolute-path URI.
    3. Fallback: replace with ``![image: filename]()`` placeholder.

    Args:
        markdown: Markdown content with asset references.
        temp_dir: Path to the temp directory containing .markitai/ assets.
        protocol: Detected terminal image protocol, or None.
        asset_store: Configured asset store, or None.
        source_name: Source document name for asset store grouping.

    Returns:
        Markdown with asset references resolved.
    """

    def _resolve_image_path(match: re.Match[str], filename: str) -> Path:
        """Resolve the actual image file path from a regex match."""
        filename_normalized = filename.replace("\\", "/")
        full_match = match.group(0)
        subdir = "assets" if "assets" in full_match else "screenshots"
        return temp_dir / ".markitai" / subdir / filename_normalized

    def _replace(match: re.Match[str]) -> str:
        filename = match.group(2)

        if protocol is not None:
            # Tier 1: terminal inline image
            image_path = _resolve_image_path(match, filename)
            if image_path.exists():
                from markitai.utils.terminal_image import render_inline_image

                try:
                    return render_inline_image(image_path, protocol)
                except Exception:
                    pass  # fall through

        if asset_store is not None:
            # Tier 2: persistent asset store
            image_path = _resolve_image_path(match, filename)
            if image_path.exists():
                try:
                    ref_path = asset_store.save(image_path, source_name)
                    uri = asset_store.ref_path_to_markdown_uri(ref_path)
                    return f"![{filename}]({uri})"
                except Exception as e:
                    logger.warning(f"Asset store save failed for {filename}: {e}")
                    # fall through to placeholder

        # Tier 3: placeholder fallback
        return f"![image: {filename}]()"

    return _ASSET_REF_PATTERN.sub(_replace, markdown)


# Backward compatibility alias
strip_asset_references = lambda markdown: resolve_asset_references(markdown, temp_dir=Path("/dev/null"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/cli/test_file_processor.py -v`
Expected: All PASS

- [ ] **Step 5: Update url.py import**

In `packages/markitai/src/markitai/cli/processors/url.py`, line 522, change:

```python
# Before:
from markitai.cli.processors.file import strip_asset_references
# ...
stdout_content = strip_asset_references(stdout_content)

# After:
from markitai.cli.processors.file import resolve_asset_references
# ...
stdout_content = resolve_asset_references(stdout_content, temp_dir=temp_dir)
```

Note: `temp_dir` must be in scope at this point. Check the `process_single_url` function for the temp directory variable name — it is `temp_dir` (created at the top of the function when `stdout_mode` is True).

- [ ] **Step 6: Run full test suite to check nothing is broken**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/ -v --timeout=30 -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/file.py packages/markitai/src/markitai/cli/processors/url.py packages/markitai/tests/unit/cli/test_file_processor.py
git commit -m "refactor: rename strip_asset_references to resolve_asset_references with three-tier cascade"
```

---

### Task 5: Integrate into stdout output path

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py:278-295` (stdout output section)
- Modify: `packages/markitai/src/markitai/cli/processors/url.py:520-531` (stdout output section)

- [ ] **Step 1: Update file.py stdout output section**

In `packages/markitai/src/markitai/cli/processors/file.py`, update the stdout output block (around lines 282-289).

Replace:

```python
        elif stdout_mode:
            # stdout mode: output markdown content
            if final_output_file and final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")
                # Strip .markitai/assets/ and .markitai/screenshots/ references
                # since the temp directory will be deleted after printing
                final_content = strip_asset_references(final_content)
                console.print(final_content, markup=False, highlight=False)
```

With:

```python
        elif stdout_mode:
            # stdout mode: output markdown content
            if final_output_file and final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")

                # Detect terminal image protocol (only if stdout is a TTY)
                from markitai.utils.terminal_image import detect_protocol

                protocol = detect_protocol()

                # Set up asset store if persistence is configured
                store = None
                if cfg.image.stdout_persist:
                    from markitai.utils.asset_store import AssetStore

                    try:
                        store = AssetStore(Path(cfg.image.stdout_persist_dir))
                    except Exception as e:
                        logger.warning(f"Asset store init failed: {e}")

                # Resolve image references (protocol > persist > placeholder)
                source_name = input_path.name
                final_content = resolve_asset_references(
                    final_content,
                    temp_dir=temp_dir,
                    protocol=protocol,
                    asset_store=store,
                    source_name=source_name,
                )

                # Bypass Rich when escape sequences are present (terminal protocol)
                if protocol is not None:
                    sys.stdout.write(final_content)
                    sys.stdout.flush()
                else:
                    console.print(final_content, markup=False, highlight=False)
```

- [ ] **Step 2: Update url.py stdout output section**

In `packages/markitai/src/markitai/cli/processors/url.py`, replace the stdout block (lines 520-531):

```python
# Before:
        if stdout_mode:
            # stdout mode: print final content to console, strip asset refs
            from markitai.cli.processors.file import strip_asset_references

            stdout_content = final_content
            # If LLM produced a .llm.md file, prefer that
            if cfg.llm.enabled:
                llm_file = output_file.with_suffix(".llm.md")
                if llm_file.exists():
                    stdout_content = llm_file.read_text(encoding="utf-8")
            stdout_content = strip_asset_references(stdout_content)
            console.print(stdout_content, markup=False, highlight=False)

# After:
        if stdout_mode:
            from markitai.cli.processors.file import resolve_asset_references

            stdout_content = final_content
            if cfg.llm.enabled:
                llm_file = output_file.with_suffix(".llm.md")
                if llm_file.exists():
                    stdout_content = llm_file.read_text(encoding="utf-8")

            # Detect terminal image protocol (only if stdout is a TTY)
            from markitai.utils.terminal_image import detect_protocol

            protocol = detect_protocol()

            # Set up asset store if persistence is configured
            store = None
            if cfg.image.stdout_persist:
                from markitai.utils.asset_store import AssetStore

                try:
                    store = AssetStore(Path(cfg.image.stdout_persist_dir))
                except Exception as e:
                    logger.warning(f"Asset store init failed: {e}")

            stdout_content = resolve_asset_references(
                stdout_content,
                temp_dir=temp_dir,
                protocol=protocol,
                asset_store=store,
                source_name=url,
            )

            if protocol is not None:
                sys.stdout.write(stdout_content)
                sys.stdout.flush()
            else:
                console.print(stdout_content, markup=False, highlight=False)
```

Note: `temp_dir` is in scope — it is created at the top of `process_single_url` when `stdout_mode` is True. `url` is the function parameter. Add `import sys` at the top if not already imported.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/ -v --timeout=30 -x`
Expected: All PASS

- [ ] **Step 4: Manual smoke test**

Run: `cd /home/oy/Work/markitai && python -m markitai packages/markitai/tests/fixtures/sample.pdf --no-cache`

Verify:
- If running in a supported terminal (Kitty/WezTerm/iTerm2): images render inline
- If not: `![image: filename]()` placeholders appear

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/file.py packages/markitai/src/markitai/cli/processors/url.py
git commit -m "feat: integrate terminal image display and asset persistence into stdout mode"
```

---

### Task 6: Final validation

- [ ] **Step 1: Run full unit test suite**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 2: Run schema sync tests specifically**

Run: `cd /home/oy/Work/markitai && python -m pytest packages/markitai/tests/unit/test_schema_sync.py -v`
Expected: All PASS

- [ ] **Step 3: Smoke test — stdout mode without persist**

```bash
cd /home/oy/Work/markitai
python -m markitai packages/markitai/tests/fixtures/sample.pdf --no-cache
```

Expected: Images shown as `![image: filename]()` (or inline if terminal supports it).

- [ ] **Step 4: Smoke test — stdout mode with persist**

Create `markitai.json`:
```json
{"image": {"stdout_persist": true}}
```

```bash
python -m markitai packages/markitai/tests/fixtures/sample.pdf --no-cache
```

Expected: Images referenced with `file:///` URIs. Check `~/.markitai/assets/blobs/` and `~/.markitai/assets/refs/` exist and contain the correct files.

Clean up: `rm markitai.json`

- [ ] **Step 5: Smoke test — pipe mode (no escape sequences)**

```bash
python -m markitai packages/markitai/tests/fixtures/sample.pdf --no-cache | head -5
```

Expected: No terminal escape sequences in output. Should show frontmatter YAML.

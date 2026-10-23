"""Content-addressed image storage with symlink-based indexing.

Stores image blobs keyed by content hash (SHA-256, first 16 hex chars).
Provides human-navigable symlink refs grouped by source file name. Where
symlinks cannot be made (Windows without administrator rights or Developer
Mode: WinError 1314), a ref is a hard link to the blob, else a copy of it.

Markdown links point at the immutable blob, never at a ref: refs are keyed
by source basename + image filename, so converting another document with
the same name (``a/report.pdf`` then ``b/report.pdf``) or a new version of
the same document re-points the ref, which would silently swap the image
behind every earlier stdout output.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from loguru import logger

_UNSAFE_CHARS = re.compile(r"[/\\:*?\"<>|]+")


def _sanitize_source_name(name: str) -> str:
    """Sanitize a source name (filename or URL) into a safe single-level directory name.

    Strips URL scheme, collapses unsafe characters to underscores, trims.
    """
    # Strip URL scheme (https://, http://, etc.)
    cleaned = re.sub(r"^[a-zA-Z]+://", "", name)
    # Replace path separators and other unsafe chars with underscore
    cleaned = _UNSAFE_CHARS.sub("_", cleaned)
    # Collapse multiple underscores, strip leading/trailing
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    # Truncate to reasonable length
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned or "unknown"


def _content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data, return first 16 hex chars."""
    return hashlib.sha256(data).hexdigest()[:16]


class AssetStore:
    """Content-addressed image store with symlink refs.

    Storage layout::

        <persist_dir>/
          blobs/<hash>.<ext>          # actual image data, deduped
          refs/<source>/<filename>    # symlinks to blobs (browse index);
                                      # hard links or copies where symlinks
                                      # are not allowed

    :meth:`save` returns the blob path: it is content-addressed and never
    rewritten, so links to it stay valid. The mutable refs are a
    convenience index only.
    """

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir).expanduser().resolve()
        self._blobs_dir = self.persist_dir / "blobs"
        self._refs_dir = self.persist_dir / "refs"
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._refs_dir.mkdir(parents=True, exist_ok=True)
        # False once this system refused a symlink: later refs go straight
        # to the fallback instead of failing the same way for every image
        self._symlinks = True

    def save(self, image_path: Path, source_name: str) -> Path:
        """Persist an image and return the absolute, immutable blob path.

        Also (re)points the browse-index symlink
        ``refs/<source_name>/<filename>`` at the blob.

        Args:
            image_path: Path to the image file. Must exist.
            source_name: Source document name (e.g., "sample.pdf").

        Returns:
            Absolute path to the content-addressed blob in ``blobs/``. Link
            to this, not to the ref: a later conversion of a same-named
            source re-points the ref.

        Raises:
            FileNotFoundError: If image_path does not exist.
        """
        # Sanitize source_name to a single safe directory name.
        # URLs, paths with slashes, etc. are collapsed to a flat name.
        safe_source = _sanitize_source_name(source_name)

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
        ref_dir = self._refs_dir / safe_source
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
        self._link_ref(ref_path, rel_target, blob_path)

        # Link target is the immutable blob: the ref above is re-pointed by
        # any later same-named source, the blob never changes
        return blob_path

    def _link_ref(self, ref_path: Path, rel_target: Path, blob_path: Path) -> None:
        """Point the browse-index entry *ref_path* at the blob.

        A symlink where the system allows one. A Windows user without
        administrator rights or Developer Mode cannot make one (WinError 1314),
        nor can some file systems (FAT, some network shares): the ref is then a
        hard link to the blob, or a copy where hard links fail too (another
        volume). Every form opens the same image. The index is a convenience:
        when even a copy fails, the image is still stored and its link, which
        names the blob, still works.
        """
        if self._symlinks:
            try:
                ref_path.symlink_to(rel_target)
                return
            except OSError as e:
                self._symlinks = False
                logger.debug(f"Asset store: no symlinks here ({e}); using hard links")
        try:
            os.link(blob_path, ref_path)
            return
        except OSError:
            pass
        try:
            shutil.copyfile(blob_path, ref_path)
        except OSError as e:
            logger.debug(f"Asset store: no browse-index entry for {ref_path}: {e}")

    def to_markdown_uri(self, path: Path) -> str:
        """Convert a stored asset path to a safe markdown-embeddable URI.

        Handles spaces and special characters via percent-encoding.
        """
        return path.as_uri()

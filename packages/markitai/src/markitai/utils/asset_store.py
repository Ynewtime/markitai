"""Content-addressed image storage with symlink-based indexing.

Stores image blobs keyed by content hash (SHA-256, first 16 hex chars).
Provides human-navigable symlink refs grouped by source file name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


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

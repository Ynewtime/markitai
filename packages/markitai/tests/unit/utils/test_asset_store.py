"""Tests for content-addressed asset store with symlink refs."""

from __future__ import annotations

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

    def test_expands_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_hash_collision_appends_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

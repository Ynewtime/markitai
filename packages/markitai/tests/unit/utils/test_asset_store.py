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
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
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

        blob_path = store.save(img, "sample.pdf")

        assert blob_path.read_bytes() == b"image-data-123"
        # The browse index still gets a ref symlink to the blob
        ref = store.persist_dir / "refs" / "sample.pdf" / "test.jpg"
        assert ref.is_symlink()
        assert ref.resolve() == blob_path.resolve()

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
        """Re-processing same file should update the browse-index symlink."""
        store = AssetStore(tmp_path / "store")
        img1 = self._make_image(tmp_path, "test.jpg", b"version-1")

        store.save(img1, "doc.pdf")
        img2 = self._make_image(tmp_path, "test.jpg", b"version-2")
        blob = store.save(img2, "doc.pdf")

        assert blob.read_bytes() == b"version-2"
        ref = store.persist_dir / "refs" / "doc.pdf" / "test.jpg"
        assert ref.read_bytes() == b"version-2"

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "test.jpg", b"data")

        ref_path = store.save(img, "source.pdf")
        assert ref_path.is_absolute()

    def test_returns_blob_path_not_ref(self, tmp_path: Path) -> None:
        """Returned path is the content-addressed blob, not the mutable ref."""
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "chart.png", b"chart-data")

        path = store.save(img, "report.docx")
        assert path.parent == store.persist_dir / "blobs"
        assert not path.is_symlink()
        assert path.suffix == ".png"

    def test_same_named_source_does_not_change_earlier_link(
        self, tmp_path: Path
    ) -> None:
        """Regression: links handed out earlier must keep their image.

        ``a/report.pdf`` and ``b/report.pdf`` (or two versions of one
        document) share ``refs/report.pdf/<image name>``; the second save
        re-points that ref. The path returned by the first save must still
        show the first image.
        """
        store = AssetStore(tmp_path / "store")
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        img_a = self._make_image(tmp_path / "a", "report.pdf-0001-01.jpg", b"red")
        img_b = self._make_image(tmp_path / "b", "report.pdf-0001-01.jpg", b"blue")

        first = store.save(img_a, "report.pdf")
        second = store.save(img_b, "report.pdf")

        assert first.read_bytes() == b"red"
        assert second.read_bytes() == b"blue"
        assert first != second

    def test_to_markdown_uri_percent_encodes(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        uri = store.to_markdown_uri(tmp_path / "a b.png")
        assert uri.startswith("file://")
        assert "a%20b.png" in uri


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


class TestAssetStoreSourceNameSanitization:
    """source_name is sanitized to a single-level safe directory name."""

    def _make_image(self, tmp_path: Path, name: str, content: bytes) -> Path:
        img = tmp_path / name
        img.write_bytes(content)
        return img

    def test_url_source_name_creates_flat_dir(self, tmp_path: Path) -> None:
        """A URL source_name must not create nested directories."""
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "img.jpg", b"data")

        blob_path = store.save(img, "https://example.com/path?q=1")

        assert blob_path.read_bytes() == b"data"
        # refs dir must be flat (no nested https:/example.com/...)
        refs_dir = tmp_path / "store" / "refs"
        subdirs = list(refs_dir.iterdir())
        assert len(subdirs) == 1
        assert "/" not in subdirs[0].name
        # Symlink must be valid and readable
        assert (subdirs[0] / "img.jpg").read_bytes() == b"data"

    def test_windows_path_source_name(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "img.jpg", b"data")

        ref_path = store.save(img, "C:\\Users\\docs\\report.pdf")
        assert ref_path.read_bytes() == b"data"

    def test_simple_filename_unchanged(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        img = self._make_image(tmp_path, "img.jpg", b"data")

        store.save(img, "sample.pdf")
        assert (store.persist_dir / "refs" / "sample.pdf" / "img.jpg").is_symlink()


class TestAssetStoreErrorHandling:
    """AssetStore errors should not propagate — fall through to placeholder."""

    def test_save_nonexistent_image_raises(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        missing = tmp_path / "missing.jpg"
        with pytest.raises(FileNotFoundError):
            store.save(missing, "doc.pdf")

"""Tests for content-addressed asset store with symlink refs."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import pytest

from markitai.utils.asset_store import AssetStore


def _symlinks_allowed(directory: Path) -> bool:
    """Whether this user may make symlinks here: a Windows user without
    administrator rights or Developer Mode may not (WinError 1314)."""
    probe = directory / "symlink-probe"
    try:
        probe.symlink_to("target")
    except OSError:
        return False
    probe.unlink()
    return True


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
        # The browse index still gets a ref to the blob: a symlink where
        # this user may make one, else a hard link to it
        ref = store.persist_dir / "refs" / "sample.pdf" / "test.jpg"
        assert ref.is_symlink() == _symlinks_allowed(tmp_path)
        assert ref.samefile(blob_path)

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

        blob = store.save(img, "sample.pdf")
        ref = store.persist_dir / "refs" / "sample.pdf" / "img.jpg"
        assert ref.is_symlink() == _symlinks_allowed(tmp_path)
        assert ref.samefile(blob)


class TestAssetStoreErrorHandling:
    """AssetStore errors should not propagate — fall through to placeholder."""

    def test_save_nonexistent_image_raises(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "store")
        missing = tmp_path / "missing.jpg"
        with pytest.raises(FileNotFoundError):
            store.save(missing, "doc.pdf")


#: What Windows raises for a user without the symlink privilege
_NO_PRIVILEGE = "[WinError 1314] A required privilege is not held by the client"


@pytest.fixture
def no_symlinks(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """This system refuses symlinks, as Windows does for a user without
    administrator rights or Developer Mode; records each attempt."""
    attempts: list[Path] = []

    def refuse(self: Path, target: object, target_is_directory: bool = False) -> None:
        attempts.append(self)
        raise OSError(errno.EPERM, _NO_PRIVILEGE)

    monkeypatch.setattr(Path, "symlink_to", refuse)
    return attempts


class TestAssetStoreWithoutSymlinks:
    """Saving never fails because symlinks are refused: the ref is a hard
    link to the blob, else a copy, and the returned blob path always works."""

    def _image(self, tmp_path: Path, name: str, content: bytes) -> Path:
        img = tmp_path / name
        img.write_bytes(content)
        return img

    def test_the_ref_is_a_hard_link_to_the_blob(
        self, tmp_path: Path, no_symlinks: list[Path]
    ) -> None:
        store = AssetStore(tmp_path / "store")
        blob = store.save(self._image(tmp_path, "chart.png", b"chart"), "report.pdf")

        ref = store.persist_dir / "refs" / "report.pdf" / "chart.png"
        assert no_symlinks == [ref]
        assert not ref.is_symlink()
        assert ref.samefile(blob)
        assert blob.read_bytes() == b"chart"

    def test_symlinks_are_tried_once_per_store(
        self, tmp_path: Path, no_symlinks: list[Path]
    ) -> None:
        store = AssetStore(tmp_path / "store")
        for number in range(3):
            store.save(
                self._image(tmp_path, f"p{number}.png", bytes([number])), "a.pdf"
            )

        assert len(no_symlinks) == 1
        for number in range(3):
            ref = store.persist_dir / "refs" / "a.pdf" / f"p{number}.png"
            assert ref.read_bytes() == bytes([number])

    def test_a_copy_where_hard_links_fail_too(
        self,
        tmp_path: Path,
        no_symlinks: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Another volume, FAT or a share: no hard links either."""

        def no_link(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", no_link)
        store = AssetStore(tmp_path / "store")
        blob = store.save(self._image(tmp_path, "chart.png", b"chart"), "report.pdf")

        ref = store.persist_dir / "refs" / "report.pdf" / "chart.png"
        assert not ref.is_symlink()
        assert not ref.samefile(blob)
        assert ref.read_bytes() == blob.read_bytes() == b"chart"

    def test_a_new_version_replaces_the_ref_and_keeps_the_old_blob(
        self, tmp_path: Path, no_symlinks: list[Path]
    ) -> None:
        """A hard-linked ref is replaced, never written through: writing
        into it would change the earlier blob, which old links name."""
        store = AssetStore(tmp_path / "store")
        first = store.save(self._image(tmp_path, "chart.png", b"version-1"), "doc.pdf")
        second = store.save(self._image(tmp_path, "chart.png", b"version-2"), "doc.pdf")

        ref = store.persist_dir / "refs" / "doc.pdf" / "chart.png"
        assert ref.samefile(second)
        assert ref.read_bytes() == b"version-2"
        assert first.read_bytes() == b"version-1"

    def test_no_index_entry_at_all_still_stores_the_image(
        self,
        tmp_path: Path,
        no_symlinks: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EACCES, "Access is denied")

        monkeypatch.setattr(os, "link", refuse)
        monkeypatch.setattr(shutil, "copyfile", refuse)
        store = AssetStore(tmp_path / "store")
        blob = store.save(self._image(tmp_path, "chart.png", b"chart"), "report.pdf")

        assert blob.read_bytes() == b"chart"
        assert not (store.persist_dir / "refs" / "report.pdf" / "chart.png").exists()

    def test_stdout_output_links_the_stored_image(
        self, tmp_path: Path, no_symlinks: list[Path]
    ) -> None:
        """What the user sees: the picture of a converted document links its
        stored copy, not the ``![image: chart.png]()`` placeholder that a
        refused symlink used to leave."""
        from markitai.runs.output import resolve_asset_references

        assets = tmp_path / "run" / ".markitai" / "assets"
        assets.mkdir(parents=True)
        (assets / "chart.png").write_bytes(b"chart")
        store = AssetStore(tmp_path / "store")

        markdown = resolve_asset_references(
            "![Q3 revenue](.markitai/assets/chart.png)",
            tmp_path / "run",
            asset_store=store,
            source_name="report.pdf",
        )

        blob = next((store.persist_dir / "blobs").iterdir())
        assert markdown == f"![Q3 revenue]({blob.as_uri()})"

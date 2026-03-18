"""Tests for utils/paths.py directory management utilities."""

from __future__ import annotations

from pathlib import Path

from markitai.utils.paths import (
    ensure_assets_dir,
    ensure_dir,
    ensure_reports_dir,
    ensure_screenshots_dir,
    ensure_subdir,
)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_creates_directory(self, tmp_path: Path):
        """Should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_dir"
        result = ensure_dir(new_dir)
        assert new_dir.is_dir()
        assert result == new_dir

    def test_idempotent(self, tmp_path: Path):
        """Calling multiple times should not raise."""
        target = tmp_path / "dir"
        ensure_dir(target)
        ensure_dir(target)
        assert target.is_dir()

    def test_creates_parents(self, tmp_path: Path):
        """Should create parent directories."""
        deep = tmp_path / "a" / "b" / "c"
        ensure_dir(deep)
        assert deep.is_dir()

    def test_returns_path_for_chaining(self, tmp_path: Path):
        """Return value should support chaining."""
        result = ensure_dir(tmp_path / "out") / "subfile.txt"
        assert str(result).endswith("subfile.txt")


class TestEnsureSubdir:
    """Tests for ensure_subdir function."""

    def test_creates_subdirectory(self, tmp_path: Path):
        """Should create named subdirectory under parent."""
        result = ensure_subdir(tmp_path, "assets")
        assert result == tmp_path / "assets"
        assert result.is_dir()

    def test_parent_not_exist(self, tmp_path: Path):
        """Should create parent if it doesn't exist."""
        parent = tmp_path / "nonexistent"
        result = ensure_subdir(parent, "child")
        assert result.is_dir()


class TestMetadataDirs:
    """Tests for .markitai/ metadata subdirectory functions."""

    def test_assets_dir_structure(self, tmp_path: Path):
        """ensure_assets_dir creates .markitai/assets/."""
        result = ensure_assets_dir(tmp_path)
        assert result == tmp_path / ".markitai" / "assets"
        assert result.is_dir()

    def test_screenshots_dir_structure(self, tmp_path: Path):
        """ensure_screenshots_dir creates .markitai/screenshots/."""
        result = ensure_screenshots_dir(tmp_path)
        assert result == tmp_path / ".markitai" / "screenshots"
        assert result.is_dir()

    def test_reports_dir_structure(self, tmp_path: Path):
        """ensure_reports_dir creates .markitai/reports/."""
        result = ensure_reports_dir(tmp_path)
        assert result == tmp_path / ".markitai" / "reports"
        assert result.is_dir()

    def test_subdirs_share_parent(self, tmp_path: Path):
        """All metadata subdirs should share the .markitai/ parent."""
        assets = ensure_assets_dir(tmp_path)
        screenshots = ensure_screenshots_dir(tmp_path)
        reports = ensure_reports_dir(tmp_path)

        assert assets.parent.parent == tmp_path
        assert screenshots.parent.parent == tmp_path
        assert reports.parent.parent == tmp_path
        # Same .markitai parent
        assert assets.parent == screenshots.parent == reports.parent

    def test_creating_one_does_not_affect_others(self, tmp_path: Path):
        """Creating one metadata dir should not create the others."""
        ensure_assets_dir(tmp_path)
        meta_dir = tmp_path / ".markitai"
        assert (meta_dir / "assets").is_dir()
        assert not (meta_dir / "screenshots").exists()
        assert not (meta_dir / "reports").exists()

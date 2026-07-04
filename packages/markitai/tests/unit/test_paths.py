"""Tests for utils/paths.py directory management utilities."""

from __future__ import annotations

from pathlib import Path

from markitai.utils.paths import (
    derive_output_name,
    ensure_assets_dir,
    ensure_dir,
    ensure_reports_dir,
    ensure_screenshots_dir,
    ensure_subdir,
    plan_output_names,
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


class TestDeriveOutputName:
    """Tests for derive_output_name extension-replacement naming."""

    def test_replaces_extension(self):
        """sample.pdf becomes sample.md."""
        assert derive_output_name("sample.pdf") == "sample.md"

    def test_md_input_keeps_name(self):
        """note.md stays note.md (identity replacement)."""
        assert derive_output_name("note.md") == "note.md"

    def test_no_extension_appends_md(self):
        """README becomes README.md."""
        assert derive_output_name("README") == "README.md"

    def test_multi_dot_replaces_last_extension(self):
        """archive.tar.gz becomes archive.tar.md."""
        assert derive_output_name("archive.tar.gz") == "archive.tar.md"

    def test_hidden_file(self):
        """Dotfiles keep their name and get .md appended."""
        assert derive_output_name(".env") == ".env.md"

    def test_avoid_triggers_legacy_fallback(self):
        """Colliding names fall back to legacy append scheme."""
        assert derive_output_name("sample.pdf", avoid={"sample.md"}) == "sample.pdf.md"

    def test_avoid_with_md_input(self):
        """Colliding .md input falls back to note.md.md."""
        assert derive_output_name("note.md", avoid={"note.md"}) == "note.md.md"

    def test_avoid_without_collision_is_ignored(self):
        """avoid entries that do not match keep replacement naming."""
        assert derive_output_name("sample.pdf", avoid={"other.md"}) == "sample.md"


class TestPlanOutputNames:
    """Tests for plan_output_names batch collision planning."""

    def test_no_collision(self, tmp_path: Path):
        """Distinct stems map to replaced names."""
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.docx"
        out = tmp_path / "out"
        planned = plan_output_names([(a, out), (b, out)])
        assert planned == {a: "a.md", b: "b.md"}

    def test_duplicate_stems_fall_back_to_legacy(self, tmp_path: Path):
        """a.pdf + a.docx in the same output dir both use legacy names."""
        a_pdf = tmp_path / "a.pdf"
        a_docx = tmp_path / "a.docx"
        out = tmp_path / "out"
        planned = plan_output_names([(a_pdf, out), (a_docx, out)])
        assert planned == {a_pdf: "a.pdf.md", a_docx: "a.docx.md"}

    def test_collision_only_affects_colliding_files(self, tmp_path: Path):
        """Non-colliding inputs keep replacement naming."""
        a_pdf = tmp_path / "a.pdf"
        a_docx = tmp_path / "a.docx"
        b = tmp_path / "b.pdf"
        out = tmp_path / "out"
        planned = plan_output_names([(a_pdf, out), (a_docx, out), (b, out)])
        assert planned[a_pdf] == "a.pdf.md"
        assert planned[a_docx] == "a.docx.md"
        assert planned[b] == "b.md"

    def test_same_stem_different_dirs_no_collision(self, tmp_path: Path):
        """Same-named inputs mapping to different output dirs do not collide."""
        a1 = tmp_path / "sub1" / "a.pdf"
        a2 = tmp_path / "sub2" / "a.pdf"
        out1 = tmp_path / "out" / "sub1"
        out2 = tmp_path / "out" / "sub2"
        planned = plan_output_names([(a1, out1), (a2, out2)])
        assert planned == {a1: "a.md", a2: "a.md"}

    def test_md_source_in_same_dir_falls_back(self, tmp_path: Path):
        """Converting note.md into its own directory must not overwrite it."""
        note = tmp_path / "note.md"
        note.write_text("# note")
        planned = plan_output_names([(note, tmp_path)])
        assert planned == {note: "note.md.md"}

    def test_md_source_different_dir_keeps_name(self, tmp_path: Path):
        """Converting note.md into another directory keeps note.md."""
        note = tmp_path / "note.md"
        note.write_text("# note")
        out = tmp_path / "out"
        planned = plan_output_names([(note, out)])
        assert planned == {note: "note.md"}

    def test_single_pdf(self, tmp_path: Path):
        """Single-entry planning replaces extension."""
        pdf = tmp_path / "sample.pdf"
        planned = plan_output_names([(pdf, tmp_path / "out")])
        assert planned == {pdf: "sample.md"}

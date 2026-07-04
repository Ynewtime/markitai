"""Tests for cli/processors split_output_file_target (-o file-target handling)."""

from __future__ import annotations

from pathlib import Path

from markitai.cli.processors import split_output_file_target


class TestSplitOutputFileTarget:
    """Interpretation of -o values that look like markdown file paths."""

    def test_bare_md_name_is_file_target(self) -> None:
        """`-o out.md` splits into cwd + explicit filename."""
        output_dir, name = split_output_file_target(Path("out.md"))
        assert output_dir == Path(".")
        assert name == "out.md"

    def test_nested_md_path_is_file_target(self, tmp_path: Path) -> None:
        """`-o results/out.md` splits into parent dir + filename."""
        target = tmp_path / "results" / "out.md"
        output_dir, name = split_output_file_target(target)
        assert output_dir == tmp_path / "results"
        assert name == "out.md"

    def test_existing_md_directory_stays_directory(self, tmp_path: Path) -> None:
        """A directory literally named `out.md` keeps directory semantics."""
        md_dir = tmp_path / "out.md"
        md_dir.mkdir()
        output_dir, name = split_output_file_target(md_dir)
        assert output_dir == md_dir
        assert name is None

    def test_non_md_value_is_directory(self, tmp_path: Path) -> None:
        """Values without .md suffix are treated as output directories."""
        output_dir, name = split_output_file_target(tmp_path / "output")
        assert output_dir == tmp_path / "output"
        assert name is None

    def test_other_extension_is_directory(self, tmp_path: Path) -> None:
        """Only `.md` triggers file-target interpretation."""
        output_dir, name = split_output_file_target(tmp_path / "out.markdown")
        assert output_dir == tmp_path / "out.markdown"
        assert name is None

    def test_existing_md_file_is_file_target(self, tmp_path: Path) -> None:
        """An existing regular file named out.md is still a file target."""
        target = tmp_path / "out.md"
        target.write_text("old")
        output_dir, name = split_output_file_target(target)
        assert output_dir == tmp_path
        assert name == "out.md"

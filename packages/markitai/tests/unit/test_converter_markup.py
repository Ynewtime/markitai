"""Tests for the native reStructuredText and Org-mode converters."""

from __future__ import annotations

from pathlib import Path

from markitai.converter.markup import OrgConverter, RstConverter

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestRstConverter:
    def test_fixture_headings_and_literals(self) -> None:
        result = RstConverter().convert(FIXTURES / "sample.rst")

        assert result.markdown.startswith("# Example Docs")
        assert "## XBRL 10-K" in result.markdown
        assert "`example-10k.html`" in result.markdown
        assert result.metadata["format"] == "RST"

    def test_code_directive_becomes_a_fenced_block(self) -> None:
        markdown = RstConverter().convert(FIXTURES / "sample.rst").markdown

        assert "```bash" in markdown
        assert "curl -O \\" in markdown
        assert ".. code::" not in markdown

    def test_adornment_order_decides_heading_levels(self, tmp_path: Path) -> None:
        path = tmp_path / "levels.rst"
        path.write_text(
            "Top\n===\n\nMiddle\n------\n\nDeep\n~~~~\n\nAlso top\n========\n",
            encoding="utf-8",
        )

        lines = RstConverter().convert(path).markdown.splitlines()

        assert lines[0] == "# Top"
        assert "## Middle" in lines
        assert "### Deep" in lines
        assert "# Also top" in lines

    def test_overline_titles_are_recognised(self, tmp_path: Path) -> None:
        path = tmp_path / "over.rst"
        path.write_text("======\nTitle\n======\n\nbody\n", encoding="utf-8")

        assert RstConverter().convert(path).markdown.startswith("# Title")

    def test_bullets_and_field_lists_pass_through(self, tmp_path: Path) -> None:
        path = tmp_path / "lists.rst"
        path.write_text("- one\n- two\n\n:author: Ada\n", encoding="utf-8")

        markdown = RstConverter().convert(path).markdown

        assert "- one" in markdown
        assert ":author: Ada" in markdown

    def test_bullet_line_is_not_read_as_an_adornment(self, tmp_path: Path) -> None:
        path = tmp_path / "dash.rst"
        path.write_text("intro\n\n- a\n- b\n", encoding="utf-8")

        assert "#" not in RstConverter().convert(path).markdown


class TestOrgConverter:
    def test_fixture_headings_and_code(self) -> None:
        result = OrgConverter().convert(FIXTURES / "sample.org")

        assert result.markdown.startswith("# Example Docs")
        assert "## XBRL 10-K" in result.markdown
        assert "`example-10k.html`" in result.markdown
        assert "```bash" in result.markdown
        assert result.metadata["format"] == "ORG"

    def test_title_keyword_becomes_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "titled.org"
        path.write_text("#+TITLE: My Notes\n\n* One\n", encoding="utf-8")

        result = OrgConverter().convert(path)

        assert result.metadata["title"] == "My Notes"
        assert "#+TITLE" not in result.markdown

    def test_links_convert_both_forms(self, tmp_path: Path) -> None:
        path = tmp_path / "links.org"
        path.write_text(
            "See [[https://example.com][the site]] and [[https://plain.example]].\n",
            encoding="utf-8",
        )

        markdown = OrgConverter().convert(path).markdown

        assert "[the site](https://example.com)" in markdown
        assert "<https://plain.example>" in markdown

    def test_verbatim_markers_become_backticks(self, tmp_path: Path) -> None:
        path = tmp_path / "code.org"
        path.write_text("run ~ls -l~ or =pwd= now\n", encoding="utf-8")

        assert OrgConverter().convert(path).markdown == "run `ls -l` or `pwd` now\n"

    def test_spaced_equals_is_not_a_code_span(self, tmp_path: Path) -> None:
        path = tmp_path / "math.org"
        path.write_text("a = b + c = d\n", encoding="utf-8")

        assert OrgConverter().convert(path).markdown == "a = b + c = d\n"

    def test_unclosed_source_block_is_still_fenced(self, tmp_path: Path) -> None:
        path = tmp_path / "open.org"
        path.write_text("#+BEGIN_SRC python\nprint(1)\n", encoding="utf-8")

        markdown = OrgConverter().convert(path).markdown

        assert markdown.splitlines()[0] == "```python"
        assert markdown.splitlines()[-1] == "```"

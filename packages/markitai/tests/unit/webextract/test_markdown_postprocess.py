"""Tests for markdown post-processing cleanup."""

from markitai.webextract.markdown import _postprocess_markdown


def test_strips_orphan_middle_dot_line() -> None:
    md = "Paragraph one.\n\n·\n"
    result = _postprocess_markdown(md)
    assert "·" not in result
    assert "Paragraph one." in result


def test_strips_orphan_dash_line() -> None:
    md = "Text.\n\n—\n\nMore text."
    result = _postprocess_markdown(md)
    assert "\n—\n" not in result


def test_preserves_separator_in_content() -> None:
    md = "Price: $5 · Free shipping"
    result = _postprocess_markdown(md)
    assert "·" in result  # not orphaned


def test_strips_multiple_orphan_separators() -> None:
    md = "Text.\n\n·\n\n|\n\nMore."
    result = _postprocess_markdown(md)
    stripped_lines = [line.strip() for line in result.strip().splitlines()]
    assert "·" not in stripped_lines
    assert "|" not in stripped_lines

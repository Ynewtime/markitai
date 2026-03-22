"""Tests for title truncation in clean_title."""

from markitai.webextract.metadata import clean_title

_MAX_TITLE_LEN = 120


def test_short_title_unchanged() -> None:
    assert clean_title("Short Title") == "Short Title"


def test_long_title_truncated_at_word_boundary() -> None:
    long = "A " * 100  # 200 chars
    result = clean_title(long)
    assert result is not None
    assert len(result) <= _MAX_TITLE_LEN + 1  # +1 for ellipsis char


def test_truncation_adds_ellipsis() -> None:
    long = "word " * 50
    result = clean_title(long)
    assert result is not None
    assert result.endswith("…")


def test_truncation_respects_word_boundary() -> None:
    long = "abcdefghij " * 15  # 165 chars
    result = clean_title(long)
    assert result is not None
    assert result.rstrip("…").endswith("abcdefghij")


def test_site_stripping_before_truncation() -> None:
    title = "Short Title | MySite"
    assert clean_title(title, site="MySite") == "Short Title"

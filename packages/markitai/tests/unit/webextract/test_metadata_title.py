"""Tests for title truncation in clean_title."""

from markitai.webextract.metadata import clean_title

# Generous cap (defuddle parity: real titles such as GitHub repo
# descriptions ~170 chars must survive untruncated).
_MAX_TITLE_LEN = 300


def test_short_title_unchanged() -> None:
    assert clean_title("Short Title") == "Short Title"


def test_realistic_long_title_not_truncated() -> None:
    title = (
        "Panniantong/Agent-Reach: Give your AI agent eyes to see the entire "
        "internet. Read & search Twitter, Reddit, YouTube, GitHub, Bilibili, "
        "XiaoHongShu — one CLI, zero API fees."
    )
    assert clean_title(title) == title


def test_pathological_title_truncated_at_word_boundary() -> None:
    long = "A " * 200  # 400 chars
    result = clean_title(long)
    assert result is not None
    assert len(result) <= _MAX_TITLE_LEN + 1  # +1 for ellipsis char


def test_truncation_adds_ellipsis() -> None:
    long = "word " * 100  # 500 chars
    result = clean_title(long)
    assert result is not None
    assert result.endswith("…")


def test_truncation_respects_word_boundary() -> None:
    long = "abcdefghij " * 40  # 440 chars
    result = clean_title(long)
    assert result is not None
    assert result.rstrip("…").endswith("abcdefghij")


def test_site_stripping_before_truncation() -> None:
    title = "Short Title | MySite"
    assert clean_title(title, site="MySite") == "Short Title"

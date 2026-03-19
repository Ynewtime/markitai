"""Semantic parity tests: freeze expected extraction behaviour before architecture change.

These tests are intentionally written against types and fields that do NOT yet
exist (``ContentProfile``, ``ExtractionInfo``, ``SemanticExtraction``).  They
will fail until Tasks 2-5 introduce those types.  The intent is to lock down
the desired behaviour so refactoring cannot silently regress it.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[3] / "fixtures" / "web"

X_URL = "https://x.com/ixiaowenz/status/2030105637204676808"
ARTICLE_URL = "https://example.com/blog/async-python"


def _load_fixture(name: str) -> str:
    """Load an HTML fixture file by name.

    Args:
        name: Fixture filename relative to the web fixtures directory.

    Returns:
        Raw HTML string.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# X/Twitter status page
# ---------------------------------------------------------------------------


def test_x_status_semantics_match_expected_fixture() -> None:
    """X status page must be classified as SOCIAL_POST with a populated thread.

    Verifies:
    - metadata.title reflects the post author
    - info.content_profile is ContentProfile.SOCIAL_POST
    - semantic.thread is not None and carries the correct author handle
    - Noise sections ("Discover more", quote cards) are absent from markdown
    """
    # These imports will raise ImportError until the new types are implemented.
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile  # type: ignore[attr-defined]

    html = _load_fixture("x_status_2030105637204676808.playwright.html")
    result = extract_web_content(html, X_URL)

    assert result.metadata.title == "Post by @ixiaowenz"

    # info and semantic fields do not exist yet → AttributeError expected
    assert result.info.content_profile == ContentProfile.SOCIAL_POST  # type: ignore[attr-defined]
    assert result.semantic is not None  # type: ignore[attr-defined]
    assert result.semantic.thread is not None  # type: ignore[attr-defined]
    assert result.semantic.thread.main_item.author_handle == "@ixiaowenz"  # type: ignore[attr-defined]

    assert "Discover more" not in result.markdown
    assert "Quote" not in result.markdown


# ---------------------------------------------------------------------------
# Generic article page
# ---------------------------------------------------------------------------


def test_generic_article_baseline_stays_non_threaded_and_clean() -> None:
    """Generic article pages must not gain a semantic thread object.

    Verifies:
    - info.content_profile is ContentProfile.GENERIC_ARTICLE
    - semantic is None (no structured thread model)
    - Sidebar noise ("Related posts") is absent from markdown
    """
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile  # type: ignore[attr-defined]

    html = _load_fixture("generic_article.playwright.html")
    result = extract_web_content(html, ARTICLE_URL)

    assert result.info.content_profile == ContentProfile.GENERIC_ARTICLE  # type: ignore[attr-defined]
    assert result.semantic is None  # type: ignore[attr-defined]

    assert "Related posts" not in result.markdown

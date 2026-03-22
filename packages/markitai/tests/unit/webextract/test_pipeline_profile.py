"""Tests for content_profile derivation in the generic extraction path."""

from markitai.webextract.pipeline import _extract_generic

_X_HTML = "<html><head><title>T</title></head><body><p>hello</p></body></html>"
_X_URL = "https://x.com/user/status/123"

_GH_HTML = "<html><head><title>T</title></head><body><p>hello</p></body></html>"
_GH_URL = "https://github.com/org/repo/issues/1"

_GENERIC_URL = "https://example.com/article"


def test_generic_path_uses_social_post_for_x_tweet() -> None:
    result = _extract_generic(_X_HTML, _X_URL)
    assert result.info is not None
    assert result.info.content_profile.value == "social_post"


def test_generic_path_uses_discussion_issue_for_github() -> None:
    result = _extract_generic(_GH_HTML, _GH_URL)
    assert result.info is not None
    assert result.info.content_profile.value == "discussion_issue"


def test_generic_path_uses_generic_article_for_unknown() -> None:
    result = _extract_generic(_X_HTML, _GENERIC_URL)
    assert result.info is not None
    assert result.info.content_profile.value == "generic_article"

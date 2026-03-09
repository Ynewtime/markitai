from __future__ import annotations


def test_registry_uses_github_issue_extractor_for_issue_urls() -> None:
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://github.com/org/repo/issues/56")

    assert extractor is not None
    assert extractor.name == "github_issue"


def test_registry_uses_x_article_extractor_for_x_article_urls() -> None:
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://x.com/i/articles/123")

    assert extractor is not None
    assert extractor.name == "x_article"

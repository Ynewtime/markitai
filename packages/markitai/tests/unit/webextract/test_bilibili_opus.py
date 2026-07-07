"""Tests for Bilibili opus (专栏/动态) post extraction.

Fixture is a trimmed real capture of a Bilibili opus page (rendered DOM,
via Playwright) — see BilibiliOpusExtractor's module docstring for why the
page needs a dedicated extractor: generic extraction pulls page chrome
(login prompt, like/comment/coin counts, share links, back-to-top) into
the output verbatim, since it lives as *siblings* of the actual content
card rather than being filtered by any noise selector.
"""

from __future__ import annotations

from pathlib import Path

from markitai.webextract.types import ExtractedWebContent

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"
OPUS_URL = "https://www.bilibili.com/opus/1053433238661365784"


def _load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _extract() -> ExtractedWebContent:
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("bilibili_opus_1053433238661365784.playwright.html")
    return extract_web_content(html, OPUS_URL)


class TestRegistry:
    def test_registry_resolves_bilibili_opus_extractor(self) -> None:
        from markitai.webextract.extractors.registry import find_extractor

        extractor = find_extractor(OPUS_URL)
        assert extractor is not None
        assert extractor.name == "bilibili_opus"

    def test_registry_does_not_match_bilibili_video_urls(self) -> None:
        """Only /opus/ posts get this extractor — video pages are a
        different content shape, not covered by this fixture/extractor."""
        from markitai.webextract.extractors.registry import find_extractor

        extractor = find_extractor("https://www.bilibili.com/video/BV1xx411c7mD")
        assert extractor is None or extractor.name != "bilibili_opus"


class TestExtractedContent:
    def test_body_contains_real_paragraphs(self) -> None:
        result = _extract()
        assert "这里看过来宝子们安装包SD" in result.markdown
        assert "以及想系统学习AI" in result.markdown

    def test_body_contains_both_images(self) -> None:
        result = _extract()
        assert "3787f6349efa1cc71bb93cadd368c3963546697288386965" in result.markdown
        assert "73ab856e48cdb0037e043668a550eab23546697288386965" in result.markdown

    def test_body_excludes_copyright_tag(self) -> None:
        """cv41273131 is an internal content-ID tag (opus-module-copyright),
        not reader-facing content."""
        result = _extract()
        assert "cv41273131" not in result.markdown

    def test_body_excludes_share_widget(self) -> None:
        result = _extract()
        assert "分享至" not in result.markdown
        assert "微信扫一扫分享" not in result.markdown

    def test_body_excludes_login_wall_and_stat_counts(self) -> None:
        """These live in DOM siblings of .bili-opus-view (comment tab pane,
        right sidebar) — confirming the extractor scopes to the right root."""
        result = _extract()
        assert "请先登录" not in result.markdown
        assert "给UP主投上" not in result.markdown
        assert "顶部" not in result.markdown


class TestMetadata:
    def test_title_is_clean(self) -> None:
        result = _extract()
        assert result.metadata.title == "资料在这里哦！"

    def test_author_name(self) -> None:
        result = _extract()
        assert result.metadata.author == "AI提示词"

    def test_published_date(self) -> None:
        result = _extract()
        assert result.metadata.published == "2026-06-10"

    def test_content_profile_is_social_post(self) -> None:
        from markitai.webextract.types import ContentProfile

        result = _extract()
        assert result.info is not None
        assert result.info.content_profile == ContentProfile.SOCIAL_POST

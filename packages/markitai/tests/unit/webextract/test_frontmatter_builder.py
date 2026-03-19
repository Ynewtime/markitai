from __future__ import annotations

from markitai.webextract import build_source_frontmatter, coerce_source_frontmatter
from markitai.webextract.types import (
    ContentProfile,
    ExtractedWebContent,
    ExtractionInfo,
    QualityAssessment,
    WebMetadata,
)


def _make_result(
    *,
    title: str | None = "Test Title",
    author: str | None = None,
    site: str | None = None,
    published: str | None = None,
    description: str | None = None,
    canonical_url: str | None = None,
    content_profile: ContentProfile = ContentProfile.GENERIC_ARTICLE,
    extractor_name: str = "generic",
    word_count: int = 100,
    score: float = 0.8,
    accepted: bool = True,
) -> ExtractedWebContent:
    info = ExtractionInfo(
        content_profile=content_profile,
        extractor_name=extractor_name,
        word_count=word_count,
    )
    quality = QualityAssessment(accepted=accepted, score=score)
    return ExtractedWebContent(
        clean_html="<p>content</p>",
        markdown="content",
        metadata=WebMetadata(
            title=title,
            author=author,
            site=site,
            published=published,
            description=description,
            canonical_url=canonical_url,
        ),
        word_count=word_count,
        info=info,
        quality=quality,
    )


class TestBuildSourceFrontmatter:
    """Tests for build_source_frontmatter()."""

    def test_exports_title_from_metadata(self) -> None:
        result = _make_result(title="My Article")
        fm = build_source_frontmatter(result)
        assert fm["title"] == "My Article"

    def test_exports_author_from_metadata(self) -> None:
        result = _make_result(author="Jane Doe")
        fm = build_source_frontmatter(result)
        assert fm["author"] == "Jane Doe"

    def test_exports_site_from_metadata(self) -> None:
        result = _make_result(site="Example Site")
        fm = build_source_frontmatter(result)
        assert fm["site"] == "Example Site"

    def test_exports_published_from_metadata(self) -> None:
        result = _make_result(published="2026-01-01")
        fm = build_source_frontmatter(result)
        assert fm["published"] == "2026-01-01"

    def test_exports_description_from_metadata(self) -> None:
        result = _make_result(description="A test article")
        fm = build_source_frontmatter(result)
        assert fm["description"] == "A test article"

    def test_exports_canonical_url_from_metadata(self) -> None:
        result = _make_result(canonical_url="https://example.com/article")
        fm = build_source_frontmatter(result)
        assert fm["canonical_url"] == "https://example.com/article"

    def test_exports_word_count_from_info(self) -> None:
        result = _make_result(word_count=241)
        fm = build_source_frontmatter(result)
        assert fm["word_count"] == 241

    def test_exports_content_profile_as_snake_case_string(self) -> None:
        result = _make_result(content_profile=ContentProfile.SOCIAL_POST)
        fm = build_source_frontmatter(result)
        assert fm["content_profile"] == "social_post"

    def test_generic_article_profile_exported_as_snake_case(self) -> None:
        result = _make_result(content_profile=ContentProfile.GENERIC_ARTICLE)
        fm = build_source_frontmatter(result)
        assert fm["content_profile"] == "generic_article"

    def test_does_not_export_quality_score(self) -> None:
        result = _make_result(score=0.95)
        fm = build_source_frontmatter(result)
        assert "score" not in fm

    def test_does_not_export_accepted_flag(self) -> None:
        result = _make_result(accepted=True)
        fm = build_source_frontmatter(result)
        assert "accepted" not in fm

    def test_does_not_export_quality_reasons(self) -> None:
        result = _make_result()
        fm = build_source_frontmatter(result)
        assert "reasons" not in fm

    def test_none_metadata_fields_omitted(self) -> None:
        result = _make_result(title="Only Title", author=None, site=None)
        fm = build_source_frontmatter(result)
        assert "author" not in fm
        assert "site" not in fm

    def test_social_post_scenario(self) -> None:
        """Scenario from the task description."""
        result = ExtractedWebContent(
            clean_html="<p>tweet</p>",
            markdown="tweet content",
            metadata=WebMetadata(
                title="Post by @ixiaowenz",
                author="@ixiaowenz",
                site="X (Twitter)",
            ),
            word_count=241,
            info=ExtractionInfo(
                content_profile=ContentProfile.SOCIAL_POST,
                word_count=241,
                extractor_name="x_tweet",
            ),
            quality=QualityAssessment(accepted=True, score=0.95),
        )
        fm = build_source_frontmatter(result)
        assert fm["title"] == "Post by @ixiaowenz"
        assert fm["word_count"] == 241
        assert fm["content_profile"] == "social_post"
        assert "score" not in fm
        assert "accepted" not in fm

    def test_without_info_still_exports_metadata(self) -> None:
        """When info is None (legacy construction), metadata still exports."""
        content = ExtractedWebContent(
            clean_html="<p>Hello</p>",
            markdown="Hello",
            metadata=WebMetadata(title="Legacy Title"),
            word_count=5,
        )
        fm = build_source_frontmatter(content)
        assert fm["title"] == "Legacy Title"
        assert "content_profile" not in fm
        assert "word_count" not in fm


class TestCoerceSourceFrontmatterBackwardCompat:
    """Tests that coerce_source_frontmatter() is preserved and still works."""

    def test_accepts_webmetadata_object(self) -> None:
        meta = WebMetadata(title="Hello", author="Author")
        fm = coerce_source_frontmatter(meta)
        assert fm["title"] == "Hello"
        assert fm["author"] == "Author"

    def test_accepts_dict(self) -> None:
        fm = coerce_source_frontmatter({"title": "From Dict"})
        assert fm["title"] == "From Dict"

    def test_accepts_none(self) -> None:
        fm = coerce_source_frontmatter(None)
        assert fm == {}

    def test_none_values_omitted(self) -> None:
        meta = WebMetadata(title="T", author=None)
        fm = coerce_source_frontmatter(meta)
        assert "author" not in fm

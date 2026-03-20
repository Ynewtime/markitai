from __future__ import annotations

from markitai.webextract.types import (
    ContentProfile,
    ExtractedWebContent,
    ExtractionInfo,
    QualityAssessment,
    SemanticExtraction,
    WebMetadata,
)


class TestContentProfile:
    """Tests for the ContentProfile enum."""

    def test_has_generic_article_variant(self) -> None:
        assert ContentProfile.GENERIC_ARTICLE is not None

    def test_has_social_post_variant(self) -> None:
        assert ContentProfile.SOCIAL_POST is not None

    def test_value_is_snake_case_string(self) -> None:
        assert ContentProfile.GENERIC_ARTICLE.value == "generic_article"
        assert ContentProfile.SOCIAL_POST.value == "social_post"


class TestExtractionInfo:
    """Tests for the ExtractionInfo dataclass."""

    def test_basic_construction(self) -> None:
        info = ExtractionInfo(
            content_profile=ContentProfile.GENERIC_ARTICLE,
            extractor_name="generic",
            word_count=100,
        )
        assert info.content_profile == ContentProfile.GENERIC_ARTICLE
        assert info.extractor_name == "generic"
        assert info.word_count == 100

    def test_optional_fields_default_to_none(self) -> None:
        info = ExtractionInfo(
            content_profile=ContentProfile.GENERIC_ARTICLE,
            extractor_name="generic",
            word_count=50,
        )
        assert info.enricher_name is None

    def test_source_kind_defaults_to_html(self) -> None:
        info = ExtractionInfo(
            content_profile=ContentProfile.SOCIAL_POST,
            extractor_name="x_tweet",
            word_count=241,
        )
        assert info.source_kind == "html"

    def test_custom_enricher_name(self) -> None:
        info = ExtractionInfo(
            content_profile=ContentProfile.SOCIAL_POST,
            extractor_name="x_tweet",
            word_count=241,
            enricher_name="thread_builder",
        )
        assert info.enricher_name == "thread_builder"


class TestQualityAssessment:
    """Tests for the QualityAssessment dataclass."""

    def test_basic_construction(self) -> None:
        q = QualityAssessment(accepted=True, score=0.95)
        assert q.accepted is True
        assert q.score == 0.95

    def test_reasons_defaults_to_empty_list(self) -> None:
        q = QualityAssessment(accepted=False, score=0.2)
        assert q.reasons == []

    def test_reasons_can_be_set(self) -> None:
        q = QualityAssessment(accepted=False, score=0.2, reasons=["too short"])
        assert q.reasons == ["too short"]


class TestSemanticExtraction:
    """Tests for the SemanticExtraction dataclass."""

    def test_construction_with_no_thread(self) -> None:
        sem = SemanticExtraction()
        assert sem.thread is None

    def test_thread_can_be_set(self) -> None:
        sem = SemanticExtraction(thread={"items": []})
        assert sem.thread == {"items": []}


class TestExtractedWebContentBackwardCompat:
    """Tests that existing construction of ExtractedWebContent still works."""

    def test_old_style_construction_with_word_count(self) -> None:
        """Pipeline.py currently constructs with word_count as positional/keyword arg."""
        content = ExtractedWebContent(
            clean_html="<p>Hello</p>",
            markdown="Hello",
            metadata=WebMetadata(title="Test"),
            word_count=1,
        )
        assert content.word_count == 1
        assert content.clean_html == "<p>Hello</p>"
        assert content.markdown == "Hello"

    def test_new_fields_default_to_none(self) -> None:
        """New optional fields must not break existing construction."""
        content = ExtractedWebContent(
            clean_html="<p>Hello</p>",
            markdown="Hello",
            metadata=WebMetadata(title="Test"),
            word_count=10,
        )
        assert content.info is None
        assert content.quality is None
        assert content.semantic is None

    def test_new_fields_can_be_set(self) -> None:
        info = ExtractionInfo(
            content_profile=ContentProfile.SOCIAL_POST,
            extractor_name="x_tweet",
            word_count=241,
        )
        quality = QualityAssessment(accepted=True, score=0.95)
        content = ExtractedWebContent(
            clean_html="<p>post</p>",
            markdown="post",
            metadata=WebMetadata(title="Post by @ixiaowenz", author="@ixiaowenz"),
            word_count=241,
            info=info,
            quality=quality,
        )
        assert content.info is info
        assert content.quality is quality
        assert content.info.content_profile == ContentProfile.SOCIAL_POST

    def test_diagnostics_defaults_to_empty_dict(self) -> None:
        content = ExtractedWebContent(
            clean_html="<p>X</p>",
            markdown="X",
            metadata=WebMetadata(),
            word_count=1,
        )
        assert content.diagnostics == {}

"""Tests for the VLM output degeneration guard (markitai.llm.degeneration)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.llm.degeneration import (
    detect_trailing_repetition,
    truncate_degenerate_tail,
)

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig


CLEAN_PREFIX = "# Report\n\nThis is a legitimate introduction paragraph.\n\n"


@pytest.fixture
def sample_test_image(tmp_path: Path) -> Path:
    """Create a sample test image file."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_page.jpg"
    img.save(img_path, "JPEG")
    return img_path


class TestDetectTrailingRepetition:
    """Unit tests for detect_trailing_repetition."""

    def test_degenerate_line_loop(self) -> None:
        """Same non-empty line repeated many times at the tail is flagged."""
        line = "All work and no play makes Jack a dull boy."
        text = CLEAN_PREFIX + (line + "\n") * 10
        offset = detect_trailing_repetition(text)
        assert offset is not None
        # Truncating at offset keeps exactly one instance of the line
        assert text[:offset].count(line) == 1

    def test_degenerate_line_loop_with_blank_lines(self) -> None:
        """Blank lines between repeated lines do not hide the loop."""
        line = "The model is stuck in a loop."
        text = CLEAN_PREFIX + (line + "\n\n") * 8
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset].count(line) == 1

    def test_degenerate_chunk_loop(self) -> None:
        """A long chunk repeated consecutively at the tail is flagged."""
        unit = "The same sentence keeps repeating endlessly here. "
        text = CLEAN_PREFIX + unit * 6
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset].count(unit) == 1

    def test_degenerate_chunk_loop_with_partial_final_unit(self) -> None:
        """Generation cut mid-unit (max_tokens) is still detected."""
        unit = "Repetition unit that got cut off at the end of output. "
        text = CLEAN_PREFIX + unit * 5 + unit[:20]
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset].count(unit) == 1

    def test_cjk_line_repetition(self) -> None:
        """CJK line loops are detected."""
        line = "这个模型开始不停地重复同样的一句话。"
        text = "# 报告\n\n这是正常的中文内容段落。\n\n" + (line + "\n") * 8
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset].count(line) == 1

    def test_cjk_chunk_repetition(self) -> None:
        """CJK chunk loops (no newlines) are detected."""
        unit = "模型输出退化成重复内容，这是一个很长的重复单元。"
        text = "# 报告\n\n这是正常的中文内容段落。\n\n" + unit * 5
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset].count(unit) == 1

    def test_clean_markdown_table_not_flagged(self) -> None:
        """Table rows with differing content are legitimate repetition."""
        rows = "\n".join(f"| row {i} | value {i} |" for i in range(50))
        text = "# Data\n\n| a | b |\n|---|---|\n" + rows + "\n"
        assert detect_trailing_repetition(text) is None

    def test_clean_list_not_flagged(self) -> None:
        """List items with differing content are not flagged."""
        items = "\n".join(f"- item number {i}" for i in range(40))
        text = "# List\n\n" + items + "\n"
        assert detect_trailing_repetition(text) is None

    def test_clean_prose_not_flagged(self) -> None:
        """Ordinary prose is not flagged."""
        assert detect_trailing_repetition(CLEAN_PREFIX) is None

    def test_empty_text(self) -> None:
        assert detect_trailing_repetition("") is None

    def test_line_repeats_below_threshold_not_flagged(self) -> None:
        """A short line repeated fewer than min_line_repeats times is fine."""
        text = CLEAN_PREFIX + "repeat me\n" * 5
        assert detect_trailing_repetition(text) is None

    def test_chunk_repeats_below_threshold_not_flagged(self) -> None:
        """A chunk repeated fewer than min_repeats times is fine."""
        unit = "only three repeats of chunk. "
        text = CLEAN_PREFIX + unit * 3
        assert detect_trailing_repetition(text) is None

    def test_fully_degenerate_text_keeps_one_unit(self) -> None:
        """Text that is nothing but repeats keeps a single unit."""
        line = "Completely degenerate output line."
        text = (line + "\n") * 12
        offset = detect_trailing_repetition(text)
        assert offset is not None
        assert text[:offset] == line + "\n"

    def test_large_text_is_fast(self) -> None:
        """~100KB clean text is scanned quickly (O(n) detectors)."""
        import time

        text = "".join(
            f"Paragraph {i} with some unique content.\n" for i in range(2500)
        )
        assert len(text) > 100_000
        start = time.perf_counter()
        assert detect_trailing_repetition(text) is None
        assert time.perf_counter() - start < 2.0


class TestTruncateDegenerateTail:
    """Unit tests for truncate_degenerate_tail."""

    def test_clean_text_untouched(self) -> None:
        text, truncated = truncate_degenerate_tail(CLEAN_PREFIX, context="src")
        assert text == CLEAN_PREFIX
        assert truncated is False

    def test_degenerate_text_truncated(self) -> None:
        line = "All work and no play makes Jack a dull boy."
        text, truncated = truncate_degenerate_tail(
            CLEAN_PREFIX + (line + "\n") * 10, context="src", stage="test"
        )
        assert truncated is True
        assert text.count(line) == 1
        assert text.endswith(line)

    def test_empty_text(self) -> None:
        text, truncated = truncate_degenerate_tail("", context="src")
        assert text == ""
        assert truncated is False


DEGENERATE_LINE = "All work and no play makes Jack a dull boy."
DEGENERATE_MARKDOWN = (
    "# Extracted\n\nReal content here.\n\n" + (DEGENERATE_LINE + "\n") * 12
)


class TestExtractFromScreenshotDegeneration:
    """Degenerate VLM responses are truncated and not cached."""

    @pytest.mark.asyncio
    async def test_degenerate_response_truncated_and_not_cached(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import EnhancedDocumentResult, Frontmatter, LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(return_value=None)
        processor._persistent_cache.set = MagicMock()

        result = EnhancedDocumentResult(
            cleaned_markdown=DEGENERATE_MARKDOWN,
            frontmatter=Frontmatter(description="A doc", tags=["test"]),
        )
        raw = MagicMock()
        raw.model = "vision-model"
        raw.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        raw.choices = [MagicMock(finish_reason="stop")]

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.engine.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.extract_from_screenshot(
                sample_test_image,
                "https://example.com/page",
            )

        assert cleaned.count(DEGENERATE_LINE) == 1
        assert "Real content here." in cleaned
        processor._persistent_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_response_still_cached(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import EnhancedDocumentResult, Frontmatter, LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(return_value=None)
        processor._persistent_cache.set = MagicMock()

        clean_markdown = "# Extracted\n\nReal content here."
        result = EnhancedDocumentResult(
            cleaned_markdown=clean_markdown,
            frontmatter=Frontmatter(description="A doc", tags=["test"]),
        )
        raw = MagicMock()
        raw.model = "vision-model"
        raw.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        raw.choices = [MagicMock(finish_reason="stop")]

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.engine.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.extract_from_screenshot(
                sample_test_image,
                "https://example.com/page",
            )

        assert cleaned == clean_markdown
        processor._persistent_cache.set.assert_called_once()


class TestEnhanceDocumentWithVisionDegeneration:
    """Degenerate vision-enhance responses are truncated and not cached."""

    @pytest.mark.asyncio
    async def test_degenerate_response_truncated_and_not_cached(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(return_value=None)
        processor._persistent_cache.set = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=DEGENERATE_MARKDOWN))
        ]
        mock_response.model = "vision-model"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        processor._router = mock_router
        processor._vision_router = mock_router

        result = await processor.enhance_document_with_vision(
            "# Title\n\nSome content.", [sample_test_image], "test.pdf"
        )

        assert result.count(DEGENERATE_LINE) == 1
        processor._persistent_cache.set.assert_not_called()


class TestVisionAnalysisDegeneration:
    """Degenerate image analysis text is truncated and not cached."""

    @pytest.mark.asyncio
    async def test_analyze_image_degenerate_extracted_text(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import ImageAnalysis, LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(return_value=None)
        processor._persistent_cache.set = MagicMock()

        analysis = ImageAnalysis(
            caption="A chart",
            description="A chart with labels",
            extracted_text=DEGENERATE_MARKDOWN,
        )
        processor._analyze_image_with_fallback = AsyncMock(  # type: ignore[method-assign]
            return_value=analysis
        )

        result = await processor.analyze_image(sample_test_image)

        assert result.extracted_text is not None
        assert result.extracted_text.count(DEGENERATE_LINE) == 1
        processor._persistent_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_image_clean_text_still_cached(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import ImageAnalysis, LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(return_value=None)
        processor._persistent_cache.set = MagicMock()

        analysis = ImageAnalysis(
            caption="A chart",
            description="A chart with labels",
            extracted_text="Quarterly revenue: 10, 20, 30",
        )
        processor._analyze_image_with_fallback = AsyncMock(  # type: ignore[method-assign]
            return_value=analysis
        )

        result = await processor.analyze_image(sample_test_image)

        assert result.extracted_text == "Quarterly revenue: 10, 20, 30"
        processor._persistent_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_page_content_degenerate_truncated(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        from markitai.llm import LLMProcessor, LLMResponse

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._call_llm = AsyncMock(  # type: ignore[method-assign]
            return_value=LLMResponse(
                content=DEGENERATE_MARKDOWN,
                model="vision-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
        )

        result = await processor.extract_page_content(sample_test_image)

        assert result.count(DEGENERATE_LINE) == 1
        assert "Real content here." in result

"""Regression tests: empty vision answers and the MIME label of sent images.

- An empty image caption (from the structured ladder, the two-call fallback,
  a batch answer or a collected ``--llm-batch`` result) and an empty
  screenshot extraction were written to the TTL-less persistent cache, so
  one bad answer blanked the image or page on every later run.
- The LLM image loader re-encodes oversized PNG/WEBP/GIF images as JPEG,
  but every data URL was labelled from the file suffix (``image/png`` on
  JPEG bytes).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.llm.document import DocumentEnhancer
from markitai.llm.engine import EmptyLLMResponseError
from markitai.llm.types import (
    BatchImageAnalysisResult,
    DocumentProcessResult,
    Frontmatter,
    ImageAnalysisResult,
    LLMResponse,
    SingleImageResult,
)
from markitai.utils.mime import get_llm_image_mime
from tests.unit.test_vision_mixin import MockVisionProcessor

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _image_url(messages: list[dict[str, Any]]) -> str:
    """First image data URL in a chat message list."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return part["image_url"]["url"]
    raise AssertionError("no image in messages")


def _enhancer(images: dict[str, bytes]) -> tuple[DocumentEnhancer, MagicMock]:
    engine = MagicMock()
    engine.persistent_cache.get.return_value = None
    engine.complete_structured = AsyncMock(side_effect=RuntimeError("stop"))
    prompts = MagicMock()
    prompts.get_prompt.return_value = "prompt"

    def get_cached_image(path: Path) -> tuple[bytes, str]:
        data = images[path.name]
        return data, base64.b64encode(data).decode()

    enhancer = DocumentEnhancer(
        engine=engine,
        prompt_manager=prompts,
        config=MagicMock(),
        cache_model_scope="pool:test",
        vision_cache_model_scope="pool:test-vision",
        get_vision_router=lambda: MagicMock(),
        get_cached_image=get_cached_image,
        get_next_call_index=lambda _context: 0,
    )
    return enhancer, engine


class TestLlmImageMime:
    @pytest.mark.parametrize(
        ("data", "suffix", "mime"),
        [
            (JPEG_BYTES, ".png", "image/jpeg"),
            (JPEG_BYTES, ".webp", "image/jpeg"),
            (PNG_BYTES, ".bmp", "image/png"),
            (b"GIF89a" + b"\x00" * 8, ".gif", "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ".webp", "image/webp"),
            # Unrecognised bytes fall back to the extension
            (b"unknown", ".png", "image/png"),
            (b"unknown", ".tiff", "image/png"),
        ],
    )
    def test_bytes_signature_wins_over_suffix(
        self, data: bytes, suffix: str, mime: str
    ) -> None:
        assert get_llm_image_mime(data, suffix) == mime

    def test_image_analysis_labels_reencoded_jpeg(self, tmp_path: Path) -> None:
        processor = MockVisionProcessor()
        image = tmp_path / "chart.png"
        processor._image_cache[str(image)] = (
            JPEG_BYTES,
            base64.b64encode(JPEG_BYTES).decode(),
        )

        plan = processor.prepare_image_plan(image)

        assert _image_url(plan.messages).startswith("data:image/jpeg;base64,")

    def test_document_vision_labels_reencoded_jpeg(self) -> None:
        enhancer, _engine = _enhancer({"page-1.png": JPEG_BYTES})

        plan = enhancer.prepare_vision_plan("text", [Path("page-1.png")], "doc.pdf")

        assert _image_url(plan.call.messages).startswith("data:image/jpeg;base64,")

    async def test_screenshot_labels_reencoded_jpeg(self) -> None:
        enhancer, engine = _enhancer({"page.full.png": JPEG_BYTES})

        with pytest.raises(RuntimeError, match="stop"):
            await enhancer.extract_from_screenshot(
                Path("page.full.png"), "https://example.com"
            )

        call = engine.complete_structured.call_args.args[0]
        assert _image_url(call.messages).startswith("data:image/jpeg;base64,")


class TestEmptyImageAnswersAreNotCached:
    def test_finalize_skips_blank_caption(self, tmp_path: Path) -> None:
        """Covers the live path and a collected --llm-batch answer alike."""
        processor = MockVisionProcessor()
        plan = processor.prepare_image_plan(tmp_path / "a.png")

        result = processor.finalize_image_plan(
            plan, ImageAnalysisResult(caption="  ", description="")
        )

        assert result.caption == ""
        processor._persistent_cache.set.assert_not_called()

    def test_finalize_still_caches_a_real_caption(self, tmp_path: Path) -> None:
        processor = MockVisionProcessor()
        plan = processor.prepare_image_plan(tmp_path / "a.png")

        processor.finalize_image_plan(
            plan, ImageAnalysisResult(caption="A chart", description="")
        )

        processor._persistent_cache.set.assert_called_once()

    async def test_blank_structured_and_two_call_answers_fail_the_image(
        self, tmp_path: Path
    ) -> None:
        processor = MockVisionProcessor()
        image = tmp_path / "a.png"
        image.write_bytes(PNG_BYTES)
        calls: list[str] = []

        async def blank_text(model: str, messages: Any, context: str = "") -> Any:
            calls.append(model)
            return LLMResponse(
                content="", model="m", input_tokens=1, output_tokens=0, cost_usd=0.0
            )

        processor._call_llm = blank_text  # type: ignore[method-assign]
        blank = ImageAnalysisResult(caption="", description="")
        with (
            patch.object(
                processor._engine,
                "complete_structured",
                new=AsyncMock(return_value=(blank, MagicMock(usage=None))),
            ),
            pytest.raises(EmptyLLMResponseError),
        ):
            await processor.analyze_image(image)

        # The structured answer fell back to the two-call path, which stopped
        # after the empty caption instead of paying for a description
        assert calls == ["default"]
        processor._persistent_cache.set.assert_not_called()

    async def test_batch_blank_caption_counts_as_failed(self, tmp_path: Path) -> None:
        processor = MockVisionProcessor()
        images = []
        for name in ("a.png", "b.png"):
            image = tmp_path / name
            image.write_bytes(PNG_BYTES + name.encode())
            images.append(image)

        response = BatchImageAnalysisResult(
            images=[
                SingleImageResult(image_index=1, caption="", description=""),
                SingleImageResult(image_index=2, caption="Chart", description="d"),
            ]
        )
        raw = MagicMock(model="test/model", usage=None)
        raw.choices = [MagicMock(finish_reason="stop")]
        with patch(
            "markitai.llm.vision.run_structured_ladder",
            new=AsyncMock(return_value=(response, raw)),
        ):
            results = await processor.analyze_batch(images, context="doc")

        assert results[0].failed is True
        assert results[1].failed is False
        cached = [
            call.args[2] for call in processor._persistent_cache.set.call_args_list
        ]
        # Only the real answer is cached
        assert len(cached) == 1
        assert cached[0]["caption"].strip()


class TestEmptyScreenshotExtractionIsNotCached:
    async def test_cache_if_rejects_blank_markdown(self) -> None:
        enhancer, engine = _enhancer({"page.full.jpg": JPEG_BYTES})

        with pytest.raises(RuntimeError, match="stop"):
            await enhancer.extract_from_screenshot(
                Path("page.full.jpg"), "https://example.com"
            )

        call = engine.complete_structured.call_args.args[0]
        frontmatter = Frontmatter.model_construct(description="d", tags=[])
        blank = DocumentProcessResult.model_construct(
            cleaned_markdown="  \n", frontmatter=frontmatter
        )
        real = DocumentProcessResult.model_construct(
            cleaned_markdown="# Page", frontmatter=frontmatter
        )
        assert call.cache_if(blank) is False
        assert call.cache_if(real) is True

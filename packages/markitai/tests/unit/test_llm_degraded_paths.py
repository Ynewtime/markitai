"""Regression tests: a degraded LLM result fails the item, and costs nothing extra.

The contract (website/guide/cli.md, ``--llm``): a refusal or a result that
fell back to unenhanced text fails the item. These tests pin the paths that
used to report such a result as a success, the paid calls whose answers were
thrown away, and the image-analysis placeholders that overwrote alt text:

- chunked documents over the request budget, and the per-chunk cleanup check
  that rejected a chunk of pure boilerplate cleaned away;
- ``clean_document_pure`` / ``enhance_document_with_vision`` /
  ``prepare_vision_plan`` / ``finalize_document_plan`` returning the input;
- ``enhance_document_complete`` paying for a vision cleaner call after the
  combined call failed, and sending every batch after an invalid key;
- ``analyze_images_batch`` placeholders ("Image N" / "Analysis failed")
  written as alt text and into images.json, without an item warning.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from markitai.config import LLMConfig, MarkitaiConfig, PromptsConfig
from markitai.constants import DEFAULT_MAX_CONTENT_CHARS
from markitai.llm import content as content_utils
from markitai.llm.document import DocumentEnhancer
from markitai.llm.engine import (
    LLMEnhancementDegradedError,
    LLMRequestBudgetExceededError,
    RequestBudget,
)
from markitai.llm.types import (
    DocumentProcessResult,
    EnhancedDocumentResult,
    Frontmatter,
    ImageAnalysis,
    LLMResponse,
)
from markitai.prompts import PromptManager
from markitai.providers.errors import ProviderError

REFUSAL = "I'm sorry, but I can't help with that request."


class _FakeEngine:
    """Engine double: canned answers, counts every call it is asked to make."""

    def __init__(
        self,
        *,
        budget_limit: int = 0,
        structured: Any = None,
        text: str = "",
    ) -> None:
        self.memory_cache = MagicMock()
        self.memory_cache.get.return_value = None
        self.persistent_cache = MagicMock()
        self.persistent_cache.get.return_value = None
        self.semaphore = asyncio.Semaphore(4)
        self.request_budget = RequestBudget(limit=budget_limit)
        self.structured = structured
        self.text = text
        self.structured_calls = 0
        self.text_calls = 0
        self.cached: set[str] = set()

    def record_cache_hit(self) -> None:
        pass

    def record_cache_miss(self) -> None:
        pass

    def try_cached(self, call: Any) -> Any:
        if call.cache_content in self.cached:
            return DocumentProcessResult(
                cleaned_markdown=call.cache_content,
                frontmatter=Frontmatter(description="d", tags=["t"]),
            )
        return None

    async def complete_structured(self, call: Any) -> tuple[Any, Any]:
        self.structured_calls += 1
        result = self.structured(call) if callable(self.structured) else None
        if call.validate is not None:
            result = call.validate(result)
        return result, None

    async def complete_text(self, **_kwargs: Any) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(
            content=self.text,
            model="fake/model",
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.0,
        )


def _enhancer(engine: _FakeEngine) -> DocumentEnhancer:
    return DocumentEnhancer(
        engine=engine,  # type: ignore[arg-type]
        prompt_manager=PromptManager(PromptsConfig()),
        config=LLMConfig(),
        cache_model_scope="pool:test",
        vision_cache_model_scope="pool:test-vision",
        get_vision_router=lambda: MagicMock(),
        get_cached_image=lambda _path: (b"img", "aW1n"),
        get_next_call_index=lambda _context: 0,
    )


def _long_document(chunks: int) -> str:
    """Markdown that splits into exactly *chunks* calls."""
    paragraph = "Quarterly revenue grew in every region we track. " * 40
    per_chunk = DEFAULT_MAX_CONTENT_CHARS // (len(paragraph) + 2)
    blocks = [f"{paragraph}{i}" for i in range(per_chunk * chunks)]
    text = "\n\n".join(blocks)
    assert len(
        content_utils.split_markdown_chunks(text, DEFAULT_MAX_CONTENT_CHARS)
    ) == (chunks)
    return text


def _echo(call: Any) -> DocumentProcessResult:
    """A faithful cleanup: the chunk comes back unchanged."""
    return DocumentProcessResult(
        cleaned_markdown=call.cache_content,
        frontmatter=Frontmatter(description="d", tags=["t"]),
    )


# =============================================================================
# 3. Chunked documents: budget pre-check and the per-chunk cleanup check
# =============================================================================


class TestChunkBudgetPrecheck:
    async def test_document_with_more_chunks_than_budget_sends_nothing(self) -> None:
        engine = _FakeEngine(budget_limit=5, structured=_echo)
        enhancer = _enhancer(engine)

        with pytest.raises(LLMEnhancementDegradedError) as exc_info:
            await enhancer.process_document(_long_document(6), "big.csv")

        assert engine.structured_calls == 0
        message = str(exc_info.value)
        assert "6 chunks need about 8 requests" in message
        assert "max_requests_per_document" in message

    async def test_retry_reserve_is_part_of_the_check(self) -> None:
        # 5 chunks + ceil(20%) = 6 requests: fits a budget of 6, not of 5
        document = _long_document(5)

        tight = _FakeEngine(budget_limit=5, structured=_echo)
        with pytest.raises(LLMEnhancementDegradedError):
            await _enhancer(tight).process_document(document, "doc.md")
        assert tight.structured_calls == 0

        roomy = _FakeEngine(budget_limit=6, structured=_echo)
        cleaned, _ = await _enhancer(roomy).process_document(document, "doc.md")
        assert roomy.structured_calls == 5
        assert "Quarterly revenue" in cleaned

    async def test_cached_chunks_cost_no_request(self) -> None:
        document = _long_document(6)
        engine = _FakeEngine(budget_limit=3, structured=_echo)
        enhancer = _enhancer(engine)
        plan = enhancer._prepare_document_plan(document, "doc.md")
        calls = [plan.call, *plan.chunk_calls]
        # A rerun after most chunks came back: only 2 still need a request
        engine.cached = {call.cache_content for call in calls[:4]}

        enhancer._check_chunk_budget(plan)

        engine.cached = {call.cache_content for call in calls[:3]}
        with pytest.raises(LLMRequestBudgetExceededError):
            enhancer._check_chunk_budget(plan)

    async def test_unlimited_budget_is_not_checked(self) -> None:
        engine = _FakeEngine(budget_limit=0, structured=_echo)
        cleaned, _ = await _enhancer(engine).process_document(
            _long_document(3), "doc.md"
        )
        assert engine.structured_calls == 3
        assert cleaned


class TestChunkCleanupCheck:
    NAV_CHUNK = "\n".join(
        f"- [Section {i}](https://example.com/section/{i}) | [Share](#share-{i})"
        for i in range(40)
    )

    def test_boilerplate_chunk_cleaned_away_passes_the_chunk_check(self) -> None:
        # The full check rejects it (nothing left), the chunk check does not
        assert content_utils.implausible_cleaning_reason(self.NAV_CHUNK, "") is not None
        assert (
            content_utils.implausible_chunk_cleaning_reason(self.NAV_CHUNK, "") is None
        )
        kept = "- [Section 3](https://example.com/section/3)"
        assert (
            content_utils.implausible_chunk_cleaning_reason(self.NAV_CHUNK, kept)
            is None
        )

    def test_refusal_still_fails_the_chunk_check(self) -> None:
        reason = content_utils.implausible_chunk_cleaning_reason(
            self.NAV_CHUNK, REFUSAL
        )
        assert reason is not None
        assert "comes from the input" in reason

    def test_chunk_call_validates_leniently(self) -> None:
        enhancer = _enhancer(_FakeEngine())
        emptied = DocumentProcessResult(
            cleaned_markdown="", frontmatter=Frontmatter(description="d", tags=["t"])
        )

        chunk_call = enhancer._build_document_call(self.NAV_CHUNK, "doc", chunked=True)
        assert chunk_call.validate is not None
        assert chunk_call.validate(emptied) is emptied

        whole_call = enhancer._build_document_call(self.NAV_CHUNK, "doc")
        assert whole_call.validate is not None
        with pytest.raises(ValueError, match="not a cleanup"):
            whole_call.validate(emptied)

    async def test_trailing_boilerplate_chunk_does_not_fail_the_document(
        self,
    ) -> None:
        document = _long_document(2) + "\n\n" + self.NAV_CHUNK
        plan = _enhancer(_FakeEngine())._prepare_document_plan(document, "page.html")
        # The navigation spills into a chunk of its own
        assert plan.chunk_calls[-1].cache_content == self.NAV_CHUNK

        def answer(call: Any) -> DocumentProcessResult:
            if call.cache_content == self.NAV_CHUNK:
                # The model strips the navigation-only chunk entirely
                return DocumentProcessResult(
                    cleaned_markdown="",
                    frontmatter=Frontmatter(description="d", tags=["t"]),
                )
            return _echo(call)

        engine = _FakeEngine(structured=answer)
        cleaned, _ = await _enhancer(engine).process_document(document, "page.html")

        assert engine.structured_calls == 3
        assert "Section 3" not in cleaned
        assert "Quarterly revenue" in cleaned

    async def test_merged_document_still_gets_the_full_check(self) -> None:
        def emptied(call: Any) -> DocumentProcessResult:
            return DocumentProcessResult(
                cleaned_markdown="",
                frontmatter=Frontmatter(description="d", tags=["t"]),
            )

        engine = _FakeEngine(structured=emptied)
        with pytest.raises(LLMEnhancementDegradedError, match="not a cleanup"):
            await _enhancer(engine).process_document(_long_document(2), "doc.md")


# =============================================================================
# 4. Refusals and fallbacks-to-input fail instead of passing as enhanced
# =============================================================================


LONG_TEXT = (
    "The committee reviewed the annual budget and approved the new library "
    "wing, the bike lanes on Main Street, and a pilot for late-night buses. "
) * 4


class TestPureCleaning:
    async def test_refusal_fails_the_item(self) -> None:
        engine = _FakeEngine(text=REFUSAL)
        with pytest.raises(LLMEnhancementDegradedError, match="not a cleanup") as exc:
            await _enhancer(engine).clean_document_pure(LONG_TEXT, "doc.md")
        assert exc.value.cleaned_markdown == LONG_TEXT

    async def test_real_cleanup_passes(self) -> None:
        engine = _FakeEngine(text=LONG_TEXT.strip())
        assert await _enhancer(engine).clean_document_pure(LONG_TEXT, "doc.md") == (
            LONG_TEXT.strip()
        )


class TestVisionCleaning:
    async def test_refusal_fails_and_is_not_cached(self, tmp_path: Path) -> None:
        engine = _FakeEngine(text=REFUSAL)
        page = tmp_path / "page-1.png"
        page.write_bytes(b"png")

        with pytest.raises(LLMEnhancementDegradedError, match="not a cleanup"):
            await _enhancer(engine).enhance_document_with_vision(
                LONG_TEXT, [page], "doc.pdf"
            )

        engine.persistent_cache.set.assert_not_called()

    def test_vision_plan_rejects_lost_page_boundaries(self, tmp_path: Path) -> None:
        text = "<!-- Page number: 1 -->\n\nAlpha\n\n<!-- Page number: 2 -->\n\nBeta"
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            text, [tmp_path / "p1.png", tmp_path / "p2.png"], "doc.pdf"
        )
        answer = EnhancedDocumentResult(
            cleaned_markdown="Alpha\n\nBeta",
            frontmatter=Frontmatter(description="d", tags=["t"]),
        )
        assert plan.call.validate is not None
        with pytest.raises(ValueError, match="page/slide boundaries"):
            plan.call.validate(answer)

    def test_vision_plan_rejects_a_refusal(self, tmp_path: Path) -> None:
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            LONG_TEXT, [tmp_path / "p1.png"], "doc.pdf"
        )
        answer = EnhancedDocumentResult(
            cleaned_markdown=REFUSAL,
            frontmatter=Frontmatter(description="d", tags=["t"]),
        )
        assert plan.call.validate is not None
        with pytest.raises(ValueError, match="not a cleanup"):
            plan.call.validate(answer)

    def test_vision_plan_accepts_a_repaired_text_layer(self, tmp_path: Path) -> None:
        """The vision model may replace a garbled text layer with what the page
        says: only the length floor applies to vision answers."""
        garbled = "".join(chr(0x2500 + (i % 90)) for i in range(400))
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            garbled, [tmp_path / "p1.png"], "doc.pdf"
        )
        answer = EnhancedDocumentResult(
            cleaned_markdown=LONG_TEXT,
            frontmatter=Frontmatter(description="d", tags=["t"]),
        )
        assert plan.call.validate is not None
        assert plan.call.validate(answer).cleaned_markdown.strip() == LONG_TEXT.strip()


class TestDocumentPlanFallbacks:
    PAGED = (
        "<!-- Page number: 1 -->\n\n# One\n\nAlpha\n\n"
        "<!-- Page number: 2 -->\n\n# Two\n\nBeta"
    )

    def _lossy(self) -> DocumentProcessResult:
        return DocumentProcessResult(
            cleaned_markdown="# One\n\nAlpha\n\n# Two\n\nBeta",
            frontmatter=Frontmatter(description="d", tags=["t"]),
        )

    def test_call_validate_rejects_dropped_boundaries_before_caching(self) -> None:
        plan = _enhancer(_FakeEngine())._prepare_document_plan(self.PAGED, "a.pdf")
        assert plan.call.validate is not None
        with pytest.raises(ValueError, match="structural placeholders"):
            plan.call.validate(self._lossy())

    def test_call_validate_rejects_dropped_image(self) -> None:
        markdown = "![chart](.markitai/assets/a.png)\n\n" + LONG_TEXT
        plan = _enhancer(_FakeEngine())._prepare_document_plan(markdown, "a.md")
        answer = DocumentProcessResult(
            cleaned_markdown=LONG_TEXT,
            frontmatter=Frontmatter(description="d", tags=["t"]),
        )
        assert plan.call.validate is not None
        with pytest.raises(ValueError, match="structural placeholders"):
            plan.call.validate(answer)

    def test_strict_finalize_fails_a_legacy_cached_fallback(self) -> None:
        enhancer = _enhancer(_FakeEngine())
        plan = enhancer._prepare_document_plan(self.PAGED, "a.pdf")

        with pytest.raises(LLMEnhancementDegradedError) as exc_info:
            enhancer.finalize_document_plan(plan, self._lossy(), strict=True)
        assert exc_info.value.cleaned_markdown == self.PAGED

        # The Batch API collector keeps the lenient default
        cleaned, _ = enhancer.finalize_document_plan(plan, self._lossy())
        assert cleaned.strip() == self.PAGED.strip()

    async def test_process_document_fails_on_a_legacy_cached_fallback(self) -> None:
        """A cache hit skips validate: the live path still must not succeed."""
        engine = _FakeEngine()
        enhancer = _enhancer(engine)

        async def cached_hit(call: Any) -> tuple[Any, Any]:
            return self._lossy(), None

        engine.complete_structured = cached_hit  # type: ignore[method-assign]
        with pytest.raises(LLMEnhancementDegradedError, match="boundaries"):
            await enhancer.process_document(self.PAGED, "a.pdf")


# =============================================================================
# 5. No thrown-away paid calls in enhance_document_complete
# =============================================================================


def _auth_error() -> ProviderError:
    return ProviderError("invalid api key", provider="copilot", retryable=False)


class TestEnhanceDocumentComplete:
    def _pages(self, tmp_path: Path, count: int) -> list[Path]:
        return [tmp_path / f"doc.page{i:04d}.png" for i in range(1, count + 1)]

    def _paged_text(self, count: int) -> str:
        return "\n\n".join(
            f"<!-- Page number: {i} -->\n\nPage {i} body." for i in range(1, count + 1)
        )

    async def test_single_batch_failure_pays_for_no_cleaner_call(
        self, tmp_path: Path
    ) -> None:
        enhancer = _enhancer(_FakeEngine())
        enhancer._enhance_with_frontmatter = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError("bad json")
        )
        enhancer.enhance_document_with_vision = AsyncMock()  # type: ignore[method-assign]
        text = self._paged_text(2)

        with pytest.raises(LLMEnhancementDegradedError) as exc_info:
            await enhancer.enhance_document_complete(
                text, self._pages(tmp_path, 2), "doc.pdf"
            )

        enhancer.enhance_document_with_vision.assert_not_called()
        assert exc_info.value.cleaned_markdown == text
        assert "llm_enhanced: false" in exc_info.value.frontmatter

    async def test_first_batch_failure_pays_for_no_cleaner_call(
        self, tmp_path: Path
    ) -> None:
        enhancer = _enhancer(_FakeEngine())
        enhancer._enhance_with_frontmatter = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError("bad json")
        )
        enhancer.enhance_document_with_vision = AsyncMock(  # type: ignore[method-assign]
            return_value="cleaned"
        )

        with pytest.raises(LLMEnhancementDegradedError):
            await enhancer.enhance_document_complete(
                self._paged_text(4),
                self._pages(tmp_path, 4),
                "doc.pdf",
                max_pages_per_batch=2,
            )

        # Only the second batch's own cleaning ran, not a retry of the first
        assert enhancer.enhance_document_with_vision.await_count == 1

    async def test_invalid_key_on_first_batch_sends_no_other_batch(
        self, tmp_path: Path
    ) -> None:
        enhancer = _enhancer(_FakeEngine())
        enhancer._enhance_with_frontmatter = AsyncMock(  # type: ignore[method-assign]
            side_effect=_auth_error()
        )
        enhancer.enhance_document_with_vision = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(LLMEnhancementDegradedError, match="invalid api key"):
            await enhancer.enhance_document_complete(
                self._paged_text(6),
                self._pages(tmp_path, 6),
                "doc.pdf",
                max_pages_per_batch=2,
            )

        enhancer.enhance_document_with_vision.assert_not_called()

    async def test_invalid_key_on_a_later_batch_stops_the_queued_ones(
        self, tmp_path: Path
    ) -> None:
        enhancer = _enhancer(_FakeEngine())
        enhancer._enhance_with_frontmatter = AsyncMock(  # type: ignore[method-assign]
            return_value=("batch 1", "title: doc")
        )
        release = asyncio.Event()
        cancelled: list[int] = []

        async def batch(text: str, _images: list[Path], context: str = "") -> str:
            if "Page 3 body" in text:
                raise _auth_error()
            try:
                await release.wait()  # still queued behind the semaphore
            except asyncio.CancelledError:
                cancelled.append(1)
                raise
            return text

        enhancer.enhance_document_with_vision = batch  # type: ignore[method-assign]

        with pytest.raises(LLMEnhancementDegradedError, match="invalid api key"):
            await asyncio.wait_for(
                enhancer.enhance_document_complete(
                    self._paged_text(8),
                    self._pages(tmp_path, 8),
                    "doc.pdf",
                    max_pages_per_batch=2,
                ),
                timeout=5,
            )

        assert len(cancelled) == 2

    async def test_later_batch_that_degrades_fails_the_document(
        self, tmp_path: Path
    ) -> None:
        enhancer = _enhancer(_FakeEngine())
        enhancer._enhance_with_frontmatter = AsyncMock(  # type: ignore[method-assign]
            return_value=("batch 1", "title: doc")
        )
        enhancer.enhance_document_with_vision = AsyncMock(  # type: ignore[method-assign]
            side_effect=LLMEnhancementDegradedError("empty answer")
        )

        with pytest.raises(LLMEnhancementDegradedError, match="empty answer"):
            await enhancer.enhance_document_complete(
                self._paged_text(4),
                self._pages(tmp_path, 4),
                "doc.pdf",
                max_pages_per_batch=2,
            )


# =============================================================================
# 6/7. Failed image analyses: no alt overwrite, no images.json entry, a warning
# =============================================================================


GOOD = ImageAnalysis(caption="A red logo", description="Logo.", extracted_text="")
BATCH_FAILED = ImageAnalysis(caption="Image 1", description="Analysis failed")
FLAGGED = ImageAnalysis(caption="Image", description="whatever", failed=True)


class TestFailedAnalysisRecognition:
    def test_every_placeholder_shape_counts_as_failed(self) -> None:
        from markitai.workflow.helpers import (
            image_analysis_failed,
            is_failed_image_entry,
        )

        assert image_analysis_failed(None)
        assert image_analysis_failed(FLAGGED)
        assert image_analysis_failed(BATCH_FAILED)
        assert not image_analysis_failed(GOOD)
        assert is_failed_image_entry({"desc": "Analysis failed"})
        assert is_failed_image_entry({"desc": "Image analysis failed"})
        assert not is_failed_image_entry({"desc": "Logo."})

    async def test_batch_failure_placeholders_are_flagged(self, tmp_path: Path) -> None:
        from markitai.llm import LLMProcessor

        config = MarkitaiConfig()
        processor = LLMProcessor(config.llm, config.prompts, no_cache=True)
        processor.vision.analyze_batch = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("boom")
        )
        images = [tmp_path / "a.png", tmp_path / "b.png"]

        results = await processor.analyze_images_batch(images, context="x")

        assert [r.failed for r in results] == [True, True]


def _url_setup(tmp_path: Path) -> tuple[MarkitaiConfig, Path, list[Path]]:
    cfg = MarkitaiConfig()
    cfg.image.alt_enabled = True
    cfg.image.desc_enabled = True
    assets = tmp_path / ".markitai" / "assets"
    assets.mkdir(parents=True)
    images = [assets / "chart.png", assets / "logo.png"]
    for image in images:
        image.write_bytes(b"png")
    llm_md = tmp_path / "page.llm.md"
    llm_md.write_text(
        "# Page\n\n![Quarterly revenue](.markitai/assets/chart.png)\n\n"
        "![](.markitai/assets/logo.png)\n",
        encoding="utf-8",
    )
    return cfg, llm_md, images


class TestUrlImageStage:
    async def test_failed_analysis_keeps_alt_and_is_not_recorded(
        self, tmp_path: Path
    ) -> None:
        from markitai.workflow.url import _analyze_url_images_stage

        cfg, llm_md, images = _url_setup(tmp_path)
        proc = MagicMock()
        proc.analyze_images_batch = AsyncMock(return_value=[BATCH_FAILED, GOOD])
        proc.get_context_cost.return_value = 0.0
        proc.get_context_usage.return_value = {}

        _cost, _usage, warnings = await _analyze_url_images_stage(
            cfg, tmp_path, llm_md, images, proc, "https://example.com"
        )

        text = llm_md.read_text(encoding="utf-8")
        assert "![Quarterly revenue](.markitai/assets/chart.png)" in text
        assert "![A red logo](.markitai/assets/logo.png)" in text
        data = json.loads(
            (tmp_path / ".markitai" / "assets" / "images.json").read_text("utf-8")
        )
        assert "Analysis failed" not in json.dumps(data)
        assert warnings == [
            "image analysis failed for chart.png; its original alt text was kept"
        ]


class TestCliImageAnalysis:
    async def test_failed_placeholder_keeps_alt_and_warns(self, tmp_path: Path) -> None:
        from markitai.cli.processors.llm import analyze_images_with_llm

        cfg, llm_md, images = _url_setup(tmp_path)
        processor = MagicMock()
        processor.analyze_images_batch = AsyncMock(return_value=[FLAGGED, GOOD])
        processor.get_context_cost.return_value = 0.0
        processor.get_context_usage.return_value = {}
        markdown = llm_md.read_text(encoding="utf-8")

        updated, _cost, _usage, result = await analyze_images_with_llm(
            images,
            markdown,
            tmp_path / "page.md",
            cfg,
            processor=processor,
        )

        assert "![Quarterly revenue](.markitai/assets/chart.png)" in updated
        assert "![A red logo](.markitai/assets/logo.png)" in updated
        assert "Quarterly revenue" in llm_md.read_text(encoding="utf-8")
        assert result is not None
        assert [Path(a["asset"]).name for a in result.assets] == ["logo.png"]
        assert result.warnings == [
            "image analysis failed for chart.png; its original alt text was kept"
        ]


class TestImageWarningsReachTheResult:
    def _ctx(self, tmp_path: Path) -> Any:
        from markitai.converter.base import ConvertResult
        from markitai.workflow.core import ConversionContext

        config = MarkitaiConfig()
        config.llm.enabled = True
        config.image.alt_enabled = True
        output_dir = tmp_path / "out"
        assets = output_dir / ".markitai" / "assets"
        assets.mkdir(parents=True)
        (assets / "doc.pdf.0001.png").write_bytes(b"png")
        ctx = ConversionContext(
            input_path=tmp_path / "doc.pdf",
            output_dir=output_dir,
            config=config,
            shared_processor=MagicMock(),
        )
        ctx.output_file = output_dir / "doc.pdf.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Doc\n\n![](.markitai/assets/doc.pdf.0001.png)", metadata={}
        )
        return ctx

    async def _run(self, ctx: Any, image_outcome: Any) -> Any:
        from unittest.mock import patch

        from markitai.workflow.core import process_with_standard_llm

        async def document(*_args: Any, **_kwargs: Any) -> tuple:
            ctx.output_file.with_suffix(".llm.md").write_text("# Doc")
            return ctx.conversion_result.markdown, 0.0, {}

        with patch("markitai.workflow.single.SingleFileWorkflow") as workflow:
            workflow.return_value.process_document_with_llm = AsyncMock(
                side_effect=document
            )
            if isinstance(image_outcome, BaseException):
                workflow.return_value.analyze_images = AsyncMock(
                    side_effect=image_outcome
                )
            else:
                workflow.return_value.analyze_images = AsyncMock(
                    return_value=image_outcome
                )
            return await process_with_standard_llm(ctx)

    async def test_per_image_failures_become_item_warnings(
        self, tmp_path: Path
    ) -> None:
        from markitai.workflow.results import document_process_result
        from markitai.workflow.single import ImageAnalysisResult

        ctx = self._ctx(tmp_path)
        warning = (
            "image analysis failed for doc.pdf.0001.png; its original alt text was kept"
        )
        analysis = ImageAnalysisResult(
            source_file="doc.pdf", assets=[], warnings=[warning]
        )

        result = await self._run(ctx, ("# Doc", 0.0, {}, analysis))

        assert result.success
        processed = document_process_result(ctx, result)
        assert processed.success
        assert processed.warnings == [warning]

    async def test_image_stage_crash_becomes_an_item_warning(
        self, tmp_path: Path
    ) -> None:
        from markitai.workflow.results import document_process_result

        ctx = self._ctx(tmp_path)

        result = await self._run(ctx, RuntimeError("vision router down"))

        processed = document_process_result(ctx, result)
        assert processed.success
        assert len(processed.warnings) == 1
        assert "vision router down" in processed.warnings[0]


# =============================================================================
# VLM OCR (--ocr --llm) of scanned pages: the transcription is the output
# =============================================================================


class TestScannedPageTranscription:
    """A scanned page has no text layer, so its extracted section is only the
    page marker; the page drift guard used to revert any transcription over
    200 characters there, and the item completed with an empty body."""

    SCANNED = "<!-- Page number: 1 -->\n\n<!-- Page number: 2 -->\n"
    PAGE_1 = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
        "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate."
    )
    PAGE_2 = (
        "第一章 概述\n\n本文档用于测试中文文本提取、表格识别和光学字符识别。"
        "日本語のテキストも含まれています。한국어 문장도 포함되어 있습니다。"
        "第二章 数据\n\n| 项目 | 数量 | 单价 |\n|---|---|---|\n| 苹果 | 3 | 5.00 |\n"
        "| 香蕉 | 12 | 2.50 |\n| 橙子 | 7 | 4.20 |\n\n以上数据仅用于识别测试，并无实际意义。"
    )

    def _transcription(self) -> str:
        return (
            f"<!-- Page number: 1 -->\n\n{self.PAGE_1}\n\n"
            f"<!-- Page number: 2 -->\n\n{self.PAGE_2}"
        )

    def test_vision_plan_keeps_the_transcription(self, tmp_path: Path) -> None:
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            self.SCANNED, [tmp_path / "p1.png", tmp_path / "p2.png"], "scan.pdf"
        )
        _, mapping = content_utils.protect_content(self.SCANNED)
        answer = self._transcription()
        for placeholder, original in mapping.items():
            answer = answer.replace(original, placeholder)
        assert plan.call.validate is not None

        result = plan.call.validate(
            EnhancedDocumentResult(
                cleaned_markdown=answer,
                frontmatter=Frontmatter(description="d", tags=["t"]),
            )
        )

        assert self.PAGE_1.strip() in result.cleaned_markdown
        assert self.PAGE_2 in result.cleaned_markdown

    async def test_vision_cleanup_keeps_the_transcription(self, tmp_path: Path) -> None:
        pages = [tmp_path / "p1.png", tmp_path / "p2.png"]
        _, mapping = content_utils.protect_content(self.SCANNED)
        answer = self._transcription()
        for placeholder, original in mapping.items():
            answer = answer.replace(original, placeholder)

        result = await _enhancer(_FakeEngine(text=answer)).enhance_document_with_vision(
            self.SCANNED, pages, "scan.pdf"
        )

        assert self.PAGE_1.strip() in result
        assert self.PAGE_2 in result

    def test_written_output_restabilization_keeps_the_transcription(
        self, tmp_path: Path
    ) -> None:
        """The workflow re-stabilizes the written .llm.md against the base
        .md; on the vision path that pass must keep the transcription too."""
        from markitai.workflow.core import stabilize_written_llm_output

        base_md = tmp_path / "scan.pdf.md"
        base_md.write_text(self.SCANNED, encoding="utf-8")
        llm_md = tmp_path / "scan.pdf.llm.md"
        llm_md.write_text(
            f"---\ntitle: scan\n---\n\n{self._transcription()}\n", encoding="utf-8"
        )
        ctx = MagicMock()
        ctx.output_file = base_md
        ctx.paged_stabilized = False
        ctx.base_written = True
        ctx.input_path.name = "scan.pdf"
        ctx.conversion_result.markdown = self.SCANNED

        stabilize_written_llm_output(ctx, _enhancer(_FakeEngine()), vision=True)

        written = llm_md.read_text(encoding="utf-8")
        assert self.PAGE_1.strip() in written
        assert self.PAGE_2 in written

    def test_text_only_cleanup_still_reverts_invented_pages(self) -> None:
        """Without page images the model cannot have read an empty page."""
        stabilized = _enhancer(_FakeEngine())._stabilize_paged_markdown(
            self.SCANNED, self._transcription(), "scan.pdf"
        )
        assert self.PAGE_1.strip() not in stabilized

    def test_vision_plan_fails_when_every_line_is_reverted(
        self, tmp_path: Path
    ) -> None:
        """An answer whose text the guards all revert is not a success with an
        empty body: the call fails (nothing is cached)."""
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            self.SCANNED, [tmp_path / "p1.png", tmp_path / "p2.png"], "scan.pdf"
        )
        _, mapping = content_utils.protect_content(self.SCANNED)
        # Pages swapped: marker drift restores the (empty) original layout
        answer = (
            f"<!-- Page number: 2 -->\n\n{self.PAGE_2}\n\n"
            f"<!-- Page number: 1 -->\n\n{self.PAGE_1}"
        )
        for placeholder, value in mapping.items():
            answer = answer.replace(value, placeholder)
        assert plan.call.validate is not None

        with pytest.raises(ValueError, match="discarded"):
            plan.call.validate(
                EnhancedDocumentResult(
                    cleaned_markdown=answer,
                    frontmatter=Frontmatter(description="d", tags=["t"]),
                )
            )

    def test_image_only_pages_keep_their_images_without_failing(
        self, tmp_path: Path
    ) -> None:
        """Reverted OCR prose on an image-only page leaves the image: not empty."""
        original = "<!-- Page number: 1 -->\n\n![](.markitai/assets/scan.png)\n"
        plan = _enhancer(_FakeEngine()).prepare_vision_plan(
            original, [tmp_path / "p1.png"], "scan.pdf"
        )
        _, mapping = content_utils.protect_content(original)
        answer = (
            "<!-- Page number: 1 -->\n\n![](.markitai/assets/scan.png)\n\n"
            f"{self.PAGE_1}"
        )
        for placeholder, value in mapping.items():
            answer = answer.replace(value, placeholder)
        assert plan.call.validate is not None

        result = plan.call.validate(
            EnhancedDocumentResult(
                cleaned_markdown=answer,
                frontmatter=Frontmatter(description="d", tags=["t"]),
            )
        )

        assert "![](.markitai/assets/scan.png)" in result.cleaned_markdown
        assert self.PAGE_1 not in result.cleaned_markdown

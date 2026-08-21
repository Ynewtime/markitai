"""Regression tests: an empty LLM answer must never become a cache entry.

``complete_text`` used to return empty content once the retries were
exhausted, and ``clean_markdown`` wrote that empty string straight into the
TTL-less SQLite cache — one bad run permanently blanked a document until the
user cleared the cache by hand. The structured path was protected by
Pydantic validation; the text path had nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import markitai.llm.engine as engine_module
from markitai.config import PromptsConfig
from markitai.llm.cache import ContentCache, PersistentCache
from markitai.llm.document import DocumentEnhancer
from markitai.llm.engine import EmptyLLMResponseError
from markitai.llm.types import LLMResponse
from markitai.prompts import PromptManager
from tests.unit.test_llm_engine import FakeRouter, Harness, make_model_response


@pytest.fixture
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out transport backoff so retry tests run instantly."""
    monkeypatch.setattr(engine_module, "DEFAULT_RETRY_BASE_DELAY", 0.0)


class TestCacheLayersRejectBlankValues:
    """Neither cache layer may persist an empty/whitespace-only result."""

    def test_memory_cache_skips_blank_value(self) -> None:
        cache = ContentCache()

        cache.set("cleaner@abc", "content", "   \n  ")

        assert cache.get("cleaner@abc", "content") is None
        assert cache.size == 0

    def test_memory_cache_keeps_real_value(self) -> None:
        cache = ContentCache()

        cache.set("cleaner@abc", "content", "# Real")

        assert cache.get("cleaner@abc", "content") == "# Real"

    def test_persistent_cache_skips_blank_value(self, tmp_path: Path) -> None:
        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        cache.set("cleaner@abc", "content", "")

        assert cache.get("cleaner@abc", "content") is None

    def test_persistent_cache_keeps_structured_value(self, tmp_path: Path) -> None:
        """Non-string payloads (dicts) are unaffected by the blank guard."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        cache.set("doc@abc", "content", {"cleaned_markdown": "", "tags": []})

        assert cache.get("doc@abc", "content") == {"cleaned_markdown": "", "tags": []}


class TestCompleteTextEmptyGuard:
    """Engine-level: the caller can tell "LLM returned nothing" apart."""

    @pytest.mark.asyncio
    async def test_require_content_raises_after_retries_exhausted(
        self, no_retry_delay: None
    ) -> None:
        harness = Harness(
            FakeRouter(
                [make_model_response("", prompt_tokens=200, completion_tokens=0)]
            )
        )

        with pytest.raises(EmptyLLMResponseError):
            await harness.engine.complete_text(
                model="default",
                messages=[{"role": "user", "content": "hi"}],
                call_id="doc:1",
                context="doc.md",
                max_retries=1,
                require_content=True,
            )

        # Retried before giving up, and the spent attempts stay accounted for
        assert len(harness.router.calls) == 2
        assert len(harness.track_calls) == 2

    @pytest.mark.asyncio
    async def test_whitespace_only_content_counts_as_empty(
        self, no_retry_delay: None
    ) -> None:
        harness = Harness(
            FakeRouter(
                [make_model_response("   \n\t ", prompt_tokens=10, completion_tokens=3)]
            )
        )

        with pytest.raises(EmptyLLMResponseError):
            await harness.engine.complete_text(
                model="default",
                messages=[{"role": "user", "content": "hi"}],
                call_id="doc:1",
                max_retries=0,
                require_content=True,
            )

    @pytest.mark.asyncio
    async def test_default_behaviour_unchanged(self, no_retry_delay: None) -> None:
        """Without require_content the empty response is still returned."""
        harness = Harness(
            FakeRouter([make_model_response("", prompt_tokens=10, completion_tokens=0)])
        )

        response = await harness.engine.complete_text(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            call_id="doc:1",
            max_retries=0,
        )

        assert response.content == ""


class _EmptyThenContentEngine:
    """Engine double: first text call comes back empty, then succeeds."""

    def __init__(self, memory_cache: Any, persistent_cache: Any) -> None:
        self.memory_cache = memory_cache
        self.persistent_cache = persistent_cache
        self.semaphore = asyncio.Semaphore(2)
        self.calls = 0
        self.hits = 0
        self.misses = 0

    def record_cache_hit(self) -> None:
        self.hits += 1

    def record_cache_miss(self) -> None:
        self.misses += 1

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str = "",
        context: str = "",
        max_retries: int = 0,
        router: Any = None,
        require_content: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            if require_content:
                raise EmptyLLMResponseError(f"[{call_id}] empty response")
            content = ""
        else:
            content = "# Cleaned\n\nReal output."
        return LLMResponse(
            content=content,
            model="fake/model",
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.0,
        )


def _make_enhancer(memory_cache: Any, persistent_cache: Any) -> DocumentEnhancer:
    return DocumentEnhancer(
        engine=_EmptyThenContentEngine(memory_cache, persistent_cache),  # type: ignore[arg-type]
        prompt_manager=PromptManager(PromptsConfig()),
        config=MagicMock(),
        cache_model_scope="pool:test",
        vision_cache_model_scope="pool:test-vision",
        get_vision_router=lambda: MagicMock(),
        get_cached_image=lambda _path: (b"img", "aW1n"),
        get_next_call_index=lambda _context: 0,
    )


class TestCleanMarkdownEmptyResponse:
    """Document text path: an empty answer must not poison the cache."""

    @pytest.mark.asyncio
    async def test_empty_response_is_not_cached_and_retries_next_run(
        self, tmp_path: Path
    ) -> None:
        memory = ContentCache()
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        enhancer = _make_enhancer(memory, persistent)
        content = "# Doc\n\nOriginal body."

        # Empty LLM answer: the original content survives untouched...
        first = await enhancer.clean_markdown(content, "doc.md")
        assert first == content

        # ...and neither layer holds an entry for it
        assert memory.size == 0
        assert persistent.stats()["cache"]["count"] == 0

        # The next run therefore issues a fresh request and caches its result
        second = await enhancer.clean_markdown(content, "doc.md")
        assert second == "# Cleaned\n\nReal output."
        assert enhancer._engine.calls == 2  # type: ignore[attr-defined]
        assert persistent.stats()["cache"]["count"] == 1

    @pytest.mark.asyncio
    async def test_enhance_document_with_vision_keeps_extracted_text_on_empty(
        self, tmp_path: Path
    ) -> None:
        memory = ContentCache()
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        enhancer = _make_enhancer(memory, persistent)
        page = tmp_path / "page-1.png"
        page.write_bytes(b"png")
        extracted = "# Doc\n\nExtracted body."

        result = await enhancer.enhance_document_with_vision(
            extracted, [page], "doc.pdf"
        )

        assert result == extracted
        assert persistent.stats()["cache"]["count"] == 0


class TestCleanDocumentPureEmptyResponse:
    """Pure mode writes the answer straight to disk — never a blank file.

    ``clean_document_pure`` caches nothing, so an empty answer cannot poison
    a cache; it goes somewhere worse, straight into the user's ``.llm.md``.
    """

    @pytest.mark.asyncio
    async def test_empty_response_returns_the_original_markdown(
        self, tmp_path: Path
    ) -> None:
        enhancer = _make_enhancer(
            ContentCache(), PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        )
        content = "# Doc\n\nOriginal body."

        assert await enhancer.clean_document_pure(content, "doc.md") == content
        # ...and the failure is not sticky: the next run calls the model again
        assert (
            await enhancer.clean_document_pure(content, "doc.md")
            == "# Cleaned\n\nReal output."
        )

    @pytest.mark.asyncio
    async def test_single_file_workflow_writes_original_not_empty_file(
        self, tmp_path: Path
    ) -> None:
        from markitai.config import MarkitaiConfig
        from markitai.workflow.single import SingleFileWorkflow

        enhancer = _make_enhancer(
            ContentCache(), PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        )
        processor = MagicMock()
        processor.clean_document_pure = enhancer.clean_document_pure
        processor.get_context_cost = MagicMock(return_value=0.0)
        processor.get_context_usage = MagicMock(return_value={})

        workflow = SingleFileWorkflow(MarkitaiConfig(), processor=processor)
        output_file = tmp_path / "doc.md"
        content = "# Doc\n\nOriginal body."

        await workflow.process_document_pure(content, "doc.md", output_file)

        assert (tmp_path / "doc.llm.md").read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_cli_pure_path_writes_original_not_empty_file(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.llm import process_with_llm
        from markitai.config import MarkitaiConfig

        enhancer = _make_enhancer(
            ContentCache(), PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        )
        processor = MagicMock()
        processor.clean_document_pure = enhancer.clean_document_pure
        processor.get_context_cost = MagicMock(return_value=0.0)
        processor.get_context_usage = MagicMock(return_value={})

        cfg = MarkitaiConfig()
        cfg.llm.pure = True
        output_file = tmp_path / "doc.md"
        content = "# Doc\n\nOriginal body."

        await process_with_llm(content, "doc.md", cfg, output_file, processor=processor)

        assert (tmp_path / "doc.llm.md").read_text(encoding="utf-8") == content

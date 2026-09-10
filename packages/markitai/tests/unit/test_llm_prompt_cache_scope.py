"""Regression tests: prompt text must scope the LLM cache keys.

Historically the persistent cache key was built from a bare category name
("cleaner", "document_process", "image_analysis"), so the prompt template
text never reached the key. Improving a prompt — or pointing the config at
a custom one — left every old cache entry live and the new prompt never ran
for already-processed documents.

Every cache category now mixes a digest of its *resolved* prompt templates
(config path -> custom dir -> built-in) plus the in-code prompt fragments
into the key, so a prompt edit is a cache miss.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from markitai.config import PromptsConfig
from markitai.llm.cache import ContentCache, PersistentCache
from markitai.llm.document import DocumentEnhancer
from markitai.llm.types import (
    DocumentProcessResult,
    Frontmatter,
    ImageAnalysis,
    LLMResponse,
)
from markitai.llm.vision import VisionAnalyzer
from markitai.prompts import PromptManager


class FakeEngine:
    """Minimal ``LLMEngine`` stand-in for cache-key round trips."""

    def __init__(
        self,
        memory_cache: Any,
        persistent_cache: Any,
        *,
        text_content: str = "CLEANED",
    ) -> None:
        self.memory_cache = memory_cache
        self.persistent_cache = persistent_cache
        self.semaphore = asyncio.Semaphore(4)
        self.text_content = text_content
        self.text_calls: list[list[dict[str, Any]]] = []
        self.structured_calls: list[Any] = []
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
        **kwargs: Any,
    ) -> LLMResponse:
        self.text_calls.append(messages)
        return LLMResponse(
            content=self.text_content,
            model="fake/model",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0,
        )

    async def complete_structured(self, call: Any) -> tuple[Any, Any]:
        self.structured_calls.append(call)
        return (
            DocumentProcessResult(
                cleaned_markdown="cleaned",
                frontmatter=Frontmatter(description="d", tags=["t"]),
            ),
            None,
        )


def _make_enhancer(
    prompts_dir: Path,
    memory_cache: Any,
    persistent_cache: Any,
    *,
    text_content: str = "CLEANED",
) -> DocumentEnhancer:
    """Build a DocumentEnhancer on a real PromptManager + real cache layers."""
    return DocumentEnhancer(
        engine=FakeEngine(memory_cache, persistent_cache, text_content=text_content),  # type: ignore[arg-type]
        prompt_manager=PromptManager(PromptsConfig(dir=str(prompts_dir))),
        config=MagicMock(),
        cache_model_scope="pool:test",
        vision_cache_model_scope="pool:test-vision",
        get_vision_router=lambda: MagicMock(),
        get_cached_image=lambda _path: (b"img", "aW1n"),
        get_next_call_index=lambda _context: 0,
    )


def _make_analyzer(
    prompts_dir: Path,
    persistent_cache: Any,
) -> VisionAnalyzer:
    """Build a VisionAnalyzer on a real PromptManager + real persistent cache."""
    engine = MagicMock()
    engine.persistent_cache = persistent_cache
    config = MagicMock()
    config.concurrency = 2
    return VisionAnalyzer(
        engine=engine,
        prompt_manager=PromptManager(PromptsConfig(dir=str(prompts_dir))),
        config=config,
        vision_cache_model_scope="pool:test-vision",
        get_vision_router=lambda: MagicMock(),
        get_cached_image=lambda _path: (b"img-bytes", "aW1nLWJ5dGVz"),
        get_next_call_index=lambda _context: 0,
    )


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Empty custom prompt directory (every name falls back to built-in)."""
    directory = tmp_path / "prompts"
    directory.mkdir()
    return directory


class TestTemplateDigest:
    """PromptManager.template_digest is the primitive the keys are built on."""

    def test_digest_is_short_stable_hex(self) -> None:
        manager = PromptManager()
        digest = manager.template_digest("cleaner_system", "cleaner_user")

        assert len(digest) == 8
        assert all(char in "0123456789abcdef" for char in digest)
        assert digest == manager.template_digest("cleaner_system", "cleaner_user")

    def test_digest_is_stable_across_manager_instances(self) -> None:
        assert PromptManager().template_digest("cleaner_system") == (
            PromptManager().template_digest("cleaner_system")
        )

    def test_digest_differs_per_template_set(self) -> None:
        manager = PromptManager()

        assert manager.template_digest("cleaner_system") != manager.template_digest(
            "document_process_system"
        )
        assert manager.template_digest("cleaner_system") != manager.template_digest(
            "cleaner_system", "cleaner_user"
        )

    def test_custom_template_changes_digest(self, prompts_dir: Path) -> None:
        """A user override in the custom prompt dir must change the digest."""
        builtin_digest = PromptManager(
            PromptsConfig(dir=str(prompts_dir))
        ).template_digest("cleaner_system")

        (prompts_dir / "cleaner_system.md").write_text(
            "Custom cleaner rules {mode_rules}", encoding="utf-8"
        )
        custom_digest = PromptManager(
            PromptsConfig(dir=str(prompts_dir))
        ).template_digest("cleaner_system")

        assert custom_digest != builtin_digest

    def test_config_path_override_changes_digest(self, tmp_path: Path) -> None:
        """A config-specified template path must change the digest too."""
        override = tmp_path / "my_cleaner.md"
        override.write_text("Totally different rules {mode_rules}", encoding="utf-8")

        assert PromptManager(
            PromptsConfig(cleaner_system=str(override))
        ).template_digest("cleaner_system") != PromptManager().template_digest(
            "cleaner_system"
        )

    def test_extra_constants_participate(self) -> None:
        """In-code prompt fragments fold into the digest via ``extra``."""
        manager = PromptManager()

        base = manager.template_digest("cleaner_system")
        with_extra = manager.template_digest("cleaner_system", extra=("RULE A",))
        other_extra = manager.template_digest("cleaner_system", extra=("RULE B",))

        assert base != with_extra
        assert with_extra != other_extra

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt"):
            PromptManager().template_digest("not_a_prompt")

    def test_clear_cache_refreshes_digest(self, prompts_dir: Path) -> None:
        manager = PromptManager(PromptsConfig(dir=str(prompts_dir)))
        before = manager.template_digest("cleaner_system")

        (prompts_dir / "cleaner_system.md").write_text("v2 {mode_rules}", "utf-8")
        manager.clear_cache()

        assert manager.template_digest("cleaner_system") != before


class TestCleanerCacheScope:
    """clean_markdown (text path, both cache layers)."""

    @pytest.mark.asyncio
    async def test_template_edit_invalidates_both_cache_layers(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        (prompts_dir / "cleaner_system.md").write_text(
            "V1 cleaner rules {mode_rules}", encoding="utf-8"
        )
        memory = ContentCache()
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        content = "# Doc\n\nSome body text."

        first = _make_enhancer(
            prompts_dir, memory, persistent, text_content="CLEANED-V1"
        )
        assert await first.clean_markdown(content, "doc.md") == "CLEANED-V1"
        assert len(first._engine.text_calls) == 1  # type: ignore[attr-defined]

        # Unchanged templates: fully cached, no second LLM call
        second = _make_enhancer(
            prompts_dir, memory, persistent, text_content="CLEANED-V2"
        )
        assert await second.clean_markdown(content, "doc.md") == "CLEANED-V1"
        assert second._engine.text_calls == []  # type: ignore[attr-defined]

        # Edited template: both the memory and the SQLite entry must miss
        (prompts_dir / "cleaner_system.md").write_text(
            "V2 cleaner rules {mode_rules}", encoding="utf-8"
        )
        third = _make_enhancer(
            prompts_dir, memory, persistent, text_content="CLEANED-V2"
        )
        assert await third.clean_markdown(content, "doc.md") == "CLEANED-V2"
        assert len(third._engine.text_calls) == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_cache_key_carries_digest(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        memory = ContentCache()
        persistent = MagicMock()
        persistent.get.return_value = None
        enhancer = _make_enhancer(prompts_dir, memory, persistent)

        await enhancer.clean_markdown("# Doc\n\nBody.", "doc.md")

        key = persistent.set.call_args[0][0]
        assert key.startswith("cleaner@")
        assert len(key) > len("cleaner@")


class TestDocumentProcessCacheScope:
    """_run_document_call (structured path)."""

    @pytest.mark.asyncio
    async def test_template_edit_changes_key_and_misses_old_entry(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        memory = ContentCache()
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        markdown = "# Doc\n\nBody."

        first = _make_enhancer(prompts_dir, memory, persistent)
        await first._run_document_call(first._build_document_call(markdown, "doc.md"))
        old_key = first._engine.structured_calls[0].cache_key  # type: ignore[attr-defined]

        (prompts_dir / "document_process_system.md").write_text(
            "New processing rules for {source}", encoding="utf-8"
        )
        second = _make_enhancer(prompts_dir, memory, persistent)
        await second._run_document_call(second._build_document_call(markdown, "doc.md"))
        new_key = second._engine.structured_calls[0].cache_key  # type: ignore[attr-defined]

        assert old_key.startswith("document_process@")
        assert new_key.startswith("document_process@")
        assert new_key != old_key

        # An entry written under the old key must not answer the new key
        persistent.set(old_key, markdown, {"cleaned_markdown": "stale"})
        assert persistent.get(new_key, markdown) is None
        assert persistent.get(old_key, markdown) is not None

    @pytest.mark.asyncio
    async def test_identical_templates_keep_key_stable(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        memory = ContentCache()
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        markdown = "# Doc\n\nBody."

        keys = []
        for _ in range(2):
            enhancer = _make_enhancer(prompts_dir, memory, persistent)
            await enhancer._run_document_call(
                enhancer._build_document_call(markdown, "doc.md")
            )
            keys.append(enhancer._engine.structured_calls[0].cache_key)  # type: ignore[attr-defined]

        assert keys[0] == keys[1]


class TestVisionCacheScope:
    """analyze_image / analyze_images_batch (image_analysis category)."""

    @pytest.mark.asyncio
    async def test_template_edit_invalidates_image_analysis_cache(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        persistent = PersistentCache(global_dir=tmp_path / "cache", enabled=True)
        image = tmp_path / "figure.png"
        image.write_bytes(b"fake-png-bytes")

        calls: list[str] = []

        def fallback_factory(caption: str) -> Any:
            async def _fallback(*_args: Any, **_kwargs: Any) -> ImageAnalysis:
                calls.append(caption)
                return ImageAnalysis(
                    caption=caption,
                    description="A detailed english description of the figure.",
                    extracted_text="figure text",
                )

            return _fallback

        first = _make_analyzer(prompts_dir, persistent)
        first._analyze_image_with_fallback = fallback_factory("caption-v1")  # type: ignore[method-assign]
        result = await first.analyze_image(image, context="doc.pdf")
        assert result.caption == "caption-v1"
        assert calls == ["caption-v1"]

        # Unchanged templates -> cache hit, no new analysis
        second = _make_analyzer(prompts_dir, persistent)
        second._analyze_image_with_fallback = fallback_factory("caption-v2")  # type: ignore[method-assign]
        result = await second.analyze_image(image, context="doc.pdf")
        assert result.caption == "caption-v1"
        assert calls == ["caption-v1"]

        # Edited template -> the stale entry must not be reused
        (prompts_dir / "image_analysis_system.md").write_text(
            "New image analysis rules in {language}", encoding="utf-8"
        )
        third = _make_analyzer(prompts_dir, persistent)
        third._analyze_image_with_fallback = fallback_factory("caption-v2")  # type: ignore[method-assign]
        result = await third.analyze_image(image, context="doc.pdf")
        assert result.caption == "caption-v2"
        assert calls == ["caption-v1", "caption-v2"]

    @pytest.mark.asyncio
    async def test_single_and_batch_paths_share_one_key(
        self, tmp_path: Path, prompts_dir: Path
    ) -> None:
        """The batch path must still reuse entries written by analyze_image."""
        persistent = MagicMock()
        persistent.get.return_value = None
        image = tmp_path / "figure.png"
        image.write_bytes(b"fake-png-bytes")

        analyzer = _make_analyzer(prompts_dir, persistent)

        async def _fallback(*_args: Any, **_kwargs: Any) -> ImageAnalysis:
            return ImageAnalysis(
                caption="a caption",
                description="a description",
                extracted_text="text",
            )

        analyzer._analyze_image_with_fallback = _fallback  # type: ignore[method-assign]
        await analyzer.analyze_image(image, context="doc.pdf")
        single_key = persistent.get.call_args[0][0]

        persistent.get.reset_mock()
        await analyzer.analyze_images_batch([image], context="doc.pdf")
        batch_key = persistent.get.call_args[0][0]

        assert single_key.startswith("image_analysis@")
        assert single_key == batch_key

    def test_vision_content_key_has_no_handwritten_version_tag(self) -> None:
        """The hand-maintained "|vision:v2" tag is replaced by the digest."""
        from markitai.llm.vision import _vision_cache_content_key

        assert "vision:v2" not in _vision_cache_content_key("fingerprint")

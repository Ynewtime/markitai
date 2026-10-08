"""Tests for cache fingerprint collision resistance."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCacheFingerprint:
    """Test that cache fingerprints use robust hashing."""

    def test_different_documents_different_fingerprints(self):
        """Documents with same prefix but different content should have different fingerprints."""
        from markitai.llm.document import _compute_document_fingerprint

        # Two documents with identical first 1000 chars but different content
        prefix = "A" * 1000
        doc1 = prefix + " DOCUMENT ONE CONTENT"
        doc2 = prefix + " DOCUMENT TWO CONTENT"

        fp1 = _compute_document_fingerprint(doc1, ["page1"])
        fp2 = _compute_document_fingerprint(doc2, ["page1"])

        assert fp1 != fp2, (
            "Documents with same prefix should have different fingerprints"
        )

    def test_fingerprint_includes_page_info(self):
        """Fingerprint should incorporate page names."""
        from markitai.llm.document import _compute_document_fingerprint

        doc = "Same content"
        fp1 = _compute_document_fingerprint(doc, ["page1", "page2"])
        fp2 = _compute_document_fingerprint(doc, ["page1", "page2", "page3"])

        assert fp1 != fp2, "Different page counts should produce different fingerprints"

    def test_fingerprint_uses_sha256(self):
        """Fingerprint should use SHA256 for collision resistance."""
        from markitai.llm.document import _compute_document_fingerprint

        fp = _compute_document_fingerprint("test content", ["page1"])
        # SHA256 hex digest is 64 characters
        assert len(fp) == 64

    def test_fingerprint_deterministic(self):
        """Same input should always produce same fingerprint."""
        from markitai.llm.document import _compute_document_fingerprint

        fp1 = _compute_document_fingerprint("hello world", ["p1", "p2"])
        fp2 = _compute_document_fingerprint("hello world", ["p1", "p2"])
        assert fp1 == fp2

    def test_empty_content(self):
        """Empty content should produce valid fingerprint."""
        from markitai.llm.document import _compute_document_fingerprint

        fp = _compute_document_fingerprint("", [])
        assert len(fp) == 64

    def test_edit_past_50k_characters_changes_fingerprint(self):
        """Regression: only the first 50,000 characters were hashed, so a
        long document edited near its end was served the old LLM output."""
        from markitai.llm.document import _compute_document_fingerprint

        body = "x" * 60_000
        fp1 = _compute_document_fingerprint(body + " old ending", ["page1"])
        fp2 = _compute_document_fingerprint(body + " new ending", ["page1"])

        assert fp1 != fp2

    def test_image_bytes_change_fingerprint(self):
        """Same text and page names but different page images must miss."""
        from markitai.llm.document import _compute_document_fingerprint

        fp1 = _compute_document_fingerprint("text", ["page1"], [b"old image"])
        fp2 = _compute_document_fingerprint("text", ["page1"], [b"new image"])

        assert fp1 != fp2


class TestVisionCacheKeysIncludeImages:
    """The vision entry points key their cache on the image bytes they send."""

    @staticmethod
    def _enhancer(images: dict[str, bytes]):
        from markitai.llm.document import DocumentEnhancer

        engine = MagicMock()
        engine.persistent_cache.get.return_value = "cached"
        engine.complete_structured = AsyncMock(side_effect=RuntimeError("stop"))
        prompts = MagicMock()
        prompts.get_prompt.return_value = "prompt"

        def get_cached_image(path: Path) -> tuple[bytes, str]:
            return images[path.name], "b64"

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

    async def test_document_vision_cache_content_follows_page_bytes(self):
        seen = []
        for page_bytes in (b"render v1", b"render v2"):
            enhancer, engine = self._enhancer({"page-1.png": page_bytes})
            await enhancer.enhance_document_with_vision(
                "same text", [Path("page-1.png")], "doc.pdf"
            )
            seen.append(engine.persistent_cache.get.call_args.args[1])

        assert seen[0] != seen[1]

    def test_vision_plan_cache_content_follows_page_bytes(self):
        seen = []
        for page_bytes in (b"render v1", b"render v2"):
            enhancer, _engine = self._enhancer({"page-1.png": page_bytes})
            plan = enhancer.prepare_vision_plan(
                "same text", [Path("page-1.png")], "doc.pdf"
            )
            seen.append(plan.call.cache_content)

        assert seen[0] != seen[1]

    async def test_url_vision_cache_content_follows_screenshot_bytes(self):
        seen = []
        for shot in (b"screenshot v1", b"screenshot v2"):
            enhancer, engine = self._enhancer({"page.full.jpg": shot})
            with pytest.raises(RuntimeError, match="stop"):
                await enhancer.enhance_url_with_vision(
                    "same text", Path("page.full.jpg"), "https://example.com"
                )
            seen.append(engine.complete_structured.call_args.args[0].cache_content)

        assert seen[0] != seen[1]

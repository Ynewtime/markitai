"""Document processing mixin for LLMProcessor.

This module contains all document-related methods extracted from processor.py
to reduce file size and improve maintainability.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from markitai.constants import (
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_PAGES_PER_BATCH,
    SCREENSHOTS_REL_PATH,
)

# Mode-specific rules injected into cleaner_system prompt via {mode_rules}
STANDARD_MODE_RULES = """\
## Image Placeholder Preservation — CRITICAL
- The document may contain `__MARKITAI_IMG_N__` placeholders (where N is a number). These represent actual images.
- You MUST preserve **every** placeholder in its **exact original position**. Do not move, reorder, merge, or remove any placeholder.
- If a placeholder appears between two paragraphs, it must remain between those same paragraphs in your output.
- Failure to preserve all placeholders will cause your output to be rejected entirely."""

PURE_MODE_RULES = """\
## YAML Frontmatter Preservation — CRITICAL
- The document may start with a YAML frontmatter block delimited by `---` lines (e.g., `---\\ntitle: ...\\n---`).
- You MUST preserve the frontmatter block exactly as-is. Do not modify, reorder, or remove any fields.
- If no frontmatter is present, do not add one."""
from markitai.llm.content import (
    protect_image_positions as _shared_protect_image_positions,
)
from markitai.llm.content import (
    restore_image_positions as _shared_restore_image_positions,
)
from markitai.llm.degeneration import truncate_degenerate_tail
from markitai.llm.engine import LLMCall

# Moved to markitai.llm.engine (Phase 2.1); alias keeps the internal
# reference in process_document() working unchanged.
from markitai.llm.engine import (
    find_non_retryable_provider_error as _find_non_retryable_provider_error,
)
from markitai.llm.types import (
    DocumentProcessResult,
    EnhancedDocumentResult,
    Frontmatter,
)
from markitai.utils.mime import get_mime_type
from markitai.utils.text import format_error_message

# Pre-compiled regex patterns for _remove_uncommented_screenshots hot path
# Screenshot references (markitai-generated .pageNNNN patterns)
_SCREENSHOT_REF_RE = re.compile(
    r"^!\[(?:Page\s+\d+|[^\]]*)\]\(\.markitai/screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
    re.MULTILINE,
)
# Page heading image labels (legacy format)
_PAGE_HEADING_LABEL_RE = re.compile(
    r"^#{2,3}\s+Page\s+\d+\s+Image:\s*\n\s*\n",
    re.MULTILINE,
)
# Placeholder and label patterns (merged: [Page/Image N], __MARKITAI_*_LABEL_N__, __MARKITAI_SLIDE_N__)
_PLACEHOLDER_LABEL_RE = re.compile(
    r"^(?:\[(Page|Image)\s+\d+\]"
    r"|__MARKITAI_(?:PAGE|IMG)_LABEL_\d+__"
    r"|__MARKITAI_SLIDE_\d+__)\s*\n",
    re.MULTILINE,
)
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
_PAGE_SECTION_RE = re.compile(
    r"(<!-- Page images for reference -->)"
    r"((?:\s*<!-- !\[Page \d+\]\([^)]+\) -->)+)"
)
_PAGE_COMMENT_RE = re.compile(r"<!-- !\[Page \d+\]\([^)]+\) -->")
_PAGE_MARKER_CAPTURE_RE = re.compile(r"<!--\s*Page number:\s*(\d+)\s*-->")
_STRUCTURED_MARKER_CAPTURE_RE = re.compile(r"<!--\s*(Page|Slide) number:\s*(\d+)\s*-->")
_BOUNDARY_MARKER_EXTRACT_RE = re.compile(r"(?:Page|Slide)\s+number:\s*(\d+)")
_LEADING_RULES_RE = re.compile(r"\A(?:[ \t]*---[ \t]*\n+)+")
_RULE_BEFORE_REF_SECTION_RE = re.compile(
    r"\n[ \t]*---[ \t]*\n(?=(?:<!-- Page images for reference -->|<!-- Screenshot for reference -->))"
)
_TRAILING_RULES_RE = re.compile(r"(?:\n[ \t]*---[ \t]*)+\Z")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+")


def _compute_document_fingerprint(
    content: str,
    page_names: list[str],
) -> str:
    """Compute a collision-resistant fingerprint for document caching.

    Uses SHA256 over the full content (truncated at DEFAULT_CACHE_CONTENT_TRUNCATE
    chars for performance) plus page structure info, rather than just the first
    1000 chars which can collide for documents with identical prefixes.

    Args:
        content: Document text content
        page_names: List of page/section names

    Returns:
        SHA256 hex digest string (64 chars)
    """
    import hashlib

    from markitai.constants import DEFAULT_CACHE_CONTENT_TRUNCATE

    truncated = content[:DEFAULT_CACHE_CONTENT_TRUNCATE]
    fingerprint_input = f"{truncated}|pages:{','.join(page_names[:50])}"
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()


def _document_result_to_cache_value(result: DocumentProcessResult) -> dict[str, Any]:
    """Serialize a document result to the legacy cache value shape.

    The flat {cleaned_markdown, description, tags} dict predates LLMEngine;
    it is preserved verbatim so existing cache entries keep hitting.
    (frontmatter_yaml is intentionally NOT cached: it contains a timestamp.)
    """
    return {
        "cleaned_markdown": result.cleaned_markdown,
        "description": result.frontmatter.description,
        "tags": result.frontmatter.tags,
    }


def _document_result_from_cache_value(cached: dict[str, Any]) -> DocumentProcessResult:
    """Rebuild a document result from the legacy cache value shape.

    Uses model_construct() to bypass validation for cached data.
    """
    return DocumentProcessResult.model_construct(
        cleaned_markdown=cached.get("cleaned_markdown", ""),
        frontmatter=Frontmatter.model_construct(
            description=cached.get("description", ""),
            tags=cached.get("tags", []),
        ),
    )


def _strip_leaked_markdown_boundaries(content: str) -> str:
    """Remove stray body separators leaked by the LLM.

    This only strips leading/trailing ``---`` lines and separators directly before
    the page reference section. Internal horizontal rules inside the body remain.
    """
    stripped = content.lstrip()
    if stripped.startswith("---\n"):
        match = re.match(
            r"\A[ \t]*---[ \t]*\n(.*?)\n[ \t]*---[ \t]*(?:\n+|$)",
            stripped,
            flags=re.DOTALL,
        )
        if match is not None:
            try:
                parsed = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                parsed = None
            if isinstance(parsed, dict):
                stripped = stripped[match.end() :].lstrip("\n")

    stripped = _LEADING_RULES_RE.sub("", stripped)
    stripped = _RULE_BEFORE_REF_SECTION_RE.sub("\n", stripped)
    stripped = _TRAILING_RULES_RE.sub("", stripped.rstrip())
    return stripped.strip()


class DocumentMixin:
    """Document processing methods for LLMProcessor.

    This mixin provides all document-related functionality including:
    - Markdown cleaning
    - Frontmatter generation
    - Vision-enhanced document processing
    - Document output formatting

    Methods access parent class attributes via self, using TYPE_CHECKING
    to avoid circular imports.
    """

    # Type hints for mixin - these are provided by LLMProcessor
    if TYPE_CHECKING:
        from markitai.llm.engine import LLMEngine

        _cache: Any
        _persistent_cache: Any
        _cache_hits: int
        _cache_misses: int
        _prompt_manager: Any
        config: Any
        router: Any
        vision_router: Any
        semaphore: asyncio.Semaphore
        engine: LLMEngine

        def _call_llm(
            self,
            model: str,
            messages: list[dict[str, Any]],
            context: str = "",
        ) -> Any: ...

        def _get_next_call_index(self, context: str) -> int: ...

        def _get_cached_image(self, image_path: Path) -> tuple[bytes, str]: ...

        def _calculate_dynamic_max_tokens(
            self,
            messages: list[dict[str, Any]],
            target_model_id: str | None = None,
            router: Any | None = None,
        ) -> int: ...

        def _get_router_primary_model(self, router: Any) -> str: ...

        def _track_usage(
            self,
            model: str,
            input_tokens: int,
            output_tokens: int,
            cost: float,
            context: str,
        ) -> None: ...

        # Static method references from content module
        extract_protected_content: Any
        protect_content: Any
        unprotect_content: Any
        fix_malformed_image_refs: Any
        strip_prompt_echo: Any
        clean_frontmatter: Any
        smart_truncate: Any
        split_text_by_pages: Any

    async def clean_markdown(self, content: str, context: str = "") -> str:
        """
        Clean and optimize markdown content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

        The cache_key parameter identifies the cache category (e.g. "cleaner");
        PersistentCache internally combines it with a content hash for lookups.

        Args:
            content: Raw markdown content
            context: Context identifier for logging (e.g., filename)

        Returns:
            Cleaned markdown content
        """
        cache_key = "cleaner"

        # 1. Check in-memory cache first (fastest)
        cached = self._cache.get(cache_key, content)
        if cached is not None:
            self._cache_hits += 1
            return cached

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, content, context=context)
        if cached is not None:
            self._cache_hits += 1
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, content, cached)
            return cached

        self._cache_misses += 1

        # 3. Protect image positions before any LLM processing
        image_protected, image_mapping = self._protect_image_positions(content)

        # 4. Extract and protect content before LLM processing
        protected = self.extract_protected_content(image_protected)
        protected_content, mapping = self.protect_content(image_protected)

        # Use separated system/user prompts to prevent prompt leakage
        system_prompt = self._prompt_manager.get_prompt(
            "cleaner_system", mode_rules=STANDARD_MODE_RULES
        )
        user_prompt = self._prompt_manager.get_prompt(
            "cleaner_user", content=protected_content
        )

        response = await self._call_llm(  # type: ignore[attr-defined]
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            context=context,
        )

        fallback = self._fallback_if_boundary_placeholders_missing(
            response.content,
            content,
            mapping,
            context or "cleaner",
            "clean_markdown",
        )
        if fallback is not None:
            result = fallback
        else:
            # Restore protected content from placeholders, with fallback for removed items
            # Disable "append missing images at end" — image positions are
            # managed by image_mapping, not by unprotect_content fallback
            result = self.unprotect_content(
                response.content,
                mapping,
                protected,
                restore_missing_images_at_end=False,
            )
        result = self._stabilize_paged_markdown(content, result, context)
        # Restore image positions (or fall back to original if placeholders were lost)
        result = self._restore_images_or_fallback(
            result, content, image_mapping, context or "cleaner", "clean_markdown"
        )

        # Cache the result in both layers
        self._cache.set(cache_key, content, result)
        self._persistent_cache.set(cache_key, content, result, model="default")

        return result

    @staticmethod
    def _protect_image_positions(text: str) -> tuple[str, dict[str, str]]:
        """Replace image references with position markers to prevent LLM from moving them.

        Delegates to shared implementation in content.py, excluding screenshots
        which have their own protection mechanism in document processing.

        Args:
            text: Markdown text with image references

        Returns:
            Tuple of (text with markers, mapping of marker -> original image reference)
        """
        return _shared_protect_image_positions(text, exclude_screenshots=True)

    @staticmethod
    def _restore_image_positions(text: str, mapping: dict[str, str]) -> str:
        """Restore original image references from position markers.

        Delegates to shared implementation in content.py.

        Args:
            text: Text with position markers
            mapping: Mapping of marker -> original image reference

        Returns:
            Text with original image references restored
        """
        return _shared_restore_image_positions(text, mapping)

    @staticmethod
    def _restore_images_or_fallback(
        llm_output: str,
        original_markdown: str,
        image_mapping: dict[str, str],
        source: str,
        stage: str,
    ) -> str:
        """Restore image placeholders, falling back to original if any are missing.

        If any __MARKITAI_IMG_*__ placeholder was dropped by the LLM,
        structural correctness is prioritized: the entire original markdown
        is returned instead of attempting partial restoration.

        Args:
            llm_output: LLM output containing image placeholders
            original_markdown: Original markdown before image protection
            image_mapping: Mapping of placeholder -> original image reference
            source: Source identifier for logging
            stage: Processing stage for logging

        Returns:
            Text with images restored to their original positions
        """
        if not image_mapping:
            return llm_output

        missing = [p for p in image_mapping if p not in llm_output]
        if missing:
            logger.warning(
                f"[{source}] {stage} dropped {len(missing)}/{len(image_mapping)} "
                f"image placeholders; using original content to preserve structure"
            )
            return original_markdown

        return _shared_restore_image_positions(llm_output, image_mapping)

    @staticmethod
    def _split_paged_sections(text: str) -> list[tuple[str, str]]:
        """Split page-marked markdown into ordered sections."""
        matches = list(_STRUCTURED_MARKER_CAPTURE_RE.finditer(text))
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_kind = match.group(1).lower()
            section_num = match.group(2)
            sections.append((f"{section_kind}:{section_num}", text[start:end].strip()))
        return sections

    @staticmethod
    def _split_reference_suffix(text: str) -> tuple[str, str]:
        """Split page reference comments from the main markdown body."""
        marker_positions = [
            pos
            for pos in (
                text.find("<!-- Page images for reference -->"),
                text.find("<!-- Screenshot for reference -->"),
            )
            if pos != -1
        ]
        if not marker_positions:
            return text.rstrip(), ""

        split_at = min(marker_positions)
        return text[:split_at].rstrip(), text[split_at:].strip()

    @staticmethod
    def _is_suspicious_page_expansion(
        original_section: str,
        cleaned_section: str,
    ) -> bool:
        """Detect page sections that grew far beyond the original extraction."""
        original_body = _STRUCTURED_MARKER_CAPTURE_RE.sub(
            "", original_section, count=1
        ).strip()
        cleaned_body = _STRUCTURED_MARKER_CAPTURE_RE.sub(
            "", cleaned_section, count=1
        ).strip()

        original_len = len(original_body)
        cleaned_len = len(cleaned_body)

        if cleaned_len <= original_len:
            return False
        if original_len == 0:
            return cleaned_len > 200
        if original_len < 400:
            return cleaned_len > max(original_len * 2, original_len + 80)
        return cleaned_len > max(original_len * 2, original_len + 800)

    @staticmethod
    def _extract_semantic_section_text(section: str) -> str:
        """Extract non-structural text from a page or slide section."""
        body = _STRUCTURED_MARKER_CAPTURE_RE.sub("", section, count=1).strip()
        semantic_lines: list[str] = []

        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("<!--") and stripped.endswith("-->"):
                continue
            if _HEADING_LINE_RE.match(stripped):
                continue
            if stripped.startswith("![") or stripped.startswith("[!["):
                continue
            if stripped in {"---", "***", "___"}:
                continue
            semantic_lines.append(stripped)

        return "\n".join(semantic_lines)

    def _is_suspicious_section_body_loss(
        self,
        original_section: str,
        cleaned_section: str,
    ) -> bool:
        """Detect when a cleaned section drops the original body text entirely."""
        original_text = self._extract_semantic_section_text(original_section)
        cleaned_text = self._extract_semantic_section_text(cleaned_section)
        return bool(original_text) and not bool(cleaned_text)

    def _is_suspicious_image_only_text_injection(
        self,
        original_section: str,
        cleaned_section: str,
    ) -> bool:
        """Detect OCR text injected into sections that originally only had images."""
        original_body = _STRUCTURED_MARKER_CAPTURE_RE.sub(
            "", original_section, count=1
        ).strip()
        cleaned_text = self._extract_semantic_section_text(cleaned_section)
        original_text = self._extract_semantic_section_text(original_section)
        had_image = "!["
        return had_image in original_body and not original_text and bool(cleaned_text)

    def _stabilize_paged_markdown(
        self,
        original_markdown: str,
        cleaned_markdown: str,
        source: str,
    ) -> str:
        """Protect page-marked documents from LLM structural drift."""
        if (
            "<!-- Page number:" not in original_markdown
            and "<!-- Slide number:" not in original_markdown
        ):
            return _strip_leaked_markdown_boundaries(cleaned_markdown)

        original_body, original_suffix = self._split_reference_suffix(original_markdown)
        cleaned_body, cleaned_suffix = self._split_reference_suffix(
            _strip_leaked_markdown_boundaries(cleaned_markdown)
        )

        original_sections = self._split_paged_sections(original_body)
        cleaned_sections = self._split_paged_sections(cleaned_body)

        suffix = cleaned_suffix or original_suffix

        if not original_sections:
            return cleaned_body if not suffix else f"{cleaned_body}\n\n{suffix}"

        if len(cleaned_sections) != len(original_sections) or [
            section_id for section_id, _ in cleaned_sections
        ] != [section_id for section_id, _ in original_sections]:
            logger.warning(
                f"[{source}] Structured marker drift detected, restoring original layout"
            )
            return original_body if not suffix else f"{original_body}\n\n{suffix}"

        stabilized_sections: list[str] = []
        reverted_sections: list[str] = []
        for (original_id, original_section), (_, cleaned_section) in zip(
            original_sections, cleaned_sections, strict=False
        ):
            if (
                self._is_suspicious_page_expansion(original_section, cleaned_section)
                or self._is_suspicious_section_body_loss(
                    original_section, cleaned_section
                )
                or self._is_suspicious_image_only_text_injection(
                    original_section, cleaned_section
                )
            ):
                stabilized_sections.append(original_section)
                reverted_sections.append(original_id.replace(":", " "))
            else:
                stabilized_sections.append(cleaned_section)

        stabilized_body = "\n\n".join(stabilized_sections).strip()
        if reverted_sections:
            logger.warning(
                f"[{source}] Reverted suspicious section drift: {', '.join(reverted_sections)}"
            )

        return stabilized_body if not suffix else f"{stabilized_body}\n\n{suffix}"

    @staticmethod
    def _find_missing_boundary_placeholders(
        llm_output: str,
        mapping: dict[str, str],
    ) -> list[str]:
        """Find dropped page/slide placeholders in raw LLM output."""
        missing: list[str] = []
        for placeholder, original in mapping.items():
            if "PAGENUM" not in placeholder and "SLIDENUM" not in placeholder:
                continue
            if placeholder not in llm_output:
                missing.append(original)
        return missing

    def _fallback_if_boundary_placeholders_missing(
        self,
        llm_output: str,
        original_markdown: str,
        mapping: dict[str, str],
        source: str,
        stage: str,
    ) -> str | None:
        """Use original paginated content when LLM drops structural placeholders."""
        missing = self._find_missing_boundary_placeholders(llm_output, mapping)
        if not missing:
            return None

        marker_nums = [
            match.group(1)
            for marker in missing
            if (match := _BOUNDARY_MARKER_EXTRACT_RE.search(marker))
        ]
        detail = ", ".join(marker_nums) if marker_nums else str(len(missing))
        logger.warning(
            f"[{source}] {stage} dropped structural placeholders "
            f"(markers: {detail}); using original paginated content"
        )
        return original_markdown

    async def extract_from_screenshot(
        self,
        screenshot_path: Path,
        context: str = "",
        original_title: str | None = None,
    ) -> tuple[str, str]:
        """
        Extract content purely from screenshot (screenshot-only mode).

        This method does NOT use any pre-extracted text - it relies entirely
        on Vision LLM to extract content from the screenshot.

        Args:
            screenshot_path: Path to full-page screenshot
            context: Source URL/filename for logging
            original_title: Optional title to preserve in frontmatter

        Returns:
            Tuple of (extracted_markdown, frontmatter_yaml)
        """
        # Load screenshot early so the cache key includes a content fingerprint.
        # A re-fetch of the same URL produces the same filename, so keying by
        # filename alone would return stale results when content changed.
        _, base64_image = self._get_cached_image(screenshot_path)  # type: ignore[attr-defined]
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()

        cache_key = f"screenshot_extract:{context}"
        cache_content = f"{screenshot_path.name}|{image_fingerprint}"

        # Use screenshot extraction prompts
        system_prompt = self._prompt_manager.get_prompt(
            "screenshot_extract_system",
            source=context,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "screenshot_extract_user",
        )

        # Build content parts with user prompt and screenshot only
        content_parts: list[dict] = [
            {"type": "text", "text": user_prompt},
        ]

        # Add screenshot (base64_image was loaded above for the cache key)
        mime_type = get_mime_type(screenshot_path.suffix)
        content_parts.append({"type": "text", "text": "\n__MARKITAI_SCREENSHOT__"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        # Guard against VLM degeneration (repetition loops) in extracted text.
        # The truncation correction runs in validate; cache_if skips persisting
        # degenerate responses so a clean retry isn't poisoned.
        degenerated = False

        def _guard_degeneration(
            result: EnhancedDocumentResult,
        ) -> EnhancedDocumentResult:
            nonlocal degenerated
            cleaned, degenerated = truncate_degenerate_tail(
                result.cleaned_markdown, context=context, stage="screenshot_extract"
            )
            return EnhancedDocumentResult(
                cleaned_markdown=cleaned, frontmatter=result.frontmatter
            )

        call = LLMCall(
            purpose="screenshot_extract",
            messages=messages,
            response_model=EnhancedDocumentResult,
            context=context,
            cache_key=cache_key,
            cache_content=cache_content,
            cache_model="vision",
            validate=_guard_degeneration,
            cache_if=lambda _result: not degenerated,
            serialize=_document_result_to_cache_value,
            deserialize=_document_result_from_cache_value,
            router=self.vision_router,
        )
        response, _raw_response = await self.engine.complete_structured(call)

        # Build frontmatter using utility function (hit and miss paths)
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        frontmatter_dict = build_frontmatter_dict(
            source=context,
            description=response.frontmatter.description,
            tags=response.frontmatter.tags,
            title=original_title,  # Preserve original title if provided
            content=response.cleaned_markdown,
        )
        frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()

        return response.cleaned_markdown, frontmatter_yaml

    async def enhance_url_with_vision(
        self,
        content: str,
        screenshot_path: Path,
        context: str = "",
        original_title: str | None = None,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Enhance URL content using screenshot as visual reference.

        Unlike enhance_document_with_vision, this method:
        - Does NOT use slide/page number protection (URLs don't have these)
        - Generates frontmatter along with cleaned content
        - Uses a simpler content protection strategy

        Args:
            content: URL content (may be multi-source combined)
            screenshot_path: Path to full-page screenshot
            context: Source URL for logging
            original_title: Optional page title from fetch result (preferred)
            fetch_strategy: Optional fetch strategy to preserve in frontmatter
            extra_meta: Optional source metadata to merge into frontmatter

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        # Use provided title, or try to extract from content frontmatter
        from markitai.utils.frontmatter import extract_frontmatter_title

        if original_title is None:
            original_title = extract_frontmatter_title(content)

        cache_key = f"enhance_url:{context}"
        cache_content = (
            f"{screenshot_path.name}|{_compute_document_fingerprint(content, [])}"
        )

        # Only protect image references, NOT slide/page markers (URLs don't have them)
        protected_text, img_mapping = self._protect_image_positions(content)

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt(
            "url_enhance_system",
            source=context,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "url_enhance_user",
            content=protected_text,
        )

        # Build content parts with user prompt and screenshot
        content_parts: list[dict] = [
            {"type": "text", "text": user_prompt},
        ]

        # Add screenshot
        _, base64_image = self._get_cached_image(screenshot_path)  # type: ignore[attr-defined]
        mime_type = get_mime_type(screenshot_path.suffix)
        content_parts.append({"type": "text", "text": "\n__MARKITAI_SCREENSHOT__"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        # Degeneration guard shared between validate and cache_if: cache_if
        # skips persisting degenerate responses so a clean retry isn't poisoned
        degenerated = False

        def _postprocess(result: EnhancedDocumentResult) -> EnhancedDocumentResult:
            """Post-process a fresh LLM result (runs before the cache write)."""
            nonlocal degenerated

            # Restore image positions (with fallback to original if placeholders lost)
            cleaned = self._restore_images_or_fallback(
                result.cleaned_markdown,
                content,
                img_mapping,
                context,
                "url_vision_enhance",
            )

            # Remove any hallucinated or leaked markers that shouldn't be in URL output
            # Remove hallucinated slide/page markers (URLs shouldn't have these)
            cleaned = re.sub(r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?", "", cleaned)
            cleaned = re.sub(r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?", "", cleaned)
            # Remove source labels that may leak from multi-source content
            cleaned = re.sub(r"<!--\s*Source:\s*[^>]+-->\s*\n?", "", cleaned)
            cleaned = re.sub(
                r"##\s*(Static Content|Browser Content|Screenshot Reference)\s*\n+",
                "",
                cleaned,
            )
            # Also remove any residual MARKITAI placeholders
            cleaned = re.sub(r"__MARKITAI_[A-Z_]+_?\d*__\s*\n?", "", cleaned)

            # Fix malformed image refs
            cleaned = self.fix_malformed_image_refs(cleaned)

            # Guard against VLM degeneration (repetition loops) in enhanced text
            cleaned, degenerated = truncate_degenerate_tail(
                cleaned, context=context, stage="url_vision_enhance"
            )
            return EnhancedDocumentResult(
                cleaned_markdown=cleaned, frontmatter=result.frontmatter
            )

        def _from_cache(cached: dict[str, Any]) -> EnhancedDocumentResult:
            # Legacy hit-path default: a cached entry without cleaned_markdown
            # falls back to the original content (not "")
            return EnhancedDocumentResult.model_construct(
                cleaned_markdown=cached.get("cleaned_markdown", content),
                frontmatter=Frontmatter.model_construct(
                    description=cached.get("description", ""),
                    tags=cached.get("tags", []),
                ),
            )

        call = LLMCall(
            purpose="enhance_url",
            messages=messages,
            response_model=EnhancedDocumentResult,
            context=context,
            cache_key=cache_key,
            cache_content=cache_content,
            cache_model="vision",
            validate=_postprocess,
            cache_if=lambda _result: not degenerated,
            serialize=_document_result_to_cache_value,
            deserialize=_from_cache,
            router=self.vision_router,
        )
        response, _raw_response = await self.engine.complete_structured(call)

        # Build frontmatter using utility function (hit and miss paths)
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        frontmatter_dict = build_frontmatter_dict(
            source=context,
            description=response.frontmatter.description,
            tags=response.frontmatter.tags,
            title=original_title,  # Preserve original title
            content=response.cleaned_markdown,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
        )
        frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()

        return response.cleaned_markdown, frontmatter_yaml

    async def enhance_document_with_vision(
        self,
        extracted_text: str,
        page_images: list[Path],
        context: str = "",
    ) -> str:
        """
        Clean document format using extracted text and page images as reference.

        This method only cleans formatting issues (removes residuals, fixes structure).
        It does NOT restructure or rewrite content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            context: Context identifier for logging (e.g., document name)

        Returns:
            Cleaned markdown content (same content, cleaner format)
        """
        if not page_images:
            return extracted_text

        # Check persistent cache using page count + text fingerprint as key
        # Create a fingerprint from text + page image names for cache lookup
        page_name_list = [p.name for p in page_images[:10]]  # First 10 page names
        cache_key = f"enhance_vision:{context}:{len(page_images)}"
        cache_content = _compute_document_fingerprint(extracted_text, page_name_list)
        cached = self._persistent_cache.get(cache_key, cache_content, context=context)
        if cached is not None:
            # Fix malformed image refs and echoed prompt tails even for cached
            # content (handles old cache entries)
            return self.strip_prompt_echo(self.fix_malformed_image_refs(cached))

        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(extracted_text)
        protected_content, mapping = self.protect_content(extracted_text)

        # Use unified document_vision prompt (no metadata section for cleaning-only)
        system_prompt = self._prompt_manager.get_prompt(
            "document_vision_system",
            source=context or "unknown",
            metadata_section="",  # No metadata generation for cleaning-only
        )
        user_prompt = self._prompt_manager.get_prompt(
            "document_vision_user", content=protected_content
        )

        # Prepare content parts with user prompt and images
        content_parts: list[dict] = [
            {"type": "text", "text": user_prompt},
        ]

        # Add page images (using cache to avoid repeated reads)
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]
            mime_type = get_mime_type(image_path.suffix)

            # Unique page label that won't conflict with document content
            content_parts.append(
                {
                    "type": "text",
                    "text": f"\n__MARKITAI_PAGE_LABEL_{i}__",
                }
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        # Append tail reminder to reinforce placeholder rules for last pages
        content_parts.append(
            {
                "type": "text",
                "text": "\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders exactly as-is. "
                "Do not remove or modify any placeholder. "
                "Output every page/slide — do not skip the last pages.",
            }
        )

        # Direct engine call (Phase 2.3): this point is only reached with
        # page_images non-empty, so the messages always contain images and
        # _call_llm's has_images check would always pick the vision router —
        # pass it explicitly.
        call_index = self._get_next_call_index(context) if context else 0
        call_id = f"{context}:{call_index}" if context else f"call:{call_index}"
        response = await self.engine.complete_text(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            call_id=call_id,
            context=context,
            max_retries=self.config.router_settings.num_retries,
            router=self.vision_router,
        )

        # Restore protected content from placeholders, with fallback for removed items
        result = self.unprotect_content(
            self.strip_prompt_echo(response.content), mapping, protected
        )

        # Fix malformed image references (e.g., extra closing parentheses)
        result = self.fix_malformed_image_refs(result)
        result = self._stabilize_paged_markdown(extracted_text, result, context)

        # Guard against VLM degeneration (repetition loops) in enhanced text
        result, degenerated = truncate_degenerate_tail(
            result, context=context, stage="document_vision_enhance"
        )

        # Store in persistent cache
        # Skip persisting degenerate responses so a clean retry isn't poisoned
        if not degenerated:
            self._persistent_cache.set(cache_key, cache_content, result, model="vision")

        return result

    async def enhance_document_complete(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str = "",
        max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
        original_title: str | None = None,
    ) -> tuple[str, str]:
        """
        Complete document enhancement: clean format + generate frontmatter.

        Architecture:
        - Single batch (pages <= max_pages_per_batch): Use Instructor for combined
          cleaning + frontmatter in one LLM call (saves one API call)
        - Multi batch (pages > max_pages_per_batch): First batch uses combined
          call for cleaning + frontmatter, remaining batches clean only

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            source: Source file name
            max_pages_per_batch: Max pages per batch (default 10)
            original_title: Optional explicit title from converter metadata

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        from markitai.utils.frontmatter import resolve_document_title

        resolved_title = resolve_document_title(
            source=source,
            explicit_title=original_title,
            content=extracted_text,
        )

        if not page_images:
            # No images, fall back to regular process_document
            return await self.process_document(
                extracted_text,
                source,
                title=resolved_title,
            )

        # Single batch: use combined Instructor call (saves one API call)
        if len(page_images) <= max_pages_per_batch:
            logger.info(
                f"[{source}] Processing {len(page_images)} pages with combined call"
            )
            try:
                return await self._enhance_with_frontmatter(
                    extracted_text,
                    page_images,
                    source,
                    original_title=resolved_title,
                )
            except Exception as e:
                # Log succinct warning instead of full exception trace
                logger.warning(
                    f"[{source}] Combined call failed: {format_error_message(e)}, "
                    "falling back to separate calls"
                )
                # Fallback to separate calls with error handling
                try:
                    cleaned = await self.enhance_document_with_vision(
                        extracted_text, page_images, context=source
                    )
                except Exception as clean_err:
                    logger.warning(
                        f"[{source}] Vision cleaning also failed: {format_error_message(clean_err)}"
                    )
                    cleaned = extracted_text

                # Use _build_fallback_frontmatter for consistent structure
                frontmatter = self._build_fallback_frontmatter(
                    source,
                    cleaned,
                    title=resolved_title,
                )

                return cleaned, frontmatter

        # Multi batch: first batch uses _enhance_with_frontmatter for Instructor-based
        # frontmatter generation, remaining batches clean only
        logger.info(
            f"[{source}] Processing {len(page_images)} pages in batches of "
            f"{max_pages_per_batch} (first batch generates frontmatter)"
        )

        # Split into batches
        image_batches = self._split_into_batches(page_images, max_pages_per_batch)
        text_batches = self._split_text_into_batches(
            extracted_text, page_images, max_pages_per_batch
        )

        # First batch: use _enhance_with_frontmatter to generate frontmatter with Instructor
        logger.info(
            f"[{source}] Batch 1/{len(image_batches)}: "
            f"pages 1-{len(image_batches[0])} (with frontmatter)"
        )
        try:
            cleaned_first, frontmatter = await self._enhance_with_frontmatter(
                text_batches[0],
                image_batches[0],
                source,
                original_title=resolved_title,
            )
        except Exception as e:
            logger.warning(
                f"[{source}] First batch failed: {format_error_message(e)}, "
                "falling back to vision-only cleaning"
            )
            try:
                cleaned_first = await self.enhance_document_with_vision(
                    text_batches[0], image_batches[0], context=source
                )
            except Exception as clean_err:
                logger.warning(
                    f"[{source}] Vision cleaning also failed: {format_error_message(clean_err)}"
                )
                cleaned_first = text_batches[0]
            frontmatter = self._build_fallback_frontmatter(
                source,
                extracted_text,
                title=resolved_title,
            )

        # Remaining batches: parallel cleaning without frontmatter
        cleaned_parts = [cleaned_first]
        if len(image_batches) > 1:
            remaining_tasks = []
            for i in range(1, len(image_batches)):
                logger.info(
                    f"[{source}] Batch {i + 1}/{len(image_batches)}: "
                    f"pages {i * max_pages_per_batch + 1}-"
                    f"{min((i + 1) * max_pages_per_batch, len(page_images))}"
                )
                remaining_tasks.append(
                    self.enhance_document_with_vision(
                        text_batches[i], image_batches[i], context=source
                    )
                )

            # Process remaining batches in parallel
            remaining_results = await asyncio.gather(
                *remaining_tasks, return_exceptions=True
            )

            # Merge results with fallbacks for failed batches
            for i, result in enumerate(remaining_results):
                if isinstance(result, BaseException):
                    logger.warning(
                        f"[{source}] Batch {i + 2} failed: {format_error_message(result)}, "
                        "using original text"
                    )
                    cleaned_parts.append(text_batches[i + 1])
                else:
                    cleaned_parts.append(result)

        # Merge all cleaned parts
        cleaned = "\n\n".join(cleaned_parts)

        return cleaned, frontmatter

    async def _enhance_with_frontmatter(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str,
        original_title: str | None = None,
    ) -> tuple[str, str]:
        """Enhance document with vision and generate frontmatter in one call.

        Uses Instructor for structured output.

        Args:
            extracted_text: Text to clean
            page_images: Page images for visual reference
            source: Source file name
            original_title: Optional explicit title from converter metadata

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        from markitai.utils.frontmatter import resolve_document_title

        resolved_title = resolve_document_title(
            source=source,
            explicit_title=original_title,
            content=extracted_text,
        )

        # Cache key: page count + source + text fingerprint
        page_name_list = [p.name for p in page_images[:10]]  # First 10 page names
        cache_key = f"enhance_frontmatter:{source}:{len(page_images)}"
        cache_content = _compute_document_fingerprint(extracted_text, page_name_list)

        # Extract protected content for fallback restoration
        protected = self.extract_protected_content(extracted_text)

        # Protect slide comments and images with placeholders before LLM processing
        protected_text, mapping = self.protect_content(extracted_text)

        # Metadata section to inject into unified document_vision prompt
        metadata_section = """
## Task 2: Metadata Generation

Generate the following fields:

- description: Summarize the core point or conclusion of the entire document in one sentence (under 100 characters, single line)
  - Focus on what the article actually discusses, not a generic description
  - Do not use templated openings like "This article discusses..."
  - If the source document already has a semantically accurate description, reuse it directly
- tags: Array of related tags (3-5, for classification and retrieval)
  - **Tags must not contain spaces** — use hyphens instead: `machine-learning`, not `machine learning`
  - Each tag must be 30 characters or fewer
  - Examples: `AI`, `software-engineering`, `web-development`

**Output language MUST match the source document** — English content → English metadata, Chinese content → Chinese metadata, etc.
"""
        # Use unified document_vision prompt with metadata section
        system_prompt = self._prompt_manager.get_prompt(
            "document_vision_system",
            source=source,
            metadata_section=metadata_section,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "document_vision_user",
            content=protected_text,
        )

        # Build content parts with user prompt and images
        content_parts: list[dict] = [
            {"type": "text", "text": user_prompt},
        ]

        # Add page images
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]
            mime_type = get_mime_type(image_path.suffix)
            # Unique page label that won't conflict with document content
            content_parts.append(
                {"type": "text", "text": f"\n__MARKITAI_PAGE_LABEL_{i}__"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        # Append tail reminder to reinforce placeholder rules for last pages
        content_parts.append(
            {
                "type": "text",
                "text": "\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders exactly as-is. "
                "Do not remove or modify any placeholder. "
                "Output every page/slide — do not skip the last pages.",
            }
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        def _postprocess(result: EnhancedDocumentResult) -> EnhancedDocumentResult:
            """Post-process a fresh LLM result (runs before the cache write)."""
            response_markdown = self.strip_prompt_echo(result.cleaned_markdown)
            fallback = self._fallback_if_boundary_placeholders_missing(
                response_markdown,
                extracted_text,
                mapping,
                source,
                "vision_enhance",
            )
            if fallback is not None:
                cleaned = fallback
            else:
                # Restore protected content from placeholders.
                # Pass protected dict for fallback restoration if LLM removed placeholders.
                cleaned = self.unprotect_content(response_markdown, mapping, protected)

            # Fix malformed image references (e.g., extra closing parentheses)
            cleaned = self.fix_malformed_image_refs(cleaned)
            cleaned = self._stabilize_paged_markdown(extracted_text, cleaned, source)
            return EnhancedDocumentResult(
                cleaned_markdown=cleaned, frontmatter=result.frontmatter
            )

        call = LLMCall(
            purpose="enhance_frontmatter",
            messages=messages,
            response_model=EnhancedDocumentResult,
            context=source,
            cache_key=cache_key,
            cache_content=cache_content,
            cache_model="vision",
            validate=_postprocess,
            serialize=_document_result_to_cache_value,
            deserialize=_document_result_from_cache_value,
            router=self.vision_router,
        )
        response, raw_response = await self.engine.complete_structured(call)

        cleaned_markdown = response.cleaned_markdown
        if raw_response is None:
            # Cache hit: fix malformed image refs and echoed prompt tails even
            # for cached content (handles old cache entries)
            cleaned_markdown = self.strip_prompt_echo(
                self.fix_malformed_image_refs(cleaned_markdown)
            )

        # Build frontmatter YAML using utility function for consistent structure
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        frontmatter_dict = build_frontmatter_dict(
            source=source,
            description=response.frontmatter.description,
            tags=response.frontmatter.tags,
            title=resolved_title,
            content=cleaned_markdown,
        )
        frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()

        return cleaned_markdown, frontmatter_yaml

    def _build_fallback_frontmatter(
        self,
        source: str,
        content: str,
        title: str | None = None,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Build fallback frontmatter when LLM fails.

        Args:
            source: Source filename
            content: Document content (for title extraction)
            title: Optional pre-extracted title to preserve
            fetch_strategy: Optional fetch strategy
            extra_meta: Optional extra metadata from external strategies

        Returns:
            YAML frontmatter string (without --- markers)
        """
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        # Extract description/tags from extra_meta as fallback
        fallback_desc = ""
        fallback_tags: list[str] = []
        if extra_meta:
            meta_desc = extra_meta.get("description")
            if isinstance(meta_desc, str) and meta_desc.strip():
                fallback_desc = meta_desc
                logger.info(f"[{source}] Using source metadata description as fallback")
            meta_tags = extra_meta.get("tags")
            if isinstance(meta_tags, list) and meta_tags:
                fallback_tags = meta_tags
                logger.info(f"[{source}] Using source metadata tags as fallback")

        frontmatter_dict = build_frontmatter_dict(
            source=source,
            description=fallback_desc,
            tags=fallback_tags,
            title=title,
            content=content,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
        )
        # Mark degraded output and log at ERROR level: quiet/stdout mode only
        # shows ERROR+, and without both signals a failed enhancement is
        # indistinguishable from a successful one.
        frontmatter_dict["llm_enhanced"] = False
        logger.error(f"[{source}] LLM enhancement failed; using fallback frontmatter")
        return frontmatter_to_yaml(frontmatter_dict).strip()

    @staticmethod
    def _split_into_batches(
        page_images: list[Path], batch_size: int
    ) -> list[list[Path]]:
        """Split page images into batches.

        Args:
            page_images: List of page image paths
            batch_size: Maximum images per batch

        Returns:
            List of batches, each containing up to batch_size images
        """
        batches: list[list[Path]] = []
        for i in range(0, len(page_images), batch_size):
            batches.append(page_images[i : i + batch_size])
        return batches

    def _split_text_into_batches(
        self, extracted_text: str, page_images: list[Path], batch_size: int
    ) -> list[str]:
        """Split text into batches corresponding to page image batches.

        Args:
            extracted_text: Full document text
            page_images: All page images
            batch_size: Pages per batch

        Returns:
            List of text chunks, one per batch
        """
        num_pages = len(page_images)
        page_texts = self.split_text_by_pages(extracted_text, num_pages)

        batches: list[str] = []
        for i in range(0, num_pages, batch_size):
            batch_texts = page_texts[i : i + batch_size]
            batches.append("\n\n".join(batch_texts))
        return batches

    async def _enhance_document_batched_simple(
        self,
        extracted_text: str,
        page_images: list[Path],
        batch_size: int,
        source: str = "",
    ) -> str:
        """Process long documents in batches - vision cleaning only.

        All batches use the same method for consistent output format.

        Args:
            extracted_text: Full document text
            page_images: All page images
            batch_size: Pages per batch
            source: Source file name

        Returns:
            Merged cleaned content
        """
        num_pages = len(page_images)
        num_batches = (num_pages + batch_size - 1) // batch_size

        # Split text by pages
        page_texts = self.split_text_by_pages(extracted_text, num_pages)

        cleaned_parts = []

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_pages)

            # Get text and images for this batch
            batch_texts = page_texts[batch_start:batch_end]
            batch_images = page_images[batch_start:batch_end]
            batch_text = "\n\n".join(batch_texts)

            logger.info(
                f"[{source}] Batch {batch_num + 1}/{num_batches}: "
                f"pages {batch_start + 1}-{batch_end}"
            )

            # All batches: clean only (no frontmatter)
            # Use source as context (not batch-specific) so all usage aggregates to same context
            batch_cleaned = await self.enhance_document_with_vision(
                batch_text, batch_images, context=source
            )

            cleaned_parts.append(batch_cleaned)

        # Merge all batches
        return "\n\n".join(cleaned_parts)

    async def process_document(
        self,
        markdown: str,
        source: str,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> tuple[str, str]:
        """Process a document with LLM: clean and generate frontmatter.

        Args:
            markdown: Raw markdown content
            source: Source file name
            fetch_strategy: Optional fetch strategy (included in frontmatter)
            extra_meta: Optional extra metadata from external strategies
            title: Optional explicit title (takes precedence over extraction)

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        from markitai.utils.frontmatter import (
            extract_frontmatter_title,
            resolve_document_title,
        )

        explicit_title = title or extract_frontmatter_title(markdown)
        original_title = resolve_document_title(
            source=source,
            explicit_title=explicit_title,
            content=markdown,
        )

        # Social posts arrive pre-curated by the site extractors: the LLM
        # cleanup adds no value there and weak models can damage structure
        # (flattened quote blocks, respaced CJK text), so the LLM result is
        # used for metadata only and the body passes through verbatim.
        body_verbatim = (
            extra_meta is not None
            and extra_meta.get("content_profile") == "social_post"
        )

        # Protect image positions before any LLM processing to prevent drift
        image_protected, image_mapping = self._protect_image_positions(markdown)

        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(image_protected)
        protected_content, mapping = self.protect_content(image_protected)

        # Try combined approach with Instructor first
        try:
            result = await self._process_document_combined(protected_content, source)

            if body_verbatim:
                logger.info(
                    f"[LLM:{source}] social_post profile: body kept verbatim, "
                    "LLM result used for metadata only"
                )
                cleaned = markdown
            else:
                fallback = self._fallback_if_boundary_placeholders_missing(
                    result.cleaned_markdown,
                    markdown,
                    mapping,
                    source,
                    "document_process",
                )
                if fallback is not None:
                    cleaned = fallback
                else:
                    # Restore protected content from placeholders, with fallback
                    # Disable "append missing images at end" — image positions are
                    # managed by image_mapping, not by unprotect_content fallback
                    cleaned = self.unprotect_content(
                        result.cleaned_markdown,
                        mapping,
                        protected,
                        restore_missing_images_at_end=False,
                    )
                cleaned = self.fix_malformed_image_refs(cleaned)
                cleaned = self._stabilize_paged_markdown(markdown, cleaned, source)
                # Restore image positions (or fall back to original if placeholders were lost)
                cleaned = self._restore_images_or_fallback(
                    cleaned, markdown, image_mapping, source, "document_process"
                )

            # Convert Frontmatter to YAML string using utility function
            from markitai.utils.frontmatter import (
                build_frontmatter_dict,
                frontmatter_to_yaml,
            )

            frontmatter_dict = build_frontmatter_dict(
                source=source,
                description=result.frontmatter.description,
                tags=result.frontmatter.tags,
                title=original_title,  # Preserve original title
                content=cleaned,
                fetch_strategy=fetch_strategy,
                extra_meta=extra_meta,
            )
            frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()
            return cleaned, frontmatter_yaml
        except Exception as e:
            fatal_provider_error = _find_non_retryable_provider_error(e)
            if fatal_provider_error is not None:
                logger.warning(
                    f"[LLM:{source}] Structured document processing failed with "
                    f"non-retryable provider error, skipping cleaner fallback: "
                    f"{format_error_message(fatal_provider_error)}"
                )
                frontmatter = self._build_fallback_frontmatter(
                    source,
                    markdown,
                    original_title,
                    fetch_strategy=fetch_strategy,
                    extra_meta=extra_meta,
                )
                return markdown, frontmatter

            logger.warning(
                f"[LLM:{source}] Structured document processing failed, "
                f"falling back to cleaner: {format_error_message(e)}"
            )

        # Fallback: Run cleaning only (no longer use generate_frontmatter)
        # Use clean_markdown which has its own protection mechanism
        if body_verbatim:
            cleaned = markdown
        else:
            try:
                cleaned = await self.clean_markdown(markdown, context=source)
            except Exception as clean_err:
                logger.warning(
                    f"Markdown cleaning failed: {format_error_message(clean_err)}"
                )
                cleaned = markdown

        # Build fallback frontmatter — prefer title from cleaned content so it
        # stays consistent with any heading corrections made by the cleaner.
        from markitai.utils.frontmatter import extract_title_from_content

        cleaned_title = extract_title_from_content(cleaned)
        fallback_title = cleaned_title if cleaned_title else original_title
        frontmatter = self._build_fallback_frontmatter(
            source,
            cleaned,
            fallback_title,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
        )

        return cleaned, frontmatter

    async def clean_document_pure(self, markdown: str, source: str) -> str:
        """Pure cleaning: send raw markdown to LLM, return response as-is.

        No content protection, no stabilization, no truncation, no frontmatter.
        The LLM decides what to clean based on the cleaner prompt.

        Args:
            markdown: Raw markdown content
            source: Source file name for logging context

        Returns:
            LLM response content as-is
        """
        system_prompt = self._prompt_manager.get_prompt(
            "cleaner_system", mode_rules=PURE_MODE_RULES
        )
        user_prompt = self._prompt_manager.get_prompt("cleaner_user", content=markdown)
        response = await self._call_llm(  # type: ignore[attr-defined]
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            context=source,
        )
        return response.content

    async def _process_document_combined(
        self,
        markdown: str,
        source: str,
    ) -> DocumentProcessResult:
        """
        Process document with combined cleaner + frontmatter using Instructor.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            DocumentProcessResult with cleaned markdown and frontmatter
        """
        cache_key = f"document_process:{source}"

        # Truncate content if needed (with warning)
        original_len = len(markdown)
        truncated_content = self.smart_truncate(markdown, DEFAULT_MAX_CONTENT_CHARS)
        if len(truncated_content) < original_len:
            logger.warning(
                f"[LLM:{source}] Content truncated: {original_len} -> {len(truncated_content)} chars "
                f"(limit: {DEFAULT_MAX_CONTENT_CHARS}). Some content may be lost."
            )

        # Get separated system and user prompts
        system_prompt = self._prompt_manager.get_prompt(
            "document_process_system",
            source=source,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "document_process_user",
            content=truncated_content,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _validate(response: DocumentProcessResult) -> DocumentProcessResult:
            """Detect prompt leakage before the cache write.

            Recoverable leakage returns a corrected result; unrecoverable
            leakage raises ValueError (nothing is cached, error propagates).
            """
            validated_markdown = self._validate_no_prompt_leakage(
                response.cleaned_markdown, source
            )
            if validated_markdown != response.cleaned_markdown:
                return DocumentProcessResult(
                    cleaned_markdown=validated_markdown,
                    frontmatter=response.frontmatter,
                )
            return response

        # router=None: this is the only structured call on the main router
        call = LLMCall(
            purpose="document_process",
            messages=messages,
            response_model=DocumentProcessResult,
            context=source,
            cache_key=cache_key,
            cache_content=markdown,
            cache_model="default",
            validate=_validate,
            serialize=_document_result_to_cache_value,
            deserialize=_document_result_from_cache_value,
        )
        try:
            response, raw_response = await self.engine.complete_structured(call)
        except Exception:
            self._cache_misses += 1
            raise
        if raw_response is None:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return response

    def _validate_no_prompt_leakage(self, cleaned: str, source: str) -> str:
        """Detect and handle prompt leakage."""
        prompt_markers = [
            "## Task 1:",
            "## Task 2:",
            "## 任务 1:",
            "## 任务 2:",
            "【核心原则】",
            "【清理规范】",
            "请处理以下",
            "你是一个专业的",
        ]

        for marker in prompt_markers:
            if marker in cleaned:
                logger.warning(
                    f"[{source}] Prompt leakage detected, attempting recovery"
                )
                if "---" in cleaned:
                    parts = cleaned.split("---", 2)
                    if len(parts) > 2:
                        return parts[2].strip()
                raise ValueError("LLM returned prompt text in cleaned_markdown")

        return cleaned

    def format_llm_output(
        self,
        markdown: str,
        frontmatter: str,
    ) -> str:
        """Format final output with frontmatter.

        Since frontmatter is now always generated programmatically with proper
        structure via build_frontmatter_dict() + frontmatter_to_yaml(), this
        function mainly handles markdown cleanup.

        Args:
            markdown: Cleaned markdown content
            frontmatter: YAML frontmatter (without --- markers)

        Returns:
            Complete markdown with frontmatter
        """
        # Clean frontmatter (remove accidental --- markers)
        frontmatter = self.clean_frontmatter(frontmatter)

        # Clean markdown content
        markdown = _strip_leaked_markdown_boundaries(markdown)
        markdown = self._remove_uncommented_screenshots(markdown)
        from markitai.utils.markdown_quality import normalize_markdown

        markdown = normalize_markdown(markdown).rstrip()

        return f"---\n{frontmatter}\n---\n\n{markdown}"

    @staticmethod
    def _remove_uncommented_screenshots(content: str) -> str:
        """Remove non-commented page screenshot references from content.

        Page screenshots should only appear as HTML comments at the end of the document.
        If LLM accidentally outputs them as regular image references, remove them.

        Also ensures that any screenshot references in the "Page images for reference"
        section are properly commented.

        Args:
            content: Markdown content

        Returns:
            Content with uncommented screenshots removed/fixed
        """
        # Find the position of "<!-- Page images for reference -->" if it exists
        page_images_header = "<!-- Page images for reference -->"
        header_pos = content.find(page_images_header)

        if header_pos == -1:
            # No page images section, just remove any stray screenshot references
            # IMPORTANT: Only match markitai-generated screenshot patterns to avoid
            # removing user's original .markitai/screenshots/ references (P0-5 fix).
            # markitai naming format: {filename}.page{NNNN}.{ext} in .markitai/screenshots/
            content = _SCREENSHOT_REF_RE.sub("", content)

            # Also remove any page/image labels that LLM may have copied
            # Pattern: ## or ### Page N Image: followed by empty line (legacy format)
            content = _PAGE_HEADING_LABEL_RE.sub("", content)
            # Merged: [Page/Image N], __MARKITAI_*_LABEL_N__, __MARKITAI_SLIDE_N__
            content = _PLACEHOLDER_LABEL_RE.sub("", content)

            # Clean up any resulting empty lines
            content = _EXCESS_NEWLINES_RE.sub("\n\n", content)
        else:
            # Split at the page images section
            before = content[:header_pos]
            after = content[header_pos:]

            # Remove screenshot references from BEFORE the page images header
            # IMPORTANT: Only match markitai-generated screenshot patterns (P0-5 fix)
            before = _SCREENSHOT_REF_RE.sub("", before)

            # Also remove any page/image labels that LLM may have copied
            before = _PAGE_HEADING_LABEL_RE.sub("", before)
            # Merged: [Page/Image N], __MARKITAI_*_LABEL_N__, __MARKITAI_SLIDE_N__
            before = _PLACEHOLDER_LABEL_RE.sub("", before)
            before = _EXCESS_NEWLINES_RE.sub("\n\n", before)

            # Fix the AFTER section: convert any non-commented page images to comments
            # Match lines with page image references that are not already commented
            # This handles: ![Page N](.markitai/screenshots/...)
            after_lines = after.split("\n")
            fixed_lines = []
            for line in after_lines:
                stripped = line.strip()
                # Check if it's an uncommented page image reference
                if (
                    stripped.startswith("![Page")
                    and f"{SCREENSHOTS_REL_PATH}/" in stripped
                    and not stripped.startswith("<!--")
                ):
                    fixed_lines.append(f"<!-- {stripped} -->")
                else:
                    fixed_lines.append(line)
            after = "\n".join(fixed_lines)

            content = before + after

        # Clean up screenshot comments section: remove blank lines between comments
        # Pattern: <!-- Page images for reference --> followed by page image comments
        def clean_page_section(match: re.Match) -> str:
            header = match.group(1)
            comments_section = match.group(2)
            # Extract individual comments and rejoin without blank lines
            comments = _PAGE_COMMENT_RE.findall(comments_section)
            return header + "\n" + "\n".join(comments)

        content = _PAGE_SECTION_RE.sub(clean_page_section, content)

        return content

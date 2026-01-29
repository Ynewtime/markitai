"""Document processing mixin for LLMProcessor.

This module contains all document-related methods extracted from processor.py
to reduce file size and improve maintainability.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import instructor
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from markitai.constants import (
    DEFAULT_INSTRUCTOR_MAX_RETRIES,
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_PAGES_PER_BATCH,
)
from markitai.llm.types import (
    DocumentProcessResult,
    EnhancedDocumentResult,
    Frontmatter,
)
from markitai.utils.mime import get_mime_type
from markitai.utils.text import format_error_message
from markitai.workflow.helpers import detect_language, get_language_name


def _context_display_name(context: str) -> str:
    """Get display name for logging context.

    Args:
        context: Context string (e.g., file path or URL)

    Returns:
        Short display name suitable for logging
    """
    if not context:
        return "unknown"
    # If it's a path, use just the filename
    if "/" in context or "\\" in context:
        return Path(context).name
    # If it's a URL, truncate it
    if len(context) > 50:
        return context[:47] + "..."
    return context


def get_response_cost(raw_response: Any) -> float:
    """Get response cost from raw LLM response.

    Args:
        raw_response: Raw response from LLM API

    Returns:
        Cost in dollars
    """
    from litellm import completion_cost

    try:
        return completion_cost(completion_response=raw_response) or 0.0
    except Exception:
        return 0.0


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
        _cache: Any
        _persistent_cache: Any
        _cache_hits: int
        _cache_misses: int
        _prompt_manager: Any
        router: Any
        vision_router: Any
        semaphore: asyncio.Semaphore

        def _call_llm(
            self,
            model: str,
            messages: list[dict[str, Any]],
            context: str = "",
        ) -> Any: ...

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
        _protect_content: Any
        _unprotect_content: Any
        _fix_malformed_image_refs: Any
        _clean_frontmatter: Any
        _smart_truncate: Any
        _split_text_by_pages: Any

    async def clean_markdown(self, content: str, context: str = "") -> str:
        """
        Clean and optimize markdown content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

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
            logger.debug(
                f"[{_context_display_name(context)}] Memory cache hit for clean_markdown"
            )
            return cached

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, content, context=context)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"[{_context_display_name(context)}] Persistent cache hit for clean_markdown"
            )
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, content, cached)
            return cached

        self._cache_misses += 1

        # 3. Extract and protect content before LLM processing
        protected = self.extract_protected_content(content)
        protected_content, mapping = self._protect_content(content)

        # Use separated system/user prompts to prevent prompt leakage
        system_prompt = self._prompt_manager.get_prompt("cleaner_system")
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

        # Restore protected content from placeholders, with fallback for removed items
        result = self._unprotect_content(response.content, mapping, protected)

        # Cache the result in both layers
        self._cache.set(cache_key, content, result)
        self._persistent_cache.set(cache_key, content, result, model="default")

        return result

    @staticmethod
    def _protect_image_positions(text: str) -> tuple[str, dict[str, str]]:
        """Replace image references with position markers to prevent LLM from moving them.

        Args:
            text: Markdown text with image references

        Returns:
            Tuple of (text with markers, mapping of marker -> original image reference)
        """
        mapping: dict[str, str] = {}
        result = text

        # Match ALL image references: ![...](...)
        # This includes both local assets and external URLs
        # Excludes screenshots placeholder which has its own protection
        img_pattern = r"!\[[^\]]*\]\([^)]+\)"
        for i, match in enumerate(re.finditer(img_pattern, text)):
            img_ref = match.group(0)
            # Skip screenshot placeholders (handled separately)
            if "screenshots/" in img_ref:
                continue
            marker = f"__MARKITAI_IMG_{i}__"
            mapping[marker] = img_ref
            result = result.replace(img_ref, marker, 1)

        return result, mapping

    @staticmethod
    def _restore_image_positions(text: str, mapping: dict[str, str]) -> str:
        """Restore original image references from position markers.

        Args:
            text: Text with position markers
            mapping: Mapping of marker -> original image reference

        Returns:
            Text with original image references restored
        """
        result = text
        for marker, original in mapping.items():
            result = result.replace(marker, original)
        return result

    async def enhance_url_with_vision(
        self,
        content: str,
        screenshot_path: Path,
        context: str = "",
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

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        start_time = time.perf_counter()

        # Check persistent cache
        cache_key = f"enhance_url:{context}"
        cache_content = f"{screenshot_path.name}|{content[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=context)
        if cached is not None:
            logger.debug(
                f"[{context}] Persistent cache hit for enhance_url_with_vision"
            )
            return cached.get("cleaned_markdown", content), cached.get(
                "frontmatter_yaml", ""
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

        async with self.semaphore:
            # Calculate dynamic max_tokens using minimum across all vision router models
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]
            max_tokens = self._calculate_dynamic_max_tokens(  # type: ignore[attr-defined]
                messages, router=self.vision_router
            )

            # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
            client = instructor.from_litellm(
                self.vision_router.acompletion, mode=instructor.Mode.MD_JSON
            )
            response, raw_response = await cast(
                Awaitable[tuple[EnhancedDocumentResult, Any]],
                client.chat.completions.create_with_completion(
                    model="default",
                    messages=cast(
                        list[ChatCompletionMessageParam],
                        messages,
                    ),
                    response_model=EnhancedDocumentResult,
                    max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                    max_tokens=max_tokens,
                ),
            )

            # Track usage and log completion
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            elapsed = time.perf_counter() - start_time

            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                cost = get_response_cost(raw_response)
                self._track_usage(  # type: ignore[attr-defined]
                    actual_model,
                    input_tokens,
                    output_tokens,
                    cost,
                    context,
                )

            logger.info(
                f"[LLM:{context}] url_vision_enhance: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={int(elapsed * 1000)}ms cost=${cost:.6f}"
            )

        # Restore image positions
        cleaned_markdown = self._restore_image_positions(
            response.cleaned_markdown, img_mapping
        )

        # Remove any hallucinated or leaked markers that shouldn't be in URL output
        # Remove hallucinated slide/page markers (URLs shouldn't have these)
        cleaned_markdown = re.sub(
            r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?", "", cleaned_markdown
        )
        cleaned_markdown = re.sub(
            r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?", "", cleaned_markdown
        )
        # Remove source labels that may leak from multi-source content
        cleaned_markdown = re.sub(
            r"<!--\s*Source:\s*[^>]+-->\s*\n?", "", cleaned_markdown
        )
        cleaned_markdown = re.sub(
            r"##\s*(Static Content|Browser Content|Screenshot Reference)\s*\n+",
            "",
            cleaned_markdown,
        )
        # Also remove any residual MARKITAI placeholders
        cleaned_markdown = re.sub(
            r"__MARKITAI_[A-Z_]+_?\d*__\s*\n?", "", cleaned_markdown
        )

        # Fix malformed image refs
        cleaned_markdown = self._fix_malformed_image_refs(cleaned_markdown)

        # Build frontmatter using utility function for consistent structure
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        frontmatter_dict = build_frontmatter_dict(
            source=context,
            description=response.frontmatter.description,
            tags=response.frontmatter.tags,
            content=cleaned_markdown,
        )
        frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()

        # Cache result
        cache_value = {
            "cleaned_markdown": cleaned_markdown,
            "frontmatter_yaml": frontmatter_yaml,
        }
        self._persistent_cache.set(
            cache_key, cache_content, cache_value, model="vision"
        )

        return cleaned_markdown, frontmatter_yaml

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
        page_names = "|".join(p.name for p in page_images[:10])  # First 10 page names
        cache_key = f"enhance_vision:{context}:{len(page_images)}"
        cache_content = f"{page_names}|{extracted_text[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=context)
        if cached is not None:
            logger.debug(
                f"[{_context_display_name(context)}] Persistent cache hit for enhance_document_with_vision"
            )
            # Fix malformed image refs even for cached content (handles old cache entries)
            return self._fix_malformed_image_refs(cached)

        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(extracted_text)
        protected_content, mapping = self._protect_content(extracted_text)

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt("document_enhance_system")
        user_prompt = self._prompt_manager.get_prompt(
            "document_enhance_user", content=protected_content
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

        response = await self._call_llm(  # type: ignore[attr-defined]
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            context=context,
        )

        # Restore protected content from placeholders, with fallback for removed items
        result = self._unprotect_content(response.content, mapping, protected)

        # Fix malformed image references (e.g., extra closing parentheses)
        result = self._fix_malformed_image_refs(result)

        # Store in persistent cache
        self._persistent_cache.set(cache_key, cache_content, result, model="vision")

        return result

    async def enhance_document_complete(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str = "",
        max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
    ) -> tuple[str, str]:
        """
        Complete document enhancement: clean format + generate frontmatter.

        Architecture:
        - Single batch (pages <= max_pages_per_batch): Use Instructor for combined
          cleaning + frontmatter in one LLM call (saves one API call)
        - Multi batch (pages > max_pages_per_batch): Clean in batches, then
          generate frontmatter separately

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            source: Source file name
            max_pages_per_batch: Max pages per batch (default 10)

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        if not page_images:
            # No images, fall back to regular process_document
            return await self.process_document(extracted_text, source)

        # Single batch: use combined Instructor call (saves one API call)
        if len(page_images) <= max_pages_per_batch:
            logger.info(
                f"[{source}] Processing {len(page_images)} pages with combined call"
            )
            try:
                return await self._enhance_with_frontmatter(
                    extracted_text, page_images, source
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
                frontmatter = self._build_fallback_frontmatter(source, cleaned)

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
                text_batches[0], image_batches[0], source
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
            frontmatter = self._build_fallback_frontmatter(source, extracted_text)

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
    ) -> tuple[str, str]:
        """Enhance document with vision and generate frontmatter in one call.

        Uses Instructor for structured output.

        Args:
            extracted_text: Text to clean
            page_images: Page images for visual reference
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        start_time = time.perf_counter()

        # Check persistent cache first
        # Use page count + source + text fingerprint as cache key
        page_names = "|".join(p.name for p in page_images[:10])  # First 10 page names
        cache_key = f"enhance_frontmatter:{source}:{len(page_images)}"
        cache_content = f"{page_names}|{extracted_text[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=source)
        if cached is not None:
            logger.debug(
                f"[{source}] Persistent cache hit for _enhance_with_frontmatter"
            )
            # Fix malformed image refs even for cached content (handles old cache entries)
            cleaned = self._fix_malformed_image_refs(cached.get("cleaned_markdown", ""))
            return cleaned, cached.get("frontmatter_yaml", "")

        # Extract protected content for fallback restoration
        protected = self.extract_protected_content(extracted_text)

        # Protect slide comments and images with placeholders before LLM processing
        protected_text, mapping = self._protect_content(extracted_text)

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt(
            "document_enhance_complete_system",
            source=source,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "document_enhance_complete_user",
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

        async with self.semaphore:
            # Calculate dynamic max_tokens using minimum across all vision router models
            # This ensures compatibility with any model the router might select
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]
            max_tokens = self._calculate_dynamic_max_tokens(  # type: ignore[attr-defined]
                messages, router=self.vision_router
            )

            # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
            client = instructor.from_litellm(
                self.vision_router.acompletion, mode=instructor.Mode.MD_JSON
            )
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            response, raw_response = await cast(
                Awaitable[tuple[EnhancedDocumentResult, Any]],
                client.chat.completions.create_with_completion(
                    model="default",
                    messages=cast(
                        list[ChatCompletionMessageParam],
                        messages,
                    ),
                    response_model=EnhancedDocumentResult,
                    max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                    max_tokens=max_tokens,
                ),
            )

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage and log completion
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                cost = get_response_cost(raw_response)
                self._track_usage(  # type: ignore[attr-defined]
                    actual_model, input_tokens, output_tokens, cost, source
                )

            # Log completion with timing
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                f"[LLM:{source}] vision_enhance: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} time={elapsed_ms}ms cost=${cost:.6f}"
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
                content=response.cleaned_markdown,
            )
            frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()

            # Restore protected content from placeholders
            # Pass protected dict for fallback restoration if LLM removed placeholders
            cleaned_markdown = self._unprotect_content(
                response.cleaned_markdown, mapping, protected
            )

            # Fix malformed image references (e.g., extra closing parentheses)
            cleaned_markdown = self._fix_malformed_image_refs(cleaned_markdown)

            # Store in persistent cache
            cache_value = {
                "cleaned_markdown": cleaned_markdown,
                "frontmatter_yaml": frontmatter_yaml,
            }
            self._persistent_cache.set(
                cache_key, cache_content, cache_value, model="vision"
            )

            return cleaned_markdown, frontmatter_yaml

    def _build_fallback_frontmatter(self, source: str, content: str) -> str:
        """Build fallback frontmatter when LLM fails.

        Uses programmatic utilities to generate consistent frontmatter structure.

        Args:
            source: Source filename
            content: Document content for title extraction

        Returns:
            YAML frontmatter string (without --- markers)
        """
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        frontmatter_dict = build_frontmatter_dict(
            source=source,
            description="",
            tags=[],
            content=content,
        )
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
        page_texts = self._split_text_by_pages(extracted_text, num_pages)

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
        page_texts = self._split_text_by_pages(extracted_text, num_pages)

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
    ) -> tuple[str, str]:
        """
        Process a document with LLM: clean and generate frontmatter.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Uses a combined prompt with Instructor for structured output,
        falling back to parallel separate calls if structured output fails.

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(markdown)
        protected_content, mapping = self._protect_content(markdown)

        # Try combined approach with Instructor first
        try:
            result = await self._process_document_combined(protected_content, source)

            # Restore protected content from placeholders, with fallback
            cleaned = self._unprotect_content(
                result.cleaned_markdown, mapping, protected
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
                content=cleaned,
            )
            frontmatter_yaml = frontmatter_to_yaml(frontmatter_dict).strip()
            logger.debug(f"[{source}] Used combined document processing")
            return cleaned, frontmatter_yaml
        except Exception as e:
            logger.debug(
                f"[{source}] Combined processing failed: {e}, using parallel fallback"
            )

        # Fallback: Run cleaning only (no longer use generate_frontmatter)
        # Use clean_markdown which has its own protection mechanism
        try:
            cleaned = await self.clean_markdown(markdown, context=source)
        except Exception as clean_err:
            logger.warning(
                f"Markdown cleaning failed: {format_error_message(clean_err)}"
            )
            cleaned = markdown

        # Build fallback frontmatter programmatically
        frontmatter = self._build_fallback_frontmatter(source, cleaned)

        return cleaned, frontmatter

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

        # Helper to reconstruct DocumentProcessResult from cached dict
        def _from_cache(cached: dict) -> DocumentProcessResult:
            return DocumentProcessResult(
                cleaned_markdown=cached.get("cleaned_markdown", ""),
                frontmatter=Frontmatter(
                    description=cached.get("description", ""),
                    tags=cached.get("tags", []),
                ),
            )

        # 1. Check in-memory cache first (fastest)
        cached = self._cache.get(cache_key, markdown)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"[{source}] Memory cache hit for _process_document_combined")
            return _from_cache(cached)

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, markdown, context=source)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"[{source}] Persistent cache hit for _process_document_combined"
            )
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, markdown, cached)
            return _from_cache(cached)

        self._cache_misses += 1

        # Detect document language
        language = get_language_name(detect_language(markdown))

        # Truncate content if needed (with warning)
        original_len = len(markdown)
        truncated_content = self._smart_truncate(markdown, DEFAULT_MAX_CONTENT_CHARS)
        if len(truncated_content) < original_len:
            logger.warning(
                f"[LLM:{source}] Content truncated: {original_len} -> {len(truncated_content)} chars "
                f"(limit: {DEFAULT_MAX_CONTENT_CHARS}). Some content may be lost."
            )

        # Get separated system and user prompts
        system_prompt = self._prompt_manager.get_prompt("document_process_system")
        user_prompt = self._prompt_manager.get_prompt(
            "document_process_user",
            content=truncated_content,
            source=source,
            language=language,
        )

        async with self.semaphore:
            start_time = time.perf_counter()

            # Build messages with separated system and user roles
            messages = cast(
                list[ChatCompletionMessageParam],
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # Calculate dynamic max_tokens using main router's model
            target_model_id = self._get_router_primary_model(self.router)  # type: ignore[attr-defined]
            max_tokens = self._calculate_dynamic_max_tokens(messages, target_model_id)  # type: ignore[attr-defined]

            # Create instructor client from router for load balancing
            # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
            client = instructor.from_litellm(
                self.router.acompletion, mode=instructor.Mode.MD_JSON
            )

            # Use create_with_completion to get both the model and the raw response
            # Use logical model name for router load balancing
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            response, raw_response = await cast(
                Awaitable[tuple[DocumentProcessResult, Any]],
                client.chat.completions.create_with_completion(
                    model="default",
                    messages=messages,
                    response_model=DocumentProcessResult,
                    max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                    max_tokens=max_tokens,
                ),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                cost = get_response_cost(raw_response)
                self._track_usage(  # type: ignore[attr-defined]
                    actual_model, input_tokens, output_tokens, cost, source
                )

            # Log detailed timing for performance analysis
            logger.info(
                f"[LLM:{source}] document_process: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
            )

            # Validate and clean prompt leakage
            validated_markdown = self._validate_no_prompt_leakage(
                response.cleaned_markdown, source
            )
            # If validation changed the content, update response
            if validated_markdown != response.cleaned_markdown:
                response = DocumentProcessResult(
                    cleaned_markdown=validated_markdown,
                    frontmatter=response.frontmatter,
                )

            # Store in both cache layers
            cache_value = {
                "cleaned_markdown": response.cleaned_markdown,
                "description": response.frontmatter.description,
                "tags": response.frontmatter.tags,
            }
            self._cache.set(cache_key, markdown, cache_value)
            self._persistent_cache.set(
                cache_key, markdown, cache_value, model="default"
            )

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
        source: str | None = None,  # noqa: ARG002
    ) -> str:
        """Format final output with frontmatter.

        Since frontmatter is now always generated programmatically with proper
        structure via build_frontmatter_dict() + frontmatter_to_yaml(), this
        function mainly handles markdown cleanup.

        Args:
            markdown: Cleaned markdown content
            frontmatter: YAML frontmatter (without --- markers)
            source: Optional source filename (unused, kept for API compatibility)

        Returns:
            Complete markdown with frontmatter
        """
        # Clean frontmatter (remove accidental --- markers)
        frontmatter = self._clean_frontmatter(frontmatter)

        # Clean markdown content
        markdown = self._remove_uncommented_screenshots(markdown)

        from markitai.utils.text import (
            clean_ppt_headers_footers,
            clean_residual_placeholders,
            fix_broken_markdown_links,
            normalize_markdown_whitespace,
        )

        markdown = fix_broken_markdown_links(markdown)
        markdown = clean_ppt_headers_footers(markdown)
        markdown = clean_residual_placeholders(markdown)
        markdown = normalize_markdown_whitespace(markdown)

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
            # removing user's original screenshots/ references (P0-5 fix).
            # markitai naming format: {filename}.page{NNNN}.{ext} in screenshots/
            # Patterns to remove:
            # 1. ![Page N](screenshots/*.page*.jpg) - markitai standard pattern
            # 2. ![...](screenshots/*.page*.jpg) - LLM-generated variants with same filename
            patterns = [
                # Matches: ![Page N](screenshots/anything.pageNNNN.jpg)
                r"^!\[Page\s+\d+\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
                # Matches: ![...](screenshots/anything.pageNNNN.jpg)
                r"^!\[[^\]]*\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
            ]
            for pattern in patterns:
                content = re.sub(pattern, "", content, flags=re.MULTILINE)

            # Also remove any page/image labels that LLM may have copied
            # Pattern: ## or ### Page N Image: followed by empty line (legacy format)
            # Pattern: [Page N] or [Image N] on its own line (simple format)
            # Pattern: __MARKITAI_PAGE_LABEL_N__ or __MARKITAI_IMG_LABEL_N__ (unique format)
            content = re.sub(
                r"^#{2,3}\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\[(Page|Image)\s+\d+\]\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^__MARKITAI_(PAGE|IMG)_LABEL_\d+__\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            # Remove any leftover slide placeholders (shouldn't exist but cleanup)
            content = re.sub(
                r"^__MARKITAI_SLIDE_\d+__\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )

            # Clean up any resulting empty lines
            content = re.sub(r"\n{3,}", "\n\n", content)
        else:
            # Split at the page images section
            before = content[:header_pos]
            after = content[header_pos:]

            # Remove screenshot references from BEFORE the page images header
            # IMPORTANT: Only match markitai-generated screenshot patterns (P0-5 fix)
            patterns = [
                # Matches: ![Page N](screenshots/anything.pageNNNN.jpg)
                r"^!\[Page\s+\d+\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
                # Matches: ![...](screenshots/anything.pageNNNN.jpg)
                r"^!\[[^\]]*\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
            ]
            for pattern in patterns:
                before = re.sub(pattern, "", before, flags=re.MULTILINE)

            # Also remove any page/image labels that LLM may have copied
            before = re.sub(
                r"^#{2,3}\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(
                r"^\[(Page|Image)\s+\d+\]\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(
                r"^__MARKITAI_(PAGE|IMG)_LABEL_\d+__\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            # Remove any leftover slide placeholders (shouldn't exist but cleanup)
            before = re.sub(
                r"^__MARKITAI_SLIDE_\d+__\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(r"\n{3,}", "\n\n", before)

            # Fix the AFTER section: convert any non-commented page images to comments
            # Match lines with page image references that are not already commented
            # This handles: ![Page N](screenshots/...)
            after_lines = after.split("\n")
            fixed_lines = []
            for line in after_lines:
                stripped = line.strip()
                # Check if it's an uncommented page image reference
                if (
                    stripped.startswith("![Page")
                    and "screenshots/" in stripped
                    and not stripped.startswith("<!--")
                ):
                    fixed_lines.append(f"<!-- {stripped} -->")
                else:
                    fixed_lines.append(line)
            after = "\n".join(fixed_lines)

            content = before + after

        # Clean up screenshot comments section: remove blank lines between comments
        # Pattern: <!-- Page images for reference --> followed by page image comments
        page_section_pattern = (
            r"(<!-- Page images for reference -->)"
            r"((?:\s*<!-- !\[Page \d+\]\([^)]+\) -->)+)"
        )

        def clean_page_section(match: re.Match) -> str:
            header = match.group(1)
            comments_section = match.group(2)
            # Extract individual comments and rejoin without blank lines
            comments = re.findall(r"<!-- !\[Page \d+\]\([^)]+\) -->", comments_section)
            return header + "\n" + "\n".join(comments)

        content = re.sub(page_section_pattern, clean_page_section, content)

        return content

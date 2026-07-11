"""Vision analysis service for LLMProcessor.

This module provides the VisionAnalyzer service class for image analysis.
LLMProcessor composes it (lazy ``vision`` property) and exposes thin
delegates for the public methods.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import instructor
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from markitai.constants import (
    DEFAULT_INSTRUCTOR_MAX_RETRIES,
    DEFAULT_MAX_IMAGES_PER_BATCH,
)
from markitai.llm.degeneration import truncate_degenerate_tail
from markitai.llm.engine import LLMCall
from markitai.llm.engine import (
    try_repair_instructor_response as _try_repair_instructor_response,
)
from markitai.llm.models import context_display_name, get_response_cost
from markitai.llm.types import (
    BatchImageAnalysisResult,
    ImageAnalysis,
    ImageAnalysisResult,
    LLMResponse,
    SingleImageResult,
)
from markitai.providers.common import has_images
from markitai.utils.mime import (
    get_llm_effective_mime,
    is_llm_supported_image,
)
from markitai.utils.text import clean_control_characters, format_error_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from markitai.config import LLMConfig
    from markitai.llm.engine import LLMEngine
    from markitai.prompts import PromptManager
    from markitai.types import LLMUsageByModel, ModelUsageStats


def _vision_cache_content_key(
    image_fingerprint: str, document_context: str = ""
) -> str:
    """Build a cache content key for vision analysis.

    Incorporates document_context hash when present so that the same
    image analyzed in different documents produces separate cache entries.

    Args:
        image_fingerprint: SHA-256 hex digest of the image data.
        document_context: Optional document context text.

    Returns:
        Content key string for use with PersistentCache.
    """
    versioned_key = f"{image_fingerprint}|vision:v2"
    if not document_context:
        return versioned_key
    ctx_hash = hashlib.sha256(document_context.encode()).hexdigest()[:16]
    return f"{versioned_key}|ctx:{ctx_hash}"


def _detect_document_language(document_context: str) -> str:
    """Infer a simple zh/en language hint from document context."""
    if not document_context:
        return "English"

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", document_context))
    latin_chars = len(re.findall(r"[A-Za-z]", document_context))
    if cjk_chars >= 4 and cjk_chars >= latin_chars:
        return "Chinese"
    if latin_chars >= 12 and cjk_chars == 0:
        return "English"
    return "English"


def _text_matches_language(text: str, language: str) -> bool:
    """Return True when text clearly matches the requested language hint."""
    if not text.strip():
        return False

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if language == "Chinese":
        return cjk_chars >= 2
    if language == "English":
        return cjk_chars == 0 and latin_chars >= 8
    return True


def _align_batch_results(
    images: list[SingleImageResult], expected_count: int
) -> tuple[list[SingleImageResult | None], bool]:
    """Align batch results to input positions using image_index.

    The batch prompt asks the model to echo a 1-based image_index for each
    result. When those indices are valid (in range, unique), use them to
    align results even if the model skipped or reordered images. Otherwise
    fall back to positional alignment.

    Args:
        images: Results returned by the model.
        expected_count: Number of images sent in the batch.

    Returns:
        Tuple of (aligned, cache_safe). ``aligned[pos]`` is the result for
        input position ``pos`` (0-based) or None when missing. ``cache_safe``
        is False when alignment is ambiguous (invalid indices AND count
        mismatch), in which case results must not be persisted to cache.
    """
    indices = [img.image_index for img in images]
    indices_valid = (
        len(images) > 0
        and all(1 <= i <= expected_count for i in indices)
        and len(set(indices)) == len(indices)
    )

    aligned: list[SingleImageResult | None] = [None] * expected_count
    if indices_valid:
        for img in images:
            aligned[img.image_index - 1] = img
        return aligned, True

    # Positional fallback: only trustworthy when counts match exactly
    for pos, img in enumerate(images[:expected_count]):
        aligned[pos] = img
    return aligned, len(images) == expected_count


def _should_retry_for_language(result: ImageAnalysis, language: str) -> bool:
    """Retry when a no-text image ignores the requested document language."""
    if result.extracted_text and str(result.extracted_text).strip():
        return False

    combined = f"{result.caption} {result.description}".strip()
    return not _text_matches_language(combined, language)


def _merge_llm_usage(
    base: LLMUsageByModel | dict[str, Any] | None,
    extra: LLMUsageByModel | dict[str, Any] | None,
) -> LLMUsageByModel:
    """Merge llm_usage dicts without importing workflow helpers."""
    merged: dict[str, Any] = copy.deepcopy(dict(base)) if base else {}
    if not extra:
        return merged  # type: ignore[return-value]

    for model, usage in extra.items():
        if model not in merged:
            merged[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        merged[model]["requests"] = merged[model].get("requests", 0) + usage.get(
            "requests", 0
        )
        merged[model]["input_tokens"] = merged[model].get(
            "input_tokens", 0
        ) + usage.get("input_tokens", 0)
        merged[model]["output_tokens"] = merged[model].get(
            "output_tokens", 0
        ) + usage.get("output_tokens", 0)
        merged[model]["cost_usd"] = merged[model].get("cost_usd", 0.0) + usage.get(
            "cost_usd", 0.0
        )

    return merged  # type: ignore[return-value]


def _guard_degenerate_extracted_text(analysis: ImageAnalysis, context: str) -> bool:
    """Truncate a degenerate repetition tail in extracted_text in place.

    Returns:
        True if truncation happened (callers should skip cache persist).
    """
    if not analysis.extracted_text:
        return False
    truncated, degenerated = truncate_degenerate_tail(
        analysis.extracted_text, context=context, stage="image_analysis"
    )
    if degenerated:
        analysis.extracted_text = truncated
    return degenerated


class VisionAnalyzer:
    """Vision analysis service used by LLMProcessor via composition.

    Provides image analysis functionality including:
    - Single image analysis
    - Batch image analysis
    - Page content extraction

    Dependencies are injected explicitly instead of being reached through a
    mixin host:

    - ``engine``: transport (text/structured calls), shared semaphore,
      cache layers, and usage accounting
    - ``prompt_manager``: prompt template lookup
    - ``config``: LLM configuration (concurrency, retry counts)
    - ``get_vision_router``: provider callback for the vision router.
      A callback (not the router instance) because the processor's
      vision router is a lazy property and tests inject doubles after
      construction — holding the instance would pin the pre-injection
      object.
    - ``get_cached_image``: processor-owned image cache accessor
      (the LRU image cache state stays on LLMProcessor)
    - ``get_next_call_index``: processor-owned per-context call counter
      (shared with the processor so call ids stay globally sequential)
    """

    def __init__(
        self,
        *,
        engine: LLMEngine,
        prompt_manager: PromptManager,
        config: LLMConfig,
        get_vision_router: Callable[[], Any],
        get_cached_image: Callable[[Path], tuple[bytes, str]],
        get_next_call_index: Callable[[str], int],
    ) -> None:
        self._engine = engine
        self._prompt_manager = prompt_manager
        self._config = config
        self._get_vision_router = get_vision_router
        self._get_cached_image = get_cached_image
        self._get_next_call_index = get_next_call_index

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        context: str = "",
    ) -> LLMResponse:
        """Make a text LLM call with smart router selection.

        Mirrors ``LLMProcessor._call_llm``: uses the vision router when the
        messages contain images, the engine's default router otherwise.
        """
        call_index = self._get_next_call_index(context) if context else 0
        call_id = f"{context}:{call_index}" if context else f"call:{call_index}"
        router = self._get_vision_router() if has_images(messages) else None
        return await self._engine.complete_text(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            max_retries=self._config.router_settings.num_retries,
            router=router,
        )

    async def analyze_image(
        self,
        image_path: Path,
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """
        Analyze an image using vision model.

        Uses Instructor for structured output with fallback mechanisms:
        1. Try Instructor with structured output
        2. Fallback to JSON mode + manual parsing
        3. Fallback to original two-call method

        Args:
            image_path: Path to the image file
            context: Context identifier for usage tracking (e.g., source filename)
            document_context: Short text snippet from the surrounding document,
                used as language hint for images without visible text.

        Returns:
            ImageAnalysis with caption and description
        """
        # Filter unsupported image formats (SVG, BMP, ICO etc.)
        if not is_llm_supported_image(image_path.suffix):
            logger.debug(
                f"[{image_path.name}] Skipping unsupported format: {image_path.suffix}"
            )
            return ImageAnalysis(
                caption=image_path.stem,
                description=f"Image format {image_path.suffix} not supported for analysis",
            )

        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Check persistent cache using image hash as key
        # Use SHA256 hash of base64 as image fingerprint to avoid collisions
        # (JPEG files share the same header, so first N chars are identical)
        cache_key = "image_analysis"
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
        cache_content_key = _vision_cache_content_key(
            image_fingerprint, document_context
        )
        cached = self._engine.persistent_cache.get(
            cache_key, cache_content_key, context=context
        )
        if cached is not None:
            logger.debug(f"[{image_path.name}] Persistent cache hit for analyze_image")
            # Reconstruct ImageAnalysis from cached dict
            return ImageAnalysis(
                caption=cached.get("caption", ""),
                description=cached.get("description", ""),
                extracted_text=cached.get("extracted_text"),
            )

        # Determine MIME type (converts BMP/TIFF → image/png)
        mime_type = get_llm_effective_mime(image_path.suffix)

        # Use separated system/user prompts to improve instruction following
        language = _detect_document_language(document_context)
        system_prompt = self._prompt_manager.get_prompt(
            "image_analysis_system",
            language=language,
        )
        doc_ctx = (
            f"\n\nDocument context: {document_context}" if document_context else ""
        )
        user_prompt = self._prompt_manager.get_prompt(
            "image_analysis_user",
            document_context=doc_ctx,
            language=language,
        )

        # Build message with system role and user role with image
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            },
        ]

        # Use "default" model name - smart router will auto-select vision-capable model
        # since the message contains image content
        vision_model = "default"

        # Try structured output methods with fallbacks
        result = await self._analyze_image_with_fallback(
            messages,
            vision_model,
            image_path.name,
            context,
            document_context=document_context,
        )

        if _should_retry_for_language(result, language):
            logger.debug(
                f"[{image_path.name}] Retrying image analysis with stronger "
                f"{language} language constraint"
            )
            retry_messages = copy.deepcopy(messages)
            retry_instruction = (
                "\n\nCRITICAL: This image appears to contain no readable text. "
                f"Return the caption and description in {language}. "
                "Do not answer in another language."
            )
            retry_messages[0]["content"] += retry_instruction
            retry_messages[1]["content"][0]["text"] += retry_instruction
            result = await self._analyze_image_with_fallback(
                retry_messages,
                vision_model,
                image_path.name,
                context,
                document_context=document_context,
            )

        if _should_retry_for_language(result, language):
            logger.debug(
                f"[{image_path.name}] Rewriting image analysis into {language} "
                "after multimodal retries"
            )
            result = await self._rewrite_analysis_language(
                result,
                language=language,
                context=context or image_path.name,
                document_context=document_context,
            )

        # Guard against VLM degeneration; skip cache persist when truncated
        degenerated = _guard_degenerate_extracted_text(result, image_path.name)

        # Store in persistent cache
        if not degenerated:
            cache_value = {
                "caption": result.caption,
                "description": result.description,
                "extracted_text": result.extracted_text,
            }
            self._engine.persistent_cache.set(
                cache_key, cache_content_key, cache_value, model="vision"
            )

        return result

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        max_images_per_batch: int = DEFAULT_MAX_IMAGES_PER_BATCH,
        context: str = "",
        document_context: str = "",
    ) -> list[ImageAnalysis]:
        """
        Analyze multiple images in batches with parallel execution.

        Batches are processed concurrently using asyncio.gather for better
        throughput. Two levels of concurrency control:
        - Batch-level semaphore limits concurrent batches (prevents memory
          pressure from loading all images at once)
        - LLM-level semaphore controls concurrent API calls

        Args:
            image_paths: List of image paths to analyze
            max_images_per_batch: Max images per LLM call (default 10)
            context: Context identifier for usage tracking (e.g., source filename)
            document_context: Short text snippet for language hinting.

        Returns:
            List of ImageAnalysis results in same order as input
        """
        if not image_paths:
            return []

        # Split into batches
        num_batches = (
            len(image_paths) + max_images_per_batch - 1
        ) // max_images_per_batch

        batches: list[tuple[int, list[Path]]] = []
        for batch_num in range(num_batches):
            batch_start = batch_num * max_images_per_batch
            batch_end = min(batch_start + max_images_per_batch, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batches.append((batch_num, batch_paths))

        # Limit concurrent batches to avoid memory pressure from loading all images
        # at once. The semaphore controls LLM API calls, but images are loaded
        # before acquiring the semaphore. This batch-level limit prevents that.
        max_concurrent_batches = min(self._config.concurrency, num_batches)
        batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        display_name = context_display_name(context) or "batch"
        logger.info(
            f"[{display_name}] Analyzing {len(image_paths)} images in "
            f"{num_batches} batches (max {max_concurrent_batches} concurrent)"
        )

        # Process batches with backpressure and streaming
        async def process_batch(
            batch_num: int, batch_paths: list[Path]
        ) -> tuple[int, list[ImageAnalysis]]:
            """Process a single batch with backpressure control."""
            async with batch_semaphore:
                try:
                    results = await self.analyze_batch(
                        batch_paths,
                        context=context,
                        document_context=document_context,
                    )
                    return (batch_num, results)
                except Exception as e:
                    logger.warning(
                        f"[{display_name}] Batch {batch_num + 1}/{num_batches} failed: {e}"
                    )
                    # Return empty results with placeholder for failed images
                    return (
                        batch_num,
                        [
                            ImageAnalysis(
                                caption=f"Image {i + 1}",
                                description="Analysis failed",
                            )
                            for i in range(len(batch_paths))
                        ],
                    )

        # Launch all batches and process results as they complete
        # Using as_completed allows earlier batches to free resources sooner
        tasks = {
            asyncio.create_task(process_batch(batch_num, paths)): batch_num
            for batch_num, paths in batches
        }

        batch_results: list[tuple[int, list[ImageAnalysis]]] = []
        for coro in asyncio.as_completed(tasks.keys()):
            try:
                result = await coro
                batch_results.append(result)
            except Exception as e:
                # Find which batch failed by checking tasks
                logger.error(
                    f"[{display_name}] Batch processing error: "
                    f"{format_error_message(e)}"
                )

        # Sort by batch number and flatten results
        batch_results_sorted = sorted(batch_results, key=lambda x: x[0])
        all_results: list[ImageAnalysis] = []
        for _, results in batch_results_sorted:
            all_results.extend(results)

        return all_results

    async def analyze_batch(
        self,
        image_paths: list[Path],
        context: str = "",
        document_context: str = "",
    ) -> list[ImageAnalysis]:
        """Batch image analysis using Instructor.

        Uses the same prompt template as single image analysis for consistency.
        Checks persistent cache first and only calls LLM for uncached images.

        Args:
            image_paths: List of image paths to analyze
            context: Context identifier for usage tracking
            document_context: Short text snippet for language hinting.

        Returns:
            List of ImageAnalysis results
        """
        # Filter unsupported formats and track their indices
        unsupported_results: dict[int, ImageAnalysis] = {}
        supported_paths: list[tuple[int, Path]] = []
        for i, image_path in enumerate(image_paths):
            if not is_llm_supported_image(image_path.suffix):
                logger.debug(
                    f"[{image_path.name}] Skipping unsupported format: {image_path.suffix}"
                )
                unsupported_results[i] = ImageAnalysis(
                    caption=image_path.stem,
                    description=f"Image format {image_path.suffix} not supported for analysis",
                )
            else:
                supported_paths.append((i, image_path))

        # If all images are unsupported, return placeholder results
        if not supported_paths:
            return [unsupported_results[i] for i in range(len(image_paths))]

        # Check persistent cache for all images first
        # Use same cache key format as analyze_image for consistency
        cache_key = "image_analysis"
        cached_results: dict[int, ImageAnalysis] = {}
        uncached_indices: list[int] = []
        image_fingerprints: dict[int, str] = {}
        image_cache_content_keys: dict[int, str] = {}

        for orig_idx, image_path in supported_paths:
            _, base64_image = self._get_cached_image(image_path)
            # Use SHA256 hash to avoid collisions (JPEG files share same header)
            fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
            image_fingerprints[orig_idx] = fingerprint
            cache_content_key = _vision_cache_content_key(fingerprint, document_context)
            image_cache_content_keys[orig_idx] = cache_content_key

            cached = self._engine.persistent_cache.get(
                cache_key, cache_content_key, context=context
            )
            if cached is not None:
                logger.debug(f"[{image_path.name}] Cache hit in batch analysis")
                cached_results[orig_idx] = ImageAnalysis(
                    caption=cached.get("caption", ""),
                    description=cached.get("description", ""),
                    extracted_text=cached.get("extracted_text"),
                )
            else:
                uncached_indices.append(orig_idx)

        # If all supported images are cached, return merged results
        display_name = context_display_name(context) or "batch"
        if not uncached_indices:
            logger.info(
                f"[{display_name}] All {len(supported_paths)} supported images found in cache"
            )
            # Merge unsupported and cached results
            return [
                unsupported_results.get(i) or cached_results[i]
                for i in range(len(image_paths))
            ]

        # Only process uncached images
        uncached_paths = [image_paths[i] for i in uncached_indices]
        logger.debug(
            f"[{display_name}] Cache: {len(cached_results)} hits, "
            f"{len(uncached_indices)} misses"
        )

        # Use separated system/user prompts to improve instruction following
        language = _detect_document_language(document_context)
        system_prompt = self._prompt_manager.get_prompt(
            "image_analysis_system",
            language=language,
        )

        # Build batch user prompt
        batch_header = f"Analyze the following {len(uncached_paths)} images in order."
        doc_ctx = (
            f"\n\nDocument context: {document_context}" if document_context else ""
        )
        language_hint = (
            f"\n\nFallback language for images without visible text: {language}."
        )
        batch_footer = (
            "\n\nReturn a JSON object with an 'images' array containing results "
            "for each image in order."
        )
        user_prompt = f"{batch_header}{doc_ctx}{language_hint}{batch_footer}"

        # Build content parts with uncached images only
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        for i, image_path in enumerate(uncached_paths, 1):
            _, base64_image = self._get_cached_image(image_path)
            mime_type = get_llm_effective_mime(image_path.suffix)

            # Unique image label that won't conflict with document content
            content_parts.append(
                {"type": "text", "text": f"\n__MARKITAI_IMG_LABEL_{i}__"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        try:
            vision_router = self._get_vision_router()
            async with self._engine.semaphore:
                # Calculate dynamic max_tokens using minimum across all vision router models
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_parts},
                ]
                max_tokens = self._engine.calculate_max_tokens(
                    messages,
                    router=vision_router,
                )

                # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
                client = instructor.from_litellm(
                    vision_router.acompletion,
                    mode=instructor.Mode.MD_JSON,
                )
                # max_retries allows Instructor to retry with validation error
                # feedback, which helps LLM fix JSON escaping issues
                try:
                    response, raw_response = await cast(
                        Awaitable[tuple[BatchImageAnalysisResult, Any]],
                        client.chat.completions.create_with_completion(
                            model="default",
                            messages=cast(
                                list[ChatCompletionMessageParam],
                                messages,
                            ),
                            response_model=BatchImageAnalysisResult,
                            max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                            max_tokens=max_tokens,
                        ),
                    )
                except Exception as e:
                    repaired = _try_repair_instructor_response(
                        e, BatchImageAnalysisResult
                    )
                    if repaired is None:
                        raise
                    response, raw_response = repaired

                # Check for truncation
                if hasattr(raw_response, "choices") and raw_response.choices:
                    finish_reason = getattr(
                        raw_response.choices[0], "finish_reason", None
                    )
                    if finish_reason == "length":
                        raise ValueError("Output truncated due to max_tokens limit")

                # Track usage
                actual_model = getattr(raw_response, "model", None) or "default"
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
                if hasattr(raw_response, "usage") and raw_response.usage is not None:
                    input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                    output_tokens = (
                        getattr(raw_response.usage, "completion_tokens", 0) or 0
                    )
                    cost = get_response_cost(raw_response)
                    self._engine.track_usage(
                        actual_model, input_tokens, output_tokens, cost, context
                    )

                # Calculate per-image usage (divide batch usage by number of images)
                num_images = max(len(response.images), 1)
                per_image_llm_usage: LLMUsageByModel = {
                    actual_model: cast(
                        "ModelUsageStats",
                        {
                            "requests": 1,  # Each image counts as 1 request share
                            "input_tokens": input_tokens // num_images,
                            "output_tokens": output_tokens // num_images,
                            "cost_usd": cost / num_images,
                        },
                    )
                }

            # Semaphore released above. Language rewrites go through
            # _call_llm which re-acquires the same semaphore, so doing them
            # while holding it would deadlock (e.g. with concurrency=1).

            # Align results to input positions using image_index
            aligned, cache_safe = _align_batch_results(
                list(response.images), len(uncached_paths)
            )
            if not cache_safe:
                logger.warning(
                    f"[{display_name}] Batch result alignment ambiguous "
                    f"({len(response.images)} results for "
                    f"{len(uncached_paths)} images); skipping cache persist"
                )

            # Convert to ImageAnalysis list and store in cache
            new_results: list[ImageAnalysis] = []
            for pos in range(len(uncached_paths)):
                img_result = aligned[pos]
                if img_result is None:
                    new_results.append(
                        ImageAnalysis(
                            caption="Image",
                            description="Image analysis failed",
                            extracted_text=None,
                            llm_usage=per_image_llm_usage,
                        )
                    )
                    continue

                analysis = ImageAnalysis(
                    caption=img_result.caption,
                    description=img_result.description,
                    extracted_text=img_result.extracted_text,
                    llm_usage=per_image_llm_usage,
                )
                if _should_retry_for_language(analysis, language):
                    analysis = await self._rewrite_analysis_language(
                        analysis,
                        language=language,
                        context=context or display_name,
                        document_context=document_context,
                    )
                # Guard against VLM degeneration; skip cache persist when truncated
                degenerated = _guard_degenerate_extracted_text(
                    analysis, uncached_paths[pos].name
                )
                new_results.append(analysis)

                # Store in persistent cache using original index
                if cache_safe and not degenerated:
                    original_idx = uncached_indices[pos]
                    content_key = image_cache_content_keys[original_idx]
                    cache_value = {
                        "caption": analysis.caption,
                        "description": analysis.description,
                        "extracted_text": analysis.extracted_text,
                    }
                    self._engine.persistent_cache.set(
                        cache_key, content_key, cache_value, model="vision"
                    )

            # Merge unsupported, cached and new results in original order
            final_results: list[ImageAnalysis] = []
            new_result_iter = iter(new_results)
            for i in range(len(image_paths)):
                if i in unsupported_results:
                    final_results.append(unsupported_results[i])
                elif i in cached_results:
                    final_results.append(cached_results[i])
                else:
                    final_results.append(next(new_result_iter))

            return final_results

        except Exception as e:
            logger.error(
                f"Batch image analysis failed: {format_error_message(e)}, "
                "falling back to individual analysis"
            )

            # Fallback: analyze each image concurrently (uses persistent cache)
            async def _analyze_one(i: int, image_path: Path) -> ImageAnalysis:
                if i in unsupported_results:
                    return unsupported_results[i]
                if i in cached_results:
                    return cached_results[i]
                try:
                    return await self.analyze_image(
                        image_path,
                        context=context,
                        document_context=document_context,
                    )
                except Exception as e:
                    logger.debug(
                        "[Vision] Image analysis failed for {}: {}", image_path.name, e
                    )
                    return ImageAnalysis(
                        caption="Image",
                        description="Image analysis failed",
                        extracted_text=None,
                    )

            fallback_results = list(
                await asyncio.gather(
                    *[
                        _analyze_one(i, image_path)
                        for i, image_path in enumerate(image_paths)
                    ]
                )
            )
            return fallback_results

    async def _analyze_image_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        image_name: str,
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """
        Analyze image with multiple fallback strategies.

        Strategy 1: Instructor structured output (most precise)
        Strategy 2: JSON mode + manual parsing
        Strategy 3: Original two-call method (most compatible)

        Args:
            messages: LLM messages with image
            model: Model name to use
            image_name: Image filename for logging
            context: Context identifier for usage tracking
            document_context: Short text snippet for language hinting.
        """
        # Strategy 1: Try Instructor
        try:
            # Deep copy to prevent Instructor from modifying original messages
            result = await self._analyze_with_instructor(
                copy.deepcopy(messages), model, context
            )
            return result
        except Exception as e:
            logger.debug(f"[{image_name}] Instructor failed: {e}, trying JSON mode")

        # Strategy 2: Try JSON mode
        try:
            result = await self._analyze_with_json_mode(
                copy.deepcopy(messages), model, context
            )
            logger.debug(f"[{image_name}] Used JSON mode fallback")
            return result
        except Exception as e:
            logger.debug(
                f"[{image_name}] JSON mode failed: {e}, using two-call fallback"
            )

        # Strategy 3: Original two-call method
        return await self._analyze_with_two_calls(
            copy.deepcopy(messages),
            context=context or image_name,
            document_context=document_context,
        )

    async def _analyze_with_instructor(
        self,
        messages: list[dict[str, Any]],
        model: str,
        context: str = "",
    ) -> ImageAnalysis:
        """Analyze using Instructor for structured output.

        cache_key=None: caching lives in the callers (analyze_image /
        analyze_batch), keyed by image fingerprint + document context.

        Any exception (ProviderError, InstructorRetryException, truncation
        ValueError) must propagate: _analyze_image_with_fallback catches all
        and moves on to the json_mode / two_calls fallback strategies.
        """
        call = LLMCall(
            purpose="image_analysis",
            messages=messages,
            response_model=ImageAnalysisResult,
            context=context,
            cache_key=None,
            router=self._get_vision_router(),
        )
        response, raw_response = await self._engine.complete_structured(call)

        # Build llm_usage dict for this analysis from the raw response
        # (aggregate usage accounting itself is handled by the engine)
        actual_model = getattr(raw_response, "model", None) or model
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        if hasattr(raw_response, "usage") and raw_response.usage is not None:
            input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
            cost = get_response_cost(raw_response)

        llm_usage: LLMUsageByModel = {
            actual_model: cast(
                "ModelUsageStats",
                {
                    "requests": 1,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                },
            )
        }

        return ImageAnalysis(
            caption=response.caption.strip(),
            description=response.description,
            extracted_text=response.extracted_text,
            llm_usage=llm_usage,
        )

    async def _analyze_with_json_mode(
        self,
        messages: list[dict[str, Any]],
        model: str,
        context: str = "",
    ) -> ImageAnalysis:
        """Analyze using JSON mode with manual parsing."""
        # Add JSON instruction to the prompt
        json_messages = messages.copy()
        json_messages[1] = {
            **messages[1],
            "content": [
                {
                    "type": "text",
                    "text": messages[1]["content"][0]["text"]
                    + "\n\nReturn a JSON object with 'caption' and 'description' fields.",
                },
                messages[1]["content"][1],  # image
            ],
        }

        vision_router = self._get_vision_router()
        async with self._engine.semaphore:
            # Calculate dynamic max_tokens using minimum across all vision router models
            max_tokens = self._engine.calculate_max_tokens(
                json_messages,
                router=vision_router,
            )

            # Use vision_router for image analysis (not main router)
            response = await vision_router.acompletion(
                model=model,
                messages=cast(list[AllMessageValues], json_messages),
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            # litellm returns Choices (not StreamingChoices) for non-streaming
            choice = cast(Choices, response.choices[0])
            content = choice.message.content if choice.message else "{}"
            actual_model = response.model or model

            # Track usage
            usage = getattr(response, "usage", None)
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = get_response_cost(response)
            self._engine.track_usage(
                actual_model, input_tokens, output_tokens, cost, context
            )

            # Clean control characters before JSON parsing to avoid errors
            cleaned_content = clean_control_characters(content or "{}")
            # Parse JSON
            data = json.loads(cleaned_content)

            # Build llm_usage dict for this analysis
            llm_usage: LLMUsageByModel = {
                actual_model: cast(
                    "ModelUsageStats",
                    {
                        "requests": 1,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                    },
                )
            }

            return ImageAnalysis(
                caption=data.get("caption", "").strip(),
                description=data.get("description", ""),
                extracted_text=data.get("extracted_text"),
                llm_usage=llm_usage,
            )

    async def _analyze_with_two_calls(
        self,
        messages: list[dict[str, Any]],
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """Original two-call method as final fallback."""
        # Extract image from messages (handle both old and new format)
        # New format: [system_msg, user_msg_with_image]
        # Old format: [user_msg_with_image]
        if messages[0].get("role") == "system":
            user_content = messages[1]["content"]
        else:
            user_content = messages[0]["content"]

        image_content = user_content[1]  # The image part
        language = _detect_document_language(document_context)

        # Generate caption using system/user prompts
        caption_system = self._prompt_manager.get_prompt(
            "image_caption_system",
            language=language,
        )
        doc_ctx = (
            f"\n\nDocument context: {document_context}" if document_context else ""
        )
        caption_user = self._prompt_manager.get_prompt(
            "image_caption_user",
            document_context=doc_ctx,
            language=language,
        )
        caption_response = await self._call_llm(
            model="default",
            messages=[
                {"role": "system", "content": caption_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_user},
                        image_content,
                    ],
                },
            ],
            context=context,
        )

        # Generate description using system/user prompts
        desc_system = self._prompt_manager.get_prompt(
            "image_description_system",
            language=language,
        )
        desc_user = self._prompt_manager.get_prompt(
            "image_description_user",
            document_context=doc_ctx,
            language=language,
        )
        desc_response = await self._call_llm(
            model="default",
            messages=[
                {"role": "system", "content": desc_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": desc_user},
                        image_content,
                    ],
                },
            ],
            context=context,
        )

        # Build aggregated llm_usage from both calls
        llm_usage: LLMUsageByModel = {}
        for resp in [caption_response, desc_response]:
            if resp.model not in llm_usage:
                llm_usage[resp.model] = cast(
                    "ModelUsageStats",
                    {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
            llm_usage[resp.model]["requests"] += 1
            llm_usage[resp.model]["input_tokens"] += resp.input_tokens
            llm_usage[resp.model]["output_tokens"] += resp.output_tokens
            llm_usage[resp.model]["cost_usd"] += resp.cost_usd

        return ImageAnalysis(
            caption=caption_response.content.strip(),
            description=desc_response.content,
            llm_usage=llm_usage,
        )

    async def _rewrite_analysis_language(
        self,
        result: ImageAnalysis,
        *,
        language: str,
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """Rewrite caption/description into the document language when needed."""
        rewritten_caption = result.caption
        rewritten_description = result.description
        merged_usage = copy.deepcopy(result.llm_usage) if result.llm_usage else {}

        if rewritten_caption.strip() and not _text_matches_language(
            rewritten_caption, language
        ):
            try:
                caption_response = await self._call_llm(
                    model="default",
                    messages=self._build_language_rewrite_messages(
                        content=rewritten_caption,
                        language=language,
                        field_name="alt text",
                        document_context=document_context,
                    ),
                    context=context,
                )
                if caption_response.content.strip():
                    rewritten_caption = caption_response.content.strip()
                merged_usage = _merge_llm_usage(
                    merged_usage,
                    {
                        caption_response.model: {
                            "requests": 1,
                            "input_tokens": caption_response.input_tokens,
                            "output_tokens": caption_response.output_tokens,
                            "cost_usd": caption_response.cost_usd,
                        }
                    },
                )
            except Exception as e:
                logger.debug(
                    "[Vision] Caption language rewrite failed for {}: {}",
                    context or "image",
                    e,
                )

        if rewritten_description.strip() and not _text_matches_language(
            rewritten_description, language
        ):
            try:
                description_response = await self._call_llm(
                    model="default",
                    messages=self._build_language_rewrite_messages(
                        content=rewritten_description,
                        language=language,
                        field_name="markdown description",
                        document_context=document_context,
                    ),
                    context=context,
                )
                if description_response.content.strip():
                    rewritten_description = description_response.content.strip()
                merged_usage = _merge_llm_usage(
                    merged_usage,
                    {
                        description_response.model: {
                            "requests": 1,
                            "input_tokens": description_response.input_tokens,
                            "output_tokens": description_response.output_tokens,
                            "cost_usd": description_response.cost_usd,
                        }
                    },
                )
            except Exception as e:
                logger.debug(
                    "[Vision] Description language rewrite failed for {}: {}",
                    context or "image",
                    e,
                )

        return ImageAnalysis(
            caption=rewritten_caption,
            description=rewritten_description,
            extracted_text=result.extracted_text,
            llm_usage=merged_usage or None,
        )

    def _build_language_rewrite_messages(
        self,
        *,
        content: str,
        language: str,
        field_name: str,
        document_context: str = "",
    ) -> list[dict[str, str]]:
        """Build a text-only rewrite prompt that preserves meaning and structure."""
        doc_ctx = (
            f"\n\nDocument context: {document_context}" if document_context else ""
        )
        preserve_structure = (
            " Preserve markdown formatting, headings, and lists."
            if field_name == "markdown description"
            else ""
        )
        system_prompt = (
            f"Rewrite the following {field_name} into {language}."
            " Preserve the original meaning."
            f"{preserve_structure}"
            " Return only the rewritten text."
        )
        user_prompt = f"Original {field_name}:\n{content}{doc_ctx}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def extract_page_content(self, image_path: Path, context: str = "") -> str:
        """
        Extract text content from a document page image.

        Used for OCR+LLM mode and PPTX+LLM mode where pages are rendered
        as images and we want to extract structured text content.

        Args:
            image_path: Path to the page image file
            context: Context identifier for logging (e.g., parent document name)

        Returns:
            Extracted markdown content from the page
        """
        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Determine MIME type (converts BMP/TIFF → image/png)
        mime_type = get_llm_effective_mime(image_path.suffix)

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt("page_content_system")
        user_prompt = self._prompt_manager.get_prompt("page_content_user")

        # Use image_path.name as context if not provided
        call_context = context or image_path.name

        response = await self._call_llm(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            context=call_context,
        )

        # Guard against VLM degeneration (repetition loops) in page text
        content, _ = truncate_degenerate_tail(
            response.content, context=call_context, stage="page_content_extract"
        )
        return content

"""Vision analysis service for LLMProcessor.

This module provides the VisionAnalyzer service class for image analysis.
LLMProcessor composes it (lazy ``vision`` property) and exposes thin
delegates for the public methods.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from markitai.constants import (
    DEFAULT_MAX_IMAGES_PER_BATCH,
)
from markitai.llm.degeneration import truncate_degenerate_tail
from markitai.llm.engine import (
    LLMCall,
    extract_cached_tokens,
    find_budget_exceeded_error,
    find_non_retryable_provider_error,
    run_structured_ladder,
)
from markitai.llm.models import context_display_name, get_response_cost
from markitai.llm.structured import router_structured_ladder
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
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from markitai.config import LLMConfig
    from markitai.llm.engine import LLMEngine
    from markitai.prompts import PromptManager
    from markitai.types import LLMUsageByModel, ModelUsageStats


# In-code prompt fragments that are part of the effective image-analysis
# prompt. They live here as constants so the same text feeds both the
# messages and the cache-key digest and cannot drift apart.
DOCUMENT_CONTEXT_PREFIX = "\n\nDocument context: "
BATCH_HEADER_TEMPLATE = "Analyze the following {count} images in order."
BATCH_LANGUAGE_HINT_TEMPLATE = (
    "\n\nFallback language for images without visible text: {language}."
)
BATCH_FOOTER = (
    "\n\nReturn a JSON object with an 'images' array containing results "
    "for each image in order."
)
IMAGE_LABEL_TEMPLATE = "\n__MARKITAI_IMG_LABEL_{index}__"
LANGUAGE_RETRY_INSTRUCTION_TEMPLATE = (
    "\n\nCRITICAL: This image appears to contain no readable text. "
    "Return the caption and description in {language}. "
    "Do not answer in another language."
)
LANGUAGE_REWRITE_SYSTEM_TEMPLATE = (
    "Rewrite the following {field_name} into {language}."
    " Preserve the original meaning."
    "{preserve_structure}"
    " Return only the rewritten text."
)
LANGUAGE_REWRITE_PRESERVE_STRUCTURE = (
    " Preserve markdown formatting, headings, and lists."
)
LANGUAGE_REWRITE_USER_TEMPLATE = "Original {field_name}:\n{content}{document_context}"

# Prompt templates + fragments behind every entry of the "image_analysis"
# cache category. analyze_image() and analyze_images_batch() share the cache,
# and analyze_image() can fall back to the caption/description two-call path,
# so one digest covers the whole surface: any of these changing invalidates
# every image-analysis entry.
VISION_PROMPT_NAMES = (
    "image_analysis_system",
    "image_analysis_user",
    "image_caption_system",
    "image_caption_user",
    "image_description_system",
    "image_description_user",
)
VISION_PROMPT_FRAGMENTS = (
    DOCUMENT_CONTEXT_PREFIX,
    BATCH_HEADER_TEMPLATE,
    BATCH_LANGUAGE_HINT_TEMPLATE,
    BATCH_FOOTER,
    IMAGE_LABEL_TEMPLATE,
    LANGUAGE_RETRY_INSTRUCTION_TEMPLATE,
    LANGUAGE_REWRITE_SYSTEM_TEMPLATE,
    LANGUAGE_REWRITE_PRESERVE_STRUCTURE,
    LANGUAGE_REWRITE_USER_TEMPLATE,
)


def _document_context_suffix(document_context: str) -> str:
    """Render the document-context tail appended to image user prompts."""
    if not document_context:
        return ""
    return f"{DOCUMENT_CONTEXT_PREFIX}{document_context}"


def _vision_cache_content_key(
    image_fingerprint: str, document_context: str = ""
) -> str:
    """Build a cache content key for vision analysis.

    Incorporates document_context hash when present so that the same
    image analyzed in different documents produces separate cache entries.

    Prompt versioning lives in the cache *key* (``image_analysis@<digest>``),
    not here — this key only identifies the inputs.

    Args:
        image_fingerprint: SHA-256 hex digest of the image data.
        document_context: Optional document context text.

    Returns:
        Content key string for use with PersistentCache.
    """
    if not document_context:
        return image_fingerprint
    ctx_hash = hashlib.sha256(document_context.encode()).hexdigest()[:16]
    return f"{image_fingerprint}|ctx:{ctx_hash}"


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


def _reraise_if_fatal(exc: BaseException) -> None:
    """Re-raise the errors a fallback strategy must never absorb."""
    fatal = find_non_retryable_provider_error(exc)
    if fatal is not None:
        raise fatal
    budget = find_budget_exceeded_error(exc)
    if budget is not None:
        raise budget


def _merge_llm_usage(
    base: LLMUsageByModel | dict[str, Any] | None,
    extra: LLMUsageByModel | dict[str, Any] | None,
) -> LLMUsageByModel:
    """Merge llm_usage dicts without importing workflow helpers."""
    merged: dict[str, Any] = copy.deepcopy(dict(base)) if base else {}
    if not extra:
        return merged

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
        merged[model]["cached_input_tokens"] = merged[model].get(
            "cached_input_tokens", 0
        ) + usage.get("cached_input_tokens", 0)
        merged[model]["cost_usd"] = merged[model].get("cost_usd", 0.0) + usage.get(
            "cost_usd", 0.0
        )

    return merged


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


@dataclass
class ImagePlan:
    """One image's analysis request plus what it takes to finish it.

    The live path builds a plan, calls, and finalizes in one breath. The
    offline path collects ``messages`` into a Batch API job and applies
    ``finalize_image_plan`` when the answer comes back.

    ``answer`` short-circuits both: an unsupported format or a cache hit is
    already the final result, and no request should be sent for it.
    """

    image_path: Path
    context: str
    document_context: str
    language: str
    cache_key: str
    cache_content_key: str
    messages: list[dict[str, Any]]
    answer: ImageAnalysis | None = None


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
    - ``vision_cache_model_scope``: persistent-cache ``model`` scope for
      image analysis results (the processor's vision model-pool
      fingerprint, see ``model_list_fingerprint``)
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
        vision_cache_model_scope: str,
        get_vision_router: Callable[[], Any],
        get_cached_image: Callable[[Path], tuple[bytes, str]],
        get_next_call_index: Callable[[str], int],
    ) -> None:
        self._engine = engine
        self._prompt_manager = prompt_manager
        self._config = config
        self._vision_cache_model_scope = vision_cache_model_scope
        self._get_vision_router = get_vision_router
        self._get_cached_image = get_cached_image
        self._get_next_call_index = get_next_call_index

    def _image_analysis_cache_key(self) -> str:
        """Cache category for image analysis, scoped by the prompt text.

        Single-image and batch analysis deliberately share one key so a
        batch miss can still be served by an entry ``analyze_image`` wrote
        (and vice versa).
        """
        digest = self._prompt_manager.template_digest(
            *VISION_PROMPT_NAMES, extra=VISION_PROMPT_FRAGMENTS
        )
        return f"image_analysis@{digest}"

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
            router=router,
        )

    def prepare_image_plan(
        self,
        image_path: Path,
        context: str = "",
        document_context: str = "",
    ) -> ImagePlan:
        """Everything decided before the model is asked about one image.

        Deterministic in (image bytes, document_context, config), so a batch
        collector can rebuild the identical plan when the answer comes back
        hours later. ``plan.answer`` is already filled for an image that
        needs no call at all — an unsupported format, or a cache hit.
        """
        if not is_llm_supported_image(image_path.suffix):
            logger.debug(
                f"[{image_path.name}] Skipping unsupported format: {image_path.suffix}"
            )
            return ImagePlan(
                image_path=image_path,
                context=context,
                document_context=document_context,
                language="",
                cache_key="",
                cache_content_key="",
                messages=[],
                answer=ImageAnalysis(
                    caption=image_path.stem,
                    description=(
                        f"Image format {image_path.suffix} not supported for analysis"
                    ),
                ),
            )

        # SHA256 of the base64 as the fingerprint: JPEG files share a header,
        # so a prefix would collide.
        _, base64_image = self._get_cached_image(image_path)
        cache_key = self._image_analysis_cache_key()
        cache_content_key = _vision_cache_content_key(
            hashlib.sha256(base64_image.encode()).hexdigest(), document_context
        )
        cached = self._engine.persistent_cache.get(
            cache_key,
            cache_content_key,
            context=context,
            model=self._vision_cache_model_scope,
        )
        if cached is not None:
            logger.debug(f"[{image_path.name}] Persistent cache hit for analyze_image")
            return ImagePlan(
                image_path=image_path,
                context=context,
                document_context=document_context,
                language="",
                cache_key=cache_key,
                cache_content_key=cache_content_key,
                messages=[],
                answer=ImageAnalysis(
                    caption=cached.get("caption", ""),
                    description=cached.get("description", ""),
                    extracted_text=cached.get("extracted_text"),
                ),
            )

        language = _detect_document_language(document_context)
        system_prompt = self._prompt_manager.get_prompt(
            "image_analysis_system",
            language=language,
        )
        user_prompt = self._prompt_manager.get_prompt(
            "image_analysis_user",
            document_context=_document_context_suffix(document_context),
            language=language,
        )
        mime_type = get_llm_effective_mime(image_path.suffix)
        return ImagePlan(
            image_path=image_path,
            context=context,
            document_context=document_context,
            language=language,
            cache_key=cache_key,
            cache_content_key=cache_content_key,
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
        )

    def finalize_image_plan(
        self, plan: ImagePlan, result: ImageAnalysis
    ) -> ImageAnalysis:
        """Guard the answer and persist it. Pure local work.

        Runs identically on the live path and on a batch result collected
        hours later. A degenerate answer is returned but never cached, so
        the next run asks again instead of serving the damage forever.
        """
        if not _guard_degenerate_extracted_text(result, plan.image_path.name):
            self._engine.persistent_cache.set(
                plan.cache_key,
                plan.cache_content_key,
                {
                    "caption": result.caption,
                    "description": result.description,
                    "extracted_text": result.extracted_text,
                },
                model=self._vision_cache_model_scope,
            )
        return result

    async def analyze_image(
        self,
        image_path: Path,
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """
        Analyze an image using vision model.

        Uses the structured ladder for the answer, with one fallback:
        1. Try the structured ladder (TOOLS -> JSON_SCHEMA -> MD_JSON)
        2. Fallback to the original two-call method

        Args:
            image_path: Path to the image file
            context: Context identifier for usage tracking (e.g., source filename)
            document_context: Short text snippet from the surrounding document,
                used as language hint for images without visible text.

        Returns:
            ImageAnalysis with caption and description
        """
        plan = self.prepare_image_plan(
            image_path, context=context, document_context=document_context
        )
        if plan.answer is not None:
            return plan.answer

        # "default" lets the router pick a vision-capable deployment: the
        # messages carry image content.
        result = await self._analyze_image_with_fallback(
            plan.messages,
            "default",
            image_path.name,
            context,
            document_context=document_context,
        )
        result = await self._retry_until_language_holds(plan, result)
        return self.finalize_image_plan(plan, result)

    async def _retry_until_language_holds(
        self, plan: ImagePlan, result: ImageAnalysis
    ) -> ImageAnalysis:
        """Two more rounds when the model answered in the wrong language.

        First re-ask multimodally with the constraint spelled out, then, if
        it still drifts, rewrite the text it produced. Both need to see an
        answer before deciding, which is why the offline path cannot run
        this half and calls it at collect time instead.
        """
        if not _should_retry_for_language(result, plan.language):
            return result

        logger.debug(
            f"[{plan.image_path.name}] Retrying image analysis with stronger "
            f"{plan.language} language constraint"
        )
        retry_messages = copy.deepcopy(plan.messages)
        retry_instruction = LANGUAGE_RETRY_INSTRUCTION_TEMPLATE.format(
            language=plan.language
        )
        retry_messages[0]["content"] += retry_instruction
        retry_messages[1]["content"][0]["text"] += retry_instruction
        result = await self._analyze_image_with_fallback(
            retry_messages,
            "default",
            plan.image_path.name,
            plan.context,
            document_context=plan.document_context,
        )

        if _should_retry_for_language(result, plan.language):
            logger.debug(
                f"[{plan.image_path.name}] Rewriting image analysis into "
                f"{plan.language} after multimodal retries"
            )
            result = await self._rewrite_analysis_language(
                result,
                language=plan.language,
                context=plan.context or plan.image_path.name,
                document_context=plan.document_context,
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
        cache_key = self._image_analysis_cache_key()
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
                cache_key,
                cache_content_key,
                context=context,
                model=self._vision_cache_model_scope,
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
        batch_header = BATCH_HEADER_TEMPLATE.format(count=len(uncached_paths))
        doc_ctx = _document_context_suffix(document_context)
        language_hint = BATCH_LANGUAGE_HINT_TEMPLATE.format(language=language)
        user_prompt = f"{batch_header}{doc_ctx}{language_hint}{BATCH_FOOTER}"

        # Build content parts with uncached images only
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        for i, image_path in enumerate(uncached_paths, 1):
            _, base64_image = self._get_cached_image(image_path)
            mime_type = get_llm_effective_mime(image_path.suffix)

            # Unique image label that won't conflict with document content
            content_parts.append(
                {"type": "text", "text": IMAGE_LABEL_TEMPLATE.format(index=i)}
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

                # Same capability staircase as every other structured call
                # (budget-guarded: instructor calls the router directly here,
                # so each attempt must still spend the document's request budget)
                response, raw_response = cast(
                    tuple[BatchImageAnalysisResult, Any],
                    await run_structured_ladder(
                        acompletion=self._engine.guard_acompletion(
                            vision_router.acompletion, context
                        ),
                        messages=messages,
                        response_model=BatchImageAnalysisResult,
                        ladder=router_structured_ladder(vision_router),
                        call_id=f"image_batch:{context}",
                        max_tokens=max_tokens,
                    ),
                )

                # Track usage first: a truncated batch was billed like any
                # other call, and truncation hits the largest (priciest)
                # batches, so raising before accounting hid real spend.
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
                        actual_model,
                        input_tokens,
                        output_tokens,
                        cost,
                        context,
                        extract_cached_tokens(raw_response),
                    )

                # Check for truncation (after accounting; the truncated
                # results are never cached because this aborts the batch)
                if hasattr(raw_response, "choices") and raw_response.choices:
                    finish_reason = getattr(
                        raw_response.choices[0], "finish_reason", None
                    )
                    if finish_reason == "length":
                        raise ValueError("Output truncated due to max_tokens limit")

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
                        cache_key,
                        content_key,
                        cache_value,
                        model=self._vision_cache_model_scope,
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
            # A blown request budget or a fatal provider error must not turn
            # into N more calls; the ladder raised them on purpose.
            _reraise_if_fatal(e)
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

        Strategy 1: the structured ladder (engine.complete_structured, which
            descends TOOLS -> JSON_SCHEMA -> MD_JSON on its own and repairs
            the JSON the model hand-writes on the bottom rung)
        Strategy 2: original two-call method (most compatible)

        Args:
            messages: LLM messages with image
            model: Model name to use
            image_name: Image filename for logging
            context: Context identifier for usage tracking
            document_context: Short text snippet for language hinting.
        """
        # Strategy 1: the structured ladder
        try:
            # Deep copy to prevent Instructor from modifying original messages
            result = await self._analyze_with_instructor(
                copy.deepcopy(messages), model, context
            )
            return result
        except Exception as e:
            _reraise_if_fatal(e)
            logger.debug(
                f"[{image_name}] Structured ladder failed: {e}, using two-call fallback"
            )

        # Strategy 2: Original two-call method
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
        and moves on to the two-call fallback.
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
        doc_ctx = _document_context_suffix(document_context)
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
        doc_ctx = _document_context_suffix(document_context)
        preserve_structure = (
            LANGUAGE_REWRITE_PRESERVE_STRUCTURE
            if field_name == "markdown description"
            else ""
        )
        system_prompt = LANGUAGE_REWRITE_SYSTEM_TEMPLATE.format(
            field_name=field_name,
            language=language,
            preserve_structure=preserve_structure,
        )
        user_prompt = LANGUAGE_REWRITE_USER_TEMPLATE.format(
            field_name=field_name, content=content, document_context=doc_ctx
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

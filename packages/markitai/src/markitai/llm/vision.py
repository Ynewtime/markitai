"""Vision analysis mixin for LLMProcessor.

This module provides vision-related methods for image analysis.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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
from markitai.llm.document import _try_repair_instructor_response
from markitai.llm.models import get_response_cost
from markitai.llm.types import (
    BatchImageAnalysisResult,
    ImageAnalysis,
    ImageAnalysisResult,
)
from markitai.utils.mime import get_mime_type, is_llm_supported_image
from markitai.utils.text import clean_control_characters, format_error_message

if TYPE_CHECKING:
    from asyncio import Semaphore

    from litellm.router import Router

    from markitai.llm.processor import HybridRouter, LocalProviderWrapper
    from markitai.prompts import PromptManager
    from markitai.types import LLMUsageByModel, ModelUsageStats


def _context_display_name(context: str) -> str:
    """Get display name for context (filename or default)."""
    return context.split("/")[-1] if context else "batch"


class VisionMixin:
    """Vision analysis methods for LLMProcessor.

    This mixin provides image analysis functionality including:
    - Single image analysis
    - Batch image analysis
    - Page content extraction

    Note: This mixin expects to be used with LLMProcessor which provides
    the attributes declared below.
    """

    # Declare expected attributes from LLMProcessor for type checking
    # Methods still use type: ignore[attr-defined] as they have complex signatures
    if TYPE_CHECKING:
        semaphore: Semaphore
        vision_router: Router | LocalProviderWrapper | HybridRouter
        _persistent_cache: Any  # PersistentCache from processor.py
        _prompt_manager: PromptManager

    async def analyze_image(
        self, image_path: Path, language: str = "en", context: str = ""
    ) -> ImageAnalysis:
        """
        Analyze an image using vision model.

        Uses Instructor for structured output with fallback mechanisms:
        1. Try Instructor with structured output
        2. Fallback to JSON mode + manual parsing
        3. Fallback to original two-call method

        Args:
            image_path: Path to the image file
            language: Language for output (e.g., "en", "zh")
            context: Context identifier for usage tracking (e.g., source filename)

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
        _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]

        # Check persistent cache using image hash + language as key
        # Use SHA256 hash of base64 as image fingerprint to avoid collisions
        # (JPEG files share the same header, so first N chars are identical)
        cache_key = f"image_analysis:{language}"
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
        cached = self._persistent_cache.get(  # type: ignore[attr-defined]
            cache_key, image_fingerprint, context=context
        )
        if cached is not None:
            logger.debug(f"[{image_path.name}] Persistent cache hit for analyze_image")
            # Reconstruct ImageAnalysis from cached dict
            return ImageAnalysis(
                caption=cached.get("caption", ""),
                description=cached.get("description", ""),
                extracted_text=cached.get("extracted_text"),
            )

        # Determine MIME type
        mime_type = get_mime_type(image_path.suffix)

        # Get language name for prompt
        lang_name = "English" if language == "en" else "中文"

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt(  # type: ignore[attr-defined]
            "image_analysis_system", language=lang_name
        )
        user_prompt = self._prompt_manager.get_prompt("image_analysis_user")  # type: ignore[attr-defined]

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
            messages, vision_model, image_path.name, context
        )

        # Store in persistent cache
        cache_value = {
            "caption": result.caption,
            "description": result.description,
            "extracted_text": result.extracted_text,
        }
        self._persistent_cache.set(  # type: ignore[attr-defined]
            cache_key, image_fingerprint, cache_value, model="vision"
        )

        return result

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        language: str = "en",
        max_images_per_batch: int = DEFAULT_MAX_IMAGES_PER_BATCH,
        context: str = "",
    ) -> list[ImageAnalysis]:
        """
        Analyze multiple images in batches with parallel execution.

        Batches are processed concurrently using asyncio.gather for better
        throughput. LLM concurrency is controlled by the shared semaphore.

        Args:
            image_paths: List of image paths to analyze
            language: Language for output ("en" or "zh")
            max_images_per_batch: Max images per LLM call (default 10)
            context: Context identifier for usage tracking (e.g., source filename)

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
        max_concurrent_batches = min(self.config.concurrency, num_batches)  # type: ignore[attr-defined]
        batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        display_name = _context_display_name(context)
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
                    results = await self.analyze_batch(batch_paths, language, context)
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
        language: str,
        context: str = "",
    ) -> list[ImageAnalysis]:
        """Batch image analysis using Instructor.

        Uses the same prompt template as single image analysis for consistency.
        Checks persistent cache first and only calls LLM for uncached images.

        Args:
            image_paths: List of image paths to analyze
            language: Language for output ("en" or "zh")
            context: Context identifier for usage tracking

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
        cache_key = f"image_analysis:{language}"
        cached_results: dict[int, ImageAnalysis] = {}
        uncached_indices: list[int] = []
        image_fingerprints: dict[int, str] = {}

        for orig_idx, image_path in supported_paths:
            _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]
            # Use SHA256 hash to avoid collisions (JPEG files share same header)
            fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
            image_fingerprints[orig_idx] = fingerprint

            cached = self._persistent_cache.get(cache_key, fingerprint, context=context)  # type: ignore[attr-defined]
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
        display_name = _context_display_name(context)
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

        # Get language name for prompt
        lang_name = "English" if language == "en" else "中文"

        # Use separated system/user prompts to improve instruction following
        system_prompt = self._prompt_manager.get_prompt(  # type: ignore[attr-defined]
            "image_analysis_system", language=lang_name
        )

        # Build batch user prompt
        batch_header = (
            f"请依次分析以下 {len(uncached_paths)} 张图片。"
            if language == "zh"
            else f"Analyze the following {len(uncached_paths)} images in order."
        )
        batch_footer = "\n\nReturn a JSON object with an 'images' array containing results for each image in order."
        user_prompt = f"{batch_header}{batch_footer}"

        # Build content parts with uncached images only
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        for i, image_path in enumerate(uncached_paths, 1):
            _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]
            mime_type = get_mime_type(image_path.suffix)

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
            async with self.semaphore:  # type: ignore[attr-defined]
                # Calculate dynamic max_tokens using minimum across all vision router models
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_parts},
                ]
                max_tokens = self._calculate_dynamic_max_tokens(  # type: ignore[attr-defined]
                    messages,
                    router=self.vision_router,  # type: ignore[attr-defined]
                )

                # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
                client = instructor.from_litellm(
                    self.vision_router.acompletion,
                    mode=instructor.Mode.MD_JSON,  # type: ignore[attr-defined]
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
                    self._track_usage(  # type: ignore[attr-defined]
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

                # Convert to ImageAnalysis list and store in cache
                new_results: list[ImageAnalysis] = []
                for idx, img_result in enumerate(response.images):
                    analysis = ImageAnalysis(
                        caption=img_result.caption,
                        description=img_result.description,
                        extracted_text=img_result.extracted_text,
                        llm_usage=per_image_llm_usage,
                    )
                    new_results.append(analysis)

                    # Store in persistent cache using original index
                    if idx < len(uncached_indices):
                        original_idx = uncached_indices[idx]
                        fingerprint = image_fingerprints[original_idx]
                        cache_value = {
                            "caption": analysis.caption,
                            "description": analysis.description,
                            "extracted_text": analysis.extracted_text,
                        }
                        self._persistent_cache.set(  # type: ignore[attr-defined]
                            cache_key, fingerprint, cache_value, model="vision"
                        )

                # Ensure we have results for all uncached images
                while len(new_results) < len(uncached_paths):
                    new_results.append(
                        ImageAnalysis(
                            caption="Image",
                            description="Image analysis failed",
                            extracted_text=None,
                            llm_usage=per_image_llm_usage,
                        )
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
            # Fallback: analyze each image individually (uses persistent cache)
            # Pass context to maintain accurate per-file usage tracking
            # Note: cached_results may already have some hits from the initial check
            fallback_results: list[ImageAnalysis] = []
            for i, image_path in enumerate(image_paths):
                if i in unsupported_results:
                    # Use unsupported placeholder result
                    fallback_results.append(unsupported_results[i])
                elif i in cached_results:
                    # Use already-cached result
                    fallback_results.append(cached_results[i])
                else:
                    try:
                        # analyze_image will also check/populate cache
                        result = await self.analyze_image(image_path, language, context)
                        fallback_results.append(result)
                    except Exception:
                        fallback_results.append(
                            ImageAnalysis(
                                caption="Image",
                                description="Image analysis failed",
                                extracted_text=None,
                            )
                        )
            return fallback_results

    async def _analyze_image_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        image_name: str,
        context: str = "",
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
            copy.deepcopy(messages), model, context=context or image_name
        )

    async def _analyze_with_instructor(
        self,
        messages: list[dict[str, Any]],
        model: str,
        context: str = "",
    ) -> ImageAnalysis:
        """Analyze using Instructor for structured output."""
        async with self.semaphore:  # type: ignore[attr-defined]
            # Calculate dynamic max_tokens using minimum across all vision router models
            max_tokens = self._calculate_dynamic_max_tokens(  # type: ignore[attr-defined]
                messages,
                router=self.vision_router,  # type: ignore[attr-defined]
            )

            # Create instructor client from vision router for load balancing
            # Use MD_JSON mode to handle LLMs that wrap JSON in ```json code blocks
            client = instructor.from_litellm(
                self.vision_router.acompletion,
                mode=instructor.Mode.MD_JSON,  # type: ignore[attr-defined]
            )

            # Use create_with_completion to get both the model and the raw response
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            try:
                response, raw_response = await cast(
                    Awaitable[tuple[ImageAnalysisResult, Any]],
                    client.chat.completions.create_with_completion(
                        model=model,
                        messages=cast(list[ChatCompletionMessageParam], messages),
                        response_model=ImageAnalysisResult,
                        max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                        max_tokens=max_tokens,
                    ),
                )
            except Exception as e:
                repaired = _try_repair_instructor_response(e, ImageAnalysisResult)
                if repaired is None:
                    raise
                response, raw_response = repaired

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or model
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                cost = get_response_cost(raw_response)
                self._track_usage(  # type: ignore[attr-defined]
                    actual_model, input_tokens, output_tokens, cost, context
                )

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
        json_messages[0] = {
            **messages[0],
            "content": [
                {
                    "type": "text",
                    "text": messages[0]["content"][0]["text"]
                    + "\n\nReturn a JSON object with 'caption' and 'description' fields.",
                },
                messages[0]["content"][1],  # image
            ],
        }

        async with self.semaphore:  # type: ignore[attr-defined]
            # Calculate dynamic max_tokens using minimum across all vision router models
            max_tokens = self._calculate_dynamic_max_tokens(  # type: ignore[attr-defined]
                json_messages,
                router=self.vision_router,  # type: ignore[attr-defined]
            )

            # Use vision_router for image analysis (not main router)
            response = await self.vision_router.acompletion(  # type: ignore[attr-defined]
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
            self._track_usage(actual_model, input_tokens, output_tokens, cost, context)  # type: ignore[attr-defined]

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
        model: str,  # noqa: ARG002
        context: str = "",
    ) -> ImageAnalysis:
        """Original two-call method as final fallback."""
        # Extract image from messages (handle both old and new format)
        # New format: [system_msg, user_msg_with_image]
        # Old format: [user_msg_with_image]
        if messages[0].get("role") == "system":
            user_content = messages[1]["content"]
            system_content = messages[0]["content"]
        else:
            user_content = messages[0]["content"]
            system_content = ""

        image_content = user_content[1]  # The image part

        # Detect language from system prompt
        lang_name = "English"
        if isinstance(system_content, str) and "中文" in system_content:
            lang_name = "中文"

        # Generate caption using system/user prompts
        caption_system = self._prompt_manager.get_prompt(  # type: ignore[attr-defined]
            "image_caption_system", language=lang_name
        )
        caption_user = self._prompt_manager.get_prompt("image_caption_user")  # type: ignore[attr-defined]
        caption_response = await self._call_llm(  # type: ignore[attr-defined]
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
        desc_system = self._prompt_manager.get_prompt(  # type: ignore[attr-defined]
            "image_description_system", language=lang_name
        )
        desc_user = self._prompt_manager.get_prompt("image_description_user")  # type: ignore[attr-defined]
        desc_response = await self._call_llm(  # type: ignore[attr-defined]
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
        _, base64_image = self._get_cached_image(image_path)  # type: ignore[attr-defined]

        # Determine MIME type
        mime_type = get_mime_type(image_path.suffix)

        # Use separated system/user prompts to improve instruction following
        # Language is set to "与源文档一致" (match source document)
        system_prompt = self._prompt_manager.get_prompt(  # type: ignore[attr-defined]
            "page_content_system", language="与源文档一致"
        )
        user_prompt = self._prompt_manager.get_prompt("page_content_user")  # type: ignore[attr-defined]

        # Use image_path.name as context if not provided
        call_context = context or image_path.name

        response = await self._call_llm(  # type: ignore[attr-defined]
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

        return response.content

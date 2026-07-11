"""Unified structured LLM call engine.

This module is the single pipeline for structured (instructor-based) LLM
calls: two-layer cache lookup, prompt-agnostic transport retries, instructor
validation retries, JSON repair, length checking, usage accounting, and
cache write-back.

Phase 2.1 of the LLM refactor: prior to this module, each structured call
site (document/vision mixins) wired ``instructor.from_litellm`` directly to
``router.acompletion``, bypassing the transport retry loop in
``LLMProcessor._call_llm_with_retry`` (backoff, quota short-circuit, empty
response retry). ``LLMEngine`` closes that gap by handing instructor a
retrying acompletion adapter, so every instructor attempt goes through the
full transport retry loop.

This module must not import ``markitai.llm.processor`` (circular import:
processor -> document -> engine).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast, get_origin

import instructor
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from loguru import logger
from pydantic import BaseModel

from markitai.constants import (
    DEFAULT_INSTRUCTOR_MAX_RETRIES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
)
from markitai.llm.models import get_response_cost
from markitai.providers.errors import ProviderError
from markitai.utils.text import format_error_message, repair_json_string

# Retryable transport exceptions. Rebuilt here from litellm types because the
# canonical tuple lives in markitai.llm.processor, which this module cannot
# import (circular). Single source will converge in Phase 2.3.
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)

# Model-level error patterns that indicate the model itself is unavailable
# (not a content/request issue). Copied verbatim from
# HybridRouter.MODEL_LEVEL_ERROR_PATTERNS in markitai.llm.processor, which
# this module cannot import (circular). Single source will converge in
# Phase 2.3.
MODEL_LEVEL_ERROR_PATTERNS = (
    "user location is not supported",
    "failed_precondition",
    "model is not available",
    "model not found",
    "model_not_available",
    "region is not supported",
    "not available in your region",
)


def try_repair_instructor_response(
    exc: Exception,
    response_model: type,
) -> tuple[Any, Any] | None:
    """Try to repair JSON from a failed instructor response.

    When instructor's retry mechanism fails (all retries exhausted), the
    last LLM completion is still available. This function extracts the raw
    text, attempts JSON repair, and constructs the Pydantic model manually.

    Args:
        exc: The InstructorRetryException (or compatible exception)
        response_model: Pydantic model class to validate against

    Returns:
        Tuple of (parsed_model, raw_response) or None if repair failed
    """
    last = getattr(exc, "last_completion", None)
    if last is None:
        return None

    # Extract text content from the completion
    try:
        content = last.choices[0].message.content
        if not content:
            return None
    except (AttributeError, IndexError):
        return None

    # Attempt JSON repair
    repaired = repair_json_string(content)
    if repaired is None:
        return None

    import json

    try:
        data = json.loads(repaired)
    except Exception:
        logger.debug(
            f"[JSON repair] Repair attempt failed for {response_model.__name__}"
        )
        return None

    # Valid JSON may still have the wrong shape: small models return the
    # bare payload without the wrapper object (e.g. one image result
    # instead of {"images": [...]}). Try the wrapped form as well.
    candidates: list[Any] = [data]
    wrapped = _wrap_bare_payload_for_model(data, response_model)
    if wrapped is not None:
        candidates.append(wrapped)

    for candidate in candidates:
        try:
            result = response_model.model_validate(candidate)
        except Exception:
            continue
        logger.info(
            f"[JSON repair] Successfully repaired malformed JSON "
            f"for {response_model.__name__}"
        )
        return result, last

    logger.debug(f"[JSON repair] Repair attempt failed for {response_model.__name__}")
    return None


def _wrap_bare_payload_for_model(
    data: Any, response_model: type
) -> dict[str, Any] | None:
    """Wrap a bare item/list into the model's single required list field.

    Only applies to wrapper models like BatchImageAnalysisResult, whose sole
    required field is a list; returns None for anything else.
    """
    fields = getattr(response_model, "model_fields", None)
    if not fields:
        return None
    required = [(name, f) for name, f in fields.items() if f.is_required()]
    if len(required) != 1:
        return None
    field_name, field = required[0]
    if get_origin(field.annotation) is not list:
        return None
    if isinstance(data, list):
        return {field_name: data}
    if isinstance(data, dict) and field_name not in data:
        return {field_name: [data]}
    return None


def find_non_retryable_provider_error(
    exc: BaseException,
    seen: set[int] | None = None,
) -> ProviderError | None:
    """Find a wrapped non-retryable ProviderError inside nested exceptions."""
    if seen is None:
        seen = set()

    exc_id = id(exc)
    if exc_id in seen:
        return None
    seen.add(exc_id)

    if isinstance(exc, ProviderError) and not exc.retryable:
        return exc

    for attr_name in ("__cause__", "__context__"):
        nested = getattr(exc, attr_name, None)
        if isinstance(nested, BaseException):
            found = find_non_retryable_provider_error(nested, seen)
            if found is not None:
                return found

    failed_attempts = getattr(exc, "failed_attempts", None)
    if failed_attempts:
        for attempt in failed_attempts:
            attempt_exc = getattr(attempt, "exception", None)
            if isinstance(attempt_exc, BaseException):
                found = find_non_retryable_provider_error(attempt_exc, seen)
                if found is not None:
                    return found

    return None


@dataclass(frozen=True)
class LLMCall:
    """One structured LLM call description.

    Attributes:
        purpose: Log tag (e.g. "document_process")
        messages: Fully assembled chat messages (system/user/vision)
        response_model: Pydantic model for instructor structured output
        context: Usage-tracking context (file name / URL)
        cache_key: Cache key; None disables both cache read and write
        cache_content: Content fingerprint parameter for the caches
        cache_model: ``model`` parameter passed to the persistent cache
        validate: Optional hook returning the (possibly corrected) result;
            if it raises, nothing is cached and the error propagates
        cache_if: Optional hook called after ``validate`` and before the
            cache write; returning False skips both cache layers' writes
            but the result is still returned normally (e.g. degenerate
            output that must not poison a clean retry)
        serialize: Result -> cacheable dict (None -> ``result.model_dump()``)
        deserialize: Cached dict -> result
            (None -> ``response_model.model_construct(**cached)``)
        router: Per-call router override (None -> engine default router)
        max_tokens: Explicit max_tokens (None -> dynamic calculation callback)
    """

    purpose: str
    messages: list[dict[str, Any]]
    response_model: type[BaseModel]
    context: str
    cache_key: str | None = None
    cache_content: str = ""
    cache_model: str = "default"
    validate: Callable[[Any], Any] | None = None
    cache_if: Callable[[Any], bool] | None = None
    serialize: Callable[[Any], dict[str, Any]] | None = None
    deserialize: Callable[[dict[str, Any]], Any] | None = None
    router: Any | None = None
    max_tokens: int | None = None


class LLMEngine:
    """Unified pipeline for structured LLM calls.

    Collaborators are injected so the engine stays independent of
    ``LLMProcessor`` (which cannot be imported here):

    - ``memory_cache``: ContentCache interface
      (``get(key, content)`` / ``set(key, content, value)``)
    - ``persistent_cache``: PersistentCache interface
      (``get(key, content, context=...)`` / ``set(key, content, value, model=...)``)
    - ``track_usage``: ``(model, input_tokens, output_tokens, cost, context)``
    - ``calculate_max_tokens``: ``(messages, model_id, router=...) -> int | None``
      (bound from ``LLMProcessor._calculate_dynamic_max_tokens``)
    - ``get_primary_model``: ``(router) -> model_id | None``
      (bound from ``LLMProcessor._get_router_primary_model``)
    """

    def __init__(
        self,
        *,
        router: Any,
        semaphore: asyncio.Semaphore,
        memory_cache: Any,
        persistent_cache: Any,
        track_usage: Callable[[str, int, int, float, str], None],
        calculate_max_tokens: Callable[..., int | None],
        get_primary_model: Callable[[Any], str | None],
    ) -> None:
        self.router = router
        self.semaphore = semaphore
        self.memory_cache = memory_cache
        self.persistent_cache = persistent_cache
        self._track_usage = track_usage
        self._calculate_max_tokens = calculate_max_tokens
        self._get_primary_model = get_primary_model

    async def complete_structured(self, call: LLMCall) -> tuple[Any, Any]:
        """Run one structured LLM call through the full pipeline.

        Returns:
            Tuple of (parsed_result, raw_response). On cache hit,
            raw_response is None.
        """

        def deserialize(cached: dict[str, Any]) -> Any:
            if call.deserialize is not None:
                return call.deserialize(cached)
            # model_construct() bypasses validation for cached data
            return call.response_model.model_construct(**cached)

        # 1. Cache lookup: in-memory first (fastest), then persistent
        if call.cache_key is not None:
            cached = self.memory_cache.get(call.cache_key, call.cache_content)
            if cached is not None:
                return deserialize(cached), None

            cached = self.persistent_cache.get(
                call.cache_key, call.cache_content, context=call.context
            )
            if cached is not None:
                # Also populate in-memory cache for faster subsequent access
                self.memory_cache.set(call.cache_key, call.cache_content, cached)
                return deserialize(cached), None

        # 2. Cache miss: one structured call occupies one concurrency slot
        # for its whole duration (transport retries included), matching the
        # existing direct-instructor call sites.
        active_router = call.router if call.router is not None else self.router
        call_id = f"{call.purpose}:{call.context}"

        async with self.semaphore:
            start_time = time.perf_counter()

            if call.max_tokens is not None:
                max_tokens = call.max_tokens
            else:
                max_tokens = self._calculate_max_tokens(
                    call.messages,
                    self._get_primary_model(active_router),
                    router=active_router,
                )

            # Instructor gets the retrying adapter instead of the router's
            # bare acompletion, so its validation retries and the transport
            # retries compose. MD_JSON mode handles LLMs that wrap JSON in
            # ```json code blocks.
            client = instructor.from_litellm(
                self._make_retrying_acompletion(active_router, call_id),
                mode=instructor.Mode.MD_JSON,
            )

            try:
                result, raw_response = await cast(
                    Awaitable[tuple[Any, Any]],
                    client.chat.completions.create_with_completion(
                        model="default",
                        # Copy: instructor mutates the messages list in place
                        # (call sites previously built a fresh list per call)
                        messages=cast(list[Any], list(call.messages)),
                        response_model=call.response_model,
                        max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                        max_tokens=max_tokens,
                    ),
                )
            except Exception as e:
                fatal_provider_error = find_non_retryable_provider_error(e)
                if fatal_provider_error is not None:
                    raise fatal_provider_error
                repaired = try_repair_instructor_response(e, call.response_model)
                if repaired is None:
                    raise
                result, raw_response = repaired

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage from the raw API response (once per structured
            # call; instructor accumulates usage across its own retries)
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                cost = get_response_cost(raw_response)
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, call.context
                )

            logger.info(
                f"[LLM:{call_id}] {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
            )

            # Validation hook: may correct the result; if it raises, nothing
            # is cached and the error propagates to the caller
            if call.validate is not None:
                result = call.validate(result)

            # Store in both cache layers (unless the cache_if hook vetoes
            # the write, e.g. for degenerate output)
            if call.cache_key is not None and (
                call.cache_if is None or call.cache_if(result)
            ):
                if call.serialize is not None:
                    cache_value = call.serialize(result)
                else:
                    cache_value = result.model_dump()
                self.memory_cache.set(call.cache_key, call.cache_content, cache_value)
                self.persistent_cache.set(
                    call.cache_key,
                    call.cache_content,
                    cache_value,
                    model=call.cache_model,
                )

        return result, raw_response

    def _make_retrying_acompletion(
        self,
        active_router: Any,
        call_id: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Callable[..., Awaitable[Any]]:
        """Build an acompletion adapter with the full transport retry loop.

        The adapter is signature-compatible with ``litellm.acompletion``
        (``model=..., messages=..., **kwargs``) and returns the raw
        ModelResponse, so it can be handed to ``instructor.from_litellm``.

        The loop is adapted from ``LLMProcessor._call_llm_with_retry``
        (markitai.llm.processor), with two deliberate differences:

        - It does NOT acquire the engine semaphore: instructor invokes the
          adapter while ``complete_structured`` already holds a slot
          ("one structured call = one concurrency slot").
        - It does NOT call ``track_usage``: usage is tracked once per
          structured call from the final raw response, which instructor
          accumulates across its own retries.
        """
        get_primary_model = self._get_primary_model

        async def retrying_acompletion(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "default")
            last_exception: Exception | None = None

            # Backoff sleeps happen at the top of the next iteration; the
            # semaphore is held by the caller for the whole structured call
            retry_delay = 0.0

            for attempt in range(max_retries + 1):
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)
                    retry_delay = 0.0
                start_time = time.perf_counter()

                try:
                    # Log request start
                    if attempt == 0:
                        logger.debug(f"[LLM:{call_id}] Request to {model}")
                    else:
                        # Log retry attempt
                        error_type = (
                            type(last_exception).__name__
                            if last_exception
                            else "Unknown"
                        )
                        status_code = getattr(last_exception, "status_code", "N/A")
                        logger.warning(
                            f"[LLM:{call_id}] Retry #{attempt}: {error_type} "
                            f"status={status_code}"
                        )

                    call_kwargs = dict(kwargs)
                    call_kwargs["metadata"] = {"call_id": call_id, "attempt": attempt}
                    response = await active_router.acompletion(*args, **call_kwargs)

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    actual_model = getattr(response, "model", None) or model

                    # Calculate cost (uses _hidden_params for local providers)
                    cost = get_response_cost(response)

                    usage = getattr(response, "usage", None)
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                    # Log result
                    logger.info(
                        f"[LLM:{call_id}] {actual_model} "
                        f"tokens={input_tokens}+{output_tokens} "
                        f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
                    )

                    # Detect empty response (0 output tokens with substantial
                    # input): usually a model failure that should be retried
                    if output_tokens == 0 and input_tokens > 100:
                        if attempt < max_retries:
                            logger.warning(
                                f"[LLM:{call_id}] Empty response (0 output tokens), "
                                f"retrying with different model..."
                            )
                            # Treat as retryable error
                            retry_delay = min(
                                DEFAULT_RETRY_BASE_DELAY * (2**attempt),
                                DEFAULT_RETRY_MAX_DELAY,
                            )
                            continue
                        else:
                            logger.error(
                                f"[LLM:{call_id}] Empty response after "
                                f"{max_retries + 1} attempts, returning as-is"
                            )

                    return response

                except RETRYABLE_ERRORS as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    last_exception = e

                    # Check for quota/billing errors that should NOT be retried
                    # These errors are wrapped by LiteLLM as APIConnectionError but
                    # are actually non-recoverable without user action.
                    # Genuine rate limits (RateLimitError) often mention "quota"
                    # in provider text (e.g. "quota will reset after 30s") but ARE
                    # retryable: the router cooldown recorded for the failing
                    # model routes the retry to another model.
                    error_msg_lower = str(e).lower()
                    if isinstance(e, RateLimitError):
                        non_retryable_patterns = (
                            "billing",
                            "payment",
                            "402",
                            "insufficient_quota",
                            "exceeded your current quota",
                        )
                    else:
                        non_retryable_patterns = (
                            "quota",
                            "billing",
                            "payment",
                            "subscription",
                            "402",
                            "insufficient_quota",
                            "exceeded your current quota",
                        )
                    if any(
                        pattern in error_msg_lower for pattern in non_retryable_patterns
                    ):
                        status_code = getattr(e, "status_code", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Quota/billing error (not retrying): "
                            f"status={status_code} {format_error_message(e)} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    if attempt == max_retries:
                        # Final failure after all retries
                        error_type = type(e).__name__
                        status_code = getattr(e, "status_code", "N/A")
                        provider = getattr(e, "llm_provider", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Failed after {max_retries + 1} attempts: "
                            f"{error_type} status={status_code} provider={provider} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    # Calculate exponential backoff delay
                    retry_delay = min(
                        DEFAULT_RETRY_BASE_DELAY * (2**attempt), DEFAULT_RETRY_MAX_DELAY
                    )

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    error_msg_lower = str(e).lower()

                    # Model-level errors are retryable (HybridRouter cooldown
                    # ensures the next attempt picks a different model)
                    if any(p in error_msg_lower for p in MODEL_LEVEL_ERROR_PATTERNS):
                        last_exception = e
                        status_code = getattr(e, "status_code", "N/A")
                        if attempt < max_retries:
                            logger.warning(
                                f"[LLM:{call_id}] Model-level error "
                                f"(status={status_code}), retrying: "
                                f"{format_error_message(e)} "
                                f"time={elapsed_ms:.0f}ms"
                            )
                            retry_delay = min(
                                DEFAULT_RETRY_BASE_DELAY * (2**attempt),
                                DEFAULT_RETRY_MAX_DELAY,
                            )
                            continue
                        else:
                            logger.error(
                                f"[LLM:{call_id}] Model-level error after "
                                f"{max_retries + 1} attempts: "
                                f"{format_error_message(e)} "
                                f"time={elapsed_ms:.0f}ms"
                            )
                            raise

                    # Check for authentication errors and provide friendly hints
                    status_code = getattr(e, "status_code", "N/A")
                    auth_patterns = (
                        "authentication",
                        "api_key",
                        "api key",
                        "unauthorized",
                        "401",
                        "403",
                        "invalid x-api-key",
                        "incorrect api key",
                    )
                    if any(p in error_msg_lower for p in auth_patterns):
                        target = get_primary_model(active_router)
                        logger.error(
                            f"[LLM:{call_id}] Authentication failed for model "
                            f"'{target}':\n"
                            f"  {format_error_message(e)}\n\n"
                            f"  Hint: Use MODEL=<provider/model> with the "
                            f"corresponding API key env var,\n"
                            f"  or run 'markitai init' to configure interactively."
                        )
                    else:
                        logger.error(
                            f"[LLM:{call_id}] Failed: status={status_code} "
                            f"{format_error_message(e)} time={elapsed_ms:.0f}ms"
                        )
                    raise

            # Should not reach here, but just in case
            raise RuntimeError(f"[LLM:{call_id}] Unexpected state in retry loop")

        return retrying_acompletion

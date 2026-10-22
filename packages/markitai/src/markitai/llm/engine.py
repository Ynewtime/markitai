"""Unified LLM call engine.

This module is the single transport layer for LLM calls. It provides two
entry points:

- ``complete_text``: plain text completions with the full transport retry
  loop (backoff, quota short-circuit, empty response retry) and per-attempt
  usage accounting.
- ``complete_structured``: structured (instructor-based) calls with
  two-layer cache lookup, transport retries, instructor validation retries,
  the structured-mode staircase, length checking, usage accounting, and
  cache write-back.

Both entry points share one retry loop (``_acompletion_with_retries``);
``LLMProcessor._call_llm_with_retry`` delegates here since Phase 2.3.

Structured calls run a capability-tiered staircase (``run_structured_ladder``)
whose rungs come from ``markitai.llm.structured``: the most native mode the
model pool supports first, one rung down per failure, ending at ``MD_JSON``.
JSON repair exists only on that last rung — every rung above it has the
provider, not the model, producing the JSON.

This module must not import ``markitai.llm.processor`` (circular import:
processor -> document -> engine).
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

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
from markitai.llm.error_redaction import redact_provider_exception
from markitai.llm.models import format_cost, get_response_cost, stamp_sent_at
from markitai.llm.router import MODEL_LEVEL_ERROR_PATTERNS, POOL_EXHAUSTED_PATTERN
from markitai.llm.structured import router_structured_ladder
from markitai.llm.types import LLMResponse
from markitai.providers.errors import ProviderError
from markitai.utils.errors import SelfExplanatoryError

# Spend tracking lives outside this package, which loads the LLM stack (the
# CLI wraps every URL in it); re-exported here for the LLM layer.
from markitai.utils.llm_spend import (
    SpendTally as SpendTally,
)
from markitai.utils.llm_spend import (
    record_llm_spend as record_llm_spend,
)
from markitai.utils.llm_spend import (
    record_llm_spend_marker as record_llm_spend_marker,
)
from markitai.utils.llm_spend import (
    track_llm_spend as track_llm_spend,
)
from markitai.utils.text import format_error_message, repair_json_string


class EmptyLLMResponseError(RuntimeError):
    """The model returned no usable content after every retry.

    Raised by ``complete_text(..., require_content=True)`` so call sites can
    tell "the model gave up" apart from a legitimate result. Callers must
    treat it as a failed call: never cache the outcome, and fall back to the
    unprocessed input instead of persisting an empty document.
    """


class LLMEnhancementDegradedError(SelfExplanatoryError, RuntimeError):
    """Enhancement fell back to unenhanced or partially enhanced output.

    Raised by the document service instead of returning a fallback result
    (the one whose frontmatter carries ``llm_enhanced: false``), so every
    surface fails the item the same way it does for any other LLM error:
    the base ``.md`` is written as the fallback and the item is reported
    failed. The best-effort result rides along for callers that still want
    to show it.

    The message is the formatted cause (``AuthenticationError: ...``), so
    this class opts out of the class-name prefix: the cause's type is the
    useful one.
    """

    def __init__(
        self,
        message: str,
        *,
        cleaned_markdown: str = "",
        frontmatter: str = "",
    ) -> None:
        super().__init__(message)
        self.cleaned_markdown = cleaned_markdown
        self.frontmatter = frontmatter


class LLMRequestBudgetExceededError(RuntimeError):
    """A document context hit its LLM request budget (circuit breaker).

    Raised *before* issuing the request that would exceed the limit, so a
    tripped context makes no further API calls. Call sites treat it like
    any other LLM failure: their existing fallbacks keep the unenhanced
    output for the remaining work.
    """


@dataclass
class CacheTally:
    """Content-cache hits and misses seen while a tally is active."""

    hits: int = 0
    misses: int = 0

    def served_from_cache(self, usage: dict[str, dict[str, Any]]) -> bool:
        """Whether every cached call hit and no request reached a model.

        ``usage`` is the item's per-model usage: an uncached call (pure
        cleaning, a language rewrite) records no miss but still spends a
        request, so a hit alone does not mean the item cost nothing.
        """
        requests = sum(stats.get("requests", 0) for stats in usage.values())
        return self.hits > 0 and self.misses == 0 and requests == 0


# The tally of the work item currently being processed. A context variable,
# not a processor attribute: batch items share one processor, and the tasks
# an item spawns (asyncio.gather) inherit the item's context, so every cache
# lookup made on its behalf lands in its own tally.
_cache_tally: ContextVar[CacheTally | None] = ContextVar(
    "markitai_llm_cache_tally", default=None
)


@contextmanager
def track_cache_hits() -> Iterator[CacheTally]:
    """Count the content-cache hits/misses of one work item.

    Yields:
        The tally, updated in place until the block exits.
    """
    tally = CacheTally()
    token = _cache_tally.set(tally)
    try:
        yield tally
    finally:
        _cache_tally.reset(token)


# The work item whose usage and request budget a tracking context belongs to.
# Same reasoning as _cache_tally: a batch shares one processor, and the
# tracking context is a display name (a file's basename, a URL), so two items
# with the same name would otherwise pool their usage and budget, and the
# first to finish would clear the other's.
_usage_scope: ContextVar[str] = ContextVar("markitai_llm_usage_scope", default="")


@contextmanager
def llm_usage_scope(scope: str) -> Iterator[None]:
    """Keep the usage and request budget of one work item apart.

    Scopes nest: an inner scope refines the enclosing one instead of
    replacing it. Each scope is also one document for the request budget
    (see :func:`llm_document_budget`).

    Args:
        scope: Unique id of the item (e.g. its resolved input path).
    """
    outer = _usage_scope.get()
    token = _usage_scope.set(f"{outer}\x00{scope}" if outer else scope)
    try:
        with llm_document_budget():
            yield
    finally:
        _usage_scope.reset(token)


class _BudgetDocument:
    """Identity of one document for :class:`RequestBudget`."""

    __slots__ = ("__weakref__",)


# The document whose request budget the current requests spend. Its
# stages track usage under different contexts (``a.pptx`` for the text,
# ``/abs/a.pptx:images`` for image analysis), but llm.max_requests_per_document
# bounds the document, not each stage.
_budget_document: ContextVar[_BudgetDocument | None] = ContextVar(
    "markitai_llm_budget_document", default=None
)


@contextmanager
def llm_document_budget() -> Iterator[None]:
    """Count every LLM request made in this block against one budget.

    A nested block starts a new document. The budget lives as long as the
    block (and the tasks it spawned) does: clearing a stage's context does
    not give the document its requests back.
    """
    token = _budget_document.set(_BudgetDocument())
    try:
        yield
    finally:
        _budget_document.reset(token)


def scoped_usage_key(context: str) -> str:
    """The key a tracking context is accounted under in the current scope."""
    scope = _usage_scope.get()
    return f"{scope}\x00{context}" if scope and context else context


def _display_context(key: str) -> str:
    """The tracking context of a scoped key, for messages."""
    return key.rpartition("\x00")[2]


@dataclass
class _BudgetState:
    """Requests and cost one budget key has used so far."""

    count: int = 0
    spent: float = 0.0
    tripped: bool = False


class RequestBudget:
    """Per-document LLM request circuit breaker.

    Counts every request attempt (retries included) per document and
    refuses further requests once the limit is reached. This bounds the
    retry multiplication of one document: transport retries x instructor
    validation retries x business-level fallback chains.

    Inside :func:`llm_document_budget` (every :func:`llm_usage_scope` is
    one) all stages of the document share one budget: its text enhancement
    and its image analysis, whatever tracking context each uses. Outside
    one, each tracking context is its own budget.
    """

    def __init__(
        self,
        limit: int,
        cost_limit: float = 0.0,
        on_exceeded: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the budget.

        Args:
            limit: Max requests per context; ``<= 0`` disables the breaker.
            cost_limit: Max USD per context; ``<= 0`` disables it. Charged
                after each answer, because a call's price is not knowable
                before it is made — so the limit bounds what a document goes
                on to spend, not the single call that crosses it.
            on_exceeded: Called once per context on the first refusal
                (e.g. to mark the trip in the usage report).
        """
        self._limit = limit
        self._cost_limit = cost_limit
        self._on_exceeded = on_exceeded
        self._by_context: dict[str, _BudgetState] = {}
        # Freed with the document block: no clear() call is needed
        self._by_document: weakref.WeakKeyDictionary[_BudgetDocument, _BudgetState] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()

    def _state(self, context: str, *, create: bool = True) -> _BudgetState:
        """The budget *context* (already scoped) spends from. Hold the lock."""
        document = _budget_document.get()
        table: Any = self._by_context if document is None else self._by_document
        key: Any = context if document is None else document
        state = table.get(key)
        if state is None:
            state = _BudgetState()
            if create:
                table[key] = state
        return state

    def _trip(self, context: str, reason: str) -> None:
        """Mark a budget tripped, reporting it once however often it refuses."""
        with self._lock:
            state = self._state(context)
            first = not state.tripped
            state.tripped = True
        if first:
            logger.warning(
                f"[LLM:{_display_context(context)}] {reason}: skipping further "
                f"LLM enhancement for this document, keeping unenhanced output."
            )
            if self._on_exceeded is not None:
                self._on_exceeded(context)

    def charge(self, context: str, cost: float) -> None:
        """Account one answer's cost, tripping the breaker once over budget.

        Never raises: the call being charged has already happened and its
        answer is worth keeping. The next :meth:`spend` for this context is
        the one that refuses.
        """
        if not context or self._cost_limit <= 0:
            return
        context = scoped_usage_key(context)
        with self._lock:
            state = self._state(context)
            state.spent += cost
            spent = state.spent
            over = spent > self._cost_limit and not state.tripped
        if over:
            self._trip(
                context,
                f"cost budget exceeded (${spent:.4f} of ${self._cost_limit:.2f})",
            )

    def unknown_cost(self, context: str) -> None:
        """A request whose price will never be known (it timed out after it
        was sent): with a cost limit, the document admits no further request,
        since what it spent is no longer known."""
        if not context or self._cost_limit <= 0:
            return
        self._trip(
            scoped_usage_key(context),
            "cost budget can no longer be enforced (a request timed out and "
            "its cost is unknown)",
        )

    def spend(self, context: str) -> None:
        """Account one request attempt for a context, or refuse it.

        No-op for empty contexts (untracked calls) or a disabled limit.

        Raises:
            LLMRequestBudgetExceededError: When the document already used up
                its budget. The refused attempt is not counted, so at most
                ``limit`` requests are ever issued per document.
        """
        if not context:
            return
        context = scoped_usage_key(context)
        with self._lock:
            state = self._state(context)
            if state.tripped:
                already_tripped = True
            else:
                already_tripped = False
                if self._limit <= 0:
                    return
                if state.count < self._limit:
                    state.count += 1
                    return
        if not already_tripped:
            self._trip(
                context,
                f"request budget exceeded ({self._limit} requests). Raise "
                f"llm.max_requests_per_document (0 disables) if this document "
                f"legitimately needs more requests",
            )
        raise LLMRequestBudgetExceededError(
            f"[LLM:{_display_context(context)}] budget exhausted"
        )

    def exceeded(self, context: str) -> bool:
        """Whether the context's document has tripped the breaker."""
        context = scoped_usage_key(context)
        with self._lock:
            return self._state(context, create=False).tripped

    @property
    def limit(self) -> int:
        """Max requests per document (``<= 0`` means unlimited)."""
        return self._limit

    def remaining(self, context: str) -> int | None:
        """Requests the context's document may still issue, or None when unlimited.

        Lets a caller that knows its request count up front (a document
        split into chunks) refuse before spending anything, instead of
        tripping the breaker halfway through.
        """
        if not context or self._limit <= 0:
            return None
        context = scoped_usage_key(context)
        with self._lock:
            state = self._state(context, create=False)
            if state.tripped:
                return 0
            return max(self._limit - state.count, 0)

    def clear(self, context: str) -> None:
        """Reset a context's budget outside a document block.

        Inside :func:`llm_document_budget` this is a no-op: the document's
        other stages may still be running, and a finished stage must not
        hand its requests back.
        """
        context = scoped_usage_key(context)
        with self._lock:
            self._by_context.pop(context, None)


# Retryable transport exceptions (canonical definition; markitai.llm.processor
# re-exports this tuple, since processor imports engine and not vice versa).
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)


def try_repair_instructor_response(
    exc: Exception,
    response_model: type,
) -> tuple[Any, Any] | None:
    """Try to repair JSON from a failed instructor response.

    Only ever called on the last rung of the structured staircase
    (``MD_JSON``), the one tier where the model hand-writes JSON into the
    answer text. Above it the provider constrains the output, so a failure
    there means something a text fixer cannot fix.

    When instructor's retry mechanism fails (all retries exhausted), the
    last LLM completion is still available. This function extracts the raw
    text, repairs it with the single JSON repair primitive
    (``markitai.utils.text.repair_json_string``), and constructs the
    Pydantic model manually.

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

    repaired = repair_json_string(content)
    if repaired is not None:
        try:
            result = response_model.model_validate_json(repaired)
        except Exception:
            pass
        else:
            logger.info(
                f"[JSON repair] Successfully repaired malformed JSON "
                f"for {response_model.__name__}"
            )
            return result, last

    logger.debug(f"[JSON repair] Repair attempt failed for {response_model.__name__}")
    return None


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Walk an exception and everything nested inside it, once each.

    Covers explicit and implicit chaining plus instructor's
    ``failed_attempts``, which holds the per-retry exceptions that would
    otherwise be invisible behind an ``InstructorRetryException``.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        for attr_name in ("__cause__", "__context__"):
            nested = getattr(current, attr_name, None)
            if isinstance(nested, BaseException):
                stack.append(nested)

        for attempt in getattr(current, "failed_attempts", None) or ():
            attempt_exc = getattr(attempt, "exception", None)
            if isinstance(attempt_exc, BaseException):
                stack.append(attempt_exc)


def find_non_retryable_provider_error(exc: BaseException) -> ProviderError | None:
    """Find a wrapped non-retryable ProviderError inside nested exceptions."""
    for nested in _iter_exception_chain(exc):
        if isinstance(nested, ProviderError) and not nested.retryable:
            return nested
    return None


def find_fatal_document_error(exc: BaseException) -> BaseException | None:
    """Find an error that every further call for the same document repeats.

    A non-retryable provider error (see
    :func:`find_non_retryable_provider_error`) or a rejected credential
    (litellm ``AuthenticationError`` / ``PermissionDeniedError``): the key
    or the model is the same for every page batch of the document, so a
    caller splitting it into several calls stops sending the rest.
    """
    from litellm.exceptions import AuthenticationError, PermissionDeniedError

    provider_error = find_non_retryable_provider_error(exc)
    if provider_error is not None:
        return provider_error
    for nested in _iter_exception_chain(exc):
        if isinstance(nested, (AuthenticationError, PermissionDeniedError)):
            return nested
    return None


def _describe_request_failure(exc: BaseException) -> str:
    """The provider's own answer inside an instructor wrapper, with its status.

    Instructor reports a rejected request as a retry exception; the HTTP
    status and the provider's message (a 400 "Thinking mode does not support
    this tool_choice") are what the user can act on.
    """
    for nested in _iter_exception_chain(exc):
        status = getattr(nested, "status_code", None)
        if isinstance(status, int) and status >= 400:
            return f"HTTP {status} {format_error_message(nested)}"
    return format_error_message(exc)


def _timed_out_after_sending(exc: BaseException) -> bool:
    """Whether *exc* is a request that timed out once it was sent.

    A connection that could not be opened in time never reached the
    provider: it retries like any connection failure.
    """
    if not isinstance(exc, Timeout):
        return False
    return not any(
        type(nested).__name__ in ("ConnectTimeout", "ConnectTimeoutError")
        for nested in _iter_exception_chain(exc)
    )


def find_budget_exceeded_error(
    exc: BaseException,
) -> LLMRequestBudgetExceededError | None:
    """Find a wrapped request-budget refusal inside nested exceptions.

    Instructor wraps whatever the adapter raised, so the circuit breaker
    would otherwise look like an ordinary structured-call failure and earn
    a pointless retry on the next staircase rung.
    """
    for nested in _iter_exception_chain(exc):
        if isinstance(nested, LLMRequestBudgetExceededError):
            return nested
    return None


def reject_unfinished_answer(raw_response: Any) -> None:
    """Refuse an answer the model did not finish.

    Cut at the output limit (``finish_reason`` ``length``) or stopped by the
    provider's content filter (``content_filter``, as LiteLLM maps Gemini's
    SAFETY and similar stops): neither is an answer to keep, cache or write
    as enhanced output (contract v1.6, section 6.6).

    Raises:
        ValueError: The answer was cut short or filtered.
    """
    choices = getattr(raw_response, "choices", None)
    if not choices:
        return
    finish_reason = getattr(choices[0], "finish_reason", None)
    if finish_reason == "length":
        raise ValueError("Output truncated due to max_tokens limit")
    if isinstance(finish_reason, str) and finish_reason.startswith("content_filter"):
        raise ValueError("The provider's content filter stopped the answer")


def extract_cached_tokens(raw_response: Any) -> int:
    """Cache-read input tokens from a litellm-normalized response usage.

    litellm maps both OpenAI's ``prompt_tokens_details.cached_tokens`` and
    Anthropic's ``cache_read_input_tokens`` onto the same
    ``usage.prompt_tokens_details.cached_tokens`` field. Returns 0 whenever
    the provider returned no cache breakdown.
    """
    usage = getattr(raw_response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", None) or 0


async def run_structured_ladder(
    *,
    acompletion: Callable[..., Awaitable[Any]],
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    ladder: Sequence[instructor.Mode],
    call_id: str,
    max_tokens: int | None = None,
) -> tuple[Any, Any]:
    """Run one structured call down the capability staircase.

    Rungs come from ``markitai.llm.structured`` (most native mode the pool
    supports first). A rung that fails drops to the next one; only the last
    rung — always ``MD_JSON`` — gets JSON repair, because it is the only
    tier where the model writes the JSON itself.

    Non-final rungs get a single instructor attempt (``max_retries=0``):
    re-asking the same model in the same rejected mode is worth less than
    changing mode, and every attempt spends from the document's request
    budget. The final rung keeps the full ``DEFAULT_INSTRUCTOR_MAX_RETRIES``
    validation retries, where instructor feeds the validation error back to
    the model.

    Two failures skip the rest of the staircase instead of descending it:
    a non-retryable ProviderError (deterministic for this process) and a
    request-budget refusal (descending would only re-trip the breaker).

    Args:
        acompletion: litellm-compatible callable handed to instructor
            (the caller decides whether it retries transport errors).
        messages: Chat messages; deep-copied per rung because instructor's
            MD_JSON mode appends its schema to the system message in place.
        response_model: Pydantic model to parse into.
        ladder: Instructor modes, most native first.
        call_id: Log tag.
        max_tokens: Explicit output cap, or None.

    Returns:
        Tuple of (parsed_result, raw_response).
    """
    rungs = tuple(ladder) or (instructor.Mode.MD_JSON,)
    previous_error: BaseException | None = None

    for index, mode in enumerate(rungs):
        is_last = index == len(rungs) - 1
        client = instructor.from_litellm(acompletion, mode=mode)
        try:
            return await cast(
                Awaitable[tuple[Any, Any]],
                client.chat.completions.create_with_completion(
                    # "default" is the logical router group every call
                    # addresses; the router resolves the deployment.
                    model="default",
                    messages=cast(list[Any], copy.deepcopy(messages)),
                    response_model=response_model,
                    max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES if is_last else 0,
                    max_tokens=max_tokens,
                ),
            )
        except Exception as e:
            fatal_provider_error = find_non_retryable_provider_error(e)
            if fatal_provider_error is not None:
                raise fatal_provider_error
            budget_error = find_budget_exceeded_error(e)
            if budget_error is not None:
                if previous_error is not None:
                    # "budget exhausted" alone hides why the budget ran out:
                    # name the failed request the refused one would replace
                    raise LLMRequestBudgetExceededError(
                        f"{budget_error} after {rungs[index - 1].value} failed: "
                        f"{_describe_request_failure(previous_error)}"
                    ) from previous_error
                raise budget_error
            if is_last:
                repaired = try_repair_instructor_response(e, response_model)
                if repaired is None:
                    raise
                return repaired
            logger.warning(
                f"[LLM:{call_id}] Structured mode {mode.value} failed "
                f"({_describe_request_failure(e)}); falling back to "
                f"{rungs[index + 1].value}"
            )
            previous_error = e

    raise RuntimeError(f"[LLM:{call_id}] Unexpected state in structured ladder")


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
        cache_model: ``model`` parameter passed to the persistent cache's
            get and set (part of the SQLite cache key). Call sites pass the
            model-pool fingerprint of the router the call goes through (see
            ``model_list_fingerprint``); the literal ``"default"`` is only
            the no-configuration fallback
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
      (``get(key, content, context=..., model=...)`` /
      ``set(key, content, value, model=...)``)
    - ``track_usage``: ``(model, input_tokens, output_tokens, cost, context,
      cached_tokens)`` — cached defaults to 0 (provider returned no cache
      breakdown).
    - ``calculate_max_tokens``: ``(messages, model_id, router=...) -> int | None``
      (bound from ``LLMProcessor._calculate_dynamic_max_tokens``)
    - ``get_primary_model``: ``(router) -> model_id | None``
      (bound from ``LLMProcessor._get_router_primary_model``)

    The engine also owns the content-cache hit/miss counters: every
    structured call with a cache key counts one hit or one miss, and the
    document and vision services record their manual cache events
    via ``record_cache_hit`` / ``record_cache_miss``.
    """

    def __init__(
        self,
        *,
        router: Any | None = None,
        get_router: Callable[[], Any] | None = None,
        semaphore: asyncio.Semaphore,
        memory_cache: Any,
        persistent_cache: Any,
        track_usage: Callable[[str, int, int, float | None, str, int], None],
        calculate_max_tokens: Callable[..., int | None],
        get_primary_model: Callable[[Any], str | None],
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_budget: RequestBudget | None = None,
        track_timed_out: Callable[[str, str], None] | None = None,
    ) -> None:
        """Exactly one of ``router`` / ``get_router`` must be provided.

        ``get_router`` defers default-router resolution to the first actual
        LLM call: router creation raises for configs without models, and
        that error must surface inside the callers' fallback handling (as
        it did when the engine itself was created lazily at call time),
        not at engine/service construction time.

        ``max_retries`` is the engine-wide transport retry count (the
        processor passes ``router_settings.num_retries``), used by every
        call that does not override it explicitly. The engine owns ALL
        transport retries: the router layer performs none.

        ``request_budget`` is the per-document circuit breaker; every
        request attempt spends from it (None disables budgeting).

        ``track_timed_out`` (``(model, context)``) records a request that
        timed out after it was sent: the provider may have billed it
        without reporting its usage, so its cost is unknown.
        """
        if (router is None) == (get_router is None):
            raise ValueError("LLMEngine requires exactly one of router/get_router")
        self._router = router
        self._get_router = get_router
        self.semaphore = semaphore
        self.memory_cache = memory_cache
        self.persistent_cache = persistent_cache
        self.track_usage = track_usage
        self.calculate_max_tokens = calculate_max_tokens
        self._get_primary_model = get_primary_model
        self.max_retries = max_retries
        self.request_budget = request_budget
        self.track_timed_out = track_timed_out
        # Content-cache hit/miss counters (moved here from LLMProcessor in
        # Phase 2.3). Plain int increments, same as the previous processor
        # attributes (GIL-safe enough for counters).
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def router(self) -> Any:
        """Default router (lazily resolved when built with ``get_router``)."""
        if self._router is not None:
            return self._router
        assert self._get_router is not None
        return self._get_router()

    def _spend_request_budget(self, context: str) -> None:
        """Spend one request from the context's budget (no-op if unbudgeted).

        Raises:
            LLMRequestBudgetExceededError: When the context's budget is
                exhausted.
        """
        if self.request_budget is not None:
            self.request_budget.spend(context)

    def guard_acompletion(
        self,
        acompletion: Callable[..., Awaitable[Any]],
        context: str,
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap an acompletion callable with the request budget check.

        Used where a router's ``acompletion`` is handed to instructor
        directly (bypassing the engine loop), so those requests still
        count against — and are stopped by — the document's budget. Each
        answer's usage is tracked as it arrives: instructor may still
        reject it (unparseable JSON, a schema mismatch), and a rejected
        answer was billed like any other.
        """

        async def guarded(*args: Any, **kwargs: Any) -> Any:
            self._spend_request_budget(context)
            sent_at = time.time()
            try:
                response = await acompletion(*args, **kwargs)
            except Exception as e:
                redact_provider_exception(e, kwargs.get("messages"))
                raise
            stamp_sent_at(response, sent_at)
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.track_usage(
                    getattr(response, "model", None) or kwargs.get("model", "default"),
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    get_response_cost(response),
                    context,
                    extract_cached_tokens(response),
                )
            return response

        return guarded

    def record_cache_hit(self) -> None:
        """Count one content-cache hit (for call-site managed caches)."""
        self._cache_hits += 1
        tally = _cache_tally.get()
        if tally is not None:
            tally.hits += 1

    def record_cache_miss(self) -> None:
        """Count one content-cache miss (for call-site managed caches)."""
        self._cache_misses += 1
        tally = _cache_tally.get()
        if tally is not None:
            tally.misses += 1

    def reset_cache_counters(self) -> None:
        """Reset the hit/miss counters (used by cache clearing)."""
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with memory cache stats, persistent cache stats, and
            combined hit rate (same shape as the historical
            ``LLMProcessor.get_cache_stats``).
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "memory": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": round(hit_rate * 100, 2),
                "size": self.memory_cache.size,
            },
            "persistent": self.persistent_cache.stats(),
        }

    def try_cached(self, call: LLMCall) -> Any | None:
        """Return the cached result for a call without invoking the model.

        Offline/batch callers use this to skip collecting requests whose
        answer is already cached. Hit/miss counters are NOT touched here —
        the batch collector reports its own aggregates.
        """
        if call.cache_key is None:
            return None
        cached = self.memory_cache.get(call.cache_key, call.cache_content)
        if cached is None:
            cached = self.persistent_cache.get(
                call.cache_key,
                call.cache_content,
                context=call.context,
                model=call.cache_model,
            )
            if cached is not None:
                self.memory_cache.set(call.cache_key, call.cache_content, cached)
        if cached is None:
            return None
        if call.deserialize is not None:
            return call.deserialize(cached)
        return call.response_model.model_construct(**cached)

    def write_cache(self, call: LLMCall, result: Any) -> None:
        """Persist a batch-produced result under the call's cache key.

        The caller must apply ``call.validate`` first (its corrections feed
        the final output too); this method only enforces the ``cache_if``
        veto, so a degenerate batch result never poisons the caches.
        """
        if call.cache_key is None:
            return
        if call.cache_if is not None and not call.cache_if(result):
            return
        cache_value = (
            call.serialize(result)
            if call.serialize is not None
            else result.model_dump()
        )
        self.memory_cache.set(call.cache_key, call.cache_content, cache_value)
        self.persistent_cache.set(
            call.cache_key,
            call.cache_content,
            cache_value,
            model=call.cache_model,
        )

    async def complete_structured(self, call: LLMCall) -> tuple[Any, Any]:
        """Run one structured LLM call through the full pipeline.

        Returns:
            Tuple of (parsed_result, raw_response). On cache hit,
            raw_response is None.
        """

        # 1. Cache lookup: in-memory first (fastest), then persistent.
        # The miss is counted before the LLM call so that failed calls also
        # count as misses (matching the historical document_process counting).
        if call.cache_key is not None:
            cached_result = self.try_cached(call)
            if cached_result is not None:
                self.record_cache_hit()
                return cached_result, None

            self.record_cache_miss()

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
                max_tokens = self.calculate_max_tokens(
                    call.messages,
                    self._get_primary_model(active_router),
                    router=active_router,
                )

            # Instructor gets the retrying adapter instead of the router's
            # bare acompletion, so its validation retries and the transport
            # retries compose.
            result, raw_response = await run_structured_ladder(
                acompletion=self._make_retrying_acompletion(
                    active_router, call_id, budget_context=call.context
                ),
                messages=call.messages,
                response_model=call.response_model,
                ladder=router_structured_ladder(active_router),
                call_id=call_id,
                max_tokens=max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Usage was tracked per answer by the adapter, as each one
            # arrived: an answer instructor rejected (unparseable JSON, a
            # failed rung) or that the checks below reject (truncation,
            # validate) was billed like any other. The raw response's usage
            # is instructor's total over its attempts, logged here.
            actual_model = getattr(raw_response, "model", None) or "default"
            usage = getattr(raw_response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            logger.info(
                f"[LLM:{call_id}] {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={elapsed_ms:.0f}ms"
            )

            # Check for truncation or a filtered answer (after accounting,
            # before the cache write: such a result must never be persisted)
            reject_unfinished_answer(raw_response)

            # Validation hook: may correct the result; if it raises, nothing
            # is cached and the error propagates to the caller
            if call.validate is not None:
                result = call.validate(result)

            # Store in both cache layers (unless the cache_if hook vetoes
            # the write, e.g. for degenerate output)
            self.write_cache(call, result)

        return result, raw_response

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str,
        context: str = "",
        max_retries: int | None = None,
        router: Any | None = None,
        require_content: bool = False,
    ) -> LLMResponse:
        """Make a plain text LLM call with the transport retry loop.

        Ported verbatim from ``LLMProcessor._call_llm_with_retry`` (Phase
        2.3): acquires one concurrency slot per attempt (backoff sleeps
        happen with the slot released), tracks usage on every successful
        attempt, and calculates dynamic max_tokens for the active router's
        model pool.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            call_id: Unique identifier for this call (for logging)
            context: Context identifier for usage tracking (e.g., filename)
            max_retries: Retry-attempt override (None -> the engine-wide
                ``max_retries`` from ``router_settings.num_retries``)
            router: Router override (None -> engine default router)
            require_content: Raise ``EmptyLLMResponseError`` instead of
                returning blank content once the retries are exhausted.
                Call sites that cache their result must set this: the text
                path has no schema validation, so an empty answer would
                otherwise be written to the TTL-less persistent cache and
                replayed forever.

        Returns:
            LLMResponse with content and usage info

        Raises:
            EmptyLLMResponseError: If ``require_content`` is set and the
                model returned empty/whitespace-only content.
        """
        if max_retries is None:
            max_retries = self.max_retries

        # Use provided router or default to main router
        active_router = router or self.router

        # Calculate dynamic max_tokens based on input size and target model
        # Pass router so the limit is the minimum across the model pool
        target_model_id = self._get_primary_model(active_router)
        max_tokens = self.calculate_max_tokens(
            messages, target_model_id, router=active_router
        )

        def build_llm_response(
            response: Any,
            actual_model: str,
            input_tokens: int,
            output_tokens: int,
            cost: float | None,
        ) -> LLMResponse:
            # Paid for and tracked already; a cut or filtered answer is still
            # no answer
            reject_unfinished_answer(response)
            # litellm returns Choices (not StreamingChoices) for non-streaming
            content = response.choices[0].message.content or ""
            return LLMResponse(
                content=content,
                model=actual_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

        response = cast(
            LLMResponse,
            await self._acompletion_with_retries(
                active_router=active_router,
                call_id=call_id,
                kwargs={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                max_retries=max_retries,
                own_semaphore=True,
                usage_context=context,
                budget_context=context,
                finalize=build_llm_response,
            ),
        )

        # Usage is already tracked above (the attempts were paid for); only
        # the *result* is rejected, so nothing downstream caches a blank.
        if require_content and not response.content.strip():
            raise EmptyLLMResponseError(
                f"[LLM:{call_id}] Model returned empty content after "
                f"{max_retries + 1} attempts"
            )

        return response

    def _make_retrying_acompletion(
        self,
        active_router: Any,
        call_id: str,
        max_retries: int | None = None,
        budget_context: str | None = None,
    ) -> Callable[..., Awaitable[Any]]:
        """Build an acompletion adapter with the full transport retry loop.

        The adapter is signature-compatible with ``litellm.acompletion``
        (``model=..., messages=..., **kwargs``) and returns the raw
        ModelResponse, so it can be handed to ``instructor.from_litellm``.

        One deliberate difference from ``complete_text`` (a parameter of
        the shared ``_acompletion_with_retries`` loop): it does NOT acquire
        the engine semaphore, since instructor invokes the adapter while
        ``complete_structured`` already holds a slot ("one structured call
        = one concurrency slot").

        Usage is tracked per answer, like the request budget is spent per
        attempt (``budget_context``): instructor may reject an answer
        (unparseable JSON, a rung the provider answered in the wrong
        shape) and ask again, and every answer was billed. The final raw
        response instructor returns carries the total of its attempts, so
        ``complete_structured`` does not track it again.
        """

        async def retrying_acompletion(*args: Any, **kwargs: Any) -> Any:
            return await self._acompletion_with_retries(
                active_router=active_router,
                call_id=call_id,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries,
                own_semaphore=False,
                usage_context=budget_context or "",
                budget_context=budget_context,
            )

        return retrying_acompletion

    def _record_timed_out(
        self,
        exc: BaseException,
        model: str,
        usage_context: str | None,
        budget_context: str | None,
    ) -> None:
        """Account a request that timed out after it was sent.

        The provider may have completed and billed it without a usage
        report, so it counts as a request of unknown price: the document's
        cost reads as incomplete, and a dollar budget admits no further
        request, since what the document spent is no longer known.
        """
        timed_out_model = str(getattr(exc, "model", None) or model)
        record_llm_spend(timed_out_model, 0, 0, None)
        if self.track_timed_out is not None and usage_context:
            self.track_timed_out(timed_out_model, usage_context)
        if budget_context and self.request_budget is not None:
            self.request_budget.unknown_cost(budget_context)

    async def _acompletion_with_retries(
        self,
        *,
        active_router: Any,
        call_id: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any],
        max_retries: int | None = None,
        own_semaphore: bool,
        usage_context: str | None = None,
        budget_context: str | None = None,
        finalize: Callable[[Any, str, int, int, float | None], Any] | None = None,
    ) -> Any:
        """Shared transport retry loop behind ``complete_text`` and
        ``_make_retrying_acompletion``.

        The two call modes differ on these axes, expressed as parameters:

        - ``own_semaphore``: when True (text calls), each attempt acquires
          the engine semaphore and backoff sleeps happen OUTSIDE it —
          sleeping while holding a slot would freeze the whole pool during
          a rate-limit burst. When False (instructor adapter), no slot is
          acquired here: the caller (``complete_structured``) holds one for
          the whole structured call, so backoff sleeps keep it.
        - ``usage_context``: when not None, every answered attempt is
          recorded via ``track_usage`` (both modes pass one: an answer the
          caller rejects afterwards was billed all the same).
        - ``finalize``: maps the successful raw response (plus derived
          model/tokens/cost) to the return value; None returns the raw
          ModelResponse (instructor needs the raw object).

        ``budget_context`` names the document context whose request budget
        every attempt spends from; the budget check runs before each
        attempt, so a tripped context stops retrying without issuing more
        requests.
        """
        if max_retries is None:
            max_retries = self.max_retries

        model = kwargs.get("model", "default")
        last_exception: Exception | None = None

        # Backoff sleeps happen at the top of the next iteration, OUTSIDE the
        # semaphore when own_semaphore is set (see docstring above)
        retry_delay = 0.0
        semaphore_ctx = self.semaphore if own_semaphore else nullcontext()
        # A request that timed out is sent again once, never more
        timeout_resent = False

        for attempt in range(max_retries + 1):
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
                retry_delay = 0.0

            # Per-document circuit breaker: refuse the attempt (outside the
            # try, so the refusal is not mistaken for a transport error)
            if budget_context is not None and self.request_budget is not None:
                try:
                    self.request_budget.spend(budget_context)
                except LLMRequestBudgetExceededError as refusal:
                    if last_exception is None:
                        raise
                    # A refused retry: report the failure it would retry
                    raise LLMRequestBudgetExceededError(
                        f"{refusal} after a failed request: "
                        f"{_describe_request_failure(last_exception)}"
                    ) from last_exception

            start_time = time.perf_counter()

            async with semaphore_ctx:
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
                    sent_at = time.time()
                    response = await active_router.acompletion(*args, **call_kwargs)
                    # DeepSeek's price depends on when the request was sent
                    stamp_sent_at(response, sent_at)

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    actual_model = getattr(response, "model", None) or model

                    # Calculate cost (uses _hidden_params for local providers)
                    cost = get_response_cost(response)

                    usage = getattr(response, "usage", None)
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                    if usage_context is not None:
                        self.track_usage(
                            actual_model,
                            input_tokens,
                            output_tokens,
                            cost,
                            usage_context,
                            extract_cached_tokens(response),
                        )

                    # Log result
                    logger.info(
                        f"[LLM:{call_id}] {actual_model} "
                        f"tokens={input_tokens}+{output_tokens} "
                        f"time={elapsed_ms:.0f}ms cost={format_cost(cost)}"
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
                                f"{max_retries + 1} attempts, returning empty content"
                            )

                    if finalize is not None:
                        return finalize(
                            response, actual_model, input_tokens, output_tokens, cost
                        )
                    return response

                except RETRYABLE_ERRORS as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    redact_provider_exception(e, kwargs.get("messages"))
                    last_exception = e

                    # Check for quota/billing errors that should NOT be retried
                    # These errors are wrapped by LiteLLM as APIConnectionError but
                    # are actually non-recoverable without user action.
                    # Genuine rate limits (RateLimitError) often mention "quota"
                    # in provider text (e.g. "quota will reset after 30s") but ARE
                    # retryable: the router cooldown recorded for the failing
                    # model routes the retry to another model.
                    # HTTP 402 is matched on the status code, not as text: a
                    # 429 saying "Used 29402" tokens is still a rate limit.
                    error_msg_lower = str(e).lower()
                    if isinstance(e, RateLimitError):
                        non_retryable_patterns = (
                            "billing",
                            "payment",
                            "insufficient_quota",
                            "exceeded your current quota",
                        )
                    else:
                        non_retryable_patterns = (
                            "quota",
                            "billing",
                            "payment",
                            "subscription",
                            "insufficient_quota",
                            "exceeded your current quota",
                        )
                    if getattr(e, "status_code", None) == 402 or any(
                        pattern in error_msg_lower for pattern in non_retryable_patterns
                    ):
                        status_code = getattr(e, "status_code", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Quota/billing error (not retrying): "
                            f"status={status_code} {format_error_message(e)} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    timed_out = _timed_out_after_sending(e)
                    if timed_out:
                        self._record_timed_out(e, model, usage_context, budget_context)
                    if attempt == max_retries or (timed_out and timeout_resent):
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

                    if timed_out and not (
                        budget_context
                        and self.request_budget is not None
                        and self.request_budget.exceeded(budget_context)
                    ):
                        timeout_resent = True
                        from markitai.notices import user_notice

                        user_notice(
                            "LLM request to {} timed out ({}) and was sent again "
                            "once; the provider may have billed the first "
                            "request, so the cost is incomplete",
                            getattr(e, "model", None) or model,
                            format_error_message(e),
                        )

                    # Calculate exponential backoff delay
                    retry_delay = min(
                        DEFAULT_RETRY_BASE_DELAY * (2**attempt), DEFAULT_RETRY_MAX_DELAY
                    )

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    redact_provider_exception(e, kwargs.get("messages"))
                    error_msg_lower = str(e).lower()

                    # Model-level errors are retryable (the router cooldown
                    # ensures the next attempt picks a different model).
                    # Pool exhaustion ("no deployments available": every
                    # LiteLLM deployment is cooling down) is retryable too:
                    # the backoff outlives short cooldowns.
                    if (
                        any(p in error_msg_lower for p in MODEL_LEVEL_ERROR_PATTERNS)
                        or POOL_EXHAUSTED_PATTERN in error_msg_lower
                    ):
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

                    # Authentication failures: one line in the log only. The
                    # item that made the call reports the failure (its error
                    # or its fallback warning), and for a refused API key
                    # the router's APIKeyRejectedError already names the
                    # variable to set; printing it here too made the console
                    # repeat the provider's 401 once per layer.
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
                        logger.warning(
                            f"[LLM:{call_id}] Authentication failed: "
                            f"{format_error_message(e)} time={elapsed_ms:.0f}ms"
                        )
                    else:
                        logger.error(
                            f"[LLM:{call_id}] Failed: status={status_code} "
                            f"{format_error_message(e)} time={elapsed_ms:.0f}ms"
                        )
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"[LLM:{call_id}] Unexpected state in retry loop")

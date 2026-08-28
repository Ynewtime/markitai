"""Single routing layer for LLM calls (``MarkitaiRouter``).

Exactly one routing decision is made per call, in one place:

- Local provider models (``claude-agent/``, ``copilot/``, ``chatgpt/``, …)
  cannot go through the LiteLLM Router (its provider resolution does not
  know them), so they are dispatched directly to their registered handler.
  Weighted selection and cooldown for these models live here, in one pure
  function (``select_weighted_model``).
- Standard models are delegated as a *group* to an inner LiteLLM Router,
  which owns their weighted load balancing, per-deployment cooldown, and
  ``fallbacks`` — none of that is re-implemented here.
- Mixed pools select between the local deployments and the standard pool
  with the same pure function. Two-stage weighted sampling (pick the
  standard pool with the pool's summed weight, then let LiteLLM pick a
  deployment inside it by weight) yields exactly the same per-model
  distribution as the historical one-stage sampling over all deployments.

Error-to-cooldown classification also lives here, in one place
(``cooldown_seconds_for_error``); ``markitai.llm.engine`` imports the
shared pattern tuples for its retry decisions.

This module must not import ``markitai.llm.engine`` or
``markitai.llm.processor`` (engine imports the pattern constants from
here).
"""

from __future__ import annotations

import random
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.router import Router
from loguru import logger

from markitai.providers.common import has_images
from markitai.utils.text import format_error_message

# Minimum system-message length worth an Anthropic cache breakpoint
# (Anthropic's smallest cacheable prefix is 1024 tokens ~= 4k chars).
_ANTHROPIC_CACHE_MIN_CHARS = 4096


def _anthropic_cache_breakpoint(messages: list[Any]) -> list[Any]:
    """Mark long system strings with Anthropic's ephemeral cache breakpoint.

    Eligible ``system`` messages convert to content-block form carrying
    ``cache_control``; everything else passes through untouched. Only valid
    for requests that will reach the Anthropic Messages API — callers gate
    on the target deployment. Idempotent: block-form content is left alone.
    """
    result: list[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if (
            isinstance(msg, dict)
            and msg.get("role") == "system"
            and isinstance(content, str)
            and len(content) >= _ANTHROPIC_CACHE_MIN_CHARS
        ):
            result.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            result.append(msg)
    return result


# =============================================================================
# Error classification (single copy; engine imports these for retryability)
# =============================================================================

# Model-level error patterns that indicate the model itself is unavailable
# (not a content/request issue). These warrant a long cooldown so retries
# pick a different model.
MODEL_LEVEL_ERROR_PATTERNS = (
    "user location is not supported",
    "failed_precondition",
    "model is not available",
    "model not found",
    "model_not_available",
    "region is not supported",
    "not available in your region",
)

# Rate-limit indicators (short cooldown; the provider recovers on its own).
RATE_LIMIT_ERROR_PATTERNS = ("429", "rate limit", "quota", "too many requests")

# LiteLLM Router raises this text when every deployment in a group is in
# cooldown. Retryable with backoff: the cooldowns expire on their own.
POOL_EXHAUSTED_PATTERN = "no deployments available"

MODEL_LEVEL_COOLDOWN_SECONDS = 3600.0
RATE_LIMIT_COOLDOWN_SECONDS = 60.0

_RETRY_AFTER_RE = re.compile(r"(\d+)\s*s")


def parse_retry_after_seconds(error_message: str) -> float | None:
    """Extract a "retry in N seconds" hint from an error message.

    Args:
        error_message: Provider error text (any case).

    Returns:
        Seconds to wait, or None if the message carries no hint.
    """
    match = _RETRY_AFTER_RE.search(error_message)
    return float(match.group(1)) if match else None


def cooldown_seconds_for_error(error_message: str) -> float | None:
    """Classify an error message into a routing cooldown duration.

    Model-level errors (region restriction, model removed, …) win over
    rate-limit matches: the model will not recover soon, so it gets the
    long cooldown.

    Args:
        error_message: The raw error text (matched case-insensitively).

    Returns:
        Cooldown in seconds, or None when the error says nothing about the
        model's routability (e.g. a content-specific 400).
    """
    msg = error_message.lower()
    if any(p in msg for p in MODEL_LEVEL_ERROR_PATTERNS):
        return MODEL_LEVEL_COOLDOWN_SECONDS
    if any(p in msg for p in RATE_LIMIT_ERROR_PATTERNS):
        return parse_retry_after_seconds(msg) or RATE_LIMIT_COOLDOWN_SECONDS
    return None


# =============================================================================
# Weighted selection (single copy, pure function)
# =============================================================================


@dataclass(frozen=True)
class RouterCandidate:
    """One selectable routing target.

    Attributes:
        model_id: Deployment id (e.g. ``"claude-agent/haiku"``) or the
            standard-pool sentinel.
        weight: Routing weight; ``<= 0`` means disabled.
        image_capable: Whether the target can serve vision requests.
    """

    model_id: str
    weight: float
    image_capable: bool = True


def select_weighted_model(
    candidates: Sequence[RouterCandidate],
    *,
    cooldowns: Mapping[str, float] | None = None,
    now: float | None = None,
    prefer_image_capable: bool = False,
) -> str | None:
    """Select a model id by weighted random choice.

    Filter order (each step falls back rather than failing):

    1. Vision requests prefer image-capable candidates when at least one
       exists; otherwise all candidates stay eligible.
    2. Candidates with ``weight <= 0`` are excluded (user-disabled); if all
       are disabled, selection is uniform over everything.
    3. Candidates in cooldown are excluded; if everything is cooling down,
       the one expiring soonest is chosen.
    4. Weighted random selection over the survivors.

    Args:
        candidates: Selectable targets.
        cooldowns: Map of model_id to monotonic expiry time.
        now: Current monotonic time (defaults to ``time.monotonic()``).
        prefer_image_capable: Whether the request contains images.

    Returns:
        The selected model id, or None when ``candidates`` is empty.
    """
    models = list(candidates)
    if not models:
        return None

    if prefer_image_capable and len(models) > 1:
        image_capable = [c for c in models if c.image_capable]
        if image_capable:
            if len(image_capable) < len(models):
                excluded = [c.model_id for c in models if not c.image_capable]
                logger.debug(
                    "[Router] Image request: preferring image-capable models, "
                    "excluding {}",
                    excluded,
                )
            models = image_capable
        # No image-capable candidate: proceed with everything and let the
        # underlying provider surface its limitation.

    if len(models) == 1:
        return models[0].model_id

    active = [c for c in models if c.weight > 0]
    if not active:
        # All weights are 0 — uniform random selection
        return random.choice(models).model_id

    if cooldowns:
        current = time.monotonic() if now is None else now
        available = [c for c in active if cooldowns.get(c.model_id, 0) <= current]
        if not available:
            soonest = min(active, key=lambda c: cooldowns.get(c.model_id, 0))
            logger.debug(
                "[Router] All models in cooldown, using soonest-expiring: {}",
                soonest.model_id,
            )
            return soonest.model_id
        active = available

    if len(active) == 1:
        return active[0].model_id

    total_weight = sum(c.weight for c in active)
    r = random.uniform(0, total_weight)
    cumulative = 0.0
    for candidate in active:
        cumulative += candidate.weight
        if r <= cumulative:
            return candidate.model_id

    # Floating-point edge: fall back to the last candidate
    return active[-1].model_id


# =============================================================================
# MarkitaiRouter
# =============================================================================

# Sentinel candidate id representing the whole LiteLLM-managed pool in the
# top-level selection. Its weight is the sum of the pool's model weights, so
# branch probability times LiteLLM's in-group weighting reproduces the
# per-model distribution of direct sampling.
STANDARD_POOL_ID = "litellm:standard-pool"


class MarkitaiRouter:
    """Unified router over local provider handlers and a LiteLLM Router.

    Accepts the same ``model_list`` entry dicts as ``litellm.Router``.
    Local provider entries are dispatched directly to their registered
    handler; standard entries are delegated to an inner LiteLLM Router by
    group name, so LiteLLM's weighted balancing, per-deployment cooldown,
    and configured ``fallbacks`` actually apply to them.
    """

    # Local provider models with confirmed image/vision support.
    # Note: Copilot has a ~2000px dimension limit, but CopilotProvider
    # handles resizing automatically via _resize_image_if_needed()
    _IMAGE_CAPABLE_PATTERNS = (
        "claude-agent/",  # All claude-agent models support vision
        "chatgpt/",  # All ChatGPT models support vision (GPT-5.x)
        "copilot/claude-",  # All Copilot Claude models
        "copilot/gemini-",  # All Copilot Gemini models
        "copilot/gpt-4.1",  # GPT-4.1 series
        "copilot/gpt-4o",  # All GPT-4o variants (including mini)
        "copilot/gpt-5",  # GPT-5 series (gpt-5, gpt-5-mini, gpt-5.1* … gpt-5.6*)
        "copilot/raptor-",  # GitHub's fine-tuned GPT-5 mini (inherits vision)
    )
    # Note: copilot/gpt-3.5*, copilot/gpt-4 (non-4o/4.1), copilot/grok-* do
    # NOT support vision

    def __init__(
        self,
        model_list: list[dict[str, Any]],
        router_settings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize from Router-format model entries.

        Args:
            model_list: Entries of shape ``{"model_name": ...,
                "litellm_params": {"model": ..., "weight": ..., ...}}``.
            router_settings: Options for the inner LiteLLM Router
                (``routing_strategy``, ``timeout``, ``fallbacks``).
                ``num_retries`` is always forced to 0: transport retries
                are owned by ``LLMEngine``'s retry loop, and the inner
                router must not multiply them.
        """
        from markitai.providers import is_local_provider_model

        self._local_entries: list[dict[str, Any]] = []
        self._standard_entries: list[dict[str, Any]] = []
        for entry in model_list:
            model_id = entry.get("litellm_params", {}).get("model", "")
            if is_local_provider_model(model_id):
                self._local_entries.append(entry)
            else:
                self._standard_entries.append(entry)

        settings = dict(router_settings or {})
        # Transport retries are owned by LLMEngine (see module docstring of
        # markitai.llm.engine); router_settings.num_retries configures that
        # loop, not LiteLLM-internal retries.
        settings["num_retries"] = 0
        self._standard_router: Router | None = None
        if self._standard_entries:
            self._standard_router = Router(
                model_list=self._standard_entries, **settings
            )

        # Anthropic prompt caching is only wired when every standard
        # deployment hits the Anthropic Messages API — a mixed pool must not
        # send cache_control blocks to providers that would reject them.
        self._standard_pool_all_anthropic = bool(self._standard_entries) and all(
            str(e.get("litellm_params", {}).get("model", "")).startswith("anthropic/")
            for e in self._standard_entries
        )

        # Selection groups: group name -> local candidates + one pool
        # candidate for the group's standard models.
        self._groups: dict[str, list[RouterCandidate]] = {}
        for entry in self._local_entries:
            group = entry.get("model_name", "default")
            params = entry.get("litellm_params", {})
            model_id = params.get("model", "")
            self._groups.setdefault(group, []).append(
                RouterCandidate(
                    model_id=model_id,
                    weight=params.get("weight", 1.0),
                    image_capable=self._is_image_capable_local(model_id),
                )
            )
        for group, pool in self._standard_pool_candidates().items():
            self._groups.setdefault(group, []).append(pool)

        # Cooldown map: model_id (or STANDARD_POOL_ID) -> monotonic expiry
        self._cooldowns: dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        logger.debug(
            "[Router] MarkitaiRouter: {} standard + {} local models",
            len(self._standard_entries),
            len(self._local_entries),
        )

    def _standard_pool_candidates(self) -> dict[str, RouterCandidate]:
        """Build one pool candidate per standard model group."""
        from markitai.llm.models import get_model_info_cached

        pools: dict[str, RouterCandidate] = {}
        by_group: dict[str, list[dict[str, Any]]] = {}
        for entry in self._standard_entries:
            by_group.setdefault(entry.get("model_name", "default"), []).append(entry)

        for group, entries in by_group.items():
            total_weight = sum(
                e.get("litellm_params", {}).get("weight", 1.0) for e in entries
            )
            image_capable = any(
                get_model_info_cached(e.get("litellm_params", {}).get("model", "")).get(
                    "supports_vision", False
                )
                for e in entries
            )
            pools[group] = RouterCandidate(
                model_id=STANDARD_POOL_ID,
                weight=total_weight,
                image_capable=image_capable,
            )
        return pools

    def _is_image_capable_local(self, model_id: str) -> bool:
        """Whether a local provider model supports image/vision requests."""
        return any(model_id.startswith(p) for p in self._IMAGE_CAPABLE_PATTERNS)

    @property
    def model_list(self) -> list[dict[str, Any]]:
        """Combined model entries (standard first, then local)."""
        return self._standard_entries + self._local_entries

    def record_cooldown(self, model_id: str, seconds: float) -> None:
        """Avoid routing to a model (or the standard pool) for a duration.

        Args:
            model_id: Deployment id or ``STANDARD_POOL_ID``.
            seconds: Cooldown duration in seconds.
        """
        with self._cooldown_lock:
            self._cooldowns[model_id] = time.monotonic() + seconds
        logger.info("[Router] Model {} in cooldown for {:.0f}s", model_id, seconds)

    def _select(self, model: str, prefer_image_capable: bool) -> str:
        """Select a routing target for a logical model name.

        Unknown names pass through unchanged so callers may address a
        concrete deployment id directly.
        """
        candidates = self._groups.get(model)
        if not candidates:
            return model
        with self._cooldown_lock:
            cooldowns = dict(self._cooldowns)
        selected = select_weighted_model(
            candidates,
            cooldowns=cooldowns,
            prefer_image_capable=prefer_image_capable,
        )
        return selected if selected is not None else model

    async def acompletion(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Route one completion call.

        Args:
            model: Logical group name (normally ``"default"``) or a
                concrete deployment id.
            messages: Chat messages.
            **kwargs: Forwarded to the backend (``metadata`` is stripped
                for local providers, which do not accept it).

        Returns:
            LiteLLM ModelResponse.
        """
        from markitai.providers import is_local_provider_model

        selected = self._select(model, has_images(messages))

        if selected == STANDARD_POOL_ID:
            logger.debug("[Router] Routing group '{}' to LiteLLM Router", model)
            return await self._standard_acompletion(model, messages, **kwargs)

        if is_local_provider_model(selected) or self._standard_router is None:
            logger.debug("[Router] Routing to local provider: {}", selected)
            return await self._local_acompletion(selected, messages, **kwargs)

        # Pass-through: a concrete standard deployment id was requested
        logger.debug("[Router] Routing to LiteLLM Router: {}", selected)
        return await self._standard_acompletion(selected, messages, **kwargs)

    async def _standard_acompletion(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Delegate to the inner LiteLLM Router.

        LiteLLM owns per-deployment cooldown and fallbacks. The only
        cooldown recorded here is for the pool as a whole, when LiteLLM
        reports that every deployment in the group is cooling down.
        """
        assert self._standard_router is not None
        if self._standard_pool_all_anthropic:
            messages = _anthropic_cache_breakpoint(messages)
        try:
            return await self._standard_router.acompletion(model, messages, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if POOL_EXHAUSTED_PATTERN in error_msg:
                seconds = (
                    parse_retry_after_seconds(error_msg) or RATE_LIMIT_COOLDOWN_SECONDS
                )
                self.record_cooldown(STANDARD_POOL_ID, seconds)
            raise

    async def _local_acompletion(
        self,
        model_id: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Call a local provider handler (or bare litellm) directly.

        LiteLLM Router's provider resolution does not know custom providers,
        so all local provider handlers are invoked directly for consistency
        — including ones like ``chatgpt/`` where LiteLLM has a native
        handler that would otherwise shadow the custom_provider_map entry.
        """
        # litellm.acompletion and local handlers don't accept metadata
        kwargs.pop("metadata", None)

        try:
            if "/" in model_id:
                from markitai.providers import get_provider

                handler = get_provider(model_id.split("/", 1)[0])
                if handler is not None:
                    return await handler.acompletion(
                        model=model_id,
                        messages=messages,
                        **kwargs,
                    )

            # Models without a registered handler fall back to litellm
            if model_id.startswith("anthropic/"):
                messages = _anthropic_cache_breakpoint(messages)
            return await litellm.acompletion(
                model=model_id,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            seconds = cooldown_seconds_for_error(str(e))
            if seconds is not None:
                self.record_cooldown(model_id, seconds)
                if seconds >= MODEL_LEVEL_COOLDOWN_SECONDS:
                    logger.warning(
                        "[Router] Model {} unavailable (model-level error), "
                        "cooldown {:.0f}s: {}",
                        model_id,
                        seconds,
                        format_error_message(e),
                    )
            raise

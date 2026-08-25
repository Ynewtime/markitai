"""Capability-tiered structured-output mode selection.

Structured (schema-validated) calls are issued in the most native mode the
configured model pool actually supports, ordered most native first:

1. ``TOOLS`` — the schema is a function signature and the provider fills it
   in. Strongest guarantee, no JSON ever appears in the answer text.
2. ``JSON_SCHEMA`` — ``response_format={"type": "json_schema", ...}`` is
   passed through and the provider constrains decoding to the schema.
3. ``MD_JSON`` — the prompt asks for a fenced JSON block and the answer text
   is parsed. The only tier where the model hand-writes JSON, so it is also
   the only tier that needs JSON repair.

Capability comes from metadata alone — LiteLLM's model table for standard
models, and the provider handler's own ``STRUCTURED_OUTPUT_MODE`` for local
providers. **No probe request is ever issued**: a capability check must not
cost a request from the document's ``RequestBudget``.

A pool is only as capable as its weakest routable member, because the router
picks the concrete deployment *after* the mode is fixed. The resolved tier is
the top of a fallback staircase (see ``markitai.llm.engine``): when a rung
fails, the call retries one rung down, ending at ``MD_JSON``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import instructor
import litellm
from loguru import logger

# Mode vocabulary, ordered most native -> most tolerant. Local provider
# handlers declare one of these strings in ``STRUCTURED_OUTPUT_MODE``.
MODE_TOOLS = "tools"
MODE_JSON_SCHEMA = "json_schema"
MODE_MD_JSON = "md_json"

MODE_ORDER: tuple[str, ...] = (MODE_TOOLS, MODE_JSON_SCHEMA, MODE_MD_JSON)

_INSTRUCTOR_MODES: dict[str, instructor.Mode] = {
    MODE_TOOLS: instructor.Mode.TOOLS,
    MODE_JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
    MODE_MD_JSON: instructor.Mode.MD_JSON,
}

# Per-model capability cache: the LiteLLM lookups behind a decision are pure
# metadata reads, but they are not free and the answer never changes within
# a process.
_model_mode_cache: dict[str, str] = {}


def instructor_mode_for_model(model_id: str) -> instructor.Mode:
    """Static instructor mode for one model — the batch/offline counterpart
    of the interactive ladder: preselects the best rung the model's
    metadata claims, with no in-flight fallback possible."""
    return _INSTRUCTOR_MODES[model_structured_mode(model_id)]


def model_structured_mode(model_id: str) -> str:
    """Resolve the best structured-output mode one model supports.

    Args:
        model_id: Deployment id (e.g. ``"openai/gpt-4o"``,
            ``"claude-agent/sonnet"``).

    Returns:
        One of ``MODE_TOOLS`` / ``MODE_JSON_SCHEMA`` / ``MODE_MD_JSON``;
        ``MODE_MD_JSON`` whenever the metadata says nothing.
    """
    cached = _model_mode_cache.get(model_id)
    if cached is not None:
        return cached
    mode = _resolve_model_structured_mode(model_id)
    _model_mode_cache[model_id] = mode
    logger.debug("[Structured] {} supports {}", model_id, mode)
    return mode


def _resolve_model_structured_mode(model_id: str) -> str:
    """Read one model's capability from metadata (never from a request)."""
    from markitai.providers import local_provider_structured_mode

    declared = local_provider_structured_mode(model_id)
    if declared is not None:
        return declared if declared in _INSTRUCTOR_MODES else MODE_MD_JSON

    # LiteLLM's capability table. Both helpers raise for unknown models, and
    # an unknown model must not be assumed capable.
    try:
        if litellm.supports_function_calling(model_id):
            return MODE_TOOLS
    except Exception:
        logger.trace("[Structured] No function-calling metadata for {}", model_id)
    try:
        if litellm.supports_response_schema(model_id):
            return MODE_JSON_SCHEMA
    except Exception:
        logger.trace("[Structured] No response-schema metadata for {}", model_id)
    return MODE_MD_JSON


def structured_mode_ladder(model_ids: Iterable[str]) -> tuple[instructor.Mode, ...]:
    """Build the fallback staircase for a model pool.

    Args:
        model_ids: Deployment ids the pool can route to.

    Returns:
        Instructor modes from the pool's best supported tier down to
        ``MD_JSON``. Always non-empty; an empty pool yields ``(MD_JSON,)``.
    """
    start = 0
    known = False
    for model_id in model_ids:
        if not model_id:
            continue
        known = True
        start = max(start, MODE_ORDER.index(model_structured_mode(model_id)))
        if start == len(MODE_ORDER) - 1:
            break
    if not known:
        return (instructor.Mode.MD_JSON,)
    return tuple(_INSTRUCTOR_MODES[name] for name in MODE_ORDER[start:])


def router_structured_ladder(router: Any) -> tuple[instructor.Mode, ...]:
    """Build the fallback staircase for a router's routable model pool.

    Disabled entries (``weight <= 0``) are excluded: the router never picks
    them, so their capability must not drag the pool down. A pool where
    every entry is disabled falls back to ``MD_JSON``, matching the router's
    own "all weights zero -> uniform over everything" behaviour.
    """
    entries = getattr(router, "model_list", None) or []
    model_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        if not isinstance(params, dict) or params.get("weight", 1) <= 0:
            continue
        model_id = params.get("model")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return structured_mode_ladder(model_ids)

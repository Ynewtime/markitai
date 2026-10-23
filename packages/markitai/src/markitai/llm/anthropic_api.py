"""Anthropic API version 2026-10-21 on every Claude Messages request.

On ``2026-10-21`` the Messages API refuses ``temperature``, ``top_p`` and
``top_k`` (HTTP 400 on every model, even at their defaults), a ``document``
block in an ``assistant`` message, and a ``text`` block with ``citations`` in
a ``user`` message. markitai builds neither of the last two (its messages
carry text and images only), and never asks for sampling settings.

Routes and the version each sends:

- ``anthropic/`` (Claude API, and gateways addressed with ``api_base``),
  ``azure_ai/claude-*`` (Microsoft Foundry) and ``bedrock/claude_platform/``
  (Claude Platform on AWS): header ``anthropic-version: 2026-10-21``.
- ``vertex_ai/claude-*``: body ``anthropic_version: "2026-10-21"`` (was
  ``vertex-2023-10-16``).
- ``bedrock/`` (Converse, LiteLLM's default for Claude), ``bedrock/invoke/``
  and ``bedrock/mantle/``: unchanged. Converse has no Anthropic version,
  InvokeModel keeps ``bedrock-2023-05-31``, and 2026-10-21 on Bedrock exists
  only on the Mantle Messages endpoint for Claude 4.7 and later, which
  LiteLLM's ``mantle/`` route addresses with the InvokeModel body.
- ``--llm-batch`` (anthropic SDK 1.x, held below 2.0 because 2.x sends the
  key as a Bearer token) and the model list: header set by markitai.

LiteLLM (1.104 to 1.107rc) writes ``anthropic-version: 2023-06-01`` after
the caller's headers and forces Vertex's body version, with no setting to
change either, so :func:`install_litellm_api_version` wraps the two places
where it builds them: :meth:`AnthropicModelInfo.get_anthropic_headers`
(whose ``anthropic_version`` argument LiteLLM never passes) and
:meth:`AnthropicConfig.transform_request`, which also drops the sampling
settings from any request on 2026-10-21 whatever added them.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from markitai.constants import ANTHROPIC_API_VERSION, ANTHROPIC_SAMPLING_PARAMS
from markitai.notices import user_notice

SAMPLING_PARAMS = ANTHROPIC_SAMPLING_PARAMS

# Vertex AI's body version before 2026-10-21; LiteLLM still writes it
_VERTEX_LEGACY_VERSION = "vertex-2023-10-16"

_reported: set[tuple[str, tuple[str, ...]]] = set()
_reported_lock = threading.Lock()
_install_lock = threading.Lock()
_installed = False


def sampling_settings(params: Mapping[str, Any]) -> tuple[str, ...]:
    """The sampling settings *params* sets, in the order of SAMPLING_PARAMS.

    An ``extra_body`` dict counts too: LiteLLM merges it into the request.
    """
    extra = params.get("extra_body")
    nested = extra if isinstance(extra, Mapping) else {}
    return tuple(
        name
        for name in SAMPLING_PARAMS
        if params.get(name) is not None or nested.get(name) is not None
    )


def sampling_notice(model: str, settings: Iterable[str]) -> str:
    """The warning for sampling settings markitai does not send to *model*.

    Same words as markitai-rust.
    """
    return (
        f"{', '.join(settings)} not sent to {model}: Anthropic API "
        f"{ANTHROPIC_API_VERSION} rejects temperature, top_p and top_k"
    )


def report_dropped_sampling(model: str, settings: Iterable[str]) -> None:
    """Warn once per process for each model and set of dropped settings."""
    ordered = tuple(name for name in SAMPLING_PARAMS if name in set(settings))
    if not ordered:
        return
    with _reported_lock:
        if (model, ordered) in _reported:
            return
        _reported.add((model, ordered))
    user_notice("{}", sampling_notice(model, ordered))


def without_sampling(params: dict[str, Any], model: str) -> dict[str, Any]:
    """*params* without the sampling settings, warning when it had any."""
    dropped = sampling_settings(params)
    if not dropped:
        return params
    report_dropped_sampling(model, dropped)
    kept = {key: value for key, value in params.items() if key not in SAMPLING_PARAMS}
    extra = kept.get("extra_body")
    if isinstance(extra, Mapping):
        kept["extra_body"] = {
            key: value for key, value in extra.items() if key not in SAMPLING_PARAMS
        }
    return kept


def uses_messages_api(model_id: str) -> bool:
    """Whether LiteLLM sends *model_id* to an Anthropic Messages endpoint on 2026-10-21.

    ``anthropic/`` (any address), and Claude on Microsoft Foundry, Vertex AI
    and Claude Platform on AWS. Bedrock's Converse and InvokeModel routes are
    not (their versions stay as LiteLLM sends them).
    """
    provider, _, name = model_id.partition("/")
    provider = provider.lower()
    if provider == "anthropic":
        return True
    if provider == "bedrock":
        return name.lower().startswith("claude_platform/")
    return provider in {"azure_ai", "vertex_ai"} and "claude" in name.lower()


def install_litellm_api_version() -> None:
    """Make LiteLLM's Anthropic requests use API version 2026-10-21. Idempotent.

    Call before the first LiteLLM completion (the router and the serve model
    test do). Imports LiteLLM.
    """
    global _installed
    with _install_lock:
        if _installed:
            return
        from markitai.utils.litellm_import import import_litellm

        import_litellm()  # the package before its submodules (import race)
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
        from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
            AmazonInvokeConfig,
        )

        get_headers = AnthropicModelInfo.get_anthropic_headers
        transform = AnthropicConfig.transform_request

        @functools.wraps(get_headers)
        def get_anthropic_headers(
            self: Any, *args: Any, anthropic_version: str | None = None, **kwargs: Any
        ) -> dict:
            return get_headers(
                self,
                *args,
                anthropic_version=anthropic_version or ANTHROPIC_API_VERSION,
                **kwargs,
            )

        @functools.wraps(transform)
        def transform_request(self: Any, model: str, *args: Any, **kwargs: Any) -> dict:
            data = transform(self, model, *args, **kwargs)
            if isinstance(self, AmazonInvokeConfig):
                # Bedrock InvokeModel and LiteLLM's Mantle route: unchanged
                return data
            dropped = sampling_settings(data)
            if dropped:
                report_dropped_sampling(model, dropped)
                for name in SAMPLING_PARAMS:
                    data.pop(name, None)
            if data.get("anthropic_version") == _VERTEX_LEGACY_VERSION:
                data["anthropic_version"] = ANTHROPIC_API_VERSION
            return data

        AnthropicModelInfo.get_anthropic_headers = get_anthropic_headers  # type: ignore[method-assign]
        AnthropicConfig.transform_request = transform_request  # type: ignore[method-assign]
        _installed = True

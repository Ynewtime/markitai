"""Shared LLM-judge plumbing for the benchmark scripts.

``scorer.score_with_llm_judge`` (rubric scoring) and
``llm_ab_eval.litellm_judge`` (A/B verdicts) both ask a model for one JSON
object and then dig that JSON out of the provider envelope. Only the
transport and the envelope handling live here; each caller keeps its own
rubric, prompt and failure policy -- the scorer fails closed, the A/B judge
degrades to a tie.

``rapidfuzz``/``litellm`` are dev-only dependencies; this module must never
be imported from ``markitai`` runtime code.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

#: Judges are asked for a bare JSON object; some models still wrap it in prose.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeTransportError(RuntimeError):
    """The judge call itself failed; no content was ever received."""


class JudgeEnvelopeError(RuntimeError):
    """The judge answered, but the envelope carried no usable text."""


class ProviderOptions(TypedDict, total=False):
    """LiteLLM credential/endpoint overrides (never logged, never cached)."""

    api_key: str
    api_base: str


def provider_options(
    api_key: str | None = None, api_base: str | None = None
) -> ProviderOptions:
    """Collect the optional provider overrides litellm accepts as kwargs."""
    options: ProviderOptions = {}
    if api_key is not None:
        options["api_key"] = api_key
    if api_base is not None:
        options["api_base"] = api_base
    return options


def judge_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    max_tokens: int,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Any:
    """One synchronous JSON-only judge completion, without retries.

    Omits ``temperature`` so reasoning models are usable.

    Raises:
        JudgeTransportError: The provider call failed. Provider exception
            chains can include credentials or source documents, so only the
            exception type crosses the boundary.
    """
    import litellm

    try:
        return litellm.completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
            timeout=timeout,
            max_tokens=max_tokens,
            num_retries=0,
            **provider_options(api_key, api_base),
        )
    except Exception as exc:
        raise JudgeTransportError(
            f"Judge request failed ({type(exc).__name__}); no score was recorded"
        ) from None


async def ajudge_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    **litellm_kwargs: Any,
) -> Any:
    """One asynchronous JSON-only judge completion. Provider errors propagate.

    Uses ``temperature=0`` for a deterministic verdict, but retries once
    without ``temperature`` if the judge model rejects it (reasoning models
    such as gpt-5.x only accept the default temperature) so one model quirk
    cannot abort a whole run.
    """
    import litellm

    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "stream": False,
        **litellm_kwargs,
    }

    async def _call(temperature: float | None) -> Any:
        kwargs = dict(base_kwargs)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return await litellm.acompletion(**kwargs)

    try:
        return await _call(0)
    except litellm.exceptions.BadRequestError as exc:
        if "temperature" in str(exc).lower():
            return await _call(None)
        raise


def envelope_content(response: Any, *, require_stop: bool = False) -> str:
    """Read the assistant text out of a chat-completion envelope.

    Args:
        response: The provider response (a litellm ``ModelResponse`` on the
            non-streaming path).
        require_stop: Reject a completion the model did not finish itself
            (truncation, refusal, an unexpected tool call).

    Raises:
        JudgeEnvelopeError: The envelope is malformed, unfinished or empty.
    """
    try:
        choice = response.choices[0]
        if require_stop and choice.finish_reason != "stop":
            raise JudgeEnvelopeError("Judge response was incomplete or refused")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise JudgeEnvelopeError("Judge returned empty or non-text content")
    except (AttributeError, IndexError, TypeError) as exc:
        raise JudgeEnvelopeError("Judge returned an invalid completion") from exc
    return content


def loads_json_object(content: str, *, embedded: bool = False) -> Any:
    """Parse the judge's JSON payload, returning ``None`` when unparseable.

    Args:
        content: The assistant text.
        embedded: Also accept a JSON object wrapped in surrounding prose.
    """
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        pass
    if embedded:
        match = _JSON_OBJECT_RE.search(content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

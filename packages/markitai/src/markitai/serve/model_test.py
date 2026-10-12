"""Why a model test in Settings -> Models failed, by category.

``POST /api/settings/llm/test`` answers ``{ok, detail, reason,
provider_detail}``. ``reason`` is one of :data:`REASONS`; ``detail`` is one
sentence that names the cause; ``provider_detail`` keeps the provider's own
words (sanitized) as a secondary detail. A refusal by HTTP status uses the
sentence shape both backends share (``Model connection returned HTTP 401:
...``), which the workbench translates by status and shows the rest of under
"Details".

A key the server does not have is found before anything is sent: LiteLLM
sends a request without a key when it finds none, and DeepSeek then answers
"Authentication Fails (governor)", which read as a wrong key although the
server had none at all (owner report, 2026-10-12).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from markitai.providers.api_keys import MissingKey, env_set_command

Reason = Literal[
    "missing_key",
    "auth_rejected",
    "model_not_found",
    "rate_limited",
    "network",
    "provider_error",
]

#: Every reason a failed model test can carry
REASONS: tuple[Reason, ...] = (
    "missing_key",
    "auth_rejected",
    "model_not_found",
    "rate_limited",
    "network",
    "provider_error",
)


@dataclass(frozen=True)
class TestFailure:
    """A failed model test as the API reports it."""

    reason: Reason
    detail: str
    provider_detail: str | None = None
    # False for an error that fits no category: its own words are all there is
    categorized: bool = True


def missing_key(
    model: str, api_key: str | None, api_base: str | None
) -> MissingKey | None:
    """The key variable a test of *model* lacks on this server, or None.

    The run's own check (``providers.api_keys.missing_api_key``: an unset
    ``env:`` reference, or one of markitai's API-key providers with no key
    anywhere), widened to every provider the workbench connects: one whose
    key variable LiteLLM reads, that needs a key, with neither a key nor an
    address of its own and that variable unset.
    """
    from markitai.config import LiteLLMParams
    from markitai.providers import is_local_provider_model
    from markitai.providers.api_keys import missing_api_key

    if is_local_provider_model(model):
        return None
    lacking = missing_api_key(
        LiteLLMParams(model=model, api_key=api_key, api_base=api_base)
    )
    if lacking is not None or api_key or api_base:
        return lacking
    from markitai.providers.connectable import CONNECTABLE

    prefix = model.partition("/")[0].lower() if "/" in model else ""
    row = next((row for row in CONNECTABLE if row.provider == prefix), None)
    if (
        row is None
        or row.key_variable is None
        or row.key_optional
        or row.cloud
        or os.environ.get(row.key_variable)
    ):
        return None
    return MissingKey(model, row.key_variable)


def missing_key_failure(item: MissingKey) -> TestFailure:
    """The failure for a test that was not sent: the server has no key."""
    where = (
        "an address of its own still needs one for this provider"
        if item.api_base
        else "nothing was sent"
    )
    return TestFailure(
        "missing_key",
        f"{item.env_var} is not set on the server, so {where}: set it where "
        f"markitai serve runs ({env_set_command(item.env_var)}) and restart "
        "the server, or add a key to this connection",
    )


# Words a provider uses for a model it does not know, whatever the status
# (DeepSeek answers 400 "Model Not Exist", OpenAI 404 "model_not_found",
# Gemini 404 "models/x is not found")
_NOT_FOUND_WORDS = (
    "model_not_found",
    "model not exist",
    "model does not exist",
    "does not exist",
    "is not found",
    "not_found_error",
    "unknown model",
    "invalid model",
    "no such model",
)
# LiteLLM's own words when it finds no key before sending
_NO_KEY_WORDS = (
    "api_key client option must be set",
    "missing api key",
    "api key not found",
    "no api key",
)


def _status_of(exc: BaseException) -> int | None:
    for candidate in (exc, exc.__cause__, exc.__context__):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def _is_network(exc: BaseException) -> bool:
    import asyncio

    if isinstance(exc, asyncio.TimeoutError | TimeoutError | ConnectionError):
        return True
    try:
        import litellm.exceptions as le
    except ImportError:  # pragma: no cover - litellm is a dependency
        return False
    if isinstance(exc, le.Timeout | le.APIConnectionError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a dependency
        return False
    return isinstance(exc, httpx.TransportError)


def _is_timeout(exc: BaseException) -> bool:
    import asyncio

    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return True
    try:
        import litellm.exceptions as le
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, le.Timeout)


def classify(
    exc: BaseException,
    *,
    model: str,
    provider_words: str,
    key_source: str | None,
    key_variable: str | None,
) -> TestFailure:
    """The category, sentence and provider wording of a failed test.

    Args:
        exc: What the test raised.
        model: The model tested.
        provider_words: The provider's own message, sanitized.
        key_source: Where the key the request carried came from ("the key
            saved with this connection", "the key in DEEPSEEK_API_KEY"), or
            None when it carried none.
        key_variable: The provider's key variable, for a key LiteLLM found
            missing before sending.
    """
    words = provider_words or None
    lowered = (provider_words or str(exc)).lower()
    status = _status_of(exc)
    try:
        import litellm.exceptions as le
    except ImportError:  # pragma: no cover
        le = None  # type: ignore[assignment]
    from markitai.providers.errors import AuthenticationError as LocalAuthError

    def http(code: int, cause: str) -> str:
        text = f"Model connection returned HTTP {code}: {cause}"
        return f"{text} ({words})" if words else text

    if key_source is None and any(w in lowered for w in _NO_KEY_WORDS):
        variable = key_variable or "the provider's key variable"
        failure = missing_key_failure(MissingKey(model, variable))
        return TestFailure(failure.reason, failure.detail, words)
    if _is_network(exc):
        if _is_timeout(exc):
            return TestFailure("network", "Model connection test timed out", words)
        cause = f": {words}" if words else ""
        return TestFailure("network", f"Model connection request failed{cause}", words)
    is_rate = status == 429 or (le is not None and isinstance(exc, le.RateLimitError))
    if is_rate:
        return TestFailure(
            "rate_limited",
            http(
                429, "the provider is limiting requests or the account is out of quota"
            ),
            words,
        )
    is_auth = status in (401, 403) or isinstance(exc, LocalAuthError)
    if le is not None:
        is_auth = is_auth or isinstance(
            exc, le.AuthenticationError | le.PermissionDeniedError
        )
    if is_auth:
        source = key_source or "a request without a key"
        return TestFailure(
            "auth_rejected",
            http(status or 401, f"the provider rejected {source}"),
            words,
        )
    is_missing_model = status == 404 or any(w in lowered for w in _NOT_FOUND_WORDS)
    if le is not None:
        is_missing_model = is_missing_model or isinstance(exc, le.NotFoundError)
    if is_missing_model:
        model_id = model.partition("/")[2] or model
        return TestFailure(
            "model_not_found",
            http(status or 404, f"the provider does not offer the model {model_id}"),
            words,
        )
    if status is not None:
        return TestFailure(
            "provider_error", http(status, "the provider returned an error"), words
        )
    return TestFailure(
        "provider_error", words or type(exc).__name__, words, categorized=False
    )

"""Why a model test in Settings -> Models failed, by category.

``POST /api/settings/llm/test`` answers a failure as ``{ok: false, detail,
reason, key_variable?}`` (API contract v1.10, the same as Rust's
``service_probe``). ``reason`` is one of :data:`REASONS`, which the
workbench translates with what to do; ``key_variable`` (with ``no_key``,
when known) names the variable the key would come from, never a value.
``detail`` is shown verbatim under "Provider's message": the sentence shape
both backends share (``Model connection returned HTTP 401: <the provider's
words>``, ``Model connection test timed out``, ``Model connection request
failed: <cause>``), with the provider's own words sanitized.

A key the server does not have is found before anything is sent: LiteLLM
sends a request without a key when it finds none, and DeepSeek then answers
"Authentication Fails (governor)", which read as a wrong key although the
server had none at all (owner report, 2026-10-12).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

from markitai.providers.api_keys import MissingKey, env_set_command

Reason = Literal[
    "no_key",
    "key_rejected",
    "model_not_found",
    "rate_limited",
    "network",
    "timeout",
    "provider_error",
]

#: Every reason a failed model test can carry (contract v1.10)
REASONS: tuple[Reason, ...] = (
    "no_key",
    "key_rejected",
    "model_not_found",
    "rate_limited",
    "network",
    "timeout",
    "provider_error",
)

# An env: reference that names a variable, not a pasted key, can be shown
_VARIABLE_NAME = re.compile(r"[A-Za-z0-9_]{1,128}")


@dataclass(frozen=True)
class TestFailure:
    """A failed model test as the API reports it."""

    reason: Reason
    detail: str
    key_variable: str | None = None
    # False for an error that fits no category: its own words are all there is
    categorized: bool = True

    def answer(self, detail: str | None = None) -> dict[str, object]:
        """The response body, ``key_variable`` only when known."""
        body: dict[str, object] = {
            "ok": False,
            "detail": detail or self.detail,
            "reason": self.reason,
        }
        if self.key_variable:
            body["key_variable"] = self.key_variable
        return body


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


def missing_key_failure(item: MissingKey | None) -> TestFailure:
    """The failure for a test that was not sent: the server has no key.

    *item* is None when the variable is not known (LiteLLM found no key for
    a provider markitai has no table row for).
    """
    variable = item.env_var if item is not None else None
    if variable is not None and not _VARIABLE_NAME.fullmatch(variable):
        # An env: reference holding something other than a name (a pasted
        # key) is never repeated
        variable = None
    where = (
        "an address of its own still needs one for this provider"
        if item is not None and item.api_base
        else "nothing was sent"
    )
    if variable is None:
        return TestFailure(
            "no_key",
            f"No API key is available for this model, so {where}: add a key "
            "to this connection, or set the provider's key variable where "
            "markitai serve runs and restart the server",
        )
    return TestFailure(
        "no_key",
        f"{variable} is not set on the server, so {where}: set it where "
        f"markitai serve runs ({env_set_command(variable)}) and restart "
        "the server, or add a key to this connection",
        key_variable=variable,
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


def _chain(exc: BaseException) -> list[BaseException]:
    # LiteLLM wraps a failure without an answer in an error with a status: a
    # refused connection is InternalServerError 500 "Connection error." over
    # OpenAIError, openai.APIConnectionError, httpx.ConnectError, ...
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 12 and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _is_timeout(exc: BaseException) -> bool:
    import asyncio

    import httpx
    import litellm.exceptions as le

    kinds = (asyncio.TimeoutError, TimeoutError, le.Timeout, httpx.TimeoutException)
    return any(isinstance(e, kinds) for e in _chain(exc))


def _is_network(exc: BaseException) -> bool:
    import httpx
    import litellm.exceptions as le
    import openai

    kinds = (
        ConnectionError,
        le.APIConnectionError,
        openai.APIConnectionError,
        httpx.TransportError,
    )
    return any(isinstance(e, kinds) for e in _chain(exc))


def classify(
    exc: BaseException,
    *,
    model: str,
    provider_words: str,
    keyless: bool,
    key_variable: str | None,
) -> TestFailure:
    """The category and detail of a failed test.

    Args:
        exc: What the test raised.
        model: The model tested.
        provider_words: The provider's own message, sanitized.
        keyless: The request carried no real key (none at all, or the
            keyless placeholder) to a provider that is not a cloud signed in
            with the server's own credentials: a 401/403 then means the key
            is missing, not wrong (as Rust's ``service_probe`` reads it).
        key_variable: The variable the missing key would come from, if one
            is named.
    """
    words = provider_words or None
    lowered = (provider_words or str(exc)).lower()
    status = _status_of(exc)
    import litellm.exceptions as le

    from markitai.providers.errors import AuthenticationError as LocalAuthError

    def refused(code: int | None) -> str:
        # The shape both backends share; the provider's words are the rest
        if code is None:
            return words or type(exc).__name__
        text = f"Model connection returned HTTP {code}"
        return f"{text}: {words}" if words else text

    def no_key() -> TestFailure:
        found = MissingKey(model, key_variable) if key_variable else None
        return missing_key_failure(found)

    if keyless and any(w in lowered for w in _NO_KEY_WORDS):
        return no_key()
    if isinstance(exc, LocalAuthError):
        # A local runtime's sign-in (claude-agent, copilot) is no API key:
        # its own words say what to do, as Rust's subscription failures do
        return TestFailure("provider_error", refused(None))
    # What the provider answered, by status or LiteLLM's class for it
    if status == 429 or isinstance(exc, le.RateLimitError):
        return TestFailure("rate_limited", refused(status or 429))
    if status in (401, 403) or isinstance(
        exc, le.AuthenticationError | le.PermissionDeniedError
    ):
        if keyless:
            return no_key()
        return TestFailure("key_rejected", refused(status or 401))
    if status == 404 or isinstance(exc, le.NotFoundError):
        return TestFailure("model_not_found", refused(status))
    # No answer at all, whatever status LiteLLM gave its wrapper
    if _is_timeout(exc):
        return TestFailure("timeout", "Model connection test timed out")
    if _is_network(exc):
        cause = f": {words}" if words else ""
        return TestFailure("network", f"Model connection request failed{cause}")
    if any(w in lowered for w in _NOT_FOUND_WORDS):
        return TestFailure("model_not_found", refused(status))
    if status is not None:
        if 300 <= status < 400:
            # A redirect is not followed: the address is wrong
            return TestFailure("network", refused(status))
        return TestFailure("provider_error", refused(status))
    return TestFailure("provider_error", refused(None), categorized=False)

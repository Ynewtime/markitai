"""API keys of the API-key LLM providers: checked before sending, named when refused.

LiteLLM reads a provider's key from the environment when the config names
none, and when it finds none it still sends the request: DeepSeek and
OpenRouter then answer HTTP 401 ("Authentication Fails"), Gemini fails
locally after transport retries, and every layer that saw the failure
printed the provider's wording again, without ever naming the variable to
set. So a model whose key is missing never joins the pool (nothing is sent,
and the one message names the variable), and a key the provider refuses is
reported as one non-retryable error that says which variable held it.

Only the providers markitai knows the key variables of are checked
(``constants.PROVIDER_API_KEY_ENV_NAMES``); local providers (``claude-agent/``,
``copilot/``, ``chatgpt/``) authenticate on their own, and any other LiteLLM
provider is passed through untouched.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_API_KEY_ENV_NAMES
from markitai.notices import user_notice
from markitai.providers.errors import ProviderError
from markitai.utils.errors import SelfExplanatoryError

if TYPE_CHECKING:
    from markitai.config import LiteLLMParams, ModelConfig

# LiteLLM's clients for these refuse to run without a key even when
# ``api_base`` points at another server. For the others (DeepSeek,
# OpenRouter) a custom ``api_base`` may be a local server or a gateway that
# needs no key, so a model with one is never held back.
_KEY_REQUIRED_WITH_API_BASE = frozenset({"anthropic", "openai", "gemini"})

# Where the CLI, ``serve`` and the MCP server also read keys from
DOTENV_HINT = "~/.markitai/.env"


def env_set_command(var: str, value: str = "your_key") -> str:
    """The command that sets an environment variable in the user's shell.

    Windows PowerShell: ``$env:VAR="value"``; Windows CMD: ``set VAR=value``;
    elsewhere ``export VAR=value``.
    """
    if sys.platform == "win32":
        if os.environ.get("PSModulePath"):  # noqa: SIM112 - Windows uses mixed case
            return f'$env:{var}="{value}"'
        return f"set {var}={value}"
    return f"export {var}={value}"


def _provider_of(model: str) -> str | None:
    prefix, sep, _ = model.partition("/")
    return prefix.lower() if sep else None


@dataclass(frozen=True)
class MissingKey:
    """A model that cannot be used because a variable it needs is not set.

    Attributes:
        model: The model id as configured (``deepseek/deepseek-flash``).
        env_var: The variable to set.
        api_base: The model has its own ``api_base`` (an OpenAI, Anthropic
            or Gemini model still needs a key there).
    """

    model: str
    env_var: str
    api_base: bool = False


def missing_api_key(params: LiteLLMParams) -> MissingKey | None:
    """The variable a model lacks to be sent anything, or None.

    A model is unusable when its ``api_key``/``api_base`` names an ``env:``
    variable that is not set (the router skips it), or when it belongs to
    an API-key provider and neither a configured ``api_key`` nor any of the
    provider's key variables is set (LiteLLM would send it without a key).
    An empty variable counts as unset, as LiteLLM treats it.
    """
    for value in (params.api_key, params.api_base):
        if value and value.startswith("env:") and os.environ.get(value[4:]) is None:
            return MissingKey(params.model, value[4:])

    provider = _provider_of(params.model)
    names = PROVIDER_API_KEY_ENV_NAMES.get(provider or "")
    if provider is None or names is None:
        return None
    if params.get_resolved_api_key(strict=False):
        return None
    has_base = bool(params.get_resolved_api_base(strict=False))
    if has_base and provider not in _KEY_REQUIRED_WITH_API_BASE:
        return None
    if any(os.environ.get(name) for name in names):
        return None
    if params.api_key and params.api_key.startswith("env:"):
        return MissingKey(params.model, params.api_key[4:], api_base=has_base)
    return MissingKey(params.model, PROVIDER_API_KEY_ENV[provider], api_base=has_base)


def split_by_api_key(
    models: Iterable[ModelConfig],
) -> tuple[list[ModelConfig], list[MissingKey]]:
    """Enabled models with what they need, and what the others are missing.

    Disabled models (``weight <= 0``) are in neither list.
    """
    usable: list[ModelConfig] = []
    missing: list[MissingKey] = []
    for model in models:
        if model.litellm_params.weight <= 0:
            continue
        lacking = missing_api_key(model.litellm_params)
        if lacking is None:
            usable.append(model)
        else:
            missing.append(lacking)
    return usable, missing


def _set_hint(var: str, *, dotenv: bool) -> str:
    hint = f"set it with: {env_set_command(var)}"
    if dotenv:
        hint += f" (or add {var}=your_key to {DOTENV_HINT})"
    return hint


def _not_set(var: str, *, api_base: bool) -> str:
    if api_base:
        # openai/, anthropic/ and gemini/ with a custom api_base
        return (
            f"{var} is not set (LiteLLM needs a key for this provider even "
            "with api_base; any value works for a server that needs none)"
        )
    return f"{var} is not set"


def _join(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items)


def no_usable_model_message(
    missing: Sequence[MissingKey], *, dotenv: bool = False
) -> str:
    """The one message for a model pool none of whose models can be used.

    Args:
        missing: What each model lacks (non-empty).
        dotenv: Also suggest ``~/.markitai/.env`` (the CLI, ``serve`` and
            the MCP server read it; the Python API does not).
    """
    by_var: dict[str, list[MissingKey]] = {}
    for item in missing:
        by_var.setdefault(item.env_var, []).append(item)
    first = missing[0].env_var
    if len(by_var) == 1:
        items = by_var[first]
        return (
            f"{_not_set(first, api_base=any(i.api_base for i in items))}, "
            f"and {_join([i.model for i in items])} "
            f"{'needs' if len(items) == 1 else 'need'} it; "
            f"{_set_hint(first, dotenv=dotenv)}"
        )
    parts = [
        f"{_not_set(var, api_base=any(i.api_base for i in items))} "
        f"({_join([i.model for i in items])})"
        for var, items in by_var.items()
    ]
    return (
        f"No LLM model can be used: {'; '.join(parts)}. Set one of them, "
        f"for example {env_set_command(first)}"
        + (f" (or in {DOTENV_HINT})" if dotenv else "")
    )


def left_out_message(item: MissingKey, *, dotenv: bool = False) -> str:
    """The notice for one model left out of a pool that still has others."""
    return (
        f"{item.model} is left out of the model pool: "
        f"{_not_set(item.env_var, api_base=item.api_base)}; "
        f"{_set_hint(item.env_var, dotenv=dotenv)}"
    )


class MissingAPIKeyError(SelfExplanatoryError, ValueError):
    """LLM enhancement was asked for, but no model has the key it needs.

    Raised before anything is sent. A ``ValueError``, like the other
    "no usable model" errors of the router.
    """

    def __init__(self, message: str, missing: Sequence[MissingKey] = ()) -> None:
        super().__init__(message)
        self.missing = list(missing)


# (model, variable) pairs already reported in this process: the CLI's
# pre-flight says it before conversion starts, and every router built later
# (the vision subset, each serve job, each API call) finds the same gap.
_reported: set[tuple[str, str]] = set()
_reported_lock = threading.Lock()


def mark_reported(items: Iterable[MissingKey]) -> None:
    """Record models whose missing key the caller has already shown."""
    with _reported_lock:
        _reported.update((item.model, item.env_var) for item in items)


def report_left_out(items: Iterable[MissingKey], *, dotenv: bool = False) -> None:
    """Raise a user notice per model left out of the pool, once per process."""
    for item in items:
        with _reported_lock:
            key = (item.model, item.env_var)
            if key in _reported:
                continue
            _reported.add(key)
        user_notice("{}", left_out_message(item, dotenv=dotenv))


# ---------------------------------------------------------------------------
# A key the provider refused
# ---------------------------------------------------------------------------


class APIKeyRejectedError(SelfExplanatoryError, ProviderError):
    """The provider refused the API key (HTTP 401).

    Non-retryable: every further request with the same key fails the same
    way, so the structured-output staircase and a document's remaining
    page batches stop here instead of sending more. The message names the
    model, the variable (or the config entry) that held the key, the
    provider's own words, and how to set a valid key.
    """

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        ProviderError.__init__(self, message, provider=provider, retryable=False)
        self.status_code = 401
        self.model = model

    def again(self) -> APIKeyRejectedError:
        """A fresh copy, for a later call refused without sending anything."""
        return APIKeyRejectedError(str(self), provider=self.provider, model=self.model)


# LiteLLM's Router appends "\n\nLiteLLM: model group 'default' failed ..."
_ROUTER_TAIL = re.compile(r"\s*LiteLLM: model group .*$", re.DOTALL)
_AUTH_ERROR_TYPE = re.compile(r"^authentication_error:\s*", re.IGNORECASE)


def _provider_words(exc: BaseException, messages: Any) -> str:
    """The provider's own message, cleaned, without LiteLLM's additions."""
    from markitai.llm.error_redaction import (
        _LITELLM_HEAD,
        redact_provider_message,
        request_texts,
    )

    text = getattr(exc, "message", None)
    if not isinstance(text, str) or not text:
        text = str(exc)
    text = redact_provider_message(text, request_texts(messages))
    text = _LITELLM_HEAD.sub("", _ROUTER_TAIL.sub("", text).strip())
    # "authentication_error: ..." (the JSON body's type) says what HTTP 401
    # already said
    text = _AUTH_ERROR_TYPE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _iter_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def api_key_rejection(
    exc: BaseException,
    entries: Sequence[dict[str, Any]] = (),
    messages: Any = None,
) -> APIKeyRejectedError | None:
    """The friendly error for an API-key provider's 401, or None.

    Args:
        exc: What a completion call raised.
        entries: The router's model entries (Router format), used to name
            the configured model and to tell a key from ``llm.model_list``
            (``litellm_params.api_key``) from one LiteLLM read from the
            environment.
        messages: The request's chat messages: provider words that repeat
            them are withheld (see ``markitai.llm.error_redaction``).
    """
    try:
        from litellm.exceptions import AuthenticationError
    except ImportError:  # pragma: no cover - litellm is a dependency
        return None

    for nested in _iter_chain(exc):
        if isinstance(nested, APIKeyRejectedError):
            return nested
        if not (
            isinstance(nested, AuthenticationError)
            or getattr(nested, "status_code", None) == 401
        ):
            continue
        provider = str(getattr(nested, "llm_provider", "") or "").lower()
        names = PROVIDER_API_KEY_ENV_NAMES.get(provider)
        if names is None:
            return None
        failed = str(getattr(nested, "model", "") or "")
        candidates = [
            e
            for e in entries
            if _provider_of(str(e.get("litellm_params", {}).get("model", "")))
            == provider
        ]
        entry = next(
            (
                e
                for e in candidates
                if e.get("litellm_params", {}).get("model")
                in (failed, f"{provider}/{failed}")
            ),
            candidates[0] if len(candidates) == 1 else None,
        )
        params = entry.get("litellm_params", {}) if entry else {}
        model = params.get("model") or (
            failed if failed.startswith(f"{provider}/") else f"{provider}/{failed}"
        )
        # What to do comes before the provider's words: format_error_message
        # keeps the first 200 characters of whatever wraps this ("LLM
        # processing failed: ..."), so only the provider's words may be
        # cut. No parentheses of its own either, as the item's fallback
        # warning wraps it in some.
        if params.get("api_key"):
            message = (
                f"{model} rejected the api_key configured for it in "
                "llm.model_list with HTTP 401; check that key"
            )
        else:
            var = next(
                (name for name in names if os.environ.get(name)),
                PROVIDER_API_KEY_ENV[provider],
            )
            message = (
                f"{model} rejected the API key in {var} with HTTP 401; "
                f"set a valid one with: {env_set_command(var)}"
            )
        words = _provider_words(nested, messages)
        if words:
            message += f". Provider message: {words}"
        logger.debug("[LLM] {} (from {})", message, type(nested).__name__)
        return APIKeyRejectedError(message, provider=provider, model=model)
    return None


def is_api_key_problem(text: str) -> bool:
    """Whether an item's LLM error is a refused API key.

    Such an item's own message already says what to set, so the generic
    ``llm.on_failure`` hint would only be noise next to it.
    """
    return " rejected the API key in " in text or (
        " rejected the api_key configured for it" in text
    )

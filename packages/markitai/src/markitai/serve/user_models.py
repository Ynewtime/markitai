"""A user token's own models (workbench contract v1.15 and v1.16, section 8).

Each user token keeps its models and connections in
``~/.markitai/serve/user-models/<token id>.json`` (``{"version": 1,
"models": [...], "providers": [...], "shared_weights": {...}}``,
``llm.model_list`` and ``llm.providers`` entries and its own weight for
models the admin shares; folder 0700, file 0600, replaced atomically),
never in ``config.json``: the admin's Edit config and the CLI never show or
read them.
The settings routes' own operations change them (``serve.app``); this module
keeps them, holds them to the rules that keep the server's credentials and
machine out of a user's reach, and builds the pool a user's conversions
route over: its own models, then the ones the admin shares.

The rules (every write, test and discovery, before anything is sent): no
provider that signs in with the server's own account or cloud login
(``server_credentials``), a key or an address of the user's own
(``key_or_address_required``), no address on the server's own machine
(``server_address``), never an ``env:`` reference, at most :data:`LIMIT`
models (``model_limit``). Every stored entry carries a literal key, or the
keyless placeholder with an address: LiteLLM reads ``OPENAI_API_KEY`` and
co. from the server's environment for an entry without one.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from loguru import logger

from markitai.serve import convert_defaults
from markitai.serve.errors import ApiError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from markitai.config import LLMProviderConfig, MarkitaiConfig, ModelConfig

#: A user token's own models at most.
LIMIT = 50

#: Providers that sign in with the server's own account or cloud login
#: whatever key an entry carries: the subscription runtimes, LiteLLM's
#: OAuth routes, Vertex AI (Google credentials), watsonx and GigaChat
#: (their token variables), SageMaker (the AWS chain). Bedrock is refused
#: only without a key of the user's own (with one, LiteLLM sends it as its
#: bearer key).
SERVER_LOGIN = frozenset(
    {
        "claude-agent",
        "copilot",
        "chatgpt",
        "github_copilot",
        "vertex_ai",
        "vertex_ai_beta",
        "watsonx",
        "watsonx_text",
        "gigachat",
        "sagemaker",
        "sagemaker_chat",
    }
)
_KEYED_CLOUDS = frozenset({"bedrock", "bedrock_mantle"})

_USER_ID = re.compile(r"u_[0-9a-f]{12}")

#: A weight at most (the largest the Rust server's settings take).
MAX_WEIGHT = 2**63 - 1

#: A user's tests of shared models at most: one at a time, and this many per
#: rolling minute (the admin's key pays for them).
SHARED_TESTS_PER_MINUTE = 6
_SHARED_TEST_WINDOW_S = 60.0


@dataclass
class Lists:
    """One user's ``llm.model_list`` and ``llm.providers`` entries, and its
    own weight for shared models (by the admin's deployment id) where it
    differs from the admin's."""

    models: list[dict[str, Any]] = field(default_factory=list)
    providers: list[dict[str, Any]] = field(default_factory=list)
    shared_weights: dict[str, int] = field(default_factory=dict)
    #: The user's own conversion defaults (contract v1.17), None when it
    #: follows the server's. Kept in the same file; not part of the revision.
    convert_defaults: dict[str, Any] | None = None


#: The largest routing weight a write may set (round 18): a weight is a share
#: among the models of one routing group, 0 turns a model off (it stays
#: saved), 100 gives 1 % steps. Files holding a bigger weight still load;
#: only writes are checked.
ROUTING_WEIGHT_MAX = 100
#: Why a routing weight is refused, in every route that writes one.
WEIGHT_RULE = "Routing weight must be a whole number from 0 to 100"


def routing_weight(value: Any) -> bool:
    """Whether a write may set *value* as a routing weight: a whole number
    from 0 to 100 (not a bool, a float or a string)."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= ROUTING_WEIGHT_MAX
    )


def invalid_weight(where: str | None = None) -> ApiError:
    """422 ``invalid_weight``; *where* names the entry of a text save."""
    return ApiError(
        422, "invalid_weight", f"{where}: {WEIGHT_RULE}" if where else WEIGHT_RULE
    )


def check_weights(
    models: list[Any], old: list[Any], name: Callable[[int, dict[str, Any]], str]
) -> None:
    """Refuse a weight of these config *models* (a whole text save's) that
    :func:`routing_weight` refuses, unless the saved *old* list holds the
    same entry (the same ``model_info.id``, else the same group and model)
    with that same weight: a file written before the limit saves unchanged.
    *name* says where an entry is."""

    def key(entry: Any) -> tuple[Any, ...]:
        if not isinstance(entry, dict):
            return ("other",)
        info = entry.get("model_info")
        params = entry.get("litellm_params")
        identity = info.get("id") if isinstance(info, dict) else None
        if identity is not None:
            return ("id", identity)
        model = params.get("model") if isinstance(params, dict) else None
        return ("name", entry.get("model_name"), model)

    def weight(entry: Any) -> Any:
        params = entry.get("litellm_params") if isinstance(entry, dict) else None
        return params.get("weight") if isinstance(params, dict) else None

    kept = {(key(entry), repr(weight(entry))) for entry in old}
    for index, entry in enumerate(models):
        value = weight(entry)
        if value is None or routing_weight(value):
            continue
        if (key(entry), repr(value)) in kept:
            continue
        raise invalid_weight(name(index, entry))


def valid_weight(value: Any) -> bool:
    """Whether *value* is a weight: a whole number from 0 to MAX_WEIGHT."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_WEIGHT
    )


# ---------------------------------------------------------------- refusals


def server_credentials() -> ApiError:
    return ApiError(
        422,
        "server_credentials",
        "this provider signs in with the server's own account; add a model "
        "with your own API key",
    )


def key_or_address_required() -> ApiError:
    return ApiError(
        422,
        "key_or_address_required",
        "a model needs your own API key, or the address of a server that takes none",
    )


def server_address() -> ApiError:
    return ApiError(
        422,
        "server_address",
        "that address is the server's own machine, which belongs to the admin; "
        "use another address",
    )


def environment_credentials() -> ApiError:
    return ApiError(
        422, "invalid_settings_request", "environment credentials are the server's"
    )


def user_token_required() -> ApiError:
    return ApiError(
        403,
        "user_token_required",
        "these routes manage a user token's own models; the admin's are under "
        "/api/settings/llm",
    )


def model_limit() -> ApiError:
    return ApiError(
        409, "model_limit", f"you have {LIMIT} models already; delete one first"
    )


def revoked() -> ApiError:
    return ApiError(
        401, "token_required", "this access token was revoked; sign in again"
    )


# ------------------------------------------------------------------- rules


def _no_key() -> str:
    from markitai.serve.app import NO_KEY

    return NO_KEY


def real_key(value: Any) -> bool:
    """A key of the entry's own: a non-empty string, not the placeholder."""
    return isinstance(value, str) and bool(value.strip()) and value != _no_key()


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def route_provider(model: Any) -> str:
    """The provider LiteLLM sends *model* through: its prefix, or for a bare
    name the provider LiteLLM routes it to (``gemini-2.5-pro`` goes to
    Vertex AI); the bare name itself when LiteLLM cannot route it."""
    if not isinstance(model, str):
        return ""
    prefix, slash, _rest = model.partition("/")
    if slash:
        return prefix.strip().lower()
    try:
        from markitai.utils.litellm_import import import_litellm

        import_litellm()
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        return str(get_llm_provider(model)[1]).lower()
    except Exception:  # a model LiteLLM cannot route sends nothing
        return model.strip().lower()


def default_base(provider: str) -> str | None:
    """*provider*'s documented address (None when every server has its own)."""
    from markitai.providers.connectable import connectable
    from markitai.providers.discovery import provider_default_api_base

    row = connectable(provider)
    return provider_default_api_base(provider) or (row.base if row else None)


def _ipv4(host: str) -> ipaddress.IPv4Address | None:
    """*host* read as a URL parser reads an IPv4 address (``127.1``,
    ``0x7f.0.0.1``, ``2130706433``), else None (a domain name)."""
    parts = host.split(".")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()
    if not 1 <= len(parts) <= 4:
        return None
    numbers: list[int] = []
    for part in parts:
        try:
            if part[:2].lower() == "0x":
                numbers.append(int(part[2:] or "0", 16))
            elif len(part) > 1 and part.startswith("0"):
                numbers.append(int(part, 8))
            else:
                if not part.isdigit() or not part.isascii():
                    return None
                numbers.append(int(part))
        except ValueError:
            return None
    *head, last = numbers
    if any(number > 255 for number in head) or last >= 256 ** (5 - len(numbers)):
        return None
    value = last
    for index, number in enumerate(head):
        value += number << (8 * (3 - index))
    return ipaddress.IPv4Address(value)


def server_machine(base: str) -> bool:
    """Whether requests to *base* would go to the server's own machine: the
    host ``localhost`` or a name under it, a loopback address (IPv4-mapped
    included) or an unspecified one (``0.0.0.0``, ``::``). Not an address
    at all: False (callers refuse it on its own)."""
    try:
        host = urlsplit(base.strip()).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    address: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        address = ipaddress.ip_address(host.partition("%")[0])
    except ValueError:
        address = _ipv4(host)
    if address is None:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is None:
            return address.is_loopback or address.is_unspecified
        address = address.ipv4_mapped
    return address.is_loopback or address.packed[0] == 0


def check_address(provider: str, base: str | None) -> None:
    """Refuse an address that is not HTTP(S), or one where *provider*'s
    requests would go to the server's own machine (its documented address
    when *base* is None)."""
    if base is not None:
        try:
            parts = urlsplit(base.strip())
            valid = parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
        except ValueError:
            valid = False
        if not valid:
            raise ApiError(
                422,
                "invalid_settings_request",
                "api_base must be an http or https address",
            )
    target = base if base is not None else default_base(provider)
    if target is not None and server_machine(target):
        raise server_address()


def check_connection(provider: str, key: Any, base: Any) -> None:
    """Refuse one connection of a user: *provider* with *key* at *base*."""
    provider = provider.strip().lower()
    if provider in SERVER_LOGIN or (provider in _KEYED_CLOUDS and not real_key(key)):
        raise server_credentials()
    if any(isinstance(v, str) and v.strip().startswith("env:") for v in (key, base)):
        raise ApiError(
            422,
            "invalid_settings_request",
            "environment references must be configured on the server",
        )
    address = _text(base)
    check_address(provider, address)
    if not real_key(key) and address is None:
        raise key_or_address_required()


def _params(entry: dict[str, Any], providers: list[dict[str, Any]]) -> dict[str, Any]:
    """An entry's key and address: its own, else its linked connection's."""
    params = entry.get("litellm_params")
    params = dict(params) if isinstance(params, dict) else {}
    info = entry.get("model_info")
    linked = info.get("provider_id") if isinstance(info, dict) else None
    if isinstance(linked, str) and linked:
        connection = next((p for p in providers if p.get("id") == linked), None)
        if connection is None:
            raise ApiError(
                404,
                "settings_entry_missing",
                "This saved provider connection no longer exists. Refresh the "
                "provider list and choose it again.",
            )
        for name in ("api_key", "api_base"):
            if params.get(name) is None and connection.get(name) is not None:
                params[name] = connection[name]
    return params


def check_entry(entry: dict[str, Any], providers: list[dict[str, Any]]) -> None:
    """Refuse a user's model entry the rules do not allow."""
    params = _params(entry, providers)
    check_connection(
        route_provider(params.get("model")),
        params.get("api_key"),
        params.get("api_base"),
    )


def check(lists: Lists) -> None:
    """Hold a user's lists after a change to the rules, then seal them: no
    entry or connection may ever use the server's environment credentials."""
    for connection in lists.providers:
        check_connection(
            str(connection.get("provider") or ""),
            connection.get("api_key"),
            connection.get("api_base"),
        )
    for entry in lists.models:
        check_entry(entry, lists.providers)
    if len(lists.models) > LIMIT:
        raise model_limit()
    for entry in lists.models:
        params = entry.setdefault("litellm_params", {})
        params["use_environment_credentials"] = False
        # Without a key of its own (or of its connection) LiteLLM would take
        # the server's key variable for the provider and send it to the
        # entry's address: the placeholder stands in, as the routes store it
        if not real_key(_params(entry, lists.providers).get("api_key")):
            params["api_key"] = _no_key()
    for connection in lists.providers:
        connection["use_environment_credentials"] = False
        if not real_key(connection.get("api_key")):
            connection["api_key"] = _no_key()


def precheck(
    model: Any, provider: Any = None, credential_provider_id: Any = None
) -> None:
    """Refuse what a request names before anything changes or is sent: a
    provider that signs in with the server's own account, or the
    environment's credentials."""
    named = provider.strip().lower() if isinstance(provider, str) else None
    if route_provider(model) in SERVER_LOGIN or named in SERVER_LOGIN:
        raise server_credentials()
    if isinstance(credential_provider_id, str) and credential_provider_id.startswith(
        "env:"
    ):
        raise environment_credentials()


def answer(value: dict[str, Any]) -> dict[str, Any]:
    """A test or discovery answer for a user: no environment variable named."""
    if value.pop("key_variable", None) is not None:
        value["detail"] = "This model needs an API key, and none is set"
    return value


# ----------------------------------------------------------------- storage


def _read(path: Path) -> Lists:
    """The saved lists; ``ValueError`` names what is wrong with the file."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return Lists()
    except OSError as e:
        raise ValueError(f"{path}: {type(e).__name__}") from None
    try:
        saved = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ValueError(f"{path}: not JSON") from None
    if not isinstance(saved, dict) or saved.get("version") != 1:
        raise ValueError(f"{path}: unsupported version")
    models, providers = saved.get("models"), saved.get("providers")
    weights = saved.get("shared_weights", {})
    if not isinstance(models, list) or not isinstance(providers, list):
        raise ValueError(f"{path}: invalid model entry")
    if not isinstance(weights, dict) or not all(
        isinstance(key, str) and valid_weight(value) for key, value in weights.items()
    ):
        raise ValueError(f"{path}: invalid model entry")
    valid_models = all(
        isinstance(entry, dict)
        and isinstance(entry.get("litellm_params"), dict)
        and isinstance(entry["litellm_params"].get("model"), str)
        for entry in models
    )
    valid_providers = all(
        isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("provider"), str)
        for entry in providers
    )
    if not valid_models or not valid_providers or len(models) > LIMIT:
        raise ValueError(f"{path}: invalid model entry")
    from pydantic import ValidationError

    from markitai.config import LLMProviderConfig, ModelConfig

    try:
        for entry in models:
            ModelConfig.model_validate(entry)
        for entry in providers:
            LLMProviderConfig.model_validate(entry)
    except ValidationError:
        raise ValueError(f"{path}: invalid model entry") from None
    own = saved.get("convert_defaults")
    try:
        defaults = None if own is None else convert_defaults.checked(own)
    except ApiError:
        # Not worth losing the user's models over: it follows the server's
        defaults = None
    return Lists(
        models=models,
        providers=providers,
        shared_weights=weights,
        convert_defaults=defaults,
    )


def _sync_directory(path: Path) -> bool:
    """Flush *path*'s entries to disk (POSIX); whether that is known done."""
    if sys.platform == "win32":
        return True
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(fd)
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def _save(directory: Path, path: Path, lists: Lists) -> None:
    """Replace *path* atomically: staged owner-only beside it, flushed,
    renamed over it, then the folder flushed."""
    from markitai.security import _replace_with_retry

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(directory, 0o700)
    saved: dict[str, Any] = {
        "version": 1,
        "models": lists.models,
        "providers": lists.providers,
    }
    if lists.shared_weights:
        # Absent when empty: a file without overrides keeps its v1.15 shape
        saved["shared_weights"] = dict(sorted(lists.shared_weights.items()))
    if lists.convert_defaults is not None:
        saved["convert_defaults"] = lists.convert_defaults
    data = (json.dumps(saved, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    staged = directory / f".tmp-{path.stem}-{os.urandom(6).hex()}"
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        _replace_with_retry(str(staged), path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    if not _sync_directory(directory):
        logger.debug("[Serve] Could not flush {}", directory)


class UserModels:
    """Every user token's own models: files on disk, a cache in memory.

    A lock per user serializes its writes; a revocation's deletion takes the
    same lock, so a write racing it either lands before (and is deleted) or
    finds the token gone and saves nothing.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._user_locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, Lists] = {}
        # Each user's tests of shared models: running, and their start times
        self._tests: dict[str, tuple[bool, deque[float]]] = {}

    def _path(self, user_id: str) -> Path:
        if not _USER_ID.fullmatch(user_id):
            raise ValueError("not a user token id")
        return self.directory / f"{user_id}.json"

    def _user_lock(self, user_id: str) -> threading.Lock:
        with self._lock:
            return self._user_locks.setdefault(user_id, threading.Lock())

    def _cached(self, user_id: str) -> Lists:
        with self._lock:
            found = self._cache.get(user_id)
        if found is not None:
            return found
        try:
            lists = _read(self._path(user_id))
        except ValueError as e:
            # Left as it is until the user's next change replaces it
            logger.warning(
                "[Serve] Ignoring a user token's saved models ({}); it has none "
                "until it saves one.",
                e,
            )
            lists = Lists()
        with self._lock:
            return self._cache.setdefault(user_id, lists)

    def lists(self, user_id: str) -> Lists:
        """A copy of the user's models and connections (none when unsaved).

        Blocking: the first read of a user reads its file.
        """
        with self._user_lock(user_id):
            return copy.deepcopy(self._cached(user_id))

    def exists(self, user_id: str) -> bool:
        """Whether the user has a file of its own."""
        return self._path(user_id).is_file()

    def count(self, user_id: str) -> int:
        """How many models of its own the user has."""
        with self._user_lock(user_id):
            return len(self._cached(user_id).models)

    def write(
        self,
        user_id: str,
        alive: Callable[[], bool],
        change: Callable[[Lists], None],
    ) -> Lists:
        """Change the user's lists with *change* and save them, if its token
        still exists (*alive*, asked under the user's lock). A change that
        changes nothing writes nothing. Blocking."""
        with self._user_lock(user_id):
            if not alive():
                raise revoked()
            before = self._cached(user_id)
            current = copy.deepcopy(before)
            change(current)
            if current == before:
                return copy.deepcopy(current)
            try:
                _save(self.directory, self._path(user_id), current)
            except OSError as e:
                logger.error(
                    "[Serve] Could not save a user token's models: {}",
                    type(e).__name__,
                )
                raise ApiError(
                    500,
                    "settings_io_failed",
                    f"failed to save your models: {type(e).__name__}",
                ) from None
            with self._lock:
                self._cache[user_id] = current
            return copy.deepcopy(current)

    def delete(self, user_id: str) -> int:
        """Delete the user's models (its token was revoked); how many there
        were. Blocking; ``OSError`` when the file stays."""
        with self._user_lock(user_id):
            count = len(self._cached(user_id).models)
            with self._lock:
                self._cache.pop(user_id, None)
            self._path(user_id).unlink(missing_ok=True)
            _sync_directory(self.directory)
            return count

    def begin_test(self, user_id: str) -> int | None:
        """Start a test of a shared model for the user, within the bound:
        None, or how many seconds until one may start."""
        now = time.monotonic()
        with self._lock:
            running, started = self._tests.get(user_id, (False, deque()))
            while started and now - started[0] >= _SHARED_TEST_WINDOW_S:
                started.popleft()
            if running:
                return 1
            if len(started) >= SHARED_TESTS_PER_MINUTE:
                left = _SHARED_TEST_WINDOW_S - (now - started[0])
                return max(1, math.ceil(left))
            started.append(now)
            self._tests[user_id] = (True, started)
            return None

    def end_test(self, user_id: str) -> None:
        """The user's test of a shared model has ended."""
        with self._lock:
            found = self._tests.get(user_id)
            if found is not None:
                self._tests[user_id] = (False, found[1])

    def sweep(self, current: Callable[[str], bool]) -> None:
        """Delete the files of tokens that no longer exist (an interrupted
        revocation) and files a stopped write left staged."""
        try:
            entries = list(self.directory.iterdir())
        except OSError:
            return
        for path in entries:
            name = path.name
            if name.endswith(".json"):
                orphan = not current(name.removesuffix(".json"))
            else:
                orphan = name.startswith(".tmp-")
            if not orphan or path.is_symlink() or not path.is_file():
                continue
            try:
                path.unlink()
            except OSError as e:
                logger.warning(
                    "[Serve] Could not delete the saved models of a revoked "
                    "access token ({}): {}",
                    path,
                    type(e).__name__,
                )


# -------------------------------------------------------------------- pool


def is_shared(model: ModelConfig) -> bool:
    """Whether the admin shares *model* (``model_info.share: "shared"``)."""
    return model.model_info is not None and model.model_info.share == "shared"


def shared_models(models: Iterable[ModelConfig]) -> list[tuple[int, ModelConfig]]:
    """The admin's saved models that are shared and on (weight above 0),
    with their index in the saved list, in its order."""
    return [
        (index, model)
        for index, model in enumerate(models)
        if is_shared(model) and model.litellm_params.weight > 0
    ]


def shared_id(model: ModelConfig, index: int) -> str:
    """The admin's deployment id of its saved *model* at *index*."""
    from markitai.serve.app import _deployment_id

    return _deployment_id(model, index)


@dataclass(frozen=True)
class Shared:
    """One admin model users see now: shared, admin weight above 0."""

    id: str
    model: str
    provider: str
    #: The admin's weight
    weight: int


def shared_list(
    admin_models: list[ModelConfig], admin_providers: list[LLMProviderConfig]
) -> list[Shared]:
    """The admin's models users see now, in the saved order: what they are,
    never a key, address or connection."""
    return [
        Shared(
            id=shared_id(model, index),
            model=model.litellm_params.model,
            provider=connection_provider(model, admin_providers),
            weight=model.litellm_params.weight,
        )
        for index, model in shared_models(admin_models)
    ]


def effective_weight(lists: Lists, shared: Shared) -> int:
    """The user's weight for *shared*: its own, else the admin's."""
    return lists.shared_weights.get(shared.id, shared.weight)


def set_weight(lists: Lists, shared: Shared, weight: int | None) -> None:
    """Set the user's weight for *shared*: an override only while it differs
    from the admin's (None or the admin's weight removes it; 0 = off for this
    user only)."""
    if weight is None or weight == shared.weight:
        lists.shared_weights.pop(shared.id, None)
    else:
        lists.shared_weights[shared.id] = weight


def connection_provider(
    model: ModelConfig, providers: Iterable[LLMProviderConfig]
) -> str:
    """The provider a shared model's connection names: ``custom`` for an
    OpenAI-compatible endpoint, else the model's prefix."""
    prefix = route_provider(model.litellm_params.model)
    linked = model.model_info.provider_id if model.model_info else None
    custom = any(p.id == linked and p.provider == "custom" for p in providers)
    return "custom" if prefix == "openai" and custom else prefix


@dataclass
class Pool:
    """The configuration one user's conversions run with."""

    config: MarkitaiConfig
    #: The models of the pool (own then shared); empty: no model processing
    models: list[ModelConfig]
    #: Hosts of the shared models' addresses: a failure a user sees never
    #: names where the admin serves a model
    hidden_hosts: tuple[str, ...] = ()


def _host_forms(base: str | None) -> list[str]:
    """``host:port`` and ``host`` of *base*, longest first (none for None)."""
    if not base:
        return []
    try:
        parts = urlsplit(base)
        host = parts.hostname
        netloc = parts.netloc.rpartition("@")[2]
    except ValueError:
        return []
    return [form for form in dict.fromkeys((netloc, host)) if form]


def build_pool(
    base: MarkitaiConfig,
    own: Lists,
    admin_models: list[ModelConfig],
    admin_providers: list[LLMProviderConfig],
) -> Pool:
    """*base* (the admin's configuration) routing to the user's own models
    that are on, then the admin's shared ones with the user's weight for
    each (0 leaves one out of this user's pool only), all in routing group
    ``default`` with no fallbacks, and nothing else. An empty pool turns
    model processing and image descriptions off; the server's environment
    keys and detected models are never in it."""
    from pydantic import ValidationError

    from markitai.config import LLMProviderConfig, ModelConfig

    models: list[ModelConfig] = []
    providers: list[LLMProviderConfig] = []
    hidden: list[str] = []
    for connection in own.providers:
        try:
            providers.append(LLMProviderConfig.model_validate(connection))
        except ValidationError:
            continue
    for entry in own.models:
        try:
            # Saved through the rules; checked again, so that no file can
            # make a user's model reach the server's credentials
            check_entry(entry, own.providers)
            model = ModelConfig.model_validate(entry)
        except (ApiError, ValidationError):
            continue
        # The key and address it is sent with: its own, else its
        # connection's, never the server's key variable (a file saved before
        # the placeholder was stored may lack both)
        merged = _params(entry, own.providers)
        model.litellm_params.api_key = (
            merged["api_key"] if real_key(merged.get("api_key")) else _no_key()
        )
        model.litellm_params.api_base = merged.get("api_base")
        if model.litellm_params.weight <= 0:
            continue
        model.model_name = "default"
        models.append(model)
    for index, saved in shared_models(admin_models):
        weight = own.shared_weights.get(
            shared_id(saved, index), saved.litellm_params.weight
        )
        if weight <= 0:
            continue
        model = saved.model_copy(deep=True)
        model.model_name = "default"
        model.litellm_params.weight = weight
        models.append(model)
        params = model.litellm_params
        linked = model.model_info.provider_id if model.model_info else None
        connection = next((p for p in admin_providers if p.id == linked), None)
        if connection is not None and all(p.id != connection.id for p in providers):
            providers.append(connection.model_copy(deep=True))
        from markitai.config import resolve_env_value

        address = (
            resolve_env_value(params.api_base, strict=False)
            if params.api_base
            else None
        )
        hidden.extend(
            _host_forms(address or default_base(route_provider(params.model)))
        )
    cfg = base.model_copy(deep=True)
    cfg.llm.router_settings.fallbacks = []
    if not models:
        cfg.llm.enabled = False
        cfg.image.alt_enabled = False
        cfg.image.desc_enabled = False
        cfg.llm.model_list = []
        cfg.llm.providers = []
        return Pool(config=cfg, models=[])
    cfg.llm.model_list = models
    cfg.llm.providers = providers
    return Pool(
        config=cfg,
        models=list(models),
        hidden_hosts=tuple(sorted(set(hidden), key=len, reverse=True)),
    )


def hide_hosts(text: str, hosts: Iterable[str]) -> str:
    """*text* without the *hosts* a shared model is served at: each URL
    naming one becomes ``<shared model address>``, each other mention
    ``<shared model host>``."""
    for host in hosts:
        if not host or host not in text:
            continue
        text = re.sub(
            r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]*" + re.escape(host) + r"[^\s\"'<>]*",
            "<shared model address>",
            text,
        )
        text = text.replace(host, "<shared model host>")
    return text

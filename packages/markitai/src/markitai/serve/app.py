"""FastAPI application factory for ``markitai serve``.

Exposes the conversion core over REST + SSE for a local single-user web UI.
Requires the ``markitai[serve]`` extra (fastapi, uvicorn, python-multipart).
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import ipaddress
import json
import os
import re
import secrets
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from loguru import logger
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from markitai import __version__
from markitai.build_info import build_identity
from markitai.constants import PROVIDER_API_KEY_ENV
from markitai.page_capture import MAX_BLOCKS_BYTES
from markitai.providers.discovery import provider_default_api_base
from markitai.runs.history import DEFAULT_SERVE_JOBS_ROOT, PENDING_BATCH_SKIP_REASON
from markitai.serve import (
    config_editor,
    convert_defaults,
    model_test,
    new_access_token,
    user_config,
    user_models,
)
from markitai.serve.access import AccessTokens, clean_label, looks_like_user_token
from markitai.serve.artifacts import (
    item_asset_files,
    markdown_pair,
    split_output_name,
)
from markitai.serve.errors import (
    ApiError,
    error_response,
    fallback_reason,
    validation_detail,
)
from markitai.serve.jobs import (
    JOB_TTL_HOURS,
    SHUTDOWN_EVENT,
    Job,
    JobItem,
    JobRegistry,
    RetryWork,
    aggregate_pricing,
    cleanup_stale_jobs,
    item_base_name,
    job_cached_dir_size,
    job_duration_ms,
    job_expired,
    public_options,
    rehydrate_jobs,
    run_history_expiry,
    run_job,
    run_retry_queue,
    write_job_meta,
)
from markitai.serve.originals import (
    MAX_SNAPSHOT_BYTES,
    VIEW_POLICY,
    ByteCache,
    Original,
    PdfWorkers,
    RenderError,
    clamp_width,
    framed_page,
    item_url,
    kept_blocks,
    kept_capture_files,
    kept_original,
    kept_upload,
    pdf_outline,
    pdf_page_text,
    pdf_pages,
    remove_empty_shots_dir,
    render_image,
    render_pdf_page,
)
from markitai.serve.originals import describe as describe_original
from markitai.serve.originals import media_type_of as original_media_type
from markitai.serve.schemas import (
    AccessTokenList,
    BatchDeleteResult,
    CancelResponse,
    Capabilities,
    ConfigDocument,
    CreatedAccessToken,
    CreateJobResponse,
    DetectedModel,
    DownloadTicket,
    DownloadTicketRequest,
    HistoryEntry,
    ItemResult,
    JobOptions,
    JobRetryBody,
    JobSnapshot,
    LLMDeploymentBatch,
    LLMModelCreate,
    LLMModelDiscoveryRequest,
    LLMModelUpdate,
    LLMProviderCredentials,
    LLMProviderUpdate,
    LLMSettingsPayload,
    LLMSettingsUpdate,
    LLMTestResult,
    ModelDiscoveryResult,
    OriginalDescriptor,
    PageText,
    PdfOutline,
    ProviderConnectionList,
    RegeneratedToken,
    RootInfo,
    SavedConfigDocument,
    UserModelsView,
    UserModelTest,
)
from markitai.serve.tickets import LIFETIME_S as TICKET_LIFETIME_S
from markitai.serve.tickets import DownloadTickets, downloadable
from markitai.serve.workbench import (
    BUNDLED_DIR,
    Workbench,
    missing_page,
    mount_workbench,
)
from markitai.utils.url_redaction import (
    mask_url_credentials,
    redact_log_record,
    url_credential_values,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence
    from typing import TypeAlias

    from markitai.config import LLMProviderConfig, MarkitaiConfig, ModelConfig

    #: One settings write's change of the raw model list and connections
    #: (the connections in place), given the backfill's legacy-id → UUID map
    _Mutation: TypeAlias = Callable[[list[Any], list[Any], dict[str, str]], list[Any]]

DEFAULT_JOBS_ROOT = DEFAULT_SERVE_JOBS_ROOT
DEFAULT_CONFIG_PATH = Path.home() / ".markitai" / "config.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per file
# A folder drop converts like the CLI's directory batch; the job's own
# concurrency limits bound the work, and 1000 is Starlette's default
# per-request file limit for multipart forms
MAX_JOB_ITEMS = 1000
# Hard cap on a whole request body (declared Content-Length), whatever the
# number of files: 5 GB of uploads plus multipart framing slack.
MAX_REQUEST_BYTES = 5 * 1024 * 1024 * 1024 + 64 * 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024
_SSE_PING_INTERVAL = 15.0
# Per-request timeout handed to litellm for POST /api/settings/llm/test, plus
# an outer backstop for local providers (claude-agent/, ...) that spawn a CLI
# and may not honor the litellm timeout parameter.
_LLM_TEST_TIMEOUT_S = 15
_LLM_TEST_BACKSTOP_S = 30.0
ROOT_HINT = "web UI not bundled; POST /api/jobs or see /api/capabilities"
DOCS_URL = "https://markitai.dev"
SOURCE_URL = "https://github.com/Ynewtime/markitai"
# A Markdown result larger than this is not sent as JSON; the file download
# endpoint serves it instead.
MAX_RESULT_BYTES = 64 * 1024 * 1024
# A submission label: what the list calls a folder or URL-list upload.
MAX_LABEL_CHARS = 120
# An uploaded ``.urls`` list larger than this is refused.
MAX_URL_LIST_BYTES = 1024 * 1024
# A download ticket request names one path.
_TICKET_BODY_LIMIT = 8 * 1024
_TOKEN_REGENERATE_PATH = "/api/access-token/regenerate"  # nosec B105 - a route
_TOKEN_REGENERATE_BODY_LIMIT = 1024
_ACCESS_TOKEN_BODY_LIMIT = 1024
# Pages drawn for the Compare view at the same time at most.
_PAGE_RENDER_SLOTS = 2

# Upload names are bounded by UTF-8 *byte* length (ext4 NAME_MAX is 255
# bytes): leave room for the " (N)" dedupe suffix plus the longest derived
# names (".llm.md", ".NNNN.<ext>" assets, ".pageNNNN.<ext>" screenshots).
_MAX_UPLOAD_NAME_BYTES = 180
# Windows reserved device names: reserved bare and with any extension.
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


@dataclass
class ServeState:
    """Application state initialized during lifespan startup.

    Attributes:
        config: Base configuration shared by all jobs (deep-copied per job).
        registry: In-memory job registry.
        llm_source: Where ``config.llm.model_list`` came from: ``"config"``
            (user config file / injected config), ``"detected"`` (startup
            provider auto-detection backfill) or ``"none"`` (empty).
        config_path: Config file the ``/api/settings/llm/models`` endpoints
            write to.
        settings_lock: Serializes the read-modify-write cycles of those
            endpoints so concurrent writes cannot drop each other's entries.
    """

    config: MarkitaiConfig
    registry: JobRegistry
    llm_source: Literal["config", "detected", "none"]
    config_path: Path
    config_origin: Literal["explicit", "environment", "project", "user", "default"]
    configured_models: list[ModelConfig]
    configured_providers: list[LLMProviderConfig]
    detected_models: list[ModelConfig]
    settings_revision: str
    settings_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The loop the app runs on, for request_shutdown() from a signal handler.
    loop: asyncio.AbstractEventLoop | None = None


class _ModelLists(Protocol):
    """The models and connections a settings route works on: the
    configuration file's (``ServeState``, the admin's) or a user token's own
    (:class:`_UserScope`)."""

    configured_models: list[ModelConfig]
    configured_providers: list[LLMProviderConfig]
    detected_models: list[ModelConfig]


@dataclass
class _UserScope:
    """A user token's own models and connections, parsed (contract v1.15)."""

    configured_models: list[ModelConfig]
    configured_providers: list[LLMProviderConfig]
    detected_models: list[ModelConfig] = field(default_factory=list)

    @classmethod
    def of(cls, lists: user_models.Lists) -> _UserScope:
        from markitai.config import LLMProviderConfig, ModelConfig

        return cls(
            [ModelConfig.model_validate(entry) for entry in lists.models],
            [LLMProviderConfig.model_validate(entry) for entry in lists.providers],
        )


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


async def _detect_provider_candidates() -> list[dict[str, Any]]:
    """Return conservative defaults from the shared provider detector."""
    from markitai.providers.discovery import (
        detect_provider_connections,
        startup_model_candidates,
    )

    connections = await detect_provider_connections()
    return startup_model_candidates(connections)


async def _backfill_llm_models(cfg: MarkitaiConfig) -> None:
    """Populate ``cfg.llm.model_list`` once when empty (MODEL env, then detect)."""
    from markitai.config import LiteLLMParams, ModelConfig

    model_env = os.environ.get("MODEL")
    if model_env:
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default", litellm_params=LiteLLMParams(model=model_env)
            )
        ]
        logger.info("[Serve] Using MODEL env var: {}", model_env)
        return
    detected = [c["model"] for c in await _detect_provider_candidates()]
    if detected:
        cfg.llm.model_list = [
            ModelConfig(model_name="default", litellm_params=LiteLLMParams(model=m))
            for m in detected
        ]
        logger.info(
            "[Serve] Auto-detected {} LLM provider(s): {}",
            len(detected),
            ", ".join(detected),
        )


def detect_static_dir() -> Path | None:
    """The vendored workbench bundle shipped in the package, when present."""
    if (BUNDLED_DIR / "manifest.json").is_file():
        return BUNDLED_DIR
    return None


class _BodyLimitMiddleware:
    """Reject requests whose declared Content-Length exceeds the job bound.

    Starlette parses the whole multipart body into spooled temp files before
    the endpoint (and any FastAPI dependency) runs, so the only place to stop
    a multi-GB upload early is ASGI middleware. Chunked bodies without a
    Content-Length header remain bounded only per-file by ``_save_upload`` —
    acceptable for a local single-user tool.
    """

    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            declared: int | None = None
            for key, value in scope.get("headers") or ():
                if key == b"content-length" and value.isdigit():
                    declared = int(value)
                    break
            if declared is not None and declared > self.max_bytes:
                response = error_response(
                    413, "request_too_large", "request body too large"
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
# Methods that may carry an Origin without it being checked: everything else
# (POST, PATCH, PUT, DELETE and any other verb) must come from this origin.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _host_header_hostname(value: str) -> str | None:
    """Extract the lowercased hostname from a ``host[:port]`` header value."""
    try:
        return urlsplit(f"//{value}").hostname
    except ValueError:
        return None


def _normalize_allowed_host(value: str) -> str:
    """Normalize a user-supplied allowlist entry to a bare lowercase hostname."""
    host = value.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _is_trusted_hostname(hostname: str | None, extra_allowed: frozenset[str]) -> bool:
    """Whether *hostname* is safe against DNS rebinding.

    IP-literal hosts are always trusted: rebinding needs the browser to
    resolve an attacker-controlled DNS name, so literal IPs (including LAN
    binds like 192.168.x.x) cannot be spoofed that way.
    """
    if not hostname:
        return False
    if hostname in _LOCAL_HOSTNAMES or hostname in extra_allowed:
        return True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_trusted_origin(
    origin_header: str,
    host_header: str | None,
    extra_allowed: frozenset[str],
) -> bool:
    """Whether *origin_header* may issue a state-changing cross-site request.

    Unlike the Host header, the Origin is fully attacker-controlled, so the
    blanket IP-literal trust of ``_is_trusted_hostname`` must not apply here:
    an attacker can host the exploit page from a bare IP and would otherwise
    pass. Trust only explicit ``--allowed-host`` entries, or an origin whose
    host *and port* equal the already Host-validated ones (i.e. the request
    is same-origin: the attacker's page would have to be served from the very
    address the victim connected to, which makes it the real app). Loopback
    names get no blanket pass: a page on another local port (Jupyter, a dev
    server, ``python -m http.server``) is a separate origin.
    """
    if host_header is None:
        return False
    try:
        origin = urlsplit(origin_header)
        host = urlsplit(f"//{host_header}")
        # A Host without a port means the default port of the origin's scheme
        default_port = _DEFAULT_PORTS.get(origin.scheme)
        origin_port = origin.port or default_port
        host_port = host.port or default_port
    except ValueError:
        return False
    # An origin is scheme://host[:port] and nothing else ("null", a path, a
    # query or credentials make it malformed, and it is refused).
    if (
        origin.scheme not in _DEFAULT_PORTS
        or not origin.hostname
        or origin.username is not None
        or origin.password is not None
        or origin.path not in ("", "/")
        or origin.query
        or origin.fragment
    ):
        return False
    if origin.hostname in extra_allowed:
        return True
    return (
        origin.hostname == host.hostname
        and origin_port is not None
        and origin_port == host_port
    )


class _HostGuardMiddleware:
    """Reject DNS-rebinding hosts and cross-site state-changing requests.

    The loopback peer check (on the settings routes with ``--no-auth``)
    cannot stop DNS rebinding: the browser connects to 127.0.0.1 (so the peer *is* loopback)
    while the page came from an attacker-controlled DNS name. Validating the
    Host header closes that hole. The Origin check covers CSRF: multipart
    POSTs are CORS-safelisted (no preflight), so any website could otherwise
    fire POST /api/jobs — which fetches caller-chosen URLs server-side —
    cross-site. Requests without an Origin header (curl, same-origin GETs)
    pass through untouched.
    """

    def __init__(self, app: Any, allowed_hosts: frozenset[str]) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            host_header: str | None = None
            origin_header: str | None = None
            for key, value in scope.get("headers") or ():
                if key == b"host":
                    host_header = value.decode("latin-1")
                elif key == b"origin":
                    origin_header = value.decode("latin-1")
            host_hostname = (
                _host_header_hostname(host_header) if host_header is not None else None
            )
            if host_header is not None and not _is_trusted_hostname(
                host_hostname, self.allowed_hosts
            ):
                response = error_response(
                    400,
                    "host_not_allowed",
                    f"host '{host_header}' is not allowed; use localhost, an IP "
                    "address, or start the server with --allowed-host",
                )
                await response(scope, receive, send)
                return
            if (
                origin_header is not None
                and scope.get("method", "").upper() not in _SAFE_METHODS
                and not _is_trusted_origin(
                    origin_header, host_header, self.allowed_hosts
                )
            ):
                response = error_response(
                    403,
                    "origin_not_allowed",
                    f"cross-site request from origin '{origin_header}' is not allowed",
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Token authentication
# ---------------------------------------------------------------------------

_AUTH_REQUIRED_DETAIL = (
    "authentication required: open the tokened URL printed at server "
    "startup, or send the token in an 'Authorization: Bearer' header "
    "('?token=' for links and event streams)"
)


def _is_loopback_host(host: str) -> bool:
    """Whether a peer address string names the machine running the server."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _is_loopback_scope(scope: Any) -> bool:
    """ASGI-scope variant of :func:`_is_loopback_host`.

    A request without a client address is not loopback: the server cannot
    tell where it came from.
    """
    client = scope.get("client")
    return bool(client) and _is_loopback_host(client[0])


def _scope_bearer_token(scope: Any) -> str | None:
    """Extract the credential of an ``Authorization: Bearer`` header, if any."""
    for key, value in scope.get("headers") or ():
        if key == b"authorization":
            prefix, _, credential = value.decode("latin-1").partition(" ")
            if prefix.lower() == "bearer" and credential.strip():
                return credential.strip()
            return None
    return None


def _scope_query_param(scope: Any, name: str) -> str | None:
    """Extract one query value (last occurrence wins)."""
    values = parse_qs(scope.get("query_string", b"").decode("latin-1")).get(name)
    return values[-1] if values else None


def _scope_query_token(scope: Any) -> str | None:
    """Extract the ``?token=`` query value (last occurrence wins)."""
    return _scope_query_param(scope, "token")


def _scope_raw_path(scope: Any) -> str:
    """The request path as the client sent it (percent-encoded)."""
    raw = scope.get("raw_path")
    if isinstance(raw, bytes) and raw:
        return raw.decode("latin-1").partition("?")[0]
    return str(scope.get("path", ""))


def _scope_presents_token(scope: Any, token: str) -> bool:
    """Whether the request carries *token* in its header or query string.

    The header is the canonical transport; the query parameter exists for
    surfaces that cannot send headers (EventSource, download links, inline
    images). Comparison is constant-time.
    """
    for presented in (_scope_bearer_token(scope), _scope_query_token(scope)):
        if presented is not None and secrets.compare_digest(
            presented.encode(), token.encode()
        ):
            return True
    return False


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8", "surrogatepass")).digest()


@dataclass(frozen=True)
class Caller:
    """Who an API request acts for (contract v1.13 access levels).

    ``admin``: the server token, or a loopback client presenting no token
    (Python's rule). ``user``: a user token created in the workbench; it
    sees only the jobs it created and never the settings. ``anonymous``: a
    client of a ``--no-auth`` server that is not loopback (it may not reach
    the settings or submit URLs, as before).
    """

    role: Literal["admin", "user", "anonymous"]
    user_id: str | None = None
    label: str | None = None

    def sees(self, job: Job) -> bool:
        """Whether this caller may see *job*: a user sees only its own."""
        return self.user_id is None or job.owner == self.user_id


ADMIN = Caller("admin")
ANONYMOUS = Caller("anonymous")
_CALLER_KEY = "markitai_caller"

#: The routes only the admin may use (403 ``settings_forbidden`` otherwise).
_ACCESS_TOKENS_PATH = "/api/access-tokens"  # nosec B105 - a route


def _admin_path(path: str) -> bool:
    return path.startswith(("/api/settings/", _ACCESS_TOKENS_PATH + "/")) or path in (
        _TOKEN_REGENERATE_PATH,
        _ACCESS_TOKENS_PATH,
    )


def _path_job(method: str, path: str) -> str | None:
    """The job a route names in its path, as the raw segment.

    ``/api/jobs/{job_id}…`` and ``DELETE /api/history/{job_id}``.
    """
    rest = path.removeprefix("/api/jobs/")
    if rest != path:
        return rest.split("/", 1)[0]
    rest = path.removeprefix("/api/history/")
    if rest != path and method == "DELETE" and "/" not in rest:
        return rest
    return None


def _loopback_only_detail(path: str) -> str:
    """Why an admin route refused an untrusted peer of a ``--no-auth`` server."""
    if path == _TOKEN_REGENERATE_PATH:
        return "the access token can be regenerated on loopback only"
    if path.startswith("/api/settings/llm"):
        return "LLM settings are available on loopback only"
    if path.startswith("/api/settings/"):
        return "settings are available on loopback only"
    return "access tokens are available on loopback only"


#: Entries one batch deletion takes at most, and its body at most.
_MAX_BATCH_DELETE = 1000
_BATCH_DELETE_BODY_LIMIT = 256 * 1024


def _delete_entries(body: object) -> list[tuple[str, str | None]]:
    """A batch deletion's entries in order, ``(job_id, None)`` for a job.

    Anything else than the contract's body is 422 ``invalid_delete_request``.
    """

    def refused(detail: str) -> ApiError:
        return ApiError(422, "invalid_delete_request", detail)

    if not isinstance(body, dict):
        raise refused("the body must be a JSON object with jobs and/or items")
    unknown = next((key for key in body if key not in ("jobs", "items")), None)
    if unknown is not None:
        raise refused(f"unknown field '{unknown}'; supported fields: jobs, items")
    jobs = body.get("jobs", [])
    items = body.get("items", [])
    for name, value in (("jobs", jobs), ("items", items)):
        if not isinstance(value, list):
            raise refused(f"{name} must be an array")
    count = len(jobs) + len(items)
    if count == 0:
        raise refused("name at least one job or item")
    if count > _MAX_BATCH_DELETE:
        raise refused(
            f"{count} entries; one request deletes at most {_MAX_BATCH_DELETE}"
        )
    entries: list[tuple[str, str | None]] = []
    for job in jobs:
        if not isinstance(job, str):
            raise refused("jobs must be an array of job ids")
        entries.append((job, None))
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"job_id", "item_id"}
            or not isinstance(item["job_id"], str)
            or not isinstance(item["item_id"], str)
        ):
            raise refused("items must be an array of {job_id, item_id} objects")
        entries.append((item["job_id"], item["item_id"]))
    return entries


def _settings_forbidden_response() -> JSONResponse:
    return error_response(
        403,
        "settings_forbidden",
        "settings access requires the admin token",
        headers={"Cache-Control": "no-store"},
    )


class _TokenAuthMiddleware:
    """Resolve the caller of every API request, and refuse the unknown.

    Once per request: the server token makes the admin; a user token (by
    its digest) makes that user, from any peer; a loopback peer presenting
    no token is the admin (Python's rule: the CLI, tests and
    ``--record-history`` tooling on the operator's machine keep working
    unchanged). Anything else on ``/api/`` is 401, as is a token
    regenerated away or a user token that is not (or no longer) known, even
    from loopback. The LLM settings additionally need the admin token from
    every peer (``protect_local_settings``). The static app shell stays
    reachable so a browser can load the UI from the tokened URL — only
    ``/api/`` carries anything worth protecting. A ``token`` of None
    disables the gate (``--no-auth``): loopback is then the admin, everyone
    else anonymous (no URLs, no settings).

    A user is refused the admin routes (403 ``settings_forbidden``) and
    never learns of another's job: every ``/api/jobs/{job_id}…`` route of a
    job it did not create is 404 ``job_not_found``.

    A ``?ticket=`` (see :mod:`markitai.serve.tickets`) stands in for the
    token on exactly one GET of the path it was issued for, as the caller it
    was issued to; presenting it spends it, and a wrong, spent or expired
    one (or one of a revoked user) is refused for every peer.

    The token is read through *token_source* on every request: the
    workbench can regenerate it (``POST /api/access-token/regenerate``).
    """

    def __init__(
        self,
        app: Any,
        token_source: Callable[[], str | None],
        tickets: DownloadTickets,
        revoked: set[bytes],
        access: AccessTokens,
        registry_source: Callable[[], JobRegistry | None],
    ) -> None:
        self.app = app
        self.token_source = token_source
        self.tickets = tickets
        self.revoked = revoked
        self.access = access
        self.registry_source = registry_source

    def _caller(self, scope: Any, token: str) -> Caller | None:
        presented = [
            value
            for value in (_scope_bearer_token(scope), _scope_query_token(scope))
            if value is not None
        ]
        if any(
            secrets.compare_digest(value.encode(), token.encode())
            for value in presented
        ):
            return ADMIN
        for value in presented:
            user = self.access.authenticate(value)
            if user is not None:
                return Caller("user", user_id=user[0], label=user[1])
        return None

    def _presents_refused(self, scope: Any) -> bool:
        """A token regenerated away, or a user token not (or no longer) known."""
        return any(
            presented is not None
            and (
                _token_digest(presented) in self.revoked
                or looks_like_user_token(presented)
            )
            for presented in (_scope_bearer_token(scope), _scope_query_token(scope))
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        token = self.token_source()
        caller: Caller | None
        if token is None:
            caller = ADMIN if _is_loopback_scope(scope) else ANONYMOUS
        else:
            caller = self._caller(scope, token)
        if caller is None:
            ticket = _scope_query_param(scope, "ticket")
            if ticket is not None:
                valid, user = self.tickets.redeem(ticket, _scope_raw_path(scope))
                if (
                    not valid
                    or method != "GET"
                    or (user is not None and not self.access.exists(user))
                ):
                    response = error_response(
                        401,
                        "ticket_invalid",
                        "download ticket is invalid, expired, already used or "
                        "issued for another path",
                    )
                    await response(scope, receive, send)
                    return
                caller = Caller("user", user_id=user) if user is not None else ADMIN
            elif not _is_loopback_scope(scope) or self._presents_refused(scope):
                # Loopback may come without a token (or with a stale one from
                # an earlier run), but never with one regenerated away or a
                # revoked user's: those must sign in again
                response = error_response(401, "token_required", _AUTH_REQUIRED_DETAIL)
                await response(scope, receive, send)
                return
            else:
                caller = ADMIN
        if caller.user_id is not None:
            if _admin_path(path):
                await _settings_forbidden_response()(scope, receive, send)
                return
            job_id = _path_job(method, path)
            registry = self.registry_source()
            job = (
                registry.get(job_id)
                if job_id is not None and registry is not None
                else None
            )
            if job is not None and not caller.sees(job):
                response = error_response(404, "job_not_found", "job not found")
                await response(scope, receive, send)
                return
        scope.setdefault("state", {})[_CALLER_KEY] = caller
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _state(request: Request) -> ServeState:
    """Return the ServeState initialized by the lifespan."""
    return request.app.state.markitai


def _caller(request: Request) -> Caller:
    """Who *request* acts for, as the token middleware resolved it."""
    caller = request.scope.get("state", {}).get(_CALLER_KEY)
    return caller if isinstance(caller, Caller) else ANONYMOUS


def _is_loopback_peer(request: Request) -> bool:
    """Whether the TCP peer of *request* is the machine running the server.

    Unlike Host/Origin, the peer address is chosen by the kernel, not by the
    caller, so it is the one trustworthy answer to "is this the operator's own
    machine?". A request without a client address is not loopback.
    """
    return request.client is not None and _is_loopback_host(request.client.host)


def _is_trusted_request(request: Request) -> bool:
    """Loopback peer, or a remote peer that presented the serve token.

    Token-bearing remote peers get the loopback trust model: they may submit
    URLs, private and credentialed ones included, and use the settings
    routes. A user token's holder is authenticated too, so it may submit
    URLs (the settings stay the admin's). With token auth disabled
    (``--no-auth``) only loopback is trusted.
    """
    if _is_loopback_peer(request) or _caller(request).role == "user":
        return True
    token = request.app.state.serve_token
    return token is not None and _scope_presents_token(request.scope, token)


_OPEN_FILES_EXHAUSTED_DETAIL = (
    "the server ran out of open files while receiving this upload (each "
    "upload over 1 MB is held open until the request ends); submit fewer "
    "files per job, or raise the open-file limit (ulimit -n) before starting "
    "markitai serve"
)


def _ran_out_of_open_files(exc: BaseException) -> bool:
    """Whether *exc* was caused by the process running out of file descriptors.

    FastAPI turns any error while parsing a form into a generic 400 ("There
    was an error parsing the body") raised from the original error. Starlette
    spools each large upload to an open temporary file for the life of the
    request, so a big folder can exhaust the open-file limit there.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in (
            errno.EMFILE,
            errno.ENFILE,
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _url_rejection_detail(url: str) -> str:
    """Explain why a remote caller's URL was refused, and what to do instead."""
    shown = url if len(url) <= 120 else f"{url[:117]}..."
    return (
        f"refusing to fetch '{shown}': this server runs without token auth "
        "(--no-auth), so it fetches URLs only for its own machine; restart "
        "`markitai serve` without --no-auth and open the printed token URL, or "
        "submit the URL on the server's own machine over loopback."
    )


async def _guard_fetch_targets(request: Request, urls: Sequence[str]) -> None:
    """Refuse any URL fetch requested by an untrusted peer.

    The Host/Origin guard only defends against a malicious *page* driving the
    victim's browser; a caller that talks to the API directly (curl, another
    LAN host) bypasses it entirely. With token auth active this gate is moot
    (an unauthenticated remote peer never reaches the route), but under
    ``--no-auth`` it keeps ``--host 0.0.0.0`` from turning the server into a
    fetch proxy for anyone on the network: public and private targets alike
    are refused (contract §2, 403 ``remote_url_forbidden``), so no DNS answer
    or redirect can decide what an anonymous caller reaches.
    """
    if not urls or _is_trusted_request(request):
        return
    logger.warning(
        "[Serve] Refused {} URL(s) for untrusted non-loopback peer {}",
        len(urls),
        request.client.host if request.client is not None else "?",
    )
    raise ApiError(403, "remote_url_forbidden", _url_rejection_detail(urls[0]))


def _get_job(request: Request, job_id: str) -> Job:
    """Return the job with *job_id* or raise 404.

    A job past history retention is gone even before the sweep removes it,
    and a user token's caller never sees a job it did not create.
    """
    job = _state(request).registry.get(job_id)
    if job is None or job_expired(job) or not _caller(request).sees(job):
        raise ApiError(404, "job_not_found", "job not found")
    return job


def _get_item(job: Job, item_id: str) -> JobItem:
    """Return the item with *item_id* of *job* or raise 404."""
    item = job.get_item(item_id)
    if item is None:
        raise ApiError(404, "item_not_found", "item not found")
    return item


def _bound_name_bytes(name: str, max_bytes: int) -> str:
    """Bound a filename's UTF-8 byte length, preserving its extension.

    ``sanitize_filename`` truncates by characters; multi-byte names (CJK)
    can still exceed NAME_MAX in bytes on Linux, so re-bound here.
    """
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    suffix = Path(name).suffix
    stem = name[: len(name) - len(suffix)]
    budget = max_bytes - len(suffix.encode("utf-8"))
    if budget < 1:  # pathological all-extension name: cut it as a whole
        stem, suffix, budget = name, "", max_bytes
    truncated = stem.encode("utf-8")[:budget].decode("utf-8", "ignore")
    return f"{truncated}{suffix}".lstrip(".") or "upload"


def _sanitize_upload_name(raw: str | None) -> str:
    """Sanitize an upload filename: no path separators, no leading dots.

    Keeps the extension, CJK characters and spaces intact. The UTF-8 byte
    length is bounded first, preserving the extension (``sanitize_filename``'s
    own 200-*char* cut would chop it off for long CJK names); then Windows
    reserved device names (``CON.txt``) are neutralized with a ``_`` prefix.
    """
    from markitai.utils.cli_helpers import sanitize_filename

    base = _bound_name_bytes(
        Path(raw).name if raw else "upload", _MAX_UPLOAD_NAME_BYTES
    )
    name = sanitize_filename(base) or "upload"
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES:
        name = f"_{name}"
    return name


async def _save_upload(
    upload: UploadFile, dest_dir: Path, taken: set[str] | None = None
) -> Path:
    """Stream one upload to disk, enforcing the per-file size limit.

    Names are de-duplicated case-insensitively: on macOS/Windows file
    systems ``Report.pdf`` and ``report.pdf`` are one file, and even on a
    case-sensitive one their outputs would collide once the job archive is
    unpacked there.

    *taken* holds the casefolded names already in *dest_dir* and receives
    the name chosen here. A caller saving many uploads passes one set for
    all of them: listing the directory again for every file made a
    1000-file submit quadratic. Without it the directory is listed.
    """
    name = _sanitize_upload_name(upload.filename)
    if taken is None:
        taken = {entry.name.casefold() for entry in dest_dir.iterdir()}
    target = dest_dir / name
    counter = 2
    while target.name.casefold() in taken or target.exists():
        target = dest_dir / f"{Path(name).stem} ({counter}){Path(name).suffix}"
        counter += 1
    taken.add(target.name.casefold())
    size = 0
    try:
        with target.open("wb") as fh:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ApiError(
                        413,
                        "file_too_large",
                        f"file '{name}' exceeds the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit",
                    )
                fh.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        taken.discard(target.name.casefold())
        raise
    return target


async def _read_url_list(upload: UploadFile) -> list[tuple[str, str | None]]:
    """Read an uploaded ``.urls`` list: its entries become URL items.

    Same formats as the CLI (one URL per line with an optional output name,
    ``#`` comments, or a JSON array of strings or ``{"url", "output_name"}``
    objects); entries that are not HTTP(S) URLs are skipped as the CLI skips
    them, but a list with no usable URL is refused rather than converting
    nothing. Returns ``(url, output file name or None)`` pairs.
    """
    from markitai.urls import UrlListParseError, parse_url_list_text

    name = _sanitize_upload_name(upload.filename)
    data = bytearray()
    while chunk := await upload.read(_UPLOAD_CHUNK):
        data += chunk
        if len(data) > MAX_URL_LIST_BYTES:
            raise ApiError(
                413, "url_list_too_large", f"{name} exceeds the 1 MiB URL list limit"
            )
    try:
        text = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ApiError(422, "invalid_url_list", f"{name} is not UTF-8 text") from None
    try:
        entries = parse_url_list_text(text, name)
    except UrlListParseError:
        raise ApiError(
            422, "invalid_url_list", f"{name} is not a valid URL list"
        ) from None
    parsed: list[tuple[str, str | None]] = []
    for entry in entries:
        url = entry.url.strip()
        if _url_problem(url) is not None:
            continue
        output_name = (entry.output_name or "").strip() or None
        if output_name is not None:
            base = output_name.removesuffix(".md")
            if "/" in output_name or "\\" in output_name or base in ("", ".", ".."):
                raise ApiError(
                    422,
                    "invalid_output_name",
                    f"{name}: a URL output name must be a filename without "
                    "directory components",
                )
            output_name = f"{_sanitize_upload_name(base)}.md"
        parsed.append((url, output_name))
    if not parsed:
        raise ApiError(422, "empty_url_list", f"{name} holds no HTTP(S) URLs")
    return parsed


def _url_problem(url: str) -> str | None:
    """The reason a submitted URL is refused, or None when it is usable."""
    try:
        parsed = urlsplit(url)
        parsed.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        return "invalid_url"
    if not parsed.scheme or not parsed.netloc:
        return "invalid_url"
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return "unsupported_url_scheme"
    return None


def _build_job_config(base: MarkitaiConfig, opts: JobOptions) -> MarkitaiConfig:
    """Deep-copy the base config and apply preset + explicit llm override.

    When LLM ends up enabled, ``keep_base`` is forced on (web semantics).
    """
    from markitai.config import get_preset

    cfg = base.model_copy(deep=True)
    if opts.preset:
        preset = get_preset(opts.preset.lower(), cfg)
        if preset is not None:
            cfg.llm.enabled = preset.llm
            cfg.image.alt_enabled = preset.alt
            cfg.image.desc_enabled = preset.desc
            cfg.ocr.enabled = preset.ocr
            cfg.screenshot.enabled = preset.screenshot
    if opts.llm is not None:
        cfg.llm.enabled = opts.llm
    if opts.ocr is not None:
        cfg.ocr.enabled = opts.ocr
    # Orthogonal to the preset and to LLM: a profile shapes the output, so a
    # plain local conversion can carry one too.
    if opts.profile is not None:
        cfg.output.profile = opts.profile
    # Individual overrides, applied after the preset so an explicit choice
    # always wins over the bundle. None means "not asked for", which stays
    # distinct from asked-for-and-off.
    if opts.alt is not None:
        cfg.image.alt_enabled = opts.alt
    if opts.desc is not None:
        cfg.image.desc_enabled = opts.desc
    if opts.screenshot is not None:
        cfg.screenshot.enabled = opts.screenshot
    if opts.screenshot_only is not None:
        cfg.screenshot.screenshot_only = opts.screenshot_only
        if opts.screenshot_only:
            cfg.screenshot.enabled = True  # as --screenshot-only implies on the CLI
    if opts.pure is not None:
        cfg.llm.pure = opts.pure
    if opts.no_cache is not None:
        cfg.cache.no_cache = opts.no_cache
    if opts.no_compress is not None:
        cfg.image.compress = not opts.no_compress
    if opts.strategy is not None:
        cfg.fetch.strategy = opts.strategy
    # -b/--backend selects a file-conversion path, which lives in two
    # different flags rather than one setting (see cli/main.py).
    if opts.backend is not None:
        cfg.fetch.cloudflare.convert_enabled = opts.backend == "cloudflare"
    if cfg.llm.enabled and not cfg.llm.model_list:
        # Only a server configuration that enables LLM gets here: a request
        # that asks for a model is refused first (_require_model).
        logger.warning(
            "[Serve] LLM enabled by the configuration but no models are "
            "configured; running without LLM enhancement"
        )
        cfg.llm.enabled = False
    if cfg.llm.enabled:
        # Web semantics (CLI untouched): the UI diff view needs the base .md
        # variant alongside .llm.md.
        cfg.llm.keep_base = True
    return cfg


def _parse_job_options(raw: str) -> JobOptions:
    """Read the ``options`` form field; problems are named, never echoed."""
    try:
        return JobOptions.model_validate_json(raw or "{}")
    except ValidationError as e:
        raise ApiError(
            422, "invalid_options", f"invalid options: {validation_detail(e.errors())}"
        ) from None


def _check_preset(state: ServeState, opts: JobOptions) -> None:
    if opts.preset is not None:
        from markitai.config import get_preset

        if get_preset(opts.preset.lower(), state.config) is None:
            raise ApiError(422, "unknown_preset", f"unknown preset '{opts.preset}'")


def _requests_llm(opts: JobOptions, base: MarkitaiConfig) -> bool:
    """Whether the request itself asks for model processing.

    As opposed to a server configuration that happens to enable it: an
    explicit ``llm``, else the chosen preset's LLM switch.
    """
    if opts.llm is not None:
        return opts.llm
    if opts.preset:
        from markitai.config import get_preset

        preset = get_preset(opts.preset.lower(), base)
        return bool(preset is not None and preset.llm)
    return False


def _llm_routable(state: ServeState) -> bool:
    effective = _effective_models(state.configured_models, state.detected_models)
    return any(_is_deployment_routable(model) for model in effective)


@dataclass
class _Conversion:
    """What a conversion runs with: the admin's configuration, or a user
    token's pool (its own models, then the shared ones; contract v1.15)."""

    config: MarkitaiConfig
    routable: bool
    user: bool = False
    hidden_hosts: tuple[str, ...] = ()


def _require_model(conversion: _Conversion, opts: JobOptions) -> None:
    """Refuse a request for model processing when no model can serve it.

    Before any job exists: the alternative is converting without the model
    the user asked for and reporting success.
    """
    if _requests_llm(opts, conversion.config) and not conversion.routable:
        raise ApiError(
            422,
            "llm_unavailable",
            "this request needs a model, but none is available; add a model in "
            "Settings → Models"
            if conversion.user
            else "this request needs a model, but none is available; add a model "
            "in Settings, or set MODEL and a provider API key",
        )


def _cloudflare_capability(cfg: MarkitaiConfig, *, trusted: bool) -> dict[str, Any]:
    """Whether this server can process a request with Cloudflare.

    An untrusted caller learns nothing: credentials are not even resolved.
    """
    from markitai.converter.cloudflare import CF_SUPPORTED_FORMATS
    from markitai.fetch_consent import _env_no_remote_fetch

    extensions = sorted({fmt.value for fmt in CF_SUPPORTED_FORMATS})
    if not trusted:
        return {
            "configured": False,
            "available": False,
            "reason": "client_not_trusted",
            "browser_rendering": False,
            "file_conversion": False,
            "file_extensions": [],
        }
    cloudflare = cfg.fetch.cloudflare
    configured = bool(
        cloudflare.get_resolved_api_token(strict=False)
        and cloudflare.get_resolved_account_id(strict=False)
    )
    reason: str | None
    if _env_no_remote_fetch() or cfg.fetch.remote_consent == "never":
        reason = "disabled_by_policy"
    else:
        reason = None if configured else "not_configured"
    available = reason is None
    return {
        "configured": configured,
        "available": available,
        "reason": reason,
        "browser_rendering": available,
        "file_conversion": available,
        "file_extensions": extensions,
    }


def _check_remote_processing(
    state: ServeState, opts: JobOptions, *, trusted: bool
) -> None:
    """Cloudflare runs only with this request's own, informed consent.

    A Cloudflare strategy or backend in the request needs
    ``remote_processing: "cloudflare"`` beside it, a trusted caller (it bills
    the server's account) and a configured, allowed Cloudflare. The consent
    is never stored, so a saved selection cannot convey it again.
    """
    selected = opts.strategy == "cloudflare" or opts.backend == "cloudflare"
    if not selected:
        if opts.remote_processing is not None:
            raise ApiError(
                422,
                "invalid_options",
                "remote_processing requires a Cloudflare strategy or backend in "
                "this request",
            )
        return
    if not trusted:
        raise ApiError(
            403,
            "remote_processing_forbidden",
            "Cloudflare processing requires a trusted authenticated request",
        )
    if opts.remote_processing != "cloudflare":
        raise ApiError(
            422,
            "remote_processing_confirmation_required",
            "Confirm Cloudflare processing for this request",
        )
    if opts.backend == "cloudflare" and opts.strategy in ("defuddle", "jina"):
        raise ApiError(
            422,
            "invalid_options",
            "Cloudflare confirmation cannot authorize another remote strategy; "
            "choose auto, static, playwright or cloudflare",
        )
    capability = _cloudflare_capability(state.config, trusted=True)
    if not capability["available"]:
        if capability["reason"] == "disabled_by_policy":
            raise ApiError(
                422,
                "remote_processing_disabled",
                "Cloudflare processing is disabled by server policy",
            )
        raise ApiError(
            422,
            "cloudflare_unavailable",
            "Cloudflare is not configured on this server",
        )


def _saved_options(opts: JobOptions) -> dict[str, Any]:
    """Options as a job remembers them: the selections, never the consent."""
    return opts.model_dump(exclude={"remote_processing"})


def _parse_label(raw: str | None) -> str | None:
    """A submission label: trimmed plain text of at most ``MAX_LABEL_CHARS``.

    A longer one, or one with a control character (a newline, a tab, ...),
    is refused; an empty one means no label.
    """
    if raw is None:
        return None
    import unicodedata

    label = raw.strip()
    if len(label) > MAX_LABEL_CHARS or any(
        unicodedata.category(char) == "Cc" for char in label
    ):
        raise ApiError(
            422,
            "invalid_label",
            f"label must be plain text of at most {MAX_LABEL_CHARS} characters",
        )
    return label or None


def _markdown_excerpt(data: bytes, max_bytes: int | None) -> tuple[str, bool]:
    """Decode a Markdown result, cut to *max_bytes* when that is given.

    The cut never splits a UTF-8 sequence and moves back to the last line
    end when one lies within the last quarter of the excerpt.
    """
    if max_bytes is None or len(data) <= max_bytes:
        return data.decode("utf-8", "replace"), False
    cut = max_bytes
    while cut > 0 and data[cut] & 0xC0 == 0x80:
        cut -= 1
    newline = data.rfind(b"\n", 0, cut)
    if newline >= 0 and newline + 1 >= cut - cut // 4:
        cut = newline + 1
    return data[:cut].decode("utf-8", "replace"), True


def _small_number(text: str, digits: int) -> bool:
    """ASCII digits only, at most *digits* of them (``"²".isdigit()`` is true)."""
    return re.fullmatch(rf"[0-9]{{1,{digits}}}", text) is not None


def _created_instant(value: object) -> float:
    """Sort key of a job's ``created_at``: its instant, oldest unknown first."""
    if isinstance(value, str) and value:
        try:
            from datetime import datetime

            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            pass
    return float("-inf")


# ---------------------------------------------------------------------------
# LLM settings helpers
# ---------------------------------------------------------------------------


def _mask_api_key(api_key: str | None) -> str | None:
    """Mask an api_key for display; never returns plaintext key material.

    ``env:VAR`` references pass through verbatim (the reference is not a
    secret and the value is never resolved here). Literal keys keep the
    first 2 + last 4 characters; short keys reveal at most the last 2 and
    never more than half.
    """
    if api_key is None or api_key == NO_KEY:
        return None
    if api_key.startswith("env:"):
        return api_key
    if len(api_key) >= 12:
        return f"{api_key[:2]}…{api_key[-4:]}"
    if len(api_key) >= 4:
        return f"…{api_key[-2:]}"
    return "…"


def _masked_base(api_base: str | None) -> str | None:
    return mask_url_credentials(api_base) if api_base is not None else None


def _shown_base(sent: str | None, stored: Any) -> bool:
    """Whether *sent* is the stored API base sent back as settings show it.

    Settings mask a base's userinfo and secret parameters (see
    ``mask_url_credentials``); that text sent back means "keep the stored
    base". Any other masked base cannot be stored: 422.
    """
    if sent is None or "…" not in sent:
        return False
    if isinstance(stored, str) and sent == mask_url_credentials(stored):
        return True
    raise ApiError(
        422,
        "invalid_settings_request",
        "api_base is masked; send the whole new address, or leave api_base out "
        "to keep the saved one.",
    )


_CONFIG_TEXT_LIMIT = 1024 * 1024
_CONFIG_BODY_LIMIT = 3 * 512 * 1024


def _config_too_large() -> ApiError:
    return ApiError(413, "config_too_large", "the configuration text exceeds 1 MiB")


def _read_config_bytes(path: Path) -> bytes | None:
    """The config file's bytes, or None when there is none."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.error("[Serve] Reading {} failed: {}", path, type(e).__name__)
        raise ApiError(
            500,
            "settings_io_failed",
            f"failed to read config file: {type(e).__name__}",
        ) from None


def _invalid_config(detail: str) -> ApiError:
    return ApiError(422, "invalid_config", detail)


def _validation_words(error: ValidationError) -> str:
    """A validator's refusal as ``loc: message`` parts (never the values)."""
    parts = [
        f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
        for item in error.errors()[:5]
    ]
    more = len(error.errors()) - len(parts)
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")


async def _config_body(request: Request) -> tuple[str, str]:
    """The ``{expected_revision, text}`` body of a config PUT (the admin's
    and a user's), within the size limits."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > _CONFIG_BODY_LIMIT:
            raise _config_too_large()
    shape = ApiError(
        422,
        "invalid_settings_request",
        'send a JSON body {"expected_revision": string, "text": string}',
    )
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise shape
    try:
        body = json.loads(bytes(data))
    except ValueError:
        raise shape from None
    if (
        not isinstance(body, dict)
        or set(body) != {"expected_revision", "text"}
        or not isinstance(body["expected_revision"], str)
        or not isinstance(body["text"], str)
    ):
        raise shape
    text: str = body["text"]
    if len(text.encode("utf-8", "surrogatepass")) > _CONFIG_TEXT_LIMIT:
        raise _config_too_large()
    return body["expected_revision"], text


#: A body of PATCH /api/me/models/shared/{id} at most
_SHARED_WEIGHT_BODY_LIMIT = 64 * 1024

#: The one sentence a failed test of a shared model says, whatever failed
SHARED_TEST_FAILED = "The shared model did not pass its test; ask the admin to check it"


async def _json_body(request: Request, limit: int) -> Any:
    """A JSON request body of at most *limit* bytes: 413 above, 422
    ``json_required`` for another content type, ``invalid_json`` when it
    does not parse."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise ApiError(413, "request_too_large", "the request is too large")
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError(422, "json_required", "send the body as application/json")
    try:
        return json.loads(bytes(data))
    except ValueError:
        raise ApiError(422, "invalid_json", "the request is not valid JSON") from None


def _replace_config_file(
    path: Path, expected_revision: str, text: str
) -> tuple[bytes, dict[str, Any] | None, dict[str, Any]]:
    """``PUT /api/settings/config``'s file work, under the settings lock.

    Returns the saved bytes, the old configuration (None when the stored file
    could not be read as a JSON object) and the new one.
    """
    from markitai.config import MarkitaiConfig
    from markitai.security import atomic_write_bytes

    original = _read_config_bytes(path)
    current = config_editor.revision(original)
    if expected_revision != current:
        raise HTTPException(
            status_code=409,
            detail={"code": "stale_revision", "current_revision": current},
        )
    try:
        parsed = config_editor.parse(text)
    except json.JSONDecodeError as e:
        raise _invalid_config(config_editor.position(e)) from None
    if not isinstance(parsed, dict):
        raise _invalid_config("the configuration must be a JSON object")
    current_text = config_editor.current_object_text(original)
    old: dict[str, Any] | None = (
        {}
        if original is None
        else json.loads(current_text)
        if current_text is not None
        else None
    )
    try:
        saved_text = config_editor.unmask(text, current_text)
    except ValueError as e:
        raise _invalid_config(str(e)) from None
    document = json.loads(saved_text)
    # Round 18: every weight the text writes is a whole number from 0 to 100;
    # one the file already held for the same entry may stay
    llm = document.get("llm")
    written = llm.get("model_list") if isinstance(llm, dict) else None
    if isinstance(written, list):
        old_llm = old.get("llm") if isinstance(old, dict) else None
        old_models = old_llm.get("model_list") if isinstance(old_llm, dict) else None
        user_models.check_weights(
            written,
            old_models if isinstance(old_models, list) else [],
            lambda index, entry: (
                f"/llm/model_list/{index} ({entry['litellm_params'].get('model', '?')} "
                f"in {entry.get('model_name', '?')})"
            ),
        )
    try:
        MarkitaiConfig.model_validate(document)
    except ValidationError as e:
        raise _invalid_config(_validation_words(e)) from None
    saved = saved_text.encode("utf-8", "surrogatepass")

    def unchanged_since_read() -> None:
        # The file may have been edited by hand since it was read
        now = _read_config_bytes(path)
        if now != original:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "config_changed",
                    "current_revision": config_editor.revision(now),
                },
            )

    try:
        # The configuration can hold literal API keys: keep it owner-only
        atomic_write_bytes(
            path, saved, private=True, before_replace=unchanged_since_read
        )
    except OSError as e:
        logger.error("[Serve] Writing {} failed: {}", path, type(e).__name__)
        raise ApiError(
            500,
            "settings_io_failed",
            f"failed to update config file: {type(e).__name__}",
        ) from None
    return saved, old, document


def _apply_config_models(state: ServeState, document: dict[str, Any]) -> None:
    """Run with the model list and connections of a saved configuration."""
    from markitai.config import MarkitaiConfig

    llm = MarkitaiConfig.model_validate(document).llm
    models = list(llm.model_list)
    providers = list(llm.providers)
    state.configured_models = models
    state.configured_providers = providers
    state.settings_revision = _document_settings_revision(document, models, providers)
    state.config.llm.providers = providers
    state.config.llm.model_list = _effective_models(models, state.detected_models)
    state.llm_source = (
        "config" if models else "detected" if state.detected_models else "none"
    )


def _open_in_editor(path: Path) -> None:
    """Hand *path* to the host's default application for its file type.

    Detached so the server never waits on a GUI editor. The OS default (not
    ``$EDITOR``, which is usually a terminal program with no terminal here)
    is what "open in my editor" means from a browser.
    """
    import shutil
    import subprocess

    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]  # nosec B606 - fixed path, no shell
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener) is None:
        raise RuntimeError(f"'{opener}' is not available on this host")
    subprocess.Popen(  # noqa: S603
        [opener, str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _display_config_path(path: Path) -> str:
    """Render a config path with the home directory collapsed to ``~``."""
    try:
        return f"~/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(path)


def _settings_revision(models: list[Any], providers: list[Any]) -> str:
    """Return a canonical revision for models and saved provider connections."""
    canonical = json.dumps(
        {"models": models, "providers": providers},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _user_revision(lists: user_models.Lists) -> str:
    """A user's revision: :func:`_settings_revision` while it overrides no
    shared model's weight (so revisions from before overrides stay valid),
    else the same digest over its models, connections and overrides
    (contract v1.16)."""
    if not lists.shared_weights:
        return _settings_revision(lists.models, lists.providers)
    canonical = json.dumps(
        {
            "models": lists.models,
            "providers": lists.providers,
            "shared_weights": lists.shared_weights,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _model_to_raw(model: ModelConfig) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True, exclude_defaults=True)


def _legacy_deployment_id(entry: Any, index: int) -> str:
    """Stable session token for an entry that predates ``model_info.id``."""
    if isinstance(entry, dict):
        routing_group = entry.get("model_name")
        params = entry.get("litellm_params")
        params = params if isinstance(params, dict) else {}
        visible = {
            "routing_group": routing_group,
            "model": params.get("model"),
            "api_base": params.get("api_base"),
            "weight": params.get("weight", 1),
            "index": index,
        }
    else:
        visible = {
            "routing_group": entry.model_name,
            "model": entry.litellm_params.model,
            "api_base": entry.litellm_params.api_base,
            "weight": entry.litellm_params.weight,
            "index": index,
        }
    digest = hashlib.sha256(
        json.dumps(visible, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"legacy-{digest}"


def _deployment_id(entry: Any, index: int) -> str:
    if isinstance(entry, dict):
        info = entry.get("model_info")
        candidate = info.get("id") if isinstance(info, dict) else None
    else:
        candidate = entry.model_info.id if entry.model_info is not None else None
    return (
        candidate
        if isinstance(candidate, str) and candidate
        else _legacy_deployment_id(entry, index)
    )


def _backfill_raw_deployment_ids(entries: list[Any]) -> dict[str, str]:
    """Assign UUIDs to legacy entries and return legacy-token → UUID."""
    mapped: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        info = entry.get("model_info")
        if not isinstance(info, dict):
            info = {}
            entry["model_info"] = info
        if not isinstance(info.get("id"), str) or not info["id"]:
            legacy_id = _legacy_deployment_id(entry, index)
            deployment_id = str(uuid.uuid4())
            info["id"] = deployment_id
            mapped[legacy_id] = deployment_id
    return mapped


def _sanitized_api_origin(api_base: str | None) -> str | None:
    """Return scheme + host + port only; never userinfo, path, query, fragment."""
    if not api_base:
        return None
    from markitai.config import resolve_env_value

    resolved = resolve_env_value(api_base, strict=False)
    if not resolved:
        return None
    try:
        parsed = urlsplit(resolved)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return None
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, "", "", ""))
    except ValueError:
        return None


def _deployment_view(
    model: ModelConfig, index: int, *, persisted: bool, share: bool = False
) -> dict[str, Any]:
    """One deployment as the settings show it; *share* adds whether the
    admin shares it (the configuration file's rows, contract v1.15)."""
    from markitai.providers import is_local_provider_model

    local = is_local_provider_model(model.litellm_params.model)
    shown: dict[str, Any] = {
        "deployment_id": _deployment_id(model, index),
        "routing_group": model.model_name,
        "model": model.litellm_params.model,
        "weight": model.litellm_params.weight,
        "api_key_configured": bool(_real_key(model.litellm_params.api_key))
        and not local,
        "api_base_configured": bool(model.litellm_params.api_base) and not local,
        "api_base": (
            None if local else _sanitized_api_origin(model.litellm_params.api_base)
        ),
        "persisted": persisted,
    }
    if share:
        shown["share"] = "shared" if user_models.is_shared(model) else "private"
    return shown


def _effective_models(
    configured: list[ModelConfig], detected: list[ModelConfig]
) -> list[ModelConfig]:
    """Configured deployments win; session-detected duplicates are omitted."""
    seen: set[tuple[str, str | None, str]] = set()
    effective: list[ModelConfig] = []
    for model in [*configured, *detected]:
        key = (
            model.litellm_params.model,
            model.litellm_params.api_base,
            model.model_name,
        )
        if key in seen:
            continue
        seen.add(key)
        effective.append(model)
    return effective


def _is_deployment_routable(model: ModelConfig) -> bool:
    if model.litellm_params.weight <= 0:
        return False
    from markitai.providers import (
        is_local_provider_available,
        is_local_provider_model,
    )
    from markitai.providers.connectable import connectable

    model_id = model.litellm_params.model
    if is_local_provider_model(model_id):
        return is_local_provider_available(model_id)
    provider = model_id.split("/", 1)[0].lower()
    row = connectable(provider)
    # Local servers take no key; Bedrock and Vertex AI sign with the
    # server's own cloud credentials
    if provider in {"ollama", "vllm"} or (row is not None and row.key_optional):
        return True
    if model.litellm_params.get_resolved_api_key(strict=False):
        return True
    return _environment_key_variable(provider) is not None


def _llm_settings_payload(state: ServeState) -> dict[str, Any]:
    effective = _effective_models(state.configured_models, state.detected_models)
    return {
        "configured": bool(state.configured_models),
        "routable": any(_is_deployment_routable(model) for model in effective),
        "source": state.llm_source,
        "config_path": _display_config_path(state.config_path),
        "config_origin": state.config_origin,
        "revision": state.settings_revision,
        "deployments": [
            _deployment_view(model, index, persisted=True, share=True)
            for index, model in enumerate(state.configured_models)
        ],
        "detected": [
            _deployment_view(model, index, persisted=False)
            for index, model in enumerate(state.detected_models)
        ],
        # Round 18: the batch route also takes `remove`
        "batch_remove": True,
    }


def _mutate_config_model_list(
    config_path: Path,
    mutate: Callable[[list[Any], list[Any], dict[str, str]], list[Any]],
    *,
    fallback: list[ModelConfig],
    fallback_providers: list[LLMProviderConfig],
    expected_revision: str | None,
    backfill_ids: bool,
) -> tuple[list[ModelConfig], list[LLMProviderConfig], str]:
    """Atomically mutate models and provider connections with revision checks."""
    from markitai.config import LLMProviderConfig, ModelConfig
    from markitai.security import atomic_write_json

    data: dict[str, Any] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("existing config file is not a JSON object")
        data = loaded

    llm = data.get("llm")
    if not isinstance(llm, dict):
        llm = {}
    model_list = llm.get("model_list")
    if not isinstance(model_list, list):
        model_list = [_model_to_raw(model) for model in fallback]
    provider_list = llm.get("providers")
    if not isinstance(provider_list, list):
        provider_list = [
            provider.model_dump(mode="json", exclude_none=True)
            for provider in fallback_providers
        ]

    current_revision = _settings_revision(model_list, provider_list)
    if expected_revision is not None and expected_revision != current_revision:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stale_revision",
                "current_revision": current_revision,
            },
        )

    legacy_mapping = _backfill_raw_deployment_ids(model_list) if backfill_ids else {}
    model_list = mutate(model_list, provider_list, legacy_mapping)
    llm["model_list"] = model_list
    if provider_list or "providers" in llm or fallback_providers:
        llm["providers"] = provider_list
    data["llm"] = llm

    merged = [ModelConfig.model_validate(model) for model in model_list]
    merged_providers = [
        LLMProviderConfig.model_validate(provider) for provider in provider_list
    ]
    revision = _settings_revision(model_list, provider_list)
    # The model list can hold literal API keys: keep the file owner-only
    atomic_write_json(config_path, data, private=True)
    return merged, merged_providers, revision


def _save_server_defaults(
    config_path: Path, running: MarkitaiConfig, expected: str, wanted: dict[str, Any]
) -> None:
    """Write the admin's conversion defaults into the configuration file,
    atomically, if *expected* is still the revision of the running defaults."""
    from markitai.security import atomic_write_json

    current = convert_defaults.revision(convert_defaults.of_config(running))
    if expected != current:
        raise HTTPException(
            status_code=409,
            detail={"code": "stale_revision", "current_revision": current},
        )
    data: dict[str, Any] = {}
    try:
        if config_path.is_file():
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("existing config file is not a JSON object")
            data = loaded
        convert_defaults.write_into_file(data, wanted)
        atomic_write_json(config_path, data, private=True)
    except (OSError, ValueError) as e:
        logger.error(
            "[Serve] Conversion defaults write failed for {}: {}",
            config_path,
            type(e).__name__,
        )
        raise ApiError(
            500,
            "settings_io_failed",
            f"failed to update config file: {type(e).__name__}",
        ) from None


def _find_raw_entries(entries: list[Any], model_name: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("model_name") == model_name
    ]


def _find_raw_deployment(
    entries: list[Any], deployment_id: str, legacy_mapping: dict[str, str]
) -> dict[str, Any] | None:
    target_id = legacy_mapping.get(deployment_id, deployment_id)
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and _deployment_id(entry, index) == target_id:
            return entry
    return None


def _raw_provider_by_id(
    providers: list[Any], provider_id: str
) -> dict[str, Any] | None:
    return next(
        (
            provider
            for provider in providers
            if isinstance(provider, dict) and provider.get("id") == provider_id
        ),
        None,
    )


def _raw_model_provider_id(entry: dict[str, Any]) -> str | None:
    info = entry.get("model_info")
    value = info.get("provider_id") if isinstance(info, dict) else None
    return value if isinstance(value, str) and value else None


def _link_raw_model_to_provider(entry: dict[str, Any], provider_id: str) -> None:
    info = entry.get("model_info")
    if not isinstance(info, dict):
        info = {}
        entry["model_info"] = info
    info["provider_id"] = provider_id


def _ensure_raw_provider(
    providers: list[Any],
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    models: list[Any] | None = None,
) -> str | None:
    """Persist one credential connection and link matching legacy models."""
    from markitai.providers import is_local_provider_model

    if is_local_provider_model(f"{provider}/model") or (
        api_key is None and api_base is None
    ):
        return None
    match = next(
        (
            candidate
            for candidate in providers
            if isinstance(candidate, dict)
            and candidate.get("provider") == provider
            and candidate.get("api_key") == api_key
            and candidate.get("api_base") == api_base
        ),
        None,
    )
    if match is None:
        match = {
            "id": str(uuid.uuid4()),
            "provider": provider,
            **({"api_key": api_key} if api_key is not None else {}),
            **({"api_base": api_base} if api_base is not None else {}),
        }
        providers.append(match)
    provider_id = str(match["id"])
    for entry in models or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params")
        if not isinstance(params, dict):
            continue
        model = params.get("model")
        if (
            isinstance(model, str)
            and model.split("/", 1)[0].lower() == provider
            and params.get("api_key") == api_key
            and params.get("api_base") == api_base
        ):
            _link_raw_model_to_provider(entry, provider_id)
    return provider_id


def _preserve_raw_deployment_provider(
    entry: dict[str, Any], entries: list[Any], providers: list[Any]
) -> None:
    """Migrate embedded credentials before a model deployment is removed."""
    if _raw_model_provider_id(entry) is not None:
        return
    params = entry.get("litellm_params")
    if not isinstance(params, dict):
        return
    model = params.get("model")
    if not isinstance(model, str):
        return
    _ensure_raw_provider(
        providers,
        provider=model.split("/", 1)[0].lower(),
        api_key=params.get("api_key")
        if isinstance(params.get("api_key"), str)
        else None,
        api_base=(
            params.get("api_base") if isinstance(params.get("api_base"), str) else None
        ),
        models=entries,
    )


def _strip_ignored_local_params(params: dict[str, Any]) -> None:
    """Drop api_key/api_base from *params* when the model is a local provider.

    Local providers (claude-agent/, copilot/, chatgpt/) authenticate via
    their CLI/OAuth session; the fields are accepted by the API but never
    persisted for them.
    """
    from markitai.providers import is_local_provider_model

    model = params.get("model")
    if isinstance(model, str) and is_local_provider_model(model):
        params.pop("api_key", None)
        params.pop("api_base", None)


def _scrub_secret(text: str, secret: str) -> str:
    """Remove a secret (and any >=6-char fragment of it) from *text*.

    Provider error messages may echo the key partially self-redacted (e.g.
    OpenAI's ``sk-12345***cdef``); scrubbing every long fragment guarantees
    no key material survives, at the cost of occasionally over-masking.
    """
    masked = _mask_api_key(secret) or "…"
    if len(secret) < 6:
        return text.replace(secret, masked)
    for length in range(len(secret), 5, -1):
        for start in range(len(secret) - length + 1):
            fragment = secret[start : start + length]
            if fragment in text:
                text = text.replace(fragment, masked)
    return text


def _sanitize_probe_error(exc: BaseException, body: LLMSettingsUpdate) -> str:
    """Build a terse failure without keys, auth headers, or credentialed URLs."""
    from markitai.config import resolve_env_value

    message = str(exc).strip()
    first_line = message.splitlines()[0] if message else ""
    detail = type(exc).__name__ + (f": {first_line}" if first_line else "")
    secrets = [
        value
        for value in (
            body.api_key,
            resolve_env_value(body.api_key, strict=False) if body.api_key else None,
        )
        if value and not value.startswith("env:")
    ]
    if body.api_base:
        base = resolve_env_value(body.api_base, strict=False) or body.api_base
        secrets += url_credential_values(base)
    for secret in secrets:
        detail = _scrub_secret(detail, secret)
    detail = re.sub(
        r"(?i)\b(?:authorization|x-api-key|api[_-]?key)\s*[:=]\s*\S+",
        "credential: …",
        detail,
    )
    detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer …", detail)
    detail = re.sub(
        r"https?://[^\s]+",
        lambda match: _sanitized_api_origin(match.group(0)) or "<url>",
        detail,
    )
    detail = re.sub(r"\bsk-[A-Za-z0-9._-]{4,}\b", "sk-…", detail)
    return detail[:300]


def _key_variables(provider: str) -> tuple[tuple[str, ...], str | None]:
    """The variables LiteLLM reads *provider*'s key from, and the main one.

    In LiteLLM's order (the first one set is the key a request carries);
    the main one is the variable markitai names when none is set. None for a
    cloud signed in with the server's own credentials.
    """
    from markitai.providers.discovery import provider_key_variables

    return provider_key_variables(provider)


def _environment_key_variable(provider: str) -> str | None:
    """The variable that serves *provider* a key on this server, if one is set.

    The key a conversion or a model test of a connection with no key of its
    own carries: LiteLLM reads it from the process environment, where the
    CLI loads ``./.env`` and ``~/.markitai/.env``.
    """
    from markitai.providers import is_local_provider_model

    if is_local_provider_model(f"{provider}/"):
        return None
    names, _main = _key_variables(provider)
    return next((name for name in names if os.environ.get(name, "").strip()), None)


def _probe_keyless(body: LLMSettingsUpdate) -> tuple[bool, str | None]:
    """Whether a test carries no real key, and the variable one would be in.

    Keyless means no key of its own, none in the provider's key variables
    and not a cloud signed in with the server's own credentials; the
    keyless placeholder counts as none, but names no variable (its entry
    reads none). The variable is the provider's main one, the one LiteLLM
    reads, for a model without a key of its own.
    """
    from markitai.providers.connectable import CONNECTABLE

    model = body.model or ""
    prefix = model.partition("/")[0].lower() if "/" in model else ""
    row = next((row for row in CONNECTABLE if row.provider == prefix), None)
    if row is not None and row.cloud:
        return False, None
    names, variable = _key_variables(prefix)
    key = body.api_key
    if key == NO_KEY:
        return True, None
    if key and key.startswith("env:"):
        return not os.environ.get(key[4:]), key[4:]
    if key:
        return False, None
    return not any(os.environ.get(name) for name in names), variable


def _probe_failure(exc: BaseException, body: LLMSettingsUpdate) -> dict[str, Any]:
    """The answer for a failed test: category and detail (contract v1.10)."""
    from markitai.llm.error_redaction import _LITELLM_HEAD

    raw = _sanitize_probe_error(exc, body)
    words = re.sub(r"\s+", " ", _LITELLM_HEAD.sub("", raw)).strip()
    keyless, variable = _probe_keyless(body)
    failure = model_test.classify(
        exc,
        model=body.model or "",
        provider_words=words,
        keyless=keyless,
        key_variable=variable,
    )
    logger.info(
        "[Serve] Model test of {} failed: {} ({})", body.model, failure.reason, raw
    )
    # An error with no status and no category keeps its own first line
    return failure.answer(None if failure.categorized else raw)


CredentialPair = tuple[str | None, str | None]

#: The key a connection or deployment saved through the settings API stores
#: when it has none but names an address. Without a
#: key LiteLLM reads the server's own key variable (OPENAI_API_KEY,
#: GROQ_API_KEY, LM_STUDIO_API_KEY, ...) and sends it to whatever address the
#: entry names, an address a settings caller can choose; with this one it
#: sends a word that is no credential. A server that checks no key (LM
#: Studio, vLLM, llama.cpp) takes it. Entries written by hand in the config
#: file are never given one: there a missing key still means "the
#: environment's". Never shown as a saved key.
NO_KEY = "markitai-no-key"


def _real_key(api_key: Any) -> Any:
    """*api_key*, or None when it is the :data:`NO_KEY` placeholder."""
    return None if api_key == NO_KEY else api_key


def _keyless_placeholder(api_key: Any, api_base: Any, model: Any) -> Any:
    """The key to store for an entry the settings API writes: *api_key*, or
    :data:`NO_KEY` when there is none while *api_base* names an address.

    Without an address the provider's own endpoint gets the environment's
    key, as configured by the server's admin; the subscription runtimes (claude-agent,
    copilot, chatgpt) take neither key nor address.
    """
    from markitai.providers import is_local_provider_model

    if api_key or not api_base:
        return api_key
    if isinstance(model, str) and is_local_provider_model(model):
        return api_key
    return NO_KEY


#: LiteLLM routes that sign in with the server's own cloud login or token
#: whatever key an entry carries (contract v1.9): Vertex AI with Google's
#: application default credentials, watsonx with WATSONX_TOKEN, GigaChat with
#: GIGACHAT_ACCESS_TOKEN. With an ``api_base`` that login goes to that
#: address, and :data:`NO_KEY` does not stop it (LiteLLM 1.104.2, checked with
#: fake credentials and a capture server). Bedrock is not here: given the
#: placeholder it sends it as its bearer key and never signs with the AWS
#: chain. Without an address these use the cloud's own endpoint, as
#: configured by the server's admin.
_AMBIENT_LOGIN: dict[str, tuple[str, str]] = {
    "vertex_ai": ("Vertex AI", "Google credentials"),
    "vertex_ai_beta": ("Vertex AI", "Google credentials"),
    "watsonx": ("watsonx", "WATSONX_TOKEN"),
    "watsonx_text": ("watsonx", "WATSONX_TOKEN"),
    "gigachat": ("GigaChat", "GIGACHAT_ACCESS_TOKEN"),
}
#: Prefixes LiteLLM must never be asked about: resolving them starts a
#: sign-in (chatgpt, github_copilot) or they are markitai's own runtimes.
_RUNTIME_PREFIXES = frozenset({"chatgpt", "github_copilot", "claude-agent", "copilot"})


def _ambient_login_provider(model: Any) -> str | None:
    """The :data:`_AMBIENT_LOGIN` route LiteLLM sends *model* through, if any.

    A bare name counts too: LiteLLM routes ``gemini-2.5-pro`` to Vertex AI.
    """
    if not isinstance(model, str) or not model.strip():
        return None
    prefix, slash, _rest = model.partition("/")
    if slash:
        prefix = prefix.strip().lower()
        if prefix in _AMBIENT_LOGIN:
            return prefix
        if prefix in _RUNTIME_PREFIXES:
            return None
    try:
        from markitai.utils.litellm_import import import_litellm

        import_litellm()
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        provider = get_llm_provider(model)[1]
    except Exception:  # a model LiteLLM cannot route sends nothing
        return None
    return provider if provider in _AMBIENT_LOGIN else None


def _refuse_ambient_login_address(model: Any, api_base: Any) -> None:
    """Refuse an entry the settings API writes or tests that would carry the
    server's own cloud login to an address (contract v1.9)."""
    if not api_base:
        return
    provider = _ambient_login_provider(model)
    if provider is None:
        return
    label, credential = _AMBIENT_LOGIN[provider]
    raise ApiError(
        422,
        "invalid_settings_request",
        (
            f"api_base: {label} models send the server's own {credential} to any "
            f"address, so here they can only use {label}'s own endpoint; set "
            "another address in the config file"
        ),
    )


def _endpoint_host(api_base: Any, provider: str) -> tuple[str, str] | None:
    """Scheme and host:port an entry's requests go to (its default when it
    names no address); None when unknown."""
    base = api_base if isinstance(api_base, str) and api_base else None
    base = base or provider_default_api_base(provider)
    if not base or base.startswith("env:"):
        return None
    parts = urlsplit(base)
    return parts.scheme.lower(), parts.netloc.rpartition("@")[2].lower()


def _refuse_keyless_move(
    old: CredentialPair, new: CredentialPair, provider: str
) -> None:
    """Refuse turning a connection with a key into a keyless one aimed at
    another host it names: its requests would leave with the placeholder (or,
    for an entry written by hand, the server's own key variable) for a server
    the admin never paired a key with. A new connection for that server says
    so. Dropping the address too (back to the provider's endpoint) is fine.
    """
    if not _real_key(old[0]) or _real_key(new[0]):
        return
    # Back on the provider's own endpoint, the environment's key goes to its
    # own provider, as for an entry that never named an address
    if not new[1]:
        return
    if _endpoint_host(new[1], provider) == _endpoint_host(old[1], provider):
        return
    raise ApiError(
        422,
        "invalid_settings_request",
        (
            "a connection with a key cannot become one without a key at another "
            "address; add a new connection for that server instead"
        ),
    )


def _configured_credential_pairs(state: _ModelLists) -> frozenset[CredentialPair]:
    """Every ``(api_key, api_base)`` pair already present in the saved config."""
    return frozenset(
        [
            (m.litellm_params.api_key, m.litellm_params.api_base)
            for m in state.configured_models
        ]
        + [(p.api_key, p.api_base) for p in state.configured_providers]
    )


def _check_credential_target(
    api_key: str | None,
    api_base: str | None,
    *,
    client_key: str | None,
    configured: frozenset[CredentialPair],
) -> None:
    """Refuse to send a credential the caller does not hold to a base it picks.

    The server resolves ``env:NAME`` from its own environment (``.env``
    included) and holds saved keys, so a settings caller that could pair
    either with any ``api_base`` could send host secrets to its own server.
    A key/base pair is accepted when the saved config already has it (the
    provider editor sends it back), when the caller sent the literal key
    itself, or for a provider key variable (the environment cards'
    ``env:OPENAI_API_KEY``, ...) used with the provider's default endpoint.
    """
    if (api_key, api_base) in configured:
        return
    if api_base is not None and api_base.startswith("env:"):
        raise ApiError(
            422,
            "invalid_settings_request",
            (
                "api_base: env: references from the web API must already be "
                "configured with this key; reference them in the config file instead"
            ),
        )
    if not api_key or api_key == NO_KEY:
        return
    if api_key.startswith("env:"):
        if api_base is None and api_key[4:] in _provider_key_variable_names():
            return
        detail = (
            "api_key: env: references from the web API are limited to the "
            "providers' own key variables (OPENAI_API_KEY, DEEPSEEK_API_KEY, "
            "...) with the provider's default api_base; to use another "
            "variable or endpoint, enter the key itself or set the pair up in "
            "the config file"
        )
    elif api_key == client_key:
        return
    else:
        detail = (
            "stored credentials only work with their saved api_base; to use "
            "a different endpoint, enter the key itself or change api_base "
            "in the config file"
        )
    raise ApiError(422, "invalid_settings_request", detail)


def _provider_key_variable_names() -> frozenset[str]:
    """Every variable a provider's key is read from (environment cards')."""
    from markitai.providers.connectable import CONNECTABLE

    return frozenset(
        name for row in CONNECTABLE for name in _key_variables(row.provider)[0]
    ) | frozenset(PROVIDER_API_KEY_ENV.values())


def _card_credential(provider: str, api_key: Any) -> dict[str, str]:
    """A configured card's ``credential``: how it authenticates from the
    environment, when it does (contract v1.11 section 8).

    ``env:VAR`` for a key that is an ``env:VAR`` reference (set or not), or,
    for a connection with no key at all (hand-written; never the keyless
    placeholder), the provider's key variable that holds a value, which
    LiteLLM sends for it. Nothing for a stored key.
    """
    if isinstance(api_key, str) and api_key:
        if api_key.startswith("env:") and len(api_key) > 4:
            return {"credential": api_key}
        return {}
    variable = _environment_key_variable(provider)
    return {"credential": f"env:{variable}"} if variable else {}


def _own_endpoint(provider: str, api_base: Any) -> bool:
    """Whether a connection's address is its provider's own endpoint."""
    if not isinstance(api_base, str) or not api_base:
        return True
    default = provider_default_api_base(provider)
    return default is not None and api_base.rstrip("/") == default.rstrip("/")


def _environment_key(
    reference: str, provider: str, *, api_key: str | None, api_base: str | None
) -> str:
    """The key reference behind an environment card's id (``env:openai``).

    Cards exist for the providers whose key variable the server reads
    (``env:openai`` -> ``env:OPENAI_API_KEY``; the first of its variables
    that is set, else its main one), never for any variable a caller names,
    and that key goes only to its own provider's endpoint: an address needs
    the caller's own key.
    """
    name = reference.removeprefix("env:")
    names, main = _key_variables(name)
    variable = _environment_key_variable(name) or (main if names else None)
    if variable is None:
        raise ApiError(
            422,
            "invalid_settings_request",
            (
                f"unknown environment provider {reference!r}; an environment "
                "connection names a provider whose API key the server reads "
                "from its environment"
            ),
        )
    if provider != name:
        raise ApiError(
            422,
            "invalid_settings_request",
            f"provider must match the environment connection ({name})",
        )
    if api_key is None and api_base is not None:
        raise ApiError(
            422,
            "invalid_settings_request",
            (
                f"the {variable} environment key only goes to {name}'s own "
                "endpoint; to use another address, enter the key itself"
            ),
        )
    return f"env:{variable}"


def _configured_cards(
    models: list[ModelConfig],
    saved_providers: list[LLMProviderConfig],
    local_connections: set[str],
    *,
    environment: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[str, bool]]]:
    """The ``configured`` provider cards of saved connections and of models
    with credentials of their own, each with whether it is aimed at its
    provider's own endpoint, in order (one provider, one saved row: the
    page folds an environment card into such a card). *environment* False
    (a user's cards) never names the server's key variables.
    """
    from markitai.providers import is_local_provider_model
    from markitai.providers.discovery import provider_label

    cards: list[dict[str, Any]] = []
    own_endpoints: list[tuple[str, bool]] = []
    for saved in saved_providers:
        model_count = sum(
            model.model_info is not None and model.model_info.provider_id == saved.id
            for model in models
        )
        cards.append(
            {
                "id": f"provider:{saved.id}",
                "provider_id": saved.id,
                "provider": saved.provider,
                "label": provider_label(saved.provider),
                "kind": "configured",
                "status": "ready"
                if saved.api_key is not None or saved.api_base is not None
                else "needs_credentials",
                "source": "config",
                "api_key_configured": bool(_real_key(saved.api_key)),
                "api_base_configured": bool(saved.api_base),
                "api_base": _sanitized_api_origin(
                    saved.api_base or provider_default_api_base(saved.provider)
                ),
                "model_count": model_count,
                "supports_discovery": True,
                **(
                    _card_credential(saved.provider, saved.api_key)
                    if environment
                    else {}
                ),
            }
        )
        own_endpoints.append(
            (saved.provider, _own_endpoint(saved.provider, saved.api_base))
        )

    seen_configured_connections: set[tuple[str, str | None, str | None]] = set()
    explicit_by_id = {provider.id for provider in saved_providers}
    explicit_connections = {
        (provider.provider, provider.api_base, provider.api_key)
        for provider in saved_providers
    }
    for index, model in enumerate(models):
        params = model.litellm_params
        provider = params.model.split("/", 1)[0].lower()
        connection = (provider, params.api_base, params.api_key)
        linked = model.model_info.provider_id if model.model_info else None
        if is_local_provider_model(params.model):
            continue
        if linked in explicit_by_id or connection in explicit_connections:
            continue
        if connection in seen_configured_connections:
            continue
        seen_configured_connections.add(connection)
        if provider in local_connections and params.api_base is None:
            continue
        deployment_id = _deployment_id(model, index)
        model_count = sum(
            (
                candidate.litellm_params.model.split("/", 1)[0].lower(),
                candidate.litellm_params.api_base,
                candidate.litellm_params.api_key,
            )
            == connection
            for candidate in models
        )
        cards.append(
            {
                "id": f"provider:legacy:{deployment_id}",
                "provider_id": f"legacy:{deployment_id}",
                "deployment_id": deployment_id,
                "provider": provider,
                "label": provider_label(provider),
                "kind": "configured",
                "status": "ready" if params.weight > 0 else "disabled",
                "source": "config",
                "api_key_configured": bool(_real_key(params.api_key)),
                "api_base_configured": bool(params.api_base),
                "api_base": _sanitized_api_origin(
                    params.api_base or provider_default_api_base(provider)
                ),
                "model_count": model_count,
                "supports_discovery": True,
                **(_card_credential(provider, params.api_key) if environment else {}),
            }
        )
        own_endpoints.append((provider, _own_endpoint(provider, params.api_base)))
    return cards, own_endpoints


def _user_provider_cards(
    scope: _ModelLists, routable: frozenset[str]
) -> list[dict[str, Any]]:
    """A user's provider cards (contract v1.15): the built-in cards a user
    may use (no subscription, no cloud login, no environment card) and its
    own ``configured`` cards; no card names a key variable, and a provider
    whose documented address is the server's own machine has none."""
    from markitai.providers import is_local_provider_model
    from markitai.providers.connectable import BUILT_IN, CONNECTABLE, installed
    from markitai.providers.discovery import provider_label
    from markitai.serve.provider_catalog import annotate_card, card_facts

    def user_base(provider: str) -> str | None:
        base = user_models.default_base(provider)
        return None if base is None or user_models.server_machine(base) else base

    taken = {provider.provider for provider in scope.configured_providers} | {
        model.litellm_params.model.split("/", 1)[0].lower()
        for model in scope.configured_models
    }
    cards: list[dict[str, Any]] = []
    for row in CONNECTABLE:
        name = row.provider
        if (
            name in user_models.SERVER_LOGIN
            or is_local_provider_model(f"{name}/")
            or not installed(row)
            or not (name in BUILT_IN or name in routable or name == "custom")
        ):
            continue
        if name not in ("openai", "custom") and name in taken:
            continue
        local = row.key_optional and not row.cloud and user_base(name) is not None
        cards.append(
            {
                "id": f"common:{name}",
                "provider": name,
                "label": provider_label(name),
                "kind": "common",
                "status": "unknown" if local else "needs_credentials",
                "source": "built_in",
                "supports_discovery": row.discovery != "manual",
            }
        )
    configured, _own = _configured_cards(
        scope.configured_models, scope.configured_providers, set(), environment=False
    )
    cards.extend(configured)
    for card in cards:
        provider = str(card["provider"])
        card_facts(annotate_card(card))
        card.pop("credential", None)
        card["key_variable"] = None
        card["default_base"] = user_base(provider)
        if provider in ("bedrock", "bedrock_mantle"):
            # A user's Bedrock connection needs its own key
            card["key_optional"] = False
    return cards


def _credentials_view(
    provider: str,
    api_key: str | None,
    api_base: str | None,
    *,
    environment: bool = True,
) -> dict[str, str | None]:
    """One connection's editable values, as the credentials route shows them.

    A connection that stores no key (not the keyless placeholder) is served
    by the server's own key variable when one is set: it shows as
    ``env:<VARIABLE>`` with ``api_key_source: environment`` (Rust's
    ``credentials``), the key its conversions and model tests carry, rather
    than "Not set". A user's connection (*environment* False) never shows
    the environment: its key is its own or none, ``api_key_source`` is null
    and a default address on the server's own machine is no placeholder.
    """
    shown = api_key
    source = None
    if not environment:
        placeholder = provider_default_api_base(provider)
        if placeholder is not None and user_models.server_machine(placeholder):
            placeholder = None
        masked = _mask_api_key(
            api_key if not (api_key or "").startswith("env:") else None
        )
        return {
            "api_key": masked,
            "api_key_masked": masked,
            "api_key_source": None,
            "api_base": _masked_base(api_base),
            "api_base_placeholder": placeholder,
        }
    if not api_key:
        variable = _environment_key_variable(provider)
        if variable is not None:
            shown, source = f"env:{variable}", "environment"
    masked = _mask_api_key(shown)
    view: dict[str, str | None] = {
        "api_key": masked,
        "api_key_masked": masked,
        "api_base": _masked_base(api_base),
        "api_base_placeholder": provider_default_api_base(provider),
    }
    if source is not None:
        view["api_key_source"] = source
    return view


def _reject_rebound_credentials(
    stored_key: str | None, stored_base: str | None, api_base: str | None
) -> None:
    """Refuse to send a stored key to an endpoint other than its saved one."""
    if _real_key(stored_key) and api_base is not None and api_base != stored_base:
        raise ApiError(
            422,
            "invalid_settings_request",
            (
                "stored credentials only work with their saved api_base; to use "
                "a different endpoint, enter the key itself or change api_base "
                "in the config file"
            ),
        )


def _fill_stored_credentials(
    state: _ModelLists, body: LLMSettingsUpdate
) -> LLMSettingsUpdate:
    """Fill a keyless probe body with the stored credentials for its model.

    A saved-row test posts only its deployment reference; resolve credentials
    from the running config exactly like a job run would. An explicit ``api_key``
    always wins, local providers never take one, and nothing is persisted.
    A stored key is only sent to its own saved ``api_base``.
    """
    from markitai.providers import is_local_provider_model

    if (
        body.api_key is not None
        or body.model is None
        or is_local_provider_model(body.model)
    ):
        return body
    stored = [
        m.litellm_params
        for m in _effective_models(state.configured_models, state.detected_models)
        if m.litellm_params.model == body.model
    ]
    if body.api_base is not None:
        # Two entries may share a model string; only a same-base one applies.
        same_base = [p for p in stored if p.api_base == body.api_base]
        if not same_base:
            for params in stored:
                _reject_rebound_credentials(
                    params.api_key, params.api_base, body.api_base
                )
        stored = same_base
    if not stored:
        return body
    params = stored[0]
    return body.model_copy(
        update={"api_key": params.api_key, "api_base": params.api_base}
    )


def _configured_deployment_by_id(
    state: _ModelLists, deployment_id: str
) -> ModelConfig | None:
    return next(
        (
            model
            for index, model in enumerate(state.configured_models)
            if _deployment_id(model, index) == deployment_id
        ),
        None,
    )


def _stored_probe_body(
    state: _ModelLists, *, deployment_id: str | None, model_name: str | None
) -> LLMSettingsUpdate:
    """Build an ad-hoc probe body from a persisted deployment reference."""
    candidates: list[ModelConfig]
    if deployment_id is not None:
        candidates = [
            model
            for models in (state.configured_models, state.detected_models)
            for index, model in enumerate(models)
            if _deployment_id(model, index) == deployment_id
        ]
        label = f"deployment_id '{deployment_id}'"
    else:
        candidates = [
            model for model in state.configured_models if model.model_name == model_name
        ]
        label = f"model_name '{model_name}'"
        if len(candidates) > 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ambiguous_legacy_model_name",
                    "model_name": model_name,
                },
            )
    if not candidates:
        raise ApiError(422, "invalid_settings_request", f"{label} not found")
    params = candidates[0].litellm_params
    return LLMSettingsUpdate(
        model=params.model,
        api_key=params.api_key,
        api_base=params.api_base,
    )


async def _probe_llm(body: LLMSettingsUpdate) -> str:
    """Run a minimal short completion against the given model.

    Mirrors the router's parameter mapping — ``env:VAR`` resolution and the
    directly-registered handlers for local providers (claude-agent/,
    copilot/, chatgpt/) — without building a full ``LLMProcessor``. Nothing
    is persisted.

    Returns:
        A short success detail string.

    Raises:
        Exception: Any config/auth/transport failure from the probe.
    """
    from markitai.config import LiteLLMParams
    from markitai.providers import (
        get_provider,
        is_local_provider_available,
        is_local_provider_model,
        register_providers,
    )

    if body.model is None:  # the endpoint resolves model_name references first
        raise ValueError("probe body carries no model")
    params = LiteLLMParams(
        model=body.model, api_key=body.api_key, api_base=body.api_base
    )
    api_key = params.get_resolved_api_key()
    api_base = params.get_resolved_api_base()

    if is_local_provider_model(body.model) and not is_local_provider_available(
        body.model
    ):
        provider = body.model.split("/", 1)[0]
        extra = "claude-agent" if provider == "claude-agent" else provider
        raise RuntimeError(
            f"{provider} runtime support is missing from the Python environment "
            "running Markitai (CLI login alone is not the SDK). Run setup again "
            f"or install markitai[{extra}]; in this repository run: "
            f"uv sync --extra serve --extra {extra}"
        )

    register_providers()
    kwargs: dict[str, Any] = {
        "model": body.model,
        # One token is too small for APIs that account for hidden reasoning
        # tokens and can turn a healthy model into an artificial length error.
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 16,
        "timeout": _LLM_TEST_TIMEOUT_S,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    handler = get_provider(body.model.split("/", 1)[0]) if "/" in body.model else None
    if handler is not None:
        await handler.acompletion(**kwargs)
    else:
        import litellm

        await litellm.acompletion(**kwargs)
    return f"{body.model} responded"


def _add_job_outputs_to_zip(
    archive: zipfile.ZipFile, job: Job, *, prefix: Path | None = None
) -> None:
    """Add one job's immutable output tree to an open ZIP archive."""
    out_dir = job.out_dir.resolve()
    for file in sorted(out_dir.rglob("*")):
        if not file.is_file() or (
            file.name.startswith(".") and file.name.endswith(".tmp")
        ):
            continue
        relative = file.relative_to(out_dir)
        try:
            archive.write(file, relative if prefix is None else prefix / relative)
        except OSError:
            # A concurrent delete-item / rerun can unlink a file between the
            # rglob and this write; skip it rather than failing the archive.
            continue


def _job_archive_folder(job: Job) -> str:
    """Human-readable, filesystem-safe folder for a multi-job archive."""
    output = next((item.output for item in job.items if item.output), None)
    if output is None:
        return f"job-{job.job_id}"
    return _sanitize_upload_name(split_output_name(Path(output).name))


#: The images an item's asset route serves, by suffix (contract v1.6 6.2):
#: rasters only, never SVG or any other file a browser could run or render
_ITEM_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
}


def _item_result_files(
    job: Job, item: JobItem
) -> tuple[Path, str, Path, Path, str, Path] | None:
    """Where a finished item's result lives, or None when it has none.

    Returns the resolved out dir, the output base name, the ``.md`` and
    ``.llm.md`` paths, the selected variant and its Markdown file. The
    result and asset routes both read it, so an image is served exactly
    when the result lists it.
    """
    if item.status != "done" or item.output is None:
        return None
    out_dir = job.out_dir.resolve()
    output_path = (out_dir / item.output).resolve()
    if not output_path.is_relative_to(out_dir):
        return None
    # The item's known base, not split_output_name(output): an upload
    # named "notes.llm" writes notes.llm.md, which stripping would read
    # as sibling "notes"'s enhanced output and serve notes.md instead.
    base_name = item_base_name(item) or split_output_name(output_path.name)
    base_path, llm_path = markdown_pair(output_path.parent, base_name)
    # Honor the item's selected variant rather than probing llm-first: a
    # non-LLM retry after an earlier enhance leaves a stale .llm.md on disk
    # (pruned on success, but this stays correct even if it lingers).
    if item.llm_enhanced and llm_path.is_file():
        return out_dir, base_name, base_path, llm_path, "llm", llm_path
    if base_path.is_file():
        return out_dir, base_name, base_path, llm_path, "base", base_path
    if llm_path.is_file():
        return out_dir, base_name, base_path, llm_path, "llm", llm_path
    return None


def _item_assets(
    item: JobItem, out_dir: Path, base_name: str, base_path: Path, llm_path: Path
) -> list[Path]:
    """The asset files a finished item's result lists.

    Same ownership rule item deletion uses (serve.artifacts): the asset
    prefix is the output base name, and references tie URL screenshots
    (named after the URL) to their item.
    """
    return item_asset_files(
        out_dir,
        base_name,
        (base_path, llm_path) if item.llm_enhanced else (base_path,),
    )


def _item_artifacts(
    item: JobItem, out_dir: Path, base_name: str, base_path: Path, llm_path: Path
) -> list[Path]:
    """Every file a finished item's result lists in ``artifacts``, in order.

    The ``.md``, the ``.llm.md`` only when the item is enhanced (a stale
    sibling never makes the UI offer a spurious base/LLM diff), then its
    assets. The result and the item archive both list exactly these.
    """
    files: list[Path] = []
    candidates = [base_path, *([llm_path] if item.llm_enhanced else [])]
    candidates += _item_assets(item, out_dir, base_name, base_path, llm_path)
    for path in candidates:
        if path.is_file() and path not in files:
            files.append(path)
    return files


def _etag_matches(header: str | None, etag: str) -> bool:
    """``If-None-Match`` against *etag* by weak comparison (lists, ``*``)."""
    ours = etag.removeprefix("W/")
    return any(
        tag == "*" or tag.removeprefix("W/") == ours
        for tag in (part.strip() for part in (header or "").split(","))
    )


def _item_claimed_files(out_dir: Path, item: JobItem) -> set[Path]:
    """Markdown outputs and asset files of *item* inside *out_dir*."""
    base_name = item_base_name(item)
    if base_name is None:
        return set()
    pair = markdown_pair(out_dir, base_name)
    claimed = {path for path in pair if path.is_file()}
    claimed.update(item_asset_files(out_dir, base_name, pair))
    return claimed


def _item_owned_files(job: Job, item: JobItem) -> list[Path]:
    """Files that deleting *item* removes, never a sibling's.

    The item's outputs and assets (``serve.artifacts``) minus anything a
    remaining item also claims, plus its original upload. Rehydrated meta is
    untrusted on-disk state, so every path stays inside the job directory.
    """
    out_dir = job.out_dir.resolve()
    claimed = _item_claimed_files(out_dir, item)
    if claimed:
        for sibling in job.items:
            if sibling is not item:
                claimed -= _item_claimed_files(out_dir, sibling)
    owned = [path for path in claimed if path.resolve().is_relative_to(out_dir)]
    if item.kind == "file" and item.name:
        uploads_dir = job.uploads_dir.resolve()
        upload = (uploads_dir / item.name).resolve()
        if upload.is_relative_to(uploads_dir) and upload != uploads_dir:
            owned.append(upload)
    # Its kept page, capture record, block boxes and compare screenshots
    owned.extend(kept_capture_files(job, item))
    return sorted(owned)


def _unlink_all(paths: Sequence[Path]) -> None:
    """Delete *paths*, tolerating ones already gone (best effort)."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("[Serve] Could not delete {}: {}", path, e)


def request_shutdown(app: Any) -> None:
    """Tell a running app its server has begun shutting down.

    Ends every open SSE stream so the server's wait for open responses
    finishes and the lifespan shutdown (which cancels running jobs and
    persists them to history) runs promptly. Safe to call from a signal
    handler or another thread; a no-op before startup or after shutdown.
    """
    state: ServeState | None = getattr(getattr(app, "state", None), "markitai", None)
    loop = state.loop if state is not None else None
    if state is None or loop is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(state.registry.begin_shutdown)
    except RuntimeError:  # the loop closed in between
        pass


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _config_origin(
    path: Path, *, explicit: bool
) -> Literal["explicit", "environment", "project", "user", "default"]:
    if explicit:
        return "explicit"
    env_path = os.environ.get("MARKITAI_CONFIG")
    if env_path and path == Path(env_path).expanduser():
        return "environment"
    if path == Path.cwd() / "markitai.json":
        return "project"
    if path == DEFAULT_CONFIG_PATH:
        return "user" if path.exists() else "default"
    return "default"


def _initial_settings_revision(
    path: Path,
    models: list[ModelConfig],
    providers: list[LLMProviderConfig],
) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = None
    return _document_settings_revision(data, models, providers)


def _document_settings_revision(
    data: object,
    models: list[ModelConfig],
    providers: list[LLMProviderConfig],
) -> str:
    """The settings revision of a config file's parsed *data*.

    Its own model list and connections when it has a model list, else
    *models* and *providers* (what the server runs with).
    """
    llm = data.get("llm") if isinstance(data, dict) else None
    entries = llm.get("model_list") if isinstance(llm, dict) else None
    raw_providers = llm.get("providers") if isinstance(llm, dict) else None
    if isinstance(entries, list):
        return _settings_revision(
            entries, raw_providers if isinstance(raw_providers, list) else []
        )
    return _settings_revision(
        [_model_to_raw(model) for model in models],
        [provider.model_dump(mode="json", exclude_none=True) for provider in providers],
    )


class _CompressionMiddleware:
    """Gzip responses, except event streams and file downloads.

    The API's JSON and Markdown results and the UI's scripts compress well,
    which matters on a LAN (``--host``). Streams must reach the browser
    event by event, and downloads are served as files (archives are already
    compressed); those paths bypass the compressor whatever Starlette's own
    exclusions are in the installed version.
    """

    _BYPASS_SUFFIXES = ("/events", "/archive", "/original/file")

    def __init__(self, app: Any) -> None:
        from starlette.middleware.gzip import GZipMiddleware

        self._app = app
        self._gzip = GZipMiddleware(app, minimum_size=1024, compresslevel=6)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if (
            scope["type"] != "http"
            or path.endswith(self._BYPASS_SUFFIXES)
            or "/files/" in path
            or "/original/pages/" in path
            or path.startswith("/ui/")
        ):
            await self._app(scope, receive, send)
            return
        await self._gzip(scope, receive, send)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    static_dir: Path | None = None,
    jobs_root: Path | None = None,
    config: MarkitaiConfig | None = None,
    configure_logging: bool = True,
    config_path: Path | None = None,
    allowed_hosts: Sequence[str] | None = None,
    token: str | None = None,
    network_origin: Callable[[], str | None] | None = None,
    token_variable: str | None = None,
    access_path: Path | None = None,
) -> FastAPI:
    """Create the markitai serve FastAPI application.

    Args:
        static_dir: Directory holding a workbench bundle (``manifest.json``
            and ``ui/``). None serves the bundle vendored in the package
            (``serve/workbench/``); a directory without a valid bundle serves
            only the API, with a JSON hint at ``/``.
        jobs_root: Directory holding per-job workspaces. Defaults to
            ``~/.markitai/serve/jobs``.
        config: Base configuration. When None it is loaded via ConfigManager
            (respecting ``~/.markitai/config.json``) and LLM models are
            backfilled once via provider auto-detection.
        configure_logging: Whether lifespan startup takes over loguru
            (disable in embedding tests to keep global state untouched).
        config_path: Config file the ``/api/settings/llm/models`` endpoints
            write to. Defaults to ``~/.markitai/config.json``.
        allowed_hosts: Extra hostnames accepted in the Host and Origin
            headers besides localhost and IP literals (DNS-rebinding and
            CSRF protection).
        token: Access token required from non-loopback peers on ``/api/``
            routes, and from every peer on the LLM settings routes; a
            token-bearing remote peer is trusted like a loopback one. None
            disables token auth: remote peers may then only submit public
            URLs, and only loopback peers reach the settings routes.
        network_origin: Returns ``http://<network address>:<port>`` when
            the server listens on all interfaces and has such an address,
            else None; a regenerated token's sign-in link is built on it.
        token_variable: The environment variable that set *token* at
            startup (``MARKITAI_SERVE_TOKEN``), or None when it was generated.
        access_path: Where the user tokens' digests are kept. Defaults to
            ``access.json`` beside the jobs root (``~/.markitai/serve/``).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if configure_logging:
            logger.remove()
            # Also when the app is created outside the CLI: mask credentials,
            # and keep variable values (keys, document text) out of tracebacks
            logger.configure(patcher=redact_log_record)
            logger.add(sys.stderr, level="INFO", diagnose=False)
        # Before the prewarm thread imports LiteLLM while requests are logged
        from markitai.utils.litellm_import import prepare_litellm_import

        prepare_litellm_import()

        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        load_dotenv(Path.home() / ".markitai" / ".env")

        from markitai import fetch

        # A server must never prompt on stdin for remote-fetch consent.
        fetch.set_remote_consent_prompt_allowed(False)

        llm_source: Literal["config", "detected", "none"]
        if config is not None:
            cfg = config
            configured_models = list(cfg.llm.model_list)
            detected_models: list[ModelConfig] = []
            llm_source = "config" if configured_models else "none"
            resolved_config_path = (
                config_path if config_path is not None else DEFAULT_CONFIG_PATH
            )
        else:
            from markitai.config import ConfigManager

            manager = ConfigManager()
            cfg = manager.load()
            configured_models = list(cfg.llm.model_list)
            detected_models = []
            if configured_models:
                llm_source = "config"
            else:
                await _backfill_llm_models(cfg)
                detected_models = list(cfg.llm.model_list)
                llm_source = "detected" if detected_models else "none"
            env_path = os.environ.get("MARKITAI_CONFIG")
            resolved_config_path = (
                config_path
                if config_path is not None
                else manager.config_path
                or (Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH)
            )

        configured_providers = list(cfg.llm.providers)
        cfg.llm.model_list = _effective_models(configured_models, detected_models)
        root = (jobs_root if jobs_root is not None else DEFAULT_JOBS_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        removed = cleanup_stale_jobs(root)
        if removed:
            logger.info("[Serve] Cleaned up {} stale job dir(s)", removed)

        state = ServeState(
            config=cfg,
            registry=JobRegistry(root),
            llm_source=llm_source,
            config_path=resolved_config_path,
            config_origin=_config_origin(
                resolved_config_path, explicit=config_path is not None
            ),
            configured_models=configured_models,
            configured_providers=configured_providers,
            detected_models=detected_models,
            settings_revision=_initial_settings_revision(
                resolved_config_path, configured_models, configured_providers
            ),
        )
        state.loop = asyncio.get_running_loop()
        app.state.markitai = state
        # Models of tokens revoked while a deletion was interrupted
        await asyncio.to_thread(users.sweep, access.exists)
        restored = rehydrate_jobs(state.registry, cfg)
        if restored:
            logger.info("[Serve] Rehydrated {} archived job(s)", restored)
        # Startup cleanup alone never expires history on a long-running server
        expiry_task = asyncio.create_task(run_history_expiry(state.registry))
        logger.info("[Serve] markitai {} ready (jobs root: {})", __version__, root)
        if cfg.llm.model_list or detected_models:
            # LiteLLM loads in the background, not on the loop when the first
            # LLM job (or settings test) needs it
            from markitai.utils.prewarm import prewarm_litellm

            prewarm_litellm()
        try:
            yield
        finally:
            expiry_task.cancel()
            with suppress(asyncio.CancelledError):
                await expiry_task
            app.state.pdf_workers.close()
            await app.state.markitai.registry.shutdown()

            from markitai.converter.pdf_parallel import shutdown_pool
            from markitai.fetch import close_shared_clients
            from markitai.utils.executor import shutdown_converter_executor

            await close_shared_clients()
            # The PDF workers stop first, as on the CLI's Ctrl-C: the job
            # tasks are cancelled by now, but a converter thread waiting on
            # the workers would otherwise finish its whole document before
            # shutdown_converter_executor (which waits for it) returns. Also
            # before uvicorn re-raises the stop signal, which ends the process
            # without running atexit or finalize_process.
            shutdown_pool()
            shutdown_converter_executor()
            # Only a LiteLLM that was loaded has clients to close; importing
            # it here just to clean up would hold shutdown for most of a second
            if "litellm" in sys.modules:
                try:
                    from litellm.llms.custom_httpx.async_client_cleanup import (
                        close_litellm_async_clients,
                    )

                    await close_litellm_async_clients()
                except Exception as e:
                    logger.debug("[Serve] LiteLLM client cleanup failed: {}", e)

    app = FastAPI(title="markitai serve", version=__version__, lifespan=lifespan)
    # Mutable: POST /api/access-token/regenerate replaces it, and every
    # guard reads it per request. Event streams opened under an older
    # generation end when it changes.
    app.state.serve_token = token
    app.state.token_generation = 0
    # Digests of the tokens regenerated away: refused even from loopback
    app.state.revoked_tokens = set()
    access = AccessTokens(
        access_path
        if access_path is not None
        else (jobs_root if jobs_root is not None else DEFAULT_JOBS_ROOT).parent
        / "access.json"
    )
    app.state.access_tokens = access
    # A user token's own models, beside access.json (contract v1.15)
    users = user_models.UserModels(access.path.parent / "user-models")
    app.state.user_models = users
    app.add_middleware(_BodyLimitMiddleware)
    app.add_middleware(
        _HostGuardMiddleware,
        allowed_hosts=frozenset(
            hostname
            for hostname in map(_normalize_allowed_host, allowed_hosts or ())
            if hostname
        ),
    )

    @app.exception_handler(Exception)
    async def _unhandled_error_with_code(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Give an unhandled 500 the same error body as every other error.

        Starlette's default handler returns a plain-text body, which would be
        the one reply a client cannot branch on. The detail stays generic on
        purpose: an internal message belongs in the server log.
        """
        logger.exception(
            "[Serve] Unhandled error on {} {}", request.method, request.url.path
        )
        return error_response(500, "internal_error", "internal server error")

    @app.exception_handler(RequestValidationError)
    async def _validation_error_with_code(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """A request body or parameter the route's schema refused.

        ``detail`` is one sentence naming the fields (never their submitted
        values); ``reason`` says which kind of request it was.
        """
        errors = list(exc.errors())
        path = request.url.path
        if any(error.get("type") == "json_invalid" for error in errors):
            reason = "invalid_json"
        elif path.startswith(("/api/settings/llm", "/api/me/models")) and any(
            "weight" in error.get("loc", ()) for error in errors
        ):
            # Round 18: one reason for every weight a route refuses
            return error_response(422, "invalid_weight", user_models.WEIGHT_RULE)
        elif path.startswith(("/api/settings/llm", "/api/me/models")):
            reason = "invalid_settings_request"
        elif path.endswith("/retry"):
            reason = "invalid_retry_body"
        elif path == "/api/download-tickets":
            reason = "invalid_ticket_path"
        elif "/original/pages/" in path:
            reason = "invalid_page_request"
        else:
            reason = "invalid_request"
        return error_response(422, reason, validation_detail(errors))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_with_code(
        request: Request, exc: StarletteHTTPException
    ) -> Response:
        """Answer every refusal with ``{detail, code, reason}``.

        ``detail`` stays the human sentence; ``code`` follows the status and
        ``reason`` is the stable cause clients translate. A browser asking
        for a page that does not exist gets the workbench instead, still 404.
        """
        workbench = getattr(request.app.state, "workbench", None)
        if exc.status_code == 404 and workbench is not None:
            page = missing_page(workbench, request)
            if page is not None:
                return page
        if exc.status_code == 400 and _ran_out_of_open_files(exc):
            # Not a malformed body: the server lacked the file descriptors
            logger.warning(
                "[Serve] Upload to {} failed: too many open files", request.url.path
            )
            return error_response(503, "unavailable", _OPEN_FILES_EXHAUSTED_DETAIL)
        if isinstance(exc, ApiError):
            reason = exc.reason
        elif exc.status_code == 400 and request.url.path == "/api/jobs":
            # FastAPI's "There was an error parsing the body"
            reason = "invalid_multipart"
        else:
            reason = fallback_reason(exc.status_code, exc.detail)
        detail = exc.detail
        if exc.status_code == 404 and reason == "route_not_found":
            detail = "route not found"
        elif exc.status_code == 405:
            detail = "method not allowed"
        return error_response(
            exc.status_code, reason, detail, headers=dict(exc.headers or {})
        )

    @app.middleware("http")
    async def protect_local_settings(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if _admin_path(path):
            # These routes read and write provider keys and the access
            # tokens, so with token auth on they need the admin token from
            # every peer: any local process, or a page on another localhost
            # origin, is a loopback peer too. (A user token never gets here:
            # the token middleware refuses it with 403.)
            serve_token = request.app.state.serve_token
            if serve_token is not None and not _scope_presents_token(
                request.scope, serve_token
            ):
                return error_response(
                    401,
                    "token_required",
                    _AUTH_REQUIRED_DETAIL,
                    headers={"Cache-Control": "no-store"},
                )
            if not _is_trusted_request(request):
                return error_response(
                    403,
                    "settings_forbidden",
                    _loopback_only_detail(path),
                    headers={"Cache-Control": "no-store"},
                )
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        if path.startswith(("/api/me/", "/api/convert-defaults")):
            # A user's own models and their keys' masks: never cached
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        return await call_next(request)

    # Added after the middlewares above so it runs outermost: a remote peer
    # without the token gets one uniform 401 before any other guard answers.
    tickets = DownloadTickets()
    app.state.download_tickets = tickets
    app.state.page_renders = asyncio.Semaphore(_PAGE_RENDER_SLOTS)
    # PyMuPDF holds the GIL: Compare's PDF work runs in worker processes
    app.state.pdf_workers = PdfWorkers(workers=_PAGE_RENDER_SLOTS)
    app.state.page_cache = ByteCache()
    app.state.original_cache = {}
    app.add_middleware(
        _TokenAuthMiddleware,
        token_source=lambda: app.state.serve_token,
        tickets=tickets,
        revoked=app.state.revoked_tokens,
        access=access,
        registry_source=lambda: (
            app.state.markitai.registry if hasattr(app.state, "markitai") else None
        ),
    )
    # Outermost: compresses whatever the inner layers answer
    app.add_middleware(_CompressionMiddleware)

    # ---------------------------------------------------------- conversions

    def conversion_for(state: ServeState, owner: str | None) -> _Conversion:
        """What a conversion of *owner* (a user token id; None: the admin)
        runs with. Blocking for a user: its file may be read."""
        if owner is None:
            return _Conversion(state.config, _llm_routable(state))
        pool = user_models.build_pool(
            state.config,
            users.lists(owner),
            state.configured_models,
            state.configured_providers,
        )
        return _Conversion(
            pool.config,
            any(_is_deployment_routable(model) for model in pool.models),
            user=True,
            hidden_hosts=pool.hidden_hosts,
        )

    async def conversion_of(state: ServeState, owner: str | None) -> _Conversion:
        if owner is None:
            return conversion_for(state, None)
        return await asyncio.to_thread(conversion_for, state, owner)

    def job_config(conversion: _Conversion, opts: JobOptions) -> MarkitaiConfig:
        cfg = _build_job_config(conversion.config, opts)
        if conversion.user and not cfg.llm.model_list:
            # No pool: no model processing of any kind, whatever was asked
            cfg.llm.enabled = False
            cfg.image.alt_enabled = False
            cfg.image.desc_enabled = False
        return cfg

    # ------------------------------------------------------------------ API

    @app.get("/api/capabilities", response_model=Capabilities)
    async def get_capabilities(request: Request) -> dict[str, Any]:
        from markitai.config import PRESET_NAMES, get_preset

        state = _state(request)
        caller = _caller(request)
        if caller.user_id is not None:
            # A user's conversions route over its pool only (contract v1.15)
            conversion = await conversion_of(state, caller.user_id)
            models = [m.litellm_params.model for m in conversion.config.llm.model_list]
            llm = {
                "configured": bool(models),
                "routable": conversion.routable,
                "effective": conversion.routable,
                "models": models,
            }
        else:
            effective = _effective_models(
                state.configured_models, state.detected_models
            )
            routable = any(_is_deployment_routable(model) for model in effective)
            llm = {
                "configured": bool(state.configured_models),
                "routable": routable,
                "effective": routable,
                "models": [model.litellm_params.model for model in effective],
            }
        return {
            "version": __version__,
            # The first call may run git (a source checkout): off the loop
            "build": await asyncio.to_thread(build_identity),
            "implementation": "python",
            "docs_url": DOCS_URL,
            "source_url": SOURCE_URL,
            "llm": llm,
            "presets": list(PRESET_NAMES),
            "preset_options": {
                name: preset.model_dump()
                for name in PRESET_NAMES
                if (preset := get_preset(name, state.config)) is not None
            },
            "extras": {
                "browser": find_spec("playwright") is not None,
                "svg": find_spec("cairosvg") is not None,
            },
            # The web UI reads server-owned limits from here instead of
            # duplicating the constants (contract: schemas.CapabilitiesLimits).
            "limits": {
                "max_job_items": MAX_JOB_ITEMS,
                "history_retention_days": round(JOB_TTL_HOURS / 24),
            },
            "features": {
                "cancel": True,
                "download_tickets": True,
                "url_lists": True,
                "history_items": True,
                "original": True,
                "result_excerpt": True,
                "job_labels": True,
                "item_assets": True,
                "item_archive": True,
                "token_regenerate": request.app.state.serve_token is not None,
                "user_tokens": request.app.state.serve_token is not None,
                "config_edit": True,
                "batch_delete": True,
                "user_models": request.app.state.serve_token is not None,
                "user_config": request.app.state.serve_token is not None,
                "convert_defaults": True,
                "pdf_outline": True,
            },
            # Anonymous callers may not use the settings, so they are users
            "access": (
                {"role": "admin"}
                if caller.role == "admin"
                else {"role": "user", "label": caller.label}
            ),
            "remote_services": {
                "cloudflare": _cloudflare_capability(
                    state.config, trusted=_is_trusted_request(request)
                )
            },
        }

    @app.get(
        "/api/settings/llm",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def get_llm_settings(request: Request) -> dict[str, Any]:
        return _llm_settings_payload(_state(request))

    @app.get("/api/settings/llm/detected", response_model=list[DetectedModel])
    async def get_detected_providers() -> list[dict[str, Any]]:
        """Legacy quick-add candidates; provider cards use the v2 endpoint."""
        return await _detect_provider_candidates()

    @app.get(
        "/api/settings/llm/providers",
        response_model=ProviderConnectionList,
        # Card keys legitimately vary per kind: absent optional keys must stay
        # absent rather than materialize as nulls.
        response_model_exclude_unset=True,
    )
    async def get_llm_providers(
        request: Request, refresh: bool = False
    ) -> dict[str, Any]:
        from markitai.providers import is_local_provider_model
        from markitai.providers.connectable import BUILT_IN, routable_providers
        from markitai.providers.discovery import (
            detect_provider_connections,
        )
        from markitai.serve.provider_catalog import annotate_card, card_facts

        state = _state(request)
        detected = await detect_provider_connections(
            state.configured_models, refresh=refresh
        )
        # A built-in card beyond those markitai always offered is shown when
        # this LiteLLM routes its prefix (LiteLLM loads on a worker thread,
        # as for a job); subscription runtimes are routed by markitai itself
        routable = await asyncio.to_thread(routable_providers)
        detected = [
            card
            for card in detected
            if card.get("kind") != "configured"
            and (
                card.get("kind") not in ("common", "environment")
                or card.get("provider") in BUILT_IN
                or is_local_provider_model(f"{card.get('provider')}/")
                or card.get("provider") in routable
            )
        ]
        explicit_by_id = {
            provider.id: provider for provider in state.configured_providers
        }
        explicit_provider_names = {
            provider.provider for provider in explicit_by_id.values()
        }
        providers = [
            card
            for card in detected
            if not (
                card.get("kind") == "common"
                and card.get("provider") in explicit_provider_names
            )
        ]
        local_connections = {
            str(card["provider"])
            for card in providers
            if card.get("kind") in {"local_cli", "oauth"}
        }

        configured_cards, own_endpoints = _configured_cards(
            state.configured_models, state.configured_providers, local_connections
        )
        providers.extend(configured_cards)
        # A connection that already authenticates with a variable stands
        # for its environment card: one at the provider's own endpoint
        # naming it in `credential`, or a saved connection storing it as its
        # key reference (contract v1.11 section 8)
        configured_cards = [
            card for card in providers if card.get("kind") == "configured"
        ]
        served = {
            (card["provider"], card["credential"])
            for card, (_provider, own) in zip(
                configured_cards, own_endpoints, strict=True
            )
            if own and card.get("credential")
        } | {
            (saved.provider, saved.api_key)
            for saved in state.configured_providers
            if isinstance(saved.api_key, str) and saved.api_key.startswith("env:")
        }
        providers = [
            card
            for card in providers
            if card.get("kind") != "environment"
            or (card.get("provider"), card.get("credential")) not in served
        ]
        return {
            "providers": [card_facts(annotate_card(dict(card))) for card in providers]
        }

    @app.get(
        "/api/settings/llm/providers/{provider_id}/credentials",
        response_model=LLMProviderCredentials,
        # api_key_source only when the environment serves the key
        response_model_exclude_unset=True,
    )
    async def get_llm_provider_credentials(
        request: Request, provider_id: str
    ) -> dict[str, str | None]:
        """Return one saved connection's editable values on explicit request.

        The stored key is never returned, only its masked form: the editor
        shows it and leaves ``api_key`` out of the update unless the user
        types a new key, so the stored one stays. The base comes without its
        userinfo and secret parameters; sent back as shown, it keeps the
        stored one (``_shown_base``).
        """
        return connection_credentials(_state(request), provider_id)

    def connection_credentials(
        state: _ModelLists, provider_id: str, *, environment: bool = True
    ) -> dict[str, str | None]:
        """The credentials route's answer for one connection of *state*'s."""
        saved = next(
            (
                provider
                for provider in state.configured_providers
                if provider.id == provider_id
            ),
            None,
        )
        if saved is not None:
            # The saved base (None when the connection uses the provider
            # default) so re-saving after only a key edit does not pin the
            # default as an explicit override; the default is offered
            # separately for the editor's placeholder.
            return _credentials_view(
                saved.provider, saved.api_key, saved.api_base, environment=environment
            )

        deployment_id = (
            provider_id.removeprefix("legacy:")
            if provider_id.startswith("legacy:")
            else None
        )
        legacy = (
            _configured_deployment_by_id(state, deployment_id)
            if deployment_id is not None
            else None
        )
        if legacy is not None:
            return _credentials_view(
                legacy.litellm_params.model.split("/", 1)[0].lower(),
                legacy.litellm_params.api_key,
                legacy.litellm_params.api_base,
                environment=environment,
            )

        raise ApiError(
            404,
            "settings_entry_missing",
            (
                "This saved provider connection no longer exists. "
                "Refresh the provider list and choose it again."
            ),
        )

    @app.post(
        "/api/settings/llm/model-discovery",
        response_model=ModelDiscoveryResult,
        # `detail` is only present when discovery has something to explain.
        response_model_exclude_unset=True,
    )
    async def discover_llm_models(
        request: Request, body: LLMModelDiscoveryRequest
    ) -> dict[str, Any]:
        return await discover(_state(request), body)

    async def discover(
        state: _ModelLists, body: LLMModelDiscoveryRequest, *, user: bool = False
    ) -> dict[str, Any]:
        """A model discovery over *state*'s connections; *user* holds it to
        a user's rules and never lets the server's environment serve it."""
        from markitai.config import resolve_env_value
        from markitai.providers.discovery import discover_models

        if user:
            if body.provider_id is not None and body.provider_id.startswith("env:"):
                raise user_models.environment_credentials()
            if body.provider.strip().lower() in user_models.SERVER_LOGIN:
                raise user_models.server_credentials()
        # An environment card's id names its provider (env:openai)
        environment_key = (
            _environment_key(
                body.provider_id,
                body.provider.strip().lower(),
                api_key=body.api_key,
                api_base=body.api_base,
            )
            if body.provider_id is not None and body.provider_id.startswith("env:")
            else None
        )
        stored_provider = next(
            (
                provider
                for provider in state.configured_providers
                if provider.id == body.provider_id
            ),
            None,
        )
        deployment_id = body.deployment_id
        if body.provider_id is not None and body.provider_id.startswith("legacy:"):
            deployment_id = body.provider_id.removeprefix("legacy:")
        stored = (
            _configured_deployment_by_id(state, deployment_id)
            if deployment_id is not None
            else None
        )
        if (
            body.provider_id is not None
            and environment_key is None
            and stored_provider is None
            and stored is None
        ):
            raise ApiError(
                404,
                "settings_entry_missing",
                (
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        if deployment_id is not None and stored is None:
            raise ApiError(
                404,
                "settings_entry_missing",
                (
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        fallback_key = (
            environment_key
            if environment_key is not None
            else stored_provider.api_key
            if stored_provider is not None
            else stored.litellm_params.api_key
            if stored is not None
            else None
        )
        fallback_base = (
            stored_provider.api_base
            if stored_provider is not None
            else stored.litellm_params.api_base
            if stored is not None
            else None
        )
        sent_base = None if _shown_base(body.api_base, fallback_base) else body.api_base
        provider = body.provider.strip().lower()
        if sent_base is not None and (
            stored is not None or stored_provider is not None
        ):
            # The connection's own address sent back as it is (or, for one
            # on the provider's default, that default): not one the caller
            # names
            own = fallback_base or provider_default_api_base(provider)
            if own is not None and sent_base.rstrip("/") == own.rstrip("/"):
                sent_base = None
        if body.api_key is None:
            _reject_rebound_credentials(fallback_key, fallback_base, sent_base)
        raw_key = body.api_key if body.api_key is not None else _real_key(fallback_key)
        raw_base = sent_base if sent_base is not None else fallback_base
        _check_credential_target(
            raw_key,
            raw_base,
            client_key=body.api_key,
            configured=_configured_credential_pairs(state),
        )
        if user:
            # A user's own key or address only, never the server's machine
            if any(
                isinstance(value, str) and value.strip().startswith("env:")
                for value in (raw_key, raw_base)
            ):
                raise user_models.environment_credentials()
            if provider in user_models.SERVER_LOGIN or (
                provider in ("bedrock", "bedrock_mantle")
                and not user_models.real_key(raw_key)
            ):
                raise user_models.server_credentials()
            user_models.check_address(
                provider, raw_base if isinstance(raw_base, str) and raw_base else None
            )
            if not user_models.real_key(raw_key):
                if not raw_base:
                    # Nothing is sent: the provider would get no key, or
                    # LiteLLM the server's own
                    return {
                        "provider": provider,
                        "status": "unavailable",
                        "source": "live_api",
                        "authoritative": False,
                        "cached": False,
                        "stale": False,
                        "models": [],
                        "detail": "This model needs an API key, and none is set",
                        "reason": "no_key",
                    }
                raw_key = NO_KEY
            return user_models.answer(
                await discover_models(
                    body.provider,
                    api_key=raw_key,
                    api_base=raw_base,
                    refresh=body.refresh,
                    key_variable=None,
                )
            )
        api_key = resolve_env_value(raw_key, strict=False) if raw_key else None
        api_base = resolve_env_value(raw_base, strict=False) if raw_base else None
        key_variable = (
            raw_key[4:]
            if isinstance(raw_key, str) and raw_key.startswith("env:")
            else None
        )
        if (
            api_key is None
            and raw_key is None
            and fallback_key != NO_KEY
            and sent_base is None
        ):
            # No key of its own: the server's key variable serves it, as it
            # serves the connection's conversions and model tests. Never for
            # an address the caller names (that takes the caller's own key)
            _names, key_variable = _key_variables(provider)
            found = _environment_key_variable(provider)
            if found is not None:
                api_key, key_variable = os.environ[found].strip(), found
        keyless_entry = body.api_key is None and fallback_key == NO_KEY
        if api_key is None and not keyless_entry:
            # A key the provider needs and nothing supplies (an unset env:
            # reference, or no key and the key variable unset): said before
            # anything is sent, as a model test does; the provider's 401
            # would read like a wrong key. Local servers, subscriptions and
            # clouds with their own credentials need none.
            lacking = model_test.missing_key(
                f"{provider}/", raw_key if isinstance(raw_key, str) else None, api_base
            )
            if lacking is not None:
                variable = model_test.missing_key_failure(lacking).key_variable
                answer = {
                    "provider": provider,
                    "status": "unavailable",
                    "source": "live_api",
                    "authoritative": False,
                    "cached": False,
                    "stale": False,
                    "models": [],
                    "detail": f"{variable} is not set on the server, so nothing was sent"
                    if variable
                    else "Model discovery needs an API key, and none is set, so "
                    "nothing was sent",
                    "reason": "no_key",
                }
                if variable:
                    answer["key_variable"] = variable
                return answer
        return await discover_models(
            body.provider,
            api_key=api_key,
            api_base=api_base,
            refresh=body.refresh,
            key_variable=key_variable,
        )

    async def _apply_model_list_mutation(
        state: ServeState,
        mutate: Callable[[list[Any], list[Any], dict[str, str]], list[Any]],
        *,
        expected_revision: str | None = None,
        backfill_ids: bool = False,
    ) -> dict[str, Any]:
        """Run one settings write: atomic file mutation + hot update."""
        async with state.settings_lock:
            try:
                merged, merged_providers, revision = await asyncio.to_thread(
                    _mutate_config_model_list,
                    state.config_path,
                    mutate,
                    fallback=state.configured_models,
                    fallback_providers=state.configured_providers,
                    expected_revision=expected_revision,
                    backfill_ids=backfill_ids,
                )
            except OSError as e:
                logger.error(
                    "[Serve] LLM settings write failed for {}: {}",
                    state.config_path,
                    type(e).__name__,
                )
                raise ApiError(
                    500,
                    "settings_io_failed",
                    f"failed to update config file: {type(e).__name__}",
                )
            except ValueError as e:
                logger.error(
                    "[Serve] LLM settings update failed for {}: {}",
                    state.config_path,
                    type(e).__name__,
                )
                raise ApiError(
                    500,
                    "settings_io_failed",
                    f"failed to update config file: {type(e).__name__}",
                )
            state.configured_models = merged
            state.configured_providers = merged_providers
            state.settings_revision = revision
            state.config.llm.providers = merged_providers
            state.config.llm.model_list = _effective_models(
                merged, state.detected_models
            )
            state.llm_source = (
                "config" if merged else "detected" if state.detected_models else "none"
            )
        return _llm_settings_payload(state)

    def raw_deployment(
        state: _ModelLists,
        body: LLMModelCreate,
        providers: list[Any],
        models: list[Any],
        *,
        user: bool = False,
    ) -> dict[str, Any]:
        credential_source = (
            _configured_deployment_by_id(state, body.credential_deployment_id)
            if body.credential_deployment_id is not None
            else None
        )
        provider_name = (body.provider or body.model.split("/", 1)[0]).strip().lower()
        # An environment card's id names its provider (env:openai): the model
        # is saved with that provider's key variable and its own endpoint
        environment_key = (
            _environment_key(
                body.credential_provider_id,
                provider_name,
                api_key=body.api_key,
                api_base=body.api_base,
            )
            if body.credential_provider_id is not None
            and body.credential_provider_id.startswith("env:")
            else None
        )
        credential_provider = (
            _raw_provider_by_id(providers, body.credential_provider_id)
            if body.credential_provider_id is not None and environment_key is None
            else None
        )
        if body.credential_deployment_id is not None and credential_source is None:
            raise ApiError(
                404,
                "settings_entry_missing",
                (
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        if (
            body.credential_provider_id is not None
            and environment_key is None
            and credential_provider is None
        ):
            raise ApiError(
                404,
                "settings_entry_missing",
                (
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )

        new_connection = (
            credential_provider is None
            and credential_source is None
            and environment_key is None
        )
        provider_id = (
            str(credential_provider["id"])
            if credential_provider is not None
            else credential_source.model_info.provider_id
            if credential_source is not None
            and credential_source.model_info is not None
            else None
        )
        fallback_key = (
            environment_key
            if environment_key is not None
            else credential_provider.get("api_key")
            if credential_provider is not None
            else credential_source.litellm_params.api_key
            if credential_source is not None
            else None
        )
        fallback_base = (
            credential_provider.get("api_base")
            if credential_provider is not None
            else credential_source.litellm_params.api_base
            if credential_source is not None
            else None
        )
        if body.api_key is None:
            _reject_rebound_credentials(fallback_key, fallback_base, body.api_base)
        api_key = body.api_key if body.api_key is not None else fallback_key
        api_base = body.api_base if body.api_base is not None else fallback_base
        if new_connection and (not user or user_models.real_key(api_key)):
            # A new connection left on the provider's default: where
            # LiteLLM's own default is another host (or api.openai.com), the
            # documented address the card showed is stored with it. A user's
            # keyless model must name an address of its own instead.
            from markitai.providers.connectable import pinned_base

            api_base = api_base or pinned_base(provider_name)
        if body.api_base is not None or new_connection:
            # An address this request names (or the pinned default of a new
            # connection) never gets the server's own key: keyless, it is
            # stored with the placeholder. A saved connection's own pair
            # stays as it is.
            api_key = _keyless_placeholder(api_key, api_base, body.model)
        _refuse_ambient_login_address(body.model, api_base)
        _check_credential_target(
            api_key,
            api_base,
            client_key=body.api_key,
            configured=_configured_credential_pairs(state),
        )
        if provider_id is None:
            provider_id = _ensure_raw_provider(
                providers,
                provider=provider_name,
                api_key=api_key,
                api_base=api_base,
                models=models,
            )

        params: dict[str, Any] = {"model": body.model}
        if api_key is not None:
            params["api_key"] = api_key
        if api_base is not None:
            params["api_base"] = api_base
        if body.weight != 1:
            params["weight"] = body.weight
        _strip_ignored_local_params(params)
        info = {"id": str(uuid.uuid4())}
        if provider_id is not None:
            info["provider_id"] = provider_id
        return {
            "model_name": body.model_name,
            "litellm_params": params,
            "model_info": info,
        }

    def update_raw_deployment(
        entry: dict[str, Any],
        body: LLMModelUpdate,
        configured: frozenset[CredentialPair],
    ) -> None:
        params = entry.get("litellm_params")
        if not isinstance(params, dict):
            params = {}
        old_pair = (params.get("api_key"), params.get("api_base"))
        base_set = "api_base" in body.model_fields_set and not _shown_base(
            body.api_base, params.get("api_base")
        )
        if "model_name" in body.model_fields_set and body.model_name is not None:
            entry["model_name"] = body.model_name
        if "model" in body.model_fields_set and body.model is not None:
            params["model"] = body.model
        if "api_key" in body.model_fields_set:
            if body.api_key is None:
                params.pop("api_key", None)
            else:
                params["api_key"] = body.api_key
        if base_set:
            if body.api_base is None:
                params.pop("api_base", None)
            else:
                params["api_base"] = body.api_base
        if "weight" in body.model_fields_set and body.weight is not None:
            params["weight"] = body.weight
        if "share" in body.model_fields_set:
            # The admin's: stored in the entry, so Edit config shows it
            info = entry.get("model_info")
            if not isinstance(info, dict):
                info = {}
                entry["model_info"] = info
            if body.share == "shared":
                info["share"] = "shared"
            else:
                info.pop("share", None)
        if "api_key" in body.model_fields_set or base_set:
            model = params.get("model")
            provider = model.split("/", 1)[0].lower() if isinstance(model, str) else ""
            _refuse_keyless_move(
                old_pair, (params.get("api_key"), params.get("api_base")), provider
            )
            placeholder = _keyless_placeholder(
                params.get("api_key"), params.get("api_base"), model
            )
            if placeholder is not None:
                params["api_key"] = placeholder
        _strip_ignored_local_params(params)
        if base_set or "model" in body.model_fields_set:
            _refuse_ambient_login_address(params.get("model"), params.get("api_base"))
        _check_credential_target(
            params.get("api_key"),
            params.get("api_base"),
            client_key=body.api_key if "api_key" in body.model_fields_set else None,
            configured=configured | {old_pair},
        )
        entry["litellm_params"] = params

    @app.post(
        "/api/settings/llm/models",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def add_llm_model(request: Request, body: LLMModelCreate) -> dict[str, Any]:
        state = _state(request)

        def mutate(
            entries: list[Any], providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            entries.append(raw_deployment(state, body, providers, entries))
            return entries

        payload = await _apply_model_list_mutation(
            state, mutate, expected_revision=body.expected_revision
        )
        logger.info("[Serve] LLM model added: {} ({})", body.model_name, body.model)
        return payload

    @app.put(
        "/api/settings/llm/models/{model_name}",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def update_llm_model(
        request: Request, model_name: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        configured = _configured_credential_pairs(_state(request))

        def mutate(
            entries: list[Any], _providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            matches = _find_raw_entries(entries, model_name)
            if not matches:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    f"model_name '{model_name}' not found",
                )
            if len(matches) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "ambiguous_legacy_model_name",
                        "model_name": model_name,
                    },
                )
            update_raw_deployment(matches[0], body, configured)
            return entries

        payload = await _apply_model_list_mutation(
            _state(request), mutate, expected_revision=body.expected_revision
        )
        logger.info("[Serve] LLM model updated through legacy route: {}", model_name)
        return payload

    @app.delete(
        "/api/settings/llm/models/{model_name}",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def delete_llm_model(request: Request, model_name: str) -> dict[str, Any]:
        def mutate(
            entries: list[Any], providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            matches = _find_raw_entries(entries, model_name)
            if not matches:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    f"model_name '{model_name}' not found",
                )
            if len(matches) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "ambiguous_legacy_model_name",
                        "model_name": model_name,
                    },
                )
            target = matches[0]
            _preserve_raw_deployment_provider(target, entries, providers)
            return [entry for entry in entries if entry is not target]

        payload = await _apply_model_list_mutation(_state(request), mutate)
        logger.info("[Serve] LLM model deleted through legacy route: {}", model_name)
        return payload

    @app.post(
        "/api/settings/llm/deployments/batch",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def add_llm_deployments_batch(
        request: Request, body: LLMDeploymentBatch
    ) -> dict[str, Any]:
        state = _state(request)
        return await _apply_model_list_mutation(
            state,
            add_deployments(state, body),
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    def add_deployments(
        state: _ModelLists, body: LLMDeploymentBatch, *, user: bool = False
    ) -> _Mutation:
        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            # Round 18: removals and additions in one change. Every id names
            # an entry of the list the revision identifies: ids to remove
            # through the backfill's mapping, and an addition's
            # credential_deployment_id through *state*, still the list before
            # this change, so removing the model a provider card stands for
            # and adding another through that card is one valid change
            targets = []
            for deployment_id in body.remove:
                entry = _find_raw_deployment(entries, deployment_id, legacy_mapping)
                if entry is None:
                    raise ApiError(
                        404,
                        "settings_entry_missing",
                        (
                            "This model configuration no longer exists. "
                            "Refresh settings and try again."
                        ),
                    )
                if any(entry is target for target in targets):
                    raise ApiError(
                        422,
                        "invalid_settings_request",
                        "remove lists one deployment twice",
                    )
                targets.append(entry)
            for entry in targets:
                _preserve_raw_deployment_provider(entry, entries, providers)
            entries = [
                entry
                for entry in entries
                if not any(entry is target for target in targets)
            ]
            for deployment in body.deployments:
                entries.append(
                    raw_deployment(state, deployment, providers, entries, user=user)
                )
            return entries

        return mutate

    @app.patch(
        "/api/settings/llm/deployments/{deployment_id}",
        response_model=LLMSettingsPayload,
    )
    async def update_llm_deployment(
        request: Request, deployment_id: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        if body.expected_revision is None:
            raise ApiError(
                422, "invalid_settings_request", "expected_revision is required"
            )
        return await _apply_model_list_mutation(
            _state(request),
            update_deployment(_state(request), deployment_id, body),
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    def update_deployment(
        state: _ModelLists, deployment_id: str, body: LLMModelUpdate
    ) -> _Mutation:
        configured = _configured_credential_pairs(state)

        def mutate(
            entries: list[Any], _providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            entry = _find_raw_deployment(entries, deployment_id, legacy_mapping)
            if entry is None:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    (
                        "This model configuration no longer exists. "
                        "Refresh settings and try again."
                    ),
                )
            update_raw_deployment(entry, body, configured)
            return entries

        return mutate

    @app.delete(
        "/api/settings/llm/deployments/{deployment_id}",
        response_model=LLMSettingsPayload,
    )
    async def delete_llm_deployment(
        request: Request, deployment_id: str, expected_revision: str
    ) -> dict[str, Any]:
        return await _apply_model_list_mutation(
            _state(request),
            delete_deployment(deployment_id),
            expected_revision=expected_revision,
            backfill_ids=True,
        )

    def delete_deployment(deployment_id: str) -> _Mutation:
        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            entry = _find_raw_deployment(entries, deployment_id, legacy_mapping)
            if entry is None:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    (
                        "This model configuration no longer exists. "
                        "Refresh settings and try again."
                    ),
                )
            _preserve_raw_deployment_provider(entry, entries, providers)
            return [candidate for candidate in entries if candidate is not entry]

        return mutate

    def locate_raw_provider(
        entries: list[Any],
        providers: list[Any],
        provider_id: str,
        legacy_mapping: dict[str, str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        target = _raw_provider_by_id(providers, provider_id)
        legacy_entry = None
        if target is None and provider_id.startswith("legacy:"):
            legacy_entry = _find_raw_deployment(
                entries,
                provider_id.removeprefix("legacy:"),
                legacy_mapping,
            )
        return target, legacy_entry

    @app.patch(
        "/api/settings/llm/providers/{provider_id}",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def update_llm_provider(
        request: Request, provider_id: str, body: LLMProviderUpdate
    ) -> dict[str, Any]:
        return await _apply_model_list_mutation(
            _state(request),
            update_provider(_state(request), provider_id, body),
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    def update_provider(
        state: _ModelLists, provider_id: str, body: LLMProviderUpdate
    ) -> _Mutation:
        configured = _configured_credential_pairs(state)

        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            target, legacy_entry = locate_raw_provider(
                entries, providers, provider_id, legacy_mapping
            )
            if target is None and legacy_entry is None:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    (
                        "This saved provider connection no longer exists. "
                        "Refresh the provider list and choose it again."
                    ),
                )

            legacy_params = (
                legacy_entry.get("litellm_params")
                if isinstance(legacy_entry, dict)
                else None
            )
            if not isinstance(legacy_params, dict):
                legacy_params = {}
            legacy_model = legacy_params.get("model")
            provider_name = (
                str(target.get("provider"))
                if target is not None
                else legacy_model.split("/", 1)[0].lower()
                if isinstance(legacy_model, str)
                else ""
            )
            old_key = (
                target.get("api_key")
                if target is not None
                else legacy_params.get("api_key")
            )
            old_base = (
                target.get("api_base")
                if target is not None
                else legacy_params.get("api_base")
            )
            key_set = "api_key" in body.model_fields_set
            base_set = "api_base" in body.model_fields_set and not _shown_base(
                body.api_base, old_base
            )
            if not key_set and not base_set:
                return entries  # only the base, sent back as shown
            if target is None:
                target = {
                    "id": str(uuid.uuid4()),
                    "provider": provider_name,
                    **({"api_key": old_key} if isinstance(old_key, str) else {}),
                    **({"api_base": old_base} if isinstance(old_base, str) else {}),
                }
                providers.append(target)

            if key_set:
                if body.api_key is None:
                    target.pop("api_key", None)
                else:
                    target["api_key"] = body.api_key
            if base_set:
                if body.api_base is None:
                    target.pop("api_base", None)
                else:
                    target["api_base"] = body.api_base
            _refuse_keyless_move(
                (old_key, old_base),
                (target.get("api_key"), target.get("api_base")),
                provider_name,
            )
            if base_set:
                _refuse_ambient_login_address(
                    f"{provider_name}/", target.get("api_base")
                )
            placeholder = _keyless_placeholder(
                target.get("api_key"), target.get("api_base"), f"{provider_name}/"
            )
            if placeholder is not None and placeholder != target.get("api_key"):
                target["api_key"] = placeholder
                key_set = True
            _check_credential_target(
                target.get("api_key"),
                target.get("api_base"),
                client_key=body.api_key if "api_key" in body.model_fields_set else None,
                configured=configured | {(old_key, old_base)},
            )
            new_key = target.get("api_key")

            saved_id = str(target["id"])
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                params = entry.get("litellm_params")
                if not isinstance(params, dict):
                    continue
                model = params.get("model")
                matches_old_connection = (
                    isinstance(model, str)
                    and model.split("/", 1)[0].lower() == provider_name
                    and params.get("api_key") == old_key
                    and params.get("api_base") == old_base
                )
                if (
                    _raw_model_provider_id(entry) != saved_id
                    and not matches_old_connection
                ):
                    continue
                if key_set:
                    if new_key is None:
                        params.pop("api_key", None)
                    else:
                        params["api_key"] = new_key
                if base_set:
                    if body.api_base is None:
                        params.pop("api_base", None)
                    else:
                        params["api_base"] = body.api_base
                    _refuse_ambient_login_address(model, params.get("api_base"))
                _link_raw_model_to_provider(entry, saved_id)
            return entries

        return mutate

    @app.delete(
        "/api/settings/llm/providers/{provider_id}",
        response_model=LLMSettingsPayload,
        # `share` only on the configuration file's rows
        response_model_exclude_unset=True,
    )
    async def delete_llm_provider(
        request: Request, provider_id: str, expected_revision: str
    ) -> dict[str, Any]:
        return await _apply_model_list_mutation(
            _state(request),
            delete_provider(provider_id),
            expected_revision=expected_revision,
            backfill_ids=True,
        )

    def delete_provider(provider_id: str) -> _Mutation:
        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            target, legacy_entry = locate_raw_provider(
                entries, providers, provider_id, legacy_mapping
            )
            if target is None and legacy_entry is None:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    (
                        "This saved provider connection no longer exists. "
                        "Refresh the provider list and choose it again."
                    ),
                )
            legacy_params = (
                legacy_entry.get("litellm_params")
                if isinstance(legacy_entry, dict)
                else None
            )
            if not isinstance(legacy_params, dict):
                legacy_params = {}
            model = legacy_params.get("model")
            provider_name = (
                str(target.get("provider"))
                if target is not None
                else model.split("/", 1)[0].lower()
                if isinstance(model, str)
                else ""
            )
            saved_id = str(target["id"]) if target is not None else None
            old_key = (
                target.get("api_key")
                if target is not None
                else legacy_params.get("api_key")
            )
            old_base = (
                target.get("api_base")
                if target is not None
                else legacy_params.get("api_base")
            )
            if target is not None:
                providers[:] = [
                    candidate for candidate in providers if candidate is not target
                ]

            remaining: list[Any] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    remaining.append(entry)
                    continue
                params = entry.get("litellm_params")
                entry_model = params.get("model") if isinstance(params, dict) else None
                matches = (
                    saved_id is not None and _raw_model_provider_id(entry) == saved_id
                )
                if not matches and isinstance(params, dict):
                    matches = (
                        isinstance(entry_model, str)
                        and entry_model.split("/", 1)[0].lower() == provider_name
                        and params.get("api_key") == old_key
                        and params.get("api_base") == old_base
                    )
                if not matches:
                    remaining.append(entry)
            return remaining

        return mutate

    # ------------------------------------------------- a user's own models

    def user_caller(request: Request) -> str:
        """The user token a ``/api/me/models`` request acts for; anyone else
        is refused (contract v1.15)."""
        caller = _caller(request)
        if caller.role != "user" or caller.user_id is None:
            raise user_models.user_token_required()
        return caller.user_id

    def shared_now(state: ServeState) -> list[user_models.Shared]:
        """The admin's models users see now (shared, weight above 0)."""
        return user_models.shared_list(
            state.configured_models, state.configured_providers
        )

    def stale(current: str) -> HTTPException:
        return HTTPException(
            status_code=409,
            detail={"code": "stale_revision", "current_revision": current},
        )

    def user_view(
        state: ServeState, user_id: str, lists: user_models.Lists
    ) -> dict[str, Any]:
        """``UserModelsView``: the user's own models, and of the admin's
        shared ones only what they are. Blocking (LiteLLM may route a bare
        model name)."""
        scope = _UserScope.of(lists)
        conversion = conversion_for(state, user_id)
        return {
            "configured": bool(lists.models),
            "routable": conversion.routable,
            "source": "user",
            "config_path": None,
            "config_origin": "user",
            "revision": _user_revision(lists),
            "deployments": [
                _deployment_view(model, index, persisted=True)
                for index, model in enumerate(scope.configured_models)
            ],
            "detected": [],
            # What a shared model is, never its key, address or connection:
            # the user's weight for it and the admin's
            "shared": [
                {
                    "id": shared.id,
                    "model": shared.model,
                    "provider": shared.provider,
                    "weight": user_models.effective_weight(lists, shared),
                    "default_weight": shared.weight,
                }
                for shared in shared_now(state)
            ],
            "limit": user_models.LIMIT,
            "batch_remove": True,
        }

    async def user_scope(user_id: str) -> _UserScope:
        return _UserScope.of(await asyncio.to_thread(users.lists, user_id))

    async def apply_user_mutation(
        request: Request,
        make: Callable[[_ModelLists], _Mutation],
        *,
        expected_revision: str,
    ) -> dict[str, Any]:
        """Run one write on the user's own lists with the settings' own
        operation (*make*), then the user rules, then save them."""
        user_id = user_caller(request)
        state = _state(request)

        def change(lists: user_models.Lists) -> None:
            current = _user_revision(lists)
            if expected_revision != current:
                raise stale(current)
            legacy_mapping = _backfill_raw_deployment_ids(lists.models)
            mutate = make(_UserScope.of(lists))
            lists.models = mutate(lists.models, lists.providers, legacy_mapping)
            user_models.check(lists)
            try:
                _UserScope.of(lists)
            except ValidationError as e:
                raise ApiError(
                    422, "invalid_settings_request", _validation_words(e)
                ) from None

        def write() -> dict[str, Any]:
            lists = users.write(user_id, lambda: access.exists(user_id), change)
            return user_view(state, user_id, lists)

        return await asyncio.to_thread(write)

    @app.get(
        "/api/me/models",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def get_user_models(request: Request) -> dict[str, Any]:
        user_id = user_caller(request)
        state = _state(request)

        def view() -> dict[str, Any]:
            return user_view(state, user_id, users.lists(user_id))

        return await asyncio.to_thread(view)

    @app.get(
        "/api/me/models/providers",
        response_model=ProviderConnectionList,
        response_model_exclude_unset=True,
    )
    async def get_user_providers(
        request: Request, refresh: bool = False
    ) -> dict[str, Any]:
        """Built-in cards a user may use, and its own connections: never an
        environment card or the server's key variables."""
        from markitai.providers.connectable import routable_providers

        del refresh  # computed on every request, as the admin's list is
        scope = await user_scope(user_caller(request))
        routable = await asyncio.to_thread(routable_providers)
        return {"providers": _user_provider_cards(scope, routable)}

    @app.get(
        "/api/me/models/providers/{provider_id}/credentials",
        response_model=LLMProviderCredentials,
        response_model_exclude_unset=True,
    )
    async def get_user_provider_credentials(
        request: Request, provider_id: str
    ) -> dict[str, str | None]:
        scope = await user_scope(user_caller(request))
        return connection_credentials(scope, provider_id, environment=False)

    @app.post(
        "/api/me/models/model-discovery",
        response_model=ModelDiscoveryResult,
        response_model_exclude_unset=True,
    )
    async def discover_user_models(
        request: Request, body: LLMModelDiscoveryRequest
    ) -> dict[str, Any]:
        scope = await user_scope(user_caller(request))
        return await discover(scope, body, user=True)

    @app.post(
        "/api/me/models/test",
        response_model=LLMTestResult,
        response_model_exclude_none=True,
    )
    async def test_user_model(request: Request, body: UserModelTest) -> dict[str, Any]:
        user_id = user_caller(request)
        if body.shared_id is not None:
            return await shared_probe(_state(request), user_id, body.shared_id)
        scope = await user_scope(user_id)
        draft = LLMSettingsUpdate.model_validate(
            body.model_dump(exclude={"shared_id"}, exclude_unset=True)
        )
        return await probe(scope, draft, user=True)

    async def shared_probe(
        state: ServeState, user_id: str, shared_id: str
    ) -> dict[str, Any]:
        """Test a model the admin shares on the server, as the admin's own
        test of it runs (its key, connection and address), one at a time and
        a few a minute per user (the admin's key pays); the answer says only
        whether it passed and the test's reason (contract v1.16)."""
        if all(shared.id != shared_id for shared in shared_now(state)):
            raise ApiError(422, "invalid_settings_request", "deployment not found")
        wait = users.begin_test(user_id)
        if wait is not None:
            raise ApiError(
                429,
                "provider_busy",
                "you can test a shared model once at a time and "
                f"{user_models.SHARED_TESTS_PER_MINUTE} times a minute; try "
                "again shortly",
                headers={"Retry-After": str(wait)},
            )
        try:
            answer = await probe(state, LLMSettingsUpdate(deployment_id=shared_id))
        except (ApiError, HTTPException):
            # Refused before anything was sent (invalid_provider_request,
            # provider_unavailable, ...): a failed test, in the same words
            answer = {"ok": False, "reason": "provider_error"}
        finally:
            users.end_test(user_id)
        if answer.get("ok") is True:
            return {"ok": True, "detail": "Model test passed"}
        reason = answer.get("reason")
        return {
            "ok": False,
            "reason": reason if reason in model_test.REASONS else "provider_error",
            "detail": SHARED_TEST_FAILED,
        }

    @app.post(
        "/api/me/models/deployments/batch",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def add_user_models(
        request: Request, body: LLMDeploymentBatch
    ) -> dict[str, Any]:
        user_caller(request)
        # What a request names is refused before anything changes
        for deployment in body.deployments:
            await asyncio.to_thread(
                user_models.precheck,
                deployment.model,
                deployment.provider,
                deployment.credential_provider_id,
            )
        return await apply_user_mutation(
            request,
            lambda scope: add_deployments(scope, body, user=True),
            expected_revision=body.expected_revision,
        )

    @app.patch(
        "/api/me/models/deployments/{deployment_id}",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def update_user_model(
        request: Request, deployment_id: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        user_caller(request)
        if "share" in body.model_fields_set:
            raise ApiError(
                422,
                "invalid_settings_request",
                "unknown field 'share': sharing is the admin's",
            )
        if body.expected_revision is None:
            raise ApiError(
                422, "invalid_settings_request", "expected_revision is required"
            )
        if body.model is not None:
            await asyncio.to_thread(user_models.precheck, body.model)
        return await apply_user_mutation(
            request,
            lambda scope: update_deployment(scope, deployment_id, body),
            expected_revision=body.expected_revision,
        )

    @app.delete(
        "/api/me/models/deployments/{deployment_id}",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def delete_user_model(
        request: Request, deployment_id: str, expected_revision: str
    ) -> dict[str, Any]:
        return await apply_user_mutation(
            request,
            lambda _scope: delete_deployment(deployment_id),
            expected_revision=expected_revision,
        )

    @app.patch(
        "/api/me/models/providers/{provider_id}",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def update_user_provider(
        request: Request, provider_id: str, body: LLMProviderUpdate
    ) -> dict[str, Any]:
        return await apply_user_mutation(
            request,
            lambda scope: update_provider(scope, provider_id, body),
            expected_revision=body.expected_revision,
        )

    @app.delete(
        "/api/me/models/providers/{provider_id}",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def delete_user_provider(
        request: Request, provider_id: str, expected_revision: str
    ) -> dict[str, Any]:
        return await apply_user_mutation(
            request,
            lambda _scope: delete_provider(provider_id),
            expected_revision=expected_revision,
        )

    @app.patch(
        "/api/me/models/shared/{shared_id}",
        response_model=UserModelsView,
        response_model_exclude_unset=True,
    )
    async def update_user_shared_weight(
        request: Request, shared_id: str
    ) -> dict[str, Any]:
        """The user's own weight for a model the admin shares (contract
        v1.16): body exactly ``{expected_revision, weight}``, ``weight`` a
        whole number or null (null, or the admin's weight: follow the admin;
        0: off for this user only)."""
        user_id = user_caller(request)
        state = _state(request)
        body = await _json_body(request, _SHARED_WEIGHT_BODY_LIMIT)
        shape = ApiError(
            422,
            "invalid_settings_request",
            'send exactly {"expected_revision": string, "weight": integer or null}',
        )
        if (
            not isinstance(body, dict)
            or set(body) != {"expected_revision", "weight"}
            or not isinstance(body["expected_revision"], str)
        ):
            raise shape
        weight = body["weight"]
        if weight is not None and not user_models.routing_weight(weight):
            raise user_models.invalid_weight()
        expected: str = body["expected_revision"]

        def write() -> dict[str, Any]:
            target = next(
                (shared for shared in shared_now(state) if shared.id == shared_id),
                None,
            )
            if target is None:
                raise ApiError(
                    404,
                    "settings_entry_missing",
                    "That shared model is no longer shared. Refresh the list.",
                )

            def change(lists: user_models.Lists) -> None:
                current = _user_revision(lists)
                if expected != current:
                    raise stale(current)
                user_models.set_weight(lists, target, weight)

            lists = users.write(user_id, lambda: access.exists(user_id), change)
            return user_view(state, user_id, lists)

        return await asyncio.to_thread(write)

    # ------------------------------------------------ a user's edit config

    def user_config_view(state: ServeState, user_id: str) -> dict[str, Any]:
        """A user's Edit config document. Blocking: its file may be read."""
        lists = users.lists(user_id)
        return user_config.view(
            lists,
            shared_now(state),
            exists=users.exists(user_id),
            revision=_user_revision(lists),
        )

    @app.get("/api/me/config", response_model=ConfigDocument)
    async def get_user_config(request: Request) -> dict[str, Any]:
        """A user's own models file as one document the server writes, every
        secret masked (contract v1.16); never the server's config.json."""
        user_id = user_caller(request)
        state = _state(request)
        return await asyncio.to_thread(user_config_view, state, user_id)

    @app.put("/api/me/config", response_model=SavedConfigDocument)
    async def put_user_config(request: Request) -> dict[str, Any]:
        """Replace a user's own models from its document: the revision, the
        kept masks, the document's shape, then the rules of the Models
        routes, saved through the user's lock and atomic write."""
        user_id = user_caller(request)
        state = _state(request)
        expected, text = await _config_body(request)

        def change(lists: user_models.Lists) -> None:
            current = _user_revision(lists)
            if expected != current:
                raise stale(current)
            try:
                config_editor.parse(text)
            except json.JSONDecodeError as e:
                raise user_config.refused(config_editor.position(e)) from None
            except ValueError as e:
                raise user_config.refused(str(e)) from None
            shared = shared_now(state)
            try:
                saved = config_editor.unmask(text, user_config.document(lists, shared))
            except ValueError as e:
                raise user_config.refused(str(e)) from None
            parsed = user_config.parse(json.loads(saved), lists, shared)
            for entry in parsed.models:
                info = entry.get("model_info")
                user_models.precheck(
                    entry["litellm_params"]["model"],
                    None,
                    info.get("provider_id") if isinstance(info, dict) else None,
                )
            user_models.check(parsed)
            _backfill_raw_deployment_ids(parsed.models)
            user_config.unique(
                "models",
                [
                    entry["model_info"].get("id")
                    for entry in parsed.models
                    if isinstance(entry.get("model_info"), dict)
                ],
            )
            user_config.unique(
                "providers", [connection["id"] for connection in parsed.providers]
            )
            # The pool must build from it
            try:
                scope = _UserScope.of(parsed)
            except ValidationError as e:
                raise user_config.refused(_validation_words(e)) from None
            del scope
            lists.models = parsed.models
            lists.providers = parsed.providers
            lists.shared_weights = parsed.shared_weights

        def write() -> dict[str, Any]:
            users.write(user_id, lambda: access.exists(user_id), change)
            return {**user_config_view(state, user_id), "restart_required": False}

        return await asyncio.to_thread(write)

    # ------------------------------------------------------- edit config

    def _config_view(state: ServeState, data: bytes | None) -> dict[str, Any]:
        return config_editor.view(
            _display_config_path(state.config_path),
            state.config_origin,
            data,
            # No session overrides of the model list exist in Python's serve
            read_only=False,
        )

    @app.get("/api/settings/config", response_model=ConfigDocument)
    async def get_config_document(request: Request) -> dict[str, Any]:
        """The configuration file as text, each secret masked (contract §8)."""
        state = _state(request)
        async with state.settings_lock:
            data = await asyncio.to_thread(_read_config_bytes, state.config_path)
        return _config_view(state, data)

    @app.put("/api/settings/config", response_model=SavedConfigDocument)
    async def put_config_document(request: Request) -> dict[str, Any]:
        """Replace the configuration file with the text the page sends.

        Body exactly ``{expected_revision, text}``. Each masked value kept
        where the GET sent it gets the stored literal back; the result must
        validate as the server's configuration, and is then saved exactly as
        sent (the diff the page showed is what is written). The model list
        and connections apply at once; ``restart_required`` says whether
        anything else changed.
        """
        state = _state(request)
        expected, text = await _config_body(request)
        async with state.settings_lock:
            saved, old, document = await asyncio.to_thread(
                _replace_config_file,
                state.config_path,
                expected,
                text,
            )
            _apply_config_models(state, document)
        logger.info(
            "[Serve] Configuration file {} saved from the web workbench",
            _display_config_path(state.config_path),
        )
        return {
            **_config_view(state, saved),
            "restart_required": config_editor.restart_required(old, document),
        }

    # ------------------------------------------------ convert defaults

    def convert_defaults_view(
        state: ServeState, caller: Caller, lists: user_models.Lists | None
    ) -> dict[str, Any]:
        server = convert_defaults.of_config(state.config)
        if caller.role == "user":
            own = lists.convert_defaults if lists is not None else None
            return convert_defaults.answer(server, own, scope="user", can_save=True)
        return convert_defaults.answer(
            server, None, scope="server", can_save=caller.role == "admin"
        )

    @app.get("/api/convert-defaults")
    async def get_convert_defaults(request: Request) -> dict[str, Any]:
        """The options the composer starts from, for every caller (v1.17)."""
        state = _state(request)
        caller = _caller(request)
        if caller.role == "user" and caller.user_id is not None:
            user_id = caller.user_id
            lists = await asyncio.to_thread(users.lists, user_id)
            return convert_defaults_view(state, caller, lists)
        return convert_defaults_view(state, caller, None)

    @app.put("/api/convert-defaults")
    async def put_convert_defaults(request: Request) -> dict[str, Any]:
        """Save the composer's defaults: the admin's into the configuration
        file (and the running configuration), a user's into its own file;
        ``defaults: null`` returns a user to the server's (v1.17)."""
        state = _state(request)
        caller = _caller(request)
        shape = convert_defaults.invalid(
            'send a JSON body {"expected_revision": string, "defaults": object}'
        )
        try:
            body = json.loads(await request.body())
        except ValueError:
            raise shape from None
        if (
            not isinstance(body, dict)
            or set(body) != {"expected_revision", "defaults"}
            or not isinstance(body["expected_revision"], str)
        ):
            raise shape
        expected: str = body["expected_revision"]
        wanted = (
            None
            if body["defaults"] is None
            else convert_defaults.checked(body["defaults"])
        )
        if caller.role == "anonymous":
            raise ApiError(
                403,
                "settings_forbidden",
                "only the server's administrator or a user token may save defaults",
            )
        if caller.role == "user" and caller.user_id is not None:
            user_id = caller.user_id

            def change(lists: user_models.Lists) -> None:
                current = convert_defaults.revision(lists.convert_defaults)
                if expected != current:
                    raise stale(current)
                lists.convert_defaults = wanted

            def write() -> user_models.Lists:
                return users.write(user_id, lambda: access.exists(user_id), change)

            lists = await asyncio.to_thread(write)
            return convert_defaults_view(state, caller, lists)
        if wanted is None:
            raise convert_defaults.invalid(
                "the server's defaults cannot be removed; send the options to use"
            )
        async with state.settings_lock:
            await asyncio.to_thread(
                _save_server_defaults, state.config_path, state.config, expected, wanted
            )
            convert_defaults.write_into_running(state.config, wanted)
        return convert_defaults_view(state, caller, None)

    @app.post("/api/settings/llm/config/open", status_code=204)
    async def open_llm_config(request: Request) -> None:
        """Open the config file the settings routes write in the host's editor.

        Loopback-only like every settings route: it launches a program on the
        server host, which only makes sense when the browser is that host.
        """
        state = _state(request)
        if not state.config_path.is_file():
            raise ApiError(
                404,
                "config_missing",
                "config file does not exist yet; save a model to create it",
            )
        try:
            await asyncio.to_thread(_open_in_editor, state.config_path)
        except Exception as e:
            raise ApiError(
                500, "settings_io_failed", f"could not open the config file: {e}"
            ) from e

    @app.post(
        "/api/settings/llm/test",
        response_model=LLMTestResult,
        # Success is {ok, detail}; key_variable only when known
        response_model_exclude_none=True,
    )
    async def test_llm_settings(
        request: Request, body: LLMSettingsUpdate
    ) -> dict[str, Any]:
        return await probe(_state(request), body)

    async def probe(
        state: _ModelLists, body: LLMSettingsUpdate, *, user: bool = False
    ) -> dict[str, Any]:
        """Test one of *state*'s deployments, or a draft; *user* holds it to
        a user's rules, before anything is sent."""
        if user and body.model is not None:
            await asyncio.to_thread(user_models.precheck, body.model)
        if body.deployment_id is not None or body.model_name is not None:
            body = _stored_probe_body(
                state,
                deployment_id=body.deployment_id,
                model_name=body.model_name,
            )
        else:
            _check_credential_target(
                body.api_key,
                body.api_base,
                client_key=body.api_key,
                configured=_configured_credential_pairs(state),
            )
            asked_base = body.api_base
            # An address the caller names; a stored one is the admin's
            _refuse_ambient_login_address(body.model, asked_base)
            body = _fill_stored_credentials(state, body)
            if asked_base is not None:
                # An address the caller names is tested with the placeholder
                # rather than the server's own key variable
                body = body.model_copy(
                    update={
                        "api_key": _keyless_placeholder(
                            body.api_key, body.api_base, body.model
                        )
                    }
                )
        if user:
            provider = await asyncio.to_thread(user_models.route_provider, body.model)
            user_models.check_connection(provider, body.api_key, body.api_base)
            if not user_models.real_key(body.api_key):
                body = body.model_copy(update={"api_key": NO_KEY})
        if body.model is not None:
            lacking = model_test.missing_key(body.model, body.api_key, body.api_base)
            if lacking is not None:
                # Nothing is sent: LiteLLM would send the request without a
                # key, and the provider's 401 reads like a wrong key
                failure = model_test.missing_key_failure(lacking).answer()
                return user_models.answer(failure) if user else failure
        try:
            detail = await asyncio.wait_for(
                _probe_llm(body), timeout=_LLM_TEST_BACKSTOP_S
            )
        except Exception as e:  # must never 500: ok flags the result
            failure = _probe_failure(e, body)
            return user_models.answer(failure) if user else failure
        return {"ok": True, "detail": detail}

    @app.post("/api/jobs", status_code=201, response_model=CreateJobResponse)
    async def create_job(
        request: Request,
        files: list[UploadFile] = File(default=[]),
        urls: str = Form(default="[]"),
        options: str = Form(default="{}"),
        label: str | None = Form(default=None),
    ) -> dict[str, Any]:
        state = _state(request)
        if state.registry.closing:
            raise ApiError(503, "shutting_down", "server is shutting down")

        try:
            url_list = json.loads(urls) if urls.strip() else []
        except json.JSONDecodeError:
            raise ApiError(
                422, "invalid_urls", "urls must be a JSON array of strings"
            ) from None
        if not isinstance(url_list, list) or not all(
            isinstance(u, str) and u.strip() for u in url_list
        ):
            raise ApiError(
                422, "invalid_urls", "urls must be a JSON array of non-empty strings"
            )
        opts = _parse_job_options(options)
        job_label = _parse_label(label)

        # A `.urls` upload is a URL list, not a document: its entries join
        # the job as URL items (after any listed ones' files, before the
        # `urls` field's), the way the CLI reads a .urls input.
        documents: list[UploadFile] = []
        url_entries: list[tuple[str, str | None]] = []
        for upload in files:
            if (upload.filename or "").lower().endswith(".urls"):
                url_entries.extend(await _read_url_list(upload))
            else:
                documents.append(upload)
        url_entries.extend((url.strip(), None) for url in url_list)

        total = len(documents) + len(url_entries)
        if total == 0:
            raise ApiError(422, "empty_job", "at least one file or url is required")
        if total > MAX_JOB_ITEMS:
            raise ApiError(
                422,
                "too_many_items",
                f"too many items: {total} (max {MAX_JOB_ITEMS})",
            )
        for url, _ in url_entries:
            problem = _url_problem(url)
            if problem == "invalid_url":
                raise ApiError(422, problem, "invalid URL: not an absolute web address")
            if problem is not None:
                raise ApiError(422, problem, "URLs must use http or https")

        await _guard_fetch_targets(request, [url for url, _ in url_entries])
        _check_preset(state, opts)
        trusted = _is_trusted_request(request)
        _check_remote_processing(state, opts, trusted=trusted)
        # Without a routable model every item would fail or silently skip the
        # model the user asked for: refuse before anything is stored. A
        # user's conversions route over its own pool (contract v1.15).
        owner = _caller(request).user_id
        conversion = await conversion_of(state, owner)
        _require_model(conversion, opts)

        cfg = job_config(conversion, opts)
        job = state.registry.create_job(
            options=_saved_options(opts), cfg=cfg, owner=owner
        )
        job.hidden_hosts = conversion.hidden_hosts
        job.public_network_only = not trusted
        job.label = job_label
        cfg.output.dir = str(job.out_dir)

        from markitai.utils.cli_helpers import url_to_filename
        from markitai.utils.paths import derive_output_name

        # Any failure below must roll the half-created job back (registry
        # entry + job dir), or it stays a permanent "running" zombie whose
        # SSE stream never emits the terminal `job` event.
        try:
            taken_uploads = {
                entry.name.casefold() for entry in job.uploads_dir.iterdir()
            }
            for index, upload in enumerate(documents, start=1):
                if index > 1:
                    # Reading an upload spooled in memory never suspends: a
                    # folder of small files would hold the loop, and every
                    # other request, until the last one is saved.
                    await asyncio.sleep(0)
                saved = await _save_upload(upload, job.uploads_dir, taken_uploads)
                job.items.append(
                    JobItem(
                        item_id=f"i{index}",
                        name=saved.name,
                        kind="file",
                        source=saved,
                        source_size=saved.stat().st_size,
                    )
                )

            # Pre-assign unique output names per job: colliding
            # url_to_filename results (or a URL colliding with an upload's
            # derived output) would otherwise silently overwrite each other
            # in LLM mode, where only the .llm.md sibling is written.
            # Compared case-insensitively: on macOS/Windows file systems
            # "Page.html.md" and "page.html.md" are one file.
            taken = {derive_output_name(item.name).casefold() for item in job.items}
            for offset, (url, named) in enumerate(
                url_entries, start=len(documents) + 1
            ):
                base_name = named or url_to_filename(url)
                output_name = base_name
                counter = 2
                while output_name.casefold() in taken:
                    output_name = (
                        f"{Path(base_name).stem} ({counter}){Path(base_name).suffix}"
                    )
                    counter += 1
                taken.add(output_name.casefold())
                job.items.append(
                    JobItem(
                        item_id=f"i{offset}",
                        name=url,
                        kind="url",
                        source=url,
                        output_name=output_name,
                    )
                )

            job.task = asyncio.create_task(run_job(state.registry, job))
        except HTTPException:
            state.registry.discard_job(job)
            raise
        except Exception as e:
            state.registry.discard_job(job)
            logger.exception("[Serve] Job {} creation failed: {}", job.job_id, e)
            raise ApiError(
                500, "internal_error", "internal error while creating the job"
            ) from e
        logger.info(
            "[Serve] Job {} created: {} file(s), {} url(s)",
            job.job_id,
            len(documents),
            len(url_entries),
        )
        return {
            "job_id": job.job_id,
            "items": [
                {"item_id": i.item_id, "name": i.name, "kind": i.kind}
                for i in job.items
            ],
        }

    @app.post(
        "/api/jobs/{job_id}/cancel", status_code=202, response_model=CancelResponse
    )
    async def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
        """Stop the job's original items still waiting for a conversion slot.

        Items already converting finish; queued retries keep their own
        lifecycle. Each stopped item ends as an error with ``error_code``
        ``cancelled`` and stays retryable.
        """
        state = _state(request)
        if state.registry.closing:
            raise ApiError(503, "shutting_down", "server is shutting down")
        job = _get_job(request, job_id)
        if job.status != "running":
            raise ApiError(409, "job_not_running", "job is not running")
        waiting = [
            item
            for item in job.items
            if item.status == "queued"
            and item.operation == "convert"
            and item.item_id not in job.retry_pending
        ]
        if not waiting:
            raise ApiError(409, "nothing_to_stop", "no queued items to stop")
        job.stop_requested = True
        for item in waiting:
            task = job.waiting.get(item.item_id)
            if task is not None and not task.done():
                task.cancel()
        logger.info("[Serve] Job {} stopping {} queued item(s)", job_id, len(waiting))
        return {"job_id": job_id, "stopping": len(waiting)}

    @app.post(
        "/api/jobs/{job_id}/items/{item_id}/retry",
        status_code=202,
        response_model=CreateJobResponse,
    )
    async def retry_job_item(
        request: Request,
        job_id: str,
        item_id: str,
        body: JobRetryBody | None = None,
    ) -> dict[str, Any]:
        """Queue a terminal item to retry or explicitly enhance with LLM."""
        state = _state(request)
        if state.registry.closing:
            raise ApiError(503, "shutting_down", "server is shutting down")
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        if item.skip_reason == PENDING_BATCH_SKIP_REASON:
            # Enhancing it here would pay for the same enhancement twice.
            raise ApiError(
                409,
                "batch_pending",
                "provider batch enhancement is pending; collect that batch "
                "(markitai --llm-batch-collect) before retrying or enhancing "
                "this item",
            )
        if item.status not in ("done", "error") or item_id in job.retry_pending:
            raise ApiError(
                409,
                "item_busy",
                "item has not reached a terminal state yet; retry when done",
            )
        if not item.retryable:
            # A CLI run (--record-history) keeps only the outputs: there is
            # no original under uploads/ to convert again.
            raise ApiError(
                409,
                "not_retryable",
                "file items recorded from a CLI run cannot be retried or "
                "enhanced here; run the markitai CLI on the file again",
            )

        operation = body.operation if body is not None else "retry"
        explicit_options = body is not None and body.options is not None
        if body is not None and body.options is not None:
            opts = body.options
        else:
            saved_options = item.options if item.options is not None else job.options
            # A saved selection never carries consent (it is not stored).
            opts = JobOptions.model_validate(
                {
                    key: value
                    for key, value in saved_options.items()
                    if key in JobOptions.model_fields and key != "remote_processing"
                },
                strict=False,
            )
        _check_preset(state, opts)

        if item.kind == "file":
            upload = kept_upload(job, item)
            # Rehydrated meta is untrusted on-disk state; never follow a
            # tampered name outside the job's upload directory.
            if upload is None:
                raise ApiError(
                    404,
                    "upload_missing",
                    f"original upload '{item.name}' is no longer on disk (job "
                    "files may have been cleaned up)",
                )
            item.source = upload
        elif item.kind == "url":
            from markitai.utils.cli_helpers import url_to_filename

            problem = _url_problem(item.name)
            if problem is not None:
                raise ApiError(422, problem, "URLs must be absolute http(s) addresses")
            # Same gate as job creation: a rerun re-fetches item.name, and
            # rehydrated meta is untrusted on-disk state.
            await _guard_fetch_targets(request, [item.name])
            item.source = item.name
            item.output_name = item.output_name or url_to_filename(item.name)
        else:
            raise ApiError(409, "no_source", "item has no supported original source")

        trusted = _is_trusted_request(request)
        _check_remote_processing(state, opts, trusted=trusted)
        # A user's job reruns over that user's pool, whoever asks (a revoked
        # user's: the shared models only)
        conversion = await conversion_of(state, job.owner)
        # The awaits above (a user's pool is read on a thread) let a delete of
        # this job or item land: it must not be converted again, paid for, and
        # written into a folder that is gone. Taking the lock a delete holds,
        # and looking again, leaves no await between the check and the queue.
        async with job.delete_lock:
            if state.registry.get(job_id) is not job:
                raise ApiError(404, "job_not_found", "job not found")
            item = _get_item(job, item_id)
            if item.status not in ("done", "error") or item_id in job.retry_pending:
                raise ApiError(
                    409,
                    "item_busy",
                    "item has not reached a terminal state yet; retry when done",
                )
            # Options inherited from the item were the caller's earlier choice;
            # a retry that names model processing now needs a model now.
            if explicit_options and operation == "retry":
                _require_model(conversion, opts)
            cfg = job_config(conversion, opts)
            if operation == "enhance" and (
                not cfg.llm.enabled or not conversion.routable
            ):
                raise ApiError(
                    409,
                    "llm_unavailable",
                    "LLM enhancement is unavailable; enable a routable LLM first",
                )
            job.public_network_only = job.public_network_only or not trusted
            job.hidden_hosts = tuple(
                dict.fromkeys((*job.hidden_hosts, *conversion.hidden_hosts))
            )
            cfg.output.dir = str(job.out_dir)
            # The rerun reuses the job's out_dir, so it would otherwise hit
            # on_conflict against its own prior output (rename → duplicate
            # versioned files; skip → the enhance can never run). Always overwrite
            # the item's own outputs; a failed rerun rolls back (RetryWork.prior_*).
            cfg.output.on_conflict = "overwrite"
            job.cfg = cfg
            if operation != "enhance":
                job.options = _saved_options(opts)
                item.options = _saved_options(opts)

            # Snapshot the previous successful (non-skipped) result so a failed
            # rerun can be rolled back to it instead of destroying a done output.
            protect = (
                item.status == "done" and item.output is not None and not item.skipped
            )
            work = RetryWork(
                item_id=item_id,
                cfg=cfg,
                operation=operation,
                prior_output=item.output if protect else None,
                prior_cost_usd=item.cost_usd if protect else None,
                prior_duration_ms=item.duration_ms if protect else None,
                prior_finished_at=item.finished_at if protect else None,
                prior_operation=item.operation if protect else "convert",
                prior_llm_enhanced=item.llm_enhanced if protect else False,
                prior_warnings=list(item.warnings) if protect else [],
                prior_pricing=item.pricing if protect else None,
                prior_models=list(item.models) if protect else [],
            )

            # Reset only this item. The job keeps its identity and item ordering;
            # the serial worker prevents rapid retries from racing output writes.
            # meta.json is left intact: an unclean shutdown mid-rerun then
            # rehydrates the job at its last persisted state rather than losing it.
            item.status = "queued"
            item.error = None
            item.error_code = None
            item.output = None
            item.duration_ms = None
            item.finished_at = None
            item.cost_usd = None
            item.pricing = None
            item.models = []
            item.llm_enhanced = False
            item.operation = operation
            item.skipped = False
            item.skip_reason = None
            item.warnings = []
            job.status = "running"
            job.generation += 1
            job.finished_at = None
            job.retry_pending.add(item_id)
            job.retry_queue.put_nowait(work)
            state.registry.publish_item(job, item)
            state.registry.publish_job(job)

            # A dedicated drainer, independent of the initial run: retrying an item
            # while sibling items are still converting must start the worker even
            # though job.task (the initial run) is not yet done.
            if job.retry_task is None or job.retry_task.done():
                job.retry_task = asyncio.create_task(
                    run_retry_queue(state.registry, job)
                )
            logger.info(
                "[Serve] Job {} queued {} for item {}", job_id, operation, item_id
            )
            return {
                "job_id": job_id,
                "items": [
                    {"item_id": item.item_id, "name": item.name, "kind": item.kind}
                ],
            }

    @app.delete("/api/jobs/{job_id}/items/{item_id}", status_code=204)
    async def delete_job_item(request: Request, job_id: str, item_id: str) -> None:
        """Permanently remove one terminal ledger item and all its files.

        Its markdown pair, its original upload and the assets/screenshots it
        owns (``serve.artifacts``) go with it; files a remaining sibling
        also claims are kept.
        """
        await _delete_item(request, job_id, item_id)

    async def _delete_item(request: Request, job_id: str, item_id: str) -> None:
        """``DELETE /api/jobs/{job_id}/items/{item_id}``'s work (also batched)."""
        state = _state(request)
        job = _get_job(request, job_id)
        async with job.delete_lock:
            # Re-resolved under the lock: a concurrent delete may already
            # have removed this row, or the job with its last row. A repeat
            # delete is then a plain 404, never a 500 or an empty job.
            if state.registry.get(job_id) is not job:
                raise ApiError(404, "job_not_found", "job not found")
            item = _get_item(job, item_id)
            if job.status == "running" or item.status not in ("done", "error"):
                raise ApiError(
                    409, "job_running", "job is still running; retry when done"
                )

            if len(job.items) == 1:
                await asyncio.to_thread(state.registry.discard_job, job)
                logger.info("[Serve] Job {} deleted with its last item", job_id)
                return

            doomed = await asyncio.to_thread(_item_owned_files, job, item)
            job.items.remove(item)
            await asyncio.to_thread(_unlink_all, [*doomed, job.job_dir / "archive.zip"])
            await asyncio.to_thread(remove_empty_shots_dir, job, item)
            await asyncio.to_thread(write_job_meta, job)
            state.registry.publish_job(job)
            logger.info("[Serve] Job {} item {} deleted", job_id, item_id)

    @app.get("/api/jobs/{job_id}", response_model=JobSnapshot)
    async def get_job(request: Request, job_id: str) -> dict[str, Any]:
        return _get_job(request, job_id).snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def get_job_events(request: Request, job_id: str) -> StreamingResponse:
        job = _get_job(request, job_id)
        registry = _state(request).registry
        # A stream lives only as long as the token it was opened with
        generation = request.app.state.token_generation

        def revoked() -> bool:
            return request.app.state.token_generation != generation

        async def stream() -> AsyncIterator[str]:
            queue = registry.subscribe(job)
            try:
                if revoked():
                    return
                yield _sse_frame("snapshot", job.snapshot())
                if registry.closing:
                    # Shutting down: a stream opened now would hold the
                    # server open until the job ends (see begin_shutdown).
                    return
                if job.status != "running":
                    # The job may have reached its terminal state while the
                    # snapshot frame was being sent: drain events already
                    # queued for this subscriber (late item output/error/
                    # duration updates) before the final `job` frame.
                    while not queue.empty():
                        event, data = queue.get_nowait()
                        if event != SHUTDOWN_EVENT:
                            yield _sse_frame(event, data)
                    yield _sse_frame("job", job.progress_payload())
                    return
                while True:
                    try:
                        event, data = await asyncio.wait_for(
                            queue.get(), timeout=_SSE_PING_INTERVAL
                        )
                    except TimeoutError:
                        if revoked():
                            return
                        yield ": ping\n\n"
                        continue
                    if event == SHUTDOWN_EVENT or revoked():
                        return
                    yield _sse_frame(event, data)
                    if event == "job" and data.get("status") != "running":
                        return
            finally:
                registry.unsubscribe(job, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/api/jobs/{job_id}/items/{item_id}/result",
        response_model=ItemResult,
    )
    async def get_item_result(
        request: Request, job_id: str, item_id: str, max_bytes: str | None = None
    ) -> dict[str, Any]:
        """The item's selected Markdown, its variant and every artifact.

        ``max_bytes`` (at least 1024) returns an excerpt instead: at most
        that many bytes, cut at a line end when one is near and never inside
        a UTF-8 sequence, with ``truncated`` and the full ``markdown_bytes``.
        """
        limit: int | None = None
        if max_bytes is not None:
            if not _small_number(max_bytes, 20) or int(max_bytes) < 1024:
                raise ApiError(
                    422,
                    "invalid_max_bytes",
                    "max_bytes must be an integer of at least 1024",
                )
            # Never more than a whole result may be: the cap is the same
            limit = min(int(max_bytes), MAX_RESULT_BYTES)
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        located = _item_result_files(job, item)
        if located is None:
            raise ApiError(404, "result_unavailable", "item result not available")
        out_dir, base_name, base_path, llm_path, variant, markdown_path = located

        def collect() -> tuple[str, int, bool, list[dict[str, Any]]]:
            artifacts = [
                {
                    "relpath": path.relative_to(out_dir).as_posix(),
                    "size": path.stat().st_size,
                }
                for path in _item_artifacts(
                    item, out_dir, base_name, base_path, llm_path
                )
            ]
            total = markdown_path.stat().st_size
            if limit is None and total > MAX_RESULT_BYTES:
                raise ApiError(
                    413,
                    "result_too_large",
                    "result is too large for JSON; use the file download endpoint",
                )
            with markdown_path.open("rb") as fh:
                data = fh.read(limit + 1 if limit is not None else -1)
            text, truncated = _markdown_excerpt(data, limit)
            return text, total, truncated, artifacts

        markdown, total, truncated, artifacts = await asyncio.to_thread(collect)
        return {
            "name": item.name,
            "variant": variant,
            "markdown": markdown,
            "artifacts": artifacts,
            "truncated": truncated,
            "markdown_bytes": total,
        }

    @app.get("/api/jobs/{job_id}/items/{item_id}/assets/{relpath:path}")
    async def get_item_asset(
        request: Request, job_id: str, item_id: str, relpath: str
    ) -> Response:
        """One of the item's own extracted images, inline, for the page.

        Contract v1.6 section 6.2: only a raster image ``…/result`` lists
        among this item's artifacts, inside the job's out dir and not
        through a symlink; anything else is 404 ``file_not_found``.
        """
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        missing = ApiError(404, "file_not_found", "file not found")
        media_type = _ITEM_IMAGE_TYPES.get(Path(relpath).suffix.lower())
        segments = relpath.split("/")
        if (
            media_type is None
            or relpath.startswith("/")
            or "\\" in relpath
            or "\0" in relpath
            or any(segment in ("", ".", "..") for segment in segments)
        ):
            raise missing
        located = _item_result_files(job, item)
        if located is None:
            raise missing
        out_dir, base_name, base_path, llm_path, _variant, _markdown = located

        def read() -> tuple[bytes, str] | None:
            owned = {
                path.relative_to(out_dir).as_posix()
                for path in _item_assets(item, out_dir, base_name, base_path, llm_path)
            }
            candidate = out_dir / relpath
            if relpath not in owned or candidate.is_symlink():
                return None
            target = candidate.resolve()
            if not target.is_relative_to(out_dir) or not target.is_file():
                return None
            stat = target.stat()
            return target.read_bytes(), f'W/"{stat.st_size:x}-{stat.st_mtime_ns:x}"'

        found = await asyncio.to_thread(read)
        if found is None:
            raise missing
        data, etag = found
        headers = {
            "Content-Disposition": (
                f"inline; filename*=UTF-8''{quote(segments[-1], safe='')}"
            ),
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Cache-Control": "private, no-cache",
            "ETag": etag,
        }
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        return Response(content=data, media_type=media_type, headers=headers)

    @app.get("/api/jobs/{job_id}/files/{relpath:path}")
    async def get_job_file(request: Request, job_id: str, relpath: str) -> FileResponse:
        job = _get_job(request, job_id)
        out_dir = job.out_dir.resolve()
        target = (out_dir / relpath).resolve()
        if not target.is_relative_to(out_dir) or not target.is_file():
            raise ApiError(404, "file_not_found", "file not found")
        return FileResponse(
            target,
            filename=target.name,
            content_disposition_type="attachment",
            # Markdown is text to every browser; never sniffed into HTML
            media_type="text/plain; charset=utf-8"
            if target.suffix.lower() == ".md"
            else None,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/jobs/{job_id}/archive")
    async def get_job_archive(request: Request, job_id: str) -> FileResponse:
        job = _get_job(request, job_id)
        if job.status == "running":
            raise ApiError(409, "job_running", "job is still running; retry when done")
        archive_path = job.job_dir / "archive.zip"

        def build() -> None:
            # Unique temp path + os.replace: readers never observe a
            # partially written zip. ".<name>.<rand>.tmp" droppings from
            # atomic_write_text are excluded from the walk.
            tmp_path = archive_path.with_name(f".archive.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    _add_job_outputs_to_zip(zf, job)
                os.replace(tmp_path, archive_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        async with job.archive_lock:  # serialize concurrent builds per job
            await asyncio.to_thread(build)
        return FileResponse(
            archive_path,
            filename=f"markitai-{job.job_id}.zip",
            media_type="application/zip",
            content_disposition_type="attachment",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/jobs/{job_id}/items/{item_id}/archive")
    async def get_item_archive(
        request: Request, job_id: str, item_id: str
    ) -> FileResponse:
        """A ZIP of one item: its result's files and its kept original.

        Every file ``…/result`` lists in ``artifacts``, at those relpaths,
        plus the original ``…/original/file`` sends, at
        ``original/<its file name>`` (contract v1.11 section 6.7).
        """
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        if item.status in ("queued", "running") or item_id in job.retry_pending:
            raise ApiError(
                409, "item_busy", "the item is being converted; retry when done"
            )
        located = _item_result_files(job, item)
        if located is None:
            raise ApiError(404, "result_unavailable", "item result not available")
        out_dir, base_name, base_path, llm_path, _variant, _markdown = located
        archive_path = job.job_dir / f".item-archive.{uuid.uuid4().hex}.zip"

        def build() -> None:
            files = _item_artifacts(item, out_dir, base_name, base_path, llm_path)
            original = kept_original(job, item)
            tmp_path = job.job_dir / f".archive.{uuid.uuid4().hex}.tmp"
            try:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for path in files:
                        with suppress(OSError):  # removed by a concurrent rerun
                            zf.write(path, path.relative_to(out_dir).as_posix())
                    if original is not None:
                        path, name = original
                        with suppress(OSError):
                            zf.write(path, f"original/{name}")
                os.replace(tmp_path, archive_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        async with job.archive_lock:
            await asyncio.to_thread(build)
        return FileResponse(
            archive_path,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(f"{base_name}.zip", safe=""),
                "X-Content-Type-Options": "nosniff",
            },
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    # ------------------------------------------------------------- originals

    def _original_paths(job: Job, item: JobItem) -> str:
        return f"/api/jobs/{job.job_id}/items/{item.item_id}/original/file"

    def _original_view_path(job: Job, item: JobItem) -> str:
        return f"/api/jobs/{job.job_id}/items/{item.item_id}/original/view"

    def _original_blocks_path(job: Job, item: JobItem) -> str:
        return f"/api/jobs/{job.job_id}/items/{item.item_id}/original/blocks"

    async def _describe(
        request: Request, job: Job, item: JobItem, *, boxes: bool
    ) -> Original:
        """The item's original, cached until the item is converted again."""
        cache: dict[tuple[Any, ...], Original] = request.app.state.original_cache
        key = (job.job_id, item.item_id, item.finished_at, item.output, boxes)
        cached = cache.get(key)
        if cached is not None:
            return cached
        workers: PdfWorkers = request.app.state.pdf_workers

        def layout(path: Path, with_boxes: bool) -> tuple[list[Any], int]:
            return workers.run(pdf_pages, path, with_boxes)

        original = await asyncio.to_thread(
            describe_original, job, item, boxes=boxes, pdf_layout=layout
        )
        if len(cache) >= 512:
            cache.pop(next(iter(cache)))
        cache[key] = original
        return original

    @app.get(
        "/api/jobs/{job_id}/items/{item_id}/original",
        response_model=OriginalDescriptor,
    )
    async def get_item_original(
        request: Request, job_id: str, item_id: str
    ) -> dict[str, Any]:
        """What the Compare view can show of the item's source."""
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        original = await _describe(request, job, item, boxes=True)
        return original.descriptor(
            _original_paths(job, item),
            _original_view_path(job, item),
            _original_blocks_path(job, item),
        )

    @app.get("/api/jobs/{job_id}/items/{item_id}/original/blocks")
    async def get_item_original_blocks(
        request: Request, job_id: str, item_id: str
    ) -> Response:
        """Where the page's blocks are drawn on its screenshots, as recorded
        when the browser took them (contract v1.14 section 7)."""
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        unavailable = ApiError(404, "original_unavailable", "no block boxes are kept")
        path = kept_blocks(job, item)
        if path is None:
            raise unavailable

        def read() -> bytes:
            with path.open("rb") as fh:
                return fh.read(MAX_BLOCKS_BYTES + 1)

        try:
            data = await asyncio.to_thread(read)
        except OSError:
            raise unavailable from None
        if len(data) > MAX_BLOCKS_BYTES:
            raise unavailable
        return Response(
            content=data,
            media_type="application/json",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/jobs/{job_id}/items/{item_id}/original/file")
    async def get_item_original_file(
        request: Request, job_id: str, item_id: str
    ) -> FileResponse:
        """The stored original bytes, never rendered by a direct visit."""
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        kept = kept_original(job, item)
        if kept is None:
            raise ApiError(404, "original_unavailable", "no original is kept")
        path, name = kept
        media_type = original_media_type(path)
        if media_type in ("text/html", "application/xhtml+xml", "image/svg+xml"):
            media_type = "text/plain; charset=utf-8"
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(name, safe=""),
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox; default-src 'none'",
            },
        )

    @app.get("/api/jobs/{job_id}/items/{item_id}/original/view")
    async def get_item_original_view(
        request: Request, job_id: str, item_id: str
    ) -> Response:
        """The kept HTML original for the workbench's sandboxed frame.

        One insertion (no DNS prefetching, a base that opens links outside
        the frame) and a policy under which the page runs no script and
        sends no form; its pictures, style sheets, fonts and media load
        over HTTPS (contract v1.12 section 7).
        """
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        original = await _describe(request, job, item, boxes=False)
        unavailable = ApiError(404, "original_unavailable", "no HTML original is kept")
        if original.kind != "html" or original.path is None:
            raise unavailable
        too_large = ApiError(
            413,
            "result_too_large",
            "the page is too large to show; download the original instead",
        )
        if (original.size or 0) > MAX_SNAPSHOT_BYTES:
            raise too_large
        path = original.path

        def read() -> bytes:
            with path.open("rb") as fh:
                return fh.read(MAX_SNAPSHOT_BYTES + 1)

        try:
            data = await asyncio.to_thread(read)
        except OSError:
            raise unavailable from None
        if len(data) > MAX_SNAPSHOT_BYTES:
            raise too_large
        # A job read back from disk keeps no source: the URL is its name
        page = framed_page(data.decode("utf-8", "replace"), item_url(item))
        return Response(
            content=page.encode("utf-8"),
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Content-Security-Policy": VIEW_POLICY,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-DNS-Prefetch-Control": "off",
                "Cache-Control": "no-store",
                "Content-Disposition": "inline",
            },
        )

    async def _page_request(
        request: Request, job_id: str, item_id: str, page: str
    ) -> tuple[Any, int]:
        if not _small_number(page, 6):
            raise ApiError(422, "invalid_page_request", "page must be a number")
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        original = await _describe(request, job, item, boxes=False)
        number = int(page)
        if original.kind in ("none", "text") or (
            original.kind == "html" and not original.screenshots
        ):
            raise ApiError(
                404, "original_unavailable", "this original has no page images"
            )
        if not 1 <= number <= original.page_count:
            raise ApiError(404, "page_not_found", "no such page")
        return original, number

    @app.get("/api/jobs/{job_id}/items/{item_id}/original/pages/{page}")
    async def get_item_original_page(
        request: Request, job_id: str, item_id: str, page: str, width: str | None = None
    ) -> Response:
        """Page *page* (1-based) as an image, *width* pixels wide."""
        if width is not None and not _small_number(width, 6):
            raise ApiError(422, "invalid_page_request", "width must be a number")
        target = clamp_width(int(width) if width is not None else None)
        original, number = await _page_request(request, job_id, item_id, page)
        source = (
            original.screenshots[number - 1]
            if original.kind in ("screenshots", "html")
            else original.path
        )
        assert source is not None
        try:
            stat = source.stat()
        except OSError:
            raise ApiError(
                404, "original_unavailable", "the original is gone"
            ) from None
        cache: ByteCache = request.app.state.page_cache
        cache_key = (str(source), stat.st_mtime_ns, stat.st_size, number, target)
        cached = cache.get(cache_key)
        if cached is not None:
            return _page_image(*cached)
        renders: asyncio.Semaphore = request.app.state.page_renders
        if renders.locked():
            raise ApiError(
                429,
                "render_busy",
                "other pages are being drawn; try again in a moment",
                headers={"Retry-After": "1"},
            )
        async with renders:
            try:
                if original.kind == "pdf":
                    workers: PdfWorkers = request.app.state.pdf_workers
                    body = await asyncio.to_thread(
                        workers.run, render_pdf_page, source, number, target
                    )
                    media_type = "image/png"
                else:
                    body, media_type = await asyncio.to_thread(
                        render_image, source, target
                    )
            except IndexError:
                raise ApiError(404, "page_not_found", "no such page") from None
            except RenderError as e:
                logger.debug("[Serve] Page {} of {} not drawn: {}", number, item_id, e)
                raise ApiError(
                    422, "render_failed", "this page cannot be drawn"
                ) from None
        cache.put(cache_key, (body, media_type))
        return _page_image(body, media_type)

    def _page_image(body: bytes, media_type: str) -> Response:
        return Response(
            content=body,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=86400",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox; default-src 'none'",
            },
        )

    @app.get(
        "/api/jobs/{job_id}/items/{item_id}/original/pages/{page}/text",
        response_model=PageText,
    )
    async def get_item_original_page_text(
        request: Request, job_id: str, item_id: str, page: str
    ) -> dict[str, Any]:
        """A PDF page's text as positioned lines (top-left origin, points)."""
        original, number = await _page_request(request, job_id, item_id, page)
        if not original.text_layer:
            raise ApiError(
                404, "original_unavailable", "this original has no text layer"
            )
        try:
            workers: PdfWorkers = request.app.state.pdf_workers
            return await asyncio.to_thread(
                workers.run, pdf_page_text, original.path, number
            )
        except IndexError:
            raise ApiError(404, "page_not_found", "no such page") from None
        except RenderError:
            raise ApiError(422, "render_failed", "this page cannot be read") from None

    @app.get(
        "/api/jobs/{job_id}/items/{item_id}/original/outline",
        response_model=PdfOutline,
    )
    async def get_item_original_outline(
        request: Request, job_id: str, item_id: str
    ) -> dict[str, Any]:
        """A PDF's own bookmarks: ``{items: [{title, page, level}], truncated}``
        (contract v1.17); empty when it has none or they cannot be read."""
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        original = await _describe(request, job, item, boxes=False)
        if original.kind != "pdf" or original.path is None:
            raise ApiError(404, "original_unavailable", "this original is not a PDF")
        workers: PdfWorkers = request.app.state.pdf_workers
        try:
            return await asyncio.to_thread(workers.run, pdf_outline, original.path)
        except Exception:
            return {"items": [], "truncated": False}

    # ---------------------------------------------------------- downloads

    @app.post("/api/download-tickets", status_code=201, response_model=DownloadTicket)
    async def create_download_ticket(request: Request) -> dict[str, Any]:
        """A single-use, 60-second ticket for one download path.

        A browser download is a navigation, which cannot carry the
        ``Authorization`` header; the ticket rides in its URL instead of the
        access token and is worthless once used or expired.
        """
        invalid = ApiError(
            422,
            "invalid_ticket_path",
            'a download ticket needs a JSON body {"path": ...} naming a job '
            "file, a job archive, an item original or the history archive",
        )
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > _TICKET_BODY_LIMIT:
                raise ApiError(
                    413, "request_too_large", "a download ticket request exceeds 8 KiB"
                )
        try:
            body = DownloadTicketRequest.model_validate_json(bytes(data))
        except ValidationError:
            raise invalid from None
        if not downloadable(body.path):
            raise invalid
        caller = _caller(request)
        # A user may name only its own jobs
        named_job = _path_job("GET", body.path)
        if caller.user_id is not None and named_job is not None:
            _get_job(request, named_job)
        tickets: DownloadTickets = request.app.state.download_tickets
        ticket = tickets.issue(body.path, user=caller.user_id)
        if ticket is None:
            raise ApiError(
                429,
                "too_many_tickets",
                "too many download tickets are outstanding; use or let them "
                "expire first",
            )
        return {
            "ticket": ticket,
            "url": f"{body.path}?ticket={ticket}",
            "expires_in": TICKET_LIFETIME_S,
        }

    # --------------------------------------------------------- access token

    @app.post(_TOKEN_REGENERATE_PATH, response_model=RegeneratedToken)
    async def regenerate_access_token(request: Request) -> JSONResponse:
        """Replace the access token with a new random one.

        Trusted as the settings routes are (``protect_local_settings``). The
        old token stops working at once: requests presenting it are 401, the
        event streams opened under it end, and outstanding download tickets
        are revoked. The new token lives in memory, as the startup one did.
        """
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > _TOKEN_REGENERATE_BODY_LIMIT:
                raise ApiError(
                    413,
                    "request_too_large",
                    "a token regeneration request exceeds 1 KiB",
                )
        if data.strip():
            try:
                body = json.loads(bytes(data))
            except ValueError:
                body = None
            if body != {}:
                raise ApiError(
                    422,
                    "invalid_json",
                    "regenerating the token takes no body, or an empty JSON object",
                )
        if request.app.state.serve_token is None:
            raise ApiError(
                409,
                "token_disabled",
                "this server runs without an access token (--no-auth), so "
                "there is none to regenerate",
            )
        token = new_access_token()
        request.app.state.revoked_tokens.add(
            _token_digest(request.app.state.serve_token)
        )
        request.app.state.serve_token = token
        request.app.state.token_generation += 1
        tickets: DownloadTickets = request.app.state.download_tickets
        tickets.clear()
        _state(request).registry.end_streams()
        logger.info(
            "[Serve] Access token regenerated from the web workbench; other tabs "
            "and devices must sign in again."
        )
        origin = network_origin() if network_origin is not None else None
        return JSONResponse(
            {
                "token": token,
                "network_url": (
                    f"{origin}/#token={quote(token, safe='')}" if origin else None
                ),
                "persistent": False,
                "startup_variable": token_variable,
            },
            headers={"Cache-Control": "no-store"},
        )

    # -------------------------------------------------------- user tokens

    def _require_auth(request: Request) -> None:
        if request.app.state.serve_token is None:
            raise ApiError(
                409,
                "token_disabled",
                "the server runs without an access token (--no-auth); user "
                "tokens are unavailable",
            )

    @app.get(_ACCESS_TOKENS_PATH, response_model=AccessTokenList)
    async def list_access_tokens(request: Request) -> JSONResponse:
        """The user tokens, oldest first (never the tokens themselves)."""
        _require_auth(request)

        def listed() -> dict[str, Any]:
            tokens = access.list()
            for token in tokens["tokens"]:
                token["model_count"] = users.count(token["id"])
            return tokens

        return JSONResponse(
            await asyncio.to_thread(listed), headers={"Cache-Control": "no-store"}
        )

    @app.post(_ACCESS_TOKENS_PATH, status_code=201, response_model=CreatedAccessToken)
    async def create_access_token(request: Request) -> JSONResponse:
        """A new user token named by ``{"label": …}``, returned this once."""
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > _ACCESS_TOKEN_BODY_LIMIT:
                raise ApiError(413, "request_too_large", "request exceeds 1 KiB")
        _require_auth(request)
        try:
            body = json.loads(bytes(data))
        except ValueError:
            body = None
        if (
            not isinstance(body, dict)
            or list(body) != ["label"]
            or not isinstance(body["label"], str)
        ):
            raise ApiError(422, "invalid_json", 'send a JSON body {"label": "…"}')
        label = clean_label(body["label"])
        if label is None:
            raise ApiError(
                422,
                "invalid_label",
                "label must be plain text of 1 to 60 characters",
            )
        created = await asyncio.to_thread(access.create, label)
        logger.info('[Serve] Access token "{}" created from the web workbench.', label)
        return JSONResponse(
            created, status_code=201, headers={"Cache-Control": "no-store"}
        )

    @app.delete(_ACCESS_TOKENS_PATH + "/{token_id}", status_code=204)
    async def revoke_access_token(request: Request, token_id: str) -> Response:
        """Revoke a user token: it stops working at once.

        Requests presenting it are 401, its download tickets are refused and
        open event streams end (their clients reconnect with the token they
        hold, so only this one's are refused).
        """
        _require_auth(request)
        label = await asyncio.to_thread(access.revoke, token_id)
        tickets: DownloadTickets = request.app.state.download_tickets
        tickets.revoke_user(token_id)
        request.app.state.token_generation += 1
        _state(request).registry.end_streams()
        # Its own models go with it (a write racing this finds it gone);
        # conversions already running finish with the models they started with
        try:
            count = await asyncio.to_thread(users.delete, token_id)
        except OSError as e:
            count = 0
            logger.warning(
                "[Serve] Could not delete the models of a revoked access token "
                "({}); the next start deletes them.",
                type(e).__name__,
            )
        if count:
            logger.info(
                '[Serve] Access token "{}" revoked from the web workbench; {}.',
                label,
                "its 1 model was deleted"
                if count == 1
                else f"its {count} models were deleted",
            )
        else:
            logger.info(
                '[Serve] Access token "{}" revoked from the web workbench.', label
            )
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    # -------------------------------------------------------------- history

    @app.get(
        "/api/history",
        response_model=list[HistoryEntry],
        # options/items only with include=items
        response_model_exclude_unset=True,
    )
    async def get_history(
        request: Request, include: str | None = None
    ) -> list[dict[str, Any]]:
        """Saved jobs, newest first; ``include=items`` adds every item."""
        state = _state(request)
        registry = state.registry
        with_items = "items" in (include or "").split(",")
        # Pick up jobs written since startup by other processes — local CLI
        # conversions recorded with --record-history land in the same jobs
        # root. Runs on the loop thread on purpose: it only stats/parses
        # directories not yet registered, and keeping registry mutations on
        # the loop preserves the single-writer invariant the read endpoints
        # (which iterate the registry from worker threads) rely on.
        rehydrate_jobs(registry, state.config)
        # The registry already dedupes archived vs this-process terminal jobs:
        # a job finished in this process is one entry, and startup rehydrate
        # never overwrites a live registry entry. A user sees only its own.
        caller = _caller(request)
        jobs = [
            job
            for job in registry.jobs.values()
            if job.status != "running" and not job_expired(job) and caller.sees(job)
        ]

        def entry(job: Job) -> dict[str, Any]:
            value: dict[str, Any] = {
                "job_id": job.job_id,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "status": job.status,
                "total": len(job.items),
                "done": job.done_count,
                "failed": job.failed_count,
                "skipped": job.skipped_count,
                "llm_enhanced": sum(i.llm_enhanced for i in job.items),
                "cost_usd": (
                    sum(i.cost_usd or 0.0 for i in job.items)
                    if any(i.cost_usd is not None for i in job.items)
                    else None
                ),
                "pricing": aggregate_pricing(job.items),
                "names_preview": [i.name for i in job.items[:3]],
                "kinds_preview": [i.kind for i in job.items[:3]],
                "duration_ms": job_duration_ms(job),
                "size_bytes": job_cached_dir_size(job),
                "origin": job.options.get("origin") or "web",
                # A mixed CLI job still offers retry for its URL items
                "retryable": any(i.retryable for i in job.items),
                "label": job.label,
            }
            if with_items:
                value["options"] = public_options(job.options)
                value["items"] = [job.item_payload(i) for i in job.items]
            return value

        entries = await asyncio.to_thread(lambda: [entry(job) for job in jobs])
        # By instant, not text: CLI history records local time with its
        # offset and the server records UTC.
        entries.sort(key=lambda e: _created_instant(e["created_at"]), reverse=True)
        return entries

    @app.get("/api/history/archive")
    async def get_history_archive(request: Request) -> FileResponse:
        """Download every completed job, including rehydrated history.

        A user's archive holds only its own jobs.
        """
        registry = _state(request).registry
        caller = _caller(request)
        jobs = sorted(
            (
                job
                for job in registry.jobs.values()
                if job.status != "running" and not job_expired(job) and caller.sees(job)
            ),
            key=lambda job: _created_instant(job.created_at),
        )
        if not jobs:
            raise ApiError(404, "history_empty", "no completed jobs to archive")

        # A response owns its archive until streaming and cleanup finish.
        # Sharing a path lets one download remove another response's file.
        archive_path = registry.jobs_root / f"archive-{uuid.uuid4().hex}.zip"

        def build() -> None:
            tmp_path = archive_path.with_name(f".archive.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    nested = len(jobs) > 1
                    used_folders: set[str] = set()
                    for job in jobs:
                        prefix: Path | None = None
                        if nested:
                            base = _job_archive_folder(job)
                            folder = base
                            counter = 2
                            while folder.casefold() in used_folders:
                                folder = f"{base} ({counter})"
                                counter += 1
                            used_folders.add(folder.casefold())
                            prefix = Path(folder)
                        _add_job_outputs_to_zip(zf, job, prefix=prefix)
                os.replace(tmp_path, archive_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        async with registry.archive_lock:
            await asyncio.to_thread(build)
        # Each response removes only its own completed archive.
        return FileResponse(
            archive_path,
            filename="markitai-all.zip",
            media_type="application/zip",
            content_disposition_type="attachment",
            headers={"X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    @app.post("/api/history/delete", response_model=BatchDeleteResult)
    async def delete_history_batch(request: Request) -> JSONResponse:
        """Delete several jobs and items in one request (contract v1.13 §6.8).

        Body exactly ``{jobs?: [job_id], items?: [{job_id, item_id}]}``, 1 to
        1000 entries in all. Each entry, in order (``jobs``, then ``items``),
        does exactly what ``DELETE /api/history/{job_id}`` or ``DELETE
        /api/jobs/{job_id}/items/{item_id}`` does, a user's only on its own
        jobs; nothing is rolled back. 200 with every entry's outcome.
        """
        content_type = request.headers.get("content-type", "")
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > _BATCH_DELETE_BODY_LIMIT:
                raise ApiError(413, "request_too_large", "request exceeds 256 KiB")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise ApiError(
                422, "json_required", "a batch deletion must use application/json"
            )
        try:
            body = json.loads(bytes(data))
        except ValueError:
            raise ApiError(
                422, "invalid_json", "the request is not valid JSON"
            ) from None
        entries = _delete_entries(body)
        results: list[dict[str, Any]] = []
        for job_id, item_id in entries:
            result: dict[str, Any] = {"job_id": job_id, "item_id": item_id}
            try:
                if item_id is None:
                    await _delete_job(request, job_id)
                else:
                    await _delete_item(request, job_id, item_id)
            except ApiError as e:
                result.update(ok=False, reason=e.reason, detail=e.detail)
            else:
                result["ok"] = True
            results.append(result)
        deleted = sum(result["ok"] for result in results)
        return JSONResponse(
            {
                "deleted": deleted,
                "failed": len(results) - deleted,
                "results": results,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/api/history/{job_id}", status_code=204)
    async def delete_history_job(request: Request, job_id: str) -> None:
        await _delete_job(request, job_id)

    async def _delete_job(request: Request, job_id: str) -> None:
        """``DELETE /api/history/{job_id}``'s work (also batched)."""
        registry = _state(request).registry
        job = _get_job(request, job_id)
        async with job.delete_lock:
            if registry.get(job_id) is not job:
                raise ApiError(404, "job_not_found", "job not found")
            if job.status == "running":
                raise ApiError(
                    409, "job_running", "job is still running; retry when done"
                )
            await asyncio.to_thread(registry.discard_job, job)
        logger.info("[Serve] History job {} deleted", job_id)

    # --------------------------------------------------------------- static

    bundle_dir = static_dir if static_dir is not None else detect_static_dir()
    workbench = Workbench.load(bundle_dir) if bundle_dir is not None else None
    app.state.workbench = workbench
    if workbench is not None:
        mount_workbench(app, workbench)
    else:

        @app.get("/", response_model=RootInfo)
        async def root_hint() -> dict[str, str]:
            return {"markitai": __version__, "hint": ROOT_HINT}

    return app

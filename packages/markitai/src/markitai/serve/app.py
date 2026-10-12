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
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from loguru import logger
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from markitai import __version__
from markitai.constants import PROVIDER_API_KEY_ENV
from markitai.providers.discovery import provider_default_api_base
from markitai.runs.history import DEFAULT_SERVE_JOBS_ROOT, PENDING_BATCH_SKIP_REASON
from markitai.serve import model_test
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
    ByteCache,
    Original,
    PdfWorkers,
    RenderError,
    clamp_width,
    kept_upload,
    pdf_page_text,
    pdf_pages,
    render_image,
    render_pdf_page,
)
from markitai.serve.originals import describe as describe_original
from markitai.serve.originals import media_type_of as original_media_type
from markitai.serve.schemas import (
    CancelResponse,
    Capabilities,
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
    ProviderConnectionList,
    RootInfo,
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

    from markitai.config import LLMProviderConfig, MarkitaiConfig, ModelConfig

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


class _TokenAuthMiddleware:
    """Reject non-loopback API requests that do not present the serve token.

    Loopback peers stay trusted without a token (the CLI, tests and
    ``--record-history`` tooling on the operator's machine keep working
    unchanged), except on the LLM settings routes, which
    ``protect_local_settings`` gates on the token for every peer. The static
    app shell stays reachable so a browser can load the UI from the tokened
    URL — only ``/api/`` carries anything worth protecting. A ``token`` of
    None disables the gate entirely (``--no-auth``): remote peers then may
    not submit URLs at all (``_guard_fetch_targets``) nor reach the settings
    (the loopback-only settings middleware).

    A ``?ticket=`` (see :mod:`markitai.serve.tickets`) stands in for the
    token on exactly one GET of the path it was issued for; presenting it
    spends it, and a wrong, spent or expired one is refused for every peer.
    """

    def __init__(self, app: Any, token: str | None, tickets: DownloadTickets) -> None:
        self.app = app
        self.token = token
        self.tickets = tickets

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope["type"] == "http"
            and self.token is not None
            and scope.get("path", "").startswith("/api/")
            and not _scope_presents_token(scope, self.token)
        ):
            ticket = _scope_query_param(scope, "ticket")
            if ticket is not None:
                if not (
                    self.tickets.redeem(ticket, _scope_raw_path(scope))
                    and scope.get("method", "").upper() == "GET"
                ):
                    response = error_response(
                        401,
                        "ticket_invalid",
                        "download ticket is invalid, expired, already used or "
                        "issued for another path",
                    )
                    await response(scope, receive, send)
                    return
            elif not _is_loopback_scope(scope):
                response = error_response(401, "token_required", _AUTH_REQUIRED_DETAIL)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _state(request: Request) -> ServeState:
    """Return the ServeState initialized by the lifespan."""
    return request.app.state.markitai


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
    routes. With token auth disabled (``--no-auth``) only loopback is
    trusted.
    """
    if _is_loopback_peer(request):
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

    A job past history retention is gone even before the sweep removes it.
    """
    job = _state(request).registry.get(job_id)
    if job is None or job_expired(job):
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


def _require_model(state: ServeState, opts: JobOptions) -> None:
    """Refuse a request for model processing when no model can serve it.

    Before any job exists: the alternative is converting without the model
    the user asked for and reporting success.
    """
    if _requests_llm(opts, state.config) and not _llm_routable(state):
        raise ApiError(
            422,
            "llm_unavailable",
            "this request needs a model, but none is available; add a model in "
            "Settings, or set MODEL and a provider API key",
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
    model: ModelConfig, index: int, *, persisted: bool
) -> dict[str, Any]:
    from markitai.providers import is_local_provider_model

    local = is_local_provider_model(model.litellm_params.model)
    return {
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
    env_keys = {PROVIDER_API_KEY_ENV.get(provider)}
    if row is not None:
        env_keys.add(row.key_variable)
    return any(name and os.environ.get(name) for name in env_keys)


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
            _deployment_view(model, index, persisted=True)
            for index, model in enumerate(state.configured_models)
        ],
        "detected": [
            _deployment_view(model, index, persisted=False)
            for index, model in enumerate(state.detected_models)
        ],
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
    from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_API_KEY_ENV_NAMES
    from markitai.providers.connectable import CONNECTABLE

    row = next((row for row in CONNECTABLE if row.provider == provider), None)
    if row is not None and row.cloud:
        return (), None
    names = PROVIDER_API_KEY_ENV_NAMES.get(provider) or (
        (row.key_variable,) if row is not None and row.key_variable else ()
    )
    return names, PROVIDER_API_KEY_ENV.get(provider) or (names[0] if names else None)


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
#: when it has none but names an address (owner rule, 2026-10-11). Without a
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
    key, as configured by the owner; the subscription runtimes (claude-agent,
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
#: configured by the owner.
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
    server's own cloud login to an address (owner decision, contract v1.9)."""
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
    the owner never paired a key with. A new connection for that server says
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


def _configured_credential_pairs(state: ServeState) -> frozenset[CredentialPair]:
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
        if api_base is None and api_key[4:] in PROVIDER_API_KEY_ENV.values():
            return
        detail = (
            "api_key: env: references from the web API are limited to "
            f"{', '.join(PROVIDER_API_KEY_ENV.values())} with the provider's "
            "default api_base; to use another variable or endpoint, enter the "
            "key itself or set the pair up in the config file"
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


def _environment_key(
    reference: str, provider: str, *, api_key: str | None, api_base: str | None
) -> str:
    """The key reference behind an environment card's id (``env:openai``).

    Cards exist only for the providers whose key variable the server reads
    (``env:openai`` -> ``env:OPENAI_API_KEY``), never for any variable a
    caller names, and that key goes only to its own provider's endpoint: an
    address needs the caller's own key.
    """
    name = reference.removeprefix("env:")
    variable = PROVIDER_API_KEY_ENV.get(name)
    if variable is None:
        raise ApiError(
            422,
            "invalid_settings_request",
            (
                f"unknown environment provider {reference!r}; environment "
                "connections are "
                f"{', '.join(f'env:{known}' for known in PROVIDER_API_KEY_ENV)}"
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


def _credentials_view(
    provider: str, api_key: str | None, api_base: str | None
) -> dict[str, str | None]:
    """One connection's editable values, as the credentials route shows them.

    A connection that stores no key (not the keyless placeholder) is served
    by the server's own key variable when one is set: it shows as
    ``env:<VARIABLE>`` with ``api_key_source: environment`` (Rust's
    ``credentials``), the key its conversions and model tests carry, rather
    than "Not set".
    """
    shown = api_key
    source = None
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
    state: ServeState, body: LLMSettingsUpdate
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
        for m in state.config.llm.model_list
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
    state: ServeState, deployment_id: str
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
    state: ServeState, *, deployment_id: str | None, model_name: str | None
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
        llm = data.get("llm") if isinstance(data, dict) else None
        entries = llm.get("model_list") if isinstance(llm, dict) else None
        raw_providers = llm.get("providers") if isinstance(llm, dict) else None
        if isinstance(entries, list):
            return _settings_revision(
                entries, raw_providers if isinstance(raw_providers, list) else []
            )
    except (OSError, ValueError):
        pass
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
    app.state.serve_token = token
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
        elif path.startswith("/api/settings/llm"):
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
        if request.url.path.startswith("/api/settings/llm"):
            # These routes read and write provider keys, so with token auth
            # on they need the token from every peer: any local process, or a
            # page on another localhost origin, is a loopback peer too.
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
                    "LLM settings are available on loopback only",
                    headers={"Cache-Control": "no-store"},
                )
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
    app.add_middleware(_TokenAuthMiddleware, token=token, tickets=tickets)
    # Outermost: compresses whatever the inner layers answer
    app.add_middleware(_CompressionMiddleware)

    # ------------------------------------------------------------------ API

    @app.get("/api/capabilities", response_model=Capabilities)
    async def get_capabilities(request: Request) -> dict[str, Any]:
        from markitai.config import PRESET_NAMES, get_preset

        state = _state(request)
        effective = _effective_models(state.configured_models, state.detected_models)
        routable = any(_is_deployment_routable(model) for model in effective)
        return {
            "version": __version__,
            "implementation": "python",
            "docs_url": DOCS_URL,
            "source_url": SOURCE_URL,
            "llm": {
                "configured": bool(state.configured_models),
                "routable": routable,
                "effective": routable,
                "models": [model.litellm_params.model for model in effective],
            },
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
            },
            "remote_services": {
                "cloudflare": _cloudflare_capability(
                    state.config, trusted=_is_trusted_request(request)
                )
            },
        }

    @app.get("/api/settings/llm", response_model=LLMSettingsPayload)
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
            provider_label,
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
                card.get("kind") != "common"
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
        # A persisted env reference and its auto-detected environment card are
        # one connection, not two provider cards.
        providers = [
            card
            for card in detected
            if not (
                card.get("kind") == "common"
                and card.get("provider") in explicit_provider_names
            )
            and not any(
                card.get("kind") == "environment"
                and card.get("provider") == saved.provider
                and saved.api_key == f"env:{card.get('source')}"
                for saved in explicit_by_id.values()
            )
        ]
        local_connections = {
            str(card["provider"])
            for card in providers
            if card.get("kind") in {"local_cli", "oauth"}
        }

        for saved in state.configured_providers:
            model_count = sum(
                model.model_info is not None
                and model.model_info.provider_id == saved.id
                for model in state.configured_models
            )
            providers.append(
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
                }
            )

        seen_configured_connections: set[tuple[str, str | None, str | None]] = set()
        explicit_connections = {
            (provider.provider, provider.api_base, provider.api_key)
            for provider in state.configured_providers
        }
        for index, model in enumerate(state.configured_models):
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
                for candidate in state.configured_models
            )
            providers.append(
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
                }
            )
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
        state = _state(request)
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
            return _credentials_view(saved.provider, saved.api_key, saved.api_base)

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
        from markitai.config import resolve_env_value
        from markitai.providers.discovery import discover_models

        state = _state(request)
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
        if api_key is None and key_variable is not None:
            lacking = model_test.missing_key(f"{provider}/", None, api_base)
            if lacking is not None:
                # Nothing is sent: the provider's 401 would read like a
                # wrong key, as for a model test
                failure = model_test.missing_key_failure(lacking)
                answer = {
                    "provider": provider,
                    "status": "unavailable",
                    "source": "live_api",
                    "authoritative": False,
                    "cached": False,
                    "stale": False,
                    "models": [],
                    "detail": failure.detail,
                    "reason": failure.reason,
                }
                if failure.key_variable:
                    answer["key_variable"] = failure.key_variable
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
        state: ServeState,
        body: LLMModelCreate,
        providers: list[Any],
        models: list[Any],
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
        if new_connection:
            # A new connection left on the provider's default: where
            # LiteLLM's own default is another host (or api.openai.com), the
            # documented address the card showed is stored with it
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

    @app.post("/api/settings/llm/models", response_model=LLMSettingsPayload)
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

    @app.put("/api/settings/llm/models/{model_name}", response_model=LLMSettingsPayload)
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
        "/api/settings/llm/models/{model_name}", response_model=LLMSettingsPayload
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

    @app.post("/api/settings/llm/deployments/batch", response_model=LLMSettingsPayload)
    async def add_llm_deployments_batch(
        request: Request, body: LLMDeploymentBatch
    ) -> dict[str, Any]:
        state = _state(request)

        def mutate(
            entries: list[Any], providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            for deployment in body.deployments:
                entries.append(raw_deployment(state, deployment, providers, entries))
            return entries

        return await _apply_model_list_mutation(
            state,
            mutate,
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

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
        configured = _configured_credential_pairs(_state(request))

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

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    @app.delete(
        "/api/settings/llm/deployments/{deployment_id}",
        response_model=LLMSettingsPayload,
    )
    async def delete_llm_deployment(
        request: Request, deployment_id: str, expected_revision: str
    ) -> dict[str, Any]:
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

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=expected_revision,
            backfill_ids=True,
        )

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
        "/api/settings/llm/providers/{provider_id}", response_model=LLMSettingsPayload
    )
    async def update_llm_provider(
        request: Request, provider_id: str, body: LLMProviderUpdate
    ) -> dict[str, Any]:
        configured = _configured_credential_pairs(_state(request))

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

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    @app.delete(
        "/api/settings/llm/providers/{provider_id}", response_model=LLMSettingsPayload
    )
    async def delete_llm_provider(
        request: Request, provider_id: str, expected_revision: str
    ) -> dict[str, Any]:
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

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=expected_revision,
            backfill_ids=True,
        )

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
        if body.deployment_id is not None or body.model_name is not None:
            body = _stored_probe_body(
                _state(request),
                deployment_id=body.deployment_id,
                model_name=body.model_name,
            )
        else:
            _check_credential_target(
                body.api_key,
                body.api_base,
                client_key=body.api_key,
                configured=_configured_credential_pairs(_state(request)),
            )
            asked_base = body.api_base
            # An address the caller names; a stored one is the owner's
            _refuse_ambient_login_address(body.model, asked_base)
            body = _fill_stored_credentials(_state(request), body)
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
        if body.model is not None:
            lacking = model_test.missing_key(body.model, body.api_key, body.api_base)
            if lacking is not None:
                # Nothing is sent: LiteLLM would send the request without a
                # key, and the provider's 401 reads like a wrong key
                return model_test.missing_key_failure(lacking).answer()
        try:
            detail = await asyncio.wait_for(
                _probe_llm(body), timeout=_LLM_TEST_BACKSTOP_S
            )
        except Exception as e:  # must never 500: ok flags the result
            return _probe_failure(e, body)
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
        # model the user asked for: refuse before anything is stored.
        _require_model(state, opts)

        cfg = _build_job_config(state.config, opts)
        job = state.registry.create_job(options=_saved_options(opts), cfg=cfg)
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
        # Options inherited from the item were the caller's earlier choice;
        # a retry that names model processing now needs a model now.
        if explicit_options and operation == "retry":
            _require_model(state, opts)
        cfg = _build_job_config(state.config, opts)
        if operation == "enhance" and (not cfg.llm.enabled or not _llm_routable(state)):
            raise ApiError(
                409,
                "llm_unavailable",
                "LLM enhancement is unavailable; enable a routable LLM first",
            )
        job.public_network_only = job.public_network_only or not trusted
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
        protect = item.status == "done" and item.output is not None and not item.skipped
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
            job.retry_task = asyncio.create_task(run_retry_queue(state.registry, job))
        logger.info("[Serve] Job {} queued {} for item {}", job_id, operation, item_id)
        return {
            "job_id": job_id,
            "items": [{"item_id": item.item_id, "name": item.name, "kind": item.kind}],
        }

    @app.delete("/api/jobs/{job_id}/items/{item_id}", status_code=204)
    async def delete_job_item(request: Request, job_id: str, item_id: str) -> None:
        """Permanently remove one terminal ledger item and all its files.

        Its markdown pair, its original upload and the assets/screenshots it
        owns (``serve.artifacts``) go with it; files a remaining sibling
        also claims are kept.
        """
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

        async def stream() -> AsyncIterator[str]:
            queue = registry.subscribe(job)
            try:
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
                        yield ": ping\n\n"
                        continue
                    if event == SHUTDOWN_EVENT:
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
            artifacts: list[dict[str, Any]] = []
            seen: set[Path] = set()

            def add_artifact(path: Path) -> None:
                if path.is_file() and path not in seen:
                    seen.add(path)
                    artifacts.append(
                        {
                            "relpath": path.relative_to(out_dir).as_posix(),
                            "size": path.stat().st_size,
                        }
                    )

            add_artifact(base_path)
            # Only expose the .llm.md when the item is actually enhanced, so a
            # stale sibling never makes the UI offer a spurious base↔llm diff.
            if item.llm_enhanced:
                add_artifact(llm_path)
            for artifact in _item_assets(item, out_dir, base_name, base_path, llm_path):
                add_artifact(artifact)
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

    # ------------------------------------------------------------- originals

    def _original_paths(job: Job, item: JobItem) -> str:
        return f"/api/jobs/{job.job_id}/items/{item.item_id}/original/file"

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
        return original.descriptor(_original_paths(job, item))

    @app.get("/api/jobs/{job_id}/items/{item_id}/original/file")
    async def get_item_original_file(
        request: Request, job_id: str, item_id: str
    ) -> FileResponse:
        """The stored original bytes, never rendered by a direct visit."""
        job = _get_job(request, job_id)
        item = _get_item(job, item_id)
        upload = kept_upload(job, item)
        if upload is None:
            raise ApiError(404, "original_unavailable", "no original is kept")
        media_type = original_media_type(upload)
        if media_type in ("text/html", "application/xhtml+xml", "image/svg+xml"):
            media_type = "text/plain; charset=utf-8"
        return FileResponse(
            upload,
            media_type=media_type,
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(item.name, safe=""),
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "sandbox; default-src 'none'",
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
        if original.kind in ("none", "html", "text"):
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
            if original.kind == "screenshots"
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
        tickets: DownloadTickets = request.app.state.download_tickets
        ticket = tickets.issue(body.path)
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
        # never overwrites a live registry entry.
        jobs = [
            job
            for job in registry.jobs.values()
            if job.status != "running" and not job_expired(job)
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
        """Download every completed job, including rehydrated history."""
        registry = _state(request).registry
        jobs = sorted(
            (
                job
                for job in registry.jobs.values()
                if job.status != "running" and not job_expired(job)
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

    @app.delete("/api/history/{job_id}", status_code=204)
    async def delete_history_job(request: Request, job_id: str) -> None:
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

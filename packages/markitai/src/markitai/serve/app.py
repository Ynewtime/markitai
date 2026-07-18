"""FastAPI application factory for ``markitai serve``.

Exposes the conversion core over REST + SSE for a local single-user web UI.
Requires the ``markitai[serve]`` extra (fastapi, uvicorn, python-multipart).
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from markitai import __version__
from markitai.providers.discovery import provider_default_api_base
from markitai.serve.jobs import (
    Job,
    JobRegistry,
    RetryWork,
    cleanup_stale_jobs,
    job_cached_dir_size,
    job_duration_ms,
    rehydrate_jobs,
    run_job,
    run_retry_queue,
    write_job_meta,
)
from markitai.serve.schemas import (
    JobOptions,
    JobRetryBody,
    LLMDeploymentBatch,
    LLMModelCreate,
    LLMModelDiscoveryRequest,
    LLMModelUpdate,
    LLMProviderUpdate,
    LLMSettingsUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from markitai.config import LLMProviderConfig, MarkitaiConfig, ModelConfig

DEFAULT_JOBS_ROOT = Path.home() / ".markitai" / "serve" / "jobs"
DEFAULT_CONFIG_PATH = Path.home() / ".markitai" / "config.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per file
MAX_JOB_ITEMS = 50
# Hard cap on a whole request body (declared Content-Length): the largest
# legitimate job plus generous multipart framing slack.
MAX_REQUEST_BYTES = MAX_JOB_ITEMS * MAX_UPLOAD_BYTES + 64 * 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024
_SSE_PING_INTERVAL = 15.0
# Per-request timeout handed to litellm for POST /api/settings/llm/test, plus
# an outer backstop for local providers (claude-agent/, ...) that spawn a CLI
# and may not honor the litellm timeout parameter.
_LLM_TEST_TIMEOUT_S = 15
_LLM_TEST_BACKSTOP_S = 30.0
ROOT_HINT = "web UI not built; POST /api/jobs or see /api/capabilities"

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
# Names an item may claim inside .markitai/assets|screenshots, after the
# "<base_name>." prefix: numbered assets ("0001.png"), paged screenshots
# ("page0001.png" / "slide0001.png") and full-page URL shots ("full.jpg").
_META_ARTIFACT_SUFFIX_RE = re.compile(
    r"^(?:\d{1,6}|page\d{1,6}|slide\d{1,6}|full)"
    r"\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg|ico|avif|heic|heif)$",
    re.IGNORECASE,
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
    """Auto-detect a built web UI: package static dir, then repo webapp/dist."""
    bundled = Path(__file__).parent / "static"
    if (bundled / "index.html").is_file():
        return bundled
    parents = Path(__file__).resolve().parents
    if len(parents) > 5:
        repo_dist = parents[5] / "webapp" / "dist"
        if (repo_dist / "index.html").is_file():
            return repo_dist
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
                from fastapi.responses import JSONResponse

                response = JSONResponse(
                    {"detail": "request body too large"}, status_code=413
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_STATE_CHANGING_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})


def _host_header_hostname(value: str) -> str | None:
    """Extract the lowercased hostname from a ``host[:port]`` header value."""
    try:
        return urlsplit(f"//{value}").hostname
    except ValueError:
        return None


def _origin_hostname(value: str) -> str | None:
    """Extract the lowercased hostname from an Origin header value.

    Opaque origins (``null``) and malformed values yield None.
    """
    try:
        return urlsplit(value).hostname
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


def _is_trusted_origin(
    origin_hostname: str | None,
    host_hostname: str | None,
    extra_allowed: frozenset[str],
) -> bool:
    """Whether *origin_hostname* may issue a state-changing cross-site request.

    Unlike the Host header, the Origin is fully attacker-controlled, so the
    blanket IP-literal trust of ``_is_trusted_hostname`` must not apply here:
    an attacker can host the exploit page from a bare IP and would otherwise
    pass. Trust only loopback, explicit ``--allowed-host`` entries, or an
    origin whose host equals the already Host-validated host (i.e. the request
    is same-origin: the attacker's page would have to be served from the very
    address the victim connected to, which makes it the real app).
    """
    if not origin_hostname:
        return False
    if origin_hostname in _LOCAL_HOSTNAMES or origin_hostname in extra_allowed:
        return True
    return host_hostname is not None and origin_hostname == host_hostname


class _HostGuardMiddleware:
    """Reject DNS-rebinding hosts and cross-site state-changing requests.

    The loopback peer check on the settings routes cannot stop DNS
    rebinding: the browser connects to 127.0.0.1 (so the peer *is* loopback)
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
                response = JSONResponse(
                    {
                        "detail": (
                            f"host '{host_header}' is not allowed; use "
                            "localhost, an IP address, or start the server "
                            "with --allowed-host"
                        )
                    },
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if (
                origin_header is not None
                and scope.get("method", "").upper() in _STATE_CHANGING_METHODS
                and not _is_trusted_origin(
                    _origin_hostname(origin_header),
                    host_hostname,
                    self.allowed_hosts,
                )
            ):
                response = JSONResponse(
                    {
                        "detail": (
                            f"cross-site request from origin "
                            f"'{origin_header}' is not allowed"
                        )
                    },
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _state(request: Request) -> ServeState:
    """Return the ServeState initialized by the lifespan."""
    return request.app.state.markitai


def _get_job(request: Request, job_id: str) -> Job:
    """Return the job with *job_id* or raise 404."""
    job = _state(request).registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


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


async def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Stream one upload to disk, enforcing the per-file size limit."""
    name = _sanitize_upload_name(upload.filename)
    target = dest_dir / name
    counter = 2
    while target.exists():
        target = dest_dir / f"{Path(name).stem} ({counter}){Path(name).suffix}"
        counter += 1
    size = 0
    try:
        with target.open("wb") as fh:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"file '{name}' exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit"
                        ),
                    )
                fh.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return target


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
    if cfg.llm.enabled and not cfg.llm.model_list:
        logger.warning(
            "[Serve] LLM requested but no models are configured; "
            "running without LLM enhancement"
        )
        cfg.llm.enabled = False
    if cfg.llm.enabled:
        # Web semantics (CLI untouched): the UI diff view needs the base .md
        # variant alongside .llm.md.
        cfg.llm.keep_base = True
    return cfg


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
    if api_key is None:
        return None
    if api_key.startswith("env:"):
        return api_key
    if len(api_key) >= 12:
        return f"{api_key[:2]}…{api_key[-4:]}"
    if len(api_key) >= 4:
        return f"…{api_key[-2:]}"
    return "…"


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
        "api_key_configured": bool(model.litellm_params.api_key) and not local,
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

    model_id = model.litellm_params.model
    if is_local_provider_model(model_id):
        return is_local_provider_available(model_id)
    provider = model_id.split("/", 1)[0].lower()
    if provider in {"ollama", "lm_studio", "vllm"}:
        return True
    if model.litellm_params.get_resolved_api_key(strict=False):
        return True
    env_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_key = env_keys.get(provider)
    return env_key is not None and bool(os.environ.get(env_key))


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
    atomic_write_json(config_path, data)
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


def _fill_stored_credentials(
    state: ServeState, body: LLMSettingsUpdate
) -> LLMSettingsUpdate:
    """Fill a keyless probe body with the stored credentials for its model.

    A saved-row test posts only its deployment reference; resolve credentials
    from the running config exactly like a job run would. An explicit ``api_key``
    always wins, local providers never take one, and nothing is persisted.
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
        # Two entries may share a model string; prefer the same-base one.
        same_base = [p for p in stored if p.api_base == body.api_base]
        stored = same_base or stored
    if not stored:
        return body
    params = stored[0]
    return body.model_copy(
        update={
            "api_key": params.api_key,
            "api_base": body.api_base if body.api_base is not None else params.api_base,
        }
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
        raise HTTPException(status_code=422, detail=f"{label} not found")
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
    return _sanitize_upload_name(_split_output_name(Path(output).name))


def _split_output_name(name: str) -> str:
    """Strip the markitai markdown suffix: 'a.pdf.llm.md' -> 'a.pdf'."""
    if name.endswith(".llm.md"):
        return name[: -len(".llm.md")]
    if name.endswith(".md"):
        return name[: -len(".md")]
    return name


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


class _SPAStaticFiles(StaticFiles):
    """Static files with SPA fallback: unknown paths serve index.html."""

    async def get_response(self, path: str, scope: Any) -> Any:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code == 404:
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


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
) -> FastAPI:
    """Create the markitai serve FastAPI application.

    Args:
        static_dir: Directory with a built web UI. Auto-detected when None
            (package ``serve/static/``, then repo ``webapp/dist/``).
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
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if configure_logging:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

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
        app.state.markitai = state
        restored = rehydrate_jobs(state.registry, cfg)
        if restored:
            logger.info("[Serve] Rehydrated {} archived job(s)", restored)
        logger.info("[Serve] markitai {} ready (jobs root: {})", __version__, root)
        try:
            yield
        finally:
            await app.state.markitai.registry.shutdown()

            from markitai.fetch import close_shared_clients
            from markitai.utils.executor import shutdown_converter_executor

            await close_shared_clients()
            shutdown_converter_executor()
            try:
                from litellm.llms.custom_httpx.async_client_cleanup import (
                    close_litellm_async_clients,
                )

                await close_litellm_async_clients()
            except Exception as e:
                logger.debug("[Serve] LiteLLM client cleanup failed: {}", e)

    app = FastAPI(title="markitai serve", version=__version__, lifespan=lifespan)
    app.add_middleware(_BodyLimitMiddleware)
    app.add_middleware(
        _HostGuardMiddleware,
        allowed_hosts=frozenset(
            hostname
            for hostname in map(_normalize_allowed_host, allowed_hosts or ())
            if hostname
        ),
    )

    @app.middleware("http")
    async def protect_local_settings(request: Request, call_next: Any) -> Response:
        if request.url.path.startswith("/api/settings/llm"):
            host = request.client.host if request.client is not None else "127.0.0.1"
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host == "localhost"
            if not loopback:
                return JSONResponse(
                    {"detail": "LLM settings are available on loopback only"},
                    status_code=403,
                    headers={"Cache-Control": "no-store"},
                )
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        return await call_next(request)

    # ------------------------------------------------------------------ API

    @app.get("/api/capabilities")
    async def get_capabilities(request: Request) -> dict[str, Any]:
        state = _state(request)
        effective = _effective_models(state.configured_models, state.detected_models)
        routable = any(_is_deployment_routable(model) for model in effective)
        return {
            "version": __version__,
            "llm": {
                "configured": bool(state.configured_models),
                "routable": routable,
                "effective": routable,
                "models": [model.litellm_params.model for model in effective],
            },
            "presets": ["minimal", "standard", "rich"],
            "extras": {
                "browser": find_spec("playwright") is not None,
                "svg": find_spec("cairosvg") is not None,
                "kreuzberg": find_spec("kreuzberg") is not None,
            },
        }

    @app.get("/api/settings/llm")
    async def get_llm_settings(request: Request) -> dict[str, Any]:
        return _llm_settings_payload(_state(request))

    @app.get("/api/settings/llm/detected")
    async def get_detected_providers() -> list[dict[str, Any]]:
        """Legacy quick-add candidates; provider cards use the v2 endpoint."""
        return await _detect_provider_candidates()

    @app.get("/api/settings/llm/providers")
    async def get_llm_providers(
        request: Request, refresh: bool = False
    ) -> dict[str, Any]:
        from markitai.providers import is_local_provider_model
        from markitai.providers.discovery import (
            detect_provider_connections,
            provider_label,
        )

        state = _state(request)
        detected = await detect_provider_connections(
            state.configured_models, refresh=refresh
        )
        detected = [card for card in detected if card.get("kind") != "configured"]
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
                    "api_key_configured": bool(saved.api_key),
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
                    "api_key_configured": bool(params.api_key),
                    "api_base_configured": bool(params.api_base),
                    "api_base": _sanitized_api_origin(
                        params.api_base or provider_default_api_base(provider)
                    ),
                    "model_count": model_count,
                    "supports_discovery": True,
                }
            )
        return {"providers": providers}

    @app.get("/api/settings/llm/providers/{provider_id}/credentials")
    async def get_llm_provider_credentials(
        request: Request, provider_id: str
    ) -> dict[str, str | None]:
        """Return one saved connection's editable values on explicit request.

        Provider collection responses remain secret-free. This endpoint is
        covered by the loopback-only settings middleware and is called only
        when the user opens the provider editor. API keys stay behind a
        password control; API bases are ordinary, non-secret URL fields.
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
            # Return the RAW saved base (None when the connection uses the
            # provider default) so re-saving after only a key edit does not
            # pin the default as an explicit override; the default is offered
            # separately for the editor's placeholder.
            return {
                "api_key": saved.api_key,
                "api_base": saved.api_base,
                "api_base_placeholder": provider_default_api_base(saved.provider),
            }

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
            legacy_provider = legacy.litellm_params.model.split("/", 1)[0].lower()
            return {
                "api_key": legacy.litellm_params.api_key,
                "api_base": legacy.litellm_params.api_base,
                "api_base_placeholder": provider_default_api_base(legacy_provider),
            }

        raise HTTPException(
            status_code=404,
            detail=(
                "This saved provider connection no longer exists. "
                "Refresh the provider list and choose it again."
            ),
        )

    @app.post("/api/settings/llm/model-discovery")
    async def discover_llm_models(
        request: Request, body: LLMModelDiscoveryRequest
    ) -> dict[str, Any]:
        from markitai.config import resolve_env_value
        from markitai.providers.discovery import discover_models

        state = _state(request)
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
        if body.provider_id is not None and stored_provider is None and stored is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        if deployment_id is not None and stored is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        fallback_key = (
            stored_provider.api_key
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
        raw_key = body.api_key if body.api_key is not None else fallback_key
        raw_base = body.api_base if body.api_base is not None else fallback_base
        api_key = resolve_env_value(raw_key, strict=False) if raw_key else None
        api_base = resolve_env_value(raw_base, strict=False) if raw_base else None
        return await discover_models(
            body.provider,
            api_key=api_key,
            api_base=api_base,
            refresh=body.refresh,
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
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to update config file: {type(e).__name__}",
                )
            except ValueError as e:
                logger.error(
                    "[Serve] LLM settings update failed for {}: {}",
                    state.config_path,
                    type(e).__name__,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to update config file: {type(e).__name__}",
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
        credential_provider = (
            _raw_provider_by_id(providers, body.credential_provider_id)
            if body.credential_provider_id is not None
            else None
        )
        if body.credential_deployment_id is not None and credential_source is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )
        if body.credential_provider_id is not None and credential_provider is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This saved provider connection no longer exists. "
                    "Refresh the provider list and choose it again."
                ),
            )

        provider_name = (body.provider or body.model.split("/", 1)[0]).strip().lower()
        provider_id = (
            str(credential_provider["id"])
            if credential_provider is not None
            else credential_source.model_info.provider_id
            if credential_source is not None
            and credential_source.model_info is not None
            else None
        )
        fallback_key = (
            credential_provider.get("api_key")
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
        api_key = body.api_key if body.api_key is not None else fallback_key
        api_base = body.api_base if body.api_base is not None else fallback_base
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

    def update_raw_deployment(entry: dict[str, Any], body: LLMModelUpdate) -> None:
        params = entry.get("litellm_params")
        if not isinstance(params, dict):
            params = {}
        if "model_name" in body.model_fields_set and body.model_name is not None:
            entry["model_name"] = body.model_name
        if "model" in body.model_fields_set and body.model is not None:
            params["model"] = body.model
        if "api_key" in body.model_fields_set:
            if body.api_key is None:
                params.pop("api_key", None)
            else:
                params["api_key"] = body.api_key
        if "api_base" in body.model_fields_set:
            if body.api_base is None:
                params.pop("api_base", None)
            else:
                params["api_base"] = body.api_base
        if "weight" in body.model_fields_set and body.weight is not None:
            params["weight"] = body.weight
        _strip_ignored_local_params(params)
        entry["litellm_params"] = params

    @app.post("/api/settings/llm/models")
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

    @app.put("/api/settings/llm/models/{model_name}")
    async def update_llm_model(
        request: Request, model_name: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        def mutate(
            entries: list[Any], _providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            matches = _find_raw_entries(entries, model_name)
            if not matches:
                raise HTTPException(
                    status_code=404, detail=f"model_name '{model_name}' not found"
                )
            if len(matches) > 1:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "ambiguous_legacy_model_name",
                        "model_name": model_name,
                    },
                )
            update_raw_deployment(matches[0], body)
            return entries

        payload = await _apply_model_list_mutation(
            _state(request), mutate, expected_revision=body.expected_revision
        )
        logger.info("[Serve] LLM model updated through legacy route: {}", model_name)
        return payload

    @app.delete("/api/settings/llm/models/{model_name}")
    async def delete_llm_model(request: Request, model_name: str) -> dict[str, Any]:
        def mutate(
            entries: list[Any], providers: list[Any], _legacy_mapping: dict[str, str]
        ) -> list[Any]:
            matches = _find_raw_entries(entries, model_name)
            if not matches:
                raise HTTPException(
                    status_code=404, detail=f"model_name '{model_name}' not found"
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

    @app.post("/api/settings/llm/deployments/batch")
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

    @app.patch("/api/settings/llm/deployments/{deployment_id}")
    async def update_llm_deployment(
        request: Request, deployment_id: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        if body.expected_revision is None:
            raise HTTPException(status_code=422, detail="expected_revision is required")

        def mutate(
            entries: list[Any], _providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            entry = _find_raw_deployment(entries, deployment_id, legacy_mapping)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "This model configuration no longer exists. "
                        "Refresh settings and try again."
                    ),
                )
            update_raw_deployment(entry, body)
            return entries

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    @app.delete("/api/settings/llm/deployments/{deployment_id}")
    async def delete_llm_deployment(
        request: Request, deployment_id: str, expected_revision: str
    ) -> dict[str, Any]:
        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            entry = _find_raw_deployment(entries, deployment_id, legacy_mapping)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
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

    @app.patch("/api/settings/llm/providers/{provider_id}")
    async def update_llm_provider(
        request: Request, provider_id: str, body: LLMProviderUpdate
    ) -> dict[str, Any]:
        def mutate(
            entries: list[Any], providers: list[Any], legacy_mapping: dict[str, str]
        ) -> list[Any]:
            target, legacy_entry = locate_raw_provider(
                entries, providers, provider_id, legacy_mapping
            )
            if target is None and legacy_entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
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
            if target is None:
                target = {
                    "id": str(uuid.uuid4()),
                    "provider": provider_name,
                    **({"api_key": old_key} if isinstance(old_key, str) else {}),
                    **({"api_base": old_base} if isinstance(old_base, str) else {}),
                }
                providers.append(target)

            if "api_key" in body.model_fields_set:
                if body.api_key is None:
                    target.pop("api_key", None)
                else:
                    target["api_key"] = body.api_key
            if "api_base" in body.model_fields_set:
                if body.api_base is None:
                    target.pop("api_base", None)
                else:
                    target["api_base"] = body.api_base

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
                if "api_key" in body.model_fields_set:
                    if body.api_key is None:
                        params.pop("api_key", None)
                    else:
                        params["api_key"] = body.api_key
                if "api_base" in body.model_fields_set:
                    if body.api_base is None:
                        params.pop("api_base", None)
                    else:
                        params["api_base"] = body.api_base
                _link_raw_model_to_provider(entry, saved_id)
            return entries

        return await _apply_model_list_mutation(
            _state(request),
            mutate,
            expected_revision=body.expected_revision,
            backfill_ids=True,
        )

    @app.delete("/api/settings/llm/providers/{provider_id}")
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
                raise HTTPException(
                    status_code=404,
                    detail=(
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

    @app.post("/api/settings/llm/test")
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
            body = _fill_stored_credentials(_state(request), body)
        try:
            detail = await asyncio.wait_for(
                _probe_llm(body), timeout=_LLM_TEST_BACKSTOP_S
            )
        except Exception as e:  # must never 500: ok flags the result
            return {"ok": False, "detail": _sanitize_probe_error(e, body)}
        return {"ok": True, "detail": detail}

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        request: Request,
        files: list[UploadFile] = File(default=[]),
        urls: str = Form(default="[]"),
        options: str = Form(default="{}"),
    ) -> dict[str, Any]:
        state = _state(request)

        try:
            url_list = json.loads(urls) if urls.strip() else []
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="urls must be a JSON array")
        if not isinstance(url_list, list) or not all(
            isinstance(u, str) and u.strip() for u in url_list
        ):
            raise HTTPException(
                status_code=422, detail="urls must be a JSON array of non-empty strings"
            )

        try:
            opts = JobOptions.model_validate_json(options or "{}")
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"invalid options: {e}")

        total = len(files) + len(url_list)
        if total == 0:
            raise HTTPException(
                status_code=422, detail="at least one file or url is required"
            )
        if total > MAX_JOB_ITEMS:
            raise HTTPException(
                status_code=422,
                detail=f"too many items: {total} (max {MAX_JOB_ITEMS})",
            )

        if opts.preset is not None:
            from markitai.config import get_preset

            if get_preset(opts.preset.lower(), state.config) is None:
                raise HTTPException(
                    status_code=422, detail=f"unknown preset '{opts.preset}'"
                )

        cfg = _build_job_config(state.config, opts)
        job = state.registry.create_job(options=opts.model_dump(), cfg=cfg)
        cfg.output.dir = str(job.out_dir)

        from markitai.serve.jobs import JobItem
        from markitai.utils.cli_helpers import url_to_filename
        from markitai.utils.paths import derive_output_name

        # Any failure below must roll the half-created job back (registry
        # entry + job dir), or it stays a permanent "running" zombie whose
        # SSE stream never emits the terminal `job` event.
        try:
            for index, upload in enumerate(files, start=1):
                saved = await _save_upload(upload, job.uploads_dir)
                job.items.append(
                    JobItem(
                        item_id=f"i{index}",
                        name=saved.name,
                        kind="file",
                        source=saved,
                    )
                )

            # Pre-assign unique output names per job: colliding
            # url_to_filename results (or a URL colliding with an upload's
            # derived output) would otherwise silently overwrite each other
            # in LLM mode, where only the .llm.md sibling is written.
            taken = {derive_output_name(item.name) for item in job.items}
            for offset, url in enumerate(url_list, start=len(files) + 1):
                clean_url = url.strip()
                base_name = url_to_filename(clean_url)
                output_name = base_name
                counter = 2
                while output_name in taken:
                    output_name = (
                        f"{Path(base_name).stem} ({counter}){Path(base_name).suffix}"
                    )
                    counter += 1
                taken.add(output_name)
                job.items.append(
                    JobItem(
                        item_id=f"i{offset}",
                        name=clean_url,
                        kind="url",
                        source=clean_url,
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
            raise HTTPException(
                status_code=500, detail="internal error while creating the job"
            ) from e
        logger.info(
            "[Serve] Job {} created: {} file(s), {} url(s)",
            job.job_id,
            len(files),
            len(url_list),
        )
        return {
            "job_id": job.job_id,
            "items": [
                {"item_id": i.item_id, "name": i.name, "kind": i.kind}
                for i in job.items
            ],
        }

    @app.post("/api/jobs/{job_id}/items/{item_id}/retry", status_code=202)
    async def retry_job_item(
        request: Request,
        job_id: str,
        item_id: str,
        body: JobRetryBody | None = None,
    ) -> dict[str, Any]:
        """Queue a terminal item to retry or explicitly enhance with LLM."""
        state = _state(request)
        job = _get_job(request, job_id)
        item = job.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")
        if item.status not in ("done", "error") or item_id in job.retry_pending:
            raise HTTPException(
                status_code=409,
                detail="item has not reached a terminal state yet; retry when done",
            )

        operation = body.operation if body is not None else "retry"
        if body is not None and body.options is not None:
            opts = body.options
        else:
            opts = JobOptions.model_validate(
                {key: job.options.get(key) for key in ("preset", "llm", "ocr")}
            )
        if opts.preset is not None:
            from markitai.config import get_preset

            if get_preset(opts.preset.lower(), state.config) is None:
                raise HTTPException(
                    status_code=422, detail=f"unknown preset '{opts.preset}'"
                )

        if item.kind == "file":
            uploads_dir = job.uploads_dir.resolve()
            upload = (uploads_dir / item.name).resolve()
            # Rehydrated meta is untrusted on-disk state; never follow a
            # tampered name outside the job's upload directory.
            if not upload.is_relative_to(uploads_dir) or not upload.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"original upload '{item.name}' is no longer on "
                        "disk (job files may have been cleaned up)"
                    ),
                )
            item.source = upload
        else:
            from markitai.utils.cli_helpers import url_to_filename

            item.source = item.name
            item.output_name = item.output_name or url_to_filename(item.name)

        cfg = _build_job_config(state.config, opts)
        if operation == "enhance" and (
            not cfg.llm.enabled or not _llm_settings_payload(state)["routable"]
        ):
            raise HTTPException(
                status_code=409,
                detail="LLM enhancement is unavailable; enable a routable LLM first",
            )
        cfg.output.dir = str(job.out_dir)
        # The rerun reuses the job's out_dir, so it would otherwise hit
        # on_conflict against its own prior output (rename → duplicate
        # versioned files; skip → the enhance can never run). Always overwrite
        # the item's own outputs; a failed rerun rolls back (RetryWork.prior_*).
        cfg.output.on_conflict = "overwrite"
        job.cfg = cfg
        if operation != "enhance":
            job.options = opts.model_dump()

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
        )

        # Reset only this item. The job keeps its identity and item ordering;
        # the serial worker prevents rapid retries from racing output writes.
        # meta.json is left intact: an unclean shutdown mid-rerun then
        # rehydrates the job at its last persisted state rather than losing it.
        item.status = "queued"
        item.error = None
        item.output = None
        item.duration_ms = None
        item.finished_at = None
        item.cost_usd = None
        item.llm_enhanced = False
        item.operation = operation
        item.skipped = False
        item.skip_reason = None
        job.status = "running"
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
        """Permanently remove one terminal ledger item and its direct output."""
        state = _state(request)
        job = _get_job(request, job_id)
        item = job.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")
        if job.status == "running" or item.status not in ("done", "error"):
            raise HTTPException(
                status_code=409, detail="job is still running; retry when done"
            )

        if len(job.items) == 1:
            await asyncio.to_thread(state.registry.discard_job, job)
            logger.info("[Serve] Job {} deleted with its last item", job_id)
            return

        output = item.output
        job.items.remove(item)
        if output:
            candidate = (job.out_dir / output).resolve()
            if candidate.is_relative_to(job.out_dir.resolve()):
                candidate.unlink(missing_ok=True)
                if candidate.name.endswith(".llm.md"):
                    base = candidate.with_name(
                        f"{candidate.name.removesuffix('.llm.md')}.md"
                    )
                    base.unlink(missing_ok=True)
        (job.job_dir / "archive.zip").unlink(missing_ok=True)
        await asyncio.to_thread(write_job_meta, job)
        state.registry.publish_job(job)
        logger.info("[Serve] Job {} item {} deleted", job_id, item_id)

    @app.get("/api/jobs/{job_id}")
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
                if job.status != "running":
                    # The job may have reached its terminal state while the
                    # snapshot frame was being sent: drain events already
                    # queued for this subscriber (late item output/error/
                    # duration updates) before the final `job` frame.
                    while not queue.empty():
                        event, data = queue.get_nowait()
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

    @app.get("/api/jobs/{job_id}/items/{item_id}/result")
    async def get_item_result(
        request: Request, job_id: str, item_id: str
    ) -> dict[str, Any]:
        job = _get_job(request, job_id)
        item = job.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")
        if item.status != "done" or item.output is None:
            raise HTTPException(status_code=404, detail="item result not available")

        out_dir = job.out_dir.resolve()
        output_path = (out_dir / item.output).resolve()
        if not output_path.is_relative_to(out_dir):
            raise HTTPException(status_code=404, detail="item result not available")

        base_name = _split_output_name(output_path.name)
        llm_path = output_path.parent / f"{base_name}.llm.md"
        base_path = output_path.parent / f"{base_name}.md"
        # Honor the item's selected variant rather than probing llm-first: a
        # non-LLM retry after an earlier enhance leaves a stale .llm.md on disk
        # (pruned on success, but this stays correct even if it lingers).
        if item.llm_enhanced and llm_path.is_file():
            variant, markdown_path = "llm", llm_path
        elif base_path.is_file():
            variant, markdown_path = "base", base_path
        elif llm_path.is_file():
            variant, markdown_path = "llm", llm_path
        else:
            raise HTTPException(status_code=404, detail="item result not available")

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
        from markitai.constants import ASSETS_REL_PATH, SCREENSHOTS_REL_PATH

        # Plain string prefix match (not glob: metacharacters in upload names
        # like "report[2024].pdf" must stay literal), and the remainder must
        # look like an asset index / screenshot suffix so item "a" never
        # claims "a.txt.0001.png" belonging to sibling item "a.txt".
        prefix = f"{base_name}."
        for rel in (ASSETS_REL_PATH, SCREENSHOTS_REL_PATH):
            meta_dir = out_dir / rel
            if meta_dir.is_dir():
                for artifact in sorted(meta_dir.iterdir()):
                    remainder = artifact.name.removeprefix(prefix)
                    if remainder != artifact.name and _META_ARTIFACT_SUFFIX_RE.match(
                        remainder
                    ):
                        add_artifact(artifact)

        return {
            "name": item.name,
            "variant": variant,
            "markdown": markdown_path.read_text(encoding="utf-8"),
            "artifacts": artifacts,
        }

    @app.get("/api/jobs/{job_id}/files/{relpath:path}")
    async def get_job_file(request: Request, job_id: str, relpath: str) -> FileResponse:
        job = _get_job(request, job_id)
        out_dir = job.out_dir.resolve()
        target = (out_dir / relpath).resolve()
        if not target.is_relative_to(out_dir) or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(
            target, filename=target.name, content_disposition_type="attachment"
        )

    @app.get("/api/jobs/{job_id}/archive")
    async def get_job_archive(request: Request, job_id: str) -> FileResponse:
        job = _get_job(request, job_id)
        if job.status == "running":
            raise HTTPException(
                status_code=409, detail="job is still running; retry when done"
            )
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
        )

    # -------------------------------------------------------------- history

    @app.get("/api/history")
    async def get_history(request: Request) -> list[dict[str, Any]]:
        registry = _state(request).registry
        # The registry already dedupes archived vs this-process terminal jobs:
        # a job finished in this process is one entry, and startup rehydrate
        # never overwrites a live registry entry.
        jobs = [job for job in registry.jobs.values() if job.status != "running"]
        entries = await asyncio.to_thread(
            lambda: [
                {
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
                    "names_preview": [i.name for i in job.items[:3]],
                    "kinds_preview": [i.kind for i in job.items[:3]],
                    "duration_ms": job_duration_ms(job),
                    "size_bytes": job_cached_dir_size(job),
                }
                for job in jobs
            ]
        )
        entries.sort(key=lambda e: str(e["created_at"]), reverse=True)
        return entries

    @app.get("/api/history/archive")
    async def get_history_archive(request: Request) -> FileResponse:
        """Download every completed job, including rehydrated history."""
        registry = _state(request).registry
        jobs = sorted(
            (job for job in registry.jobs.values() if job.status != "running"),
            key=lambda job: job.created_at,
        )
        if not jobs:
            raise HTTPException(status_code=404, detail="no completed jobs to archive")

        archive_path = registry.jobs_root / "archive.zip"

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
        # The whole-history zip is rebuilt on every download, so it is pure
        # dead weight afterwards. Unlink once the response is sent (POSIX keeps
        # the open fd valid for the in-flight stream); nothing else removes it.
        return FileResponse(
            archive_path,
            filename="markitai-all.zip",
            media_type="application/zip",
            content_disposition_type="attachment",
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    @app.delete("/api/history/{job_id}", status_code=204)
    async def delete_history_job(request: Request, job_id: str) -> None:
        job = _get_job(request, job_id)
        if job.status == "running":
            raise HTTPException(
                status_code=409, detail="job is still running; retry when done"
            )
        await asyncio.to_thread(_state(request).registry.discard_job, job)
        logger.info("[Serve] History job {} deleted", job_id)

    # --------------------------------------------------------------- static

    resolved_static = static_dir if static_dir is not None else detect_static_dir()
    if resolved_static is not None and (resolved_static / "index.html").is_file():
        app.mount(
            "/", _SPAStaticFiles(directory=resolved_static, html=True), name="static"
        )
    else:

        @app.get("/")
        async def root_hint() -> dict[str, str]:
            return {"markitai": __version__, "hint": ROOT_HINT}

    return app

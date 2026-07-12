"""FastAPI application factory for ``markitai serve``.

Exposes the conversion core over REST + SSE for a local single-user web UI.
Requires the ``markitai[serve]`` extra (fastapi, uvicorn, python-multipart).
"""

from __future__ import annotations

import asyncio
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from markitai import __version__
from markitai.serve.jobs import (
    Job,
    JobRegistry,
    cleanup_stale_jobs,
    job_dir_size,
    rehydrate_jobs,
    run_job,
)
from markitai.serve.schemas import (
    JobOptions,
    LLMModelCreate,
    LLMModelUpdate,
    LLMSettingsUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from markitai.config import MarkitaiConfig, ModelConfig

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
    settings_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


async def _detect_provider_candidates() -> list[dict[str, Any]]:
    """Auto-detect available LLM providers (async mirror of the CLI detector).

    Kept in serve because ``markitai.serve`` must not import ``markitai.cli``
    (import-linter top-layer contract); the CLI variant also relies on
    ``asyncio.run``, which is unavailable inside a running event loop.

    Returns:
        Candidates in CLI-detector priority order, each shaped as the
        ``GET /api/settings/llm/detected`` payload: ``{provider, model,
        label, requires_api_key}``. ``requires_api_key`` is always False —
        a provider is only detected when its auth already exists (CLI/OAuth
        login or an API key environment variable).
    """
    import shutil

    from markitai.providers.auth import AuthManager

    candidates: list[dict[str, Any]] = []

    def add(provider: str, model: str, label: str) -> None:
        candidates.append(
            {
                "provider": provider,
                "model": model,
                "label": label,
                "requires_api_key": False,
            }
        )

    auth = AuthManager()
    for binary, provider, model, label in (
        ("claude", "claude-agent", "claude-agent/sonnet", "Claude Code CLI"),
        ("copilot", "copilot", "copilot/claude-haiku-4.5", "GitHub Copilot CLI"),
    ):
        if shutil.which(binary):
            try:
                status = await auth.check_auth(provider)
            except Exception:
                continue
            if status.authenticated:
                add(provider, model, label)
    try:
        chatgpt = await auth.check_auth("chatgpt")
        if chatgpt.authenticated:
            add("chatgpt", "chatgpt/gpt-5.4-mini", "ChatGPT (Codex OAuth)")
    except Exception:
        logger.debug("[Serve] ChatGPT auth detection failed")
    for env_var, provider, model in (
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-haiku-4-5"),
        ("OPENAI_API_KEY", "openai", "openai/gpt-5.4-nano"),
        ("GEMINI_API_KEY", "gemini", "gemini/gemini-3.1-flash-lite-preview"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek/deepseek-v4-flash"),
        ("OPENROUTER_API_KEY", "openrouter", "openrouter/google/gemini-3.1-flash-lite"),
    ):
        if os.environ.get(env_var):
            add(provider, model, f"{env_var} (environment)")
    return candidates


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
    """Deep-copy the base config and apply preset + explicit llm override."""
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
    if cfg.llm.enabled and not cfg.llm.model_list:
        logger.warning(
            "[Serve] LLM requested but no models are configured; "
            "running without LLM enhancement"
        )
        cfg.llm.enabled = False
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


def _llm_settings_payload(state: ServeState) -> dict[str, Any]:
    """Build the masked ``GET /api/settings/llm`` response body.

    Local-provider entries (claude-agent/, copilot/, chatgpt/) always render
    ``api_key_masked`` as null: their auth comes from the CLI/OAuth session,
    so any stored key is meaningless and must not look configurable.
    """
    from markitai.providers import is_local_provider_model

    return {
        "configured": bool(state.config.llm.model_list),
        "source": state.llm_source,
        "config_path": _display_config_path(state.config_path),
        "models": [
            {
                "model_name": m.model_name,
                "model": m.litellm_params.model,
                "api_key_masked": (
                    None
                    if is_local_provider_model(m.litellm_params.model)
                    else _mask_api_key(m.litellm_params.api_key)
                ),
                "api_base": m.litellm_params.api_base,
            }
            for m in state.config.llm.model_list
        ],
    }


def _mutate_config_model_list(
    config_path: Path, mutate: Callable[[list[Any]], list[Any]]
) -> list[ModelConfig]:
    """Apply *mutate* to the raw ``llm.model_list`` of the user's config file.

    The existing file is loaded as raw JSON so keys unknown to
    ``MarkitaiConfig`` (top-level, inside ``llm``, and inside each entry)
    survive untouched; only ``llm.model_list`` is replaced with the mutated
    list. The file is written atomically with owner-only permissions
    (mkstemp temp file + rename, matching how ``markitai init`` writes it),
    and only after the mutated list validates — a bad merge never lands on
    disk.

    Args:
        config_path: Target config file (created when missing).
        mutate: Callback receiving the raw entry list; returns the new list.
            May raise ``HTTPException`` for request-level errors (409/404).

    Returns:
        The validated mutated model list, mirroring exactly what a fresh
        startup would load from the file (for the running-config hot update).

    Raises:
        ValueError: When the existing file is not valid JSON / not a JSON
            object, or the mutated model list fails validation.
        OSError: When reading or writing the file fails.
    """
    from markitai.config import ModelConfig
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
        model_list = []
    model_list = mutate(model_list)
    llm["model_list"] = model_list
    data["llm"] = llm

    merged = [ModelConfig.model_validate(m) for m in model_list]
    atomic_write_json(config_path, data)
    return merged


def _find_raw_entry(entries: list[Any], model_name: str) -> dict[str, Any] | None:
    """Return the first raw model_list entry with *model_name*, or None."""
    for entry in entries:
        if isinstance(entry, dict) and entry.get("model_name") == model_name:
            return entry
    return None


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
    """Build a terse, key-free probe failure detail (class + first line)."""
    from markitai.config import resolve_env_value

    message = str(exc).strip()
    first_line = message.splitlines()[0] if message else ""
    detail = type(exc).__name__ + (f": {first_line}" if first_line else "")
    secret = resolve_env_value(body.api_key, strict=False) if body.api_key else None
    if secret:
        detail = _scrub_secret(detail, secret)
    return detail[:300]


async def _probe_llm(body: LLMSettingsUpdate) -> str:
    """Run a minimal one-token completion against the given model.

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
        register_providers,
    )

    params = LiteLLMParams(
        model=body.model, api_key=body.api_key, api_base=body.api_base
    )
    api_key = params.get_resolved_api_key()
    api_base = params.get_resolved_api_base()

    if not is_local_provider_available(body.model):
        provider = body.model.split("/", 1)[0]
        raise RuntimeError(f"{provider} SDK is not installed")

    register_providers()
    kwargs: dict[str, Any] = {
        "model": body.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
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
            llm_source = "config" if cfg.llm.model_list else "none"
        else:
            from markitai.config import ConfigManager

            cfg = ConfigManager().load()
            if cfg.llm.model_list:
                llm_source = "config"
            else:
                await _backfill_llm_models(cfg)
                llm_source = "detected" if cfg.llm.model_list else "none"

        root = (jobs_root if jobs_root is not None else DEFAULT_JOBS_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        removed = cleanup_stale_jobs(root)
        if removed:
            logger.info("[Serve] Cleaned up {} stale job dir(s)", removed)

        state = ServeState(
            config=cfg,
            registry=JobRegistry(root),
            llm_source=llm_source,
            config_path=config_path if config_path is not None else DEFAULT_CONFIG_PATH,
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

    # ------------------------------------------------------------------ API

    @app.get("/api/capabilities")
    async def get_capabilities(request: Request) -> dict[str, Any]:
        cfg = _state(request).config
        return {
            "version": __version__,
            "llm": {
                "configured": bool(cfg.llm.model_list),
                "models": [m.litellm_params.model for m in cfg.llm.model_list],
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
        return await _detect_provider_candidates()

    async def _apply_model_list_mutation(
        state: ServeState, mutate: Callable[[list[Any]], list[Any]]
    ) -> dict[str, Any]:
        """Run one settings write: file mutation + running-config hot update."""
        async with state.settings_lock:
            try:
                merged = await asyncio.to_thread(
                    _mutate_config_model_list, state.config_path, mutate
                )
            except OSError as e:
                logger.error(
                    "[Serve] LLM settings write failed for {}: {}",
                    state.config_path,
                    e,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to update config file: {type(e).__name__}",
                )
            except ValueError as e:
                # Not logged with str(e): a ValidationError repr can echo input
                # values of pre-existing file entries (potentially key material).
                logger.error(
                    "[Serve] LLM settings update failed for {}: {}",
                    state.config_path,
                    type(e).__name__,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to update config file: {type(e).__name__}",
                )
            # Hot-update the running base config to exactly what a fresh
            # startup would now load from the file; capabilities reads it per
            # request.
            state.config.llm.model_list = merged
            state.llm_source = "config" if merged else "none"
        return _llm_settings_payload(state)

    @app.post("/api/settings/llm/models")
    async def add_llm_model(request: Request, body: LLMModelCreate) -> dict[str, Any]:
        def mutate(entries: list[Any]) -> list[Any]:
            if _find_raw_entry(entries, body.model_name) is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"model_name '{body.model_name}' already exists",
                )
            params: dict[str, Any] = {"model": body.model}
            if body.api_key is not None:
                params["api_key"] = body.api_key
            if body.api_base is not None:
                params["api_base"] = body.api_base
            _strip_ignored_local_params(params)
            entries.append({"model_name": body.model_name, "litellm_params": params})
            return entries

        payload = await _apply_model_list_mutation(_state(request), mutate)
        logger.info("[Serve] LLM model added: {} ({})", body.model_name, body.model)
        return payload

    @app.put("/api/settings/llm/models/{model_name}")
    async def update_llm_model(
        request: Request, model_name: str, body: LLMModelUpdate
    ) -> dict[str, Any]:
        def mutate(entries: list[Any]) -> list[Any]:
            entry = _find_raw_entry(entries, model_name)
            if entry is None:
                raise HTTPException(
                    status_code=404, detail=f"model_name '{model_name}' not found"
                )
            params = entry.get("litellm_params")
            if not isinstance(params, dict):
                params = {}
            # Omitted field = keep the current raw value (env: references and
            # literal keys survive untouched); explicit null clears it.
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
            _strip_ignored_local_params(params)
            entry["litellm_params"] = params
            return entries

        payload = await _apply_model_list_mutation(_state(request), mutate)
        logger.info("[Serve] LLM model updated: {}", model_name)
        return payload

    @app.delete("/api/settings/llm/models/{model_name}")
    async def delete_llm_model(request: Request, model_name: str) -> dict[str, Any]:
        def mutate(entries: list[Any]) -> list[Any]:
            kept = [
                e
                for e in entries
                if not (isinstance(e, dict) and e.get("model_name") == model_name)
            ]
            if len(kept) == len(entries):
                raise HTTPException(
                    status_code=404, detail=f"model_name '{model_name}' not found"
                )
            return kept

        payload = await _apply_model_list_mutation(_state(request), mutate)
        logger.info("[Serve] LLM model deleted: {}", model_name)
        return payload

    @app.post("/api/settings/llm/test")
    async def test_llm_settings(body: LLMSettingsUpdate) -> dict[str, Any]:
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
        if llm_path.is_file():
            variant, markdown_path = "llm", llm_path
        elif base_path.is_file():
            variant, markdown_path = "base", base_path
        else:
            raise HTTPException(status_code=404, detail="item result not available")

        artifacts: list[dict[str, Any]] = []
        seen: set[Path] = set()

        def add_artifact(path: Path) -> None:
            if path.is_file() and path not in seen:
                seen.add(path)
                artifacts.append(
                    {
                        "relpath": str(path.relative_to(out_dir)),
                        "size": path.stat().st_size,
                    }
                )

        add_artifact(base_path)
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
        out_dir = job.out_dir.resolve()
        archive_path = job.job_dir / "archive.zip"

        def build() -> None:
            # Unique temp path + os.replace: readers never observe a
            # partially written zip. ".<name>.<rand>.tmp" droppings from
            # atomic_write_text are excluded from the walk.
            tmp_path = archive_path.with_name(f".archive.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in sorted(out_dir.rglob("*")):
                        if file.is_file() and not (
                            file.name.startswith(".") and file.name.endswith(".tmp")
                        ):
                            zf.write(file, file.relative_to(out_dir))
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
                    "names_preview": [i.name for i in job.items[:3]],
                    "size_bytes": job_dir_size(job.job_dir),
                }
                for job in jobs
            ]
        )
        entries.sort(key=lambda e: str(e["created_at"]), reverse=True)
        return entries

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

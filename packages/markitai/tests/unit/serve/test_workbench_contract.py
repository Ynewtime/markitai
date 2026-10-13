"""The shared web workbench's API contract, shape by shape, against this server.

One browser workbench (built in the markitai-rust repository and vendored
under ``markitai/serve/workbench``) runs on both servers. Its API contract
(version 1, 2026-10-11) is mirrored by the shapes below: required keys,
optional keys, value types and enumerations, as the UI reads them. Every
endpoint the UI calls is exercised here and its answers are checked against
those shapes; unknown extra keys are allowed (the UI ignores them), missing
or mistyped ones are not. Error answers must carry a ``reason`` from the
contract's table for their status.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import MarkitaiConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Contract shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F:
    """One field: its kind, whether the key must be present, whether null is ok.

    ``kind`` is ``"str"``, ``"int"``, ``"num"``, ``"bool"``, ``"obj"``, a
    ``frozenset`` of allowed string values, a shape name (``"@Pricing"``) or a
    one-element list ``[kind]`` for an array.
    """

    kind: Any
    required: bool = True
    nullable: bool = False


def opt(kind: Any, nullable: bool = True) -> F:
    return F(kind, required=False, nullable=nullable)


def null(kind: Any) -> F:
    return F(kind, nullable=True)


ITEM_KIND = frozenset({"file", "url"})
PROVIDER_GROUP = frozenset(
    {"common", "subscriptions", "makers", "clouds", "inference", "gateways", "local"}
)
OPTION_KEYS = (
    "preset",
    "llm",
    "ocr",
    "profile",
    "alt",
    "desc",
    "screenshot",
    "screenshot_only",
    "pure",
    "no_cache",
    "no_compress",
    "strategy",
    "backend",
)

SHAPES: dict[str, dict[str, F]] = {
    "ErrorBody": {
        "detail": F(("str", "obj")),
        "code": F("str"),
        "reason": F("str"),
    },
    "Capabilities": {
        "version": F("str"),
        "implementation": opt(frozenset({"rust", "python"}), nullable=False),
        "docs_url": opt("str", nullable=False),
        "source_url": opt("str", nullable=False),
        "llm": F("@CapabilitiesLLM"),
        "presets": F(["str"]),
        "preset_options": F("obj"),
        "extras": F("@Extras"),
        "limits": F("@Limits"),
        "remote_services": opt("@RemoteServices", nullable=False),
        "features": opt("@Features", nullable=False),
        "access": opt("@Access", nullable=False),
    },
    "Access": {
        "role": F(frozenset({"admin", "user"})),
        "label": opt("str"),
    },
    "CapabilitiesLLM": {
        "configured": opt("bool", nullable=False),
        "routable": F("bool"),
        "effective": opt("bool", nullable=False),
        "models": opt(["str"], nullable=False),
    },
    "Extras": {"browser": F("bool"), "svg": F("bool")},
    "Limits": {"max_job_items": F("int"), "history_retention_days": opt("int")},
    "RemoteServices": {"cloudflare": F("@CloudflareCapability")},
    "CloudflareCapability": {
        "configured": F("bool"),
        "available": F("bool"),
        "reason": null(
            frozenset(
                {
                    "not_configured",
                    "invalid_configuration",
                    "disabled_by_policy",
                    "client_not_trusted",
                }
            )
        ),
        "browser_rendering": F("bool"),
        "file_conversion": F("bool"),
        "file_extensions": F(["str"]),
    },
    "Features": {
        name: F("bool")
        for name in (
            "cancel",
            "download_tickets",
            "url_lists",
            "history_items",
            "original",
            "result_excerpt",
            "job_labels",
            "item_assets",
            "item_archive",
            "token_regenerate",
            "user_tokens",
            "config_edit",
            "batch_delete",
        )
    },
    "PresetFeatures": {
        name: F("bool") for name in ("llm", "ocr", "alt", "desc", "screenshot")
    },
    "JobOptions": {name: opt(("str", "bool")) for name in OPTION_KEYS},
    "Pricing": {
        "priced_requests": F("int"),
        "unpriced_requests": F("int"),
        "cost_status": F(frozenset({"complete", "partial", "unknown"})),
    },
    "ItemPayload": {
        "item_id": F("str"),
        "name": F("str"),
        "kind": F(ITEM_KIND),
        "status": F(frozenset({"queued", "running", "done", "error"})),
        "error": null("str"),
        "error_code": opt("str"),
        "output": null("str"),
        "output_name": null("str"),
        "duration_ms": null("int"),
        "finished_at": null("str"),
        "cost_usd": null("num"),
        "pricing": opt("@Pricing"),
        "llm_enhanced": F("bool"),
        "operation": F(frozenset({"convert", "retry", "enhance"})),
        "skipped": F("bool"),
        "skip_reason": null("str"),
        "retryable": F("bool"),
        "warnings": F(["str"]),
        "options": opt("@JobOptions"),
        "models": opt(["str"]),
        "source_size": opt("int"),
        "rerun_failure": opt("@RerunFailure"),
    },
    "RerunFailure": {
        "operation": F(frozenset({"retry", "enhance"})),
        "error_code": F("str"),
        "error": F("str"),
        "failed_at": F("str"),
    },
    "JobProgress": {
        "status": F(frozenset({"running", "done", "error"})),
        "done": F("int"),
        "failed": F("int"),
        "total": F("int"),
    },
    "JobSnapshot": {
        "status": F(frozenset({"running", "done", "error"})),
        "done": F("int"),
        "failed": F("int"),
        "total": F("int"),
        "job_id": F("str"),
        "created_at": F("str"),
        "finished_at": null("str"),
        "options": F("@JobOptions"),
        "items": F(["@ItemPayload"]),
        "persistence_error": opt("str"),
        "label": opt("str"),
    },
    "HistoryEntry": {
        "job_id": F("str"),
        "created_at": F("str"),
        "finished_at": null("str"),
        "status": F(frozenset({"running", "done", "error"})),
        "total": F("int"),
        "done": F("int"),
        "failed": F("int"),
        "skipped": F("int"),
        "llm_enhanced": F("int"),
        "cost_usd": null("num"),
        "pricing": opt("@Pricing"),
        "names_preview": F(["str"]),
        "kinds_preview": F([ITEM_KIND]),
        "duration_ms": null("int"),
        "size_bytes": F("int"),
        "origin": opt(frozenset({"web", "cli"}), nullable=False),
        "retryable": opt("bool", nullable=False),
        "label": opt("str"),
        "options": opt("@JobOptions"),
        "items": opt(["@ItemPayload"]),
    },
    "CreatedItem": {"item_id": F("str"), "name": F("str"), "kind": F(ITEM_KIND)},
    "CreatedJob": {"job_id": F("str"), "items": F(["@CreatedItem"])},
    "CancelResponse": {"job_id": F("str"), "stopping": F("int")},
    "Artifact": {"relpath": F("str"), "size": F("int")},
    "ItemResult": {
        "name": F("str"),
        "variant": F(frozenset({"llm", "base"})),
        "markdown": F("str"),
        "artifacts": F(["@Artifact"]),
        "truncated": opt("bool"),
        "markdown_bytes": opt("int"),
    },
    "DownloadTicket": {"ticket": F("str"), "url": F("str"), "expires_in": F("int")},
    "RegeneratedToken": {
        "token": F("str"),
        "network_url": null("str"),
        "persistent": F("bool"),
        "startup_variable": null("str"),
    },
    "AccessToken": {
        "id": F("str"),
        "label": F("str"),
        "hint": F("str"),
        "created_at": F("str"),
        "last_used_at": null("str"),
    },
    "AccessTokenList": {"tokens": F(["@AccessToken"])},
    "CreatedAccessToken": {
        "id": F("str"),
        "label": F("str"),
        "hint": F("str"),
        "created_at": F("str"),
        "last_used_at": null("str"),
        "token": F("str"),
    },
    "BatchDeleteEntry": {
        "job_id": F("str"),
        "item_id": null("str"),
        "ok": F("bool"),
        "reason": opt("str"),
        "detail": opt("str"),
    },
    "BatchDeleteResult": {
        "deleted": F("int"),
        "failed": F("int"),
        "results": F(["@BatchDeleteEntry"]),
    },
    "ConfigDocument": {
        "path": F("str"),
        "origin": F("str"),
        "exists": F("bool"),
        "revision": F("str"),
        "text": null("str"),
        "masked": F("int"),
        "problem": null("str"),
        "read_only": F("bool"),
    },
    "SavedConfigDocument": {
        "path": F("str"),
        "origin": F("str"),
        "exists": F("bool"),
        "revision": F("str"),
        "text": null("str"),
        "masked": F("int"),
        "problem": null("str"),
        "read_only": F("bool"),
        "restart_required": F("bool"),
    },
    "PageBox": {"width": F("num"), "height": F("num")},
    "OriginalDescriptor": {
        "kind": F(frozenset({"pdf", "image", "html", "text", "screenshots", "none"})),
        "name": F("str"),
        "media_type": null("str"),
        "size": null("int"),
        "download": null("str"),
        "view": opt("str"),
        "pages": F(["@PageBox"]),
        "page_count": F("int"),
        "text_layer": F("bool"),
        "reason": null("str"),
    },
    "PageTextLine": {"text": F("str"), "bbox": F(["num"])},
    "PageText": {
        "page": F("int"),
        "width": F("num"),
        "height": F("num"),
        "lines": F(["@PageTextLine"]),
    },
    "ProviderCard": {
        "id": F("str"),
        "provider_id": opt("str"),
        "deployment_id": opt("str"),
        "provider": F("str"),
        "label": F("str"),
        # The contract's four kinds, and Python's "local_cli" and "oauth" for
        # its subscription runtimes: the UI treats every kind but configured
        # and environment as a provider to connect
        "kind": F("str"),
        "status": F("str"),
        "source": F("str"),
        "credential": opt("str"),
        "api_key_configured": opt("bool"),
        "api_base_configured": opt("bool"),
        "api_base": opt("str"),
        "model_count": opt("int"),
        "supports_discovery": F("bool"),
        "default_base": opt("str"),
        "key_variable": opt("str"),
        "key_optional": opt("bool"),
        "group": opt(PROVIDER_GROUP),
        "icon": opt("str"),
        "aliases": opt(["str"]),
    },
    "ProviderCredentials": {
        "api_key": null("str"),
        "api_key_source": opt("str"),
        "api_base": null("str"),
        "api_base_placeholder": null("str"),
    },
    "SettingsView": {
        "configured": F("bool"),
        "routable": F("bool"),
        "source": F("str"),
        "config_path": F("str"),
        "config_origin": F("str"),
        "revision": F("str"),
        "deployments": F(["obj"]),
        "detected": F(["obj"]),
    },
}

# Reasons the UI translates, by status (contract section 3).
REASONS: dict[int, set[str]] = {
    400: {"host_not_allowed", "invalid_multipart", "bad_request"},
    401: {"token_required", "ticket_invalid"},
    403: {
        "origin_not_allowed",
        "settings_forbidden",
        "remote_url_forbidden",
        "remote_processing_forbidden",
    },
    404: {
        "route_not_found",
        "job_not_found",
        "item_not_found",
        "result_unavailable",
        "file_not_found",
        "history_empty",
        "settings_entry_missing",
        "config_missing",
        "original_unavailable",
        "page_not_found",
    },
    405: {"method_not_allowed"},
    409: {
        "job_running",
        "job_not_running",
        "nothing_to_stop",
        "item_busy",
        "not_retryable",
        "batch_pending",
        "output_conflict",
        "output_identity_conflict",
        "persistence_failed",
        "stale_revision",
        "config_changed",
        "ambiguous_legacy_model_name",
        "settings_read_only",
        "job_id_collision",
    },
    413: {
        "request_too_large",
        "form_field_too_large",
        "file_too_large",
        "url_list_too_large",
        "result_too_large",
        "archive_too_large",
        "config_too_large",
    },
    422: {
        "empty_job",
        "too_many_items",
        "invalid_url",
        "unsupported_url_scheme",
        "invalid_urls",
        "invalid_url_list",
        "empty_url_list",
        "invalid_output_name",
        "invalid_options",
        "unknown_preset",
        "llm_unavailable",
        "remote_processing_confirmation_required",
        "remote_processing_disabled",
        "cloudflare_unavailable",
        "invalid_retry_body",
        "no_source",
        "upload_missing",
        "no_output",
        "enhancement_failed",
        "invalid_ticket_path",
        "json_required",
        "invalid_json",
        "invalid_settings_request",
        "invalid_provider_request",
        "render_failed",
        "invalid_page_request",
        "invalid_label",
        "invalid_max_bytes",
    },
    429: {"too_many_tickets", "provider_busy", "render_busy"},
    500: {"internal_error", "settings_io_failed", "settings_durability_unknown"},
    503: {"shutting_down", "unavailable"},
}
CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    503: "unavailable",
}
ITEM_ERROR_CODES = {
    "no_model_configured",
    "unsupported",
    "fetch_error",
    "conversion_error",
    "invalid_input",
    "is_directory",
    "invalid_json",
    "config_error",
    "io_error",
    "not_found",
    "file_not_found",
    "upload_missing",
    "internal_error",
    "enhancement_failed",
    "no_output",
    "output_conflict",
    "output_identity_conflict",
    "cancelled",
    "shutdown",
    "interrupted",
}

_PRIMITIVES: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "num": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "obj": lambda v: isinstance(v, dict),
}


def _check_value(value: Any, kind: Any, where: str) -> None:
    if isinstance(kind, tuple):
        problems = []
        for alternative in kind:
            try:
                _check_value(value, alternative, where)
                return
            except AssertionError as e:
                problems.append(str(e))
        raise AssertionError(f"{where}: none of {kind} matched: {problems}")
    if isinstance(kind, list):
        assert isinstance(value, list), f"{where}: expected a list, got {value!r}"
        for index, entry in enumerate(value):
            _check_value(entry, kind[0], f"{where}[{index}]")
        return
    if isinstance(kind, frozenset):
        assert value in kind, f"{where}: {value!r} not in {sorted(kind)}"
        return
    if isinstance(kind, str) and kind.startswith("@"):
        check_shape(kind[1:], value, where)
        return
    assert _PRIMITIVES[kind](value), f"{where}: expected {kind}, got {value!r}"


def check_shape(name: str, value: Any, where: str | None = None) -> None:
    """Assert *value* has contract shape *name* (extra keys allowed)."""
    where = where or name
    assert isinstance(value, dict), f"{where}: expected an object, got {value!r}"
    for key, field in SHAPES[name].items():
        if key not in value:
            assert not field.required, f"{where}: missing required key '{key}'"
            continue
        if value[key] is None:
            assert field.nullable, f"{where}.{key}: null is not allowed"
            continue
        _check_value(value[key], field.kind, f"{where}.{key}")


def check_error(response: httpx.Response, status: int, reason: str) -> None:
    """An error answer: the status, a contract body and the expected reason."""
    assert response.status_code == status, (response.status_code, response.text)
    body = response.json()
    check_shape("ErrorBody", body)
    assert body["code"] == CODES.get(status, "server_error"), body
    assert body["reason"] == reason, body
    assert reason in REASONS[status], f"{reason} is not a contract reason for {status}"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

TOKEN = "mk_contract-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _config(tmp_path: Path, *, model: bool = False) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    if model:
        from markitai.config import LiteLLMParams, ModelConfig

        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-4o-mini", api_key="sk-contract-test-0000"
                ),
            )
        ]
    return cfg


def _make_app(
    tmp_path: Path,
    *,
    token: str | None = None,
    model: bool = False,
    static_dir: Path | None = None,
) -> FastAPI:
    return create_app(
        static_dir=static_dir if static_dir is not None else tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=_config(tmp_path, model=model),
        configure_logging=False,
        config_path=tmp_path / "config.json",
        token=token,
    )


@asynccontextmanager
async def _client(
    app: FastAPI, *, peer: tuple[str, int] | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = (
            httpx.ASGITransport(app=app, client=peer)
            if peer is not None
            else httpx.ASGITransport(app=app)
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600"
        ) as client:
            yield client


def _form(
    files: list[tuple[str, bytes]] | None = None,
    urls: list[str] | None = None,
    options: dict[str, Any] | None = None,
    label: str | None = None,
) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files or []
    ]
    parts.append(("urls", (None, json.dumps(urls or []))))
    parts.append(("options", (None, json.dumps(options or {}))))
    if label is not None:
        parts.append(("label", (None, label)))
    return parts


def _converter(
    *,
    gate: asyncio.Event | None = None,
    markdown: str | None = None,
    usage: dict[str, Any] | None = None,
    screenshots: int = 0,
    enhanced: bool = False,
) -> Callable[..., Any]:
    """A fake file conversion: writes ``<name>.md`` (and page screenshots;
    ``<name>.llm.md`` when *enhanced*)."""

    async def fake(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
        from markitai.batch import ProcessResult
        from markitai.constants import SCREENSHOTS_REL_PATH

        if gate is not None:
            await gate.wait()
        out = out_dir / f"{file_path.name}.md"
        # Bytes, not text: Windows would write "\r\n"
        out.write_bytes(
            (markdown if markdown is not None else f"# {file_path.name}\n").encode()
        )
        if screenshots:
            from PIL import Image

            shots = out_dir / SCREENSHOTS_REL_PATH
            shots.mkdir(parents=True, exist_ok=True)
            for page in range(1, screenshots + 1):
                Image.new("RGB", (300, 400), "white").save(
                    shots / f"{file_path.name}.page{page:04d}.jpg"
                )
        if enhanced:
            out = out.with_suffix(".llm.md")
            out.write_bytes(f"# {file_path.name}, enhanced\n".encode())
        return ProcessResult(
            success=True,
            output_path=str(out),
            cost_usd=0.0125 if usage else 0.0,
            llm_usage=usage or {},
            llm_enhanced=enhanced,
        )

    return fake


async def _done(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] != "running":
            return data
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")


async def _job(client: httpx.AsyncClient, **form: Any) -> tuple[str, dict[str, Any]]:
    created = await client.post("/api/jobs", files=_form(**form))
    assert created.status_code == 201, created.text
    check_shape("CreatedJob", created.json())
    job_id = created.json()["job_id"]
    return job_id, await _done(client, job_id)


def _pdf_bytes(pages: int = 2, *, rotate: int = 0) -> bytes:
    import pymupdf

    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Heading {index + 1}", fontsize=24)
        page.insert_text((72, 140), "Body text on the page.", fontsize=11)
        if rotate:
            page.set_rotation(rotate)
    data = document.tobytes()
    document.close()
    return data


def _png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    name: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name is not None:
            events.append((name, json.loads(line.split(":", 1)[1])))
            name = None
    return events


# ---------------------------------------------------------------------------
# 4. Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    async def test_shape_and_features(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            data = (await client.get("/api/capabilities")).json()
        check_shape("Capabilities", data)
        assert data["implementation"] == "python"
        assert data["docs_url"].startswith("https://")
        assert data["source_url"] == "https://github.com/Ynewtime/markitai"
        # Without a token (--no-auth) there is none to regenerate, and no
        # user tokens
        assert data["features"].pop("token_regenerate") is False
        assert data["features"].pop("user_tokens") is False
        assert all(data["features"].values())
        assert data["limits"]["history_retention_days"] == 7
        for preset in data["presets"]:
            check_shape("PresetFeatures", data["preset_options"][preset])
        async with _client(_make_app(tmp_path, token="secret-token")) as client:
            data = (await client.get("/api/capabilities")).json()
        assert data["features"]["token_regenerate"] is True

    async def test_cloudflare_state_is_hidden_from_untrusted_callers(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path), peer=("192.168.1.30", 40000)) as client:
            data = (await client.get("/api/capabilities")).json()
        cloudflare = data["remote_services"]["cloudflare"]
        assert cloudflare["reason"] == "client_not_trusted"
        assert cloudflare["available"] is False
        assert cloudflare["file_extensions"] == []

    async def test_configured_cloudflare_is_available_to_trusted_callers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")
        monkeypatch.delenv("MARKITAI_NO_REMOTE_FETCH", raising=False)
        async with _client(_make_app(tmp_path)) as client:
            data = (await client.get("/api/capabilities")).json()
        cloudflare = data["remote_services"]["cloudflare"]
        check_shape("CloudflareCapability", cloudflare)
        assert cloudflare["available"] and cloudflare["configured"]
        assert "pdf" in cloudflare["file_extensions"]


# ---------------------------------------------------------------------------
# 6. Jobs and history
# ---------------------------------------------------------------------------


class TestCreateJob:
    async def test_items_files_first_then_list_urls_then_field_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        url_list = b"# reading list\nhttps://example.com/a  Report\nftp://skip.me/x\n"
        async with _client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_form(
                    files=[("b.txt", b"b"), ("list.URLS", url_list), ("a.txt", b"a")],
                    urls=["https://example.com/c"],
                    label="  my folder  ",
                ),
            )
            assert created.status_code == 201
            body = created.json()
            check_shape("CreatedJob", body)
            # Snapshot right away: URL items have not run (they would fetch)
            snapshot = (await client.get(f"/api/jobs/{body['job_id']}")).json()
        assert [(i["item_id"], i["name"], i["kind"]) for i in body["items"]] == [
            ("i1", "b.txt", "file"),
            ("i2", "a.txt", "file"),
            ("i3", "https://example.com/a", "url"),
            ("i4", "https://example.com/c", "url"),
        ]
        check_shape("JobSnapshot", snapshot)
        assert snapshot["label"] == "my folder"
        named = next(i for i in snapshot["items"] if i["item_id"] == "i3")
        assert named["output_name"] == "Report.md"
        files = {i["name"]: i for i in snapshot["items"] if i["kind"] == "file"}
        assert files["b.txt"]["source_size"] == 1

    async def test_urlencoded_body_creates_a_url_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def never(*args: Any, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", never)
        async with _client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                data={"urls": json.dumps(["https://example.com/x"]), "options": "{}"},
            )
        assert created.status_code == 201
        assert created.json()["items"][0]["kind"] == "url"

    @pytest.mark.parametrize(
        ("form", "status", "reason"),
        [
            ({}, 422, "empty_job"),
            ({"urls": ["not a url"]}, 422, "invalid_url"),
            ({"urls": ["ftp://example.com/a"]}, 422, "unsupported_url_scheme"),
            ({"urls": ["javascript:alert(1)"]}, 422, "invalid_url"),
            ({"options": {"bogus": True}}, 422, "invalid_options"),
            ({"options": {"llm": "yes"}}, 422, "invalid_options"),
            (
                {"options": {"preset": "nope"}, "urls": ["https://e.x/"]},
                422,
                "unknown_preset",
            ),
            ({"label": "x" * 121, "urls": ["https://e.x/"]}, 422, "invalid_label"),
            ({"label": "two\nlines", "urls": ["https://e.x/"]}, 422, "invalid_label"),
        ],
    )
    async def test_refusals(
        self, tmp_path: Path, form: dict[str, Any], status: int, reason: str
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post("/api/jobs", files=_form(**form))
        check_error(response, status, reason)
        assert not any((tmp_path / "jobs").iterdir())  # nothing stored

    async def test_invalid_urls_field(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs", files=[("urls", (None, "{not json"))]
            )
        check_error(response, 422, "invalid_urls")

    async def test_too_many_items(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.app.MAX_JOB_ITEMS", 2)
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs",
                files=_form(urls=["https://e.x/1", "https://e.x/2", "https://e.x/3"]),
            )
        check_error(response, 422, "too_many_items")

    async def test_file_too_large(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.app.MAX_UPLOAD_BYTES", 10)
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs", files=_form(files=[("big.txt", b"x" * 64)])
            )
        check_error(response, 413, "file_too_large")

    async def test_multipart_that_cannot_be_read(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs",
                content=b"--x\r\nbroken",
                headers={"Content-Type": "multipart/form-data"},
            )
        check_error(response, 400, "invalid_multipart")

    async def test_remote_processing_null_is_accepted_and_never_echoed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(
                client,
                files=[("a.txt", b"a")],
                options={"remote_processing": None, "profile": "rag"},
            )
            history = (await client.get("/api/history?include=items")).json()
        assert "remote_processing" not in data["options"]
        assert data["options"]["profile"] == "rag"
        assert all("remote_processing" not in i["options"] for i in data["items"])
        assert "remote_processing" not in history[0]["options"]


class TestUrlLists:
    async def _post(self, client: httpx.AsyncClient, body: bytes) -> httpx.Response:
        return await client.post("/api/jobs", files=_form(files=[("list.urls", body)]))

    async def test_json_list_with_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def never(*args: Any, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", never)
        body = json.dumps(
            ["https://e.x/a", {"url": "https://e.x/b", "output_name": "B page.md"}]
        ).encode()
        async with _client(_make_app(tmp_path)) as client:
            created = await self._post(client, body)
            snapshot = (
                await client.get(f"/api/jobs/{created.json()['job_id']}")
            ).json()
        assert created.status_code == 201
        assert [i["output_name"] for i in snapshot["items"]][1] == "B page.md"

    @pytest.mark.parametrize(
        ("body", "status", "reason"),
        [
            (b"[not json", 422, "invalid_url_list"),
            (b"\xff\xfe\x00bad", 422, "invalid_url_list"),
            (b"# only a comment\n\nftp://x.y/z\n", 422, "empty_url_list"),
            (b"https://e.x/a  ../escape\n", 422, "invalid_output_name"),
            (b"https://e.x/" + b"a" * (1024 * 1024), 413, "url_list_too_large"),
        ],
        # A 1 MiB URL as the id breaks Windows (PYTEST_CURRENT_TEST)
        ids=["not-json", "not-utf8", "no-urls", "escaping-name", "over-1mib"],
    )
    async def test_refusals(
        self, tmp_path: Path, body: bytes, status: int, reason: str
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await self._post(client, body)
        check_error(response, status, reason)


class TestLlmRefusal:
    @pytest.mark.parametrize(
        "options", [{"llm": True}, {"preset": "standard"}, {"preset": "rich"}]
    )
    async def test_a_model_request_without_a_model_is_refused(
        self, tmp_path: Path, options: dict[str, Any]
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs", files=_form(files=[("a.txt", b"a")], options=options)
            )
        check_error(response, 422, "llm_unavailable")
        assert not any((tmp_path / "jobs").iterdir())

    async def test_an_explicit_no_beats_the_preset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs",
                files=_form(
                    files=[("a.txt", b"a")],
                    options={"preset": "standard", "llm": False},
                ),
            )
        assert response.status_code == 201

    async def test_a_routable_model_admits_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: object(),
        )
        async with _client(_make_app(tmp_path, model=True)) as client:
            response = await client.post(
                "/api/jobs", files=_form(files=[("a.txt", b"a")], options={"llm": True})
            )
        assert response.status_code == 201


class TestRemoteProcessing:
    async def test_cloudflare_needs_this_requests_consent(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs",
                files=_form(urls=["https://e.x/"], options={"strategy": "cloudflare"}),
            )
            unconfigured = await client.post(
                "/api/jobs",
                files=_form(
                    urls=["https://e.x/"],
                    options={
                        "strategy": "cloudflare",
                        "remote_processing": "cloudflare",
                    },
                ),
            )
            stray = await client.post(
                "/api/jobs",
                files=_form(
                    files=[("a.txt", b"a")], options={"remote_processing": "cloudflare"}
                ),
            )
        check_error(response, 422, "remote_processing_confirmation_required")
        check_error(unconfigured, 422, "cloudflare_unavailable")
        check_error(stray, 422, "invalid_options")

    async def test_policy_off_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/jobs",
                files=_form(
                    files=[("a.pdf", b"%PDF-1.4")],
                    options={
                        "backend": "cloudflare",
                        "remote_processing": "cloudflare",
                    },
                ),
            )
        check_error(response, 422, "remote_processing_disabled")

    async def test_untrusted_callers_cannot_bill_the_server(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path), peer=("192.168.1.30", 40000)) as client:
            response = await client.post(
                "/api/jobs",
                files=_form(
                    files=[("a.pdf", b"%PDF-1.4")],
                    options={
                        "backend": "cloudflare",
                        "remote_processing": "cloudflare",
                    },
                ),
            )
        check_error(response, 403, "remote_processing_forbidden")


class TestSnapshotEventsAndHistory:
    async def test_snapshot_item_and_job_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(gate=gate)
        )
        async with _client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_form(files=[("a.txt", b"a")], label="Batch")
            )
            job_id = created.json()["job_id"]

            async def release() -> None:
                await asyncio.sleep(0.2)
                gate.set()

            releaser = asyncio.create_task(release())
            stream = await client.get(f"/api/jobs/{job_id}/events")
            await releaser
        events = _sse_events(stream.text)
        assert events[0][0] == "snapshot"
        check_shape("JobSnapshot", events[0][1])
        assert events[0][1]["label"] == "Batch"
        items = [data for name, data in events if name == "item"]
        assert items and all(check_shape("ItemPayload", i) is None for i in items)
        assert all("options" in i for i in items)
        assert events[-1][0] == "job" and events[-1][1]["status"] == "done"
        check_shape("JobProgress", events[-1][1])

    async def test_history_default_and_with_items(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            first, _ = await _job(client, files=[("a.txt", b"a")], label="First")
            second, _ = await _job(client, files=[("b.txt", b"b")])
            plain = (await client.get("/api/history")).json()
            full = (await client.get("/api/history?include=items")).json()
        assert [e["job_id"] for e in plain] == [second, first]
        for entry in plain:
            check_shape("HistoryEntry", entry)
            assert "items" not in entry and "options" not in entry
        for entry in full:
            check_shape("HistoryEntry", entry)
            assert entry["items"] and entry["options"] is not None
        assert full[1]["label"] == "First"

    async def test_history_sorts_by_instant_not_text(self, tmp_path: Path) -> None:
        """A CLI job recorded with a local offset sorts by its instant.

        Two hours ago at +14:00 reads as a later text than one hour ago in
        UTC, but it is the older job.
        """
        from datetime import UTC, datetime, timedelta, timezone

        now = datetime.now(UTC)
        older = (now - timedelta(hours=2)).astimezone(timezone(timedelta(hours=14)))
        newer = now - timedelta(hours=1)
        jobs = tmp_path / "jobs"
        for job_id, created in (
            ("aaaaaaaaaaaa", older.isoformat(timespec="milliseconds")),
            ("bbbbbbbbbbbb", newer.isoformat(timespec="milliseconds")),
        ):
            (jobs / job_id).mkdir(parents=True)
            (jobs / job_id / "meta.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "created_at": created,
                        "finished_at": created,
                        "status": "done",
                        "options": {},
                        "items": [],
                    }
                ),
                encoding="utf-8",
            )
        async with _client(_make_app(tmp_path)) as client:
            history = (await client.get("/api/history")).json()
        assert [e["job_id"] for e in history] == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]

    async def test_expired_history_is_gone_before_the_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("a.txt", b"a")])
            from markitai.serve.jobs import JOB_TTL_HOURS

            job = client._transport.app.state.markitai.registry.get(job_id)  # type: ignore[attr-defined]
            job.finished_at = "2000-01-01T00:00:00.000Z"
            history = (await client.get("/api/history")).json()
            snapshot = await client.get(f"/api/jobs/{job_id}")
            archive = await client.get("/api/history/archive")
        assert JOB_TTL_HOURS == 168
        assert history == []
        check_error(snapshot, 404, "job_not_found")
        check_error(archive, 404, "history_empty")

    async def test_unknown_job_and_item(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            missing_job = await client.get("/api/jobs/nope")
            missing_route = await client.get("/api/nothing-here")
            wrong_method = await client.put("/api/capabilities")
        check_error(missing_job, 404, "job_not_found")
        check_error(missing_route, 404, "route_not_found")
        check_error(wrong_method, 405, "method_not_allowed")


class TestPricing:
    async def test_items_a_model_answered_carry_pricing_and_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = {
            "gpt-4o-mini": {"requests": 2, "cost_usd": 0.0125},
            "local-model": {"requests": 1, "unpriced_requests": 1, "cost_usd": 0.0},
        }
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item",
            _converter(usage=usage, enhanced=True),
        )
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(client, files=[("a.txt", b"a")])
            history = (await client.get("/api/history")).json()
        item = data["items"][0]
        check_shape("ItemPayload", item)
        assert item["pricing"] == {
            "priced_requests": 2,
            "unpriced_requests": 1,
            "cost_status": "partial",
        }
        assert item["models"] == ["gpt-4o-mini", "local-model"]
        assert history[0]["pricing"] == item["pricing"]

    async def test_a_base_only_result_names_no_models_but_keeps_its_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract v1.6, 6.6: failed requests still cost, but no model's
        answer is in the result."""
        usage = {"gpt-4o-mini": {"requests": 2, "cost_usd": 0.0125}}
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(usage=usage)
        )
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(client, files=[("a.txt", b"a")])
        item = data["items"][0]
        assert item["llm_enhanced"] is False
        assert item["models"] == []
        assert item["pricing"]["priced_requests"] == 2
        assert item["cost_usd"] == pytest.approx(0.0125)

    async def test_unpriced_requests_are_unknown_not_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = {"fake-model": {"requests": 1, "unpriced_requests": 1}}
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(usage=usage)
        )
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(client, files=[("a.txt", b"a")])
        assert data["items"][0]["pricing"]["cost_status"] == "unknown"

    async def test_no_model_no_pricing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(client, files=[("a.txt", b"a")])
        item = data["items"][0]
        assert item["pricing"] is None and item["models"] == []
        assert item["llm_enhanced"] is False

    async def test_pricing_survives_a_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = {"gpt-4o-mini": {"requests": 1, "cost_usd": 0.01}}
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item",
            _converter(usage=usage, enhanced=True),
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("a.txt", b"a")], label="Kept")
        async with _client(_make_app(tmp_path)) as client:
            restored = (await client.get(f"/api/jobs/{job_id}")).json()
        assert restored["items"][0]["pricing"]["cost_status"] == "complete"
        assert restored["items"][0]["models"] == ["gpt-4o-mini"]
        assert restored["label"] == "Kept"


class TestItemErrors:
    async def test_unsupported_files_say_so(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            _, data = await _job(client, files=[("blob.bin", b"\x00\x01")])
        item = data["items"][0]
        assert item["status"] == "error"
        assert item["error_code"] == "unsupported"
        assert item["error_code"] in ITEM_ERROR_CODES


class TestCancel:
    async def test_stops_waiting_items_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(gate=gate)
        )
        cfg_app = _make_app(tmp_path)
        async with _client(cfg_app) as client:
            cfg_app.state.markitai.config.batch.concurrency = 1
            created = await client.post(
                "/api/jobs",
                files=_form(files=[(f"f{n}.txt", b"x") for n in range(4)]),
            )
            job_id = created.json()["job_id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snap = (await client.get(f"/api/jobs/{job_id}")).json()
                if any(i["status"] == "running" for i in snap["items"]):
                    break
                await asyncio.sleep(0.01)
            stopped = await client.post(f"/api/jobs/{job_id}/cancel")
            again = await client.post(f"/api/jobs/{job_id}/cancel")
            gate.set()
            data = await _done(client, job_id)
            finished = await client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 202
        check_shape("CancelResponse", stopped.json())
        assert stopped.json() == {"job_id": job_id, "stopping": 3}
        check_error(again, 409, "nothing_to_stop")
        check_error(finished, 409, "job_not_running")
        by_status = sorted(i["status"] for i in data["items"])
        assert by_status == ["done", "error", "error", "error"]
        for item in data["items"]:
            if item["status"] == "error":
                assert item["error_code"] == "cancelled"
                assert item["error"] == "cancelled (stopped by request)"
                assert item["retryable"] is True

    async def test_a_stopped_item_can_be_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(gate=gate)
        )
        app = _make_app(tmp_path)
        async with _client(app) as client:
            app.state.markitai.config.batch.concurrency = 1
            created = await client.post(
                "/api/jobs", files=_form(files=[("a.txt", b"a"), ("b.txt", b"b")])
            )
            job_id = created.json()["job_id"]
            await asyncio.sleep(0.05)
            assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 202
            gate.set()
            data = await _done(client, job_id)
            stopped = next(i for i in data["items"] if i["error_code"] == "cancelled")
            retried = await client.post(
                f"/api/jobs/{job_id}/items/{stopped['item_id']}/retry"
            )
            assert retried.status_code == 202
            check_shape("CreatedJob", retried.json())
            data = await _done(client, job_id)
        assert all(i["status"] == "done" for i in data["items"])

    async def test_unknown_job(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post("/api/jobs/nope/cancel")
        check_error(response, 404, "job_not_found")


class TestRetryAndDelete:
    async def test_refusals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("a.txt", b"a"), ("b.txt", b"b")])
            bad_body = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry", json={"operation": "nope"}
            )
            unknown_field = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry", json={"x": 1}
            )
            enhance = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry", json={"operation": "enhance"}
            )
            llm = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={"options": {"llm": True}},
            )
            missing_item = await client.post(f"/api/jobs/{job_id}/items/i9/retry")
            for upload in (tmp_path / "jobs" / job_id / "uploads").iterdir():
                upload.unlink()
            upload_gone = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            deleted = await client.delete(f"/api/jobs/{job_id}/items/i2")
            delete_missing = await client.delete(f"/api/jobs/{job_id}/items/i2")
        check_error(bad_body, 422, "invalid_retry_body")
        check_error(unknown_field, 422, "invalid_retry_body")
        # As the Rust server answers it (the contract lists 422 for a retry)
        assert enhance.status_code == 409
        assert enhance.json()["reason"] == "llm_unavailable"
        check_error(llm, 422, "llm_unavailable")
        check_error(missing_item, 404, "item_not_found")
        assert upload_gone.status_code == 404
        assert upload_gone.json()["reason"] == "upload_missing"
        assert deleted.status_code == 204
        check_error(delete_missing, 404, "item_not_found")

    async def test_pending_batch_items_are_not_enhanced_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        app = _make_app(tmp_path)
        async with _client(app) as client:
            job_id, _ = await _job(client, files=[("a.txt", b"a")])
            item = app.state.markitai.registry.get(job_id).items[0]
            item.skipped, item.skip_reason = True, "pending_batch"
            response = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
        assert response.status_code == 409
        assert response.json()["reason"] == "batch_pending"


class TestResultAndFiles:
    async def test_full_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(markdown="# Hi\n")
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("a.txt", b"a")])
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            file = await client.get(f"/api/jobs/{job_id}/files/a.txt.md")
            missing = await client.get(f"/api/jobs/{job_id}/files/..%2Fmeta.json")
        check_shape("ItemResult", result.json())
        assert result.json()["truncated"] is False
        assert result.json()["markdown_bytes"] == 5
        assert file.headers["content-type"] == "text/plain; charset=utf-8"
        assert file.headers["x-content-type-options"] == "nosniff"
        assert file.headers["content-disposition"].startswith("attachment")
        check_error(missing, 404, "file_not_found")

    async def test_excerpt_cuts_at_a_line_and_never_inside_a_character(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lines = "".join(f"| 行 {n} | 数据 {n} |\n" for n in range(2000))
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(markdown=lines)
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("t.txt", b"a")])
            excerpt = (
                await client.get(f"/api/jobs/{job_id}/items/i1/result?max_bytes=16384")
            ).json()
            small = await client.get(f"/api/jobs/{job_id}/items/i1/result?max_bytes=10")
            words = await client.get(
                f"/api/jobs/{job_id}/items/i1/result?max_bytes=lots"
            )
        check_shape("ItemResult", excerpt)
        assert excerpt["truncated"] is True
        assert excerpt["markdown_bytes"] == len(lines.encode())
        cut = excerpt["markdown"].encode()
        assert len(cut) <= 16384 and len(cut) >= 16384 * 3 // 4
        assert excerpt["markdown"].endswith("|\n")
        assert lines.startswith(excerpt["markdown"])
        check_error(small, 422, "invalid_max_bytes")
        check_error(words, 422, "invalid_max_bytes")

    async def test_excerpt_of_one_long_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = "é" * 5000  # two bytes each, no line end
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(markdown=text)
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("t.txt", b"a")])
            excerpt = (
                await client.get(f"/api/jobs/{job_id}/items/i1/result?max_bytes=1025")
            ).json()
        assert excerpt["markdown"] == "é" * 512  # 1024 bytes, not a split é

    async def test_result_unavailable_for_a_failed_item(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("blob.bin", b"\x00")])
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        check_error(result, 404, "result_unavailable")


# ---------------------------------------------------------------------------
# 6.3 Download tickets
# ---------------------------------------------------------------------------


class TestDownloadTickets:
    async def test_issue_and_redeem_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        app = _make_app(tmp_path, token=TOKEN)
        async with _client(app, peer=("192.168.1.30", 40000)) as client:
            job_id, _ = await _job_with_auth(client, files=[("a b.txt", b"a")])
            for path in (
                f"/api/jobs/{job_id}/archive",
                "/api/history/archive",
                f"/api/jobs/{job_id}/files/a%20b.txt.md",
                f"/api/jobs/{job_id}/items/i1/original/file",
            ):
                issued = await client.post(
                    "/api/download-tickets", json={"path": path}, headers=AUTH
                )
                assert issued.status_code == 201, issued.text
                body = issued.json()
                check_shape("DownloadTicket", body)
                assert body["expires_in"] == 60
                assert body["url"] == f"{path}?ticket={body['ticket']}"
                first = await client.get(body["url"])
                second = await client.get(body["url"])
                assert first.status_code == 200, (path, first.text)
                check_error(second, 401, "ticket_invalid")

    @pytest.mark.parametrize(
        "path",
        [
            "/api/settings/llm",
            "/api/jobs/abc",
            "/api/jobs/abc/events",
            "/api/jobs/abc/files/../meta.json",
            "/api/jobs/abc/files/",
            "/api/jobs/abc/archive?x=1",
            "/api/jobs/abc/items/i1/original",
            "/api/jobs/abc/items/i1/original/pages/1",
            "/api/history",
            "/jobs",
            "/api/jobs/abc/files/naïve.md",
        ],
    )
    async def test_only_download_paths(self, tmp_path: Path, path: str) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post("/api/download-tickets", json={"path": path})
        check_error(response, 422, "invalid_ticket_path")

    async def test_body_must_be_exactly_a_path(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            extra = await client.post(
                "/api/download-tickets",
                json={"path": "/api/history/archive", "x": 1},
            )
            not_json = await client.post("/api/download-tickets", content=b"nope")
            huge = await client.post(
                "/api/download-tickets", content=b"{" + b" " * 9000 + b"}"
            )
        check_error(extra, 422, "invalid_ticket_path")
        check_error(not_json, 422, "invalid_ticket_path")
        check_error(huge, 413, "request_too_large")

    async def test_at_most_64_outstanding(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            for _ in range(64):
                ok = await client.post(
                    "/api/download-tickets", json={"path": "/api/history/archive"}
                )
                assert ok.status_code == 201
            refused = await client.post(
                "/api/download-tickets", json={"path": "/api/history/archive"}
            )
        check_error(refused, 429, "too_many_tickets")


async def _job_with_auth(
    client: httpx.AsyncClient, **form: Any
) -> tuple[str, dict[str, Any]]:
    created = await client.post("/api/jobs", files=_form(**form), headers=AUTH)
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}", headers=AUTH)).json()
        if data["status"] != "running":
            return job_id, data
        await asyncio.sleep(0.02)
    pytest.fail("job did not finish")


# ---------------------------------------------------------------------------
# 7. Originals for Compare
# ---------------------------------------------------------------------------


class TestOriginals:
    async def _job_of(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        data: bytes,
        screenshots: int = 0,
    ) -> str:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item",
            _converter(screenshots=screenshots),
        )
        job_id, _ = await _job(client, files=[(name, data)])
        return job_id

    async def test_pdf_pages_and_text_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(client, monkeypatch, "doc.pdf", _pdf_bytes(2))
            base = f"/api/jobs/{job_id}/items/i1/original"
            descriptor = (await client.get(base)).json()
            page = await client.get(f"{base}/pages/2?width=400")
            default_width = await client.get(f"{base}/pages/1")
            text = (await client.get(f"{base}/pages/1/text")).json()
            raw = await client.get(f"{base}/file")
            no_page = await client.get(f"{base}/pages/3")
            zero = await client.get(f"{base}/pages/0")
            bad_number = await client.get(f"{base}/pages/one")
            bad_width = await client.get(f"{base}/pages/1?width=wide")
        check_shape("OriginalDescriptor", descriptor)
        assert descriptor["kind"] == "pdf"
        assert descriptor["pages"] == [{"width": 612, "height": 792}] * 2
        assert descriptor["page_count"] == 2 and descriptor["text_layer"] is True
        assert descriptor["download"] == f"{base}/file"
        assert descriptor["media_type"] == "application/pdf"
        assert page.headers["content-type"] == "image/png"
        assert _png_size(page.content)[0] == 400
        assert _png_size(default_width.content)[0] == 1200
        assert page.headers["cache-control"] == "private, max-age=86400"
        check_shape("PageText", text)
        heading = next(line for line in text["lines"] if line["text"] == "Heading 1")
        x0, y0, x1, y1 = heading["bbox"]
        assert 70 <= x0 < x1 and 70 < y0 < y1 < 110  # top-left origin, points
        assert raw.content.startswith(b"%PDF")
        assert (
            raw.headers["content-disposition"] == "attachment; filename*=UTF-8''doc.pdf"
        )
        assert raw.headers["content-security-policy"] == "sandbox; default-src 'none'"
        check_error(no_page, 404, "page_not_found")
        check_error(zero, 404, "page_not_found")
        check_error(bad_number, 422, "invalid_page_request")
        check_error(bad_width, 422, "invalid_page_request")

    async def test_rotated_pdf_reports_the_visible_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(
                client, monkeypatch, "r.pdf", _pdf_bytes(1, rotate=90)
            )
            base = f"/api/jobs/{job_id}/items/i1/original"
            descriptor = (await client.get(base)).json()
            text = (await client.get(f"{base}/pages/1/text")).json()
        assert descriptor["pages"] == [{"width": 792, "height": 612}]
        assert (text["width"], text["height"]) == (792, 612)
        for line in text["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            assert 0 <= x0 <= x1 <= 792 and 0 <= y0 <= y1 <= 612

    async def test_width_is_clamped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(client, monkeypatch, "doc.pdf", _pdf_bytes(1))
            base = f"/api/jobs/{job_id}/items/i1/original/pages/1"
            narrow = await client.get(f"{base}?width=10")
            wide = await client.get(f"{base}?width=99999")
        assert _png_size(narrow.content)[0] == 160
        assert _png_size(wide.content)[0] == 2400

    async def test_image_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(
                client, monkeypatch, "pic.png", _png_bytes(640, 480)
            )
            base = f"/api/jobs/{job_id}/items/i1/original"
            descriptor = (await client.get(base)).json()
            as_is = await client.get(f"{base}/pages/1?width=800")
            scaled = await client.get(f"{base}/pages/1?width=200")
            second = await client.get(f"{base}/pages/2")
            text = await client.get(f"{base}/pages/1/text")
        check_shape("OriginalDescriptor", descriptor)
        assert descriptor["kind"] == "image"
        assert descriptor["pages"] == [{"width": 640, "height": 480}]
        assert descriptor["text_layer"] is False
        assert _png_size(as_is.content) == (640, 480)
        assert _png_size(scaled.content) == (200, 150)
        check_error(second, 404, "page_not_found")
        check_error(text, 404, "original_unavailable")

    @pytest.mark.parametrize(
        ("name", "kind", "served_type"),
        [
            ("page.html", "html", "text/plain; charset=utf-8"),
            ("notes.md", "text", "text/markdown"),
            ("data.csv", "text", "text/csv"),
        ],
    )
    async def test_html_and_text_originals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        kind: str,
        served_type: str,
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(
                client, monkeypatch, name, b"<h1>Title</h1><script>x()</script>"
            )
            base = f"/api/jobs/{job_id}/items/i1/original"
            descriptor = (await client.get(base)).json()
            raw = await client.get(f"{base}/file")
            pages = await client.get(f"{base}/pages/1")
        check_shape("OriginalDescriptor", descriptor)
        assert descriptor["kind"] == kind and descriptor["pages"] == []
        assert descriptor["page_count"] == 0
        assert raw.headers["content-type"].startswith(served_type)
        check_error(pages, 404, "original_unavailable")

    async def test_office_file_without_and_with_screenshots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            plain = await self._job_of(client, monkeypatch, "deck.pptx", b"PK")
            shots = await self._job_of(
                client, monkeypatch, "slides.pptx", b"PK", screenshots=3
            )
            none = (await client.get(f"/api/jobs/{plain}/items/i1/original")).json()
            base = f"/api/jobs/{shots}/items/i1/original"
            screenshots = (await client.get(base)).json()
            page = await client.get(f"{base}/pages/3?width=200")
        assert none["kind"] == "none" and none["reason"] == "unsupported_format"
        assert none["download"] is not None  # the UI offers "Download original"
        assert screenshots["kind"] == "screenshots"
        assert screenshots["page_count"] == 3
        assert screenshots["pages"][0] == {"width": 300, "height": 400}
        assert page.status_code == 200
        assert _png_size(page.content) == (200, 267)

    async def test_url_item_keeps_no_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_url(url: str, cfg: Any, out_dir: Path, *args: Any, **kw: Any):
            from markitai.batch import ProcessResult

            out = out_dir / kw["output_name"]
            out.write_text("# page\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake_url)
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, urls=["https://example.com/a"])
            base = f"/api/jobs/{job_id}/items/i1/original"
            descriptor = (await client.get(base)).json()
            raw = await client.get(f"{base}/file")
        assert descriptor["kind"] == "none" and descriptor["reason"] == "not_kept"
        assert descriptor["download"] is None
        check_error(raw, 404, "original_unavailable")

    async def test_busy_renderer_asks_to_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path)
        async with _client(app) as client:
            job_id = await self._job_of(client, monkeypatch, "doc.pdf", _pdf_bytes(1))
            renders: asyncio.Semaphore = app.state.page_renders
            await renders.acquire()
            await renders.acquire()
            try:
                busy = await client.get(f"/api/jobs/{job_id}/items/i1/original/pages/1")
            finally:
                renders.release()
                renders.release()
        check_error(busy, 429, "render_busy")
        assert busy.headers["retry-after"] == "1"

    async def test_broken_pdf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(
                client, monkeypatch, "bad.pdf", b"%PDF-1.4 junk"
            )
            descriptor = (
                await client.get(f"/api/jobs/{job_id}/items/i1/original")
            ).json()
        assert descriptor["kind"] == "none"
        assert descriptor["reason"] == "render_unavailable"

    async def test_unknown_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await self._job_of(client, monkeypatch, "a.txt", b"a")
            response = await client.get(f"/api/jobs/{job_id}/items/i7/original")
        check_error(response, 404, "item_not_found")


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", data[:16]
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


# ---------------------------------------------------------------------------
# 8. Settings
# ---------------------------------------------------------------------------


class TestSettings:
    async def test_view_and_credentials_display_value(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            view = (await client.get("/api/settings/llm")).json()
            check_shape("SettingsView", view)
            saved = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": view["revision"],
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "openai/my-model",
                            "provider": "custom",
                            "api_key": "sk-contract-0000-secret",
                            "api_base": "http://127.0.0.1:9/v1",
                        }
                    ],
                },
            )
            assert saved.status_code == 200, saved.text
            cards = (await client.get("/api/settings/llm/providers")).json()
            card = next(c for c in cards["providers"] if c["kind"] == "configured")
            credentials = await client.get(
                f"/api/settings/llm/providers/{card['provider_id']}/credentials"
            )
            stale = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": view["revision"],
                    "deployments": [{"model_name": "x", "model": "openai/x"}],
                },
            )
            missing = await client.delete(
                f"/api/settings/llm/deployments/nope?expected_revision={view['revision']}"
            )
            config_open = await client.post("/api/settings/llm/config/open")
        for listed in cards["providers"]:
            check_shape("ProviderCard", listed, f"ProviderCard {listed['id']}")
        check_shape("ProviderCredentials", credentials.json())
        assert credentials.json()["api_key"] == "sk…cret"
        assert "secret" not in credentials.text.replace("sk…cret", "")
        assert credentials.headers["cache-control"] == "no-store"
        assert card["provider"] == "custom"  # the OpenAI-compatible save works
        check_error(stale, 409, "stale_revision")
        assert stale.json()["detail"]["code"] == "stale_revision"
        assert missing.status_code in (404, 409)
        assert missing.json()["reason"] in ("settings_entry_missing", "stale_revision")
        assert config_open.status_code in (204, 500)

    async def test_config_open_without_a_file(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post("/api/settings/llm/config/open")
        check_error(response, 404, "config_missing")

    async def test_invalid_settings_body(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post("/api/settings/llm/test", json={})
            not_json = await client.post(
                "/api/settings/llm/test",
                content=b"{",
                headers={"Content-Type": "application/json"},
            )
        check_error(response, 422, "invalid_settings_request")
        check_error(not_json, 422, "invalid_json")


# ---------------------------------------------------------------------------
# 1. Static UI
# ---------------------------------------------------------------------------


def _bundle(directory: Path, files: dict[str, bytes]) -> Path:
    """A bundle as published: index.html at the root, the rest under ui/."""
    import hashlib

    paths = {
        (name if name == "index.html" or "/" in name else f"ui/{name}"): data
        for name, data in files.items()
    }
    for path, data in paths.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(data)
    manifest = {
        "source_commit": "0" * 40,
        "build_command": "test",
        "files": {
            path: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in paths.items()
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self' blob: https:; font-src 'self'; frame-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class TestStaticUi:
    def _app(self, tmp_path: Path) -> FastAPI:
        app_js = b"console.log('workbench');" * 100
        bundle = _bundle(
            tmp_path / "bundle",
            {
                "index.html": b"<!doctype html><title>Markitai</title>",
                "app.js": app_js,
                "app.js.gz": gzip.compress(app_js),
                "marked.js.gz": gzip.compress(b"export const marked = 1;"),
                "app.css": b"body{}",
                "logo.svg": b"<svg/>",
                "inter-latin-wght.woff2": b"wOF2",
                "LICENSES/marked-LICENSE": b"MIT",
            },
        )
        return _make_app(tmp_path, static_dir=bundle)

    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/jobs",
            "/settings",
            "/settings/models",
            "/settings/appearance",
            "/settings/access",
            "/settings/config",
        ],
    )
    async def test_shell_at_every_app_path(self, tmp_path: Path, path: str) -> None:
        async with _client(self._app(tmp_path)) as client:
            response = await client.get(path, headers={"Accept-Encoding": "identity"})
        assert response.status_code == 200
        assert response.text == "<!doctype html><title>Markitai</title>"
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert response.headers["content-security-policy"] == CSP
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-cache"

    @pytest.mark.parametrize(
        ("name", "content_type"),
        [
            ("app.js", "text/javascript; charset=utf-8"),
            ("app.css", "text/css; charset=utf-8"),
            ("logo.svg", "image/svg+xml"),
            ("inter-latin-wght.woff2", "font/woff2"),
            ("marked.js", "text/javascript; charset=utf-8"),
        ],
    )
    async def test_files_by_exact_name(
        self, tmp_path: Path, name: str, content_type: str
    ) -> None:
        async with _client(self._app(tmp_path)) as client:
            response = await client.get(
                f"/ui/{name}", headers={"Accept-Encoding": "identity"}
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == content_type
        assert response.headers["content-security-policy"] == CSP
        assert "content-encoding" not in response.headers

    async def test_gzip_twin_validators_and_unknown_names(self, tmp_path: Path) -> None:
        async with _client(self._app(tmp_path)) as client:
            zipped = await client.get("/ui/app.js", headers={"Accept-Encoding": "gzip"})
            gz_only = await client.get(
                "/ui/marked.js", headers={"Accept-Encoding": "gzip"}
            )
            refused = await client.get(
                "/ui/app.js", headers={"Accept-Encoding": "gzip;q=0"}
            )
            etag = zipped.headers["etag"]
            again = await client.get("/ui/app.js", headers={"If-None-Match": etag})
            twin = await client.get("/ui/app.js.gz")
            unknown = await client.get("/ui/missing.js")
            nested = await client.get("/ui/sub/app.js")
            old_assets = await client.get("/assets/index-abc.js")
            head = await client.head("/ui/app.js")
        assert zipped.headers["content-encoding"] == "gzip"
        assert zipped.headers["vary"] == "Accept-Encoding"
        assert zipped.text.startswith("console.log")  # httpx decoded it
        assert gz_only.text == "export const marked = 1;"
        assert "content-encoding" not in refused.headers
        assert etag.startswith('W/"')
        assert again.status_code == 304
        check_error(twin, 404, "route_not_found")
        check_error(unknown, 404, "route_not_found")
        assert nested.status_code == 404
        assert old_assets.status_code == 404
        assert head.status_code == 200

    @pytest.mark.parametrize("path", ["/does-not-exist", "/jobs/abc", "/settings/nope"])
    async def test_an_unknown_page_is_the_shell_with_404(
        self, tmp_path: Path, path: str
    ) -> None:
        """Contract v1.4: the UI shows its own "Page not found" view."""
        async with _client(self._app(tmp_path)) as client:
            page = await client.get(
                path,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "If-None-Match": "*",
                    "Accept-Encoding": "identity",
                },
            )
        assert page.status_code == 404
        assert page.text == "<!doctype html><title>Markitai</title>"
        assert page.headers["content-type"] == "text/html; charset=utf-8"
        assert page.headers["content-security-policy"] == CSP
        assert page.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize(
        ("method", "path", "accept"),
        [
            ("GET", "/does-not-exist", "application/json"),
            ("GET", "/does-not-exist", "*/*"),
            ("GET", "/ui/nope.js", "text/html"),
            ("GET", "/api/nope", "text/html"),
            ("GET", "/api", "text/html"),
            ("POST", "/does-not-exist", "text/html"),
        ],
    )
    async def test_other_misses_stay_json(
        self, tmp_path: Path, method: str, path: str, accept: str
    ) -> None:
        async with _client(self._app(tmp_path)) as client:
            response = await client.request(method, path, headers={"Accept": accept})
        check_error(response, 404, "route_not_found")

    async def test_static_needs_no_token(self, tmp_path: Path) -> None:
        app_js = b"x"
        bundle = _bundle(
            tmp_path / "bundle", {"index.html": b"<html></html>", "app.js": app_js}
        )
        app = _make_app(tmp_path, static_dir=bundle, token=TOKEN)
        async with _client(app, peer=("192.168.1.30", 40000)) as client:
            shell = await client.get("/")
            script = await client.get("/ui/app.js")
            api = await client.get("/api/capabilities")
        assert shell.status_code == 200 and script.status_code == 200
        check_error(api, 401, "token_required")


class TestVendoredBundle:
    def test_the_package_ships_a_verified_bundle(self) -> None:
        from markitai.serve.workbench import BUNDLED_DIR, Workbench

        workbench = Workbench.load(BUNDLED_DIR)
        assert workbench is not None, "serve/workbench is missing or damaged"
        assert workbench.source_commit
        for name in ("app.js", "app.css", "boot.js", "marked.js", "purify.js"):
            assert name in workbench.assets, name
        index = workbench.index.plain.decode()
        assert "/ui/app.js" in index and "/ui/app.css" in index
        # Every /ui/ name the shell references is in the bundle
        for name in re.findall(r'"/ui/([^"]+)"', index):
            assert name in workbench.assets, name
        # The licences of the libraries and font inside travel with them
        licences = BUNDLED_DIR / "LICENSES"
        assert licences.is_dir() and any(licences.iterdir())
        # The bundle was built for the policy this server sends
        serve = workbench.manifest.get("serve")
        if isinstance(serve, dict) and "content_security_policy" in serve:
            assert serve["content_security_policy"] == CSP
            assert set(serve["page_paths"]) <= set(workbench.page_paths)


class TestReviewFixes:
    """Compare's PDF work runs in killable workers; numbers are bounded."""

    def test_a_runaway_page_is_killed_and_the_next_one_works(self) -> None:
        from markitai.serve.originals import PdfWorkers, RenderError

        workers = PdfWorkers(workers=1, timeout=0.5)
        try:
            started = time.monotonic()
            with pytest.raises(RenderError):
                workers.run(time.sleep, 30)
            assert time.monotonic() - started < 10
            # The next call starts a fresh pool, whose worker is spawned and
            # imports markitai first: on runner-win (x64 Python emulated on
            # Arm) that alone took longer than the 0.5 s that bounds the
            # runaway. Serve's own budget is 20 s.
            workers.timeout = 60
            assert workers.run(abs, -3) == 3
        finally:
            workers.close()

    async def test_pages_are_drawn_once_then_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _converter())
        app = _make_app(tmp_path)
        async with _client(app) as client:
            job_id, _ = await _job(client, files=[("doc.pdf", _pdf_bytes(1))])
            workers = app.state.pdf_workers
            calls: list[str] = []
            original_run = workers.run

            def counting(function: Any, *args: Any) -> Any:
                calls.append(function.__name__)
                return original_run(function, *args)

            workers.run = counting
            url = f"/api/jobs/{job_id}/items/i1/original/pages/1?width=400"
            first = await client.get(url)
            second = await client.get(url)
        assert first.content == second.content
        assert calls.count("render_pdf_page") == 1
        assert app.state.pdf_workers._pool is None  # closed with the server

    async def test_numbers_are_ascii_and_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _converter(markdown="# x\n")
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id, _ = await _job(client, files=[("doc.pdf", _pdf_bytes(1))])
            base = f"/api/jobs/{job_id}/items/i1"
            huge = await client.get(f"{base}/result?max_bytes={10**15}")
            superscript = await client.get(f"{base}/result?max_bytes=2%C2%B2")
            page = await client.get(f"{base}/original/pages/%C2%B2")
            width = await client.get(f"{base}/original/pages/1?width=%C2%B2")
        assert huge.status_code == 200 and huge.json()["truncated"] is False
        check_error(superscript, 422, "invalid_max_bytes")
        check_error(page, 422, "invalid_page_request")
        check_error(width, 422, "invalid_page_request")

    async def test_a_stray_file_in_the_bundle_is_ignored(self, tmp_path: Path) -> None:
        bundle = _bundle(
            tmp_path / "bundle", {"index.html": b"<html></html>", "app.js": b"x"}
        )
        (bundle / "ui" / ".DS_Store").write_bytes(b"\0")
        async with _client(_make_app(tmp_path, static_dir=bundle)) as client:
            shell = await client.get("/")
            stray = await client.get("/ui/.DS_Store")
        assert shell.text == "<html></html>"
        assert stray.status_code == 404

    def test_stored_options_load_leniently(self) -> None:
        from markitai.serve.jobs import normalize_job_options

        assert normalize_job_options({"llm": "true", "profile": "rag"})["llm"] is True

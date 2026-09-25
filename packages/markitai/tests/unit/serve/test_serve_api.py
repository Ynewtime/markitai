"""Tests for the markitai serve REST + SSE API.

All tests run against ``create_app`` with a tmp_path-backed jobs root and an
injected default config, over httpx.ASGITransport. Note ASGITransport buffers
responses, so SSE tests gate the (monkeypatched) converter until the stream
has subscribed and then parse the fully buffered event stream.
"""

from __future__ import annotations

import asyncio
import json
import time
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx


def _all_job_options(**set_values: object) -> dict[str, object]:
    """Every JobOptions field, defaulting to None.

    Written from the model rather than by hand: the snapshot echoes the
    options back in full, so a hand-listed expectation goes stale the next
    time an option is added.
    """
    from markitai.serve.schemas import JobOptions

    return {name: set_values.get(name) for name in JobOptions.model_fields}


from markitai.config import MarkitaiConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def _make_app(tmp_path: Path, cfg: MarkitaiConfig | None = None) -> FastAPI:
    """Build an app with hermetic config and tmp_path-backed jobs root."""
    cfg = cfg or MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        # A nonexistent static dir disables the SPA mount even when the repo
        # webapp/dist has been built — keeps these tests hermetic.
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
    )


@asynccontextmanager
async def _serve_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app lifespan and yield an ASGI-backed client."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


def _multipart(
    files: list[tuple[str, bytes]] | None = None,
    urls: list[str] | None = None,
    options: dict[str, Any] | None = None,
    raw_options: str | None = None,
) -> list[tuple[str, Any]]:
    """Build httpx multipart payload for POST /api/jobs."""
    parts: list[tuple[str, Any]] = []
    for name, content in files or []:
        parts.append(("files", (name, content, "application/octet-stream")))
    parts.append(("urls", (None, json.dumps(urls or []))))
    opts = raw_options if raw_options is not None else json.dumps(options or {})
    parts.append(("options", (None, opts)))
    return parts


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event, data) tuples, ignoring comments."""
    events: list[tuple[str, dict[str, Any]]] = []
    event_name: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name is not None:
            events.append((event_name, json.loads(line.split(":", 1)[1].strip())))
            event_name = None
    return events


async def _wait_job_done(
    client: httpx.AsyncClient, job_id: str, timeout: float = 60.0
) -> dict[str, Any]:
    """Poll GET /api/jobs/{id} until the job reaches its terminal state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


class TestCapabilitiesAndRoot:
    """GET /api/capabilities and the JSON hint at /."""

    async def test_capabilities_shape(self, tmp_path: Path) -> None:
        from markitai import __version__
        from markitai.serve.app import MAX_JOB_ITEMS

        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == __version__
        assert data["llm"] == {
            "configured": False,
            "routable": False,
            "effective": False,
            "models": [],
        }
        assert data["presets"] == ["minimal", "standard", "rich"]
        from markitai.config import BUILTIN_PRESETS

        assert data["preset_options"] == {
            name: preset.model_dump() for name, preset in BUILTIN_PRESETS.items()
        }
        assert set(data["extras"]) == {"browser", "svg"}
        assert all(isinstance(v, bool) for v in data["extras"].values())
        # The webapp reads server-owned limits from here (no constant copies).
        assert data["limits"] == {"max_job_items": MAX_JOB_ITEMS}

    async def test_capabilities_exposes_configured_preset_values(
        self, tmp_path: Path
    ) -> None:
        from markitai.config import PresetConfig

        cfg = MarkitaiConfig()
        cfg.presets["rich"] = PresetConfig(llm=True, ocr=True, desc=True)
        app = create_app(
            jobs_root=tmp_path / "jobs", config=cfg, configure_logging=False
        )
        async with _serve_client(app) as client:
            data = (await client.get("/api/capabilities")).json()
        assert data["preset_options"]["rich"] == cfg.presets["rich"].model_dump()

    async def test_capabilities_reports_configured_models(self, tmp_path: Path) -> None:
        from markitai.config import LiteLLMParams, ModelConfig

        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini"),
            )
        ]
        app = create_app(
            jobs_root=tmp_path / "jobs", config=cfg, configure_logging=False
        )
        async with _serve_client(app) as client:
            data = (await client.get("/api/capabilities")).json()
        assert data["llm"] == {
            "configured": True,
            "routable": False,  # no key is available for this deployment
            "effective": False,
            "models": ["openai/gpt-4o-mini"],
        }

    async def test_root_returns_json_hint_without_static(self, tmp_path: Path) -> None:
        from markitai import __version__

        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["markitai"] == __version__
        assert "hint" in data

    async def test_static_dir_is_mounted_as_spa(self, tmp_path: Path) -> None:
        static = tmp_path / "static"
        static.mkdir()
        (static / "index.html").write_text("<html>ui</html>", encoding="utf-8")
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        app = create_app(
            static_dir=static,
            jobs_root=tmp_path / "jobs",
            config=cfg,
            configure_logging=False,
        )
        async with _serve_client(app) as client:
            root = await client.get("/")
            spa_route = await client.get("/jobs/some-client-route")
            api = await client.get("/api/capabilities")
        assert root.status_code == 200 and "ui" in root.text
        assert spa_route.status_code == 200 and "ui" in spa_route.text
        assert api.status_code == 200  # /api keeps priority over the SPA mount

    async def test_hashed_assets_are_cached_and_the_index_revalidated(
        self, tmp_path: Path
    ) -> None:
        static = tmp_path / "static"
        (static / "assets").mkdir(parents=True)
        (static / "index.html").write_text("<html>ui</html>", encoding="utf-8")
        (static / "assets" / "index-abc123.js").write_text(
            "console.log('ui');" * 200, encoding="utf-8"
        )
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        app = create_app(
            static_dir=static,
            jobs_root=tmp_path / "jobs",
            config=cfg,
            configure_logging=False,
        )
        async with _serve_client(app) as client:
            asset = await client.get(
                "/assets/index-abc123.js", headers={"Accept-Encoding": "gzip"}
            )
            root = await client.get("/")
            spa_route = await client.get("/jobs/some-client-route")
        assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert asset.headers["content-encoding"] == "gzip"
        assert asset.text.startswith("console.log")  # httpx decoded it
        assert root.headers["cache-control"] == "no-cache"
        assert spa_route.headers["cache-control"] == "no-cache"


class TestShutdownStopsPdfWorkers:
    async def test_the_lifespan_stops_the_extraction_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """uvicorn re-raises its stop signal after the lifespan ends, which
        skips atexit and finalize_process: the workers must stop here."""
        from markitai.converter import pdf_parallel

        monkeypatch.setattr(pdf_parallel, "_enabled", True)
        async with _serve_client(_make_app(tmp_path)):
            pdf_parallel._get_pool(2)
            assert pdf_parallel._pool is not None
        assert pdf_parallel._pool is None


class TestCompression:
    """JSON and Markdown compress; event streams and downloads never do."""

    async def test_json_is_gzipped_when_the_client_accepts_it(
        self, tmp_path: Path
    ) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get(
                "/api/capabilities", headers={"Accept-Encoding": "gzip"}
            )
        assert resp.status_code == 200
        assert resp.headers.get("content-encoding") == "gzip"
        assert resp.json()["version"]

    def test_streams_and_downloads_bypass_the_compressor(self) -> None:
        from markitai.serve.app import _CompressionMiddleware

        seen: list[str] = []

        async def inner(scope: Any, receive: Any, send: Any) -> None:
            seen.append(scope["path"])

        middleware = _CompressionMiddleware(inner)
        middleware._gzip = None  # type: ignore[assignment]  # must not be reached

        async def run(path: str) -> None:
            await middleware({"type": "http", "path": path}, None, None)

        for path in (
            "/api/jobs/j1/events",
            "/api/jobs/j1/archive",
            "/api/history/archive",
            "/api/jobs/j1/files/out/a.png",
        ):
            asyncio.run(run(path))
        assert len(seen) == 4


class TestJobCreationValidation:
    """POST /api/jobs input validation."""

    async def test_empty_input_is_422(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post("/api/jobs", files=_multipart())
        assert resp.status_code == 422

    async def test_invalid_options_json_is_422(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.txt", b"hi")], raw_options="not json"),
            )
        assert resp.status_code == 422

    async def test_unknown_options_key_is_422(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.txt", b"hi")], raw_options='{"bogus": 1}'),
            )
        assert resp.status_code == 422

    async def test_unknown_preset_is_422(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.txt", b"hi")], options={"preset": "nope"}),
            )
        assert resp.status_code == 422

    async def test_invalid_urls_payload_is_422(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=[("urls", (None, '"not-a-list"')), ("options", (None, "{}"))],
            )
        assert resp.status_code == 422

    async def test_too_many_items_is_422(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.app.MAX_JOB_ITEMS", 3)
        urls = [f"https://example.com/{i}" for i in range(4)]
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post("/api/jobs", files=_multipart(urls=urls))
        assert resp.status_code == 422
        assert "max 3" in resp.json()["detail"]

    def test_a_folder_drop_fits_like_a_cli_directory_batch(self) -> None:
        """The web cap follows Starlette's per-request file limit, not 50."""
        from markitai.serve.app import MAX_JOB_ITEMS, MAX_REQUEST_BYTES

        assert MAX_JOB_ITEMS == 1000
        assert MAX_REQUEST_BYTES >= 5 * 1024**3

    async def test_oversized_upload_is_413(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.app.MAX_UPLOAD_BYTES", 8)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart(files=[("big.txt", b"0123456789abcdef")]),
            )
            assert resp.status_code == 413
            # Creation-time rollback: no half-created job directories remain
            jobs_root = app.state.markitai.registry.jobs_root
            assert list(jobs_root.iterdir()) == []

    async def test_creation_failure_rolls_back_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-HTTPException failures must not leave a zombie 'running' job."""

        async def exploding_save(upload: Any, dest_dir: Path) -> Path:
            raise OSError(63, "File name too long")

        monkeypatch.setattr("markitai.serve.app._save_upload", exploding_save)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            resp = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hi")])
            )
            assert resp.status_code == 500
            registry = app.state.markitai.registry
            assert registry.jobs == {}  # no zombie registry entry
            assert list(registry.jobs_root.iterdir()) == []  # job dir removed

    async def test_oversized_content_length_is_413_before_parsing(
        self, tmp_path: Path
    ) -> None:
        from markitai.serve.app import MAX_REQUEST_BYTES

        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                content=b"tiny",
                headers={
                    "content-length": str(MAX_REQUEST_BYTES + 1),
                    "content-type": "multipart/form-data; boundary=x",
                },
            )
        assert resp.status_code == 413
        # Middleware-produced errors carry the same machine code as route ones.
        assert resp.json()["code"] == "payload_too_large"

    async def test_unknown_job_and_item_are_404(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            assert (await client.get("/api/jobs/nope")).status_code == 404
            assert (
                await client.get("/api/jobs/nope/items/i1/result")
            ).status_code == 404
            assert (await client.get("/api/jobs/nope/archive")).status_code == 404

    async def test_error_bodies_carry_a_machine_code(self, tmp_path: Path) -> None:
        """``detail`` stays human; ``code`` is the stable part clients branch on."""
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/jobs/nope")

        assert resp.status_code == 404
        body = resp.json()
        assert body["code"] == "not_found"
        assert isinstance(body["detail"], str) and body["detail"]


class TestJobConfigMapping:
    """Preset + llm override semantics (mirrors the CLI mapping)."""

    def _build(self, base: MarkitaiConfig, **options: Any) -> MarkitaiConfig:
        from markitai.serve.app import _build_job_config
        from markitai.serve.schemas import JobOptions

        return _build_job_config(base, JobOptions(**options))

    def _base_with_model(self) -> MarkitaiConfig:
        from markitai.config import LiteLLMParams, ModelConfig

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini"),
            )
        ]
        return cfg

    def test_profile_reaches_the_config(self) -> None:
        cfg = self._build(MarkitaiConfig(), profile="rag")
        assert cfg.output.profile == "rag"

    def test_profile_survives_a_conversion_with_no_llm(self) -> None:
        """A profile shapes the output, so it is not an LLM feature.

        Gating it on the LLM would put it out of reach of exactly the users
        who convert locally, which is who the visible assets/ layout helps
        most.
        """
        cfg = self._build(MarkitaiConfig(), llm=False, profile="obsidian")
        assert cfg.llm.enabled is False
        assert cfg.output.profile == "obsidian"

    def test_omitting_the_profile_leaves_the_base_config_alone(self) -> None:
        base = MarkitaiConfig()
        base.output.profile = "okf"
        assert self._build(base, preset="minimal").output.profile == "okf"

    def test_preset_rich_maps_all_five_booleans(self) -> None:
        cfg = self._build(self._base_with_model(), preset="rich")
        assert cfg.llm.enabled is True
        assert cfg.image.alt_enabled is True
        assert cfg.image.desc_enabled is True
        assert cfg.ocr.enabled is False
        assert cfg.screenshot.enabled is True

    def test_preset_rich_with_llm_false_override(self) -> None:
        cfg = self._build(self._base_with_model(), preset="rich", llm=False)
        assert cfg.llm.enabled is False
        assert cfg.image.alt_enabled is True  # preset values kept
        assert cfg.screenshot.enabled is True

    def test_preset_minimal_disables_features(self) -> None:
        base = self._base_with_model()
        base.llm.enabled = True
        base.image.alt_enabled = True
        cfg = self._build(base, preset="minimal")
        assert cfg.llm.enabled is False
        assert cfg.image.alt_enabled is False

    def test_explicit_ocr_overrides_preset(self) -> None:
        cfg = self._build(self._base_with_model(), preset="minimal", ocr=True)
        assert cfg.ocr.enabled is True
        cfg = self._build(self._base_with_model(), preset="rich", ocr=False)
        assert cfg.ocr.enabled is False

    def test_explicit_image_overrides_can_disable_a_rich_preset(self) -> None:
        cfg = self._build(
            self._base_with_model(),
            preset="rich",
            alt=False,
            desc=False,
            screenshot=False,
        )
        assert not cfg.image.alt_enabled
        assert not cfg.image.desc_enabled
        assert not cfg.screenshot.enabled

    def test_screenshot_source_implies_capture_but_not_llm(self) -> None:
        cfg = self._build(
            MarkitaiConfig(), preset="minimal", screenshot=False, screenshot_only=True
        )
        assert cfg.screenshot.enabled
        assert cfg.screenshot.screenshot_only
        assert not cfg.llm.enabled

    @pytest.mark.parametrize("skip", [False, True])
    def test_no_cache_controls_cache_reads_without_changing_storage(
        self, skip: bool
    ) -> None:
        base = self._base_with_model()
        base.cache.no_cache = not skip
        cfg = self._build(base, no_cache=skip)
        assert cfg.cache.no_cache is skip
        assert cfg.cache.enabled == base.cache.enabled
        assert base.cache.no_cache is not skip

    @pytest.mark.parametrize("backend", ["native", "cloudflare"])
    def test_backend_replaces_the_inherited_converter_flag(self, backend: str) -> None:
        base = self._base_with_model()
        base.fetch.cloudflare.convert_enabled = True
        cfg = self._build(base, backend=backend)
        assert cfg.fetch.cloudflare.convert_enabled is (backend == "cloudflare")
        assert base.fetch.cloudflare.convert_enabled

    def test_llm_true_without_models_degrades_to_disabled(self) -> None:
        cfg = self._build(MarkitaiConfig(), llm=True)
        assert cfg.llm.enabled is False

    def test_base_config_is_not_mutated(self) -> None:
        base = self._base_with_model()
        assert base.llm.enabled is False
        self._build(base, preset="rich")
        assert base.llm.enabled is False

    def test_llm_enabled_forces_keep_base(self) -> None:
        """Web semantics: LLM jobs always keep the base .md (diff view)."""
        base = self._base_with_model()
        cfg = self._build(base, llm=True)
        assert cfg.llm.enabled is True
        assert cfg.llm.keep_base is True
        assert base.llm.keep_base is False  # base config untouched

    def test_keep_base_not_forced_when_llm_disabled(self) -> None:
        base = self._base_with_model()
        assert self._build(base).llm.keep_base is False
        assert self._build(base, llm=False).llm.keep_base is False
        assert self._build(base, preset="minimal").llm.keep_base is False
        # llm=True without configured models degrades to disabled: no keep_base
        assert self._build(MarkitaiConfig(), llm=True).llm.keep_base is False


class TestJobLifecycle:
    """Job execution, SSE, results and downloads with a stubbed converter."""

    @staticmethod
    def _fake_converter(gate: asyncio.Event | None = None, cost: float = 0.25):
        async def fake_process_file_item(
            file_path: Path, cfg: Any, out_dir: Path, shared: Any
        ):
            from markitai.batch import ProcessResult

            if gate is not None:
                await gate.wait()
            out = out_dir / f"{file_path.name}.md"
            out.write_bytes(f"# converted {file_path.name}\n".encode())
            return ProcessResult(success=True, output_path=str(out), cost_usd=cost)

        return fake_process_file_item

    async def test_sse_snapshot_then_item_lifecycle_to_terminal_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter(gate)
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]

            # Subscribe while the item is gated, then release it. The
            # buffered response contains the whole stream once it closes.
            sse_task = asyncio.create_task(client.get(f"/api/jobs/{job_id}/events"))
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            while not job.subscribers:
                await asyncio.sleep(0.01)
            gate.set()
            resp = await asyncio.wait_for(sse_task, timeout=30)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)

        assert events[0][0] == "snapshot"
        snapshot = events[0][1]
        assert snapshot["job_id"] == job_id
        assert snapshot["total"] == 1
        assert snapshot["items"][0]["item_id"] == "i1"

        item_events = [d for name, d in events if name == "item"]
        done_items = [d for d in item_events if d["status"] == "done"]
        assert done_items, f"no terminal item event in {events}"
        done = done_items[-1]
        assert done["item_id"] == "i1"
        assert done["name"] == "doc.txt"
        assert done["kind"] == "file"
        assert done["error"] is None
        assert done["output"] == "doc.txt.md"
        assert isinstance(done["duration_ms"], int)
        assert isinstance(done["finished_at"], str)
        assert done["cost_usd"] == 0.25
        assert done["skipped"] is False
        assert done["skip_reason"] is None

        assert events[-1][0] == "job"
        assert events[-1][1] == {
            "status": "done",
            "done": 1,
            "failed": 0,
            "total": 1,
        }

    async def test_sse_on_finished_job_sends_snapshot_and_terminal_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            resp = await client.get(f"/api/jobs/{job_id}/events")
        events = _parse_sse(resp.text)
        assert [name for name, _ in events] == ["snapshot", "job"]
        assert events[0][1]["status"] == "done"
        assert events[1][1]["status"] == "done"

    async def test_sse_drains_events_queued_before_terminal_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Item events queued while the snapshot frame is in flight are not
        dropped when the job turns terminal in that window."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            registry = app.state.markitai.registry
            job = registry.get(job_id)
            assert job is not None
            orig_subscribe = registry.subscribe

            def racing_subscribe(j: Any) -> Any:
                # Simulate the race: final item/job events land on the fresh
                # subscriber queue before the stream's terminal-status check.
                queue = orig_subscribe(j)
                queue.put_nowait(("item", j.items[0].to_payload()))
                queue.put_nowait(("job", j.progress_payload()))
                return queue

            monkeypatch.setattr(registry, "subscribe", racing_subscribe)
            resp = await client.get(f"/api/jobs/{job_id}/events")

        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert names[0] == "snapshot" and names[-1] == "job"
        item_frames = [d for name, d in events if name == "item"]
        assert item_frames, f"queued item frame was dropped: {events}"
        assert item_frames[0]["status"] == "done"
        assert item_frames[0]["output"] == "doc.txt.md"

    async def test_archive_on_running_job_is_409(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter(gate)
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            resp = await client.get(f"/api/jobs/{job_id}/archive")
            assert resp.status_code == 409
            gate.set()
            await _wait_job_done(client, job_id)
            done = await client.get(f"/api/jobs/{job_id}/archive")
            assert done.status_code == 200

    async def test_archive_excludes_atomic_write_droppings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            # ".<name>.<rand>.tmp" dropping as left by atomic_write_text
            (job.out_dir / ".doc.txt.md.x1y2z3.tmp").write_text(
                "partial", encoding="utf-8"
            )
            archive = await client.get(f"/api/jobs/{job_id}/archive")
        assert archive.status_code == 200
        with zipfile.ZipFile(BytesIO(archive.content)) as zf:
            assert zf.namelist() == ["doc.txt.md"]

    async def test_long_cjk_upload_name_is_byte_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 240-CJK-char name (720 UTF-8 bytes) must convert end to end."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        name = "汉" * 240 + ".txt"
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[(name, b"x")])
            )
            assert created.status_code == 201
            saved = created.json()["items"][0]["name"]
            assert saved.endswith(".txt")
            assert len(saved.encode("utf-8")) <= 180
            data = await _wait_job_done(client, created.json()["job_id"])
        item = data["items"][0]
        assert item["status"] == "done"
        assert item["output"] == f"{saved}.md"

    async def test_windows_reserved_upload_name_is_neutralized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("CON.txt", b"x"), ("nul", b"y")]),
            )
            assert created.status_code == 201
            names = [i["name"] for i in created.json()["items"]]
            assert names == ["_CON.txt", "_nul"]
            data = await _wait_job_done(client, created.json()["job_id"])
        assert data["done"] == 2 and data["failed"] == 0

    @staticmethod
    def _converter_with_asset(gate: asyncio.Event | None = None):
        """Fake converter that also drops a numbered asset for the item."""

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult
            from markitai.constants import ASSETS_REL_PATH

            if gate is not None:
                await gate.wait()
            out = out_dir / f"{file_path.name}.md"
            out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
            assets = out_dir / ASSETS_REL_PATH
            assets.mkdir(parents=True, exist_ok=True)
            (assets / f"{file_path.name}.0001.png").write_bytes(b"\x89PNG")
            return ProcessResult(success=True, output_path=str(out))

        return convert

    async def test_artifacts_with_glob_metachars_in_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.constants import ASSETS_REL_PATH

        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._converter_with_asset()
        )
        name = "report[2024].pdf"
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[(name, b"x")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        assert result.status_code == 200
        relpaths = {a["relpath"] for a in result.json()["artifacts"]}
        assert relpaths == {f"{name}.md", f"{ASSETS_REL_PATH}/{name}.0001.png"}

    async def test_artifact_prefix_does_not_bleed_across_items(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.constants import ASSETS_REL_PATH

        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._converter_with_asset()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("a", b"x"), ("a.txt", b"y")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        assert result.status_code == 200
        relpaths = {a["relpath"] for a in result.json()["artifacts"]}
        # Item "a" must not claim "a.txt.md" or "a.txt.0001.png".
        assert relpaths == {"a.md", f"{ASSETS_REL_PATH}/a.0001.png"}

    async def test_failed_item_reported_in_snapshot_and_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            raise RuntimeError("converter exploded")

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", failing)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
        assert data["failed"] == 1 and data["done"] == 0
        item = data["items"][0]
        assert item["status"] == "error"
        assert "converter exploded" in item["error"]
        assert data["options"] == _all_job_options()

    async def test_result_files_archive_and_cjk_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        name = "测试 文档.txt"
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[(name, "中文内容".encode())]),
            )
            assert created.status_code == 201
            body = created.json()
            assert body["items"][0]["name"] == name
            job_id = body["job_id"]
            data = await _wait_job_done(client, job_id)
            item = data["items"][0]
            assert item["output"] == f"{name}.md"

            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            assert result.status_code == 200
            payload = result.json()
            assert payload["name"] == name
            assert payload["variant"] == "base"
            assert f"converted {name}" in payload["markdown"]
            assert payload["artifacts"] == [
                {
                    "relpath": f"{name}.md",
                    "size": len(f"# converted {name}\n".encode()),
                }
            ]

            download = await client.get(f"/api/jobs/{job_id}/files/{item['output']}")
            assert download.status_code == 200
            assert "attachment" in download.headers["content-disposition"]
            assert f"converted {name}" in download.text

            archive = await client.get(f"/api/jobs/{job_id}/archive")
            assert archive.status_code == 200
            assert archive.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(BytesIO(archive.content)) as zf:
                assert zf.namelist() == [f"{name}.md"]

    async def test_path_traversal_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        app = _make_app(tmp_path)
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            for relpath in (
                "..%2F..%2F..%2Fsecret.txt",
                "..%2Fuploads%2Fdoc.txt",
                "%2Fetc%2Fpasswd",
            ):
                resp = await client.get(f"/api/jobs/{job_id}/files/{relpath}")
                assert resp.status_code == 404, relpath

    async def test_upload_filename_is_sanitized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._fake_converter()
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("../../.hidden/../evil.txt", b"x")]),
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            saved = job.items[0].name
            assert "/" not in saved and "\\" not in saved
            assert not saved.startswith(".")
            assert saved.endswith(".txt")
            await _wait_job_done(client, job_id)


class TestRealConversion:
    """End-to-end: a real .txt through the real conversion core (no LLM)."""

    async def test_txt_file_converts_to_markdown(self, tmp_path: Path) -> None:
        content = "# Real Doc\n\nServe end-to-end test content."
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("real.txt", content.encode())]),
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)

            assert data["done"] == 1 and data["failed"] == 0
            item = data["items"][0]
            assert item["status"] == "done"
            assert item["output"] == "real.txt.md"

            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            assert result.status_code == 200
            payload = result.json()
            assert payload["variant"] == "base"
            assert "Serve end-to-end test content." in payload["markdown"]

            archive = await client.get(f"/api/jobs/{job_id}/archive")
            assert archive.status_code == 200
            with zipfile.ZipFile(BytesIO(archive.content)) as zf:
                assert "real.txt.md" in zf.namelist()


class TestUrlPipeline:
    """URL jobs: lifecycle with a stubbed item processor, plus the real
    ``process_url_item`` over a monkeypatched ``markitai.fetch.fetch_url``."""

    @staticmethod
    def _canned_fetch(content_by_url: dict[str, str] | None = None):
        from markitai.fetch_types import FetchResult

        async def fake_fetch_url(url: str, *args: Any, **kwargs: Any) -> FetchResult:
            content = (
                "# Fetched\n\nbody text"
                if content_by_url is None
                else content_by_url[url]
            )
            return FetchResult(
                content=content, strategy_used="static", title="Fetched", url=url
            )

        return fake_fetch_url

    @staticmethod
    def _url_ctx_and_cfg(tmp_path: Path) -> tuple[Any, MarkitaiConfig, Path]:
        from markitai.serve.jobs import UrlJobContext

        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        return UrlJobContext.build(cfg, out_dir), cfg, out_dir

    async def test_urls_only_job_lifecycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str | None]] = []

        async def fake_process_url_item(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            from markitai.batch import ProcessResult

            calls.append((url, output_name))
            out = out_dir / (output_name or "x.md")
            out.write_text(f"# fetched {url}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", fake_process_url_item
        )
        url = "https://example.com/page.html"
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post("/api/jobs", files=_multipart(urls=[url]))
            assert created.status_code == 201
            body = created.json()
            assert body["items"] == [{"item_id": "i1", "name": url, "kind": "url"}]
            data = await _wait_job_done(client, body["job_id"])
            item = data["items"][0]
            assert item["status"] == "done"
            assert item["kind"] == "url"
            assert item["output"] == "page.html.md"
            result = await client.get(f"/api/jobs/{body['job_id']}/items/i1/result")
            assert result.status_code == 200
            assert f"fetched {url}" in result.json()["markdown"]
        assert calls == [(url, "page.html.md")]

    async def test_process_url_item_writes_base_md_with_frontmatter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr("markitai.fetch.fetch_url", self._canned_fetch())
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )
        assert result.success is True and result.error is None
        out = out_dir / "page.html.md"
        assert result.output_path == str(out)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("---\n")  # basic frontmatter block
        assert "body text" in text

    async def test_process_url_item_plain_without_llm_keeps_raw_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr("markitai.fetch.fetch_url", self._canned_fetch())
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.llm.pure = True
        cfg.llm.enabled = False
        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )
        assert result.success is True
        assert (out_dir / "page.html.md").read_text(encoding="utf-8") == (
            "# Fetched\n\nbody text"
        )

    @pytest.mark.parametrize("content", ["", "Text layer must not become output"])
    async def test_process_url_item_capture_only_without_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
    ) -> None:
        from markitai.constants import SCREENSHOTS_REL_PATH
        from markitai.fetch_types import FetchResult
        from markitai.serve.jobs import process_url_item

        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.llm.enabled = False
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True
        screenshot = out_dir / SCREENSHOTS_REL_PATH / "page.html.0001.png"
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(b"screenshot")

        async def fake_fetch(url: str, *args: Any, **kwargs: Any) -> FetchResult:
            assert kwargs["screenshot"] is True
            return FetchResult(
                content=content,
                strategy_used="playwright",
                url=url,
                screenshot_path=screenshot,
            )

        monkeypatch.setattr("markitai.fetch.fetch_url", fake_fetch)
        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )
        assert result.success is True
        assert result.screenshots == 1
        assert result.output_path == str(out_dir / "page.html.md")
        text = (out_dir / "page.html.md").read_text(encoding="utf-8")
        assert f"{SCREENSHOTS_REL_PATH}/{screenshot.name}" in text
        assert "Text layer must not become output" not in text

    async def test_process_url_item_notices_a_missing_screenshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.notices import capture_task_notices
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr("markitai.fetch.fetch_url", self._canned_fetch())
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.screenshot.enabled = True
        with capture_task_notices() as notices:
            result = await process_url_item(
                "https://example.com/page.html?token=secret",
                cfg,
                out_dir,
                None,
                url_ctx,
            )
        assert result.success is True
        assert len(notices) == 1
        assert notices[0].startswith("[URL] Screenshot not captured for ")
        assert "secret" not in notices[0]

    async def test_capture_only_job_exposes_screenshot_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.fetch_types import FetchResult

        async def fake_fetch(url: str, *args: Any, **kwargs: Any) -> FetchResult:
            screenshot = kwargs["screenshot_dir"] / "page.html.0001.png"
            screenshot.write_bytes(b"screenshot")
            return FetchResult(
                content="Text layer must not become output",
                strategy_used="playwright",
                url=url,
                screenshot_path=screenshot,
            )

        monkeypatch.setattr("markitai.fetch.fetch_url", fake_fetch)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    urls=["https://example.com/page.html"],
                    options={"llm": False, "screenshot_only": True},
                ),
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            job = await _wait_job_done(client, job_id)
            assert job["items"][0]["status"] == "done"
            response = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            assert response.status_code == 200
            result = response.json()
            assert "Text layer must not become output" not in result["markdown"]
            screenshot = next(
                artifact
                for artifact in result["artifacts"]
                if artifact["relpath"].endswith(".png")
            )
            assert screenshot["relpath"] in result["markdown"]
            download = await client.get(
                f"/api/jobs/{job_id}/files/{screenshot['relpath']}"
            )
            assert download.status_code == 200
            assert download.content == b"screenshot"

    async def test_process_url_item_capture_only_requires_a_capture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr("markitai.fetch.fetch_url", self._canned_fetch())
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True
        cfg.llm.enabled = False
        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )
        assert result.success is False
        assert "No screenshot captured" in (result.error or "")
        assert not (out_dir / "page.html.md").exists()

    async def test_process_url_item_skips_image_download_without_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr(
            "markitai.fetch.fetch_url",
            self._canned_fetch(
                {
                    "https://example.com/page.html": (
                        "# Fetched\n\n![](https://images.example.com/slow.png)"
                    )
                }
            ),
        )
        downloaded = False

        async def fake_download(*args: Any, **kwargs: Any) -> Any:
            nonlocal downloaded
            downloaded = True
            raise AssertionError("images must not be downloaded with LLM disabled")

        monkeypatch.setattr("markitai.image.download_url_images", fake_download)
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.llm.enabled = False
        cfg.image.alt_enabled = True
        cfg.image.desc_enabled = True

        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )

        assert result.success is True
        assert downloaded is False
        assert "https://images.example.com/slow.png" in (
            out_dir / "page.html.md"
        ).read_text(encoding="utf-8")

    async def test_process_url_item_maps_fetch_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.fetch import FetchError, JinaRateLimitError
        from markitai.serve.jobs import process_url_item

        async def failing_fetch(url: str, *args: Any, **kwargs: Any) -> Any:
            raise FetchError(f"All fetch strategies failed for {url}:\n  - s: boom")

        monkeypatch.setattr("markitai.fetch.fetch_url", failing_fetch)
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        result = await process_url_item(
            "https://x.invalid/a", cfg, out_dir, None, url_ctx
        )
        assert result.success is False
        assert result.error is not None and "boom" in result.error

        async def rate_limited(url: str, *args: Any, **kwargs: Any) -> Any:
            raise JinaRateLimitError()

        monkeypatch.setattr("markitai.fetch.fetch_url", rate_limited)
        result = await process_url_item(
            "https://x.invalid/a", cfg, out_dir, None, url_ctx
        )
        assert result.success is False
        assert result.error == "Jina Reader rate limit exceeded (20 RPM)"

    async def test_process_url_item_skips_existing_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_url_item

        monkeypatch.setattr("markitai.fetch.fetch_url", self._canned_fetch())
        url_ctx, cfg, out_dir = self._url_ctx_and_cfg(tmp_path)
        cfg.output.on_conflict = "skip"
        (out_dir / "page.html.md").write_text("previous run", encoding="utf-8")
        result = await process_url_item(
            "https://example.com/page.html", cfg, out_dir, None, url_ctx
        )
        assert result.success is True
        assert result.error == "skipped (exists)"
        assert (out_dir / "page.html.md").read_text(encoding="utf-8") == "previous run"

    async def test_colliding_url_names_get_distinct_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two URLs mapping to the same filename must not clobber each other."""
        url_a = "https://a.example/index.html"
        url_b = "https://b.example/index.html"
        monkeypatch.setattr(
            "markitai.fetch.fetch_url",
            self._canned_fetch({url_a: "# Doc A", url_b: "# Doc B"}),
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(urls=[url_a, url_b])
            )
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            outputs = {i["name"]: i["output"] for i in data["items"]}
            assert outputs == {url_a: "index.html.md", url_b: "index.html (2).md"}
            text_a = (job.out_dir / "index.html.md").read_text(encoding="utf-8")
            text_b = (job.out_dir / "index.html (2).md").read_text(encoding="utf-8")
            assert "# Doc A" in text_a
            assert "# Doc B" in text_b


class TestRetry:
    """POST /api/jobs/{job_id}/items/{item_id}/retry."""

    @staticmethod
    def _stub_converter(gate: asyncio.Event | None = None):
        """Stub converter writing '<name>.md' (optionally gated)."""

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            if gate is not None:
                await gate.wait()
            out = out_dir / f"{file_path.name}.md"
            out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        return convert

    async def test_retry_while_sibling_still_running_is_not_stranded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying a terminal item mid-job must start the retry worker even
        though the initial run (job.task) is not yet done."""
        gate = asyncio.Event()
        calls: list[str] = []

        async def conv(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            calls.append(file_path.name)
            if file_path.name == "doc2.txt":
                await gate.wait()  # hold the initial run open
                out = out_dir / f"{file_path.name}.md"
                out.write_text("ok", encoding="utf-8")
                return ProcessResult(success=True, output_path=str(out))
            if calls.count("doc1.txt") == 1:  # first pass: i1 fails
                return ProcessResult(success=False, error="boom")
            out = out_dir / f"{file_path.name}.md"  # retry pass: i1 succeeds
            out.write_text("ok", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", conv)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("doc1.txt", b"a"), ("doc2.txt", b"b")]),
            )
            job_id = created.json()["job_id"]

            # i1 has failed while i2 is still gated (job running).
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snap = (await client.get(f"/api/jobs/{job_id}")).json()
                if snap["items"][0]["status"] == "error":
                    break
                await asyncio.sleep(0.02)
            assert snap["status"] == "running"
            assert snap["items"][0]["status"] == "error"

            queued = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert queued.status_code == 202
            gate.set()
            done = await _wait_job_done(client, job_id)

        assert done["items"][0]["status"] == "done"
        assert done["items"][0]["output"] == "doc1.txt.md"
        assert done["items"][1]["status"] == "done"

    async def test_retry_failed_file_item_reuses_upload_and_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Path] = []

        async def flaky(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            calls.append(file_path)
            if len(calls) == 1:
                return ProcessResult(success=False, error="converter exploded")
            out = out_dir / f"{file_path.name}.md"
            out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", flaky)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["items"][0]["status"] == "error"

            resp = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert resp.status_code == 202
            body = resp.json()
            assert body["job_id"] == job_id
            assert body["items"] == [
                {"item_id": "i1", "name": "doc.txt", "kind": "file"}
            ]

            second = await _wait_job_done(client, job_id)
            assert second["done"] == 1 and second["failed"] == 0
            assert second["items"][0]["output"] == "doc.txt.md"
            assert set(app.state.markitai.registry.jobs) == {job_id}
            history = (await client.get("/api/history")).json()
            assert [entry["job_id"] for entry in history] == [job_id]
        # Both runs consume the same durable upload and ledger item.
        assert [path.name for path in calls] == ["doc.txt", "doc.txt"]
        assert calls[1] == tmp_path / "jobs" / job_id / "uploads" / "doc.txt"

    async def test_retry_url_item_reenters_cascade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str | None]] = []

        async def flaky_url(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            from markitai.batch import ProcessResult

            calls.append((url, output_name))
            if len(calls) == 1:
                return ProcessResult(success=False, error="fetch failed")
            out = out_dir / (output_name or "x.md")
            out.write_text(f"# fetched {url}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", flaky_url)
        url = "https://example.com/page.html"
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post("/api/jobs", files=_multipart(urls=[url]))
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["items"][0]["status"] == "error"

            resp = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert resp.status_code == 202
            body = resp.json()
            assert body["job_id"] == job_id
            assert body["items"] == [{"item_id": "i1", "name": url, "kind": "url"}]
            second = await _wait_job_done(client, job_id)
            assert second["items"][0]["status"] == "done"
            assert second["items"][0]["output"] == "page.html.md"
        assert calls == [(url, "page.html.md"), (url, "page.html.md")]

    async def test_explicit_llm_enhancement_reuses_row_and_records_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.batch import ProcessResult
        from markitai.config import LiteLLMParams, ModelConfig

        attempts: list[bool] = []

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            attempts.append(cfg.llm.enabled)
            suffix = ".llm.md" if cfg.llm.enabled else ".md"
            output = out_dir / f"{file_path.name}{suffix}"
            output.write_text(
                "enhanced" if cfg.llm.enabled else "base", encoding="utf-8"
            )
            return ProcessResult(
                success=True,
                output_path=str(output),
                cost_usd=0.0123 if cfg.llm.enabled else 0.0,
                llm_enhanced=cfg.llm.enabled,
            )

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
            )
        ]
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: object(),
        )
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("doc.txt", b"hello")],
                    options={"preset": "minimal", "llm": False},
                ),
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["items"][0]["llm_enhanced"] is False

            queued = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={
                    "operation": "enhance",
                    "options": {"preset": "minimal", "llm": True},
                },
            )
            assert queued.status_code == 202
            enhanced = await _wait_job_done(client, job_id)

            # An existing .llm.md result remains explicitly re-enhanceable so
            # changed model settings can be applied without creating a new row.
            repeated = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={
                    "operation": "enhance",
                    "options": {"preset": "minimal", "llm": True},
                },
            )
            assert repeated.status_code == 202
            enhanced = await _wait_job_done(client, job_id)
            history = (await client.get("/api/history")).json()[0]

        item = enhanced["items"][0]
        assert item["item_id"] == "i1"
        assert item["operation"] == "enhance"
        assert item["llm_enhanced"] is True
        assert item["output"] == "doc.txt.llm.md"
        assert item["cost_usd"] == pytest.approx(0.0123)
        # Enhancing one item must not rewrite the source job's shared options.
        assert enhanced["options"]["llm"] is False
        assert history["llm_enhanced"] == 1
        assert history["cost_usd"] == pytest.approx(0.0123)
        assert history["duration_ms"] == item["duration_ms"]
        assert attempts == [False, True, True]

    async def test_failed_enhance_keeps_the_previous_llm_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: serve judged enhancement by the .llm.md suffix, so a
        degraded rerun that rewrote .llm.md counted as enhanced, replaced
        the previous good file and never rolled back."""
        from markitai.batch import ProcessResult
        from markitai.config import LiteLLMParams, ModelConfig

        runs: list[str] = []

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            runs.append("x")
            output = out_dir / f"{file_path.name}.llm.md"
            if len(runs) == 1:
                output.write_text("good enhancement", encoding="utf-8")
                return ProcessResult(
                    success=True, output_path=str(output), llm_enhanced=True
                )
            # A rerun whose LLM failed: whatever it left on disk, the
            # pipeline reports no real enhancement
            output.write_text("degraded rewrite", encoding="utf-8")
            (out_dir / f"{file_path.name}.md").write_text("base", encoding="utf-8")
            return ProcessResult(
                success=True, output_path=str(output), llm_enhanced=False
            )

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
            )
        ]
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: object(),
        )
        app = _make_app(tmp_path, cfg)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("doc.txt", b"hello")],
                    options={"preset": "minimal", "llm": True},
                ),
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["items"][0]["llm_enhanced"] is True

            queued = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={
                    "operation": "enhance",
                    "options": {"preset": "minimal", "llm": True},
                },
            )
            assert queued.status_code == 202
            after = await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")

        item = after["items"][0]
        assert item["status"] == "done"
        assert item["llm_enhanced"] is True
        assert item["output"] == "doc.txt.llm.md"
        assert len(runs) == 2
        # The previous files are back byte for byte; the base .md the failed
        # rerun created is not left behind
        out_dir = next((tmp_path / "jobs").glob("*/out"))
        assert (out_dir / "doc.txt.llm.md").read_text(encoding="utf-8") == (
            "good enhancement"
        )
        assert not (out_dir / "doc.txt.md").exists()
        assert "good enhancement" in result.text

    async def test_failed_url_enhance_restores_the_base_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An Enhance of a plain URL result re-fetches the page and rewrites
        the base .md before the LLM step; when the LLM then fails, the base
        .md must go back to what the row showed before the rerun."""
        from markitai.batch import ProcessResult
        from markitai.config import LiteLLMParams, ModelConfig

        runs: list[str] = []

        async def fake_process_url_item(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            out = out_dir / (output_name or "x.md")
            runs.append(url)
            if len(runs) == 1:
                out.write_text("ORIGINAL-V1", encoding="utf-8")
                return ProcessResult(success=True, output_path=str(out))
            out.write_text("CHANGED-V2", encoding="utf-8")
            return ProcessResult(success=False, error="LLM processing failed: 500")

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
            )
        ]
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", fake_process_url_item
        )
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: object(),
        )
        url = "https://example.com/page.html"
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            created = await client.post("/api/jobs", files=_multipart(urls=[url]))
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            queued = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={"operation": "enhance", "options": {"llm": True}},
            )
            assert queued.status_code == 202
            after = await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")

        item = after["items"][0]
        assert len(runs) == 2
        assert item["status"] == "done"
        assert item["output"] == "page.html.md"
        assert result.json()["markdown"] == "ORIGINAL-V1"

    async def test_explicit_llm_enhancement_requires_an_llm_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.batch import ProcessResult
        from markitai.config import LiteLLMParams, ModelConfig

        async def base_only(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            output = out_dir / f"{file_path.name}.md"
            output.write_text("base", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
            )
        ]
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", base_only)
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: object(),
        )
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            queued = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={
                    "operation": "enhance",
                    "options": {"preset": "minimal", "llm": True},
                },
            )
            assert queued.status_code == 202
            failed = await _wait_job_done(client, job_id)

        # A failed enhance must not destroy the item's existing base result:
        # the row rolls back to its prior successful (base) state.
        item = failed["items"][0]
        assert item["status"] == "done"
        assert item["operation"] == "convert"
        assert item["llm_enhanced"] is False
        assert item["output"] == "doc.txt.md"
        assert item["error"] is None

    async def test_explicit_llm_enhancement_is_rejected_without_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            response = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={
                    "operation": "enhance",
                    "options": {"preset": "minimal", "llm": True},
                },
            )

        assert response.status_code == 409
        assert "unavailable" in response.json()["detail"]

    async def test_retry_non_terminal_item_is_409(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter(gate)
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hi")])
            )
            job_id = created.json()["job_id"]
            resp = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert resp.status_code == 409
            assert set(app.state.markitai.registry.jobs) == {job_id}
            gate.set()
            await _wait_job_done(client, job_id)
            # done items are retryable too (terminal = done or error)
            done = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert done.status_code == 202
            assert done.json()["job_id"] == job_id
            await _wait_job_done(client, job_id)

    async def test_retry_unknown_job_or_item_is_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            missing_job = await client.post("/api/jobs/nope/items/i1/retry")
            assert missing_job.status_code == 404
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hi")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            missing_item = await client.post(f"/api/jobs/{job_id}/items/i99/retry")
            assert missing_item.status_code == 404

    async def test_retry_missing_upload_is_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hi")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            (tmp_path / "jobs" / job_id / "uploads" / "doc.txt").unlink()

            resp = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert resp.status_code == 404
            assert "no longer on disk" in resp.json()["detail"]
            assert set(app.state.markitai.registry.jobs) == {job_id}

    async def test_skipped_image_retries_in_place_with_ocr_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[bool] = []

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            attempts.append(cfg.ocr.enabled)
            if not cfg.ocr.enabled:
                return ProcessResult(success=True, error="skipped (image_only)")
            output = out_dir / f"{file_path.name}.md"
            output.write_text("recognized text", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("sample.jpg", b"image")],
                    options={"llm": False, "ocr": False},
                ),
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["items"][0]["skipped"] is True

            retried = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={"options": {"preset": "minimal", "llm": False, "ocr": True}},
            )
            assert retried.status_code == 202
            assert retried.json()["job_id"] == job_id
            second = await _wait_job_done(client, job_id)

        assert attempts == [False, True]
        assert second["items"][0]["item_id"] == "i1"
        assert second["items"][0]["status"] == "done"
        assert second["items"][0]["skipped"] is False
        assert second["options"]["ocr"] is True

    async def test_retry_inherits_and_overrides_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("doc.txt", b"hi")], options={"preset": "minimal"}
                ),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)

            # No body: the source job's options are inherited.
            inherited = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert inherited.status_code == 202
            assert inherited.json()["job_id"] == job_id
            await _wait_job_done(client, job_id)
            snap = (await client.get(f"/api/jobs/{job_id}")).json()
            assert snap["options"] == _all_job_options(
                preset="minimal", llm=None, ocr=None
            )

            # Body options replace the inherited ones as a whole.
            overridden = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={"options": {"preset": "standard", "llm": False}},
            )
            assert overridden.status_code == 202
            assert overridden.json()["job_id"] == job_id
            await _wait_job_done(client, job_id)
            snap = (await client.get(f"/api/jobs/{job_id}")).json()
            assert snap["options"] == _all_job_options(
                preset="standard", llm=False, ocr=None
            )

            # Same validation as POST /api/jobs.
            bad_key = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry", json={"options": {"bogus": 1}}
            )
            assert bad_key.status_code == 422
            bad_preset = await client.post(
                f"/api/jobs/{job_id}/items/i1/retry",
                json={"options": {"preset": "nope"}},
            )
            assert bad_preset.status_code == 422

    async def test_retries_share_one_serial_background_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: dict[str, int] = {}
        retry_started = asyncio.Event()
        release_retry = asyncio.Event()
        retry_order: list[str] = []
        active = 0
        max_active = 0

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            nonlocal active, max_active
            from markitai.batch import ProcessResult

            name = file_path.name
            attempts[name] = attempts.get(name, 0) + 1
            if attempts[name] == 1:
                return ProcessResult(success=False, error=f"failed {name}")
            active += 1
            max_active = max(max_active, active)
            retry_order.append(name)
            try:
                if name == "first.txt":
                    retry_started.set()
                    await release_retry.wait()
                output = out_dir / f"{name}.md"
                output.write_text("done", encoding="utf-8")
                return ProcessResult(success=True, output_path=str(output))
            finally:
                active -= 1

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("first.txt", b"one"), ("second.txt", b"two")]),
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            assert first["failed"] == 2

            queued_first = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert queued_first.status_code == 202
            await asyncio.wait_for(retry_started.wait(), timeout=1)
            queued_second = await client.post(f"/api/jobs/{job_id}/items/i2/retry")
            assert queued_second.status_code == 202
            assert set(app.state.markitai.registry.jobs) == {job_id}

            release_retry.set()
            final = await _wait_job_done(client, job_id)
            assert final["done"] == 2
            assert retry_order == ["first.txt", "second.txt"]
            assert max_active == 1

    @pytest.mark.parametrize(
        ("retried", "gated"),
        [("notes.llm", "notes"), ("notes", "notes.llm")],
    )
    async def test_failed_rerun_never_deletes_a_sibling_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        retried: str,
        gated: str,
    ) -> None:
        """Regression: the rollback snapshot reverse-engineered the stem from
        the previous output, so ``notes.llm.md`` (upload ``notes.llm``) was
        read as stem ``notes`` and a failed rerun unlinked sibling ``notes``'s
        ``notes.md``. Conversely a file a sibling claims (``notes.llm.md`` of
        upload ``notes.llm``) must survive ``notes``'s rollback."""
        from markitai.batch import ProcessResult

        attempts: dict[str, int] = {}
        gate = asyncio.Event()
        rerun_started = asyncio.Event()
        fail_rerun = asyncio.Event()

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            name = file_path.name
            attempts[name] = attempts.get(name, 0) + 1
            if name == gated:
                await gate.wait()
            if attempts[name] == 2:
                rerun_started.set()
                await fail_rerun.wait()
                return ProcessResult(success=False, error="rerun failed")
            output = out_dir / f"{name}.md"
            output.write_text(f"# {name}", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("notes", b"a"), ("notes.llm", b"b")]),
            )
            job_id = created.json()["job_id"]
            job = app.state.markitai.registry.get(job_id)
            ids = {item.name: item.item_id for item in job.items}
            retried_item = job.get_item(ids[retried])
            gated_item = job.get_item(ids[gated])
            for _ in range(500):
                if retried_item.status == "done":
                    break
                await asyncio.sleep(0.01)
            assert retried_item.status == "done"

            queued = await client.post(f"/api/jobs/{job_id}/items/{ids[retried]}/retry")
            assert queued.status_code == 202
            await asyncio.wait_for(rerun_started.wait(), 5)
            # The sibling writes its output while the rerun is in flight.
            gate.set()
            for _ in range(500):
                if gated_item.status == "done":
                    break
                await asyncio.sleep(0.01)
            fail_rerun.set()
            final = await _wait_job_done(client, job_id)

        assert final["done"] == 2 and final["failed"] == 0
        out_dir = tmp_path / "jobs" / job_id / "out"
        assert (out_dir / "notes.md").read_text(encoding="utf-8") == "# notes"
        assert (out_dir / "notes.llm.md").read_text(encoding="utf-8") == ("# notes.llm")

    async def test_result_never_serves_a_sibling_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upload ``notes.llm`` writes base ``notes.llm.md``; its result must
        not be resolved as sibling ``notes``'s pair and serve ``notes.md``."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("notes", b"a"), ("notes.llm", b"b")]),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            first = (await client.get(f"/api/jobs/{job_id}/items/i1/result")).json()
            second = (await client.get(f"/api/jobs/{job_id}/items/i2/result")).json()

        assert first["markdown"] == "# converted notes\n"
        assert second["markdown"] == "# converted notes.llm\n"
        assert [a["relpath"] for a in second["artifacts"]] == ["notes.llm.md"]

    async def test_retry_archived_job_after_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", self._stub_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
            )
            source_id = created.json()["job_id"]
            await _wait_job_done(client, source_id)

        # "Restart": a fresh app over the same jobs root rehydrates the job.
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(f"/api/jobs/{source_id}/items/i1/retry")
            assert resp.status_code == 202
            assert resp.json()["job_id"] == source_id
            assert (tmp_path / "jobs" / source_id / "uploads" / "doc.txt").is_file()
            data = await _wait_job_done(client, source_id)
            assert data["done"] == 1 and data["failed"] == 0
            assert data["items"][0]["output"] == "doc.txt.md"


class TestDeleteJobItem:
    """DELETE /api/jobs/{job_id}/items/{item_id}."""

    async def test_delete_one_row_then_last_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            output = out_dir / f"{file_path.name}.md"
            output.write_text(f"# {file_path.name}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("first.txt", b"one"), ("second.txt", b"two")]),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            first_output = job.out_dir / "first.txt.md"
            assert first_output.is_file()

            deleted = await client.delete(f"/api/jobs/{job_id}/items/i1")
            assert deleted.status_code == 204
            assert not first_output.exists()
            snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
            assert [item["item_id"] for item in snapshot["items"]] == ["i2"]
            history = (await client.get("/api/history")).json()
            assert history[0]["total"] == 1

            deleted_last = await client.delete(f"/api/jobs/{job_id}/items/i2")
            assert deleted_last.status_code == 204
            assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
            assert not (tmp_path / "jobs" / job_id).exists()

    async def test_delete_waits_for_whole_job_to_finish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            if file_path.name == "second.txt":
                await gate.wait()
            output = out_dir / f"{file_path.name}.md"
            output.write_text("done", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("first.txt", b"one"), ("second.txt", b"two")]),
            )
            job_id = created.json()["job_id"]
            for _ in range(100):
                snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
                if snapshot["items"][0]["status"] == "done":
                    break
                await asyncio.sleep(0.005)
            response = await client.delete(f"/api/jobs/{job_id}/items/i1")
            assert response.status_code == 409
            gate.set()
            await _wait_job_done(client, job_id)

    async def test_concurrent_deletes_are_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the row check and the off-thread file work ran before
        the row was removed, so a repeated DELETE crashed with a 500 and
        deleting the last two rows at once left an empty job in history."""

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            output = out_dir / f"{file_path.name}.md"
            output.write_text(f"# {file_path.name}\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("a.txt", b"a"), ("b.txt", b"b"), ("c.txt", b"c")]
                ),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)

            same_row = await asyncio.gather(
                client.delete(f"/api/jobs/{job_id}/items/i1"),
                client.delete(f"/api/jobs/{job_id}/items/i1"),
            )
            assert sorted(r.status_code for r in same_row) == [204, 404]
            snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
            assert [item["item_id"] for item in snapshot["items"]] == ["i2", "i3"]

            last_rows = await asyncio.gather(
                client.delete(f"/api/jobs/{job_id}/items/i2"),
                client.delete(f"/api/jobs/{job_id}/items/i3"),
            )
            assert [r.status_code for r in last_rows] == [204, 204]
            assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
            assert (await client.get("/api/history")).json() == []
            assert not (tmp_path / "jobs" / job_id).exists()
            again = await client.delete(f"/api/jobs/{job_id}/items/i3")
            assert again.status_code == 404

        # Nothing empty comes back after a restart either.
        async with _serve_client(_make_app(tmp_path)) as client:
            assert (await client.get("/api/history")).json() == []

    async def test_concurrent_history_and_item_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            output = out_dir / f"{file_path.name}.md"
            output.write_text("done", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.txt", b"a"), ("b.txt", b"b")]),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            responses = await asyncio.gather(
                client.delete(f"/api/history/{job_id}"),
                client.delete(f"/api/jobs/{job_id}/items/i1"),
            )
            # Whichever runs second sees the other's result: a row delete
            # behind the history delete finds no job (404) instead of
            # writing meta.json into the removed directory (500).
            assert responses[0].status_code == 204
            assert responses[1].status_code in (204, 404)
            assert not (tmp_path / "jobs" / job_id).exists()
            assert (await client.get("/api/history")).json() == []


class TestItemWarnings:
    """User notices raised while an item converts land on that item."""

    @staticmethod
    def _noticing_converter(
        both_running: asyncio.Barrier | None = None, fail_rerun: bool = False
    ):
        attempts: dict[str, int] = {}

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult
            from markitai.notices import user_notice
            from markitai.utils.executor import run_in_converter_thread

            name = file_path.name
            attempts[name] = attempts.get(name, 0) + 1
            if both_running is not None:
                await both_running.wait()
            # Raised from the converter thread pool, like the real converters
            await run_in_converter_thread(
                user_notice,
                "[PDF] {}: 2 page(s) look scanned/garbled (run {})",
                name,
                attempts[name],
            )
            if both_running is not None:
                await both_running.wait()
            if fail_rerun and attempts[name] > 1:
                return ProcessResult(success=False, error="rerun failed")
            output = out_dir / f"{name}.md"
            output.write_text(f"# {name}", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(output))

        return convert

    async def test_concurrent_items_keep_their_own_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item",
            self._noticing_converter(asyncio.Barrier(2)),
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.pdf", b"a"), ("b.pdf", b"b")]),
            )
            job_id = created.json()["job_id"]
            done = await _wait_job_done(client, job_id)
            events = _parse_sse((await client.get(f"/api/jobs/{job_id}/events")).text)

        assert [item["warnings"] for item in done["items"]] == [
            ["[PDF] a.pdf: 2 page(s) look scanned/garbled (run 1)"],
            ["[PDF] b.pdf: 2 page(s) look scanned/garbled (run 1)"],
        ]
        assert events[0][0] == "snapshot"
        assert events[0][1]["items"][0]["warnings"] == done["items"][0]["warnings"]
        meta = json.loads((tmp_path / "jobs" / job_id / "meta.json").read_text())
        assert meta["items"][1]["warnings"] == done["items"][1]["warnings"]

        # Rehydrated history keeps them.
        async with _serve_client(_make_app(tmp_path)) as client:
            snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
        assert snapshot["items"][0]["warnings"] == done["items"][0]["warnings"]

    async def test_rerun_replaces_warnings_and_a_rollback_restores_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item",
            self._noticing_converter(fail_rerun=True),
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("a.pdf", b"a")])
            )
            job_id = created.json()["job_id"]
            first = await _wait_job_done(client, job_id)
            queued = await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            assert queued.status_code == 202
            after = await _wait_job_done(client, job_id)

        assert first["items"][0]["warnings"] == [
            "[PDF] a.pdf: 2 page(s) look scanned/garbled (run 1)"
        ]
        # The failed rerun rolled back to the first result, warnings included.
        assert after["items"][0]["status"] == "done"
        assert after["items"][0]["warnings"] == first["items"][0]["warnings"]


class TestKeepBase:
    """serve forces llm.keep_base: LLM jobs keep .md next to .llm.md."""

    def _make_llm_app(self, tmp_path: Path) -> FastAPI:
        from markitai.config import LiteLLMParams, ModelConfig

        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        cfg.cache.global_dir = str(tmp_path / "cache")
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini"),
            )
        ]
        return create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            config=cfg,
            configure_logging=False,
        )

    async def test_llm_job_produces_base_and_llm_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real .txt conversion with a fake LLM step: keep_base is forced, so
        write_base_markdown keeps the .md the diff view needs."""
        from unittest.mock import MagicMock

        from markitai.workflow.core import ConversionStepResult

        async def fake_standard_llm(ctx: Any) -> ConversionStepResult:
            llm_out = ctx.output_file.with_suffix(".llm.md")
            llm_out.write_text("# llm enhanced\n", encoding="utf-8")
            # Like the real step: only a successful LLM write sets it
            ctx.llm_output_file = llm_out
            return ConversionStepResult(success=True)

        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "markitai.workflow.core.process_with_standard_llm", fake_standard_llm
        )
        async with _serve_client(self._make_llm_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("keep.txt", b"# body")], options={"llm": True}
                ),
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
            item = data["items"][0]
            assert item["status"] == "done"
            assert item["output"] == "keep.txt.llm.md"
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        assert result.status_code == 200
        payload = result.json()
        assert payload["variant"] == "llm"
        relpaths = {a["relpath"] for a in payload["artifacts"]}
        assert relpaths == {"keep.txt.md", "keep.txt.llm.md"}

    async def test_llm_disabled_job_keeps_base_only(self, tmp_path: Path) -> None:
        async with _serve_client(self._make_llm_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("keep.txt", b"# body")], options={"llm": False}
                ),
            )
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
            assert data["items"][0]["output"] == "keep.txt.md"
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        payload = result.json()
        assert payload["variant"] == "base"
        assert {a["relpath"] for a in payload["artifacts"]} == {"keep.txt.md"}


class TestSkipSemantics:
    """Skips complete as status=done with skipped=true + skip_reason."""

    async def test_image_without_llm_surfaces_as_skipped(
        self, tmp_path: Path, create_test_image: Any
    ) -> None:
        png = create_test_image(16, 16, "blue")
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(files=[("photo.png", png)])
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
            item = data["items"][0]
            assert item["status"] == "done"  # status enum unchanged
            assert item["skipped"] is True
            assert item["skip_reason"] == "image_only"
            assert item["output"] is None
            assert data["done"] == 1 and data["failed"] == 0
            # No output -> no previewable result
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            assert result.status_code == 404


class TestJobHousekeeping:
    """Stale job cleanup."""

    def test_cleanup_removes_only_stale_job_dirs(self, tmp_path: Path) -> None:
        import os

        from markitai.serve.jobs import cleanup_stale_jobs

        jobs_root = tmp_path / "jobs"
        stale = jobs_root / "old-job"
        fresh = jobs_root / "new-job"
        stale.mkdir(parents=True)
        fresh.mkdir(parents=True)
        old = time.time() - 25 * 3600
        os.utime(stale, (old, old))

        assert cleanup_stale_jobs(jobs_root, ttl_hours=24.0) == 1
        assert not stale.exists()
        assert fresh.exists()


class TestValidationErrorContract:
    """Pydantic 422s carry the same machine code as route-raised errors."""

    async def test_validation_errors_carry_a_machine_code(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            # An empty body fails the model-create schema (a real pydantic
            # 422), not a route-raised one.
            resp = await client.post("/api/settings/llm/models", json={})

        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "invalid_request"
        assert isinstance(body["detail"], list)


class TestItemEnhancedSignal:
    """Serve reads "enhanced" from the pipeline, not from the file suffix."""

    async def test_url_llm_fallback_completes_on_the_base_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llm.on_failure = "fallback" (default): the cascade kept the base
        .md and reports the error; the item completes, not enhanced."""
        from markitai.serve.jobs import UrlJobContext, process_url_item
        from markitai.workflow.url import UrlCascadeResult

        base = tmp_path / "page.md"
        base.write_text("base", encoding="utf-8")

        async def cascade(*_args: Any, **_kwargs: Any) -> UrlCascadeResult:
            return UrlCascadeResult(
                markdown="base",
                output_path=base,
                llm_output_path=None,
                target_file=base,
                llm_error="AuthenticationError: invalid api key",
            )

        monkeypatch.setattr("markitai.workflow.url.convert_url_cascade", cascade)
        result = await process_url_item(
            "https://example.com/page",
            MarkitaiConfig(),
            tmp_path,
            None,
            UrlJobContext(strategy=None, cache=None, screenshot_dir=None),
        )

        assert result.success is True
        assert result.output_path == str(base)
        assert result.llm_enhanced is False

    async def test_url_llm_failure_fails_the_item_under_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a URL whose LLM stage failed was reported done."""
        from markitai.serve.jobs import UrlJobContext, process_url_item
        from markitai.utils.errors import ConversionError

        async def cascade(*_args: Any, **_kwargs: Any) -> Any:
            # What the cascade does under llm.on_failure = "fail"
            raise ConversionError(
                "LLM processing failed: AuthenticationError: invalid api key"
            )

        monkeypatch.setattr("markitai.workflow.url.convert_url_cascade", cascade)
        cfg = MarkitaiConfig()
        cfg.llm.on_failure = "fail"
        result = await process_url_item(
            "https://example.com/page",
            cfg,
            tmp_path,
            None,
            UrlJobContext(strategy=None, cache=None, screenshot_dir=None),
        )

        assert result.success is False
        assert result.error is not None and "invalid api key" in result.error
        assert result.llm_enhanced is False

    async def test_file_item_reports_the_pipeline_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.serve.jobs import process_file_item
        from markitai.workflow.core import ConversionStepResult

        source = tmp_path / "doc.txt"
        source.write_text("hello", encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        async def core(ctx: Any, _max_size: int) -> ConversionStepResult:
            ctx.output_file = out_dir / "doc.txt.md"
            ctx.llm_output_file = out_dir / "doc.txt.llm.md"
            ctx.llm_output_file.write_text("enhanced", encoding="utf-8")
            return ConversionStepResult(success=True)

        async def core_without_llm_write(
            ctx: Any, _max_size: int
        ) -> ConversionStepResult:
            ctx.output_file = out_dir / "doc.txt.md"
            return ConversionStepResult(success=True)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        monkeypatch.setattr("markitai.workflow.core.convert_document_core", core)
        enhanced = await process_file_item(source, cfg, out_dir, None)
        monkeypatch.setattr(
            "markitai.workflow.core.convert_document_core", core_without_llm_write
        )
        missing = await process_file_item(source, cfg, out_dir, None)

        assert enhanced.success is True
        assert enhanced.llm_enhanced is True
        assert enhanced.output_path == str(out_dir / "doc.txt.llm.md")
        assert missing.success is False
        assert missing.llm_enhanced is False

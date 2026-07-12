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

from markitai.config import MarkitaiConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def _make_app(tmp_path: Path) -> FastAPI:
    """Build an app with hermetic config and tmp_path-backed jobs root."""
    cfg = MarkitaiConfig()
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
            transport=transport, base_url="http://test"
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

        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == __version__
        assert data["llm"] == {"configured": False, "models": []}
        assert data["presets"] == ["minimal", "standard", "rich"]
        assert set(data["extras"]) == {"browser", "svg", "kreuzberg"}
        assert all(isinstance(v, bool) for v in data["extras"].values())

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
        assert data["llm"] == {"configured": True, "models": ["openai/gpt-4o-mini"]}

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

    async def test_too_many_items_is_422(self, tmp_path: Path) -> None:
        urls = [f"https://example.com/{i}" for i in range(51)]
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post("/api/jobs", files=_multipart(urls=urls))
        assert resp.status_code == 422

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

    async def test_unknown_job_and_item_are_404(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            assert (await client.get("/api/jobs/nope")).status_code == 404
            assert (
                await client.get("/api/jobs/nope/items/i1/result")
            ).status_code == 404
            assert (await client.get("/api/jobs/nope/archive")).status_code == 404


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

    def test_llm_true_without_models_degrades_to_disabled(self) -> None:
        cfg = self._build(MarkitaiConfig(), llm=True)
        assert cfg.llm.enabled is False

    def test_base_config_is_not_mutated(self) -> None:
        base = self._base_with_model()
        assert base.llm.enabled is False
        self._build(base, preset="rich")
        assert base.llm.enabled is False


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
            out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
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
        assert data["options"] == {"preset": None, "llm": None}

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

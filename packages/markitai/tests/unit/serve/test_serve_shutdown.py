"""Server shutdown with jobs still running.

An SSE stream following a running job must end as soon as the server starts
shutting down — otherwise the ASGI server keeps waiting on it, a second
Ctrl-C force-quits past the lifespan shutdown, and the job never writes its
meta.json (it vanishes from history). The lifespan shutdown then cancels the
job and persists it with its items marked ``cancelled (server shutdown)``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
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
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
    )


@asynccontextmanager
async def _serve_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


def _multipart(files: list[tuple[str, bytes]]) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files
    ]
    parts.append(("urls", (None, "[]")))
    parts.append(("options", (None, "{}")))
    return parts


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    event_name: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name is not None:
            events.append((event_name, json.loads(line.split(":", 1)[1].strip())))
            event_name = None
    return events


def _blocking_converter(started: asyncio.Event):
    """A converter that never finishes on its own (a long conversion)."""

    async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return convert


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            pytest.fail("condition not reached")
        await asyncio.sleep(0.01)


class TestSseEndsOnShutdown:
    async def test_open_stream_ends_when_shutdown_begins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _blocking_converter(started)
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("long.txt", b"x")])
            )
            job_id = created.json()["job_id"]
            await asyncio.wait_for(started.wait(), 5)
            registry = app.state.markitai.registry
            job = registry.get(job_id)
            assert job is not None

            stream = asyncio.create_task(client.get(f"/api/jobs/{job_id}/events"))
            await _wait_for(lambda: bool(job.subscribers))
            registry.begin_shutdown()
            response = await asyncio.wait_for(stream, 5)

            assert response.status_code == 200
            events = _parse_sse(response.text)
            # The snapshot went out; the stream then closed without a
            # (never-coming) terminal frame and without leaking the sentinel.
            assert [name for name, _ in events] == ["snapshot"]
            assert "__shutdown__" not in response.text
            assert job.subscribers == []

            # A stream opened while shutting down ends right after its snapshot.
            late = await asyncio.wait_for(client.get(f"/api/jobs/{job_id}/events"), 5)
            assert [name for name, _ in _parse_sse(late.text)] == ["snapshot"]

    async def test_request_shutdown_is_thread_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """uvicorn's signal handler calls request_shutdown off the loop."""
        from markitai.serve.app import request_shutdown

        started = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _blocking_converter(started)
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("long.txt", b"x")])
            )
            job_id = created.json()["job_id"]
            await asyncio.wait_for(started.wait(), 5)
            job = app.state.markitai.registry.get(job_id)
            stream = asyncio.create_task(client.get(f"/api/jobs/{job_id}/events"))
            await _wait_for(lambda: bool(job.subscribers))

            thread = threading.Thread(target=request_shutdown, args=(app,))
            thread.start()
            thread.join()
            response = await asyncio.wait_for(stream, 5)
        assert [name for name, _ in _parse_sse(response.text)] == ["snapshot"]

    def test_request_shutdown_before_startup_is_a_noop(self, tmp_path: Path) -> None:
        from markitai.serve.app import request_shutdown

        request_shutdown(_make_app(tmp_path))
        request_shutdown(object())


class TestShutdownPersistsRunningJobs:
    async def test_running_job_is_cancelled_persisted_and_rehydrated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _blocking_converter(started)
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart([("long.txt", b"x"), ("queued.txt", b"y")]),
            )
            job_id = created.json()["job_id"]
            await asyncio.wait_for(started.wait(), 5)
            stream = asyncio.create_task(client.get(f"/api/jobs/{job_id}/events"))
            job = app.state.markitai.registry.get(job_id)
            await _wait_for(lambda: bool(job.subscribers))
            # What the CLI's uvicorn subclass does on the first Ctrl-C.
            app.state.markitai.registry.begin_shutdown()
            await asyncio.wait_for(stream, 5)
        # Leaving the lifespan ran registry.shutdown().

        meta = json.loads((tmp_path / "jobs" / job_id / "meta.json").read_text())
        assert meta["status"] == "done"
        assert {item["status"] for item in meta["items"]} == {"error"}
        assert {item["error"] for item in meta["items"]} == {
            "cancelled (server shutdown)"
        }

        async with _serve_client(_make_app(tmp_path)) as client:
            history = (await client.get("/api/history")).json()
        assert [entry["job_id"] for entry in history] == [job_id]
        assert history[0]["failed"] == 2


class TestShutdownDuringRerun:
    async def test_interrupted_reruns_keep_the_previous_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: Ctrl-C while re-running (Enhance/Retry) done items
        persisted them as ``cancelled (server shutdown)`` errors without an
        output; after a restart the working result was gone. The running
        rerun and the one still queued behind it must both roll back."""
        from markitai.batch import ProcessResult

        attempts: dict[str, int] = {}
        rerun_started = asyncio.Event()

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            name = file_path.name
            attempts[name] = attempts.get(name, 0) + 1
            output = out_dir / f"{name}.md"
            if attempts[name] == 1:
                output.write_text(f"original {name}", encoding="utf-8")
                return ProcessResult(success=True, output_path=str(output))
            # The rerun overwrites the output in place, then never finishes.
            output.write_text("half-written rerun", encoding="utf-8")
            rerun_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("a.txt", b"a"), ("b.txt", b"b")])
            )
            job_id = created.json()["job_id"]
            job = app.state.markitai.registry.get(job_id)
            await _wait_for(lambda: job.status == "done")

            assert (
                await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            ).status_code == 202
            await asyncio.wait_for(rerun_started.wait(), 5)
            # Queued behind the running rerun on the serial worker.
            assert (
                await client.post(f"/api/jobs/{job_id}/items/i2/retry")
            ).status_code == 202
        # Leaving the lifespan ran registry.shutdown().

        out_dir = tmp_path / "jobs" / job_id / "out"
        assert (out_dir / "a.txt.md").read_text(encoding="utf-8") == "original a.txt"
        assert (out_dir / "b.txt.md").read_text(encoding="utf-8") == "original b.txt"
        meta = json.loads((tmp_path / "jobs" / job_id / "meta.json").read_text())
        assert meta["status"] == "done"
        assert [
            (item["status"], item["error"], item["output"], item["operation"])
            for item in meta["items"]
        ] == [
            ("done", None, "a.txt.md", "convert"),
            ("done", None, "b.txt.md", "convert"),
        ]

        async with _serve_client(_make_app(tmp_path)) as client:
            snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        assert snapshot["done"] == 2 and snapshot["failed"] == 0
        assert result.json()["markdown"] == "original a.txt"

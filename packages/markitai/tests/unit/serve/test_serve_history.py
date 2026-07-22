"""Tests for conversion history: meta.json, startup rehydrate, /api/history.

Same harness as test_serve_api.py: ``create_app`` with a tmp_path-backed jobs
root and an injected config over httpx.ASGITransport. A server restart is
simulated by exiting one app's lifespan and starting a fresh app on the same
jobs root.
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
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
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


def _multipart(files: list[tuple[str, bytes]]) -> list[tuple[str, Any]]:
    """Build httpx multipart payload for POST /api/jobs."""
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files
    ]
    parts.append(("urls", (None, "[]")))
    parts.append(("options", (None, "{}")))
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


def _fake_converter(gate: asyncio.Event | None = None):
    """Stub converter writing '<name>.md' into the job out dir."""

    async def fake_process_file_item(
        file_path: Path, cfg: Any, out_dir: Path, shared: Any
    ):
        from markitai.batch import ProcessResult

        if gate is not None:
            await gate.wait()
        out = out_dir / f"{file_path.name}.md"
        out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(out), cost_usd=0.25)

    return fake_process_file_item


async def _run_one_job(app: FastAPI, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Run one job to its terminal state inside a fresh lifespan."""
    async with _serve_client(app) as client:
        created = await client.post("/api/jobs", files=_multipart(files))
        assert created.status_code == 201
        return await _wait_job_done(client, created.json()["job_id"])


class TestJobMeta:
    """meta.json is written when a job reaches its terminal state."""

    async def test_meta_written_at_terminal_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app(tmp_path)
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _fake_converter(gate)
        )
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            meta_path = tmp_path / "jobs" / job_id / "meta.json"
            running = (await client.get(f"/api/jobs/{job_id}")).json()
            assert running["finished_at"] is None  # not terminal yet
            assert not meta_path.exists()  # meta only lands at terminal state
            gate.set()
            data = await _wait_job_done(client, job_id)

        assert isinstance(data["finished_at"], str)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["job_id"] == job_id
        assert meta["created_at"] == data["created_at"]
        assert meta["finished_at"] == data["finished_at"]
        assert meta["status"] == "done"
        assert meta["options"] == {"preset": None, "llm": None, "ocr": None}
        assert meta["items"] == data["items"]  # full item snapshot
        item = meta["items"][0]
        assert item["output"] == "doc.txt.md"
        assert item["finished_at"] is not None
        assert item["cost_usd"] == 0.25


class TestRehydrate:
    """Startup rehydrate registers terminal jobs found on disk as archived."""

    async def test_archived_job_serves_all_read_endpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        first_run = await _run_one_job(_make_app(tmp_path), [("doc.txt", b"hello")])
        job_id = first_run["job_id"]

        # "Restart": a fresh app over the same jobs root.
        async with _serve_client(_make_app(tmp_path)) as client:
            snapshot = await client.get(f"/api/jobs/{job_id}")
            assert snapshot.status_code == 200
            data = snapshot.json()
            assert data["status"] == "done"
            assert data["created_at"] == first_run["created_at"]
            assert data["finished_at"] == first_run["finished_at"]
            assert data["items"] == first_run["items"]

            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
            assert result.status_code == 200
            payload = result.json()
            assert payload["variant"] == "base"
            assert "converted doc.txt" in payload["markdown"]

            download = await client.get(f"/api/jobs/{job_id}/files/doc.txt.md")
            assert download.status_code == 200
            assert "attachment" in download.headers["content-disposition"]

            archive = await client.get(f"/api/jobs/{job_id}/archive")
            assert archive.status_code == 200
            with zipfile.ZipFile(BytesIO(archive.content)) as zf:
                assert "doc.txt.md" in zf.namelist()

    async def test_archived_job_events_replay_snapshot_then_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        first_run = await _run_one_job(_make_app(tmp_path), [("doc.txt", b"hello")])
        job_id = first_run["job_id"]

        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await asyncio.wait_for(
                client.get(f"/api/jobs/{job_id}/events"), timeout=10
            )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert [name for name, _ in events] == ["snapshot", "job"]
        assert events[0][1]["job_id"] == job_id
        assert events[0][1]["status"] == "done"
        assert events[1][1] == {"status": "done", "done": 1, "failed": 0, "total": 1}

    async def test_rehydrate_backfills_legacy_item_completion_time(
        self, tmp_path: Path
    ) -> None:
        job_dir = tmp_path / "jobs" / "legacy-job"
        job_dir.mkdir(parents=True)
        finished_at = "2026-07-12T10:01:00+00:00"
        (job_dir / "meta.json").write_text(
            json.dumps(
                {
                    "job_id": "legacy-job",
                    "created_at": "2026-07-12T10:00:00+00:00",
                    "finished_at": finished_at,
                    "status": "done",
                    "items": [
                        {
                            "item_id": "i1",
                            "name": "legacy.txt",
                            "kind": "file",
                            "status": "done",
                            "duration_ms": 1000,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        async with _serve_client(_make_app(tmp_path)) as client:
            snapshot = (await client.get("/api/jobs/legacy-job")).json()

        assert snapshot["items"][0]["finished_at"] == finished_at

    async def test_rehydrate_skips_dirs_without_valid_terminal_meta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jobs_root = tmp_path / "jobs"
        (jobs_root / "no-meta" / "out").mkdir(parents=True)
        broken = jobs_root / "broken-meta"
        broken.mkdir(parents=True)
        (broken / "meta.json").write_text("{not json", encoding="utf-8")
        crashed = jobs_root / "still-running"
        crashed.mkdir(parents=True)
        (crashed / "meta.json").write_text(
            json.dumps({"status": "running", "items": []}), encoding="utf-8"
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            history = (await client.get("/api/history")).json()
            assert history == []
            for job_id in ("no-meta", "broken-meta", "still-running"):
                assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404


class TestHistory:
    """GET /api/history listing and DELETE /api/history/{job_id}."""

    async def test_list_shape_order_and_dedupe_across_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        first_run = await _run_one_job(
            _make_app(tmp_path),
            [("old.txt", b"1"), ("b.txt", b"2"), ("c.txt", b"3"), ("d.txt", b"4")],
        )

        # Restart, rehydrate the first job, then finish a second one in-process.
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("new.txt", b"hello")])
            )
            second = await _wait_job_done(client, created.json()["job_id"])
            history = (await client.get("/api/history")).json()

        assert [h["job_id"] for h in history] == [
            second["job_id"],
            first_run["job_id"],
        ]  # newest first, each terminal job exactly once
        newest, oldest = history
        assert newest == {
            "job_id": second["job_id"],
            "created_at": second["created_at"],
            "finished_at": second["finished_at"],
            "status": "done",
            "total": 1,
            "done": 1,
            "failed": 0,
            "skipped": 0,
            "llm_enhanced": 0,
            "cost_usd": 0.25,
            "names_preview": ["new.txt"],
            "kinds_preview": ["file"],
            "duration_ms": newest["duration_ms"],
            "size_bytes": newest["size_bytes"],
            "origin": "web",
        }
        assert newest["duration_ms"] >= 0
        assert newest["size_bytes"] > 0
        assert oldest["total"] == 4
        assert oldest["names_preview"] == ["old.txt", "b.txt", "c.txt"]  # first 3

    async def test_download_all_includes_live_and_rehydrated_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        await _run_one_job(_make_app(tmp_path), [("report.txt", b"old")])

        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("report.txt", b"new")])
            )
            await _wait_job_done(client, created.json()["job_id"])
            archive = await client.get("/api/history/archive")

        assert archive.status_code == 200
        assert "markitai-all.zip" in archive.headers["content-disposition"]
        with zipfile.ZipFile(BytesIO(archive.content)) as zf:
            assert set(zf.namelist()) == {
                "report.txt/report.txt.md",
                "report.txt (2)/report.txt.md",
            }

    async def test_download_all_without_completed_jobs_is_404(
        self, tmp_path: Path
    ) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            archive = await client.get("/api/history/archive")
        assert archive.status_code == 404

    async def test_running_job_is_not_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _fake_converter(gate)
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            assert (await client.get("/api/history")).json() == []
            gate.set()
            await _wait_job_done(client, job_id)
            history = (await client.get("/api/history")).json()
        assert [h["job_id"] for h in history] == [job_id]

    async def test_delete_removes_dir_and_registry_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job_dir = tmp_path / "jobs" / job_id
            assert job_dir.is_dir()

            resp = await client.delete(f"/api/history/{job_id}")
            assert resp.status_code == 204
            assert not job_dir.exists()
            assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
            assert (await client.get("/api/history")).json() == []
            missing = await client.delete(f"/api/history/{job_id}")
            assert missing.status_code == 404

    async def test_delete_running_job_is_409(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = asyncio.Event()
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _fake_converter(gate)
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            resp = await client.delete(f"/api/history/{job_id}")
            assert resp.status_code == 409
            assert (tmp_path / "jobs" / job_id).is_dir()  # nothing deleted
            gate.set()
            await _wait_job_done(client, job_id)
            done = await client.delete(f"/api/history/{job_id}")
            assert done.status_code == 204


class TestHistorySizeCache:
    """size_bytes comes from the terminal-state cache, not a per-refresh rglob."""

    async def test_terminal_meta_caches_size_and_history_reuses_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("doc.txt", b"hello")])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            meta = json.loads(
                (tmp_path / "jobs" / job_id / "meta.json").read_text(encoding="utf-8")
            )
            assert meta["dir_size_bytes"] > 0

            def fail_rglob(job_dir: Path) -> int:
                pytest.fail("history refresh must use the cached size")

            monkeypatch.setattr("markitai.serve.jobs.job_dir_size", fail_rglob)
            history = (await client.get("/api/history")).json()
        assert history[0]["size_bytes"] == meta["dir_size_bytes"]

    async def test_legacy_meta_without_size_is_computed_once_and_persisted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_dir = tmp_path / "jobs" / "legacy-job"
        (job_dir / "out").mkdir(parents=True)
        (job_dir / "out" / "legacy.txt.md").write_text("x" * 64, encoding="utf-8")
        meta_path = job_dir / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "job_id": "legacy-job",
                    "created_at": "2026-07-12T10:00:00+00:00",
                    "finished_at": "2026-07-12T10:01:00+00:00",
                    "status": "done",
                    "items": [{"item_id": "i1", "name": "legacy.txt", "kind": "file"}],
                }
            ),
            encoding="utf-8",
        )

        async with _serve_client(_make_app(tmp_path)) as client:
            first = (await client.get("/api/history")).json()
            assert first[0]["size_bytes"] >= 64
            # The lazy compute is persisted so a restart also skips the rglob.
            persisted = json.loads(meta_path.read_text(encoding="utf-8"))
            assert persisted["dir_size_bytes"] == first[0]["size_bytes"]

            def fail_rglob(job_dir: Path) -> int:
                pytest.fail("second refresh must use the cached size")

            monkeypatch.setattr("markitai.serve.jobs.job_dir_size", fail_rglob)
            second = (await client.get("/api/history")).json()
        assert second[0]["size_bytes"] == first[0]["size_bytes"]


class TestCliRecordedHistory:
    """CLI runs recorded via record_cli_job show up as origin "cli"."""

    def _record(self, jobs_root: Path, work: Path) -> Path:
        """Write a two-item CLI history job (one done file, one failed URL)."""
        from markitai.runs.history import record_cli_job
        from markitai.runs.types import Outcome

        work.mkdir(exist_ok=True)
        out = work / "notes.txt.md"
        out.write_text("# converted notes\n", encoding="utf-8")
        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="notes.txt",
                    status="completed",
                    output_path=out,
                    duration=0.5,
                ),
                Outcome(
                    kind="url",
                    source="https://example.com",
                    status="failed",
                    error="fetch boom",
                ),
            ],
            options={"preset": None, "llm": False, "ocr": False, "origin": "cli"},
            jobs_root=jobs_root,
        )
        assert job_dir is not None
        return job_dir

    async def test_recorded_job_serves_all_read_endpoints(self, tmp_path: Path) -> None:
        job_dir = self._record(tmp_path / "jobs", tmp_path / "work")

        async with _serve_client(_make_app(tmp_path)) as client:
            history = (await client.get("/api/history")).json()
            assert [h["job_id"] for h in history] == [job_dir.name]
            entry = history[0]
            assert entry["origin"] == "cli"
            assert entry["done"] == 1
            assert entry["failed"] == 1
            assert entry["names_preview"] == ["notes.txt", "https://example.com"]

            snapshot = (await client.get(f"/api/jobs/{job_dir.name}")).json()
            assert snapshot["options"]["origin"] == "cli"
            items = {item["name"]: item for item in snapshot["items"]}
            assert items["notes.txt"]["status"] == "done"
            assert items["notes.txt"]["output"] == "notes.txt.md"
            assert items["notes.txt"]["duration_ms"] == 500
            assert items["https://example.com"]["status"] == "error"
            assert items["https://example.com"]["error"] == "fetch boom"

            result = await client.get(f"/api/jobs/{job_dir.name}/items/i1/result")
            assert result.status_code == 200
            assert "converted notes" in result.json()["markdown"]

            download = await client.get(f"/api/jobs/{job_dir.name}/files/notes.txt.md")
            assert download.status_code == 200
            assert "attachment" in download.headers["content-disposition"]

            archive = await client.get(f"/api/jobs/{job_dir.name}/archive")
            assert archive.status_code == 200
            with zipfile.ZipFile(BytesIO(archive.content)) as zf:
                assert "notes.txt.md" in zf.namelist()

    async def test_recorded_after_startup_is_discovered_live(
        self, tmp_path: Path
    ) -> None:
        """GET /api/history picks up CLI records written while serving."""
        work = tmp_path / "work"
        work.mkdir()
        async with _serve_client(_make_app(tmp_path)) as client:
            assert (await client.get("/api/history")).json() == []
            job_dir = self._record(tmp_path / "jobs", work)
            history = (await client.get("/api/history")).json()
            # ...and a second refresh neither duplicates nor re-reads it.
            again = (await client.get("/api/history")).json()

        assert [h["job_id"] for h in history] == [job_dir.name]
        assert history[0]["origin"] == "cli"
        assert [h["job_id"] for h in again] == [job_dir.name]

    async def test_web_and_cli_jobs_keep_their_origins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        work = tmp_path / "work"
        work.mkdir()
        cli_job = self._record(tmp_path / "jobs", work)

        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart([("web.txt", b"hi")])
            )
            web = await _wait_job_done(client, created.json()["job_id"])
            history = (await client.get("/api/history")).json()

        origins = {h["job_id"]: h["origin"] for h in history}
        assert origins == {web["job_id"]: "web", cli_job.name: "cli"}


class TestHistoryTTL:
    """Stale-job cleanup keeps history for 7 days."""

    def test_default_ttl_is_seven_days(self, tmp_path: Path) -> None:
        import os

        from markitai.serve.jobs import JOB_TTL_HOURS, cleanup_stale_jobs

        assert JOB_TTL_HOURS == 7 * 24.0
        jobs_root = tmp_path / "jobs"
        recent = jobs_root / "six-days-old"
        stale = jobs_root / "eight-days-old"
        recent.mkdir(parents=True)
        stale.mkdir(parents=True)
        six_days = time.time() - 6 * 24 * 3600
        eight_days = time.time() - 8 * 24 * 3600
        os.utime(recent, (six_days, six_days))
        os.utime(stale, (eight_days, eight_days))

        assert cleanup_stale_jobs(jobs_root) == 1
        assert recent.exists()
        assert not stale.exists()

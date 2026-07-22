"""Tests for recording CLI conversions into the serve job history."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from markitai.runs.history import DEFAULT_SERVE_JOBS_ROOT, record_cli_job
from markitai.runs.types import Outcome

OPTIONS = {"preset": None, "llm": False, "ocr": False, "origin": "cli"}


def _read_meta(job_dir: Path) -> dict:
    return json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """A fake conversion output directory with one output and assets."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "doc.txt.md").write_text("# converted\n", encoding="utf-8")
    assets = work / ".markitai" / "assets"
    assets.mkdir(parents=True)
    (assets / "doc.txt.0001.png").write_bytes(b"png")
    states = work / ".markitai" / "states"
    states.mkdir(parents=True)
    (states / "markitai.state.json").write_text("{}", encoding="utf-8")
    return work


class TestRecordCliJob:
    """Meta shape, item mapping, and atomic publish."""

    def test_meta_shape_and_item_mapping(self, tmp_path: Path, work_dir: Path) -> None:
        items = [
            Outcome(
                kind="file",
                source="doc.txt",
                status="completed",
                output_path=work_dir / "doc.txt.md",
                duration=1.25,
                cost_usd=0.5,
            ),
            Outcome(
                kind="url",
                source="https://example.com",
                status="failed",
                error="fetch boom",
            ),
            Outcome(
                kind="file",
                source="other.txt",
                status="skipped",
                skip_reason="exists",
                output_path=work_dir / "doc.txt.md",
            ),
        ]
        started = datetime(2026, 7, 22, 10, 0, 0).astimezone()
        job_dir = record_cli_job(
            items, options=OPTIONS, jobs_root=tmp_path / "jobs", started_at=started
        )

        assert job_dir is not None
        assert len(job_dir.name) == 12
        meta = _read_meta(job_dir)
        assert meta["job_id"] == job_dir.name
        assert meta["status"] == "done"
        assert meta["options"] == OPTIONS
        assert meta["created_at"].startswith("2026-07-22T10:00:00")
        assert meta["finished_at"] >= meta["created_at"]
        assert meta["dir_size_bytes"] > 0

        done, failed, skipped = meta["items"]
        assert done == {
            "item_id": "i1",
            "name": "doc.txt",
            "kind": "file",
            "status": "done",
            "error": None,
            "output": "doc.txt.md",
            "output_name": "doc.txt.md",
            "duration_ms": 1250,
            "finished_at": done["finished_at"],
            "cost_usd": 0.5,
            "llm_enhanced": False,
            "operation": "convert",
            "skipped": False,
            "skip_reason": None,
        }
        assert failed["status"] == "error"
        assert failed["error"] == "fetch boom"
        assert failed["kind"] == "url"
        assert failed["output"] is None
        # Second item referencing the same file gets a de-conflicted name.
        assert skipped["status"] == "done"
        assert skipped["skipped"] is True
        assert skipped["skip_reason"] == "exists"
        assert skipped["output"] == "doc.txt (2).md"

        # Outputs and referenced assets are copied; batch bookkeeping is not.
        copied = {
            p.relative_to(job_dir / "out").as_posix()
            for p in (job_dir / "out").rglob("*")
            if p.is_file()
        }
        assert copied == {
            "doc.txt.md",
            "doc.txt (2).md",
            ".markitai/assets/doc.txt.0001.png",
        }
        # Sources are not copied and no stray dirs are left behind.
        assert not (job_dir / "uploads").exists()
        assert list((tmp_path / "jobs").glob(".tmp-*")) == []

    def test_llm_enhanced_output_flag(self, tmp_path: Path, work_dir: Path) -> None:
        enhanced = work_dir / "doc.llm.md"
        enhanced.write_text("# enhanced\n", encoding="utf-8")
        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="doc.md",
                    status="completed",
                    output_path=enhanced,
                )
            ],
            options=OPTIONS,
            jobs_root=tmp_path / "jobs",
        )
        assert job_dir is not None
        item = _read_meta(job_dir)["items"][0]
        assert item["output"] == "doc.llm.md"
        assert item["llm_enhanced"] is True

    def test_nested_batch_outputs_find_root_assets(self, tmp_path: Path) -> None:
        """A nested batch output references assets at the output root."""
        out_root = tmp_path / "out"
        nested = out_root / "sub"
        nested.mkdir(parents=True)
        (nested / "a.txt.md").write_text("# a\n", encoding="utf-8")
        shots = out_root / ".markitai" / "screenshots"
        shots.mkdir(parents=True)
        (shots / "a.txt.page0001.jpg").write_bytes(b"jpg")

        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="sub/a.txt",
                    status="completed",
                    output_path=nested / "a.txt.md",
                )
            ],
            options=OPTIONS,
            jobs_root=tmp_path / "jobs",
        )
        assert job_dir is not None
        assert (job_dir / "out" / ".markitai" / "screenshots").is_dir()

    def test_empty_items_records_nothing(self, tmp_path: Path) -> None:
        assert record_cli_job([], options=OPTIONS, jobs_root=tmp_path / "jobs") is None
        assert not (tmp_path / "jobs").exists()


class TestBestEffort:
    """Recording must never break a conversion."""

    def test_missing_output_records_item_without_output(self, tmp_path: Path) -> None:
        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="gone.txt",
                    status="completed",
                    output_path=tmp_path / "gone.md",
                )
            ],
            options=OPTIONS,
            jobs_root=tmp_path / "jobs",
        )
        assert job_dir is not None
        item = _read_meta(job_dir)["items"][0]
        assert item["status"] == "done"
        assert item["output"] is None

    def test_uncopyable_output_records_item_without_output(
        self, tmp_path: Path, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_copy2 = shutil.copy2

        def flaky_copy2(src: Path, dst: Path) -> None:
            if Path(src).name == "doc.txt.md":
                raise OSError("simulated read failure")
            real_copy2(src, dst)

        monkeypatch.setattr(shutil, "copy2", flaky_copy2)
        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="doc.txt",
                    status="completed",
                    output_path=work_dir / "doc.txt.md",
                )
            ],
            options=OPTIONS,
            jobs_root=tmp_path / "jobs",
        )
        assert job_dir is not None
        item = _read_meta(job_dir)["items"][0]
        assert item["output"] is None
        # The job dir is still published (with the remaining assets).
        assert (job_dir / "out").is_dir()
        assert list((tmp_path / "jobs").glob(".tmp-*")) == []

    def test_total_failure_returns_none_and_cleans_up(self, tmp_path: Path) -> None:
        # A file where the jobs root should be: mkdir must fail.
        blocker = tmp_path / "jobs"
        blocker.write_text("not a dir", encoding="utf-8")
        result = record_cli_job(
            [Outcome(kind="file", source="a.txt", status="failed", error="x")],
            options=OPTIONS,
            jobs_root=blocker,
        )
        assert result is None
        assert blocker.read_text(encoding="utf-8") == "not a dir"


class TestRehydrateRoundtrip:
    """A recorded job registers as archived serve history with origin cli."""

    def test_rehydrate_registers_recorded_job(self, tmp_path: Path) -> None:
        pytest.importorskip("fastapi")
        from markitai.config import MarkitaiConfig
        from markitai.serve.jobs import JobRegistry, rehydrate_jobs

        out = tmp_path / "doc.md"
        out.write_text("# doc\n", encoding="utf-8")
        job_dir = record_cli_job(
            [
                Outcome(
                    kind="file",
                    source="doc.md",
                    status="completed",
                    output_path=out,
                    duration=0.25,
                )
            ],
            options=OPTIONS,
            jobs_root=tmp_path / "jobs",
        )
        assert job_dir is not None

        registry = JobRegistry(tmp_path / "jobs")
        assert rehydrate_jobs(registry, MarkitaiConfig()) == 1
        job = registry.get(job_dir.name)
        assert job is not None
        assert job.status == "done"
        assert job.options["origin"] == "cli"
        assert job.options["llm"] is False
        item = job.items[0]
        assert item.name == "doc.md"
        assert item.output == "doc.md"
        assert item.duration_ms == 250

        # A second pass (as /api/history does) neither duplicates nor errors.
        assert rehydrate_jobs(registry, MarkitaiConfig()) == 0
        assert len(registry.jobs) == 1

    def test_tmp_dir_is_never_registered(self, tmp_path: Path) -> None:
        pytest.importorskip("fastapi")
        from markitai.config import MarkitaiConfig
        from markitai.serve.jobs import JobRegistry, rehydrate_jobs

        # Simulate a crash leftover: a half-assembled job dir without meta.
        tmp_dir = tmp_path / "jobs" / ".tmp-deadbeef"
        (tmp_dir / "out").mkdir(parents=True)
        (tmp_dir / "out" / "a.md").write_text("# a\n", encoding="utf-8")

        registry = JobRegistry(tmp_path / "jobs")
        assert rehydrate_jobs(registry, MarkitaiConfig()) == 0
        assert registry.jobs == {}


def test_default_jobs_root_points_at_serve_jobs() -> None:
    assert DEFAULT_SERVE_JOBS_ROOT.parts[-3:] == (".markitai", "serve", "jobs")

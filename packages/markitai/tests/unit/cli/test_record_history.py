"""Tests for --record-history: precedence, stdout-mode skip, and recording."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from markitai.cli.main import app
from markitai.runs import Outcome


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture record_cli_job calls without touching the real jobs root."""
    calls: list[dict] = []

    def fake_record(items, *, options, jobs_root, started_at=None):
        calls.append(
            {
                "items": items,
                "options": options,
                "jobs_root": jobs_root,
                "started_at": started_at,
            }
        )
        return jobs_root / "abc123def456"

    monkeypatch.setattr("markitai.runs.history.record_cli_job", fake_record)
    return calls


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKITAI_RECORD_HISTORY", raising=False)


def _outcome(tmp_path: Path, name: str = "doc.txt") -> Outcome:
    out = tmp_path / f"{name}.md"
    out.write_text("# converted\n", encoding="utf-8")
    return Outcome(
        kind="file", source=name, status="completed", output_path=out, duration=0.1
    )


def _fake_single_file(outcome: Outcome):
    async def fake(input_path, output_dir, cfg, *args, **kwargs):
        history = kwargs.get("history")
        if history is not None:
            history.append(outcome)

    return fake


def _invoke_single_file(
    tmp_path: Path, recorded: list[dict], extra_args: list[str], **invoke_kwargs
):
    """Invoke the CLI on a single file with a faked processor."""
    src = tmp_path / "doc.txt"
    src.write_text("hello", encoding="utf-8")
    runner = CliRunner()
    with patch(
        "markitai.cli.processors.file.process_single_file",
        side_effect=_fake_single_file(_outcome(tmp_path)),
    ):
        return runner.invoke(app, [str(src), *extra_args], **invoke_kwargs)


class TestRecordHistoryPrecedence:
    """flag > MARKITAI_RECORD_HISTORY > history.record config > off."""

    def test_default_off(self, tmp_path: Path, recorded: list[dict]) -> None:
        result = _invoke_single_file(tmp_path, recorded, ["-o", str(tmp_path / "o")])
        assert result.exit_code == 0, result.output
        assert recorded == []

    def test_flag_enables(self, tmp_path: Path, recorded: list[dict]) -> None:
        result = _invoke_single_file(
            tmp_path, recorded, ["-o", str(tmp_path / "o"), "--record-history"]
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1

    def test_config_enables(self, tmp_path: Path, recorded: list[dict]) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            [
                "-o",
                str(tmp_path / "o"),
                "--config-json",
                '{"history": {"record": true}}',
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1

    def test_env_enables(self, tmp_path: Path, recorded: list[dict]) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            ["-o", str(tmp_path / "o")],
            env={"MARKITAI_RECORD_HISTORY": "1"},
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_falsy_env_overrides_config(
        self, tmp_path: Path, recorded: list[dict], value: str
    ) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            [
                "-o",
                str(tmp_path / "o"),
                "--config-json",
                '{"history": {"record": true}}',
            ],
            env={"MARKITAI_RECORD_HISTORY": value},
        )
        assert result.exit_code == 0, result.output
        assert recorded == []

    def test_flag_overrides_env(self, tmp_path: Path, recorded: list[dict]) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            ["-o", str(tmp_path / "o"), "--no-record-history"],
            env={"MARKITAI_RECORD_HISTORY": "1"},
        )
        assert result.exit_code == 0, result.output
        assert recorded == []

    def test_flag_overrides_falsy_env(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            ["-o", str(tmp_path / "o"), "--record-history"],
            env={"MARKITAI_RECORD_HISTORY": "0"},
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1


class TestRecordHistoryCall:
    """The recorder receives one job with the run's items and options."""

    def test_records_single_job_with_origin_cli(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        result = _invoke_single_file(
            tmp_path, recorded, ["-o", str(tmp_path / "o"), "--record-history"]
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1
        call = recorded[0]
        assert [item.source for item in call["items"]] == ["doc.txt"]
        assert call["options"]["origin"] == "cli"
        assert set(call["options"]) == {"preset", "llm", "ocr", "origin"}
        assert call["started_at"] is not None
        assert "Recorded in history" in result.output
        assert "markitai serve" in result.output

    def test_quiet_suppresses_confirmation(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        result = _invoke_single_file(
            tmp_path,
            recorded,
            ["-o", str(tmp_path / "o"), "--record-history", "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1
        assert "Recorded in history" not in result.output

    def test_stdout_mode_skips_recording(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        # No -o: single-file stdout mode must stay clean (and has no
        # durable output to record anyway).
        result = _invoke_single_file(tmp_path, recorded, ["--record-history"])
        assert result.exit_code == 0, result.output
        assert recorded == []
        assert "Recorded in history" not in result.output

    def test_partial_failure_still_records(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        async def fake_then_exit(input_path, output_dir, cfg, *args, **kwargs):
            kwargs["history"].append(_outcome(tmp_path))
            raise SystemExit(10)  # PARTIAL_FAILURE, as the processors do

        src = tmp_path / "doc.txt"
        src.write_text("hello", encoding="utf-8")
        runner = CliRunner()
        with patch(
            "markitai.cli.processors.file.process_single_file",
            side_effect=fake_then_exit,
        ):
            result = runner.invoke(
                app,
                [str(src), "-o", str(tmp_path / "o"), "--record-history"],
            )
        assert result.exit_code == 10
        assert len(recorded) == 1

    def test_dry_run_records_nothing(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        async def fake_dry_run(input_path, output_dir, cfg, *args, **kwargs):
            # The real processors raise SystemExit(0) on --dry-run before
            # any conversion (and any outcome) happens.
            raise SystemExit(0)

        src = tmp_path / "doc.txt"
        src.write_text("hello", encoding="utf-8")
        runner = CliRunner()
        with patch(
            "markitai.cli.processors.file.process_single_file",
            side_effect=fake_dry_run,
        ):
            result = runner.invoke(
                app,
                [
                    str(src),
                    "-o",
                    str(tmp_path / "o"),
                    "--record-history",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert recorded == []


class TestRecordHistoryDispatch:
    """Every input mode threads the collector to its processor."""

    def test_directory_batch_records_all_items_as_one_job(
        self, tmp_path: Path, recorded: list[dict]
    ) -> None:
        async def fake_batch(input_dir, output_dir, cfg, *args, **kwargs):
            kwargs["history"].extend(
                [
                    _outcome(tmp_path, "a.txt"),
                    Outcome(
                        kind="file",
                        source="b.txt",
                        status="failed",
                        error="boom",
                        duration=0.2,
                    ),
                ]
            )

        runner = CliRunner()
        with patch(
            "markitai.cli.processors.batch.process_batch", side_effect=fake_batch
        ):
            result = runner.invoke(
                app,
                [str(tmp_path), "-o", str(tmp_path / "o"), "--record-history"],
            )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1
        items = recorded[0]["items"]
        assert [(i.source, i.status) for i in items] == [
            ("a.txt", "completed"),
            ("b.txt", "failed"),
        ]

    def test_single_url_records(self, tmp_path: Path, recorded: list[dict]) -> None:
        async def fake_url(url, output_dir, cfg, *args, **kwargs):
            kwargs["history"].append(
                Outcome(
                    kind="url",
                    source=url,
                    status="completed",
                    output_path=_outcome(tmp_path).output_path,
                    duration=0.3,
                )
            )

        runner = CliRunner()
        with patch("markitai.cli.processors.url.process_url", side_effect=fake_url):
            result = runner.invoke(
                app,
                [
                    "https://example.com/page",
                    "-o",
                    str(tmp_path / "o"),
                    "--record-history",
                ],
            )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1
        (item,) = recorded[0]["items"]
        assert item.kind == "url"
        assert item.source == "https://example.com/page"

    def test_url_list_batch_records(self, tmp_path: Path, recorded: list[dict]) -> None:
        urls_file = tmp_path / "list.urls"
        urls_file.write_text(
            "https://example.com/a\nhttps://example.com/b\n", encoding="utf-8"
        )

        async def fake_url_batch(entries, output_dir, cfg, *args, **kwargs):
            for entry in entries:
                kwargs["history"].append(
                    Outcome(
                        kind="url",
                        source=entry.url,
                        status="failed",
                        error="boom",
                        duration=0.1,
                    )
                )

        runner = CliRunner()
        with patch(
            "markitai.cli.processors.url.process_url_batch",
            side_effect=fake_url_batch,
        ):
            result = runner.invoke(
                app,
                [str(urls_file), "-o", str(tmp_path / "o"), "--record-history"],
            )
        assert result.exit_code == 0, result.output
        assert len(recorded) == 1
        assert [i.source for i in recorded[0]["items"]] == [
            "https://example.com/a",
            "https://example.com/b",
        ]


class TestProcessorOutcomes:
    """The real processors append well-formed outcomes to the collector."""

    async def test_process_single_file_appends_completed_outcome(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.file import process_single_file
        from markitai.config import MarkitaiConfig

        src = tmp_path / "doc.txt"
        src.write_text("# hello\n\nworld\n", encoding="utf-8")
        history: list[Outcome] = []
        await process_single_file(
            src,
            tmp_path / "out",
            MarkitaiConfig(),
            False,
            quiet=True,
            history=history,
        )
        assert len(history) == 1
        (outcome,) = history
        assert outcome.kind == "file"
        assert outcome.source == "doc.txt"
        assert outcome.status == "completed"
        assert outcome.output_path is not None
        assert outcome.output_path.is_file()
        assert outcome.duration is not None and outcome.duration >= 0

    async def test_process_single_file_appends_failed_outcome(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.file import process_single_file
        from markitai.config import MarkitaiConfig

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"garbage not a real pdf")
        history: list[Outcome] = []
        with pytest.raises(SystemExit):
            await process_single_file(
                src,
                tmp_path / "out",
                MarkitaiConfig(),
                False,
                quiet=True,
                history=history,
            )
        assert len(history) == 1
        (outcome,) = history
        assert outcome.status == "failed"
        assert outcome.error

    async def test_unsupported_format_records_nothing(self, tmp_path: Path) -> None:
        """Validation failures (unsupported format) are usage errors, not
        conversion results — nothing is recorded."""
        from markitai.cli.processors.file import process_single_file
        from markitai.config import MarkitaiConfig

        src = tmp_path / "doc.xyz"
        src.write_text("garbage", encoding="utf-8")
        history: list[Outcome] = []
        with pytest.raises(SystemExit):
            await process_single_file(
                src,
                tmp_path / "out",
                MarkitaiConfig(),
                False,
                quiet=True,
                history=history,
            )
        assert history == []

    async def test_process_batch_appends_per_item_outcomes(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch import process_batch
        from markitai.config import MarkitaiConfig

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        sub = input_dir / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("# b\n", encoding="utf-8")
        history: list[Outcome] = []
        await process_batch(
            input_dir,
            tmp_path / "out",
            MarkitaiConfig(),
            False,
            False,
            quiet=True,
            history=history,
        )
        assert {(o.source, o.status) for o in history} == {
            ("a.txt", "completed"),
            ("sub/b.txt", "completed"),
        }
        for outcome in history:
            assert outcome.kind == "file"
            assert outcome.output_path is not None
            assert outcome.duration is not None


class TestFullStackRecording:
    """Real conversion + real recorder writing into a patched jobs root."""

    def test_single_file_run_writes_history_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jobs_root = tmp_path / "jobs"
        monkeypatch.setattr("markitai.runs.history.DEFAULT_SERVE_JOBS_ROOT", jobs_root)

        src = tmp_path / "doc.txt"
        src.write_text("# hello\n\nworld\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [str(src), "-o", str(tmp_path / "out"), "--record-history"],
        )
        assert result.exit_code == 0, result.output
        assert "Recorded in history" in result.output

        job_dirs = [p for p in jobs_root.iterdir() if p.is_dir()]
        assert len(job_dirs) == 1
        import json

        meta = json.loads((job_dirs[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "done"
        assert meta["options"]["origin"] == "cli"
        assert [(i["name"], i["status"]) for i in meta["items"]] == [
            ("doc.txt", "done")
        ]
        output = meta["items"][0]["output"]
        assert (job_dirs[0] / "out" / output).is_file()

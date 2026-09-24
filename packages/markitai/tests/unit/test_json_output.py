"""Tests for the ``markitai --json`` result envelope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click import unstyle
from click.testing import CliRunner

from markitai.runs import Outcome
from markitai.runs import json_output as json_result


class TestEnvelope:
    """The envelope is the machine contract scripts depend on."""

    def test_completed_item_is_ok(self, tmp_path: Path) -> None:
        outcome = Outcome(
            kind="file",
            source="a.txt",
            status="completed",
            output_path=tmp_path / "a.txt.md",
            duration=1.2345,
        )
        envelope = json_result.build_envelope([outcome])

        assert envelope["ok"] is True
        assert envelope["version"] == json_result.ENVELOPE_VERSION
        assert envelope["totals"] == {
            "total": 1,
            "completed": 1,
            "failed": 0,
            "skipped": 0,
            "pending": 0,
            "cost_usd": 0.0,
            "duration_s": 1.234,
        }
        item = envelope["items"][0]
        assert item["output"] == str(tmp_path / "a.txt.md")
        assert item["error"] is None

    def test_failed_item_flips_ok(self) -> None:
        envelope = json_result.build_envelope(
            [
                Outcome(kind="file", source="ok.txt", status="completed"),
                Outcome(
                    kind="file",
                    source="bad.pdf",
                    status="failed",
                    error="Unsupported file format",
                ),
            ]
        )

        assert envelope["ok"] is False
        assert envelope["totals"]["failed"] == 1
        assert envelope["totals"]["completed"] == 1

    def test_pending_item_is_not_failed_and_carries_the_batch(self) -> None:
        """An --llm-batch handoff: the base .md exists and the enhancement is
        paid for, just not collected. Calling it failed sent scripts to
        re-convert (and re-pay); leaving out the batch id left them nothing
        to collect with."""
        batch = {
            "id": "batch_abc",
            "status": "in_progress",
            "collect_command": "markitai --llm-batch-collect batch_abc -o out",
        }
        envelope = json_result.build_envelope(
            [Outcome(kind="file", source="a.pdf", status="pending")], batch=batch
        )

        assert envelope["ok"] is False  # the run has not finished
        assert envelope["error"] is None
        assert envelope["batch"] == batch
        assert envelope["totals"]["pending"] == 1
        assert envelope["totals"]["failed"] == 0
        assert envelope["items"][0]["status"] == "pending"

    def test_batch_is_null_without_a_handoff(self) -> None:
        envelope = json_result.build_envelope(
            [Outcome(kind="file", source="a.txt", status="completed")]
        )

        assert envelope["batch"] is None

    def test_item_warnings_are_rendered(self) -> None:
        outcome = Outcome(
            kind="file",
            source="a.pdf",
            status="completed",
            warnings=["image analysis failed for a-0001.png"],
        )

        envelope = json_result.build_envelope([outcome])

        assert envelope["ok"] is True
        assert envelope["items"][0]["warnings"] == [
            "image analysis failed for a-0001.png"
        ]

    def test_skipped_item_keeps_ok_true(self) -> None:
        """A skip is not a failure; batch runs report skips separately."""
        envelope = json_result.build_envelope(
            [
                Outcome(
                    kind="file",
                    source="img.png",
                    status="skipped",
                    skip_reason="image_only",
                )
            ]
        )

        assert envelope["ok"] is True
        assert envelope["totals"]["skipped"] == 1
        assert envelope["items"][0]["skip_reason"] == "image_only"

    def test_costs_and_durations_are_summed(self) -> None:
        envelope = json_result.build_envelope(
            [
                Outcome(
                    kind="url",
                    source="https://a.test",
                    status="completed",
                    cost_usd=0.1,
                    duration=2.0,
                ),
                Outcome(
                    kind="url",
                    source="https://b.test",
                    status="completed",
                    cost_usd=0.25,
                    duration=3.5,
                ),
            ]
        )

        assert envelope["totals"]["cost_usd"] == pytest.approx(0.35)
        assert envelope["totals"]["duration_s"] == pytest.approx(5.5)

    def test_render_is_parseable_json_with_newline(self) -> None:
        rendered = json_result.render(
            [Outcome(kind="file", source="a.txt", status="completed")]
        )

        assert rendered.endswith("\n")
        assert json.loads(rendered)["items"][0]["source"] == "a.txt"

    def test_empty_run_still_renders(self) -> None:
        envelope = json_result.build_envelope([])

        assert envelope["ok"] is True
        assert envelope["error"] is None
        assert envelope["totals"]["total"] == 0

    def test_partial_item_failure_leaves_the_run_error_empty(self) -> None:
        """`ok` is false because an item failed, not because the run did.

        `error` stays for a run-level rejection (missing input, bad flag
        combination), so a consumer can tell "your invocation was wrong" from
        "one of your files was".
        """
        envelope = json_result.build_envelope(
            [
                Outcome(kind="file", source="ok.txt", status="completed"),
                Outcome(kind="file", source="bad.pdf", status="failed", error="boom"),
            ]
        )

        assert envelope["ok"] is False
        assert envelope["error"] is None
        # Items keep input order: a consumer may pair them with its own list.
        assert [item["source"] for item in envelope["items"]] == [
            "ok.txt",
            "bad.pdf",
        ]

    def test_run_level_error_flips_ok_without_items(self) -> None:
        """A run that died before its first item must not look successful."""
        envelope = json_result.build_envelope([], error="Path 'x' does not exist.")

        assert envelope["ok"] is False
        assert envelope["error"] == "Path 'x' does not exist."
        assert envelope["items"] == []


class TestCliContract:
    """The flag is only valid where stdout can carry the JSON."""

    @pytest.mark.parametrize("color", [False, True])
    def test_json_without_output_is_a_usage_error(
        self, color: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.cli.main import app

        monkeypatch.setattr("rich_click.rich_click.FORCE_TERMINAL", color)
        monkeypatch.setattr(
            "rich_click.rich_click.COLOR_SYSTEM", "standard" if color else None
        )
        result = CliRunner().invoke(
            app,
            ["some.txt", "--json"],
            color=color,
            env={"NO_COLOR": None, "TERM": "xterm"},
        )

        assert result.exit_code == 2
        # Rich styles individual option names on CI and color terminals.
        assert "--json needs -o" in unstyle(result.output)

    def test_json_with_batch_collect_is_a_usage_error(self) -> None:
        """The collect path writes its own output; the envelope would lie."""
        from markitai.cli.main import app

        result = CliRunner().invoke(
            app, ["--json", "-o", "out", "--llm-batch-collect", "batch_123"]
        )

        assert result.exit_code == 2
        assert "--llm-batch-collect" in unstyle(result.output)

    def test_json_with_dry_run_is_a_usage_error(self) -> None:
        """A dry run writes no item, so the envelope could only lie."""
        from markitai.cli.main import app

        result = CliRunner().invoke(
            app, ["--json", "-o", "out", "--dry-run", "some.txt"]
        )

        assert result.exit_code == 2
        assert "--dry-run" in unstyle(result.output)

    def test_missing_path_still_reports_ok_false(self, tmp_path: Path) -> None:
        """stdout stays pure JSON even on an input error."""
        from markitai.cli.main import app

        result = CliRunner().invoke(
            app,
            [str(tmp_path / "nope.pdf"), "-o", str(tmp_path / "out"), "--json"],
        )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "does not exist" in body["error"]

    def test_unsupported_format_still_reports_ok_false(self, tmp_path: Path) -> None:
        from markitai.cli.main import app

        bad = tmp_path / "notes.xyz"
        bad.write_text("x", encoding="utf-8")
        result = CliRunner().invoke(
            app, [str(bad), "-o", str(tmp_path / "out"), "--json"]
        )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert body["items"] == []
        assert "Unsupported file format" in body["error"]

    def test_successful_run_writes_only_json_on_stdout(self, tmp_path: Path) -> None:
        from markitai.cli.main import app

        src = tmp_path / "ok.txt"
        src.write_text("# hi\n", encoding="utf-8")
        result = CliRunner().invoke(
            app, [str(src), "-o", str(tmp_path / "out"), "--json"]
        )

        assert result.exit_code == 0
        assert result.stdout.lstrip().startswith("{")
        body = json.loads(result.stdout)
        assert body["ok"] is True
        assert body["items"][0]["status"] == "completed"

    def test_cjk_source_names_stay_valid_json(self, tmp_path: Path) -> None:
        """An ASCII stdout (LC_ALL=C) must not choke on a CJK filename."""
        from markitai.cli.main import app

        src = tmp_path / "中文 报告.txt"
        src.write_text("# 标题\n", encoding="utf-8")
        result = CliRunner().invoke(
            app, [str(src), "-o", str(tmp_path / "out"), "--json"]
        )

        assert result.exit_code == 0
        body = json.loads(result.stdout)
        assert body["items"][0]["source"] == "中文 报告.txt"

    def test_empty_url_list_reports_ok_false(self, tmp_path: Path) -> None:
        """A .urls file with nothing usable is an input error, not success."""
        from markitai.cli.main import app

        urls = tmp_path / "empty.urls"
        urls.write_text("# only a comment\n", encoding="utf-8")
        result = CliRunner().invoke(
            app, [str(urls), "-o", str(tmp_path / "out"), "--json"]
        )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "No valid URLs" in body["error"]

    def test_malformed_config_still_reports_ok_false(self, tmp_path: Path) -> None:
        """A broken config file is an input error, not a traceback.

        It used to escape as a JSONDecodeError before the envelope was written,
        so ``--json`` produced an empty stdout.
        """
        from markitai.cli.main import app

        src = tmp_path / "doc.txt"
        src.write_text("# hi\n", encoding="utf-8")
        config_file = tmp_path / "config.json"
        config_file.write_text("{ broken", encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [str(src), "-o", str(tmp_path / "out"), "--json", "-c", str(config_file)],
        )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "Invalid JSON" in body["error"]
        assert body["items"] == []
        assert "Traceback" not in result.output

    def test_config_with_invalid_values_still_reports_ok_false(
        self, tmp_path: Path
    ) -> None:
        """The validation path keeps its field-level message and exits 1."""
        from markitai.cli.main import app

        src = tmp_path / "doc.txt"
        src.write_text("# hi\n", encoding="utf-8")
        config_file = tmp_path / "config.json"
        config_file.write_text('{"llm": {"concurrency": 0}}', encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [str(src), "-o", str(tmp_path / "out"), "--json", "-c", str(config_file)],
        )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "Invalid configuration" in body["error"]
        assert "llm.concurrency" in body["error"]

    def test_existing_output_reports_a_skip_with_exit_zero(
        self, tmp_path: Path
    ) -> None:
        """A skip is not a failure, and the reason has to survive into JSON.

        `output.on_conflict = "skip"` is a config-file setting, so the run is
        exercised through MARKITAI_CONFIG exactly as a user would set it.
        """
        from markitai.cli.main import app

        out = tmp_path / "out"
        out.mkdir()
        (out / "doc.txt.md").write_text("# old\n", encoding="utf-8")
        src = tmp_path / "doc.txt"
        src.write_text("# new\n", encoding="utf-8")
        config_file = tmp_path / "config.json"
        config_file.write_text('{"output": {"on_conflict": "skip"}}', encoding="utf-8")

        result = CliRunner().invoke(
            app,
            [str(src), "-o", str(out), "--json", "-c", str(config_file)],
        )

        assert result.exit_code == 0, result.output
        body = json.loads(result.stdout)
        assert body["ok"] is True
        assert body["totals"]["skipped"] == 1
        item = body["items"][0]
        assert item["status"] == "skipped"
        assert item["skip_reason"] == "exists"
        # The kept file is named: it says what the run left alone.
        assert item["output"] == str(out / "doc.txt.md")
        # The untouched file keeps its old content.
        assert (out / "doc.txt.md").read_text(encoding="utf-8") == "# old\n"

    def test_batch_partial_failure_exits_10_and_reports_both(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.main import app

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "ok.txt").write_text("# ok\n", encoding="utf-8")
        (source_dir / "broken.pdf").write_bytes(b"not a pdf")
        result = CliRunner().invoke(
            app, [str(source_dir), "-o", str(tmp_path / "out"), "--json"]
        )

        assert result.exit_code == 10
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert body["error"] is None  # a failed item is not a failed run
        assert body["totals"] == {
            "total": 2,
            "completed": 1,
            "failed": 1,
            "skipped": 0,
            "pending": 0,
            "cost_usd": 0.0,
            "duration_s": body["totals"]["duration_s"],
        }


_OPENAI_POOL = json.dumps(
    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-t",
                    },
                }
            ]
        }
    }
)


class TestLlmBatchCli:
    """--llm-batch results a script can act on."""

    def test_handoff_carries_the_batch_id_and_pending_items(
        self, tmp_path: Path
    ) -> None:
        """Past the wait the batch keeps running. The envelope used to mark
        every item failed and omit the batch id, and stderr was empty under
        --json — nothing left to collect the paid-for results with."""
        from unittest.mock import patch

        from markitai.cli.main import app
        from markitai.cli.processors.batch_llm import BatchHandoff

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        out = tmp_path / "out"

        async def fake_enhancement(cfg, output_dir, *, items, **_kwargs):
            for item in items:
                item.status = "pending"
            return BatchHandoff(
                "batch_h1", "in_progress", output_dir, "Batch still in flight"
            )

        with patch(
            "markitai.cli.processors.batch_llm.run_batch_llm_enhancement",
            side_effect=fake_enhancement,
        ):
            result = CliRunner().invoke(
                app,
                [
                    str(source_dir),
                    "-o",
                    str(out),
                    "--llm",
                    "--llm-batch",
                    "--json",
                    "--config-json",
                    _OPENAI_POOL,
                ],
            )

        assert result.exit_code == 2
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert body["error"] is None
        assert body["batch"]["id"] == "batch_h1"
        assert body["batch"]["status"] == "in_progress"
        assert "--llm-batch-collect batch_h1" in body["batch"]["collect_command"]
        assert [item["status"] for item in body["items"]] == ["pending"]
        assert body["totals"]["pending"] == 1
        assert "--llm-batch-collect batch_h1" in result.stderr

    def test_partial_conversion_failure_still_enhances_the_rest(
        self, tmp_path: Path
    ) -> None:
        """One unconvertible file used to end the run with exit 10 before the
        batch was ever submitted, leaving every good document unenhanced."""
        from unittest.mock import patch

        from markitai.cli.main import app

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        (source_dir / "broken.pdf").write_bytes(b"not a pdf")
        out = tmp_path / "out"
        seen: list[str] = []

        async def fake_enhancement(cfg, output_dir, *, items, **_kwargs):
            seen.extend(item.source for item in items if item.status == "completed")
            return None

        with patch(
            "markitai.cli.processors.batch_llm.run_batch_llm_enhancement",
            side_effect=fake_enhancement,
        ):
            result = CliRunner().invoke(
                app,
                [
                    str(source_dir),
                    "-o",
                    str(out),
                    "--llm",
                    "--llm-batch",
                    "--json",
                    "--config-json",
                    _OPENAI_POOL,
                ],
            )

        assert seen == ["a.txt"]
        assert result.exit_code == 10
        body = json.loads(result.stdout)
        statuses = {item["source"]: item["status"] for item in body["items"]}
        assert statuses == {"a.txt": "completed", "broken.pdf": "failed"}

    def _invoke(
        self, source_dir: Path, out: Path, *extra: str, json_output: bool = True
    ) -> Any:
        from markitai.cli.main import app

        return CliRunner().invoke(
            app,
            [
                str(source_dir),
                "-o",
                str(out),
                "--llm",
                "--llm-batch",
                *(["--json"] if json_output else []),
                "--config-json",
                _OPENAI_POOL,
                *extra,
            ],
        )

    def test_handoff_with_failed_items_exits_2_and_says_so(
        self, tmp_path: Path
    ) -> None:
        """Exit 2 (collect needed) wins over 10, but the failed item must not
        vanish behind it: it is in the envelope and named on stderr."""
        from unittest.mock import patch

        from markitai.cli.processors.batch_llm import BatchHandoff

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        (source_dir / "broken.pdf").write_bytes(b"not a pdf")
        out = tmp_path / "out"

        async def fake_enhancement(cfg, output_dir, *, items, **_kwargs):
            for item in items:
                if item.status == "completed":
                    item.status = "pending"
            return BatchHandoff("batch_h2", "in_progress", output_dir, "In flight")

        with patch(
            "markitai.cli.processors.batch_llm.run_batch_llm_enhancement",
            side_effect=fake_enhancement,
        ):
            result = self._invoke(source_dir, out)

        assert result.exit_code == 2
        body = json.loads(result.stdout)
        assert body["batch"]["id"] == "batch_h2"
        assert body["totals"]["failed"] == 1
        assert body["totals"]["pending"] == 1
        assert "--llm-batch-collect batch_h2" in result.stderr
        assert "1 item(s) failed" in result.stderr

    def test_nothing_converted_skips_the_enhancement(self, tmp_path: Path) -> None:
        """Every conversion failed; the run still entered the enhancement and
        printed "All documents already cached — nothing to submit."."""
        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "broken.pdf").write_bytes(b"not a pdf")
        out = tmp_path / "out"

        # Not --json: that implies --quiet, which hid the line anyway
        result = self._invoke(source_dir, out, json_output=False)

        assert result.exit_code == 10
        assert "cached" not in result.output
        assert "broken.pdf" in result.output

    def test_ctrl_c_while_waiting_prints_the_collect_command(
        self, tmp_path: Path
    ) -> None:
        """Ctrl-C during the wait printed only "Interrupted" (and nothing at
        all under --json), though the batch keeps running and billing."""
        from unittest.mock import patch

        from markitai.cli.processors.batch_llm import BatchHandoff

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        out = tmp_path / "out"

        async def fake_enhancement(cfg, output_dir, *, items, on_submitted, **_kw):
            on_submitted(
                BatchHandoff(
                    "batch_cc",
                    "in_progress",
                    output_dir,
                    "Interrupted while waiting for batch batch_cc.",
                )
            )
            for item in items:
                item.status = "pending"
            raise KeyboardInterrupt

        with patch(
            "markitai.cli.processors.batch_llm.run_batch_llm_enhancement",
            side_effect=fake_enhancement,
        ):
            result = self._invoke(source_dir, out)

        assert result.exit_code != 0
        assert "--llm-batch-collect batch_cc" in result.stderr
        body = json.loads(result.stdout)
        assert body["batch"]["id"] == "batch_cc"
        assert body["error"] == "interrupted before the run finished"
        assert [item["status"] for item in body["items"]] == ["pending"]

    def test_resume_names_a_batch_that_was_never_collected(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import patch

        from markitai.llm.batch_api import BatchDocItem, BatchRunState

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        out = tmp_path / "out"
        BatchRunState(
            batch_id="batch_old",
            model="gpt-4o-mini",
            mode="tool_call",
            provider="openai",
            created_at="2026-09-24T00:00:00+00:00",
            items=[BatchDocItem("doc_0_b", "b.txt", "inputs/0.md", "b.txt.md")],
        ).save(BatchRunState.state_dir_for(out, "batch_old"))

        async def fake_enhancement(cfg, output_dir, *, items, **_kwargs):
            return None

        with patch(
            "markitai.cli.processors.batch_llm.run_batch_llm_enhancement",
            side_effect=fake_enhancement,
        ):
            result = self._invoke(source_dir, out, "--resume")

        assert result.exit_code == 0, result.stderr
        assert "batch_old" in result.stderr
        assert "--llm-batch-collect batch_old" in result.stderr

    def test_unsupported_pool_is_refused_before_converting(
        self, tmp_path: Path
    ) -> None:
        """Found after the conversion, the refusal cost a full pass, and the
        corrected re-run wrote .v2 copies next to the first outputs."""
        from unittest.mock import AsyncMock, patch

        from markitai.cli.main import app

        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("# a\n", encoding="utf-8")
        out = tmp_path / "out"
        gemini_pool = _OPENAI_POOL.replace("openai/gpt-4o-mini", "gemini/flash")

        with patch(
            "markitai.cli.processors.batch.process_batch", new_callable=AsyncMock
        ) as process_batch:
            result = CliRunner().invoke(
                app,
                [
                    str(source_dir),
                    "-o",
                    str(out),
                    "--llm",
                    "--llm-batch",
                    "--json",
                    "--config-json",
                    gemini_pool,
                ],
            )

        assert result.exit_code == 1
        process_batch.assert_not_awaited()
        assert not out.exists()
        body = json.loads(result.stdout)
        assert "openai and anthropic pools" in body["error"]


class TestLlmBatchCollectCli:
    def test_collect_sets_up_logging_like_a_normal_run(self, tmp_path: Path) -> None:
        """Without it loguru's default handler printed DEBUG/INFO internals
        ([Cache] Global cache..., [Providers] Registered...) to the console."""
        from unittest.mock import AsyncMock, patch

        from markitai.cli.main import app

        with (
            patch(
                "markitai.cli.main.setup_logging", return_value=(1, None)
            ) as setup_logging,
            patch(
                "markitai.cli.processors.batch_llm.collect_batch_llm",
                new_callable=AsyncMock,
                return_value=0,
            ),
        ):
            result = CliRunner().invoke(
                app, ["--llm-batch-collect", "batch_1", "-o", str(tmp_path)]
            )

        assert result.exit_code == 0
        setup_logging.assert_called_once()
        assert setup_logging.call_args.kwargs["quiet"] is False


class TestOutputDirectoryErrors:
    """An output directory that cannot be created is a runtime error: one line
    on stderr and the envelope's ``error``, never a traceback."""

    @staticmethod
    def _blocked_output(tmp_path: Path) -> Path:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        return blocker

    def test_directory_batch_reports_error_in_json(self, tmp_path: Path) -> None:
        from markitai.cli.main import app

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("# a\n", encoding="utf-8")
        blocker = self._blocked_output(tmp_path)

        result = CliRunner().invoke(app, [str(docs), "-o", str(blocker), "--json"])

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "Cannot create output directory" in body["error"]
        assert str(blocker) in body["error"]
        assert "Traceback" not in result.output

    def test_url_list_reports_error_in_json(self, tmp_path: Path) -> None:
        from markitai.cli.main import app

        urls = tmp_path / "list.urls"
        urls.write_text("https://example.invalid/a\n", encoding="utf-8")
        blocker = self._blocked_output(tmp_path)

        result = CliRunner().invoke(app, [str(urls), "-o", str(blocker), "--json"])

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "Cannot create output directory" in body["error"]

    def test_single_url_reports_error_in_json(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock, patch

        from markitai.cli.main import app

        target = tmp_path / "blocker" / "sub"
        self._blocked_output(tmp_path)
        with patch(
            "markitai.cli.processors.url.process_url",
            new_callable=AsyncMock,
            side_effect=NotADirectoryError(20, "Not a directory", str(target)),
        ):
            result = CliRunner().invoke(
                app, ["https://example.invalid/", "-o", str(target), "--json"]
            )

        assert result.exit_code == 1
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert "Not a directory" in body["error"]

    def test_without_json_prints_one_line_and_no_traceback(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.main import app

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("# a\n", encoding="utf-8")
        blocker = self._blocked_output(tmp_path)

        result = CliRunner().invoke(app, [str(docs), "-o", str(blocker)])

        assert result.exit_code == 1
        assert "Error: Cannot create output directory" in result.output
        assert "Traceback" not in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)


class TestDescribeOsError:
    def test_names_the_output_directory(self, tmp_path: Path) -> None:
        from markitai.cli.main import describe_os_error

        out = tmp_path / "out"
        error = PermissionError(13, "Permission denied", str(out))

        assert (
            describe_os_error(error, out)
            == f"Cannot create output directory '{out}': Permission denied"
        )

    def test_names_the_failing_path_below_the_output(self, tmp_path: Path) -> None:
        from markitai.cli.main import describe_os_error

        out = tmp_path / "out"
        error = PermissionError(13, "Permission denied", str(out / "assets"))

        message = describe_os_error(error, out)

        assert message.startswith(f"Cannot create output directory '{out}'")
        assert str(out / "assets") in message

    def test_unrelated_path_is_reported_as_is(self, tmp_path: Path) -> None:
        from markitai.cli.main import describe_os_error

        error = OSError(28, "No space left on device", str(tmp_path / "x"))

        assert describe_os_error(error, None) == (
            f"No space left on device: '{tmp_path / 'x'}'"
        )

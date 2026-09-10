"""Tests for the ``markitai --json`` result envelope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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

    def test_json_without_output_is_a_usage_error(self) -> None:
        from markitai.cli.main import app

        result = CliRunner().invoke(app, ["some.txt", "--json"])

        assert result.exit_code == 2
        assert "--json needs -o" in result.output

    def test_json_with_batch_collect_is_a_usage_error(self) -> None:
        """The collect path writes its own output; the envelope would lie."""
        from markitai.cli.main import app

        result = CliRunner().invoke(
            app, ["--json", "-o", "out", "--llm-batch-collect", "batch_123"]
        )

        assert result.exit_code == 2
        assert "--llm-batch-collect" in result.output

    def test_json_with_dry_run_is_a_usage_error(self) -> None:
        """A dry run writes no item, so the envelope could only lie."""
        from markitai.cli.main import app

        result = CliRunner().invoke(
            app, ["--json", "-o", "out", "--dry-run", "some.txt"]
        )

        assert result.exit_code == 2
        assert "--dry-run" in result.output

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
            "cost_usd": 0.0,
            "duration_s": body["totals"]["duration_s"],
        }

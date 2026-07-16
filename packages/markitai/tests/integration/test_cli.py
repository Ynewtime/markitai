"""Integration tests for CLI commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import app

# Note: Uses fixtures from conftest.py:
# - cli_runner (aliased as runner below for backward compatibility)
# - sample_txt_file (aliased as sample_txt below)
# - sample_md_file (aliased as sample_md below)


@pytest.fixture
def runner(cli_runner: CliRunner) -> CliRunner:
    """Alias for cli_runner from conftest.py."""
    return cli_runner


@pytest.fixture
def sample_txt(sample_txt_file: Path) -> Path:
    """Alias for sample_txt_file from conftest.py."""
    return sample_txt_file


@pytest.fixture
def sample_md(sample_md_file: Path) -> Path:
    """Alias for sample_md_file from conftest.py."""
    return sample_md_file


class TestVersionCommand:
    """Tests for version command."""

    def test_version_flag(self, runner: CliRunner):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "markitai" in result.output

    def test_version_short_flag(self, runner: CliRunner):
        """Test -v flag."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "markitai" in result.output


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_list(self, runner: CliRunner):
        """Test config list command."""
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        # Should output JSON
        assert "output" in result.output
        assert "llm" in result.output

    def test_config_path(self, runner: CliRunner):
        """Test config path command."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert (
            "Configuration Sources" in result.output
            or "Currently using" in result.output
        )

    def test_config_validate(self, runner: CliRunner):
        """Test config validate command."""
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_get(self, runner: CliRunner):
        """Test config get command."""
        result = runner.invoke(app, ["config", "get", "llm.enabled"])
        # May exit 0 or 1 depending on whether key exists
        assert result.exit_code in (0, 1)

    def test_config_init(self, runner: CliRunner, tmp_path: Path):
        """Test init command (moved from config init to top-level init)."""
        output_file = tmp_path / "markitai.json"
        result = runner.invoke(app, ["init", "-y", "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify it's valid JSON
        content = json.loads(output_file.read_text())
        assert "output" in content


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_txt(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test converting a text file."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.txt.md").exists()

    def test_convert_md(self, runner: CliRunner, sample_md: Path, tmp_path: Path):
        """Test converting a markdown file (passthrough)."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_md),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.md.md").exists()

    def test_convert_dry_run(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test dry run mode."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Would convert" in result.output or "Dry Run" in result.output
        # File should not be created in dry run
        assert not (output_dir / "sample.txt.md").exists()

    def test_convert_verbose(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test verbose mode."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--verbose",
            ],
        )
        assert result.exit_code == 0

    def test_convert_unsupported_format(self, runner: CliRunner, tmp_path: Path):
        """Test converting unsupported file format."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(unsupported),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 1
        assert "Unsupported" in result.output or "Error" in result.output

    def test_convert_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test converting non-existent file."""
        result = runner.invoke(
            app,
            [
                str(tmp_path / "nonexistent.txt"),
                "-o",
                str(tmp_path / "output"),
            ],
        )
        assert result.exit_code != 0


class TestOutputFileTarget:
    """Tests for `-o <name>.md` targeting an output FILE (not a directory)."""

    def test_single_file_md_target_writes_file(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """`-o out.md` writes the markdown exactly to out.md."""
        target = tmp_path / "out.md"
        result = runner.invoke(app, [str(sample_txt), "-o", str(target)])

        assert result.exit_code == 0
        assert target.is_file(), "-o out.md must create a file, not a directory"
        content = target.read_text()
        assert "---" in content

    def test_single_file_md_target_nested_parent(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """`-o sub/out.md` writes to sub/out.md (parent is the output dir)."""
        target = tmp_path / "sub" / "out.md"
        result = runner.invoke(app, [str(sample_txt), "-o", str(target)])

        assert result.exit_code == 0
        assert target.is_file()
        # No directory named out.md anywhere
        assert not (tmp_path / "sub" / "out.md").is_dir()

    def test_existing_md_directory_keeps_directory_behavior(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """A pre-existing directory named out.md is still used as a directory."""
        md_dir = tmp_path / "out.md"
        md_dir.mkdir()
        result = runner.invoke(app, [str(sample_txt), "-o", str(md_dir)])

        assert result.exit_code == 0
        assert md_dir.is_dir()
        assert (md_dir / "sample.txt.md").is_file()

    def test_batch_md_target_is_usage_error(self, runner: CliRunner, tmp_path: Path):
        """Directory input with `-o out.md` must fail with a clear error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")

        result = runner.invoke(app, [str(input_dir), "-o", str(tmp_path / "out.md")])

        assert result.exit_code == 2
        # rich wraps the usage panel at console width (narrow on Windows CI):
        # normalize whitespace/borders before matching the phrase
        flat = " ".join(result.output.replace("\u2502", " ").split())
        assert "pass an output directory" in flat
        assert not (tmp_path / "out.md").exists()

    def test_url_list_md_target_is_usage_error(self, runner: CliRunner, tmp_path: Path):
        """URL list input with `-o out.md` must fail with a clear error."""
        urls_file = tmp_path / "links.urls"
        urls_file.write_text("https://example.com/\n")

        result = runner.invoke(app, [str(urls_file), "-o", str(tmp_path / "out.md")])

        assert result.exit_code == 2
        flat = " ".join(result.output.replace("\u2502", " ").split())
        assert "pass an output directory" in flat
        assert not (tmp_path / "out.md").exists()

    def test_url_list_without_output_reports_usage_on_stderr(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        urls_file = tmp_path / "links.urls"
        urls_file.write_text("https://example.com/\n")
        # Keep this usage test independent of the developer's user config,
        # which may legitimately define output.dir as a batch fallback.
        config_file = tmp_path / "empty-config.json"
        config_file.write_text("{}")

        result = runner.invoke(app, [str(urls_file), "-c", str(config_file)])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "requires -o/--output directory" in result.stderr


class TestStdoutPipePurity:
    """In stdout mode (no -o), diagnostics must go to stderr, not stdout."""

    def test_quiet_preserves_primary_markdown_payload(
        self, runner: CliRunner, sample_txt: Path
    ) -> None:
        """Quiet suppresses diagnostics, never the requested stdout payload."""
        result = runner.invoke(app, [str(sample_txt), "--quiet"])

        assert result.exit_code == 0
        assert result.stdout.lstrip().startswith("---")
        assert "This is test content." in result.stdout
        assert result.stderr == ""

    def test_quiet_single_url_file_mode_only_writes_artifact(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        fetch_result = MagicMock()
        fetch_result.content = "# Quiet URL\n"
        fetch_result.cache_hit = False
        fetch_result.strategy_used = "static"
        fetch_result.screenshot_path = None
        fetch_result.title = "Quiet URL"
        fetch_result.metadata = {}
        output_file = tmp_path / "result.md"

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            result = runner.invoke(
                app,
                [
                    "https://example.com/page",
                    "-o",
                    str(output_file),
                    "--quiet",
                    "--no-cache",
                ],
            )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert output_file.exists()

    def test_quiet_single_url_stdout_mode_preserves_markdown(
        self, runner: CliRunner
    ) -> None:
        fetch_result = MagicMock()
        fetch_result.content = "# Quiet URL\n\nPayload stays on stdout.\n"
        fetch_result.cache_hit = False
        fetch_result.strategy_used = "static"
        fetch_result.screenshot_path = None
        fetch_result.title = "Quiet URL"
        fetch_result.metadata = {}

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=fetch_result,
        ):
            result = runner.invoke(
                app,
                ["https://example.com/page", "--quiet", "--no-cache"],
            )

        assert result.exit_code == 0
        assert "# Quiet URL" in result.stdout
        assert "Payload stays on stdout." in result.stdout
        assert result.stderr == ""

    def test_missing_input_error_goes_to_stderr(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A failed conversion must not write diagnostics into stdout."""
        missing = tmp_path / "missing.pdf"

        result = runner.invoke(app, [str(missing)])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "does not exist" in result.stderr

    def test_image_without_ocr_or_llm_is_not_a_successful_empty_conversion(
        self, runner: CliRunner, fixtures_dir: Path
    ) -> None:
        """A supported image request without an extraction mode fails clearly."""
        result = runner.invoke(app, [str(fixtures_dir / "sample.jpg")])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Use --llm or --ocr" in result.stderr

    def test_quiet_unsupported_input_still_explains_failure(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        unsupported = tmp_path / "input.unsupported"
        unsupported.write_text("content")

        result = runner.invoke(app, [str(unsupported), "--quiet"])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Unsupported file format" in result.stderr

    def test_file_output_mode_failure_still_uses_stderr(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        unsupported = tmp_path / "input.unsupported"
        unsupported.write_text("content")

        result = runner.invoke(
            app,
            [str(unsupported), "-o", str(tmp_path / "output"), "--quiet"],
        )

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Unsupported file format" in result.stderr

    def test_quiet_image_failure_still_explains_required_feature(
        self, runner: CliRunner, fixtures_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [str(fixtures_dir / "sample.jpg"), "--quiet"],
        )

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Use --llm or --ocr" in result.stderr

    def test_quiet_runtime_failure_still_reaches_stderr(
        self, runner: CliRunner, sample_txt: Path
    ) -> None:
        with patch(
            "markitai.workflow.core.convert_document_core",
            side_effect=RuntimeError("conversion exploded"),
        ):
            result = runner.invoke(app, [str(sample_txt), "--quiet"])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "conversion exploded" in result.stderr

    def test_alt_warning_goes_to_stderr(self, runner: CliRunner, sample_txt: Path):
        """`--alt` without LLM warns on stderr; stdout stays pure markdown."""
        result = runner.invoke(app, [str(sample_txt), "--alt", "--no-llm"])

        assert result.exit_code == 0
        assert "Image analysis" not in result.stdout
        assert "Image analysis" in result.stderr
        # stdout is pure markdown content (starts with frontmatter)
        assert result.stdout.lstrip().startswith("---")

    def test_pipe_purity_subprocess(self, sample_txt: Path):
        """End-to-end: `markitai <file> --alt 2>/dev/null` yields pure markdown."""
        result = subprocess.run(
            ["uv", "run", "markitai", str(sample_txt), "--alt", "--no-llm"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Image analysis" not in result.stdout
        assert result.stdout.lstrip().startswith("---")
        # The diagnostic still reaches the user via stderr
        assert "Image analysis" in result.stderr

    def test_long_lines_not_hard_wrapped(self, runner: CliRunner, tmp_path: Path):
        """stdout content bypasses Rich: long URLs must never be wrapped
        at terminal width (Rich's default 80 cols would break them)."""
        long_url = "https://example.com/" + "a" * 300
        source = tmp_path / "long.md"
        source.write_text(f"See [link]({long_url}) for details.\n")

        result = runner.invoke(app, [str(source)])

        assert result.exit_code == 0
        assert long_url in result.stdout


class TestUrlBatchExitContract:
    """The public CLI distinguishes partial output from complete success."""

    def test_url_list_forwards_quiet_to_batch_processor(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        urls_file = tmp_path / "links.urls"
        urls_file.write_text("https://example.com/page\n")
        output_dir = tmp_path / "output"

        with patch(
            "markitai.cli.processors.url.process_url_batch",
            new_callable=AsyncMock,
        ) as process_batch_mock:
            result = runner.invoke(
                app,
                [str(urls_file), "-o", str(output_dir), "--quiet"],
            )

        assert result.exit_code == 0
        assert process_batch_mock.await_args is not None
        assert process_batch_mock.await_args.kwargs["quiet"] is True

    def test_url_list_partial_failure_exits_ten(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from markitai.fetch import FetchError

        urls_file = tmp_path / "links.urls"
        urls_file.write_text("https://example.com/success\nhttps://example.com/fail\n")
        output_dir = tmp_path / "output"

        async def fake_fetch(url: str, *args: object, **kwargs: object) -> MagicMock:
            if url.endswith("/fail"):
                raise FetchError("Connection refused")
            result = MagicMock()
            result.content = "# Success\n"
            result.cache_hit = False
            result.strategy_used = "static"
            result.screenshot_path = None
            result.title = "Success"
            result.metadata = {}
            return result

        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch):
            result = runner.invoke(
                app,
                [str(urls_file), "-o", str(output_dir), "--no-cache"],
            )

        assert result.exit_code == 10
        assert (output_dir / "success.md").exists()
        assert not (output_dir / "fail.md").exists()


class TestRemoteFetchHardOptOut:
    """The environment opt-out wins over explicit remote CLI flags."""

    def test_explicit_remote_strategy_is_blocked(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")

        result = runner.invoke(
            app,
            [
                "https://example.com/article",
                "-s",
                "jina",
                "--config-json",
                '{"cache":{"enabled":false}}',
            ],
        )

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "MARKITAI_NO_REMOTE_FETCH" in result.stderr

    def test_config_remote_strategy_still_respects_consent_never(
        self, runner: CliRunner
    ) -> None:
        with patch(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            new_callable=AsyncMock,
        ) as remote_fetch:
            result = runner.invoke(
                app,
                [
                    "https://example.com/article",
                    "--config-json",
                    (
                        '{"fetch":{"strategy":"jina",'
                        '"remote_consent":"never"},'
                        '"cache":{"enabled":false}}'
                    ),
                ],
            )

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "remote extraction is not allowed" in result.stderr
        remote_fetch.assert_not_awaited()

    def test_url_output_mode_failure_still_uses_stderr(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")

        result = runner.invoke(
            app,
            [
                "https://example.com/article",
                "-s",
                "jina",
                "-o",
                str(tmp_path / "output"),
                "--quiet",
                "--config-json",
                '{"cache":{"enabled":false}}',
            ],
        )

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "MARKITAI_NO_REMOTE_FETCH" in result.stderr


class TestBatchConvert:
    """Tests for batch conversion."""

    def test_quiet_batch_writes_outputs_without_informational_ui(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file.txt").write_text("Content")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir), "--quiet"],
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert (output_dir / "file.txt.md").exists()

    def test_quiet_batch_failure_is_reported_on_stderr(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "broken.txt").write_text("Content")
        output_dir = tmp_path / "output"

        with patch(
            "markitai.workflow.core.convert_document_core",
            new_callable=AsyncMock,
            side_effect=RuntimeError("conversion exploded"),
        ):
            result = runner.invoke(
                app,
                [str(input_dir), "-o", str(output_dir), "--quiet"],
            )

        assert result.exit_code == 10
        assert result.stdout == ""
        assert "broken.txt" in result.stderr
        assert "conversion exploded" in result.stderr

    def test_quiet_directory_url_failure_redacts_signed_url_on_stderr(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from markitai.fetch import FetchError

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "links.urls").write_text(
            "https://alice:password@example.com/private/report"
            "?token=query-secret#access_token=fragment-secret\n"
        )
        output_dir = tmp_path / "output"

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                side_effect=FetchError("Connection refused"),
            ),
            patch(
                "markitai.fetch._get_playwright_renderer",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=False,
            ),
        ):
            result = runner.invoke(
                app,
                [
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--quiet",
                    "--no-cache",
                ],
            )

        assert result.exit_code == 10
        assert result.stdout == ""
        assert "https://example.com/private/report" in result.stderr
        assert "alice" not in result.stderr
        assert "password" not in result.stderr
        assert "token" not in result.stderr
        assert "query-secret" not in result.stderr
        assert "fragment-secret" not in result.stderr
        assert "Playwright is not installed" not in result.stderr

    def test_quiet_batch_dry_run_suppresses_preview(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file.txt").write_text("Content")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir), "--dry-run", "--quiet"],
        )

        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert not output_dir.exists()

    def test_batch_dry_run(self, runner: CliRunner, tmp_path: Path):
        """Test batch mode dry run."""
        # Create input directory with files
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")
        (input_dir / "file2.txt").write_text("Content 2")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "2 files" in result.output or "file1.txt" in result.output

    def test_batch_convert(self, runner: CliRunner, tmp_path: Path):
        """Test batch conversion."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")
        (input_dir / "file2.txt").write_text("Content 2")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "file1.txt.md").exists()
        assert (output_dir / "file2.txt.md").exists()

    def test_batch_empty_directory(self, runner: CliRunner, tmp_path: Path):
        """Test batch mode with empty directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert "No supported files" in result.output


class TestCLIWithSubprocess:
    """Tests using subprocess for more realistic CLI testing."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "rich-click renders --help through a rich Console whose writes to a "
            "redirected pipe are unreliable on Windows CI (empty capture, rc 0). "
            "Help content is covered cross-platform by the CliRunner tests in "
            "test_cli_main.py; subprocess entry-point launch is covered by "
            "test_version_command, which passes on Windows."
        ),
    )
    def test_help_command(self):
        """Test help command via subprocess (non-Windows)."""
        result = subprocess.run(
            [sys.executable, "-m", "markitai", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        combined = (result.stdout or "") + (result.stderr or "")
        assert "Opinionated Markdown converter" in combined

    def test_version_command(self):
        """Test version command via subprocess."""
        result = subprocess.run(
            ["uv", "run", "markitai", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "markitai" in result.stdout


class TestCaseSensitivityWarning:
    """Tests for case-sensitivity warning in --no-cache-for patterns."""

    def test_warn_case_sensitivity_mismatches_detects_mismatch(self, tmp_path: Path):
        """Test that case mismatches are detected."""
        from markitai.cli import _warn_case_sensitivity_mismatches

        # Create files with uppercase extensions
        (tmp_path / "image.JPG").touch()
        (tmp_path / "doc.PDF").touch()
        (tmp_path / "normal.txt").touch()

        files = list(tmp_path.glob("*"))
        patterns = ["*.jpg", "*.pdf"]  # lowercase patterns

        # Should not raise, just log warnings
        # We can't easily test the console output, but we can verify no exceptions
        _warn_case_sensitivity_mismatches(files, tmp_path, patterns)

    def test_warn_case_sensitivity_no_mismatch(self, tmp_path: Path):
        """Test that matching cases don't trigger warnings."""
        from markitai.cli import _warn_case_sensitivity_mismatches

        # Create files with matching case
        (tmp_path / "image.jpg").touch()
        (tmp_path / "doc.pdf").touch()

        files = list(tmp_path.glob("*"))
        patterns = ["*.jpg", "*.pdf"]

        # Should not trigger warnings
        _warn_case_sensitivity_mismatches(files, tmp_path, patterns)

    def test_warn_case_sensitivity_with_glob_patterns(self, tmp_path: Path):
        """Test case sensitivity with ** patterns."""
        from markitai.cli import _warn_case_sensitivity_mismatches

        # Create nested files with uppercase extensions
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "image.JPG").touch()

        files = list(tmp_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        patterns = ["**/*.jpg"]

        # Should detect mismatch
        _warn_case_sensitivity_mismatches(files, tmp_path, patterns)

    def test_warn_case_sensitivity_empty_patterns(self, tmp_path: Path):
        """Test with empty patterns list."""
        from markitai.cli import _warn_case_sensitivity_mismatches

        (tmp_path / "image.JPG").touch()
        files = list(tmp_path.glob("*"))

        # Should not raise with empty patterns
        _warn_case_sensitivity_mismatches(files, tmp_path, [])


class TestWorkflowCoreV2:
    """Tests for workflow/core pipeline.

    These tests verify the core conversion pipeline produces correct output.
    """

    def test_convert_txt(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test converting a text file (uses workflow/core by default)."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.txt.md").exists()

        # Verify content has frontmatter
        content = (output_dir / "sample.txt.md").read_text()
        assert "---" in content
        assert "title:" in content or "source:" in content

    def test_convert_dry_run(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test dry run mode."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Would convert" in result.output or "Dry Run" in result.output
        assert not (output_dir / "sample.txt.md").exists()

    def test_unsupported_format(self, runner: CliRunner, tmp_path: Path):
        """Test with unsupported file format."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(unsupported),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 1
        assert "Unsupported" in result.output or "Error" in result.output

    def test_skip_existing(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test skip mode when output exists."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create existing output file
        existing = output_dir / "sample.txt.md"
        existing.write_text("existing content")

        # Create config with skip mode
        config_path = tmp_path / "config.json"
        config_path.write_text('{"output": {"on_conflict": "skip"}}')

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "-c",
                str(config_path),
            ],
        )
        assert result.exit_code == 0
        assert "Skipped" in result.output or "exists" in result.output.lower()
        # Content should not change
        assert existing.read_text() == "existing content"

    def test_no_report_by_default(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test single-file conversion does not generate a report by default."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0

        # No report for single-file conversions unless output.report=true
        reports_dir = output_dir / ".markitai" / "reports"
        assert not reports_dir.exists()

    def test_generates_report_when_enabled(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test generates report file when output.report=true."""
        output_dir = tmp_path / "output"
        config_path = tmp_path / "config.json"
        config_path.write_text('{"output": {"report": true}}')

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "-c",
                str(config_path),
            ],
        )
        assert result.exit_code == 0

        # Check report was generated
        reports_dir = output_dir / ".markitai" / "reports"
        assert reports_dir.exists()

        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1

        # Verify report structure
        report = json.loads(report_files[0].read_text())
        assert "version" in report
        assert "summary" in report
        assert report["summary"]["total_documents"] == 1
        assert report["summary"]["completed_documents"] == 1

    def test_verbose_mode(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test verbose mode produces more output."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        # Verbose mode should have log messages
        assert "Converting" in result.output or "Written" in result.output

    def test_batch_convert(self, runner: CliRunner, tmp_path: Path):
        """Test batch conversion."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")
        (input_dir / "file2.txt").write_text("Content 2")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "file1.txt.md").exists()
        assert (output_dir / "file2.txt.md").exists()

    def test_batch_preserves_subdirs(self, runner: CliRunner, tmp_path: Path):
        """Test batch preserves subdirectory structure."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "root.txt").write_text("Root content")

        subdir = input_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "root.txt.md").exists()
        assert (output_dir / "subdir" / "nested.txt.md").exists()

    def test_batch_report_opt_out(self, runner: CliRunner, tmp_path: Path):
        """output.report=false disables batch report generation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")

        output_dir = tmp_path / "output"
        config_path = tmp_path / "config.json"
        config_path.write_text('{"output": {"report": false}}')

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
                "-c",
                str(config_path),
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "file1.txt.md").exists()
        reports_dir = output_dir / ".markitai" / "reports"
        report_files = list(reports_dir.glob("*.json")) if reports_dir.exists() else []
        assert report_files == []

    def test_batch_same_stem_inputs_stay_distinct(
        self, runner: CliRunner, tmp_path: Path
    ):
        """Same-stem inputs (a.txt + a.html) map to distinct output names."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("Text content")
        (input_dir / "a.html").write_text("<h1>HTML content</h1>")
        (input_dir / "b.txt").write_text("Other content")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        # Append naming keeps the source extension in the output name
        assert (output_dir / "a.txt.md").exists()
        assert (output_dir / "a.html.md").exists()
        assert not (output_dir / "a.md").exists()
        assert (output_dir / "b.txt.md").exists()

    def test_md_input_into_own_dir_keeps_source(
        self, runner: CliRunner, tmp_path: Path
    ):
        """Converting note.md into its own directory must not overwrite it."""
        note = tmp_path / "note.md"
        note.write_text("# Original note")

        result = runner.invoke(
            app,
            [
                str(note),
                "-o",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        # Source untouched; append naming yields note.md.md
        assert note.read_text() == "# Original note"
        assert (tmp_path / "note.md.md").exists()

    def test_batch_generates_report(self, runner: CliRunner, tmp_path: Path):
        """Test batch generates report file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0

        # Check report was generated
        reports_dir = output_dir / ".markitai" / "reports"
        assert reports_dir.exists()

        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1

        # Verify report structure
        report = json.loads(report_files[0].read_text())
        assert "version" in report
        assert "summary" in report
        assert report["summary"]["completed_documents"] >= 1


class TestBatchResume:
    """Tests for --resume skipping completed batch work.

    Regression: the CLI batch entry point accepted the resume flag but never
    used it — every --resume run re-initialized state and re-converted all
    files (piling up .v2.md rename copies under the default on_conflict).
    """

    def test_quiet_resume_suppresses_resume_and_summary_ui(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("Content A")
        output_dir = tmp_path / "output"

        first = runner.invoke(app, [str(input_dir), "-o", str(output_dir)])
        assert first.exit_code == 0

        resumed = runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir), "--resume", "--quiet"],
        )

        assert resumed.exit_code == 0
        assert resumed.stdout == ""
        assert resumed.stderr == ""

    def test_resume_skips_completed_files(self, runner: CliRunner, tmp_path: Path):
        """A fully completed batch re-run with --resume does no work."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("Content A")
        (input_dir / "b.txt").write_text("Content B")
        output_dir = tmp_path / "output"

        first = runner.invoke(app, [str(input_dir), "-o", str(output_dir)])
        assert first.exit_code == 0
        a_md = output_dir / "a.txt.md"
        b_md = output_dir / "b.txt.md"
        assert a_md.exists() and b_md.exists()
        a_mtime = a_md.stat().st_mtime_ns
        b_mtime = b_md.stat().st_mtime_ns

        second = runner.invoke(app, [str(input_dir), "-o", str(output_dir), "--resume"])
        assert second.exit_code == 0
        # Completed files must not be re-converted: no rename-versioned
        # copies, and the original outputs are untouched
        assert not (output_dir / "a.txt.v2.md").exists()
        assert not (output_dir / "b.txt.v2.md").exists()
        assert a_md.stat().st_mtime_ns == a_mtime
        assert b_md.stat().st_mtime_ns == b_mtime

    def test_resume_processes_new_and_failed_files(
        self, runner: CliRunner, tmp_path: Path
    ):
        """--resume retries failed entries and picks up new files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("Content A")
        (input_dir / "b.txt").write_text("Content B")
        output_dir = tmp_path / "output"

        first = runner.invoke(app, [str(input_dir), "-o", str(output_dir)])
        assert first.exit_code == 0

        # Simulate a failed entry: mark b.txt failed and remove its output
        states_dir = output_dir / ".markitai" / "states"
        state_file = next(states_dir.glob("markitai.*.state.json"))
        state = json.loads(state_file.read_text())
        state["documents"]["b.txt"]["status"] = "failed"
        state_file.write_text(json.dumps(state))
        (output_dir / "b.txt.md").unlink()

        # New file added after the first run
        (input_dir / "c.txt").write_text("Content C")

        second = runner.invoke(app, [str(input_dir), "-o", str(output_dir), "--resume"])
        assert second.exit_code == 0
        # Failed entry retried, new file converted, completed file untouched
        assert (output_dir / "b.txt.md").exists()
        assert (output_dir / "c.txt.md").exists()
        assert not (output_dir / "a.txt.v2.md").exists()

    @staticmethod
    def _fetch_stub() -> MagicMock:
        """Mock FetchResult with the fields the batch URL processor reads."""
        result = MagicMock()
        result.content = "# Test Page\n\nSome content.\n"
        result.cache_hit = False
        result.strategy_used = "static"
        result.screenshot_path = None
        result.title = "Test Page"
        result.static_content = None
        result.browser_content = None
        result.metadata = {}
        return result

    def test_resume_skips_completed_urls(self, runner: CliRunner, tmp_path: Path):
        """URLs already COMPLETED in the state are not re-fetched on --resume."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "links.urls").write_text("https://example.com/alpha\n")
        output_dir = tmp_path / "output"

        async def fake_fetch(url, *args, **kwargs):
            return self._fetch_stub()

        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch) as first_fetch:
            first = runner.invoke(
                app, [str(input_dir), "-o", str(output_dir), "--no-cache"]
            )
        assert first.exit_code == 0
        assert first_fetch.call_count == 1
        assert (output_dir / "alpha.md").exists()

        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch) as second_fetch:
            second = runner.invoke(
                app,
                [str(input_dir), "-o", str(output_dir), "--no-cache", "--resume"],
            )
        assert second.exit_code == 0
        # Completed URL skipped entirely: no fetch, no rename-versioned copy
        assert second_fetch.call_count == 0
        assert not (output_dir / "alpha.v2.md").exists()

    def test_resume_fetches_only_new_urls(self, runner: CliRunner, tmp_path: Path):
        """--resume picks up URLs added to the .urls file since the last run."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        urls_file = input_dir / "links.urls"
        urls_file.write_text("https://example.com/alpha\n")
        output_dir = tmp_path / "output"

        fetched: list[str] = []

        async def fake_fetch(url, *args, **kwargs):
            fetched.append(url)
            return self._fetch_stub()

        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch):
            first = runner.invoke(
                app, [str(input_dir), "-o", str(output_dir), "--no-cache"]
            )
        assert first.exit_code == 0
        assert fetched == ["https://example.com/alpha"]

        # New URL appended after the first run
        urls_file.write_text("https://example.com/alpha\nhttps://example.com/beta\n")

        fetched.clear()
        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch):
            second = runner.invoke(
                app,
                [str(input_dir), "-o", str(output_dir), "--no-cache", "--resume"],
            )
        assert second.exit_code == 0
        # Only the new URL is fetched; existing output untouched
        assert fetched == ["https://example.com/beta"]
        assert (output_dir / "beta.md").exists()
        assert not (output_dir / "alpha.v2.md").exists()

    def test_resume_retries_failed_urls(self, runner: CliRunner, tmp_path: Path):
        """--resume re-fetches failed URLs but not completed ones."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "links.urls").write_text(
            "https://example.com/alpha\nhttps://example.com/beta\n"
        )
        output_dir = tmp_path / "output"

        fetched: list[str] = []

        async def fake_fetch(url, *args, **kwargs):
            fetched.append(url)
            return self._fetch_stub()

        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch):
            first = runner.invoke(
                app, [str(input_dir), "-o", str(output_dir), "--no-cache"]
            )
        assert first.exit_code == 0
        assert sorted(fetched) == [
            "https://example.com/alpha",
            "https://example.com/beta",
        ]

        # Simulate a failed URL: mark alpha failed and remove its output
        states_dir = output_dir / ".markitai" / "states"
        state_file = next(states_dir.glob("markitai.*.state.json"))
        state = json.loads(state_file.read_text())
        state["urls"]["https://example.com/alpha"]["status"] = "failed"
        state_file.write_text(json.dumps(state))
        (output_dir / "alpha.md").unlink()

        fetched.clear()
        with patch("markitai.fetch.fetch_url", side_effect=fake_fetch):
            second = runner.invoke(
                app,
                [str(input_dir), "-o", str(output_dir), "--no-cache", "--resume"],
            )
        assert second.exit_code == 0
        # Only the failed URL is re-fetched; the completed one is skipped
        assert fetched == ["https://example.com/alpha"]
        assert (output_dir / "alpha.md").exists()
        assert not (output_dir / "beta.v2.md").exists()


class TestConfigOutputDir:
    """Tests for using output.dir from config file as default output directory.

    Note: Single file mode always outputs to stdout (no -o = stdout).
    Config output.dir is only used as fallback for batch/URL modes.
    """

    def test_single_file_outputs_to_stdout_ignoring_config(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test that single file mode outputs to stdout, ignoring config output.dir."""
        # Setup: create project config with output.dir
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        output_dir = project_dir / "my_output"

        config_file = project_dir / "markitai.json"
        config_file.write_text(json.dumps({"output": {"dir": str(output_dir)}}))

        # Copy sample file to project dir
        input_file = project_dir / "test.txt"
        input_file.write_text(sample_txt.read_text())

        # Run from project directory (where markitai.json exists)

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            result = runner.invoke(app, [str(input_file)])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        # Single file mode should output to stdout, NOT create file
        assert not output_dir.exists(), (
            "Config output.dir should NOT be used for single file mode"
        )
        # Verify content is in stdout
        assert "# test" in result.output or "test" in result.output.lower()

    def test_single_file_with_explicit_output_saves_to_file(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test that single file mode with -o saves to file."""
        output_dir = tmp_path / "output"
        input_file = tmp_path / "test.txt"
        input_file.write_text(sample_txt.read_text())

        result = runner.invoke(app, [str(input_file), "-o", str(output_dir)])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "test.txt.md").exists()

    def test_cli_output_overrides_config(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test that -o CLI flag overrides config output.dir."""
        # Setup: create config with output.dir
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        config_output = project_dir / "config_output"
        cli_output = project_dir / "cli_output"

        config_file = project_dir / "markitai.json"
        config_file.write_text(json.dumps({"output": {"dir": str(config_output)}}))

        input_file = project_dir / "test.txt"
        input_file.write_text(sample_txt.read_text())

        # Run with explicit -o flag

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            result = runner.invoke(app, [str(input_file), "-o", str(cli_output)])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert cli_output.exists(), "CLI -o should be used"
        assert (cli_output / "test.txt.md").exists(), "Output should be in CLI -o dir"
        assert not config_output.exists(), (
            "Config output.dir should NOT be used when -o specified"
        )

    def test_explicit_config_single_file_outputs_to_stdout(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test that single file with -c config still outputs to stdout (not config's output.dir)."""
        # Setup: create explicit config file with output.dir
        config_file = tmp_path / "custom_config.json"
        output_dir = tmp_path / "custom_output"
        config_file.write_text(json.dumps({"output": {"dir": str(output_dir)}}))

        input_file = tmp_path / "test.txt"
        input_file.write_text(sample_txt.read_text())

        # Run with explicit -c flag but no -o
        result = runner.invoke(app, [str(input_file), "-c", str(config_file)])

        assert result.exit_code == 0
        # Single file mode should output to stdout, NOT use config's output.dir
        assert not output_dir.exists(), (
            "Config output.dir should NOT be used for single file mode"
        )
        # Verify content is in stdout
        assert "# test" in result.output or "test" in result.output.lower()

    def test_batch_mode_uses_config_output_dir(self, runner: CliRunner, tmp_path: Path):
        """Test that batch mode (directory input) uses config output.dir."""
        # Setup: create project with config and input files
        project_dir = tmp_path / "project"
        input_dir = project_dir / "input"
        input_dir.mkdir(parents=True)
        output_dir = project_dir / "batch_output"

        config_file = project_dir / "markitai.json"
        config_file.write_text(json.dumps({"output": {"dir": str(output_dir)}}))

        # Create test files
        (input_dir / "file1.txt").write_text("Content 1")
        (input_dir / "file2.txt").write_text("Content 2")

        # Run batch mode from project directory

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            result = runner.invoke(app, [str(input_dir)])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert output_dir.exists(), "Batch mode should use config output.dir"
        assert (output_dir / "file1.txt.md").exists(), "Batch output file1 should exist"
        assert (output_dir / "file2.txt.md").exists(), "Batch output file2 should exist"

"""Integration tests for CLI commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
        assert "Configuration file" in result.output

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
        """Test config init command."""
        output_file = tmp_path / "test_config.json"
        result = runner.invoke(app, ["config", "init", "-o", str(output_file)])
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


class TestBatchConvert:
    """Tests for batch conversion."""

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

    def test_help_command(self):
        """Test help command via subprocess."""
        result = subprocess.run(
            ["uv", "run", "markitai", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Document to Markdown converter" in result.stdout

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

    def test_generates_report(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test generates report file."""
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

        # Check report was generated
        reports_dir = output_dir / "reports"
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
        reports_dir = output_dir / "reports"
        assert reports_dir.exists()

        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1

        # Verify report structure
        report = json.loads(report_files[0].read_text())
        assert "version" in report
        assert "summary" in report
        assert report["summary"]["completed_documents"] >= 1

"""Integration tests for CLI commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from markit.cli import app


@pytest.fixture
def runner():
    """Return a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("# Test Document\n\nThis is test content.", encoding="utf-8")
    return txt_file


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Test Document\n\nThis is test content.\n\n## Section 1\n\nSome text here.\n",
        encoding="utf-8",
    )
    return md_file


class TestVersionCommand:
    """Tests for version command."""

    def test_version_flag(self, runner: CliRunner):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "markit" in result.output

    def test_version_short_flag(self, runner: CliRunner):
        """Test -v flag."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "markit" in result.output


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
            ["uv", "run", "markit", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Document to Markdown converter" in result.stdout

    def test_version_command(self):
        """Test version command via subprocess."""
        result = subprocess.run(
            ["uv", "run", "markit", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "markit" in result.stdout

"""Full CLI integration tests for markitai.

This module provides comprehensive tests for all CLI commands and options.
Tests are organized by feature area and use CliRunner for fast execution.

Note: Real file conversion tests are in test_real_scenarios.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from markitai.cli import app

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
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
        "# Test Document\n\nThis is test content.\n\n## Section 1\n\nSome text.\n",
        encoding="utf-8",
    )
    return md_file


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_file = tmp_path / "markitai.json"
    config_file.write_text(
        json.dumps(
            {
                "output": {"dir": "./output", "on_conflict": "overwrite"},
                "llm": {"enabled": False},
                "image": {"compress": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_file


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple files for batch testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file1.txt").write_text("Content 1", encoding="utf-8")
    (input_dir / "file2.txt").write_text("Content 2", encoding="utf-8")
    (input_dir / "file3.md").write_text("# Markdown\n\nContent 3", encoding="utf-8")

    # Create subdirectory
    sub_dir = input_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "nested.txt").write_text("Nested content", encoding="utf-8")

    return input_dir


# =============================================================================
# Test: Help and Version Commands
# =============================================================================


class TestHelpAndVersion:
    """Tests for help and version commands."""

    def test_help_short_flag(self, runner: CliRunner):
        """Test -h displays help message."""
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        assert "Markitai" in result.output
        assert "Opinionated Markdown converter" in result.output

    def test_help_long_flag(self, runner: CliRunner):
        """Test --help displays help message."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Markitai" in result.output

    def test_help_shows_all_options(self, runner: CliRunner):
        """Test help shows all main options."""
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        # Check for key options
        options = [
            "--output",
            "--config",
            "--preset",
            "--llm",
            "--alt",
            "--desc",
            "--ocr",
            "--screenshot",
            "--resume",
            "--no-compress",
            "--no-cache",
            "--verbose",
            "--quiet",
            "--dry-run",
            "--version",
        ]
        for opt in options:
            assert opt in result.output, f"Option {opt} not found in help"

    def test_help_shows_all_subcommands(self, runner: CliRunner):
        """Test help shows all subcommands."""
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        subcommands = ["cache", "config", "check-deps"]
        for cmd in subcommands:
            assert cmd in result.output, f"Subcommand {cmd} not found in help"

    def test_help_shows_presets(self, runner: CliRunner):
        """Test help shows preset descriptions."""
        result = runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        assert "rich" in result.output
        assert "standard" in result.output
        assert "minimal" in result.output

    def test_version_short_flag(self, runner: CliRunner):
        """Test -v displays version."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "markitai" in result.output.lower()

    def test_version_long_flag(self, runner: CliRunner):
        """Test --version displays version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "markitai" in result.output.lower()

    def test_no_input_shows_help(self, runner: CliRunner):
        """Test running without input shows help."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output


# =============================================================================
# Test: Preset Configuration
# =============================================================================


class TestPresetConfiguration:
    """Tests for --preset option."""

    def test_preset_rich(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --preset rich enables all enhancement features."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--preset", "rich", "--dry-run"],
        )
        assert result.exit_code == 0

    def test_preset_standard(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --preset standard enables standard enhancement."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--preset",
                "standard",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_preset_minimal(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --preset minimal disables enhancement."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--preset", "minimal"],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.txt.md").exists()

    def test_preset_short_flag(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test -p short flag for preset."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-p", "minimal"],
        )
        assert result.exit_code == 0

    def test_preset_override_with_cli_flag(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test preset can be overridden by explicit CLI flags."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(output_dir),
                "--preset",
                "rich",
                "--no-desc",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_preset_case_insensitive(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test preset names are case insensitive."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--preset", "MINIMAL"],
        )
        assert result.exit_code == 0

    def test_invalid_preset(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test invalid preset value shows error."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--preset", "invalid"],
        )
        assert result.exit_code != 0


# =============================================================================
# Test: LLM Options
# =============================================================================


class TestLLMOptions:
    """Tests for LLM-related CLI options."""

    def test_llm_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --llm enables LLM processing."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--llm", "--dry-run"],
        )
        assert result.exit_code in (0, 1)

    def test_no_llm_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --no-llm disables LLM processing."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-llm"],
        )
        assert result.exit_code == 0

    def test_alt_and_desc_flags(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --alt and --desc flags."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-alt", "--no-desc"],
        )
        assert result.exit_code == 0

    def test_llm_concurrency(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --llm-concurrency sets concurrent LLM requests."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--llm-concurrency", "5"],
        )
        assert result.exit_code == 0

    def test_llm_concurrency_min_value(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --llm-concurrency rejects values < 1."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--llm-concurrency", "0"],
        )
        assert result.exit_code != 0


# =============================================================================
# Test: OCR and Screenshot Options
# =============================================================================


class TestOCRAndScreenshotOptions:
    """Tests for OCR and screenshot CLI options."""

    def test_ocr_flags(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --ocr and --no-ocr flags."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-ocr"],
        )
        assert result.exit_code == 0

    def test_screenshot_flags(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --screenshot and --no-screenshot flags."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-screenshot"],
        )
        assert result.exit_code == 0

    def test_no_compress_flag(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --no-compress disables image compression."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-compress"],
        )
        assert result.exit_code == 0


# =============================================================================
# Test: Cache Options
# =============================================================================


class TestCacheOptions:
    """Tests for cache-related CLI options."""

    def test_no_cache_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --no-cache disables caching."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-cache"],
        )
        assert result.exit_code == 0

    def test_no_cache_for_patterns(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --no-cache-for with patterns."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--no-cache-for", "*.txt,*.pdf"],
        )
        assert result.exit_code == 0


# =============================================================================
# Test: Concurrency Options
# =============================================================================


class TestConcurrencyOptions:
    """Tests for concurrency control options."""

    def test_batch_concurrency(
        self, runner: CliRunner, batch_dir: Path, tmp_path: Path
    ):
        """Test -j/--batch-concurrency sets batch task concurrency."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(batch_dir), "-o", str(output_dir), "-j", "2"],
        )
        assert result.exit_code == 0

    def test_url_concurrency(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --url-concurrency sets URL fetch concurrency."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--url-concurrency", "5"],
        )
        assert result.exit_code == 0

    def test_concurrency_min_value(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test concurrency rejects values < 1."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-j", "0"],
        )
        assert result.exit_code != 0


# =============================================================================
# Test: URL Fetch Strategy Options
# =============================================================================


class TestURLFetchStrategyOptions:
    """Tests for URL fetch strategy options."""

    def test_playwright_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --playwright flag is accepted."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--playwright"],
        )
        assert result.exit_code == 0

    def test_jina_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --jina flag is accepted."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--jina"],
        )
        assert result.exit_code == 0

    def test_mutually_exclusive_fetch_strategies(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test --playwright and --jina are mutually exclusive."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--playwright", "--jina"],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# =============================================================================
# Test: Output Control Options
# =============================================================================


class TestOutputControlOptions:
    """Tests for output control options."""

    def test_output_directory(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test -o/--output specifies output directory."""
        output_dir = tmp_path / "custom_output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.txt.md").exists()

    def test_output_directory_created(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_verbose_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --verbose enables verbose output."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--verbose"],
        )
        assert result.exit_code == 0

    def test_quiet_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test -q/--quiet suppresses output."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-q"],
        )
        assert result.exit_code == 0

    def test_dry_run_flag(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test --dry-run previews without writing."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "--dry-run"],
        )
        assert result.exit_code == 0
        assert not (output_dir / "sample.txt.md").exists()


# =============================================================================
# Test: Config Subcommand
# =============================================================================


class TestConfigSubcommand:
    """Tests for config subcommand."""

    def test_config_help(self, runner: CliRunner):
        """Test config --help."""
        result = runner.invoke(app, ["config", "-h"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "path" in result.output
        assert "init" in result.output

    def test_config_list_formats(self, runner: CliRunner):
        """Test config list with different formats."""
        # JSON (default)
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "output" in result.output

        # Table
        result = runner.invoke(app, ["config", "list", "-f", "table"])
        assert result.exit_code == 0

    def test_config_path(self, runner: CliRunner):
        """Test config path shows file paths."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0

    def test_config_init(self, runner: CliRunner, tmp_path: Path):
        """Test config init creates config file."""
        output_file = tmp_path / "test_config.json"
        result = runner.invoke(app, ["config", "init", "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert "output" in content or "llm" in content

    def test_config_validate(self, runner: CliRunner, sample_config: Path):
        """Test config validate with valid config."""
        result = runner.invoke(app, ["config", "validate", str(sample_config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_get(self, runner: CliRunner):
        """Test config get commands."""
        result = runner.invoke(app, ["config", "get", "llm.enabled"])
        assert result.exit_code in (0, 1)

        result = runner.invoke(app, ["config", "get", "nonexistent.key"])
        assert result.exit_code == 1


# =============================================================================
# Test: Cache Subcommand
# =============================================================================


class TestCacheSubcommand:
    """Tests for cache subcommand."""

    def test_cache_help(self, runner: CliRunner):
        """Test cache --help."""
        result = runner.invoke(app, ["cache", "-h"])
        assert result.exit_code == 0
        assert "stats" in result.output
        assert "clear" in result.output
        assert "spa-domains" in result.output

    def test_cache_stats(self, runner: CliRunner):
        """Test cache stats shows statistics."""
        result = runner.invoke(app, ["cache", "stats"])
        assert result.exit_code == 0

    def test_cache_stats_json(self, runner: CliRunner):
        """Test cache stats --json output."""
        result = runner.invoke(app, ["cache", "stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "enabled" in data or "cache" in data

    def test_cache_clear_yes(self, runner: CliRunner):
        """Test cache clear -y skips confirmation."""
        result = runner.invoke(app, ["cache", "clear", "-y"])
        assert result.exit_code == 0

    def test_cache_spa_domains(self, runner: CliRunner):
        """Test cache spa-domains commands."""
        result = runner.invoke(app, ["cache", "spa-domains"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["cache", "spa-domains", "--json"])
        assert result.exit_code == 0
        json.loads(result.output)  # Should be valid JSON


# =============================================================================
# Test: check-deps Subcommand
# =============================================================================


class TestCheckDepsSubcommand:
    """Tests for check-deps subcommand."""

    def test_check_deps_table_output(self, runner: CliRunner):
        """Test check-deps shows table output."""
        result = runner.invoke(app, ["check-deps"])
        assert result.exit_code == 0

    def test_check_deps_json_output(self, runner: CliRunner):
        """Test check-deps --json output."""
        result = runner.invoke(app, ["check-deps", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert any(
            key in data for key in ["playwright", "libreoffice", "rapidocr", "llm-api"]
        )


# =============================================================================
# Test: Single File Conversion (Fast - txt/md only)
# =============================================================================


class TestSingleFileConversion:
    """Tests for single file conversion with fast file types."""

    def test_convert_txt(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test converting text file."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.txt.md").exists()

        content = (output_dir / "sample.txt.md").read_text()
        assert "---" in content  # Has frontmatter

    def test_convert_md_passthrough(
        self, runner: CliRunner, sample_md: Path, tmp_path: Path
    ):
        """Test markdown file conversion (passthrough)."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_md), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert (output_dir / "sample.md.md").exists()

    def test_convert_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test converting non-existent file shows error."""
        result = runner.invoke(
            app,
            [str(tmp_path / "nonexistent.txt"), "-o", str(tmp_path / "output")],
        )
        assert result.exit_code != 0

    def test_convert_unsupported_format(self, runner: CliRunner, tmp_path: Path):
        """Test converting unsupported format shows error."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test content")
        result = runner.invoke(
            app,
            [str(unsupported), "-o", str(tmp_path / "output")],
        )
        assert result.exit_code == 1


# =============================================================================
# Test: Batch Conversion (Fast - txt/md only)
# =============================================================================


class TestBatchConversion:
    """Tests for batch (directory) conversion."""

    def test_batch_convert(self, runner: CliRunner, batch_dir: Path, tmp_path: Path):
        """Test batch conversion of directory."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(batch_dir), "-o", str(output_dir)],
        )
        assert result.exit_code == 0
        assert (output_dir / "file1.txt.md").exists()
        assert (output_dir / "file2.txt.md").exists()
        assert (output_dir / "subdir" / "nested.txt.md").exists()

    def test_batch_dry_run(self, runner: CliRunner, batch_dir: Path, tmp_path: Path):
        """Test batch dry run shows files without converting."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(batch_dir), "-o", str(output_dir), "--dry-run"],
        )
        assert result.exit_code == 0
        assert not (output_dir / "file1.txt.md").exists()

    def test_batch_empty_directory(self, runner: CliRunner, tmp_path: Path):
        """Test batch conversion of empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(
            app,
            [str(empty_dir), "-o", str(tmp_path / "output")],
        )
        assert result.exit_code == 0
        assert "No supported files" in result.output

    def test_batch_generates_report(
        self, runner: CliRunner, batch_dir: Path, tmp_path: Path
    ):
        """Test batch conversion generates report."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(batch_dir), "-o", str(output_dir)],
        )
        assert result.exit_code == 0

        reports_dir = output_dir / "reports"
        assert reports_dir.exists()
        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1


# =============================================================================
# Test: Configuration File Loading
# =============================================================================


class TestConfigFileLoading:
    """Tests for configuration file loading priority."""

    def test_config_flag(
        self, runner: CliRunner, sample_txt: Path, sample_config: Path, tmp_path: Path
    ):
        """Test -c/--config specifies config file."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-c", str(sample_config)],
        )
        assert result.exit_code == 0

    def test_invalid_config_file(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test invalid config file path shows error."""
        result = runner.invoke(
            app,
            [
                str(sample_txt),
                "-o",
                str(tmp_path / "output"),
                "-c",
                "/nonexistent/config.json",
            ],
        )
        assert result.exit_code != 0


# =============================================================================
# Test: Conflict Handling
# =============================================================================


class TestConflictHandling:
    """Tests for output file conflict handling."""

    def test_skip_existing(self, runner: CliRunner, sample_txt: Path, tmp_path: Path):
        """Test skip mode keeps existing files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "sample.txt.md"
        existing.write_text("existing content")

        config = tmp_path / "config.json"
        config.write_text('{"output": {"on_conflict": "skip"}}')

        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-c", str(config)],
        )
        assert result.exit_code == 0
        assert existing.read_text() == "existing content"

    def test_overwrite_existing(
        self, runner: CliRunner, sample_txt: Path, tmp_path: Path
    ):
        """Test overwrite mode replaces existing files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "sample.txt.md"
        existing.write_text("existing content")

        config = tmp_path / "config.json"
        config.write_text('{"output": {"on_conflict": "overwrite"}}')

        result = runner.invoke(
            app,
            [str(sample_txt), "-o", str(output_dir), "-c", str(config)],
        )
        assert result.exit_code == 0
        assert existing.read_text() != "existing content"


# =============================================================================
# Test: Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_special_characters_in_filename(self, runner: CliRunner, tmp_path: Path):
        """Test handling of special characters in filename."""
        special_file = tmp_path / "test (1) [copy].txt"
        special_file.write_text("content")
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(special_file), "-o", str(output_dir)],
        )
        assert result.exit_code == 0

    def test_unicode_content(self, runner: CliRunner, tmp_path: Path):
        """Test handling of unicode content."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("中文内容 日本語 한국어", encoding="utf-8")
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(unicode_file), "-o", str(output_dir)],
        )
        assert result.exit_code == 0

    def test_empty_file(self, runner: CliRunner, tmp_path: Path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(empty_file), "-o", str(output_dir)],
        )
        assert result.exit_code == 0


# =============================================================================
# Test: URL Conversion
# =============================================================================


class TestURLConversion:
    """Tests for URL conversion."""

    def test_url_input_recognized(self, runner: CliRunner, tmp_path: Path):
        """Test URL input is recognized."""
        result = runner.invoke(
            app,
            ["https://example.com", "-o", str(tmp_path / "output"), "--dry-run"],
        )
        assert result.exit_code == 0


# =============================================================================
# Test: Case Sensitivity Warning
# =============================================================================


class TestCaseSensitivityWarning:
    """Tests for case-sensitivity warning in --no-cache-for patterns."""

    def test_warn_case_sensitivity_function(self, tmp_path: Path):
        """Test case sensitivity warning function."""
        from markitai.cli import _warn_case_sensitivity_mismatches

        (tmp_path / "image.JPG").touch()
        files = list(tmp_path.glob("*"))
        patterns = ["*.jpg"]
        # Should not raise
        _warn_case_sensitivity_mismatches(files, tmp_path, patterns)

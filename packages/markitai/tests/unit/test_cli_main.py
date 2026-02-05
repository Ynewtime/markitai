"""Unit tests for CLI main module.

Tests CLI option parsing, configuration merging, output path handling,
dry run mode, and error handling paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from markitai.cli import app

# =============================================================================
# CLI Option Parsing Tests
# =============================================================================


class TestCLIOptions:
    """Tests for CLI option parsing."""

    def test_help_displays_without_error(self, cli_runner: CliRunner) -> None:
        """Test that --help works correctly."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Markitai" in result.output
        assert "INPUT" in result.output
        assert "Options" in result.output

    def test_short_help_option(self, cli_runner: CliRunner) -> None:
        """Test that -h works as help shortcut."""
        result = cli_runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        assert "Markitai" in result.output

    def test_no_input_shows_help(self, cli_runner: CliRunner) -> None:
        """Test that invoking without input shows help."""
        result = cli_runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_output_option_short(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test -o output option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        assert result.exit_code == 0

    def test_output_option_long(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --output option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(app, [str(test_file), "--output", str(output_dir)])
        assert result.exit_code == 0

    def test_verbose_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --verbose flag enables verbose output."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--verbose"]
        )
        assert result.exit_code == 0

    def test_quiet_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --quiet/-q flag suppresses output."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(app, [str(test_file), "-o", str(output_dir), "-q"])
        assert result.exit_code == 0

    def test_llm_flag_enable(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --llm flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--llm", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "LLM" in result.output

    def test_llm_flag_disable(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-llm flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--no-llm", "--dry-run"]
        )
        assert result.exit_code == 0


# =============================================================================
# Preset Tests
# =============================================================================


class TestPresets:
    """Tests for preset configuration."""

    def test_preset_rich(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --preset rich applies correct settings."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--preset", "rich", "--dry-run"],
        )
        assert result.exit_code == 0
        # Rich preset enables LLM, alt, desc, screenshot
        assert "LLM" in result.output

    def test_preset_standard(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --preset standard applies correct settings."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--preset",
                "standard",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_preset_minimal(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --preset minimal applies correct settings."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--preset", "minimal", "--dry-run"],
        )
        assert result.exit_code == 0
        # Minimal preset disables everything
        assert "Features: none" in result.output

    def test_preset_short_option(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test -p preset option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "-p", "minimal", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_preset_override_with_flag(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test that CLI flags can override preset settings."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Rich preset enables desc, but --no-desc should override
        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--preset",
                "rich",
                "--no-desc",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


# =============================================================================
# Configuration Merging Tests
# =============================================================================


class TestConfigMerging:
    """Tests for configuration merging logic."""

    def test_cli_overrides_preset(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test CLI flags override preset values."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Rich preset enables LLM, but --no-llm should disable it
        # Note: alt/desc/screenshot may still be enabled (they depend on LLM at runtime)
        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--preset",
                "rich",
                "--no-llm",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        # --no-llm disables LLM, but alt/desc/screenshot are still listed as features
        # (they just won't work without LLM at runtime)
        assert "Would convert" in result.output
        # Should show the LLM Required warning since alt/desc are enabled but LLM is disabled
        assert "LLM Required" in result.output or "Features:" in result.output

    def test_config_file_option(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --config/-c option loads config file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Create a config file
        config_file = tmp_path / "markitai.json"
        config_file.write_text(
            json.dumps(
                {
                    "llm": {"enabled": True},
                    "image": {"compress": False},
                }
            )
        )

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "-c",
                str(config_file),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_batch_concurrency_option(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --batch-concurrency/-j option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "-j", "4", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_llm_concurrency_option(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --llm-concurrency option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--llm-concurrency",
                "8",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_url_concurrency_option(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --url-concurrency option."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--url-concurrency",
                "3",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


# =============================================================================
# Output Path Handling Tests
# =============================================================================


class TestOutputPathHandling:
    """Tests for output path handling."""

    def test_output_dir_created(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test output directory is created if it doesn't exist."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "new_output"

        result = cli_runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_nested_output_dir(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test nested output directories are created."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "a" / "b" / "c"

        result = cli_runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        assert result.exit_code == 0
        assert output_dir.exists()


# =============================================================================
# Dry Run Mode Tests
# =============================================================================


class TestDryRunMode:
    """Tests for dry run mode."""

    def test_dry_run_file(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --dry-run with file input."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "Would convert" in result.output

    def test_dry_run_url(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test --dry-run with URL input."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["https://example.com", "-o", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Would convert URL" in result.output

    def test_dry_run_no_files_created(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test that dry run doesn't create any files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        # Output dir should NOT be created in dry run
        assert not output_dir.exists()

    def test_dry_run_shows_features(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test dry run shows enabled features."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--llm",
                "--alt",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Features:" in result.output
        assert "LLM" in result.output


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling paths."""

    def test_nonexistent_file(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test error handling for non-existent file."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["/nonexistent/file.txt", "-o", str(output_dir)]
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_nonexistent_config_file(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test error handling for non-existent config file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "-c", "/nonexistent/config.json"],
        )
        # Click validates exists=True for config path
        assert result.exit_code != 0

    def test_mutually_exclusive_fetch_options(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test error for mutually exclusive --playwright and --jina."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                "https://example.com",
                "-o",
                str(output_dir),
                "--playwright",
                "--jina",
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_invalid_preset(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test error handling for invalid preset."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--preset", "invalid"]
        )
        # Click validates Choice options
        assert result.exit_code != 0


# =============================================================================
# Image/Screenshot Options Tests
# =============================================================================


class TestImageOptions:
    """Tests for image-related CLI options."""

    def test_alt_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --alt flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--alt", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_no_alt_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-alt flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--no-alt", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_desc_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --desc flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--desc", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_screenshot_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --screenshot flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--screenshot", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_screenshot_only_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --screenshot-only flag enables screenshot."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--screenshot-only", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "screenshot" in result.output.lower()

    def test_no_compress_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-compress flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--no-compress", "--dry-run"]
        )
        assert result.exit_code == 0


# =============================================================================
# OCR Options Tests
# =============================================================================


class TestOCROptions:
    """Tests for OCR-related CLI options."""

    def test_ocr_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --ocr flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--ocr", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "OCR" in result.output

    def test_no_ocr_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-ocr flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--no-ocr", "--dry-run"]
        )
        assert result.exit_code == 0


# =============================================================================
# Cache Options Tests
# =============================================================================


class TestCacheOptions:
    """Tests for cache-related CLI options."""

    def test_no_cache_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-cache flag."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--no-cache", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_no_cache_for_option(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --no-cache-for option with patterns."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--no-cache-for",
                "*.pdf,*.docx",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


# =============================================================================
# Fetch Strategy Tests
# =============================================================================


class TestFetchStrategy:
    """Tests for fetch strategy options."""

    def test_playwright_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --playwright flag."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "--playwright", "--dry-run"],
        )
        assert result.exit_code == 0

    def test_jina_flag(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test --jina flag."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["https://example.com", "-o", str(output_dir), "--jina", "--dry-run"]
        )
        assert result.exit_code == 0


# =============================================================================
# URL List Processing Tests
# =============================================================================


class TestURLListProcessing:
    """Tests for .urls file processing."""

    def test_empty_urls_file(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test handling of empty .urls file."""
        urls_file = tmp_path / "urls.urls"
        urls_file.write_text("")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(app, [str(urls_file), "-o", str(output_dir)])
        assert result.exit_code == 0
        assert "No valid URLs" in result.output

    def test_urls_file_with_comments(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test .urls file parsing with comments."""
        urls_file = tmp_path / "urls.urls"
        urls_file.write_text(
            """# This is a comment
https://example.com

# Another comment
https://example.org
"""
        )
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(urls_file), "-o", str(output_dir), "--dry-run"]
        )
        # Should process without error
        assert result.exit_code == 0


# =============================================================================
# Config Subcommand Tests
# =============================================================================


class TestConfigSubcommand:
    """Tests for config subcommand."""

    def test_config_list_json(self, cli_runner: CliRunner) -> None:
        """Test config list with JSON format."""
        result = cli_runner.invoke(app, ["config", "list", "-f", "json"])
        assert result.exit_code == 0
        # Should output valid JSON
        assert "{" in result.output

    def test_config_list_table(self, cli_runner: CliRunner) -> None:
        """Test config list with table format."""
        result = cli_runner.invoke(app, ["config", "list", "-f", "table"])
        assert result.exit_code == 0

    def test_config_path(self, cli_runner: CliRunner) -> None:
        """Test config path shows search order."""
        result = cli_runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        # Supports both English and Chinese UI
        assert (
            "Configuration" in result.output
            or "配置来源" in result.output
            or "◆" in result.output  # Unified UI title marker
        )

    def test_config_init(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test config init creates config file."""
        config_path = tmp_path / "markitai.json"

        result = cli_runner.invoke(app, ["config", "init", "-o", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()

    def test_config_init_to_directory(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test config init with directory path creates markitai.json."""
        result = cli_runner.invoke(app, ["config", "init", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "markitai.json").exists()

    def test_config_validate_valid(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test config validate with valid config."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"llm": {"enabled": False}}))

        result = cli_runner.invoke(app, ["config", "validate", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_config_get(self, cli_runner: CliRunner) -> None:
        """Test config get retrieves value."""
        result = cli_runner.invoke(app, ["config", "get", "llm.enabled"])
        assert result.exit_code == 0
        # Should output the value (True or False)
        assert result.output.strip() in ("True", "False")

    def test_config_get_nonexistent(self, cli_runner: CliRunner) -> None:
        """Test config get with non-existent key."""
        result = cli_runner.invoke(app, ["config", "get", "nonexistent.key"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


# =============================================================================
# Cache Subcommand Tests
# =============================================================================


class TestCacheSubcommand:
    """Tests for cache subcommand."""

    def test_cache_stats(self, cli_runner: CliRunner) -> None:
        """Test cache stats displays without error."""
        result = cli_runner.invoke(app, ["cache", "stats"])
        assert result.exit_code == 0
        # Support both English and Chinese output
        assert (
            "Cache" in result.output
            or "cache" in result.output.lower()
            or "缓存" in result.output
        )

    def test_cache_stats_json(self, cli_runner: CliRunner) -> None:
        """Test cache stats with JSON output."""
        result = cli_runner.invoke(app, ["cache", "stats", "--json"])
        assert result.exit_code == 0
        # Should output valid JSON
        data = json.loads(result.output)
        assert "enabled" in data

    def test_cache_spa_domains(self, cli_runner: CliRunner) -> None:
        """Test cache spa-domains displays without error."""
        result = cli_runner.invoke(app, ["cache", "spa-domains"])
        assert result.exit_code == 0


# =============================================================================
# Input Position Tests
# =============================================================================


class TestInputPosition:
    """Tests for INPUT argument position handling."""

    def test_input_before_options(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test INPUT can come before options."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(test_file), "-o", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0

    def test_input_after_options(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test INPUT can come after options."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["-o", str(output_dir), "--dry-run", str(test_file)]
        )
        assert result.exit_code == 0

    def test_input_between_options(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test INPUT can come between options."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["-o", str(output_dir), str(test_file), "--dry-run"]
        )
        assert result.exit_code == 0


# =============================================================================
# Batch Mode Tests
# =============================================================================


class TestBatchMode:
    """Tests for batch (directory) mode."""

    def test_batch_mode_directory(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """Test batch mode with directory input."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("content 1")
        (input_dir / "file2.txt").write_text("content 2")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(app, [str(input_dir), "-o", str(output_dir)])
        assert result.exit_code == 0

    def test_batch_mode_resume_flag(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --resume flag for batch mode."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("content")
        output_dir = tmp_path / "out"

        # First run
        cli_runner.invoke(app, [str(input_dir), "-o", str(output_dir)])

        # Second run with --resume
        result = cli_runner.invoke(
            app, [str(input_dir), "-o", str(output_dir), "--resume"]
        )
        assert result.exit_code == 0

"""Unit tests for CLI main module.

Tests CLI option parsing, configuration merging, output path handling,
dry run mode, and error handling paths.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import app

# rich force-enables color when GITHUB_ACTIONS is set, so CI output carries
# ANSI codes; strip them before matching option names in rendered output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Return ``text`` without ANSI style escapes."""
    return _ANSI_RE.sub("", text)


# =============================================================================
# CLI Option Parsing Tests
# =============================================================================


class TestCLIOptions:
    """Tests for CLI option parsing."""

    def test_cli_sets_local_litellm_cost_map_by_default(self) -> None:
        """CLI startup should avoid LiteLLM remote pricing fetch noise."""
        assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"

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

    def test_help_sections_single_blank_line(self, cli_runner: CliRunner) -> None:
        """Docstring sections are separated by exactly one blank line each."""
        result = cli_runner.invoke(app, ["-h"])
        assert result.exit_code == 0
        # rich force-enables color when GITHUB_ACTIONS is set, so CI output
        # carries ANSI codes (\x1b[2mPresets:\x1b[0m) — strip before matching
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        lines = [ansi.sub("", ln).rstrip() for ln in result.output.splitlines()]
        for header in ("Presets:", "Examples:"):
            idx = next(
                (i for i, ln in enumerate(lines) if ln.strip().startswith(header)),
                None,
            )
            assert idx is not None, (
                f"section header {header!r} not found in -h output:\n" + result.output
            )
            assert lines[idx - 1] == "", f"expected blank line before {header}"
            assert lines[idx - 2] != "", f"expected single blank before {header}"

    def test_ocr_help_mentions_vlm_path(self, cli_runner: CliRunner) -> None:
        """--ocr help must name both OCR paths (C5 naming)."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        text = _strip_ansi(result.output)
        assert "VLM OCR" in text
        assert "RapidOCR" in text

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


class TestPureModeWarnings:
    """Tests for --pure mode warnings when combined with features it overrides."""

    def test_pure_with_alt_warns(self, tmp_path: Path, cli_runner: CliRunner) -> None:
        """--pure with --alt should warn that --alt is ignored."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--pure",
                "--llm",
                "--alt",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "--alt" in result.output
        assert "--pure" in result.output

    def test_pure_with_preset_rich_warns(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """--preset rich --pure should warn about ignored features."""
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
                "rich",
                "--pure",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "--pure" in result.output
        # Should mention at least one of the ignored features
        assert any(
            flag in result.output for flag in ("--alt", "--desc", "--screenshot")
        )

    def test_pure_with_screenshot_only_does_not_warn_screenshot(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """--pure --screenshot-only should NOT warn about --screenshot being ignored."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--pure",
                "--llm",
                "--screenshot-only",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "--screenshot" not in result.output

    def test_pure_without_conflicting_flags_no_warning(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """--pure alone should not produce any warning."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--pure", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "ignored" not in result.output.lower()


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
        # New unified UI format uses "◆ Dry Run" header
        assert "Dry Run" in result.output
        # Should show the LLM Required warning since alt/desc are enabled but LLM is disabled
        assert "LLM" in result.output or "Features:" in result.output

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
        # New unified UI format uses "◆ Dry Run" header with file listing
        assert "Dry Run" in result.output
        assert "Files (1)" in result.output or "test.txt" in result.output

    def test_dry_run_url(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test --dry-run with URL input."""
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, ["https://example.com", "-o", str(output_dir), "--dry-run"]
        )
        assert result.exit_code == 0
        # New format uses "Dry Run" panel instead of "Would convert URL"
        assert "Dry Run" in result.output or "URL:" in result.output

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

    def test_missing_env_var_shows_friendly_error(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test that missing env var API key shows friendly error, not traceback."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file.txt").write_text("content")
        output_dir = tmp_path / "out"

        config = {
            "llm": {
                "enabled": True,
                "model_list": [
                    {
                        "model_name": "default",
                        "litellm_params": {
                            "model": "gemini/gemini-2.0-flash",
                            "api_key": "env:NONEXISTENT_CLI_TEST_KEY_999",
                        },
                    }
                ],
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = cli_runner.invoke(
            app, [str(input_dir), "-o", str(output_dir), "-c", str(config_file)]
        )
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "NONEXISTENT_CLI_TEST_KEY_999" in result.output


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


class TestExplicitBooleanOverrides:
    """Explicit false wins over inherited config; omitted flags preserve it."""

    @pytest.mark.parametrize("inherited", [False, True])
    @pytest.mark.parametrize("explicit", [None, False, True])
    @pytest.mark.parametrize(
        ("section", "field", "positive", "negative", "inverse"),
        [
            (
                "screenshot",
                "screenshot_only",
                "--screenshot-only",
                "--no-screenshot-only",
                False,
            ),
            ("llm", "pure", "--pure", "--no-pure", False),
            ("cache", "no_cache", "--no-cache", "--cache", False),
            ("image", "compress", "--no-compress", "--compress", True),
        ],
    )
    def test_boolean_override(
        self,
        inherited: bool,
        explicit: bool | None,
        section: str,
        field: str,
        positive: str,
        negative: str,
        inverse: bool,
        tmp_path: Path,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from markitai.config import ConfigManager, MarkitaiConfig

        cfg = MarkitaiConfig()
        target = getattr(cfg, section)
        setattr(target, field, not inherited if inverse else inherited)
        cfg.screenshot.enabled = False
        monkeypatch.delenv("MARKITAI_PURE", raising=False)
        monkeypatch.setattr(ConfigManager, "load", lambda *_args, **_kwargs: cfg)
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        flags = [] if explicit is None else [positive if explicit else negative]
        result = cli_runner.invoke(app, [str(sample), *flags, "--dry-run"])
        assert result.exit_code == 0, result.output
        expected = inherited if explicit is None else explicit
        assert getattr(target, field) is (not expected if inverse else expected)
        if section == "screenshot":
            assert cfg.screenshot.enabled is (explicit is True)

    @pytest.mark.parametrize("flag", [None, "--pure", "--no-pure"])
    def test_explicit_pure_overrides_environment(
        self,
        flag: str | None,
        tmp_path: Path,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from markitai.config import ConfigManager, MarkitaiConfig

        cfg = MarkitaiConfig()
        monkeypatch.setenv("MARKITAI_PURE", "true")
        monkeypatch.setattr(ConfigManager, "load", lambda *_args, **_kwargs: cfg)
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        result = cli_runner.invoke(
            app, [str(sample), *([flag] if flag else []), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert cfg.llm.pure is (flag != "--no-pure")


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

    def test_strategy_option_static(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test unified -s/--strategy option."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "-s", "static", "--dry-run"],
        )
        assert result.exit_code == 0

    def test_strategy_option_long_form(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --strategy long form."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            [
                "https://example.com",
                "-o",
                str(output_dir),
                "--strategy",
                "playwright",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_strategy_option_rejects_unknown_value(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test -s validates against the strategy choices."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            ["https://example.com", "-o", str(output_dir), "-s", "bogus", "--dry-run"],
        )
        assert result.exit_code != 0

    def test_strategy_option_shown_in_help(self, cli_runner: CliRunner) -> None:
        """Test -s/--strategy appears in --help."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output

    def test_backend_option_shown_in_help(self, cli_runner: CliRunner) -> None:
        """Test -b/--backend appears in --help."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    @pytest.mark.parametrize("backend", ["native", "kreuzberg", "cloudflare"])
    def test_explicit_backend_replaces_inherited_flags(
        self,
        backend: str,
        tmp_path: Path,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from markitai.config import ConfigManager, MarkitaiConfig

        cfg = MarkitaiConfig()
        cfg.fetch.kreuzberg_convert_enabled = True
        cfg.fetch.cloudflare.convert_enabled = True
        monkeypatch.setattr(ConfigManager, "load", lambda *_args, **_kwargs: cfg)
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        result = cli_runner.invoke(app, [str(sample), "-b", backend, "--dry-run"])
        assert result.exit_code == 0, result.output
        assert cfg.fetch.kreuzberg_convert_enabled is (backend == "kreuzberg")
        assert cfg.fetch.cloudflare.convert_enabled is (backend == "cloudflare")

    def test_backend_kreuzberg_conflicts_with_cloudflare_strategy(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        result = cli_runner.invoke(
            app,
            [str(sample), "-b", "kreuzberg", "-s", "cloudflare", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "exclusive" in result.output


# =============================================================================
# Removed Deprecated Alias Tests
# =============================================================================

# Removed flag name -> the supported spelling that replaced it.
REMOVED_ALIASES = {
    "--playwright": "-s playwright",
    "--defuddle": "-s defuddle",
    "--static": "-s static",
    "--jina": "-s jina",
    "--cloudflare": "-s cloudflare",
    "--kreuzberg": "-b kreuzberg",
}


class TestRemovedDeprecatedAliases:
    """The six removed strategy/backend aliases must fail with a migration hint."""

    @pytest.mark.parametrize(("flag", "replacement"), sorted(REMOVED_ALIASES.items()))
    def test_removed_alias_names_the_replacement(
        self, flag: str, replacement: str, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """A removed alias errors out pointing at the supported spelling."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app, [str(sample), "-o", str(output_dir), flag, "--dry-run"]
        )

        # Same exit code as any other usage error
        assert result.exit_code == 2
        stderr = _strip_ansi(result.stderr)
        assert flag in stderr
        assert replacement in stderr
        # Not the bare click message
        assert "No such option" not in stderr

    @pytest.mark.parametrize(("flag", "replacement"), sorted(REMOVED_ALIASES.items()))
    def test_removed_alias_before_input_also_reports(
        self, flag: str, replacement: str, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Options may precede INPUT; detection must not depend on position."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        result = cli_runner.invoke(app, [flag, str(sample), "--dry-run"])

        assert result.exit_code == 2
        assert replacement in _strip_ansi(result.stderr)

    @pytest.mark.parametrize("flag", sorted(REMOVED_ALIASES))
    def test_removed_alias_absent_from_help(
        self, flag: str, cli_runner: CliRunner
    ) -> None:
        """--help must not advertise the removed aliases any more."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert flag not in _strip_ansi(result.output)

    def test_removed_alias_message_goes_to_stderr_only(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """The migration error must not pollute stdout."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        result = cli_runner.invoke(app, [str(sample), "--playwright"])

        assert result.exit_code == 2
        assert "--playwright" not in result.stdout

    def test_replacement_strategy_still_works(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """`-s playwright` (the replacement) is unaffected."""
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app,
            [
                "https://example.com",
                "-o",
                str(output_dir),
                "-s",
                "playwright",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_replacement_backend_still_works(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """`-b kreuzberg` (the replacement) is unaffected."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")
        output_dir = tmp_path / "out"
        result = cli_runner.invoke(
            app, [str(sample), "-o", str(output_dir), "-b", "kreuzberg", "--dry-run"]
        )
        assert result.exit_code == 0

    def test_unknown_option_keeps_default_click_error(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Options we never had still get click's own error, not a hint."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        result = cli_runner.invoke(app, [str(sample), "--nope"])

        assert result.exit_code == 2
        assert "No such option" in _strip_ansi(result.stderr)


# =============================================================================
# Console Verbosity Tests
# =============================================================================


def _flattened_help(cli_runner: CliRunner) -> str:
    """Return --help as one whitespace-normalized line, free of ANSI and box art."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = _strip_ansi(result.output)
    text = re.sub(r"[│┃╭╮╰╯─━┌┐└┘]", " ", text)
    return re.sub(r"\s+", " ", text)


class TestConsoleVerbosity:
    """Pin the implicit quiet rules and require --help to spell them out."""

    @staticmethod
    def _console_quiet(cli_runner: CliRunner, args: list[str]) -> bool:
        """Return the ``quiet`` value the CLI computed for the console logger."""
        with patch("markitai.cli.main.setup_logging", return_value=(1, None)) as setup:
            result = cli_runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert setup.call_count == 1
        return bool(setup.call_args.kwargs["quiet"])

    def test_single_file_is_quiet_by_default(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """A single file conversion suppresses console logs without asking."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        quiet = self._console_quiet(
            cli_runner, [str(sample), "-o", str(tmp_path / "out"), "--dry-run"]
        )

        assert quiet is True

    def test_verbose_unquiets_single_file(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """-v is what turns the hidden single-file output back on."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        quiet = self._console_quiet(
            cli_runner, [str(sample), "-o", str(tmp_path / "out"), "-v", "--dry-run"]
        )

        assert quiet is False

    def test_stdout_mode_stays_quiet_even_with_verbose(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Without -o the Markdown is the output, so -v must not pollute it."""
        sample = tmp_path / "sample.txt"
        sample.write_text("hello")

        quiet = self._console_quiet(cli_runner, [str(sample), "-v", "--dry-run"])

        assert quiet is True

    def test_directory_batch_is_not_quiet_by_default(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Batch runs keep their progress output unless asked otherwise."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "sample.txt").write_text("hello")

        quiet = self._console_quiet(
            cli_runner, [str(src), "-o", str(tmp_path / "out"), "--dry-run"]
        )

        assert quiet is False

    def test_quiet_flag_silences_directory_batch(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """-q is what a batch run needs to go silent."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "sample.txt").write_text("hello")

        quiet = self._console_quiet(
            cli_runner, [str(src), "-o", str(tmp_path / "out"), "-q", "--dry-run"]
        )

        assert quiet is True

    def test_quiet_help_documents_the_default(self, cli_runner: CliRunner) -> None:
        """--help must say single conversions are quiet already, batches are not."""
        text = _flattened_help(cli_runner)

        assert "already quiet by default" in text
        assert "batch" in text.lower()

    def test_verbose_help_documents_stdout_exception(
        self, cli_runner: CliRunner
    ) -> None:
        """--help must say -v does nothing when the result goes to stdout."""
        text = _flattened_help(cli_runner)

        assert "quiet by default" in text
        assert "stdout" in text


class TestScreenshotOnlyHelp:
    """--screenshot-only means two different things; --help must say both."""

    def test_help_describes_both_llm_modes(self, cli_runner: CliRunner) -> None:
        text = _flattened_help(cli_runner)

        assert "With --llm" in text
        assert "Without --llm" in text
        assert "implies --screenshot" in text


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

    def test_urls_batch_receives_console_handler_id(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """URL list batch mode should suspend console logs during progress."""
        urls_file = tmp_path / "urls.urls"
        urls_file.write_text("https://example.com\n")
        output_dir = tmp_path / "out"

        with patch(
            "markitai.cli.processors.url.process_url_batch",
            new_callable=AsyncMock,
        ) as mock_process_url_batch:
            result = cli_runner.invoke(
                app,
                [str(urls_file), "-o", str(output_dir), "--verbose"],
            )

        assert result.exit_code == 0
        assert mock_process_url_batch.await_count == 1
        assert mock_process_url_batch.await_args is not None
        assert (
            mock_process_url_batch.await_args.kwargs["console_handler_id"] is not None
        )


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

        # New: "config init" replaced by "init" (Task 5.3)
        # Use -y for non-interactive mode
        result = cli_runner.invoke(app, ["init", "-y", "-o", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()

    def test_config_init_to_directory(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test config init with directory path creates markitai.json."""
        # New: "config init" replaced by "init" (Task 5.3)
        # Use -y for non-interactive mode
        result = cli_runner.invoke(app, ["init", "-y", "-o", str(tmp_path)])
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

    def test_batch_mode_glob_filters_directory_input(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test repeated --glob filters only affect directory batch mode."""
        input_dir = tmp_path / "input"
        include_dir = input_dir / "reports"
        exclude_dir = include_dir / "private"
        include_dir.mkdir(parents=True)
        exclude_dir.mkdir(parents=True)

        (include_dir / "public.pdf").write_text("public")
        (exclude_dir / "secret.pdf").write_text("secret")
        (input_dir / "notes.txt").write_text("notes")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "-g",
                "reports/**/*.pdf",
                "-g",
                "!reports/private/**",
            ],
        )

        assert result.exit_code == 0
        assert "Files (1)" in result.output
        assert "public.pdf" in result.output
        assert "secret.pdf" not in result.output

    def test_batch_mode_max_depth_overrides_config(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test --max-depth limits discovery depth for directory batch mode."""
        input_dir = tmp_path / "input"
        nested_dir = input_dir / "nested"
        input_dir.mkdir()
        nested_dir.mkdir()

        (input_dir / "root.txt").write_text("root")
        (nested_dir / "deep.txt").write_text("deep")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "--max-depth",
                "0",
            ],
        )

        assert result.exit_code == 0
        assert "Files (1)" in result.output
        assert "root.txt" in result.output
        assert "deep.txt" not in result.output

    def test_batch_mode_glob_filters_url_list_files(
        self, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        """Test repeated --glob filters also constrain discovered .urls files."""
        input_dir = tmp_path / "input"
        include_dir = input_dir / "feeds"
        exclude_dir = input_dir / "archive"
        include_dir.mkdir(parents=True)
        exclude_dir.mkdir(parents=True)

        (include_dir / "links.urls").write_text("https://example.com/feed\n")
        (exclude_dir / "old.urls").write_text("https://example.com/archive\n")
        output_dir = tmp_path / "out"

        result = cli_runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "-g",
                "feeds/**",
                "-g",
                "!archive/**",
            ],
        )

        assert result.exit_code == 0
        assert "URLs (1)" in result.output
        assert "https://example.com/feed" in result.output
        assert "https://example.com/archive" not in result.output

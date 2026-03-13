"""Tests for CLI main module interactive mode."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from click.testing import CliRunner

from markitai.cli.main import app


class TestInteractiveFlag:
    """Tests for -I/--interactive flag."""

    def test_interactive_short_flag(self) -> None:
        """Should recognize -I flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            runner.invoke(app, ["-I"])
            # Should call interactive mode, not show help
            mock_run.assert_called_once()

    def test_interactive_long_flag(self) -> None:
        """Should recognize --interactive flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            runner.invoke(app, ["--interactive"])
            mock_run.assert_called_once()

    def test_no_args_shows_help(self) -> None:
        """Should show help when no arguments provided."""
        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "usage:" in result.output.lower()


class TestInteractiveModeExitCode:
    """Tests for interactive mode exit code propagation."""

    def test_propagates_nonzero_exit_code(self) -> None:
        """Should propagate subprocess non-zero exit code instead of always exiting 0."""
        from markitai.cli.interactive import InteractiveSession

        session = InteractiveSession(input_path="test.pdf")

        # subprocess.run returns a CompletedProcess with returncode=1
        mock_result = subprocess.CompletedProcess(
            args=["python", "-m", "markitai", "test.pdf"],
            returncode=1,
        )

        with (
            patch(
                "markitai.cli.interactive.run_interactive",
                return_value=session,
            ),
            patch(
                "markitai.cli.interactive.session_to_cli_args",
                return_value=["test.pdf", "-o", "./output", "--no-llm"],
            ),
            patch("questionary.confirm") as mock_confirm,
            patch("subprocess.run", return_value=mock_result),
        ):
            mock_confirm.return_value.ask.return_value = True
            runner = CliRunner()
            result = runner.invoke(app, ["-I"])
            assert result.exit_code == 1

    def test_propagates_zero_exit_code_on_success(self) -> None:
        """Should exit 0 when subprocess succeeds."""
        from markitai.cli.interactive import InteractiveSession

        session = InteractiveSession(input_path="test.pdf")

        mock_result = subprocess.CompletedProcess(
            args=["python", "-m", "markitai", "test.pdf"],
            returncode=0,
        )

        with (
            patch(
                "markitai.cli.interactive.run_interactive",
                return_value=session,
            ),
            patch(
                "markitai.cli.interactive.session_to_cli_args",
                return_value=["test.pdf", "-o", "./output", "--no-llm"],
            ),
            patch("questionary.confirm") as mock_confirm,
            patch("subprocess.run", return_value=mock_result),
        ):
            mock_confirm.return_value.ask.return_value = True
            runner = CliRunner()
            result = runner.invoke(app, ["-I"])
            assert result.exit_code == 0


class TestDryRunSkipsAuthPreflight:
    """Tests for dry-run skipping auth preflight check."""

    def test_dry_run_does_not_trigger_auth_preflight(self, tmp_path: object) -> None:
        """dry-run should return before auth preflight is reached."""
        import tempfile
        from pathlib import Path

        # Create a real temporary file to satisfy existence checks
        tmpdir = Path(tempfile.mkdtemp())
        test_file = tmpdir / "test.txt"
        test_file.write_text("hello")

        runner = CliRunner()
        with patch("markitai.providers.preflight_auth_check") as mock_preflight:
            result = runner.invoke(
                app,
                [str(test_file), "-o", str(tmpdir), "--llm", "--dry-run"],
            )
            # Auth preflight should NOT have been called
            mock_preflight.assert_not_called()
            # dry-run should succeed
            assert result.exit_code == 0


class TestAuthPreflightAfterValidation:
    """Tests for auth preflight happening AFTER parameter validation.

    Medium-2: Auth preflight should not run before basic parameter errors
    (like missing -o for URL mode) are caught. Users should see parameter
    errors immediately, not after a slow auth/login attempt.
    """

    def test_url_without_output_errors_before_auth(self) -> None:
        """URL mode without -o should fail with param error, NOT trigger auth preflight."""
        from unittest.mock import AsyncMock

        runner = CliRunner()
        with (
            patch(
                "markitai.providers.preflight_auth_check",
                new_callable=AsyncMock,
            ) as mock_preflight,
            # Override config output.dir so validation actually fires
            patch("markitai.cli.main.ConfigManager") as mock_cm,
        ):
            from markitai.config import MarkitaiConfig

            mock_cfg = MarkitaiConfig()
            mock_cfg.output.dir = ""  # No default output dir
            mock_cfg.llm.enabled = True
            from markitai.config import LiteLLMParams, ModelConfig

            mock_cfg.llm.model_list = [
                ModelConfig(
                    model_name="test",
                    litellm_params=LiteLLMParams(model="gpt-4o-mini"),
                )
            ]
            mock_cm.return_value.load.return_value = mock_cfg
            mock_cm.return_value.config_path = None

            result = runner.invoke(
                app,
                ["https://example.com", "--llm"],
            )
            # Auth preflight should NOT have been called — param error comes first
            mock_preflight.assert_not_called()
            # Should get an error about missing output
            assert result.exit_code != 0

    def test_url_batch_without_output_errors_before_auth(self) -> None:
        """URL batch mode without -o should fail with param error, NOT trigger auth preflight."""
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock

        tmpdir = Path(tempfile.mkdtemp())
        urls_file = tmpdir / "urls.urls"
        urls_file.write_text("https://example.com\nhttps://example.org\n")

        runner = CliRunner()
        with (
            patch(
                "markitai.providers.preflight_auth_check",
                new_callable=AsyncMock,
            ) as mock_preflight,
            patch("markitai.cli.main.ConfigManager") as mock_cm,
        ):
            from markitai.config import MarkitaiConfig

            mock_cfg = MarkitaiConfig()
            mock_cfg.output.dir = ""
            mock_cfg.llm.enabled = True
            from markitai.config import LiteLLMParams, ModelConfig

            mock_cfg.llm.model_list = [
                ModelConfig(
                    model_name="test",
                    litellm_params=LiteLLMParams(model="gpt-4o-mini"),
                )
            ]
            mock_cm.return_value.load.return_value = mock_cfg
            mock_cm.return_value.config_path = None

            result = runner.invoke(
                app,
                [str(urls_file), "--llm"],
            )
            mock_preflight.assert_not_called()
            assert result.exit_code != 0

    def test_batch_dir_without_output_errors_before_auth(self) -> None:
        """Batch directory mode without -o should fail with param error, NOT trigger auth preflight."""
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock

        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "file.txt").write_text("hello")

        runner = CliRunner()
        with (
            patch(
                "markitai.providers.preflight_auth_check",
                new_callable=AsyncMock,
            ) as mock_preflight,
            patch("markitai.cli.main.ConfigManager") as mock_cm,
        ):
            from markitai.config import MarkitaiConfig

            mock_cfg = MarkitaiConfig()
            mock_cfg.output.dir = ""
            mock_cfg.llm.enabled = True
            from markitai.config import LiteLLMParams, ModelConfig

            mock_cfg.llm.model_list = [
                ModelConfig(
                    model_name="test",
                    litellm_params=LiteLLMParams(model="gpt-4o-mini"),
                )
            ]
            mock_cm.return_value.load.return_value = mock_cfg
            mock_cm.return_value.config_path = None

            result = runner.invoke(
                app,
                [str(tmpdir), "--llm"],
            )
            mock_preflight.assert_not_called()
            assert result.exit_code != 0


class TestPureCLIFlag:
    """Test --pure CLI flag."""

    def test_pure_flag_recognized(self, tmp_path):
        """--pure should be a recognized CLI flag."""
        from click.testing import CliRunner

        from markitai.cli.main import app

        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        result = runner.invoke(
            app,
            [str(txt_file), "--pure", "--dry-run", "-o", str(tmp_path / "out")],
        )
        # Should not fail with "no such option: --pure"
        assert "no such option" not in (result.output or "").lower()

    def test_pure_env_var(self, tmp_path):
        """MARKITAI_PURE=1 should enable pure mode."""
        from click.testing import CliRunner

        from markitai.cli.main import app

        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        result = runner.invoke(
            app,
            [str(txt_file), "--dry-run", "-o", str(tmp_path / "out")],
            env={"MARKITAI_PURE": "1"},
        )
        assert "no such option" not in (result.output or "").lower()

    def test_pure_flag_does_not_enable_llm(self, tmp_path):
        """--pure alone should NOT set llm.enabled = True."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from markitai.cli.main import app

        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        captured_cfg = {}

        async def capture_cfg(input_path, output_dir, cfg, *args, **kwargs):
            captured_cfg["llm_enabled"] = cfg.llm.enabled
            captured_cfg["llm_pure"] = cfg.llm.pure

        with patch(
            "markitai.cli.processors.file.process_single_file",
            side_effect=capture_cfg,
        ):
            runner.invoke(
                app,
                [str(txt_file), "--pure", "-o", str(tmp_path / "out")],
            )

        assert captured_cfg.get("llm_pure") is True
        assert captured_cfg.get("llm_enabled") is False

    def test_pure_env_var_does_not_enable_llm(self, tmp_path):
        """MARKITAI_PURE=1 should NOT set llm.enabled = True."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from markitai.cli.main import app

        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        captured_cfg = {}

        async def capture_cfg(input_path, output_dir, cfg, *args, **kwargs):
            captured_cfg["llm_enabled"] = cfg.llm.enabled
            captured_cfg["llm_pure"] = cfg.llm.pure

        with patch(
            "markitai.cli.processors.file.process_single_file",
            side_effect=capture_cfg,
        ):
            runner.invoke(
                app,
                [str(txt_file), "-o", str(tmp_path / "out")],
                env={"MARKITAI_PURE": "1"},
            )

        assert captured_cfg.get("llm_pure") is True
        assert captured_cfg.get("llm_enabled") is False


class TestKeepBaseCLIFlag:
    """Test --keep-base CLI flag."""

    def test_keep_base_option_exists(self) -> None:
        """--keep-base should appear in --help output."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "--keep-base" in result.output

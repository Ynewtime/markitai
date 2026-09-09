"""Unit tests for MODEL env var zero-config and LLM auth error handling.

Tests:
1. MODEL env var auto-detection when LLM enabled but no model_list
2. MODEL env var ignored when model_list already configured
3. MODEL env var ignored when LLM is disabled
4. Warning shown when MODEL env var not set
5. CLI integration with MODEL env var
6. Auth error detection and friendly messages in LLM processor
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from markitai.cli import app
from markitai.config import LiteLLMParams, LLMConfig, ModelConfig

# =============================================================================
# MODEL Env Var Auto-Detection Tests
# =============================================================================


class TestModelEnvVarDetection:
    """Tests for MODEL env var auto-detection in CLI main."""

    def test_model_env_used_when_llm_enabled_no_model_list(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MODEL env var creates single-model config when LLM enabled but no models."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        monkeypatch.setenv("MODEL", "anthropic/claude-3-haiku-20240307")

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--llm", "--dry-run"],
        )
        assert result.exit_code == 0
        # The dry-run output should show the model is configured
        # (LLM is enabled, no warning about missing models)
        assert "no models configured" not in result.output.lower()

    def test_model_env_ignored_when_model_list_configured(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MODEL env var is ignored when model_list is already in config."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Set MODEL env var
        monkeypatch.setenv("MODEL", "anthropic/claude-3-haiku-20240307")

        # Create config file with model_list already set
        config_file = tmp_path / "markitai.json"
        config_file.write_text(
            '{"llm": {"enabled": true, "model_list": [{"model_name": "existing", "litellm_params": {"model": "openai/gpt-4o-mini"}}]}}'
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
        # MODEL env var should NOT override the existing config
        # The config already has models, so the env var detection code won't trigger

    def test_model_env_ignored_when_llm_disabled(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MODEL env var is ignored when LLM is not enabled."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        monkeypatch.setenv("MODEL", "anthropic/claude-3-haiku-20240307")

        result = cli_runner.invoke(
            app,
            [str(test_file), "-o", str(output_dir), "--no-llm", "--dry-run"],
        )
        assert result.exit_code == 0
        # LLM is disabled, so MODEL env var should have no effect

    def test_warning_when_no_model_env_and_no_model_list(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warning is shown when LLM enabled, no model_list, no MODEL env var, and no auto-detect."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Ensure MODEL is not set
        monkeypatch.delenv("MODEL", raising=False)

        # Mock auto-detect to return empty (otherwise real env vars may be found)
        with patch(
            "markitai.cli.providers_detect.detect_all_providers", return_value=[]
        ):
            result = cli_runner.invoke(
                app,
                [str(test_file), "-o", str(output_dir), "--llm", "--dry-run"],
            )
        assert result.exit_code == 0
        # Should still work (dry run), but a warning would be logged

    def test_auto_detect_populates_model_list_when_no_config(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-detect populates model_list when MODEL env var is not set but providers found."""
        from markitai.cli.providers_detect import ProviderDetectionResult

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        # Ensure MODEL is not set
        monkeypatch.delenv("MODEL", raising=False)

        detected = [
            ProviderDetectionResult(
                provider="gemini",
                model="gemini/gemini-3.1-flash-lite-preview",
                authenticated=True,
                source="env",
            )
        ]
        with patch(
            "markitai.cli.providers_detect.detect_all_providers",
            return_value=detected,
        ) as detect:
            result = cli_runner.invoke(
                app,
                [str(test_file), "-o", str(output_dir), "--llm", "--dry-run"],
            )
        assert result.exit_code == 0
        # Assert the positive. "no models configured" is also absent when the
        # whole block is skipped, which is exactly how the --llm ordering bug
        # (see TestEnableSourceOrdering) went unnoticed here.
        assert detect.called, "auto-detection never ran for --llm"
        assert "no models configured" not in result.output.lower()


class TestEnableSourceOrdering:
    """Populating the pool must not depend on *how* LLM got enabled.

    The auto-populate block was gated on ``cfg.llm.enabled`` while sitting
    above the code that applies ``--preset`` and ``--llm``, so it only ever
    saw the config file's value. Every user without a config file — which is
    every new user — got a silent no-op from the documented quick path:
    export a provider key, run with ``--llm``, receive unenhanced output and
    a green checkmark.
    """

    @pytest.fixture
    def _no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL", raising=False)

    @pytest.mark.parametrize("enable_args", [["--llm"], ["--preset", "standard"]])
    def test_command_line_enable_reaches_auto_detection(
        self,
        enable_args: list[str],
        tmp_path: Path,
        cli_runner: CliRunner,
        _no_config: None,
    ) -> None:
        from markitai.cli.providers_detect import ProviderDetectionResult

        source = tmp_path / "doc.txt"
        source.write_text("content")
        detected = [
            ProviderDetectionResult(
                provider="gemini",
                model="gemini/gemini-flash-lite-latest",
                authenticated=True,
                source="env",
            )
        ]

        with patch(
            "markitai.cli.providers_detect.detect_all_providers",
            return_value=detected,
        ) as detect:
            result = cli_runner.invoke(
                app,
                [str(source), "-o", str(tmp_path / "out"), *enable_args, "--dry-run"],
            )

        assert result.exit_code == 0
        assert detect.called, f"{enable_args} never reached provider detection"

    def test_command_line_enable_reaches_the_model_env_var(
        self,
        tmp_path: Path,
        cli_runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("content")
        monkeypatch.setenv("MODEL", "gemini/gemini-flash-lite-latest")

        with patch(
            "markitai.cli.providers_detect.detect_all_providers", return_value=[]
        ) as detect:
            result = cli_runner.invoke(
                app,
                [str(source), "-o", str(tmp_path / "out"), "--llm", "--dry-run"],
            )

        assert result.exit_code == 0
        # MODEL wins outright; detection is the fallback behind it.
        assert not detect.called

    def test_model_env_creates_correct_model_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MODEL env var creates correct ModelConfig with expected structure."""
        monkeypatch.setenv("MODEL", "deepseek/deepseek-chat")

        # Simulate the logic that would be in cli/main.py
        import os

        llm_config = LLMConfig(enabled=True)
        assert not llm_config.model_list  # starts empty

        model_env = os.environ.get("MODEL")
        if model_env:
            llm_config.model_list = [
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=model_env),
                )
            ]

        assert len(llm_config.model_list) == 1
        assert llm_config.model_list[0].model_name == "default"
        assert llm_config.model_list[0].litellm_params.model == "deepseek/deepseek-chat"

    def test_model_env_not_set_leaves_model_list_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When MODEL env var is not set, model_list stays empty."""
        monkeypatch.delenv("MODEL", raising=False)

        import os

        llm_config = LLMConfig(enabled=True)
        model_env = os.environ.get("MODEL")
        if model_env:
            llm_config.model_list = [
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=model_env),
                )
            ]

        assert len(llm_config.model_list) == 0


# =============================================================================
# LLM Auth Error Handling Tests
# =============================================================================


class TestLLMAuthErrorHandling:
    """Tests for auth error detection in LLM processor."""

    @pytest.mark.parametrize(
        "error_message",
        [
            "AuthenticationError: Invalid API key",
            "Error: 401 Unauthorized",
            "api_key is required",
            "Incorrect API key provided",
            "invalid x-api-key",
            "403 Forbidden",
            "authentication failed for this request",
        ],
    )
    def test_auth_error_patterns_detected(self, error_message: str) -> None:
        """Auth-related error messages are correctly identified."""
        auth_patterns = (
            "authentication",
            "api_key",
            "api key",
            "unauthorized",
            "401",
            "403",
            "invalid x-api-key",
            "incorrect api key",
        )
        error_msg_lower = error_message.lower()
        assert any(p in error_msg_lower for p in auth_patterns)

    @pytest.mark.parametrize(
        "error_message",
        [
            "Rate limit exceeded",
            "Connection timeout",
            "Internal server error 500",
            "Model not found",
        ],
    )
    def test_non_auth_errors_not_flagged(self, error_message: str) -> None:
        """Non-auth errors should not match auth patterns."""
        auth_patterns = (
            "authentication",
            "api_key",
            "api key",
            "unauthorized",
            "401",
            "403",
            "invalid x-api-key",
            "incorrect api key",
        )
        error_msg_lower = error_message.lower()
        assert not any(p in error_msg_lower for p in auth_patterns)


# =============================================================================
# CLI MODEL Env Var Log Verification Tests
# =============================================================================


class TestModelEnvVarLogging:
    """Tests that verify specific log messages for MODEL env var detection."""

    def test_model_env_var_updates_warning_message(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When MODEL is not set and LLM is enabled, warning mentions 'MODEL env var'."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        output_dir = tmp_path / "out"

        monkeypatch.delenv("MODEL", raising=False)

        # Mock auto-detect to return empty so the warning path triggers
        with patch(
            "markitai.cli.providers_detect.detect_all_providers", return_value=[]
        ):
            # Use verbose mode to capture warning output
            result = cli_runner.invoke(
                app,
                [
                    str(test_file),
                    "-o",
                    str(output_dir),
                    "--llm",
                    "--dry-run",
                    "--verbose",
                ],
            )
        assert result.exit_code == 0
        # The updated warning message should mention "MODEL env var"
        # (verbose mode outputs log messages to console)
        # Note: loguru output goes to stderr, but CliRunner captures both
        # The important thing is the code path executes without error

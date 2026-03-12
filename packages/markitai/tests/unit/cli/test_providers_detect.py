"""Tests for shared provider detection module."""

from __future__ import annotations

from unittest.mock import patch

from markitai.cli.providers_detect import (
    detect_all_providers,
    detect_first_provider,
    format_model_list,
    get_active_models_from_config,
)


class TestDetectAllProviders:
    """Tests for detect_all_providers()."""

    def test_detect_claude_cli_authenticated(self) -> None:
        """Should detect authenticated Claude CLI."""
        with (
            patch(
                "markitai.cli.providers_detect.shutil.which",
                side_effect=lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
            ),
            patch(
                "markitai.cli.providers_detect._check_claude_auth",
                return_value=True,
            ),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            results = detect_all_providers()
            assert len(results) == 1
            assert results[0].provider == "claude-agent"
            assert results[0].model == "claude-agent/sonnet"
            assert results[0].authenticated is True
            assert results[0].source == "cli"

    def test_detect_copilot_cli_authenticated(self) -> None:
        """Should detect authenticated Copilot CLI."""
        with (
            patch(
                "markitai.cli.providers_detect.shutil.which",
                side_effect=lambda x: "/usr/bin/copilot" if x == "copilot" else None,
            ),
            patch(
                "markitai.cli.providers_detect._check_copilot_auth",
                return_value=True,
            ),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            results = detect_all_providers()
            assert len(results) == 1
            assert results[0].provider == "copilot"
            assert results[0].model == "copilot/claude-sonnet-4.5"

    def test_detect_env_providers(self) -> None:
        """Should detect environment variable providers."""
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict(
                "os.environ",
                {"ANTHROPIC_API_KEY": "sk-test", "OPENAI_API_KEY": "sk-test2"},
                clear=True,
            ),
        ):
            results = detect_all_providers()
            assert len(results) == 2
            providers = [r.provider for r in results]
            assert providers == ["anthropic", "openai"]

    def test_detect_all_returns_multiple(self) -> None:
        """Should return all available providers ordered by priority."""
        with (
            patch(
                "markitai.cli.providers_detect.shutil.which",
                side_effect=lambda x: "/usr/bin/claude" if x == "claude" else None,
            ),
            patch(
                "markitai.cli.providers_detect._check_claude_auth",
                return_value=True,
            ),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict(
                "os.environ",
                {"GEMINI_API_KEY": "test-key", "OPENAI_API_KEY": "sk-test"},
                clear=True,
            ),
        ):
            results = detect_all_providers()
            assert len(results) == 3
            providers = [r.provider for r in results]
            assert providers == ["claude-agent", "openai", "gemini"]

    def test_detect_no_provider(self) -> None:
        """Should return empty list when no provider is available."""
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            results = detect_all_providers()
            assert results == []

    def test_detect_chatgpt_provider(self) -> None:
        """Should detect ChatGPT when authenticated."""
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=True,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            results = detect_all_providers()
            assert len(results) == 1
            assert results[0].provider == "chatgpt"
            assert results[0].model == "chatgpt/gpt-5.2"

    def test_detect_gemini_cli_provider(self) -> None:
        """Should detect Gemini CLI when authenticated."""
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=True,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            results = detect_all_providers()
            assert len(results) == 1
            assert results[0].provider == "gemini-cli"
            assert results[0].model == "gemini-cli/gemini-2.5-pro"


class TestDetectFirstProvider:
    """Tests for detect_first_provider()."""

    def test_returns_first_provider(self) -> None:
        """Should return the highest-priority provider."""
        with (
            patch(
                "markitai.cli.providers_detect.shutil.which",
                side_effect=lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
            ),
            patch(
                "markitai.cli.providers_detect._check_claude_auth",
                return_value=True,
            ),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_first_provider()
            assert result is not None
            assert result.provider == "claude-agent"

    def test_returns_none_when_no_provider(self) -> None:
        """Should return None when no provider detected."""
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch(
                "markitai.cli.providers_detect._check_chatgpt_auth",
                return_value=False,
            ),
            patch(
                "markitai.cli.providers_detect._check_gemini_cli_auth",
                return_value=False,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_first_provider()
            assert result is None


class TestGetActiveModelsFromConfig:
    """Tests for get_active_models_from_config function."""

    def test_returns_models_with_positive_weight(self) -> None:
        """Should return only models with weight > 0."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-lite-preview",
                    "weight": 10,
                },
            },
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 0},
            },
            {
                "model_name": "default",
                "litellm_params": {"model": "copilot/claude-haiku-4.5", "weight": 5},
            },
        ]
        result = get_active_models_from_config(model_list)
        assert result == [
            "gemini/gemini-3.1-flash-lite-preview",
            "copilot/claude-haiku-4.5",
        ]

    def test_returns_empty_when_all_weight_zero(self) -> None:
        """Should return empty list when all models are disabled."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 0},
            },
            {
                "model_name": "default",
                "litellm_params": {"model": "copilot/gpt-4", "weight": 0},
            },
        ]
        result = get_active_models_from_config(model_list)
        assert result == []

    def test_returns_empty_for_empty_list(self) -> None:
        """Should return empty list for empty model_list."""
        assert get_active_models_from_config([]) == []

    def test_handles_missing_weight(self) -> None:
        """Models without explicit weight should be included (default weight > 0)."""
        model_list = [
            {
                "model_name": "default",
                "litellm_params": {"model": "anthropic/claude-sonnet"},
            },
        ]
        result = get_active_models_from_config(model_list)
        assert result == ["anthropic/claude-sonnet"]


class TestFormatModelList:
    """Tests for format_model_list function."""

    def test_format_short_list(self) -> None:
        """Should format a short list without suffix."""
        result = format_model_list(["model-a", "model-b"])
        assert result == "model-a, model-b"

    def test_format_exact_max(self) -> None:
        """Should format exactly max_show items without suffix."""
        result = format_model_list(["a", "b", "c"], max_show=3)
        assert result == "a, b, c"

    def test_format_exceeds_max(self) -> None:
        """Should add +N more suffix when exceeding max_show."""
        result = format_model_list(["a", "b", "c", "d", "e"], max_show=3)
        assert result == "a, b, c (+2 more)"

    def test_format_single_item(self) -> None:
        """Should format a single item."""
        result = format_model_list(["only-one"])
        assert result == "only-one"

    def test_format_empty_list(self) -> None:
        """Should handle empty list."""
        result = format_model_list([])
        assert result == ""


class TestBackwardCompatImports:
    """Test that backward-compatible imports from interactive still work."""

    def test_detect_all_llm_providers_importable(self) -> None:
        """detect_all_llm_providers should still be importable from interactive."""
        from markitai.cli.interactive import detect_all_llm_providers

        assert callable(detect_all_llm_providers)

    def test_detect_llm_provider_importable(self) -> None:
        """detect_llm_provider should still be importable from interactive."""
        from markitai.cli.interactive import detect_llm_provider

        assert callable(detect_llm_provider)

    def test_provider_detection_result_importable(self) -> None:
        """ProviderDetectionResult should still be importable from interactive."""
        from markitai.cli.interactive import (
            ProviderDetectionResult as InteractiveResult,
        )
        from markitai.cli.providers_detect import ProviderDetectionResult

        assert InteractiveResult is ProviderDetectionResult

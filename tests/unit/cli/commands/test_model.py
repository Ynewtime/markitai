"""Tests for model command."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from markit.cli.commands.model import (
    _get_models_sync,
    model_app,
)


class TestGetModelsSync:
    """Tests for _get_models_sync function."""

    def test_get_openai_models(self):
        """Test getting OpenAI models."""
        mock_model = MagicMock()
        mock_model.id = "gpt-4"
        mock_model.created = 1234567890
        mock_model.owned_by = "openai"

        mock_result = MagicMock()
        mock_result.data = [mock_model]

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_result
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("openai", "test-key", None)

            assert len(models) == 1
            assert models[0]["id"] == "gpt-4"
            assert models[0]["owned_by"] == "openai"

    def test_get_anthropic_models(self):
        """Test getting Anthropic models."""
        mock_model = MagicMock()
        mock_model.id = "claude-3-opus"
        mock_model.display_name = "Claude 3 Opus"
        mock_model.created_at = None

        mock_result = MagicMock()
        mock_result.data = [mock_model]

        with patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_result
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("anthropic", "test-key", None)

            assert len(models) == 1
            assert models[0]["id"] == "claude-3-opus"
            assert models[0]["display_name"] == "Claude 3 Opus"

    def test_get_gemini_models(self):
        """Test getting Gemini models."""
        mock_model = MagicMock()
        mock_model.name = "models/gemini-pro"
        mock_model.display_name = "Gemini Pro"
        mock_model.description = "A powerful model"

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = [mock_model]
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("gemini", "test-key", None)

            assert len(models) == 1
            assert models[0]["id"] == "models/gemini-pro"
            assert models[0]["display_name"] == "Gemini Pro"

    def test_get_openrouter_models(self):
        """Test getting OpenRouter models."""
        mock_model = MagicMock()
        mock_model.id = "anthropic/claude-3"
        mock_model.created = 1234567890
        mock_model.owned_by = "anthropic"

        mock_result = MagicMock()
        mock_result.data = [mock_model]

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_result
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("openrouter", "test-key", None)

            assert len(models) == 1
            assert models[0]["id"] == "anthropic/claude-3"
            # Verify OpenRouter base URL
            call_kwargs = mock_client_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_get_ollama_models(self):
        """Test getting Ollama models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 4700000000, "modified_at": "2024-01-01"},
            ]
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("ollama", None, None)

            assert len(models) == 1
            assert models[0]["id"] == "llama3:latest"
            assert models[0]["size"] == 4700000000

    def test_get_ollama_custom_host(self):
        """Test getting Ollama models with custom host."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=None)
            mock_client_cls.return_value = mock_client

            _get_models_sync("ollama", None, "http://custom:11434")

            mock_client.get.assert_called_with("http://custom:11434/api/tags")

    def test_get_models_unknown_provider(self):
        """Test getting models for unknown provider."""
        models = _get_models_sync("unknown", "key", None)
        assert models == []

    def test_get_models_no_api_key(self):
        """Test getting models without API key for providers that need it."""
        # For providers that require API key, empty list is returned
        models = _get_models_sync("openai", None, None)
        assert models == []


class TestModelCommands:
    """Integration tests for model commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_list_no_models(self, runner):
        """Test list command with no models configured."""
        mock_settings = MagicMock()
        mock_settings.llm.models = []

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert result.exit_code == 0
            assert "No models configured" in result.stdout

    def test_list_with_models(self, runner):
        """Test list command with models configured."""
        mock_model = MagicMock()
        mock_model.name = "gpt-4o"
        mock_model.model = "gpt-4o"
        mock_model.credential_id = "openai-main"
        mock_model.capabilities = ["text", "vision"]
        mock_model.timeout = 120

        mock_settings = MagicMock()
        mock_settings.llm.models = [mock_model]

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert result.exit_code == 0
            assert "gpt-4o" in result.stdout
            assert "openai-main" in result.stdout

    def test_add_no_credentials(self, runner):
        """Test add command with no credentials."""
        mock_settings = MagicMock()

        with (
            patch("markit.cli.commands.model.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.model.get_unique_credentials", return_value=[]),
        ):
            result = runner.invoke(model_app, ["add"], input="\n")
            assert result.exit_code == 1
            assert "No credentials configured" in result.stdout


class TestModelListDisplay:
    """Tests for model list display formatting."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_list_shows_capabilities(self, runner):
        """Test that list shows model capabilities."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.model = "test-model-id"
        mock_model.credential_id = "test-cred"
        mock_model.capabilities = ["text"]
        mock_model.timeout = 60

        mock_settings = MagicMock()
        mock_settings.llm.models = [mock_model]

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert "text" in result.stdout

    def test_list_shows_timeout(self, runner):
        """Test that list shows timeout."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.model = "test-model-id"
        mock_model.credential_id = "test-cred"
        mock_model.capabilities = None
        mock_model.timeout = 300

        mock_settings = MagicMock()
        mock_settings.llm.models = [mock_model]

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert "300s" in result.stdout

    def test_list_multiple_models(self, runner):
        """Test listing multiple models."""
        mock_models = []
        for i in range(3):
            m = MagicMock()
            m.name = f"model-{i}"
            m.model = f"model-id-{i}"
            m.credential_id = f"cred-{i}"
            m.capabilities = ["text"]
            m.timeout = 120
            mock_models.append(m)

        mock_settings = MagicMock()
        mock_settings.llm.models = mock_models

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert result.exit_code == 0
            assert "model-0" in result.stdout
            assert "model-1" in result.stdout
            assert "model-2" in result.stdout
            assert "Total: 3" in result.stdout


class TestCapabilitiesInference:
    """Tests for capability inference in model display."""

    def test_infer_vision_capabilities(self):
        """Test that vision models are tagged correctly."""
        from markit.utils.capabilities import infer_capabilities

        # Models with 'vision' in name should have vision capability
        caps = infer_capabilities("gpt-4o")
        assert "text" in caps
        assert "vision" in caps

    def test_infer_text_only(self):
        """Test that text-only models are tagged correctly."""
        from markit.utils.capabilities import infer_capabilities

        # DeepSeek chat is text-only
        caps = infer_capabilities("deepseek-chat")
        assert caps == ["text"]


class TestAddModelCommand:
    """Tests for add model command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_add_model_fetch_failure(self, runner):
        """Test add model when fetch fails."""
        mock_settings = MagicMock()
        creds = [("openai", "test-key", None, "OpenAI", "openai-cred")]

        with (
            patch("markit.cli.commands.model.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.model.get_unique_credentials", return_value=creds),
            patch(
                "markit.cli.commands.model._get_models_sync",
                side_effect=Exception("Connection failed"),
            ),
            patch("markit.cli.commands.model.questionary.select") as mock_select,
        ):
            mock_select.return_value.ask.return_value = ("openai", "test-key", None, "openai-cred")
            result = runner.invoke(model_app, ["add"])
            assert result.exit_code == 1
            assert "failed" in result.stdout

    def test_add_model_no_models_found(self, runner):
        """Test add model when no models are returned."""
        mock_settings = MagicMock()
        creds = [("openai", "test-key", None, "OpenAI", "openai-cred")]

        with (
            patch("markit.cli.commands.model.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.model.get_unique_credentials", return_value=creds),
            patch("markit.cli.commands.model._get_models_sync", return_value=[]),
            patch("markit.cli.commands.model.questionary.select") as mock_select,
        ):
            mock_select.return_value.ask.return_value = ("openai", "test-key", None, "openai-cred")
            result = runner.invoke(model_app, ["add"])
            assert result.exit_code == 1
            assert "no models found" in result.stdout

    def test_add_model_cancel_credential_selection(self, runner):
        """Test add model when user cancels credential selection."""
        mock_settings = MagicMock()
        creds = [("openai", "test-key", None, "OpenAI", "openai-cred")]

        with (
            patch("markit.cli.commands.model.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.model.get_unique_credentials", return_value=creds),
            patch("markit.cli.commands.model.questionary.select") as mock_select,
        ):
            mock_select.return_value.ask.return_value = None
            result = runner.invoke(model_app, ["add"])
            assert result.exit_code == 0  # Clean exit on cancel


class TestListModelsEdgeCases:
    """Edge case tests for list models command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_list_model_with_no_capabilities(self, runner):
        """Test list shows default capabilities when none specified."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.model = "test-model-id"
        mock_model.credential_id = "test-cred"
        mock_model.capabilities = None
        mock_model.timeout = 60

        mock_settings = MagicMock()
        mock_settings.llm.models = [mock_model]

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert result.exit_code == 0
            # When capabilities is None, it should display "text"
            assert "text" in result.stdout

    def test_list_model_with_vision_capabilities(self, runner):
        """Test list correctly shows vision capabilities."""
        mock_model = MagicMock()
        mock_model.name = "gpt-4o"
        mock_model.model = "gpt-4o"
        mock_model.credential_id = "openai-main"
        mock_model.capabilities = ["text", "vision"]
        mock_model.timeout = 120

        mock_settings = MagicMock()
        mock_settings.llm.models = [mock_model]

        with patch("markit.cli.commands.model.get_settings", return_value=mock_settings):
            result = runner.invoke(model_app, ["list"])
            assert result.exit_code == 0
            assert "vision" in result.stdout


class TestGetModelsSyncExtended:
    """Extended tests for _get_models_sync function."""

    def test_get_openai_models_with_base_url(self):
        """Test getting OpenAI models with custom base URL."""
        mock_model = MagicMock()
        mock_model.id = "deepseek-chat"
        mock_model.created = 1234567890
        mock_model.owned_by = "deepseek"

        mock_result = MagicMock()
        mock_result.data = [mock_model]

        with patch("openai.OpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_result
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("openai", "test-key", "https://api.deepseek.com")

            assert len(models) == 1
            mock_client_cls.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                timeout=30.0,
            )

    def test_get_anthropic_models_with_created_at(self):
        """Test getting Anthropic models with created_at date."""
        from datetime import datetime

        mock_model = MagicMock()
        mock_model.id = "claude-3-opus"
        mock_model.display_name = "Claude 3 Opus"
        mock_model.created_at = datetime(2024, 1, 1, 12, 0, 0)

        mock_result = MagicMock()
        mock_result.data = [mock_model]

        with patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_result
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("anthropic", "test-key", None)

            assert len(models) == 1
            assert models[0]["created_at"] is not None

    def test_get_gemini_models_with_long_description(self):
        """Test getting Gemini models with long description (truncated)."""
        mock_model = MagicMock()
        mock_model.name = "models/gemini-pro"
        mock_model.display_name = "Gemini Pro"
        mock_model.description = "x" * 200  # Long description

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = [mock_model]
            mock_client_cls.return_value = mock_client

            models = _get_models_sync("gemini", "test-key", None)

            assert len(models) == 1
            # Description should be truncated to 100 chars
            assert len(models[0]["description"]) == 100

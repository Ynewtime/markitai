"""Tests for provider command."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from markit.cli.commands.provider import (
    PROVIDER_TYPES,
    ProviderTestResult,
    display_test_results,
    get_cache_dir,
    get_models_cache_path,
    provider_app,
    test_all_providers,
)


class TestProviderTestResult:
    """Tests for ProviderTestResult dataclass."""

    def test_create_connected_result(self):
        """Test creating a connected result."""
        result = ProviderTestResult(
            provider="openai",
            status="connected",
            latency_ms=150.5,
            models_count=10,
        )
        assert result.provider == "openai"
        assert result.status == "connected"
        assert result.latency_ms == 150.5
        assert result.models_count == 10
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating a failed result."""
        result = ProviderTestResult(
            provider="anthropic",
            status="failed",
            latency_ms=500.0,
            error="Connection refused",
        )
        assert result.provider == "anthropic"
        assert result.status == "failed"
        assert result.error == "Connection refused"

    def test_create_skipped_result(self):
        """Test creating a skipped result."""
        result = ProviderTestResult(
            provider="gemini",
            status="skipped",
            error="No API key configured",
        )
        assert result.status == "skipped"


class TestProviderTypes:
    """Tests for PROVIDER_TYPES constant."""

    def test_all_providers_defined(self):
        """Test that all expected providers are defined."""
        expected = ["openai", "openai_compatible", "anthropic", "gemini", "openrouter", "ollama"]
        for provider in expected:
            assert provider in PROVIDER_TYPES

    def test_provider_has_display(self):
        """Test each provider has a display name."""
        for _provider, info in PROVIDER_TYPES.items():
            assert "display" in info
            assert isinstance(info["display"], str)

    def test_openai_compatible_has_actual_provider(self):
        """Test openai_compatible maps to openai."""
        assert PROVIDER_TYPES["openai_compatible"]["actual_provider"] == "openai"

    def test_ollama_no_api_key(self):
        """Test ollama doesn't require API key."""
        assert PROVIDER_TYPES["ollama"]["default_api_key_env"] is None


class TestCacheFunctions:
    """Tests for cache-related functions."""

    def test_get_cache_dir_creates_dir(self, tmp_path):
        """Test get_cache_dir creates directory."""
        with patch("markit.cli.commands.provider.CACHE_DIR", tmp_path / "cache"):
            result = get_cache_dir()
            assert result.exists()
            assert result.is_dir()

    def test_get_models_cache_path(self, tmp_path):
        """Test get_models_cache_path returns correct path."""
        with patch("markit.cli.commands.provider.CACHE_DIR", tmp_path / "cache"):
            result = get_models_cache_path()
            assert result.name == "models.json"


class TestTestOpenAI:
    """Tests for _test_openai function."""

    @pytest.mark.asyncio
    async def test_openai_success(self):
        """Test successful OpenAI connectivity test."""
        from markit.cli.commands.provider import _test_openai

        mock_models = MagicMock()
        mock_models.data = [MagicMock() for _ in range(5)]

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            result = await _test_openai("test-key")

            assert result.status == "connected"
            assert result.models_count == 5
            assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_openai_failure(self):
        """Test OpenAI test failure."""
        from markit.cli.commands.provider import _test_openai

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(side_effect=Exception("Auth failed"))
            mock_client_cls.return_value = mock_client

            result = await _test_openai("invalid-key")

            assert result.status == "failed"
            assert result.error is not None and "Auth failed" in result.error


class TestTestAnthropic:
    """Tests for _test_anthropic function."""

    @pytest.mark.asyncio
    async def test_anthropic_success(self):
        """Test successful Anthropic connectivity test."""
        from markit.cli.commands.provider import _test_anthropic

        mock_models = MagicMock()
        mock_models.data = [MagicMock() for _ in range(3)]

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            result = await _test_anthropic("test-key")

            assert result.status == "connected"
            assert result.models_count == 3

    @pytest.mark.asyncio
    async def test_anthropic_failure(self):
        """Test Anthropic test failure."""
        from markit.cli.commands.provider import _test_anthropic

        with patch("anthropic.AsyncAnthropic") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(side_effect=Exception("Invalid API key"))
            mock_client_cls.return_value = mock_client

            result = await _test_anthropic("invalid-key")

            assert result.status == "failed"
            assert result.error is not None and "Invalid API key" in result.error


class TestTestGemini:
    """Tests for _test_gemini function."""

    @pytest.mark.asyncio
    async def test_gemini_success(self):
        """Test successful Gemini connectivity test."""
        from markit.cli.commands.provider import _test_gemini

        mock_models = [MagicMock() for _ in range(10)]

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.return_value = iter(mock_models)
            mock_client_cls.return_value = mock_client

            result = await _test_gemini("test-key")

            assert result.status == "connected"
            assert result.models_count == 10

    @pytest.mark.asyncio
    async def test_gemini_failure(self):
        """Test Gemini test failure."""
        from markit.cli.commands.provider import _test_gemini

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.list.side_effect = Exception("API error")
            mock_client_cls.return_value = mock_client

            result = await _test_gemini("invalid-key")

            assert result.status == "failed"
            assert result.error is not None and "API error" in result.error


class TestTestOpenRouter:
    """Tests for _test_openrouter function."""

    @pytest.mark.asyncio
    async def test_openrouter_success(self):
        """Test successful OpenRouter connectivity test."""
        from markit.cli.commands.provider import _test_openrouter

        mock_models = MagicMock()
        mock_models.data = [MagicMock() for _ in range(100)]

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            result = await _test_openrouter("test-key")

            assert result.status == "connected"
            assert result.models_count == 100

    @pytest.mark.asyncio
    async def test_openrouter_failure(self):
        """Test OpenRouter test failure."""
        from markit.cli.commands.provider import _test_openrouter

        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(side_effect=Exception("Rate limited"))
            mock_client_cls.return_value = mock_client

            result = await _test_openrouter("test-key")

            assert result.status == "failed"
            assert result.error is not None and "Rate limited" in result.error


class TestTestOllama:
    """Tests for _test_ollama function."""

    @pytest.mark.asyncio
    async def test_ollama_success(self):
        """Test successful Ollama connectivity test."""
        from markit.cli.commands.provider import _test_ollama

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()

            # Mock health check response
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.text = "Ollama is running"

            # Mock tags response
            mock_tags_response = MagicMock()
            mock_tags_response.raise_for_status = MagicMock()
            mock_tags_response.json.return_value = {"models": [{"name": "llama3"}]}

            mock_client.get = AsyncMock(side_effect=[mock_health_response, mock_tags_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await _test_ollama()

            assert result.status == "connected"
            assert result.models_count == 1

    @pytest.mark.asyncio
    async def test_ollama_connection_refused(self):
        """Test Ollama when connection is refused."""
        import httpx

        from markit.cli.commands.provider import _test_ollama

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await _test_ollama()

            assert result.status == "failed"
            assert result.error is not None and "ollama serve" in result.error.lower()


class TestTestAllProviders:
    """Tests for test_all_providers function."""

    @pytest.mark.asyncio
    async def test_all_providers_no_creds(self):
        """Test with no credentials configured."""
        mock_settings = MagicMock()

        with patch("markit.cli.commands.provider.get_unique_credentials", return_value=[]):
            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert results == []

    @pytest.mark.asyncio
    async def test_all_providers_skipped_no_key(self):
        """Test that providers without API key are skipped."""
        mock_settings = MagicMock()
        creds = [("openai", None, None, "OpenAI", "openai-cred")]

        with patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds):
            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "skipped"
            assert results[0].error is not None and "No API key" in results[0].error


class TestDisplayTestResults:
    """Tests for display_test_results function."""

    def test_display_results(self):
        """Test displaying test results."""
        results = [
            ProviderTestResult(
                provider="openai",
                name="OpenAI",
                status="connected",
                latency_ms=100,
                models_count=5,
            ),
            ProviderTestResult(
                provider="anthropic",
                name="Anthropic",
                status="failed",
                latency_ms=500,
                error="Auth failed",
            ),
        ]

        with patch("markit.cli.commands.provider.console") as mock_console:
            display_test_results(results)
            mock_console.print.assert_called()


class TestProviderCommands:
    """Integration tests for provider commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_list_no_credentials(self, runner):
        """Test list command with no credentials."""
        mock_settings = MagicMock()
        mock_settings.llm.credentials = []

        with patch("markit.cli.commands.provider.get_settings", return_value=mock_settings):
            result = runner.invoke(provider_app, ["list"])
            assert result.exit_code == 0
            assert "No credentials configured" in result.stdout

    def test_list_with_credentials(self, runner):
        """Test list command with credentials."""
        mock_cred = MagicMock()
        mock_cred.id = "test-openai"
        mock_cred.provider = "openai"
        mock_cred.base_url = None
        mock_cred.api_key_env = "OPENAI_API_KEY"

        mock_settings = MagicMock()
        mock_settings.llm.credentials = [mock_cred]

        with patch("markit.cli.commands.provider.get_settings", return_value=mock_settings):
            result = runner.invoke(provider_app, ["list"])
            assert result.exit_code == 0
            assert "test-openai" in result.stdout

    def test_add_invalid_provider(self, runner):
        """Test add command with invalid provider."""
        result = runner.invoke(
            provider_app,
            ["add", "--provider", "invalid", "--id", "test"],
        )
        assert result.exit_code == 1
        assert "Invalid provider type" in result.stdout

    def test_add_missing_id(self, runner):
        """Test add command without ID."""
        result = runner.invoke(
            provider_app,
            ["add", "--provider", "openai"],
        )
        assert result.exit_code == 1
        assert "--id is required" in result.stdout


class TestWriteCredentialToConfig:
    """Tests for _write_credential_to_config function."""

    def test_write_missing_config(self):
        """Test writing when config file doesn't exist."""
        from markit.cli.commands.provider import _write_credential_to_config

        with patch("markit.cli.commands.provider.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            result = _write_credential_to_config({"id": "test"})
            assert result is False

    def test_write_duplicate_id(self, tmp_path):
        """Test writing duplicate credential ID."""
        from markit.cli.commands.provider import _write_credential_to_config

        config_file = tmp_path / "markit.yaml"
        config_file.write_text(
            """
llm:
  credentials:
    - id: existing
      provider: openai
"""
        )

        with patch("markit.cli.commands.provider.Path", return_value=config_file):
            result = _write_credential_to_config({"id": "existing", "provider": "openai"})
            assert result is False


class TestProviderTestCommand:
    """Tests for provider test command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_test_command_success(self, runner):
        """Test provider test command with successful results."""
        mock_results = [
            ProviderTestResult(
                provider="openai",
                name="OpenAI",
                status="connected",
                latency_ms=100,
                models_count=5,
            )
        ]

        with (
            patch("markit.cli.commands.provider.asyncio.run", return_value=mock_results),
            patch("markit.cli.commands.provider.display_test_results"),
        ):
            result = runner.invoke(provider_app, ["test"])
            assert result.exit_code == 0

    def test_test_command_with_failures(self, runner):
        """Test provider test command with some failures."""
        mock_results = [
            ProviderTestResult(
                provider="openai",
                name="OpenAI",
                status="connected",
                latency_ms=100,
            ),
            ProviderTestResult(
                provider="anthropic",
                name="Anthropic",
                status="failed",
                error="Auth failed",
            ),
        ]

        with (
            patch("markit.cli.commands.provider.asyncio.run", return_value=mock_results),
            patch("markit.cli.commands.provider.display_test_results"),
        ):
            result = runner.invoke(provider_app, ["test"])
            assert result.exit_code == 1

    def test_test_command_verbose(self, runner):
        """Test provider test command with verbose option."""
        mock_results = [
            ProviderTestResult(
                provider="openai",
                name="OpenAI",
                status="connected",
                latency_ms=100,
            )
        ]

        with (
            patch("markit.cli.commands.provider.asyncio.run", return_value=mock_results),
            patch("markit.cli.commands.provider.display_test_results"),
            patch("markit.utils.logging.setup_logging") as mock_logging,
        ):
            result = runner.invoke(provider_app, ["test", "-v"])
            assert result.exit_code == 0
            mock_logging.assert_called_once_with(level="DEBUG")


class TestProviderFetchCommand:
    """Tests for provider fetch command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_fetch_no_cache(self, runner):
        """Test fetch command with --no-cache option."""
        mock_settings = MagicMock()

        with (
            patch("markit.cli.commands.provider.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=[]),
        ):
            result = runner.invoke(provider_app, ["fetch", "--no-cache"])
            assert result.exit_code == 0

    def test_fetch_with_provider_filter(self, runner):
        """Test fetch command with provider filter."""
        mock_settings = MagicMock()
        creds = [
            ("openai", "test-key", None, "OpenAI", "openai-cred"),
            ("anthropic", "test-key2", None, "Anthropic", "anthropic-cred"),
        ]

        with (
            patch("markit.cli.commands.provider.get_settings", return_value=mock_settings),
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("markit.cli.commands.provider._get_openai_models") as mock_get_openai,
        ):
            mock_get_openai.return_value = [{"id": "gpt-4o", "owned_by": "openai"}]
            runner.invoke(provider_app, ["fetch", "--provider", "openai", "--no-cache"])
            # Only OpenAI should be fetched
            mock_get_openai.assert_called_once()


class TestTestAllProvidersExtended:
    """Extended tests for test_all_providers function."""

    @pytest.mark.asyncio
    async def test_unknown_provider(self):
        """Test handling of unknown provider type."""
        mock_settings = MagicMock()
        creds = [("unknown_provider", "key", None, "Unknown", "cred")]

        with patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds):
            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "skipped"
            assert results[0].error is not None and "Unknown provider" in results[0].error

    @pytest.mark.asyncio
    async def test_openai_with_base_url(self):
        """Test OpenAI provider with custom base URL."""
        mock_settings = MagicMock()
        creds = [
            ("openai", "test-key", "https://custom.openai.com/v1", "DeepSeek", "deepseek-cred")
        ]

        mock_models = MagicMock()
        mock_models.data = [MagicMock()]

        with (
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("openai.AsyncOpenAI") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "connected"
            # Verify base_url was passed
            mock_client_cls.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.openai.com/v1",
                timeout=10.0,
            )

    @pytest.mark.asyncio
    async def test_anthropic_provider(self):
        """Test Anthropic provider test."""
        mock_settings = MagicMock()
        creds = [("anthropic", "test-key", None, "Anthropic", "anthropic-cred")]

        mock_models = MagicMock()
        mock_models.data = [MagicMock()]

        with (
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("anthropic.AsyncAnthropic") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "connected"

    @pytest.mark.asyncio
    async def test_gemini_provider(self):
        """Test Gemini provider test."""
        mock_settings = MagicMock()
        creds = [("gemini", "test-key", None, "Gemini", "gemini-cred")]

        mock_models = [MagicMock()]

        with (
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("google.genai.Client") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = iter(mock_models)
            mock_client_cls.return_value = mock_client

            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "connected"

    @pytest.mark.asyncio
    async def test_openrouter_provider(self):
        """Test OpenRouter provider test."""
        mock_settings = MagicMock()
        creds = [("openrouter", "test-key", None, "OpenRouter", "openrouter-cred")]

        mock_models = MagicMock()
        mock_models.data = [MagicMock()]

        with (
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("openai.AsyncOpenAI") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models)
            mock_client_cls.return_value = mock_client

            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "connected"

    @pytest.mark.asyncio
    async def test_ollama_provider(self):
        """Test Ollama provider test."""
        mock_settings = MagicMock()
        creds = [("ollama", None, "http://localhost:11434", "Ollama", "ollama-cred")]

        with (
            patch("markit.cli.commands.provider.get_unique_credentials", return_value=creds),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.text = "Ollama is running"
            mock_tags_response = MagicMock()
            mock_tags_response.raise_for_status = MagicMock()
            mock_tags_response.json.return_value = {"models": [{"name": "llama3"}]}
            mock_client.get = AsyncMock(side_effect=[mock_health_response, mock_tags_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            results = await test_all_providers(settings=mock_settings, show_progress=False)

            assert len(results) == 1
            assert results[0].status == "connected"


class TestDisplayTestResultsExtended:
    """Extended tests for display_test_results function."""

    def test_display_results_with_long_error(self):
        """Test displaying results with long error message."""
        results = [
            ProviderTestResult(
                provider="openai",
                name="OpenAI",
                status="failed",
                latency_ms=100,
                error="x" * 100,  # Long error message
            )
        ]

        with patch("markit.cli.commands.provider.console") as mock_console:
            display_test_results(results)
            mock_console.print.assert_called()

    def test_display_results_with_no_latency(self):
        """Test displaying results with no latency."""
        results = [
            ProviderTestResult(
                provider="gemini",
                name="Gemini",
                status="skipped",
            )
        ]

        with patch("markit.cli.commands.provider.console") as mock_console:
            display_test_results(results)
            mock_console.print.assert_called()

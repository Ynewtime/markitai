from unittest.mock import AsyncMock, patch

import pytest

from markit.config.settings import (
    LLMConfig,
    LLMCredentialConfig,
    LLMModelConfig,
    LLMProviderConfig,
)
from markit.exceptions import LLMError, ProviderNotFoundError
from markit.llm.base import LLMResponse, TokenUsage
from markit.llm.manager import ProviderManager, ProviderState


@pytest.fixture
def mock_providers():
    p1 = AsyncMock()
    p1.complete.return_value = LLMResponse(
        content="response1",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        model="gpt-4",
        finish_reason="stop",
    )
    p1.analyze_image.return_value = LLMResponse(
        content="analysis1",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        model="gpt-4",
        finish_reason="stop",
    )
    p2 = AsyncMock()
    p2.complete.return_value = LLMResponse(
        content="response2",
        usage=TokenUsage(prompt_tokens=15, completion_tokens=25),
        model="claude",
        finish_reason="stop",
    )
    p2.analyze_image.return_value = LLMResponse(
        content="analysis2",
        usage=TokenUsage(prompt_tokens=120, completion_tokens=60),
        model="claude",
        finish_reason="stop",
    )
    return {"openai_gpt-4": p1, "anthropic_claude": p2}


@pytest.fixture
def provider_manager(mock_providers):
    configs = [
        LLMProviderConfig(provider="openai", model="gpt-4", api_key="sk-123"),
        LLMProviderConfig(provider="anthropic", model="claude", api_key="sk-456"),
    ]
    pm = ProviderManager(configs)

    # Manually set up provider states to bypass initialization logic
    # The new architecture uses _provider_states with ProviderState objects
    pm._provider_states = {
        "openai_gpt-4": ProviderState(
            config=configs[0],
            capabilities=["text", "vision"],
            provider=mock_providers["openai_gpt-4"],
            initialized=True,
            valid=True,
        ),
        "anthropic_claude": ProviderState(
            config=configs[1],
            capabilities=["text", "vision"],
            provider=mock_providers["anthropic_claude"],
            initialized=True,
            valid=True,
        ),
    }
    # Also populate _providers dict (used by complete_with_fallback)
    pm._providers = {
        "openai_gpt-4": mock_providers["openai_gpt-4"],
        "anthropic_claude": mock_providers["anthropic_claude"],
    }
    pm._valid_providers = ["openai_gpt-4", "anthropic_claude"]
    pm._provider_capabilities = {
        "openai_gpt-4": ["text", "vision"],
        "anthropic_claude": ["text", "vision"],
    }
    pm._config_loaded = True
    pm._initialized = True
    pm._current_index = 0

    return pm


@pytest.mark.asyncio
async def test_round_robin_load_balancing(provider_manager, mock_providers):
    """Test that requests rotate through providers."""
    # 1st call -> openai
    await provider_manager.complete_with_fallback([])
    assert mock_providers["openai_gpt-4"].complete.called
    assert not mock_providers["anthropic_claude"].complete.called

    mock_providers["openai_gpt-4"].complete.reset_mock()

    # 2nd call -> anthropic
    await provider_manager.complete_with_fallback([])
    assert mock_providers["anthropic_claude"].complete.called
    assert not mock_providers["openai_gpt-4"].complete.called

    mock_providers["anthropic_claude"].complete.reset_mock()

    # 3rd call -> openai (wrap around)
    await provider_manager.complete_with_fallback([])
    assert mock_providers["openai_gpt-4"].complete.called
    assert not mock_providers["anthropic_claude"].complete.called


@pytest.mark.asyncio
async def test_fallback_logic(provider_manager, mock_providers):
    """Test fallback when primary provider fails."""
    # Make openai fail
    mock_providers["openai_gpt-4"].complete.side_effect = Exception("OpenAI Error")

    # Call (starts at index 0 -> openai)
    # OpenAI fails, should fallback to Anthropic
    await provider_manager.complete_with_fallback([])

    assert mock_providers["openai_gpt-4"].complete.called
    assert mock_providers["anthropic_claude"].complete.called


@pytest.mark.asyncio
async def test_all_providers_fail(provider_manager, mock_providers):
    """Test error raised when all providers fail."""
    mock_providers["openai_gpt-4"].complete.side_effect = Exception("Error 1")
    mock_providers["anthropic_claude"].complete.side_effect = Exception("Error 2")

    with pytest.raises(LLMError) as exc:
        await provider_manager.complete_with_fallback([])

    assert "All providers failed" in str(exc.value)
    assert "Error 1" in str(exc.value)
    assert "Error 2" in str(exc.value)


@pytest.mark.asyncio
async def test_image_analysis_round_robin(provider_manager, mock_providers):
    """Test round robin for image analysis sharing same counter."""
    # Reset index manually for predictability
    provider_manager._current_index = 0

    # 1st call -> openai
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called
    assert not mock_providers["anthropic_claude"].analyze_image.called

    mock_providers["openai_gpt-4"].analyze_image.reset_mock()

    # 2nd call -> anthropic
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["anthropic_claude"].analyze_image.called
    assert not mock_providers["openai_gpt-4"].analyze_image.called


class TestProviderManagerInit:
    """Tests for ProviderManager initialization."""

    def test_init_with_legacy_list(self):
        """Test initialization with legacy list of configs."""
        configs = [
            LLMProviderConfig(provider="openai", model="gpt-4", api_key="sk-123"),
        ]
        pm = ProviderManager(configs)

        assert isinstance(pm.config, LLMConfig)
        assert len(pm.config.providers) == 1
        assert pm.config.providers[0].provider == "openai"

    def test_init_with_llm_config(self):
        """Test initialization with LLMConfig object."""
        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="openai-main", provider="openai", api_key="sk-123"),
            ],
            models=[
                LLMModelConfig(name="GPT-4o", model="gpt-4o", credential_id="openai-main"),
            ],
        )
        pm = ProviderManager(llm_config)

        assert pm.config is llm_config
        assert len(pm.config.credentials) == 1
        assert len(pm.config.models) == 1

    def test_init_with_none(self):
        """Test initialization with None creates empty config."""
        pm = ProviderManager(None)

        assert isinstance(pm.config, LLMConfig)
        assert len(pm.config.providers) == 0
        assert len(pm.config.credentials) == 0
        assert len(pm.config.models) == 0


class TestProviderManagerNewSchema:
    """Tests for ProviderManager with new credentials + models schema."""

    @pytest.mark.asyncio
    async def test_initialize_with_new_schema(self):
        """Test initializing providers from credentials + models schema."""
        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="openai-main", provider="openai", api_key="sk-test"),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="openai-main",
                    capabilities=["text", "vision"],
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        with (
            patch.object(pm, "_create_provider", return_value=AsyncMock()) as mock_create,
            patch.object(pm, "_validate_provider", return_value=True),
        ):
            await pm.initialize()

            # Verify provider was created
            assert mock_create.called
            # Verify model is registered
            assert "GPT-4o" in pm._valid_providers
            assert pm._provider_capabilities["GPT-4o"] == ["text", "vision"]

    @pytest.mark.asyncio
    async def test_initialize_skips_missing_credential(self):
        """Test that models with missing credentials are skipped."""
        llm_config = LLMConfig(
            credentials=[],  # No credentials
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="nonexistent",
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        await pm.initialize()

        # Model should be skipped
        assert "GPT-4o" not in pm._valid_providers
        assert pm._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_mixed_legacy_and_new(self):
        """Test initializing with both legacy and new schema."""
        llm_config = LLMConfig(
            providers=[
                LLMProviderConfig(
                    provider="ollama",
                    model="llama3.2-vision",
                    name="Ollama-LLaMA",
                    capabilities=["text", "vision"],
                ),
            ],
            credentials=[
                LLMCredentialConfig(id="openai-main", provider="openai", api_key="sk-test"),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="openai-main",
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        with (
            patch.object(pm, "_create_provider", return_value=AsyncMock()),
            patch.object(pm, "_validate_provider", return_value=True),
        ):
            await pm.initialize()

            # Both should be registered
            assert "Ollama-LLaMA" in pm._valid_providers
            assert "GPT-4o" in pm._valid_providers

    @pytest.mark.asyncio
    async def test_capabilities_default_to_optimistic(self):
        """Test that capabilities default to ['text', 'vision'] if not specified."""
        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="openai-main", provider="openai", api_key="sk-test"),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="openai-main",
                    # capabilities not specified
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        with (
            patch.object(pm, "_create_provider", return_value=AsyncMock()),
            patch.object(pm, "_validate_provider", return_value=True),
        ):
            await pm.initialize()

            # Should default to optimistic capabilities
            assert pm._provider_capabilities["GPT-4o"] == ["text", "vision"]


class TestProviderManagerMethods:
    """Tests for ProviderManager methods."""

    def test_get_default_no_providers(self):
        """Test get_default raises when no providers available."""
        pm = ProviderManager(None)
        pm._initialized = True

        with pytest.raises(ProviderNotFoundError) as exc:
            pm.get_default()

        assert "No valid LLM provider available" in str(exc.value)

    def test_get_provider_not_found(self):
        """Test get_provider raises for unknown provider."""
        pm = ProviderManager(None)
        pm._initialized = True

        with pytest.raises(ProviderNotFoundError) as exc:
            pm.get_provider("nonexistent")

        assert "not available" in str(exc.value)

    def test_has_capability(self, provider_manager):
        """Test has_capability correctly checks provider capabilities."""
        assert provider_manager.has_capability("text") is True
        assert provider_manager.has_capability("vision") is True
        assert provider_manager.has_capability("embedding") is False

    def test_available_providers(self, provider_manager):
        """Test available_providers returns copy of list."""
        providers = provider_manager.available_providers

        assert providers == ["openai_gpt-4", "anthropic_claude"]
        # Ensure it's a copy
        providers.append("test")
        assert provider_manager.available_providers == ["openai_gpt-4", "anthropic_claude"]

    def test_has_providers(self, provider_manager):
        """Test has_providers property."""
        assert provider_manager.has_providers is True

        empty_pm = ProviderManager(None)
        empty_pm._initialized = True
        assert empty_pm.has_providers is False


class TestProviderManagerApiKeyResolution:
    """Tests for API key resolution logic."""

    def test_get_api_key_from_env_openai(self):
        """Test API key resolution for OpenAI."""
        pm = ProviderManager(None)

        with patch("os.environ.get") as mock_env:
            mock_env.return_value = "sk-openai-key"
            key = pm._get_api_key_from_env("openai")

            mock_env.assert_called_with("OPENAI_API_KEY")
            assert key == "sk-openai-key"

    def test_get_api_key_from_env_anthropic(self):
        """Test API key resolution for Anthropic."""
        pm = ProviderManager(None)

        with patch("os.environ.get") as mock_env:
            mock_env.return_value = "sk-anthropic-key"
            key = pm._get_api_key_from_env("anthropic")

            mock_env.assert_called_with("ANTHROPIC_API_KEY")
            assert key == "sk-anthropic-key"

    def test_get_api_key_from_env_gemini(self):
        """Test API key resolution for Gemini."""
        pm = ProviderManager(None)

        with patch("os.environ.get") as mock_env:
            mock_env.return_value = "gemini-key"
            key = pm._get_api_key_from_env("gemini")

            mock_env.assert_called_with("GOOGLE_API_KEY")
            assert key == "gemini-key"

    def test_get_api_key_from_env_openrouter(self):
        """Test API key resolution for OpenRouter."""
        pm = ProviderManager(None)

        with patch("os.environ.get") as mock_env:
            mock_env.return_value = "or-key"
            key = pm._get_api_key_from_env("openrouter")

            mock_env.assert_called_with("OPENROUTER_API_KEY")
            assert key == "or-key"

    def test_get_api_key_from_env_ollama(self):
        """Test API key resolution for Ollama (no key needed)."""
        pm = ProviderManager(None)
        key = pm._get_api_key_from_env("ollama")

        assert key is None

    def test_get_api_key_from_env_unknown_provider(self):
        """Test API key resolution for unknown provider."""
        pm = ProviderManager(None)
        key = pm._get_api_key_from_env("unknown")

        assert key is None

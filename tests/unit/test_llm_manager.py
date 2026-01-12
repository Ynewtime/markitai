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
    pm._configs_loaded = True
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
async def test_image_analysis_prioritizes_last_successful(provider_manager, mock_providers):
    """Test that image analysis prioritizes last successful provider for same capability.

    This behavior was changed from round-robin to improve performance and reliability.
    The first successful call sets the preferred provider for subsequent calls.
    """
    # Reset state for predictability
    provider_manager._current_index = 0
    provider_manager._last_successful_provider.clear()

    # 1st call -> openai (round-robin starts with openai)
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called
    assert not mock_providers["anthropic_claude"].analyze_image.called

    mock_providers["openai_gpt-4"].analyze_image.reset_mock()

    # 2nd call -> still openai (prioritizes last successful provider)
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called
    assert not mock_providers["anthropic_claude"].analyze_image.called


@pytest.mark.asyncio
async def test_image_analysis_falls_back_on_failure(provider_manager, mock_providers):
    """Test that image analysis falls back when preferred provider fails."""
    # Reset state for predictability
    provider_manager._current_index = 0
    provider_manager._last_successful_provider.clear()

    # 1st call -> openai succeeds, sets as preferred
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called

    # Reset and make openai fail
    mock_providers["openai_gpt-4"].analyze_image.reset_mock()
    mock_providers["openai_gpt-4"].analyze_image.side_effect = Exception("Provider failed")

    # 2nd call -> openai fails, falls back to anthropic
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called  # Tried first
    assert mock_providers["anthropic_claude"].analyze_image.called  # Fallback succeeded


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


class TestProviderManagerRedundantValidation:
    """Tests for preventing redundant validation of shared credentials."""

    @pytest.mark.asyncio
    async def test_initialize_shared_credential_validates_once(self):
        """Test that validation happens only once per credential."""
        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="shared-cred", provider="openai", api_key="sk-test"),
            ],
            models=[
                LLMModelConfig(
                    name="Model-A",
                    model="gpt-4o",
                    credential_id="shared-cred",
                ),
                LLMModelConfig(
                    name="Model-B",
                    model="gpt-4o-mini",
                    credential_id="shared-cred",
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        # Mock create_provider and validate_provider
        mock_provider = AsyncMock()
        mock_provider.validate.return_value = True

        # We need to mock _create_provider to return our mock
        with (
            patch.object(pm, "_create_provider", return_value=mock_provider) as mock_create,
            patch.object(pm, "_validate_provider", return_value=True) as mock_validate,
        ):
            await pm.initialize()

            # Verify providers created (should be 2, one for each model)
            assert mock_create.call_count == 2

            # Verify validation called only ONCE (for the first model that used the credential)
            assert mock_validate.call_count == 1

            # Both models should be valid
            assert "Model-A" in pm._valid_providers
            assert "Model-B" in pm._valid_providers


class TestProviderManagerConcurrentCredentialValidation:
    """Tests for concurrent credential validation (credential-level locking)."""

    @pytest.mark.asyncio
    async def test_concurrent_init_same_credential_validates_once(self):
        """Test that concurrent initialization with same credential validates only once.

        This test verifies the credential-level locking fix: when multiple models
        sharing the same credential are initialized concurrently (via asyncio.gather),
        the validation should happen exactly once, not multiple times.
        """
        import asyncio

        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="shared-openai", provider="openai", api_key="sk-test"),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="shared-openai",
                ),
                LLMModelConfig(
                    name="GPT-4o-mini",
                    model="gpt-4o-mini",
                    credential_id="shared-openai",
                ),
                LLMModelConfig(
                    name="GPT-5",
                    model="gpt-5",
                    credential_id="shared-openai",
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        # Track validation calls
        validation_call_count = 0
        validation_lock = asyncio.Lock()

        async def mock_validate(_provider, _config):
            nonlocal validation_call_count
            # Add small delay to increase chance of race conditions
            await asyncio.sleep(0.01)
            async with validation_lock:
                validation_call_count += 1
            return True

        mock_provider = AsyncMock()
        mock_provider.validate.return_value = True

        with (
            patch.object(pm, "_create_provider", return_value=mock_provider),
            patch.object(pm, "_validate_provider", side_effect=mock_validate),
        ):
            # Load configs first
            await pm._load_configs()

            # Concurrently initialize all three models
            results = await asyncio.gather(
                pm._ensure_provider_initialized("GPT-4o"),
                pm._ensure_provider_initialized("GPT-4o-mini"),
                pm._ensure_provider_initialized("GPT-5"),
            )

            # All should succeed
            assert all(results), "All models should initialize successfully"

            # Validation should be called exactly ONCE (credential-level lock ensures this)
            assert validation_call_count == 1, (
                f"Expected 1 validation call, got {validation_call_count}. "
                "Credential-level locking may not be working."
            )

            # All models should be valid
            assert "GPT-4o" in pm._valid_providers
            assert "GPT-4o-mini" in pm._valid_providers
            assert "GPT-5" in pm._valid_providers

    @pytest.mark.asyncio
    async def test_concurrent_init_different_credentials_validates_each(self):
        """Test that different credentials are validated separately.

        When models use different credentials, each credential should be
        validated independently.
        """
        import asyncio

        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="openai-cred", provider="openai", api_key="sk-openai"),
                LLMCredentialConfig(
                    id="anthropic-cred", provider="anthropic", api_key="sk-anthropic"
                ),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="openai-cred",
                ),
                LLMModelConfig(
                    name="Claude-Sonnet",
                    model="claude-sonnet",
                    credential_id="anthropic-cred",
                ),
            ],
        )
        pm = ProviderManager(llm_config)

        # Track validation calls per credential
        validation_calls: dict[str, int] = {}
        validation_lock = asyncio.Lock()

        async def mock_validate(_provider, config):
            await asyncio.sleep(0.01)
            async with validation_lock:
                cred_type = config.provider
                validation_calls[cred_type] = validation_calls.get(cred_type, 0) + 1
            return True

        mock_provider = AsyncMock()

        with (
            patch.object(pm, "_create_provider", return_value=mock_provider),
            patch.object(pm, "_validate_provider", side_effect=mock_validate),
        ):
            await pm._load_configs()

            # Concurrently initialize both models
            results = await asyncio.gather(
                pm._ensure_provider_initialized("GPT-4o"),
                pm._ensure_provider_initialized("Claude-Sonnet"),
            )

            assert all(results)

            # Each credential should be validated once
            assert validation_calls.get("openai", 0) == 1
            assert validation_calls.get("anthropic", 0) == 1

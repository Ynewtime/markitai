from unittest.mock import AsyncMock, patch

import pytest

from markit.config.settings import LLMProviderConfig
from markit.exceptions import LLMError
from markit.llm.base import LLMResponse, TokenUsage
from markit.llm.manager import ProviderManager, ProviderState


@pytest.fixture
def mock_providers():
    p1 = AsyncMock()  # Vision capable
    p1.analyze_image.return_value = LLMResponse(
        content="vision_result",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        model="gpt-4",
        finish_reason="stop",
    )
    p2 = AsyncMock()  # Text only
    p2.analyze_image.side_effect = Exception("Not supported")
    return {"openai_gpt-4": p1, "openai_deepseek": p2}


@pytest.fixture
def provider_manager(mock_providers):
    configs = [
        LLMProviderConfig(
            provider="openai", model="gpt-4", api_key="sk-123", capabilities=["text", "vision"]
        ),
        LLMProviderConfig(
            provider="openai", model="deepseek", api_key="sk-456", capabilities=["text"]
        ),
    ]
    pm = ProviderManager(configs)

    # Manually set up provider states to bypass initialization logic
    pm._provider_states = {
        "openai_gpt-4": ProviderState(
            config=configs[0],
            capabilities=["text", "vision"],
            provider=mock_providers["openai_gpt-4"],
            initialized=True,
            valid=True,
        ),
        "openai_deepseek": ProviderState(
            config=configs[1],
            capabilities=["text"],
            provider=mock_providers["openai_deepseek"],
            initialized=True,
            valid=True,
        ),
    }
    # Also populate _providers dict (used by complete_with_fallback and analyze_image_with_fallback)
    pm._providers = {
        "openai_gpt-4": mock_providers["openai_gpt-4"],
        "openai_deepseek": mock_providers["openai_deepseek"],
    }
    pm._valid_providers = ["openai_gpt-4", "openai_deepseek"]
    pm._provider_capabilities = {
        "openai_gpt-4": ["text", "vision"],
        "openai_deepseek": ["text"],
    }
    pm._config_loaded = True
    pm._initialized = True
    pm._current_index = 0

    return pm


@pytest.mark.asyncio
async def test_analyze_image_skips_text_only(provider_manager, mock_providers):
    """Test that analyze_image skips providers without vision capability."""
    # Should only call openai_gpt-4, never openai_deepseek

    # 1st call -> openai_gpt-4
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called
    assert not mock_providers["openai_deepseek"].analyze_image.called

    mock_providers["openai_gpt-4"].analyze_image.reset_mock()

    # 2nd call -> openai_gpt-4 (should NOT rotate to openai_deepseek)
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai_gpt-4"].analyze_image.called
    assert not mock_providers["openai_deepseek"].analyze_image.called


@pytest.mark.asyncio
async def test_no_vision_provider_raises_error(mock_providers):
    """Test error when no vision provider is available."""
    configs = [
        LLMProviderConfig(
            provider="openai", model="deepseek", api_key="sk-456", capabilities=["text"]
        ),
    ]
    pm = ProviderManager(configs)
    pm._provider_states = {
        "openai_deepseek": ProviderState(
            config=configs[0],
            capabilities=["text"],
            provider=mock_providers["openai_deepseek"],
            initialized=True,
            valid=True,
        ),
    }
    # Also populate _providers dict
    pm._providers = {"openai_deepseek": mock_providers["openai_deepseek"]}
    pm._valid_providers = ["openai_deepseek"]
    pm._provider_capabilities = {"openai_deepseek": ["text"]}
    pm._config_loaded = True
    pm._initialized = True

    with pytest.raises(LLMError) as exc:
        await pm.analyze_image_with_fallback(b"data", "prompt")

    assert "No provider with 'vision' capability" in str(exc.value)


@pytest.mark.asyncio
async def test_optimistic_default_capabilities():
    """Test that capabilities default to ['text', 'vision'] if None."""
    # We need to test the initialization logic, so we'll mock _create_provider
    configs = [
        LLMProviderConfig(
            provider="openai", model="gpt-5.2", api_key="sk-123"
        ),  # None capabilities
    ]
    pm = ProviderManager(configs)

    with (
        patch.object(pm, "_create_provider", return_value=AsyncMock()),
        patch.object(pm, "_validate_provider", return_value=True),
    ):
        await pm.initialize()

        provider_key = "openai_gpt-5.2"
        assert provider_key in pm._provider_capabilities
        assert pm._provider_capabilities[provider_key] == ["text", "vision"]


@pytest.mark.asyncio
async def test_api_key_env_support():
    """Test reading API key from custom env var."""
    configs = [
        LLMProviderConfig(provider="openai", model="deepseek", api_key_env="DEEPSEEK_TEST_KEY"),
    ]
    pm = ProviderManager(configs)

    # Mock environment and _get_api_key_from_env
    with patch("os.environ.get") as mock_env:
        mock_env.return_value = "sk-custom-key"

        # We invoke _create_provider directly to test logic
        # But _create_provider imports classes, so we might hit import errors if dependencies missing
        # Instead, let's mock the class instantiation or just test logic in isolation if possible.
        # Actually, _create_provider calls OpenAIProvider(...). We can patch the class.

        with patch("markit.llm.openai.OpenAIProvider") as MockProvider:
            pm._create_provider(configs[0])

            # Check if os.environ.get was called with correct key
            mock_env.assert_called_with("DEEPSEEK_TEST_KEY")

            # Check if Provider was initialized with the key from env
            call_args = MockProvider.call_args
            assert call_args.kwargs["api_key"] == "sk-custom-key"

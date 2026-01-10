from unittest.mock import AsyncMock

import pytest

from markit.config.settings import LLMProviderConfig
from markit.exceptions import LLMError
from markit.llm.manager import ProviderManager


@pytest.fixture
def mock_providers():
    p1 = AsyncMock()
    p1.complete.return_value = "response1"
    p2 = AsyncMock()
    p2.complete.return_value = "response2"
    return {"openai": p1, "anthropic": p2}


@pytest.fixture
def provider_manager(mock_providers):
    configs = [
        LLMProviderConfig(provider="openai", model="gpt-4", api_key="sk-123"),
        LLMProviderConfig(provider="anthropic", model="claude", api_key="sk-456"),
    ]
    pm = ProviderManager(configs)

    # Manually inject providers to bypass initialization logic
    pm._providers = mock_providers
    pm._valid_providers = ["openai", "anthropic"]
    # Manually inject capabilities (simulating optimistic default)
    pm._provider_capabilities = {
        "openai": ["text", "vision"],
        "anthropic": ["text", "vision"],
    }
    pm._initialized = True
    pm._current_index = 0

    return pm


@pytest.mark.asyncio
async def test_round_robin_load_balancing(provider_manager, mock_providers):
    """Test that requests rotate through providers."""
    # 1st call -> openai
    await provider_manager.complete_with_fallback([])
    assert mock_providers["openai"].complete.called
    assert not mock_providers["anthropic"].complete.called

    mock_providers["openai"].complete.reset_mock()

    # 2nd call -> anthropic
    await provider_manager.complete_with_fallback([])
    assert mock_providers["anthropic"].complete.called
    assert not mock_providers["openai"].complete.called

    mock_providers["anthropic"].complete.reset_mock()

    # 3rd call -> openai (wrap around)
    await provider_manager.complete_with_fallback([])
    assert mock_providers["openai"].complete.called
    assert not mock_providers["anthropic"].complete.called


@pytest.mark.asyncio
async def test_fallback_logic(provider_manager, mock_providers):
    """Test fallback when primary provider fails."""
    # Make openai fail
    mock_providers["openai"].complete.side_effect = Exception("OpenAI Error")

    # Call (starts at index 0 -> openai)
    # OpenAI fails, should fallback to Anthropic
    await provider_manager.complete_with_fallback([])

    assert mock_providers["openai"].complete.called
    assert mock_providers["anthropic"].complete.called


@pytest.mark.asyncio
async def test_all_providers_fail(provider_manager, mock_providers):
    """Test error raised when all providers fail."""
    mock_providers["openai"].complete.side_effect = Exception("Error 1")
    mock_providers["anthropic"].complete.side_effect = Exception("Error 2")

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

    mock_providers["openai"].analyze_image.return_value = "analysis1"
    mock_providers["anthropic"].analyze_image.return_value = "analysis2"

    # 1st call -> openai
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["openai"].analyze_image.called
    assert not mock_providers["anthropic"].analyze_image.called

    mock_providers["openai"].analyze_image.reset_mock()

    # 2nd call -> anthropic
    await provider_manager.analyze_image_with_fallback(b"data", "prompt")
    assert mock_providers["anthropic"].analyze_image.called
    assert not mock_providers["openai"].analyze_image.called

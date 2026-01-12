"""Test ChaosMockProvider for resilience testing.

ChaosMockProvider simulates real-world failure conditions:
- Random latency (Gaussian distribution)
- Rate limits (429 errors)
- Server errors (500)
- Timeouts
"""

import contextlib

import pytest

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import LLMMessage
from markit.llm.chaos import ChaosConfig, ChaosMockProvider


@pytest.mark.asyncio
async def test_chaos_default_config() -> None:
    """Test ChaosMockProvider with default configuration."""
    config = ChaosConfig()
    assert config.latency_mean >= 0
    assert 0 <= config.failure_rate <= 1
    assert 0 <= config.rate_limit_prob <= 1


@pytest.mark.asyncio
async def test_chaos_custom_config() -> None:
    """Test ChaosMockProvider with custom configuration."""
    custom_config = ChaosConfig(
        latency_mean=0.5,
        latency_stddev=0.1,
        failure_rate=0.05,
        rate_limit_prob=0.1,
        timeout_prob=0.0,
        response_text="Custom response for testing",
    )
    provider = ChaosMockProvider(custom_config)
    assert provider.config.latency_mean == 0.5
    assert provider.config.failure_rate == 0.05


@pytest.mark.asyncio
async def test_chaos_complete() -> None:
    """Test complete() method with no failures."""
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.005,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    messages = [LLMMessage.user("Hello, this is a test message")]
    response = await provider.complete(messages)

    assert response.content
    assert response.model


@pytest.mark.asyncio
async def test_chaos_analyze_image() -> None:
    """Test analyze_image() method."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    response = await provider.analyze_image(b"fake_image_data", "Analyze this image")
    assert "alt_text" in response.content


@pytest.mark.asyncio
async def test_chaos_validate() -> None:
    """Test validate() method."""
    provider = ChaosMockProvider(ChaosConfig())
    valid = await provider.validate()
    assert valid is True


@pytest.mark.asyncio
async def test_chaos_rate_limit() -> None:
    """Test rate limit simulation."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=1.0,  # Always rate limit
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    with pytest.raises(RateLimitError):
        await provider.complete([LLMMessage.user("test")])


@pytest.mark.asyncio
async def test_chaos_server_error() -> None:
    """Test server error simulation."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=1.0,  # Always fail
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    with pytest.raises(LLMError):
        await provider.complete([LLMMessage.user("test")])


@pytest.mark.asyncio
async def test_chaos_timeout() -> None:
    """Test timeout simulation."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=1.0,  # Always timeout
    )
    provider = ChaosMockProvider(config)

    with pytest.raises(TimeoutError):
        await provider.complete([LLMMessage.user("test")])


@pytest.mark.asyncio
async def test_chaos_statistics() -> None:
    """Test statistics tracking."""
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.005,
        failure_rate=0.2,
        rate_limit_prob=0.1,
        timeout_prob=0.05,
    )
    provider = ChaosMockProvider(config)

    for _ in range(50):
        with contextlib.suppress(LLMError, RateLimitError, TimeoutError):
            await provider.complete([LLMMessage.user("test")])

    stats = provider.stats
    total = stats.success_count + stats.failure_count + stats.rate_limit_count + stats.timeout_count
    assert total == stats.call_count


@pytest.mark.asyncio
async def test_chaos_call_history() -> None:
    """Test call history tracking."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    await provider.complete([LLMMessage.user("First call")])
    await provider.complete([LLMMessage.user("Second call")])
    await provider.analyze_image(b"image", "Analyze")

    assert len(provider.call_history) == 3

    # Test reset
    provider.reset_stats()
    assert len(provider.call_history) == 0
    assert provider.stats.call_count == 0


@pytest.mark.asyncio
async def test_chaos_runtime_config() -> None:
    """Test runtime configuration changes."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    # Success call
    await provider.complete([LLMMessage.user("test")])

    # Change config at runtime
    provider.configure(failure_rate=1.0)

    with pytest.raises(LLMError):
        await provider.complete([LLMMessage.user("test")])


@pytest.mark.asyncio
async def test_chaos_stream() -> None:
    """Test streaming with chaos."""
    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
        response_text="Hello world from chaos",
    )
    provider = ChaosMockProvider(config)

    words = []
    async for chunk in provider.stream([LLMMessage.user("test")]):
        words.append(chunk.strip())

    full_response = " ".join(words)
    assert full_response == config.response_text

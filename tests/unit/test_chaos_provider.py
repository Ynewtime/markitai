"""Tests for ChaosMockProvider."""

import pytest

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import LLMMessage
from markit.llm.chaos import ChaosConfig, ChaosMockProvider, ChaosStats


class TestChaosConfig:
    """Tests for ChaosConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChaosConfig()
        assert config.latency_mean == 2.0
        assert config.latency_stddev == 1.0
        assert config.failure_rate == 0.1
        assert config.rate_limit_prob == 0.2
        assert config.timeout_prob == 0.05
        assert config.oom_trigger is False
        assert config.model_name == "chaos-mock-v1"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChaosConfig(
            latency_mean=5.0,
            failure_rate=0.5,
            rate_limit_prob=0.1,
            model_name="custom-chaos",
        )
        assert config.latency_mean == 5.0
        assert config.failure_rate == 0.5
        assert config.rate_limit_prob == 0.1
        assert config.model_name == "custom-chaos"


class TestChaosStats:
    """Tests for ChaosStats dataclass."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = ChaosStats()
        assert stats.call_count == 0
        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.rate_limit_count == 0
        assert stats.total_requests == 0
        assert stats.success_rate == 100.0  # No requests = 100% success
        assert stats.rate_limit_rate == 0.0

    def test_total_requests(self):
        """Test total_requests property."""
        stats = ChaosStats(
            success_count=10,
            failure_count=2,
            rate_limit_count=3,
        )
        assert stats.total_requests == 15

    def test_success_rate(self):
        """Test success_rate calculation."""
        stats = ChaosStats(
            call_count=100,
            success_count=80,
            failure_count=15,
            rate_limit_count=5,
        )
        assert stats.success_rate == 80.0

    def test_rate_limit_rate(self):
        """Test rate_limit_rate calculation."""
        stats = ChaosStats(
            call_count=100,
            success_count=80,
            failure_count=10,
            rate_limit_count=10,
        )
        assert stats.rate_limit_rate == 10.0


class TestChaosMockProvider:
    """Tests for ChaosMockProvider."""

    @pytest.fixture
    def provider(self):
        """Create a chaos provider with no chaos for predictable tests."""
        config = ChaosConfig(
            latency_mean=0.0,
            latency_stddev=0.0,
            failure_rate=0.0,
            rate_limit_prob=0.0,
            timeout_prob=0.0,
        )
        return ChaosMockProvider(config)

    @pytest.fixture
    def failure_provider(self):
        """Create a chaos provider that always fails."""
        config = ChaosConfig(
            latency_mean=0.0,
            latency_stddev=0.0,
            failure_rate=1.0,  # Always fail
            rate_limit_prob=0.0,
            timeout_prob=0.0,
        )
        return ChaosMockProvider(config)

    @pytest.fixture
    def rate_limit_provider(self):
        """Create a chaos provider that always rate limits."""
        config = ChaosConfig(
            latency_mean=0.0,
            latency_stddev=0.0,
            failure_rate=0.0,
            rate_limit_prob=1.0,  # Always rate limit
            timeout_prob=0.0,
        )
        return ChaosMockProvider(config)

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        messages = [LLMMessage.user("Hello")]
        response = await provider.complete(messages)

        assert response.content == "Chaos mock response"
        assert response.model == "chaos-mock-v1"
        assert response.finish_reason == "stop"
        assert provider.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_complete_failure(self, failure_provider):
        """Test completion failure."""
        messages = [LLMMessage.user("Hello")]

        with pytest.raises(LLMError, match="Simulated random server error"):
            await failure_provider.complete(messages)

        assert failure_provider.stats.failure_count == 1

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self, rate_limit_provider):
        """Test rate limit error."""
        messages = [LLMMessage.user("Hello")]

        with pytest.raises(RateLimitError):
            await rate_limit_provider.complete(messages)

        assert rate_limit_provider.stats.rate_limit_count == 1

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, provider):
        """Test successful image analysis."""
        response = await provider.analyze_image(b"fake image data", "Describe this")

        assert "alt_text" in response.content
        assert response.model == "chaos-mock-v1"
        assert provider.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_stream(self, provider):
        """Test streaming response."""
        messages = [LLMMessage.user("Hello")]
        chunks = []

        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert provider.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_validate(self, provider):
        """Test validation always returns True."""
        result = await provider.validate()
        assert result is True

    def test_reset_stats(self, provider):
        """Test resetting statistics."""
        provider._stats.success_count = 10
        provider._stats.failure_count = 5
        provider._call_history.append({"test": "data"})

        provider.reset_stats()

        assert provider.stats.success_count == 0
        assert provider.stats.failure_count == 0
        assert len(provider.call_history) == 0

    def test_configure(self, provider):
        """Test runtime configuration update."""
        provider.configure(failure_rate=0.5, latency_mean=3.0)

        assert provider.config.failure_rate == 0.5
        assert provider.config.latency_mean == 3.0

    @pytest.mark.asyncio
    async def test_call_history(self, provider):
        """Test that call history is recorded."""
        messages = [LLMMessage.user("Hello")]
        await provider.complete(messages)

        history = provider.call_history
        assert len(history) == 1
        assert history[0]["operation"] == "complete"
        assert history[0]["outcome"] == "success"
        assert "request_id" in history[0]
        assert "latency_ms" in history[0]

    @pytest.mark.asyncio
    async def test_avg_latency(self, provider):
        """Test average latency calculation."""
        messages = [LLMMessage.user("Hello")]

        # Make a few calls
        for _ in range(3):
            await provider.complete(messages)

        # With 0 latency configured, avg should be very low
        assert provider.stats.avg_latency_ms >= 0

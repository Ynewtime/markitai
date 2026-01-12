"""Tests for AIMD AdaptiveRateLimiter."""

import asyncio

import pytest

from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig, AIMDStats


class TestAIMDConfig:
    """Tests for AIMDConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AIMDConfig()
        assert config.initial_concurrency == 5
        assert config.min_concurrency == 1
        assert config.max_concurrency == 50
        assert config.success_threshold == 10
        assert config.additive_increase == 1
        assert config.multiplicative_decrease == 0.5
        assert config.cooldown_seconds == 5.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AIMDConfig(
            initial_concurrency=10,
            max_concurrency=100,
            success_threshold=20,
        )
        assert config.initial_concurrency == 10
        assert config.max_concurrency == 100
        assert config.success_threshold == 20


class TestAIMDStats:
    """Tests for AIMDStats dataclass."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = AIMDStats()
        assert stats.current_concurrency == 5
        assert stats.total_successes == 0
        assert stats.total_failures == 0
        assert stats.rate_limit_hits == 0
        assert stats.total_requests == 0

    def test_total_requests(self):
        """Test total_requests calculation."""
        stats = AIMDStats(
            total_successes=100,
            total_failures=10,
            rate_limit_hits=5,
        )
        assert stats.total_requests == 115

    def test_success_rate(self):
        """Test success_rate calculation."""
        stats = AIMDStats(
            total_successes=90,
            total_failures=5,
            rate_limit_hits=5,
        )
        assert stats.success_rate == 90.0

    def test_success_rate_no_requests(self):
        """Test success_rate with no requests."""
        stats = AIMDStats()
        assert stats.success_rate == 100.0

    def test_rate_limit_rate(self):
        """Test rate_limit_rate calculation."""
        stats = AIMDStats(
            total_successes=80,
            total_failures=10,
            rate_limit_hits=10,
        )
        assert stats.rate_limit_rate == 10.0


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create a limiter with small thresholds for testing."""
        config = AIMDConfig(
            initial_concurrency=5,
            min_concurrency=1,
            max_concurrency=20,
            success_threshold=3,  # Small for testing
            cooldown_seconds=0.0,  # No cooldown for testing
        )
        return AdaptiveRateLimiter(config)

    @pytest.mark.asyncio
    async def test_initial_state(self, limiter):
        """Test initial limiter state."""
        assert limiter.current_concurrency == 5
        assert limiter.stats.total_successes == 0
        assert limiter.stats.increase_count == 0
        assert limiter.stats.decrease_count == 0

    @pytest.mark.asyncio
    async def test_acquire_release(self, limiter):
        """Test basic acquire/release."""
        await limiter.acquire()
        limiter.release()
        # No exception means success
        assert True

    @pytest.mark.asyncio
    async def test_context_manager(self, limiter):
        """Test async context manager."""
        async with limiter:
            pass  # Should not raise
        assert True

    @pytest.mark.asyncio
    async def test_additive_increase(self, limiter):
        """Test concurrency increases after success threshold."""
        initial = limiter.current_concurrency

        # Record enough successes to trigger increase
        for _ in range(3):  # success_threshold = 3
            await limiter.record_success()

        assert limiter.current_concurrency == initial + 1
        assert limiter.stats.increase_count == 1
        assert limiter.stats.total_successes == 3

    @pytest.mark.asyncio
    async def test_max_concurrency_cap(self, limiter):
        """Test concurrency doesn't exceed max."""
        # Set to max
        limiter._current_concurrency = 20

        # Try to increase
        for _ in range(3):
            await limiter.record_success()

        assert limiter.current_concurrency == 20  # Still at max

    @pytest.mark.asyncio
    async def test_multiplicative_decrease(self, limiter):
        """Test concurrency halves on rate limit."""
        # Start at 10
        limiter._current_concurrency = 10
        limiter._semaphore = asyncio.Semaphore(10)

        await limiter.record_rate_limit()

        assert limiter.current_concurrency == 5  # 10 * 0.5 = 5
        assert limiter.stats.decrease_count == 1
        assert limiter.stats.rate_limit_hits == 1

    @pytest.mark.asyncio
    async def test_min_concurrency_floor(self, limiter):
        """Test concurrency doesn't go below min."""
        # Start at 2
        limiter._current_concurrency = 2
        limiter._semaphore = asyncio.Semaphore(2)

        await limiter.record_rate_limit()

        assert limiter.current_concurrency == 1  # min_concurrency

    @pytest.mark.asyncio
    async def test_failure_resets_streak(self, limiter):
        """Test failures reset success streak without decreasing."""
        initial = limiter.current_concurrency

        # Record some successes (not enough to increase)
        await limiter.record_success()
        await limiter.record_success()

        # Record failure - should reset streak
        await limiter.record_failure()

        # Record more successes (now needs 3 more)
        await limiter.record_success()
        await limiter.record_success()

        # Should NOT have increased (only 2 after reset)
        assert limiter.current_concurrency == initial
        assert limiter.stats.total_failures == 1

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_decrease(self):
        """Test cooldown prevents rapid decreases."""
        config = AIMDConfig(
            initial_concurrency=10,
            cooldown_seconds=1.0,  # 1 second cooldown
        )
        limiter = AdaptiveRateLimiter(config)

        # First rate limit
        await limiter.record_rate_limit()
        first = limiter.current_concurrency
        assert first == 5

        # Second rate limit immediately (should be ignored due to cooldown)
        await limiter.record_rate_limit()
        assert limiter.current_concurrency == 5  # No further decrease

        # Rate limit count should still increase
        assert limiter.stats.rate_limit_hits == 2

    @pytest.mark.asyncio
    async def test_peak_trough_tracking(self, limiter):
        """Test peak and trough tracking."""
        # Increase to peak
        for _ in range(3):
            await limiter.record_success()
        peak = limiter.current_concurrency

        # Decrease to trough
        await limiter.record_rate_limit()
        trough = limiter.current_concurrency

        assert limiter.stats.peak_concurrency == peak
        assert limiter.stats.trough_concurrency == trough

    def test_reset(self, limiter):
        """Test limiter reset."""
        limiter._current_concurrency = 10
        limiter._stats.total_successes = 100
        limiter._success_streak = 5

        limiter.reset()

        assert limiter.current_concurrency == 5  # Back to initial
        assert limiter.stats.total_successes == 0
        assert limiter._success_streak == 0

    @pytest.mark.asyncio
    async def test_callback_on_adjust(self):
        """Test callback is called on concurrency adjustment."""
        adjustments = []

        def on_adjust(new_concurrency, direction):
            adjustments.append((new_concurrency, direction))

        config = AIMDConfig(
            initial_concurrency=5,
            success_threshold=2,
            cooldown_seconds=0.0,
        )
        limiter = AdaptiveRateLimiter(config, on_adjust=on_adjust)

        # Trigger increase
        await limiter.record_success()
        await limiter.record_success()

        assert len(adjustments) == 1
        assert adjustments[0][1] == "increase"

        # Trigger decrease
        await limiter.record_rate_limit()

        assert len(adjustments) == 2
        assert adjustments[1][1] == "decrease"

    @pytest.mark.asyncio
    async def test_concurrent_access(self, limiter):
        """Test limiter handles concurrent access safely."""

        async def worker():
            await limiter.acquire()
            await asyncio.sleep(0.01)
            await limiter.record_success()
            limiter.release()

        # Run many workers concurrently
        tasks = [worker() for _ in range(20)]
        await asyncio.gather(*tasks)

        assert limiter.stats.total_successes == 20

"""Test AIMD adaptive rate limiter under simulated load.

AIMD (Additive Increase Multiplicative Decrease) dynamically adjusts
concurrency based on success/rate-limit responses.
"""

import asyncio

import pytest

from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig


@pytest.mark.asyncio
async def test_aimd_additive_increase() -> None:
    """Test additive increase after consecutive successes."""
    config = AIMDConfig(
        initial_concurrency=5,
        max_concurrency=20,
        min_concurrency=1,
        success_threshold=3,
        cooldown_seconds=0.1,
    )
    limiter = AdaptiveRateLimiter(config)

    initial = limiter.current_concurrency

    # Record enough successes to trigger increase
    for _ in range(5):
        await limiter.acquire()
        await limiter.record_success()
        limiter.release()

    assert limiter.current_concurrency >= initial + 1


@pytest.mark.asyncio
async def test_aimd_multiplicative_decrease() -> None:
    """Test multiplicative decrease on rate limit."""
    config = AIMDConfig(
        initial_concurrency=10,
        max_concurrency=20,
        min_concurrency=1,
        cooldown_seconds=0.1,
    )
    limiter = AdaptiveRateLimiter(config)

    before = limiter.current_concurrency
    await limiter.record_rate_limit()
    after = limiter.current_concurrency

    assert after < before


@pytest.mark.asyncio
async def test_aimd_record_error_interface() -> None:
    """Test unified record_error() interface."""
    config = AIMDConfig(
        initial_concurrency=10,
        max_concurrency=20,
        min_concurrency=1,
        cooldown_seconds=0.05,
    )
    limiter = AdaptiveRateLimiter(config)

    # Wait for any cooldown
    await asyncio.sleep(0.1)

    current = limiter.current_concurrency

    # Non-rate-limit error should not affect concurrency
    await limiter.record_error(is_rate_limit=False)
    assert limiter.current_concurrency == current

    # Rate-limit error should decrease concurrency
    await limiter.record_error(is_rate_limit=True)
    assert limiter.current_concurrency < current


@pytest.mark.asyncio
async def test_aimd_under_load() -> None:
    """Test AIMD behavior under simulated mixed load."""
    config = AIMDConfig(
        initial_concurrency=10,
        max_concurrency=50,
        min_concurrency=2,
        success_threshold=5,
        cooldown_seconds=0.05,
    )
    limiter = AdaptiveRateLimiter(config)

    for i in range(100):
        await limiter.acquire()
        try:
            # Simulate ~14% rate limit rate
            if i % 7 == 0:
                await limiter.record_rate_limit()
            else:
                await limiter.record_success()
        finally:
            limiter.release()

    # Verify adaptation occurred
    stats = limiter.stats
    assert stats.increase_count > 0 or stats.decrease_count > 0


@pytest.mark.asyncio
async def test_aimd_concurrent_workers() -> None:
    """Test AIMD with concurrent workers acquiring slots."""
    config = AIMDConfig(
        initial_concurrency=5,
        max_concurrency=10,
        min_concurrency=1,
    )
    limiter = AdaptiveRateLimiter(config)

    active_count = 0
    max_active = 0

    async def worker(_worker_id: int) -> None:
        nonlocal active_count, max_active
        await limiter.acquire()
        active_count += 1
        max_active = max(max_active, active_count)

        await asyncio.sleep(0.05)

        active_count -= 1
        await limiter.record_success()
        limiter.release()

    # Run 10 workers concurrently
    await asyncio.gather(*[worker(i) for i in range(10)])

    assert max_active <= config.initial_concurrency

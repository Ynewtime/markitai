#!/usr/bin/env python3
"""UAT: Test AIMD adaptive rate limiter under simulated load.

AIMD (Additive Increase Multiplicative Decrease) dynamically adjusts
concurrency based on success/rate-limit responses.

Usage:
    uv run python uat/test_aimd_limiter.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig


async def test_aimd_basic():
    """Test basic AIMD operations."""
    print("=" * 60)
    print("UAT: AIMD Adaptive Rate Limiter - Basic Operations")
    print("=" * 60)
    print()

    config = AIMDConfig(
        initial_concurrency=5,
        max_concurrency=20,
        min_concurrency=1,
        success_threshold=3,
        cooldown_seconds=0.1,
    )
    limiter = AdaptiveRateLimiter(config)

    print(f"Initial concurrency: {limiter.current_concurrency}")
    print(f"Config: max={config.max_concurrency}, min={config.min_concurrency}")
    print(f"Success threshold for increase: {config.success_threshold}")
    print("-" * 40)

    # Test additive increase
    print("\n[Test 1] Additive Increase after consecutive successes:")
    for i in range(5):
        await limiter.acquire()
        await limiter.record_success()
        limiter.release()
        print(f"  Request {i + 1}: concurrency = {limiter.current_concurrency}")

    expected_increase = config.initial_concurrency + 1  # Should increase by 1
    if limiter.current_concurrency >= expected_increase:
        print(f"  ✅ Concurrency increased to {limiter.current_concurrency}")
    else:
        print(f"  ❌ Expected increase, got {limiter.current_concurrency}")
        return False

    # Test multiplicative decrease
    print("\n[Test 2] Multiplicative Decrease on rate limit:")
    before_decrease = limiter.current_concurrency
    await limiter.record_rate_limit()
    after_decrease = limiter.current_concurrency
    print(f"  Before rate limit: {before_decrease}")
    print(f"  After rate limit: {after_decrease}")

    if after_decrease < before_decrease:
        print(f"  ✅ Concurrency decreased (multiplier={config.multiplicative_decrease})")
    else:
        print("  ❌ Expected decrease")
        return False

    # Test record_error unified interface
    print("\n[Test 3] Unified record_error() interface:")

    # Wait for cooldown to expire before testing decrease
    await asyncio.sleep(0.15)

    current = limiter.current_concurrency

    await limiter.record_error(is_rate_limit=False)
    print(f"  After non-rate-limit error: {limiter.current_concurrency} (no change)")

    if limiter.current_concurrency == current:
        print("  ✅ Non-rate-limit error doesn't affect concurrency")
    else:
        print("  ❌ Concurrency changed unexpectedly")
        return False

    await limiter.record_error(is_rate_limit=True)
    print(f"  After rate-limit error: {limiter.current_concurrency} (decreased)")

    if limiter.current_concurrency < current:
        print("  ✅ record_error() correctly routes to record_rate_limit()")
    else:
        print("  ❌ Expected decrease for rate limit error")
        return False

    # Final stats
    print("\n[Statistics]")
    stats = limiter.stats
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Successes: {stats.total_successes}")
    print(f"  Rate limit hits: {stats.rate_limit_hits}")
    print(f"  Total failures: {stats.total_failures}")
    print(f"  Increases: {stats.increase_count}")
    print(f"  Decreases: {stats.decrease_count}")

    return True


async def test_aimd_under_load():
    """Test AIMD behavior under simulated mixed load."""
    print()
    print("=" * 60)
    print("UAT: AIMD Adaptive Rate Limiter - Simulated Load")
    print("=" * 60)
    print()

    config = AIMDConfig(
        initial_concurrency=10,
        max_concurrency=50,
        min_concurrency=2,
        success_threshold=5,
        cooldown_seconds=0.05,
    )
    limiter = AdaptiveRateLimiter(config)

    print("Simulating 100 requests with 15% rate limit rate...")
    print(f"Initial concurrency: {limiter.current_concurrency}")
    print("-" * 40)

    rate_limits = 0
    successes = 0

    for i in range(100):
        await limiter.acquire()
        try:
            # Simulate 15% rate limit rate
            if i % 7 == 0:  # ~14% rate limit
                await limiter.record_rate_limit()
                rate_limits += 1
            else:
                await limiter.record_success()
                successes += 1
        finally:
            limiter.release()

    print("\nResults:")
    print(f"  Successes: {successes}")
    print(f"  Rate limits: {rate_limits}")
    print(f"  Final concurrency: {limiter.current_concurrency}")
    print(f"  Peak concurrency: {limiter.stats.peak_concurrency}")
    print(f"  Trough concurrency: {limiter.stats.trough_concurrency}")

    # Verify adaptation occurred
    if limiter.stats.increase_count > 0 or limiter.stats.decrease_count > 0:
        print(
            f"  ✅ AIMD adapted: {limiter.stats.increase_count} increases, {limiter.stats.decrease_count} decreases"
        )
        return True
    else:
        print("  ❌ No adaptation occurred")
        return False


async def test_aimd_concurrent_workers():
    """Test AIMD with concurrent workers acquiring slots."""
    print()
    print("=" * 60)
    print("UAT: AIMD Adaptive Rate Limiter - Concurrent Workers")
    print("=" * 60)
    print()

    config = AIMDConfig(
        initial_concurrency=5,
        max_concurrency=10,
        min_concurrency=1,
    )
    limiter = AdaptiveRateLimiter(config)

    print("Spawning 10 concurrent workers with concurrency limit of 5...")
    print("-" * 40)

    active_count = 0
    max_active = 0

    async def worker(worker_id: int):
        nonlocal active_count, max_active
        await limiter.acquire()
        active_count += 1
        max_active = max(max_active, active_count)
        print(f"  Worker {worker_id} acquired slot (active: {active_count})")

        await asyncio.sleep(0.05)  # Simulate work

        active_count -= 1
        await limiter.record_success()
        limiter.release()
        print(f"  Worker {worker_id} released slot")

    # Run 10 workers concurrently
    await asyncio.gather(*[worker(i) for i in range(10)])

    print("\nResults:")
    print(f"  Max concurrent workers: {max_active}")
    print(f"  Concurrency limit was: {config.initial_concurrency}")

    if max_active <= config.initial_concurrency:
        print("  ✅ Concurrency properly limited")
        return True
    else:
        print("  ❌ Exceeded concurrency limit")
        return False


async def main():
    """Run all AIMD UAT tests."""
    results = []

    results.append(("Basic Operations", await test_aimd_basic()))
    results.append(("Simulated Load", await test_aimd_under_load()))
    results.append(("Concurrent Workers", await test_aimd_concurrent_workers()))

    print()
    print("=" * 60)
    print("SUMMARY: AIMD Rate Limiter UAT")
    print("=" * 60)
    print()

    all_passed = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_passed = False

    print()
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

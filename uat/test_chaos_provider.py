#!/usr/bin/env python3
"""UAT: Test ChaosMockProvider for resilience testing.

ChaosMockProvider simulates real-world failure conditions:
- Random latency (Gaussian distribution)
- Rate limits (429 errors)
- Server errors (500)
- Timeouts

Usage:
    uv run python uat/test_chaos_provider.py
"""

import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import LLMMessage
from markit.llm.chaos import ChaosConfig, ChaosMockProvider


async def test_chaos_config():
    """Test ChaosMockProvider with different configurations."""
    print("=" * 60)
    print("UAT: ChaosMockProvider - Configuration")
    print("=" * 60)
    print()

    # Test with default config
    print("[Test 1] Default configuration:")
    config = ChaosConfig()
    print(f"  latency_mean: {config.latency_mean}s")
    print(f"  latency_stddev: {config.latency_stddev}s")
    print(f"  failure_rate: {config.failure_rate * 100}%")
    print(f"  rate_limit_prob: {config.rate_limit_prob * 100}%")
    print(f"  timeout_prob: {config.timeout_prob * 100}%")
    print("  ✅ Default config loaded")

    # Test with custom config
    print("\n[Test 2] Custom configuration:")
    custom_config = ChaosConfig(
        latency_mean=0.5,
        latency_stddev=0.1,
        failure_rate=0.05,
        rate_limit_prob=0.1,
        timeout_prob=0.0,
        response_text="Custom response for testing",
    )
    provider = ChaosMockProvider(custom_config)
    print(f"  latency_mean: {provider.config.latency_mean}s")
    print(f"  failure_rate: {provider.config.failure_rate * 100}%")
    print(f"  response_text: '{provider.config.response_text[:30]}...'")
    print("  ✅ Custom config applied")

    return True


async def test_chaos_basic_calls():
    """Test basic call behavior with chaos effects."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Basic Calls")
    print("=" * 60)
    print()

    # Use fast latency, no failures for predictable testing
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.005,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    print("[Test 1] complete() method:")
    messages = [LLMMessage.user("Hello, this is a test message")]
    response = await provider.complete(messages)

    print(f"  Content: '{response.content[:40]}...'")
    print(f"  Model: {response.model}")
    print(f"  Finish reason: {response.finish_reason}")
    print(
        f"  Usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output"
    )

    if response.content and response.model:
        print("  ✅ complete() works correctly")
    else:
        return False

    print("\n[Test 2] analyze_image() method:")
    response = await provider.analyze_image(b"fake_image_data", "Analyze this image")
    print(f"  Content: '{response.content[:50]}...'")

    if "alt_text" in response.content:
        print("  ✅ analyze_image() returns JSON structure")
    else:
        return False

    print("\n[Test 3] validate() method:")
    valid = await provider.validate()
    if valid:
        print("  ✅ validate() returns True")
    else:
        return False

    return True


async def test_chaos_failure_modes():
    """Test different failure modes."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Failure Modes")
    print("=" * 60)
    print()

    # Test rate limit
    print("[Test 1] Rate limit simulation:")
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.0,
        failure_rate=0.0,
        rate_limit_prob=1.0,  # Always rate limit
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    try:
        await provider.complete([LLMMessage.user("test")])
        print("  ❌ Expected RateLimitError")
        return False
    except RateLimitError as e:
        print(f"  ✅ RateLimitError raised (retry_after={e.retry_after})")

    # Test random failure
    print("\n[Test 2] Server error simulation:")
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.0,
        failure_rate=1.0,  # Always fail
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    try:
        await provider.complete([LLMMessage.user("test")])
        print("  ❌ Expected LLMError")
        return False
    except LLMError:
        print("  ✅ LLMError raised (500 simulation)")

    # Test timeout
    print("\n[Test 3] Timeout simulation:")
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.0,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=1.0,  # Always timeout
    )
    provider = ChaosMockProvider(config)

    try:
        await provider.complete([LLMMessage.user("test")])
        print("  ❌ Expected TimeoutError")
        return False
    except TimeoutError:
        print("  ✅ TimeoutError raised")

    return True


async def test_chaos_statistics():
    """Test statistics tracking."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Statistics")
    print("=" * 60)
    print()

    # Configure with mixed outcomes
    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.005,
        failure_rate=0.2,
        rate_limit_prob=0.1,
        timeout_prob=0.05,
    )
    provider = ChaosMockProvider(config)

    print("Running 50 requests with mixed failure rates...")
    print(f"  failure_rate: {config.failure_rate * 100}%")
    print(f"  rate_limit_prob: {config.rate_limit_prob * 100}%")
    print(f"  timeout_prob: {config.timeout_prob * 100}%")
    print("-" * 40)

    for _ in range(50):
        with contextlib.suppress(LLMError, RateLimitError, TimeoutError):
            await provider.complete([LLMMessage.user("test")])

    stats = provider.stats
    print("\n[Statistics]")
    print(f"  Total calls: {stats.call_count}")
    print(f"  Successes: {stats.success_count}")
    print(f"  Failures (500): {stats.failure_count}")
    print(f"  Rate limits (429): {stats.rate_limit_count}")
    print(f"  Timeouts: {stats.timeout_count}")
    print(f"  Avg latency: {stats.avg_latency_ms:.1f}ms")
    print(f"  Success rate: {stats.success_rate:.1f}%")

    # Verify counts add up
    total = stats.success_count + stats.failure_count + stats.rate_limit_count + stats.timeout_count
    if total == stats.call_count:
        print(f"  ✅ All calls accounted for ({total} = {stats.call_count})")
    else:
        print(f"  ❌ Counts don't add up ({total} != {stats.call_count})")
        return False

    return True


async def test_chaos_call_history():
    """Test call history tracking."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Call History")
    print("=" * 60)
    print()

    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.0,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    # Make some calls
    await provider.complete([LLMMessage.user("First call")])
    await provider.complete([LLMMessage.user("Second call")])
    await provider.analyze_image(b"image", "Analyze")

    history = provider.call_history
    print(f"Call history has {len(history)} entries:")
    for i, record in enumerate(history):
        print(
            f"  [{i}] operation={record['operation']}, outcome={record['outcome']}, latency={record['latency_ms']}ms"
        )

    if len(history) == 3:
        print("  ✅ All calls recorded in history")
    else:
        return False

    # Test reset
    print("\n[Test reset]:")
    provider.reset_stats()
    if len(provider.call_history) == 0 and provider.stats.call_count == 0:
        print("  ✅ reset_stats() clears history and stats")
    else:
        return False

    return True


async def test_chaos_runtime_config():
    """Test runtime configuration changes."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Runtime Configuration")
    print("=" * 60)
    print()

    config = ChaosConfig(
        latency_mean=0.01,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
    )
    provider = ChaosMockProvider(config)

    print(f"Initial failure_rate: {provider.config.failure_rate * 100}%")

    # Success call
    await provider.complete([LLMMessage.user("test")])
    print("  Call succeeded (failure_rate=0%)")

    # Change config at runtime
    provider.configure(failure_rate=1.0)
    print(f"\nUpdated failure_rate: {provider.config.failure_rate * 100}%")

    try:
        await provider.complete([LLMMessage.user("test")])
        print("  ❌ Expected failure")
        return False
    except LLMError:
        print("  ✅ Call failed after runtime config change")

    return True


async def test_chaos_stream():
    """Test streaming with chaos."""
    print()
    print("=" * 60)
    print("UAT: ChaosMockProvider - Streaming")
    print("=" * 60)
    print()

    config = ChaosConfig(
        latency_mean=0.01,
        latency_stddev=0.0,
        failure_rate=0.0,
        rate_limit_prob=0.0,
        timeout_prob=0.0,
        response_text="Hello world from chaos",
    )
    provider = ChaosMockProvider(config)

    print("Streaming response:")
    words = []
    async for chunk in provider.stream([LLMMessage.user("test")]):
        words.append(chunk.strip())
        print(f"  chunk: '{chunk.strip()}'")

    full_response = " ".join(words)
    if full_response == config.response_text:
        print(f"  ✅ Stream complete: '{full_response}'")
        return True
    else:
        print(f"  ❌ Stream mismatch: '{full_response}'")
        return False


async def main():
    """Run all ChaosMockProvider UAT tests."""
    results = []

    results.append(("Configuration", await test_chaos_config()))
    results.append(("Basic Calls", await test_chaos_basic_calls()))
    results.append(("Failure Modes", await test_chaos_failure_modes()))
    results.append(("Statistics", await test_chaos_statistics()))
    results.append(("Call History", await test_chaos_call_history()))
    results.append(("Runtime Config", await test_chaos_runtime_config()))
    results.append(("Streaming", await test_chaos_stream()))

    print()
    print("=" * 60)
    print("SUMMARY: ChaosMockProvider UAT")
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

#!/usr/bin/env python3
"""Run all resilience UAT tests.

This script runs all resilience-related tests for the cc branch features:
- AIMD Adaptive Rate Limiter
- Dead Letter Queue (DLQ)
- BoundedQueue (Backpressure)
- ChaosMockProvider

Usage:
    uv run python uat/run_resilience.py
"""

import asyncio
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_all_resilience_tests():
    """Run all resilience UAT tests and summarize results."""
    print("=" * 70)
    print("UAT: Resilience Features Test Suite (cc branch)")
    print("=" * 70)
    print()

    tests = [
        ("AIMD Rate Limiter", "test_aimd_limiter"),
        ("Dead Letter Queue", "test_dead_letter_queue"),
        ("BoundedQueue (Backpressure)", "test_bounded_queue"),
        ("ChaosMockProvider", "test_chaos_provider"),
    ]

    results = {}

    for name, module_name in tests:
        print()
        print("#" * 70)
        print(f"# Running: {name}")
        print("#" * 70)
        print()

        try:
            module = importlib.import_module(module_name)
            main_func = getattr(module, "main")

            if asyncio.iscoroutinefunction(main_func):
                success = await main_func()
            else:
                success = main_func()

            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = "ERROR"

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: Resilience UAT Suite")
    print("=" * 70)
    print()

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if v == "FAILED")
    errors = sum(1 for v in results.values() if v == "ERROR")

    for name, status in results.items():
        icon = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}[status]
        print(f"  {icon} {name}: {status}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {errors} errors")
    print()

    # Feature overview
    print("Resilience Features Tested:")
    print("-" * 40)
    print("  - AIMD (Additive Increase Multiplicative Decrease)")
    print("    Auto-adjusts concurrency based on 429 rate limits")
    print()
    print("  - Dead Letter Queue (DLQ)")
    print("    Tracks failures per file, isolates permanent failures")
    print()
    print("  - BoundedQueue (Backpressure)")
    print("    Prevents memory exhaustion during batch processing")
    print()
    print("  - ChaosMockProvider")
    print("    Simulates failures for resilience testing")
    print()

    return failed == 0 and errors == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_resilience_tests())
    sys.exit(0 if success else 1)

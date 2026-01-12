#!/usr/bin/env python3
"""UAT: Test BoundedQueue backpressure mechanism.

BoundedQueue prevents memory exhaustion during high-volume batch
processing by blocking producers when the queue is full.

Usage:
    uv run python uat/test_bounded_queue.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.utils.flow_control import BoundedQueue


async def test_backpressure_basic():
    """Test basic backpressure behavior."""
    print("=" * 60)
    print("UAT: BoundedQueue - Basic Backpressure")
    print("=" * 60)
    print()

    queue: BoundedQueue[str] = BoundedQueue(max_size=5, put_timeout=0.5)

    print(f"Queue max size: {queue.max_size}")
    print("Put timeout: 0.5s")
    print("-" * 40)

    # Fill queue
    print("\n[Step 1] Fill queue to capacity:")
    for i in range(5):
        result = await queue.put(f"item_{i}")
        print(f"  Put item_{i}: {'✅' if result else '❌'}")

    print(f"\n  Queue size: {queue.size}")
    print(f"  Queue full: {queue.full}")

    if not queue.full:
        print("  ❌ Queue should be full")
        return False

    # Try to add more (should timeout)
    print("\n[Step 2] Attempt to add item when full (should timeout):")
    result = await queue.put("overflow_item")

    if result:
        print("  ❌ Put should have failed (queue full)")
        return False
    else:
        print("  ✅ Put correctly timed out")

    # Check stats
    print(f"\n  Dropped items: {queue.stats.total_dropped}")
    if queue.stats.total_dropped == 1:
        print("  ✅ Dropped count correct")
    else:
        print("  ❌ Expected 1 dropped")
        return False

    return True


async def test_producer_consumer():
    """Test producer-consumer pattern with backpressure."""
    print()
    print("=" * 60)
    print("UAT: BoundedQueue - Producer/Consumer Pattern")
    print("=" * 60)
    print()

    queue: BoundedQueue[int] = BoundedQueue(max_size=10, put_timeout=1.0)
    processed = []
    producer_done = asyncio.Event()

    print("Queue size: 10")
    print("Producing 50 items, consuming with delay...")
    print("-" * 40)

    async def producer():
        for i in range(50):
            success = await queue.put(i)
            if not success:
                print(f"  [Producer] Item {i} dropped!")
        producer_done.set()

    async def consumer():
        while not (producer_done.is_set() and queue.empty):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                processed.append(item)
                await asyncio.sleep(0.01)  # Simulate processing
                queue.task_done()
            except TimeoutError:
                if producer_done.is_set():
                    break

    # Run producer and consumer
    await asyncio.gather(producer(), consumer())

    print("\n[Results]")
    print("  Items produced: 50")
    print(f"  Items consumed: {len(processed)}")
    print(f"  Items dropped: {queue.stats.total_dropped}")
    print(f"  Throughput ratio: {queue.stats.throughput_ratio:.2%}")
    print(f"  Drop rate: {queue.stats.drop_rate:.2%}")

    # Some items may be dropped due to backpressure
    total_accounted = len(processed) + queue.stats.total_dropped
    if total_accounted == 50:
        print("  ✅ All items accounted for (processed + dropped)")
        return True
    else:
        print(f"  ❌ Expected 50 items total, got {total_accounted}")
        return False


async def test_queue_statistics():
    """Test queue statistics tracking."""
    print()
    print("=" * 60)
    print("UAT: BoundedQueue - Statistics")
    print("=" * 60)
    print()

    queue: BoundedQueue[str] = BoundedQueue(max_size=20)

    # Enqueue items
    for i in range(15):
        await queue.put(f"item_{i}")

    # Dequeue some
    for _ in range(10):
        await queue.get()
        queue.task_done()

    stats = queue.stats
    print("[Queue Statistics]")
    print("-" * 40)
    print(f"  Current size: {stats.current_size}")
    print(f"  Max size: {stats.max_size}")
    print(f"  Total enqueued: {stats.total_enqueued}")
    print(f"  Total dequeued: {stats.total_dequeued}")
    print(f"  Total completed: {stats.total_completed}")
    print(f"  Total dropped: {stats.total_dropped}")
    print(f"  Throughput ratio: {stats.throughput_ratio:.2%}")
    print(f"  Drop rate: {stats.drop_rate:.2%}")

    # Verify stats
    if stats.total_enqueued == 15:
        print("  ✅ Enqueue count correct")
    else:
        return False

    if stats.total_dequeued == 10:
        print("  ✅ Dequeue count correct")
    else:
        return False

    if stats.total_completed == 10:
        print("  ✅ Completed count correct")
    else:
        return False

    if stats.current_size == 5:
        print("  ✅ Current size correct")
    else:
        return False

    return True


async def test_concurrent_producers():
    """Test multiple producers with backpressure."""
    print()
    print("=" * 60)
    print("UAT: BoundedQueue - Concurrent Producers")
    print("=" * 60)
    print()

    queue: BoundedQueue[tuple[int, int]] = BoundedQueue(max_size=20, put_timeout=0.5)
    producer_results = {}

    print("Spawning 5 producers, each sending 30 items...")
    print("Queue size: 20, expect backpressure...")
    print("-" * 40)

    async def producer(producer_id: int):
        success_count = 0
        for i in range(30):
            result = await queue.put((producer_id, i))
            if result:
                success_count += 1
        producer_results[producer_id] = success_count

    async def consumer():
        consumed = 0
        while consumed < 100:  # Try to consume up to 100 items
            try:
                await asyncio.wait_for(queue.get(), timeout=1.0)
                queue.task_done()
                consumed += 1
                await asyncio.sleep(0.01)  # Slow consumer to cause backpressure
            except TimeoutError:
                break
        return consumed

    # Run all producers and one slow consumer concurrently
    producer_tasks = [producer(i) for i in range(5)]
    consumer_task = asyncio.create_task(consumer())

    await asyncio.gather(*producer_tasks)
    consumed = await consumer_task

    print("\n[Results per producer]")
    total_produced = 0
    for pid, count in sorted(producer_results.items()):
        total_produced += count
        print(f"  Producer {pid}: {count}/30 items accepted")

    print(f"\n  Total produced: {total_produced}/150")
    print(f"  Total consumed: {consumed}")
    print(f"  Dropped by backpressure: {queue.stats.total_dropped}")

    if queue.stats.total_dropped > 0:
        print("  ✅ Backpressure activated (some items dropped)")
    else:
        print("  ⚠️  No drops (consumer fast enough)")

    return True


async def main():
    """Run all BoundedQueue UAT tests."""
    results = []

    results.append(("Basic Backpressure", await test_backpressure_basic()))
    results.append(("Producer/Consumer", await test_producer_consumer()))
    results.append(("Statistics", await test_queue_statistics()))
    results.append(("Concurrent Producers", await test_concurrent_producers()))

    print()
    print("=" * 60)
    print("SUMMARY: BoundedQueue UAT")
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

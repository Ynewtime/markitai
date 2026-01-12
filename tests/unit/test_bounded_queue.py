"""Test BoundedQueue backpressure mechanism.

BoundedQueue prevents memory exhaustion during high-volume batch
processing by blocking producers when the queue is full.
"""

import asyncio

import pytest

from markit.utils.flow_control import BoundedQueue


@pytest.mark.asyncio
async def test_backpressure_basic() -> None:
    """Test basic backpressure behavior."""
    queue: BoundedQueue[str] = BoundedQueue(max_size=5, put_timeout=0.5)

    # Fill queue
    for i in range(5):
        result = await queue.put(f"item_{i}")
        assert result is True

    assert queue.full

    # Try to add more (should timeout)
    result = await queue.put("overflow_item")
    assert result is False
    assert queue.stats.total_dropped == 1


@pytest.mark.asyncio
async def test_producer_consumer() -> None:
    """Test producer-consumer pattern with backpressure."""
    queue: BoundedQueue[int] = BoundedQueue(max_size=10, put_timeout=1.0)
    processed: list[int] = []
    producer_done = asyncio.Event()

    async def producer() -> None:
        for i in range(50):
            await queue.put(i)
        producer_done.set()

    async def consumer() -> None:
        while not (producer_done.is_set() and queue.empty):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                processed.append(item)
                await asyncio.sleep(0.01)  # Simulate processing
                queue.task_done()
            except TimeoutError:
                if producer_done.is_set():
                    break

    await asyncio.gather(producer(), consumer())

    # All items should be accounted for (processed + dropped)
    total_accounted = len(processed) + queue.stats.total_dropped
    assert total_accounted == 50


@pytest.mark.asyncio
async def test_queue_statistics() -> None:
    """Test queue statistics tracking."""
    queue: BoundedQueue[str] = BoundedQueue(max_size=20)

    # Enqueue items
    for i in range(15):
        await queue.put(f"item_{i}")

    # Dequeue some
    for _ in range(10):
        await queue.get()
        queue.task_done()

    stats = queue.stats
    assert stats.total_enqueued == 15
    assert stats.total_dequeued == 10
    assert stats.total_completed == 10
    assert stats.current_size == 5


@pytest.mark.asyncio
async def test_concurrent_producers() -> None:
    """Test multiple producers with backpressure."""
    queue: BoundedQueue[tuple[int, int]] = BoundedQueue(max_size=20, put_timeout=0.5)
    producer_results: dict[int, int] = {}

    async def producer(producer_id: int) -> None:
        success_count = 0
        for i in range(30):
            result = await queue.put((producer_id, i))
            if result:
                success_count += 1
        producer_results[producer_id] = success_count

    async def consumer() -> int:
        consumed = 0
        while consumed < 100:
            try:
                await asyncio.wait_for(queue.get(), timeout=1.0)
                queue.task_done()
                consumed += 1
                await asyncio.sleep(0.01)  # Slow consumer
            except TimeoutError:
                break
        return consumed

    producer_tasks = [producer(i) for i in range(5)]
    consumer_task = asyncio.create_task(consumer())

    await asyncio.gather(*producer_tasks)
    await consumer_task

    # Total produced should be <= 150
    total_produced = sum(producer_results.values())
    assert total_produced <= 150

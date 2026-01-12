"""Tests for LLMTaskQueue with AIMD integration."""

import asyncio
from pathlib import Path

import pytest

from markit.exceptions import RateLimitError
from markit.llm.queue import LLMTask, LLMTaskQueue
from markit.utils.adaptive_limiter import AIMDConfig


class TestLLMTaskQueueStatic:
    """Tests for LLMTaskQueue with static concurrency."""

    @pytest.fixture
    def queue(self):
        """Create a queue with static concurrency."""
        return LLMTaskQueue(max_concurrent=5, max_pending=10)

    @pytest.mark.asyncio
    async def test_current_concurrency_static(self, queue):
        """Test current_concurrency returns max_concurrent for static mode."""
        assert queue.current_concurrency == 5

    @pytest.mark.asyncio
    async def test_adaptive_stats_none_in_static_mode(self, queue):
        """Test adaptive_stats returns None in static mode."""
        assert queue.adaptive_stats is None


class TestLLMTaskQueueAIMD:
    """Tests for LLMTaskQueue with AIMD integration."""

    @pytest.fixture
    def aimd_config(self):
        """Create AIMD config for testing."""
        return AIMDConfig(
            initial_concurrency=5,
            min_concurrency=1,
            max_concurrency=20,
            success_threshold=3,
            cooldown_seconds=0.0,
        )

    @pytest.fixture
    def aimd_queue(self, aimd_config):
        """Create a queue with AIMD enabled."""
        return LLMTaskQueue(
            max_pending=20,
            use_adaptive=True,
            aimd_config=aimd_config,
        )

    def test_aimd_initialization(self, aimd_queue):
        """Test AIMD is initialized correctly."""
        assert aimd_queue.use_adaptive is True
        assert aimd_queue._adaptive_limiter is not None
        assert aimd_queue.current_concurrency == 5

    def test_adaptive_stats_available(self, aimd_queue):
        """Test adaptive_stats returns stats in AIMD mode."""
        stats = aimd_queue.adaptive_stats
        assert stats is not None
        assert stats.current_concurrency == 5

    @pytest.mark.asyncio
    async def test_submit_success_records_aimd(self, aimd_queue):
        """Test successful task records success in AIMD."""

        async def success_coro():
            return "success"

        task = LLMTask(
            source_file=Path("/test/file.md"),
            task_type="chunk_enhancement",
            task_id="test_1",
            coro=success_coro(),
        )

        async_task = await aimd_queue.submit(task)
        result = await async_task

        assert result.success is True
        assert aimd_queue.adaptive_stats.total_successes == 1

    @pytest.mark.asyncio
    async def test_submit_rate_limit_records_aimd(self, aimd_queue):
        """Test rate limit error triggers AIMD decrease."""
        # Start at higher concurrency to see decrease
        aimd_queue._adaptive_limiter._current_concurrency = 10
        aimd_queue._adaptive_limiter._semaphore = asyncio.Semaphore(10)
        initial = aimd_queue.current_concurrency

        async def rate_limit_coro():
            raise RateLimitError(retry_after=5)

        task = LLMTask(
            source_file=Path("/test/file.md"),
            task_type="chunk_enhancement",
            task_id="test_1",
            coro=rate_limit_coro(),
        )

        async_task = await aimd_queue.submit(task)
        result = await async_task

        assert result.success is False
        assert "Rate limited" in result.error
        assert aimd_queue.adaptive_stats.rate_limit_hits == 1
        # Concurrency should decrease (10 * 0.5 = 5)
        assert aimd_queue.current_concurrency < initial

    @pytest.mark.asyncio
    async def test_submit_failure_records_aimd(self, aimd_queue):
        """Test non-rate-limit failure records failure in AIMD."""

        async def failure_coro():
            raise ValueError("Something went wrong")

        task = LLMTask(
            source_file=Path("/test/file.md"),
            task_type="chunk_enhancement",
            task_id="test_1",
            coro=failure_coro(),
        )

        async_task = await aimd_queue.submit(task)
        result = await async_task

        assert result.success is False
        assert aimd_queue.adaptive_stats.total_failures == 1

    @pytest.mark.asyncio
    async def test_concurrency_increase_after_threshold(self, aimd_queue):
        """Test concurrency increases after success threshold."""
        initial = aimd_queue.current_concurrency

        async def success_coro():
            return "success"

        # Submit enough tasks to trigger increase (threshold = 3)
        for i in range(4):
            task = LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id=f"test_{i}",
                coro=success_coro(),
            )
            async_task = await aimd_queue.submit(task)
            await async_task

        assert aimd_queue.current_concurrency > initial
        assert aimd_queue.adaptive_stats.increase_count >= 1

    @pytest.mark.asyncio
    async def test_reset_clears_aimd(self, aimd_queue):
        """Test reset clears AIMD state."""

        # Record some activity
        async def success_coro():
            return "success"

        task = LLMTask(
            source_file=Path("/test/file.md"),
            task_type="chunk_enhancement",
            task_id="test_1",
            coro=success_coro(),
        )
        async_task = await aimd_queue.submit(task)
        await async_task

        assert aimd_queue.adaptive_stats.total_successes == 1

        # Reset
        aimd_queue.reset()

        assert aimd_queue.adaptive_stats.total_successes == 0
        assert aimd_queue.submitted_count == 0


class TestLLMTaskQueueBackpressure:
    """Tests for LLMTaskQueue backpressure mechanism."""

    @pytest.mark.asyncio
    async def test_backpressure_blocks(self):
        """Test backpressure blocks when max_pending reached."""
        queue = LLMTaskQueue(max_concurrent=1, max_pending=2)

        submitted = []

        async def slow_coro():
            await asyncio.sleep(0.5)
            return "done"

        # Submit 2 tasks (max_pending)
        for i in range(2):
            task = LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id=f"test_{i}",
                coro=slow_coro(),
            )
            submitted.append(await queue.submit(task))

        # 3rd task should be blocked
        task3 = LLMTask(
            source_file=Path("/test/file.md"),
            task_type="chunk_enhancement",
            task_id="test_2",
            coro=slow_coro(),
        )

        # Use wait_for to detect blocking
        try:
            await asyncio.wait_for(queue.submit(task3), timeout=0.1)
            blocked = False
        except TimeoutError:
            blocked = True

        # Wait for tasks to complete
        await asyncio.gather(*submitted)

        assert blocked is True


class TestLLMTaskQueueIntegration:
    """Integration tests for LLMTaskQueue."""

    @pytest.mark.asyncio
    async def test_batch_submit_with_aimd(self):
        """Test batch submission with AIMD."""
        config = AIMDConfig(
            initial_concurrency=3,
            success_threshold=2,
            cooldown_seconds=0.0,
        )
        queue = LLMTaskQueue(use_adaptive=True, aimd_config=config)

        async def success_coro():
            return "success"

        tasks = [
            LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id=f"test_{i}",
                coro=success_coro(),
            )
            for i in range(5)
        ]

        await queue.submit_batch(tasks)
        results = await queue.wait_all()

        assert len(results) == 5
        assert all(r.success for r in results)
        assert queue.adaptive_stats.total_successes == 5

    @pytest.mark.asyncio
    async def test_mixed_results_with_aimd(self):
        """Test mixed success/failure results with AIMD."""
        config = AIMDConfig(
            initial_concurrency=5,
            success_threshold=3,
            cooldown_seconds=0.0,
        )
        queue = LLMTaskQueue(use_adaptive=True, aimd_config=config)

        async def success_coro():
            return "success"

        async def failure_coro():
            raise ValueError("Error")

        tasks = [
            LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id="success_1",
                coro=success_coro(),
            ),
            LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id="failure_1",
                coro=failure_coro(),
            ),
            LLMTask(
                source_file=Path("/test/file.md"),
                task_type="chunk_enhancement",
                task_id="success_2",
                coro=success_coro(),
            ),
        ]

        await queue.submit_batch(tasks)
        results = await queue.wait_all()

        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        assert succeeded == 2
        assert failed == 1
        assert queue.adaptive_stats.total_successes == 2
        assert queue.adaptive_stats.total_failures == 1

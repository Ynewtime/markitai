"""Tests for utils/executor.py module."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from markitai.utils.executor import (
    _CONVERTER_MAX_WORKERS,
    get_converter_executor,
    run_in_converter_thread,
    shutdown_converter_executor,
)


class TestGetConverterExecutor:
    """Tests for get_converter_executor function."""

    def teardown_method(self):
        """Clean up executor after each test."""
        shutdown_converter_executor()

    def test_returns_thread_pool_executor(self):
        """Test that it returns a ThreadPoolExecutor instance."""
        executor = get_converter_executor()
        assert isinstance(executor, ThreadPoolExecutor)

    def test_returns_same_instance(self):
        """Test that multiple calls return the same executor instance."""
        executor1 = get_converter_executor()
        executor2 = get_converter_executor()
        assert executor1 is executor2

    def test_max_workers_limit(self):
        """Test that max_workers is capped at 8."""
        assert _CONVERTER_MAX_WORKERS <= 8

    def test_thread_safe_initialization(self):
        """Test that concurrent calls return the same executor (thread safety)."""
        results: list[ThreadPoolExecutor] = []
        errors: list[Exception] = []

        def get_executor():
            try:
                executor = get_converter_executor()
                results.append(executor)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that call get_converter_executor simultaneously
        threads = [threading.Thread(target=get_executor) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all threads got the same executor instance
        assert len(results) == 10
        first_executor = results[0]
        for executor in results[1:]:
            assert executor is first_executor


class TestRunInConverterThread:
    """Tests for run_in_converter_thread function."""

    def teardown_method(self):
        """Clean up executor after each test."""
        shutdown_converter_executor()

    @pytest.mark.asyncio
    async def test_runs_function_and_returns_result(self):
        """Test that the function is executed and result is returned."""

        def compute(x: int, y: int) -> int:
            return x + y

        result = await run_in_converter_thread(compute, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_runs_function_with_kwargs(self):
        """Test that kwargs are passed correctly."""

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await run_in_converter_thread(greet, "World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_runs_in_thread_pool(self):
        """Test that function runs in a separate thread."""
        main_thread_id = threading.current_thread().ident
        thread_ids: list[int | None] = []

        def capture_thread_id() -> int | None:
            tid = threading.current_thread().ident
            thread_ids.append(tid)
            return tid

        await run_in_converter_thread(capture_thread_id)

        assert len(thread_ids) == 1
        # The function should run in a different thread
        assert thread_ids[0] != main_thread_id

    @pytest.mark.asyncio
    async def test_thread_name_prefix(self):
        """Test that thread has correct name prefix."""
        thread_names: list[str] = []

        def capture_thread_name() -> str:
            name = threading.current_thread().name
            thread_names.append(name)
            return name

        await run_in_converter_thread(capture_thread_name)

        assert len(thread_names) == 1
        assert thread_names[0].startswith("markitai-converter")

    @pytest.mark.asyncio
    async def test_propagates_exception(self):
        """Test that exceptions from the function are propagated."""

        def raise_error() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await run_in_converter_thread(raise_error)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test that multiple tasks can run concurrently."""
        execution_times: list[float] = []

        def slow_task(task_id: int) -> int:
            start = time.time()
            time.sleep(0.1)  # 100ms sleep
            execution_times.append(time.time() - start)
            return task_id

        start = time.time()
        # Run 4 tasks concurrently
        results = await asyncio.gather(
            run_in_converter_thread(slow_task, 1),
            run_in_converter_thread(slow_task, 2),
            run_in_converter_thread(slow_task, 3),
            run_in_converter_thread(slow_task, 4),
        )
        total_time = time.time() - start

        assert sorted(results) == [1, 2, 3, 4]
        # If truly concurrent, total time should be close to 0.1s, not 0.4s
        # Allow some tolerance for thread scheduling overhead
        assert total_time < 0.3, f"Tasks not concurrent: {total_time:.2f}s"


class TestShutdownConverterExecutor:
    """Tests for shutdown_converter_executor function."""

    def teardown_method(self):
        """Ensure cleanup after each test."""
        shutdown_converter_executor()

    def test_shutdown_clears_executor(self):
        """Test that shutdown clears the global executor."""
        # First, create an executor
        executor1 = get_converter_executor()
        assert executor1 is not None

        # Shutdown
        shutdown_converter_executor()

        # Get a new executor - should be a different instance
        executor2 = get_converter_executor()
        assert executor2 is not None
        assert executor2 is not executor1

    def test_shutdown_when_none_is_safe(self):
        """Test that shutdown when executor is None doesn't raise."""
        # Ensure executor is None
        shutdown_converter_executor()

        # Should not raise
        shutdown_converter_executor()

    def test_shutdown_waits_for_tasks(self):
        """Test that shutdown waits for pending tasks to complete."""
        results: list[int] = []

        def slow_task() -> int:
            time.sleep(0.1)
            results.append(1)
            return 1

        executor = get_converter_executor()
        # Submit a task
        future = executor.submit(slow_task)

        # Shutdown should wait for the task
        shutdown_converter_executor()

        # Task should have completed
        assert len(results) == 1
        assert future.result() == 1


class TestExecutorIntegration:
    """Integration tests for executor module."""

    def teardown_method(self):
        """Clean up executor after each test."""
        shutdown_converter_executor()

    @pytest.mark.asyncio
    async def test_mixed_sync_async_usage(self):
        """Test using executor from both sync and async contexts."""
        # Get executor synchronously
        executor1 = get_converter_executor()

        # Use run_in_converter_thread (async)
        result = await run_in_converter_thread(lambda: 42)
        assert result == 42

        # Get executor again - should be same instance
        executor2 = get_converter_executor()
        assert executor1 is executor2

    @pytest.mark.asyncio
    async def test_executor_reusable_after_shutdown(self):
        """Test that executor can be recreated after shutdown."""
        # Use executor
        result1 = await run_in_converter_thread(lambda: 1)
        assert result1 == 1

        # Shutdown
        shutdown_converter_executor()

        # Use again - should work with new executor
        result2 = await run_in_converter_thread(lambda: 2)
        assert result2 == 2

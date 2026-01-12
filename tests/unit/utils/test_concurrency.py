"""Tests for concurrency management module."""

import asyncio

import pytest

from markit.utils.concurrency import (
    BatchProcessor,
    ConcurrencyManager,
    TaskResult,
)


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_success_result(self):
        """Test successful task result."""
        result = TaskResult(
            item="test_item",
            success=True,
            result="completed",
        )

        assert result.item == "test_item"
        assert result.success is True
        assert result.result == "completed"
        assert result.error is None

    def test_failure_result(self):
        """Test failed task result."""
        result = TaskResult(
            item="test_item",
            success=False,
            error="Something went wrong",
        )

        assert result.item == "test_item"
        assert result.success is False
        assert result.result is None
        assert result.error == "Something went wrong"


class TestConcurrencyManagerInit:
    """Tests for ConcurrencyManager initialization."""

    def test_default_values(self):
        """Test default initialization."""
        manager = ConcurrencyManager()

        assert manager.file_workers == 8
        assert manager.image_workers == 16
        assert manager.llm_workers == 10

    def test_custom_values(self):
        """Test custom initialization."""
        manager = ConcurrencyManager(
            file_workers=4,
            image_workers=8,
            llm_workers=5,
        )

        assert manager.file_workers == 4
        assert manager.image_workers == 8
        assert manager.llm_workers == 5

    def test_semaphores_initially_none(self):
        """Test that semaphores are not created until needed."""
        manager = ConcurrencyManager()

        assert manager._file_semaphore is None
        assert manager._image_semaphore is None
        assert manager._llm_semaphore is None


class TestConcurrencyManagerSemaphores:
    """Tests for semaphore creation and retrieval."""

    def test_file_semaphore_creation(self):
        """Test file semaphore is created on demand."""
        manager = ConcurrencyManager(file_workers=4)

        semaphore = manager._get_file_semaphore()

        assert semaphore is not None
        assert isinstance(semaphore, asyncio.Semaphore)
        assert manager._file_semaphore is semaphore

    def test_file_semaphore_reused(self):
        """Test file semaphore is reused."""
        manager = ConcurrencyManager()

        sem1 = manager._get_file_semaphore()
        sem2 = manager._get_file_semaphore()

        assert sem1 is sem2

    def test_image_semaphore_creation(self):
        """Test image semaphore is created on demand."""
        manager = ConcurrencyManager(image_workers=8)

        semaphore = manager._get_image_semaphore()

        assert semaphore is not None
        assert isinstance(semaphore, asyncio.Semaphore)
        assert manager._image_semaphore is semaphore

    def test_llm_semaphore_creation(self):
        """Test LLM semaphore is created on demand."""
        manager = ConcurrencyManager(llm_workers=5)

        semaphore = manager._get_llm_semaphore()

        assert semaphore is not None
        assert isinstance(semaphore, asyncio.Semaphore)
        assert manager._llm_semaphore is semaphore

    def test_get_llm_semaphore_public(self):
        """Test public get_llm_semaphore method."""
        manager = ConcurrencyManager(llm_workers=5)

        semaphore = manager.get_llm_semaphore()

        assert semaphore is not None
        assert semaphore is manager._get_llm_semaphore()


class TestConcurrencyManagerRunTasks:
    """Tests for run_*_task methods."""

    @pytest.mark.asyncio
    async def test_run_file_task(self):
        """Test running file task with semaphore."""
        manager = ConcurrencyManager(file_workers=2)

        async def task():
            return "done"

        result = await manager.run_file_task(task())

        assert result == "done"

    @pytest.mark.asyncio
    async def test_run_image_task(self):
        """Test running image task with semaphore."""
        manager = ConcurrencyManager(image_workers=2)

        async def task():
            return "processed"

        result = await manager.run_image_task(task())

        assert result == "processed"

    @pytest.mark.asyncio
    async def test_run_llm_task(self):
        """Test running LLM task with semaphore."""
        manager = ConcurrencyManager(llm_workers=2)

        async def task():
            return "response"

        result = await manager.run_llm_task(task())

        assert result == "response"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is properly limited."""
        manager = ConcurrencyManager(file_workers=2)
        active_count = 0
        max_active = 0

        async def task():
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return "done"

        # Run 5 tasks with limit of 2
        tasks = [manager.run_file_task(task()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Max active should be limited to 2
        assert max_active <= 2


class TestConcurrencyManagerMapTasks:
    """Tests for map_*_tasks methods."""

    @pytest.mark.asyncio
    async def test_map_file_tasks(self):
        """Test mapping file tasks."""
        manager = ConcurrencyManager(file_workers=2)

        async def process(x: int) -> int:
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await manager.map_file_tasks(items, process)

        assert len(results) == 5
        assert all(r.success for r in results)
        assert [r.result for r in results] == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_map_image_tasks(self):
        """Test mapping image tasks."""
        manager = ConcurrencyManager(image_workers=2)

        async def process(x: str) -> str:
            return x.upper()

        items = ["a", "b", "c"]
        results = await manager.map_image_tasks(items, process)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert [r.result for r in results] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_map_llm_tasks(self):
        """Test mapping LLM tasks."""
        manager = ConcurrencyManager(llm_workers=2)

        async def process(prompt: str) -> str:
            return f"Response to: {prompt}"

        items = ["Hello", "World"]
        results = await manager.map_llm_tasks(items, process)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_map_tasks_with_failure(self):
        """Test mapping tasks when some fail."""
        manager = ConcurrencyManager(file_workers=2)

        async def process(x: int) -> int:
            if x == 3:
                raise ValueError("Cannot process 3")
            return x * 2

        items = [1, 2, 3, 4]
        results = await manager.map_file_tasks(items, process)

        assert len(results) == 4
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False
        assert results[2].error is not None and "Cannot process 3" in results[2].error
        assert results[3].success is True

    @pytest.mark.asyncio
    async def test_map_tasks_with_progress_callback(self):
        """Test mapping tasks with progress callback."""
        manager = ConcurrencyManager(file_workers=2)
        progress_calls = []

        async def process(x: int) -> int:
            return x * 2

        def on_progress(item, result, error):
            progress_calls.append((item, result, error))

        items = [1, 2, 3]
        await manager.map_file_tasks(items, process, on_progress=on_progress)

        assert len(progress_calls) == 3
        # Check that all callbacks were called with correct items
        callback_items = [call[0] for call in progress_calls]
        assert set(callback_items) == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_map_tasks_progress_on_error(self):
        """Test progress callback is called on error."""
        manager = ConcurrencyManager(file_workers=2)
        progress_calls = []

        async def process(x: int) -> int:
            if x == 2:
                raise ValueError("Error")
            return x * 2

        def on_progress(item, result, error):
            progress_calls.append((item, result, error))

        items = [1, 2, 3]
        await manager.map_file_tasks(items, process, on_progress=on_progress)

        # Find the error callback
        error_calls = [call for call in progress_calls if call[2] is not None]
        assert len(error_calls) == 1
        assert error_calls[0][0] == 2


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_init(self):
        """Test initialization."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        assert processor.concurrency is manager
        assert processor.completed == 0
        assert processor.failed == 0
        assert processor.total == 0

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test processing a batch."""
        manager = ConcurrencyManager(file_workers=2)
        processor = BatchProcessor(concurrency=manager)

        async def process(x: int) -> int:
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(items, process)

        assert len(results) == 5
        assert processor.total == 5
        assert processor.completed == 5
        assert processor.failed == 0

    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self):
        """Test processing batch with failures."""
        manager = ConcurrencyManager(file_workers=2)
        processor = BatchProcessor(concurrency=manager)

        async def process(x: int) -> int:
            if x % 2 == 0:
                raise ValueError("Even number")
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(items, process)

        assert len(results) == 5
        assert processor.total == 5
        assert processor.completed == 3  # 1, 3, 5
        assert processor.failed == 2  # 2, 4

    @pytest.mark.asyncio
    async def test_process_batch_with_callback(self):
        """Test processing batch with completion callback."""
        manager = ConcurrencyManager(file_workers=2)
        processor = BatchProcessor(concurrency=manager)
        callbacks = []

        async def process(x: int) -> int:
            return x * 2

        def on_complete(item, result):
            callbacks.append((item, result))

        items = [1, 2, 3]
        await processor.process_batch(items, process, on_item_complete=on_complete)

        assert len(callbacks) == 3

    def test_progress_property(self):
        """Test progress property."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        processor.total = 10
        processor.completed = 3
        processor.failed = 2

        assert processor.progress == 0.5  # 5 out of 10

    def test_progress_zero_total(self):
        """Test progress with zero total."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        assert processor.progress == 0.0

    def test_success_rate_property(self):
        """Test success_rate property."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        processor.completed = 7
        processor.failed = 3

        assert processor.success_rate == 0.7

    def test_success_rate_zero_processed(self):
        """Test success_rate with zero processed."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        assert processor.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success_rate with all success."""
        manager = ConcurrencyManager()
        processor = BatchProcessor(concurrency=manager)

        processor.completed = 10
        processor.failed = 0

        assert processor.success_rate == 1.0

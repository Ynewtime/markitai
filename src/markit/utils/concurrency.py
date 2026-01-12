"""Concurrency management for batch processing."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from markit.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult(Generic[T]):
    """Result of a concurrent task."""

    item: T
    success: bool
    result: Any | None = None
    error: str | None = None


class ConcurrencyManager:
    """Manages concurrent task execution with configurable limits.

    Provides separate semaphores for different types of operations:
    - File processing (I/O bound)
    - Image processing (CPU bound)
    - LLM API calls (rate limited)
    """

    def __init__(
        self,
        file_workers: int = 8,
        image_workers: int = 16,
        llm_workers: int = 10,
    ) -> None:
        """Initialize the concurrency manager.

        Args:
            file_workers: Maximum concurrent file operations
            image_workers: Maximum concurrent image operations
            llm_workers: Maximum concurrent LLM API calls
        """
        self.file_workers = file_workers
        self.image_workers = image_workers
        self.llm_workers = llm_workers

        self._file_semaphore: asyncio.Semaphore | None = None
        self._image_semaphore: asyncio.Semaphore | None = None
        self._llm_semaphore: asyncio.Semaphore | None = None

    def _get_file_semaphore(self) -> asyncio.Semaphore:
        """Get or create file semaphore."""
        if self._file_semaphore is None:
            self._file_semaphore = asyncio.Semaphore(self.file_workers)
        return self._file_semaphore

    def _get_image_semaphore(self) -> asyncio.Semaphore:
        """Get or create image semaphore."""
        if self._image_semaphore is None:
            self._image_semaphore = asyncio.Semaphore(self.image_workers)
        return self._image_semaphore

    def _get_llm_semaphore(self) -> asyncio.Semaphore:
        """Get or create LLM semaphore."""
        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(self.llm_workers)
        return self._llm_semaphore

    def get_llm_semaphore(self) -> asyncio.Semaphore:
        """Get the LLM semaphore for external use (e.g., LLMTaskQueue).

        This allows the LLMTaskQueue to share the same semaphore
        for global rate limiting across all LLM operations.

        Returns:
            The LLM semaphore instance
        """
        return self._get_llm_semaphore()

    async def run_file_task(
        self,
        coro: Awaitable[R],
    ) -> R:
        """Run a file task with semaphore protection.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of the coroutine
        """
        async with self._get_file_semaphore():
            return await coro

    async def run_image_task(
        self,
        coro: Awaitable[R],
    ) -> R:
        """Run an image task with semaphore protection.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of the coroutine
        """
        async with self._get_image_semaphore():
            return await coro

    async def run_llm_task(
        self,
        coro: Awaitable[R],
    ) -> R:
        """Run an LLM task with semaphore protection.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of the coroutine
        """
        async with self._get_llm_semaphore():
            return await coro

    async def map_file_tasks(
        self,
        items: list[T],
        func: Callable[[T], Awaitable[R]],
        on_progress: Callable[[T, R | None, Exception | None], None] | None = None,
    ) -> list[TaskResult[T]]:
        """Process items concurrently with file concurrency limits.

        Args:
            items: Items to process
            func: Async function to apply to each item
            on_progress: Optional callback for progress updates

        Returns:
            List of TaskResult objects
        """
        return await self._map_with_semaphore(
            items=items,
            func=func,
            semaphore=self._get_file_semaphore(),
            on_progress=on_progress,
        )

    async def map_image_tasks(
        self,
        items: list[T],
        func: Callable[[T], Awaitable[R]],
        on_progress: Callable[[T, R | None, Exception | None], None] | None = None,
    ) -> list[TaskResult[T]]:
        """Process items concurrently with image concurrency limits.

        Args:
            items: Items to process
            func: Async function to apply to each item
            on_progress: Optional callback for progress updates

        Returns:
            List of TaskResult objects
        """
        return await self._map_with_semaphore(
            items=items,
            func=func,
            semaphore=self._get_image_semaphore(),
            on_progress=on_progress,
        )

    async def map_llm_tasks(
        self,
        items: list[T],
        func: Callable[[T], Awaitable[R]],
        on_progress: Callable[[T, R | None, Exception | None], None] | None = None,
    ) -> list[TaskResult[T]]:
        """Process items concurrently with LLM concurrency limits.

        Args:
            items: Items to process
            func: Async function to apply to each item
            on_progress: Optional callback for progress updates

        Returns:
            List of TaskResult objects
        """
        return await self._map_with_semaphore(
            items=items,
            func=func,
            semaphore=self._get_llm_semaphore(),
            on_progress=on_progress,
        )

    async def _map_with_semaphore(
        self,
        items: list[T],
        func: Callable[[T], Awaitable[R]],
        semaphore: asyncio.Semaphore,
        on_progress: Callable[[T, R | None, Exception | None], None] | None = None,
    ) -> list[TaskResult[T]]:
        """Process items with semaphore-controlled concurrency.

        Args:
            items: Items to process
            func: Async function to apply
            semaphore: Semaphore for concurrency control
            on_progress: Optional progress callback

        Returns:
            List of TaskResult objects
        """

        async def process_item(item: T) -> TaskResult[T]:
            async with semaphore:
                try:
                    result = await func(item)
                    if on_progress:
                        on_progress(item, result, None)
                    return TaskResult(
                        item=item,
                        success=True,
                        result=result,
                    )
                except Exception as e:
                    log.warning(
                        "Task failed",
                        item=str(item),
                        error=str(e),
                    )
                    if on_progress:
                        on_progress(item, None, e)
                    return TaskResult(
                        item=item,
                        success=False,
                        error=str(e),
                    )

        tasks = [process_item(item) for item in items]
        return await asyncio.gather(*tasks)


class BatchProcessor:
    """High-level batch processor with progress tracking."""

    def __init__(
        self,
        concurrency: ConcurrencyManager,
    ) -> None:
        """Initialize the batch processor.

        Args:
            concurrency: Concurrency manager
        """
        self.concurrency = concurrency
        self.completed = 0
        self.failed = 0
        self.total = 0

    async def process_batch(
        self,
        items: list[T],
        processor: Callable[[T], Awaitable[R]],
        on_item_complete: Callable[[T, TaskResult[T]], None] | None = None,
    ) -> list[TaskResult[T]]:
        """Process a batch of items.

        Args:
            items: Items to process
            processor: Async function to process each item
            on_item_complete: Callback when item completes

        Returns:
            List of task results
        """
        self.total = len(items)
        self.completed = 0
        self.failed = 0

        def progress_callback(
            item: T,
            result: R | None,
            error: Exception | None,
        ) -> None:
            if error:
                self.failed += 1
            else:
                self.completed += 1

            if on_item_complete:
                task_result = TaskResult(
                    item=item,
                    success=error is None,
                    result=result,
                    error=str(error) if error else None,
                )
                on_item_complete(item, task_result)

        results = await self.concurrency.map_file_tasks(
            items=items,
            func=processor,
            on_progress=progress_callback,
        )

        return results

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        return self.completed / processed

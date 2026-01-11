"""Global LLM task queue for batch processing with rate limiting."""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMTask:
    """Represents an LLM task with source tracking for error correlation.

    Attributes:
        source_file: The file that originated this task
        task_type: Type of LLM task
        task_id: Unique identifier for this task (e.g., image filename)
        coro: The coroutine to execute
    """

    source_file: Path
    task_type: Literal["image_analysis", "chunk_enhancement", "summary"]
    task_id: str
    coro: Coroutine[Any, Any, Any]


@dataclass
class LLMTaskResult:
    """Result of an LLM task execution.

    Attributes:
        task: The original task
        success: Whether the task completed successfully
        result: The result if successful
        error: The error message if failed
        model: Model name used for this task
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        estimated_cost: Estimated cost in USD
        duration: Execution duration in seconds
        start_time: When the task started (Unix timestamp)
        end_time: When the task completed (Unix timestamp)
    """

    task: LLMTask
    success: bool
    result: Any = None
    error: str | None = None
    # LLM statistics
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0
    duration: float = 0.0
    # Timing for wall-clock duration tracking
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def source_file(self) -> Path:
        """Convenience accessor for source file."""
        return self.task.source_file

    @property
    def task_type(self) -> str:
        """Convenience accessor for task type."""
        return self.task.task_type

    @property
    def task_id(self) -> str:
        """Convenience accessor for task ID."""
        return self.task.task_id


class LLMTaskQueue:
    """Global queue for all LLM tasks across files with rate limiting.

    This queue provides:
    - Concurrent execution with configurable limit (semaphore)
    - Backpressure to prevent memory exhaustion (pending limit)
    - Task tracking for error correlation back to source files

    Usage:
        queue = LLMTaskQueue(max_concurrent=10, max_pending=100)

        # Submit tasks
        await queue.submit(LLMTask(...))
        await queue.submit(LLMTask(...))

        # Wait for all to complete
        results = await queue.wait_all()
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_pending: int = 100,
    ) -> None:
        """Initialize the LLM task queue.

        Args:
            max_concurrent: Maximum number of concurrent LLM API calls
            max_pending: Maximum number of pending tasks (backpressure)
        """
        self.max_concurrent = max_concurrent
        self.max_pending = max_pending

        self._semaphore: asyncio.Semaphore | None = None
        self._pending_semaphore: asyncio.Semaphore | None = None
        self._tasks: list[asyncio.Task[LLMTaskResult]] = []
        self._submitted_count = 0
        self._completed_count = 0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the concurrency semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    def _get_pending_semaphore(self) -> asyncio.Semaphore:
        """Get or create the backpressure semaphore."""
        if self._pending_semaphore is None:
            self._pending_semaphore = asyncio.Semaphore(self.max_pending)
        return self._pending_semaphore

    async def submit(self, task: LLMTask) -> asyncio.Task[LLMTaskResult]:
        """Submit an LLM task for execution with rate limiting.

        This method will block if max_pending tasks are already queued
        (backpressure mechanism).

        Args:
            task: The LLM task to submit

        Returns:
            The asyncio.Task wrapping the execution
        """
        from time import time

        from markit.llm.base import LLMTaskResultWithStats

        # Backpressure: wait if too many tasks pending
        pending_sem = self._get_pending_semaphore()
        await pending_sem.acquire()

        async def execute_task() -> LLMTaskResult:
            """Execute the task with semaphore protection."""
            semaphore = self._get_semaphore()
            start_time = time()
            try:
                async with semaphore:
                    log.debug(
                        "Executing LLM task",
                        task_type=task.task_type,
                        task_id=task.task_id,
                        source=str(task.source_file.name),
                    )
                    result = await task.coro
                    end_time = time()
                    duration = end_time - start_time

                    # Extract statistics if result is LLMTaskResultWithStats
                    if isinstance(result, LLMTaskResultWithStats):
                        return LLMTaskResult(
                            task=task,
                            success=True,
                            result=result.result,  # Extract the actual result
                            model=result.model,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            estimated_cost=result.estimated_cost,
                            duration=duration,
                            start_time=start_time,
                            end_time=end_time,
                        )
                    return LLMTaskResult(
                        task=task,
                        success=True,
                        result=result,
                        duration=duration,
                        start_time=start_time,
                        end_time=end_time,
                    )
            except Exception as e:
                end_time = time()
                duration = end_time - start_time
                log.warning(
                    "LLM task failed",
                    task_type=task.task_type,
                    task_id=task.task_id,
                    source=str(task.source_file.name),
                    error=str(e),
                )
                return LLMTaskResult(
                    task=task,
                    success=False,
                    error=str(e),
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                )
            finally:
                # Release backpressure slot
                pending_sem.release()
                self._completed_count += 1

        async_task = asyncio.create_task(execute_task())
        self._tasks.append(async_task)
        self._submitted_count += 1

        # Note: Individual task submission logs removed to reduce log noise
        # Use submit_batch() for batch logging or check pending_count property

        return async_task

    async def submit_batch(self, tasks: list[LLMTask]) -> list[asyncio.Task[LLMTaskResult]]:
        """Submit multiple tasks at once.

        Args:
            tasks: List of LLM tasks to submit

        Returns:
            List of asyncio.Tasks
        """
        if tasks:
            # Log batch submission summary instead of individual tasks
            task_types = {}
            for t in tasks:
                task_types[t.task_type] = task_types.get(t.task_type, 0) + 1
            log.debug(
                "Submitting LLM task batch",
                count=len(tasks),
                types=task_types,
            )
        return [await self.submit(task) for task in tasks]

    async def wait_all(self) -> list[LLMTaskResult]:
        """Wait for all submitted tasks to complete.

        Returns:
            List of LLMTaskResult objects (preserves submission order)
        """
        if not self._tasks:
            return []

        log.info(
            "Waiting for LLM tasks to complete",
            total=len(self._tasks),
        )

        results = await asyncio.gather(*self._tasks, return_exceptions=True)

        # Convert any unexpected exceptions to LLMTaskResult
        final_results: list[LLMTaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, LLMTaskResult):
                final_results.append(result)
            elif isinstance(result, Exception):
                # This shouldn't happen as execute_task catches exceptions,
                # but handle it gracefully just in case
                log.error(
                    "Unexpected exception in LLM task",
                    error=str(result),
                )
                # We don't have the original task here, so create a minimal result
                final_results.append(
                    LLMTaskResult(
                        task=LLMTask(
                            source_file=Path("unknown"),
                            task_type="chunk_enhancement",
                            task_id=f"unknown_{i}",
                            coro=asyncio.sleep(0),  # Dummy coro
                        ),
                        success=False,
                        error=f"Unexpected error: {result}",
                    )
                )
            else:
                final_results.append(result)

        succeeded = sum(1 for r in final_results if r.success)
        failed = len(final_results) - succeeded

        log.info(
            "LLM tasks completed",
            total=len(final_results),
            succeeded=succeeded,
            failed=failed,
        )

        return final_results

    def get_results_for_file(
        self,
        results: list[LLMTaskResult],
        source_file: Path,
    ) -> list[LLMTaskResult]:
        """Filter results for a specific source file.

        Args:
            results: List of all results
            source_file: The source file to filter by

        Returns:
            Results belonging to the specified file
        """
        return [r for r in results if r.source_file == source_file]

    def get_results_by_type(
        self,
        results: list[LLMTaskResult],
        task_type: str,
    ) -> list[LLMTaskResult]:
        """Filter results by task type.

        Args:
            results: List of all results
            task_type: The task type to filter by

        Returns:
            Results of the specified type
        """
        return [r for r in results if r.task_type == task_type]

    @property
    def pending_count(self) -> int:
        """Number of tasks pending (submitted but not completed)."""
        return self._submitted_count - self._completed_count

    @property
    def submitted_count(self) -> int:
        """Total number of tasks submitted."""
        return self._submitted_count

    @property
    def completed_count(self) -> int:
        """Number of tasks completed."""
        return self._completed_count

    def reset(self) -> None:
        """Reset the queue state for reuse.

        Note: Only call this after wait_all() has completed.
        """
        self._tasks.clear()
        self._submitted_count = 0
        self._completed_count = 0
        # Keep semaphores - they can be reused

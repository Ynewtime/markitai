"""Flow control utilities for backpressure and failure handling.

This module provides reusable components for:
- BoundedQueue: Generic producer-consumer queue with backpressure
- DeadLetterQueue: Tracking and managing failed items with retry logic

These components are designed to be independent and can be used across
different parts of the application.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from markit.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# BoundedQueue - Generic backpressure queue
# =============================================================================


@dataclass
class QueueStats:
    """Statistics for bounded queue operations.

    Attributes:
        current_size: Current number of items in queue
        max_size: Maximum queue capacity
        total_enqueued: Total items added to queue
        total_dequeued: Total items removed from queue
        total_dropped: Items dropped due to timeout
        total_completed: Items marked as done via task_done()
    """

    current_size: int
    max_size: int
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_dropped: int = 0
    total_completed: int = 0

    @property
    def throughput_ratio(self) -> float:
        """Ratio of completed items to enqueued items."""
        if self.total_enqueued == 0:
            return 0.0
        return self.total_completed / self.total_enqueued

    @property
    def drop_rate(self) -> float:
        """Percentage of items dropped."""
        if self.total_enqueued == 0:
            return 0.0
        return (self.total_dropped / self.total_enqueued) * 100


class BoundedQueue(Generic[T]):
    """Bounded queue with backpressure for producer-consumer pattern.

    This queue prevents memory exhaustion by blocking producers when
    the queue is full. It's a generic implementation that can hold
    any type of item.

    Features:
    - Maximum queue size to prevent memory overflow
    - Backpressure: producers block when queue is full
    - Configurable timeout for put operations
    - Statistics tracking for monitoring

    Example usage:
        ```python
        queue: BoundedQueue[LLMTask] = BoundedQueue(max_size=100)

        # Producer
        success = await queue.put(task)
        if not success:
            log.warning("Queue full, task dropped")

        # Consumer
        task = await queue.get()
        try:
            await process(task)
        finally:
            queue.task_done()

        # Wait for all tasks
        await queue.join()
        ```
    """

    def __init__(
        self,
        max_size: int = 1000,
        put_timeout: float = 60.0,
    ) -> None:
        """Initialize bounded queue.

        Args:
            max_size: Maximum queue size (producers block when full)
            put_timeout: Timeout in seconds for put operations
        """
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._max_size = max_size
        self._put_timeout = put_timeout
        self._stats = QueueStats(current_size=0, max_size=max_size)

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def max_size(self) -> int:
        """Get maximum queue size."""
        return self._max_size

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    async def put(self, item: T) -> bool:
        """Put item in queue (blocks if full, with timeout).

        Args:
            item: Item to add to queue

        Returns:
            True if item was added, False if dropped due to timeout
        """
        self._stats.total_enqueued += 1

        try:
            await asyncio.wait_for(
                self._queue.put(item),
                timeout=self._put_timeout,
            )
            self._stats.current_size = self._queue.qsize()
            return True
        except TimeoutError:
            self._stats.total_dropped += 1
            log.warning(
                "Queue put timed out, dropping item",
                timeout=self._put_timeout,
                queue_size=self._queue.qsize(),
            )
            return False

    def put_nowait(self, item: T) -> bool:
        """Put item in queue without waiting.

        Args:
            item: Item to add to queue

        Returns:
            True if item was added, False if queue is full
        """
        self._stats.total_enqueued += 1

        try:
            self._queue.put_nowait(item)
            self._stats.current_size = self._queue.qsize()
            return True
        except asyncio.QueueFull:
            self._stats.total_dropped += 1
            return False

    async def get(self) -> T:
        """Get item from queue (blocks if empty).

        Returns:
            The next item from the queue
        """
        item = await self._queue.get()
        self._stats.current_size = self._queue.qsize()
        self._stats.total_dequeued += 1
        return item

    def get_nowait(self) -> T:
        """Get item from queue without waiting.

        Returns:
            The next item from the queue

        Raises:
            asyncio.QueueEmpty: If queue is empty
        """
        item = self._queue.get_nowait()
        self._stats.current_size = self._queue.qsize()
        self._stats.total_dequeued += 1
        return item

    def task_done(self) -> None:
        """Mark a task as completed.

        Call this after processing an item obtained via get().
        """
        self._queue.task_done()
        self._stats.total_completed += 1

    async def join(self) -> None:
        """Wait for all tasks to be completed.

        Blocks until every item that has been put into the queue
        has been marked as done via task_done().
        """
        await self._queue.join()

    @property
    def stats(self) -> QueueStats:
        """Get queue statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics while keeping queue contents."""
        self._stats = QueueStats(
            current_size=self._queue.qsize(),
            max_size=self._max_size,
        )


# =============================================================================
# DeadLetterQueue - Generic failure tracking
# =============================================================================


@dataclass
class DLQEntry:
    """Entry in the dead letter queue.

    Attributes:
        item_id: Unique identifier for the failed item
        error: Error message from the last failure
        failure_count: Number of times this item has failed
        timestamp: Unix timestamp of the last failure
        metadata: Optional additional context (provider_id, model_id, etc.)
    """

    item_id: str
    error: str
    failure_count: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "error": self.error,
            "failure_count": self.failure_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DLQEntry:
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            error=data["error"],
            failure_count=data["failure_count"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


class DeadLetterQueue:
    """Dead letter queue for tracking failed items with retry logic.

    This is a standalone DLQ component that can be used for any type
    of item that may fail and need retry tracking. It's independent
    of the file-based StateManager and can be used for LLM requests,
    API calls, or any other retriable operations.

    Features:
    - Track failure count per item
    - Mark items as permanent failures after max retries
    - Optional persistent storage (JSON file)
    - Automatic cleanup on success (record_success)
    - Generate failure reports

    Example usage:
        ```python
        dlq = DeadLetterQueue(max_retries=3)

        # Record a failure
        should_retry = dlq.record_failure(
            item_id="request_123",
            error="API timeout",
            metadata={"provider": "openai", "model": "gpt-4"},
        )

        if should_retry:
            # Retry the operation
            ...
        else:
            # Item marked as permanent failure
            log.error("Giving up on request_123")

        # On success, remove from DLQ
        dlq.record_success("request_123")

        # Get report
        report = dlq.generate_report()
        ```
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize dead letter queue.

        Args:
            storage_path: Optional path to JSON file for persistence.
                         If None, entries are only kept in memory.
            max_retries: Maximum retries before marking as permanent failure
        """
        self._storage_path = storage_path
        self._max_retries = max_retries
        self._entries: dict[str, DLQEntry] = {}
        self._load()

    @property
    def max_retries(self) -> int:
        """Get maximum retry count."""
        return self._max_retries

    def _load(self) -> None:
        """Load entries from storage file."""
        if self._storage_path is None or not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            for item_id, entry_data in data.items():
                self._entries[item_id] = DLQEntry.from_dict(entry_data)
            log.debug("Loaded DLQ entries", count=len(self._entries))
        except Exception as e:
            log.warning("Failed to load DLQ", error=str(e))

    def _save(self) -> None:
        """Save entries to storage file."""
        if self._storage_path is None:
            return

        try:
            data = {item_id: entry.to_dict() for item_id, entry in self._entries.items()}
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("Failed to save DLQ", error=str(e))

    def record_failure(
        self,
        item_id: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a failure for an item.

        This increments the failure count and updates the error message.
        If the failure count exceeds max_retries, the item is marked
        as a permanent failure.

        Args:
            item_id: Unique identifier for the item
            error: Error message
            metadata: Optional additional context

        Returns:
            True if item should be retried, False if permanent failure
        """
        existing = self._entries.get(item_id)
        failure_count = (existing.failure_count + 1) if existing else 1

        # Merge metadata if existing
        merged_metadata = {}
        if existing and existing.metadata:
            merged_metadata.update(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)

        entry = DLQEntry(
            item_id=item_id,
            error=error,
            failure_count=failure_count,
            timestamp=time.time(),
            metadata=merged_metadata,
        )

        self._entries[item_id] = entry
        self._save()

        if failure_count >= self._max_retries:
            log.warning(
                "Item marked as permanent failure",
                item_id=item_id,
                failure_count=failure_count,
                max_retries=self._max_retries,
            )
            return False

        log.debug(
            "Item failure recorded",
            item_id=item_id,
            failure_count=failure_count,
            will_retry=True,
        )
        return True

    def record_success(self, item_id: str) -> bool:
        """Remove item from DLQ on successful retry.

        Call this when an item that previously failed is successfully
        processed. This removes it from the DLQ entirely.

        Args:
            item_id: Unique identifier for the item

        Returns:
            True if item was in DLQ and removed, False if not found
        """
        if item_id in self._entries:
            del self._entries[item_id]
            self._save()
            log.debug("Item removed from DLQ after success", item_id=item_id)
            return True
        return False

    def get_entry(self, item_id: str) -> DLQEntry | None:
        """Get DLQ entry for an item.

        Args:
            item_id: Unique identifier for the item

        Returns:
            DLQEntry if found, None otherwise
        """
        return self._entries.get(item_id)

    def get_permanent_failures(self) -> list[DLQEntry]:
        """Get all items marked as permanent failures.

        Returns:
            List of DLQ entries that have exceeded max_retries
        """
        return [
            entry for entry in self._entries.values() if entry.failure_count >= self._max_retries
        ]

    def get_retryable_items(self) -> list[DLQEntry]:
        """Get all items eligible for retry.

        Returns:
            List of DLQ entries that can still be retried
        """
        return [
            entry for entry in self._entries.values() if entry.failure_count < self._max_retries
        ]

    def is_permanent_failure(self, item_id: str) -> bool:
        """Check if an item is a permanent failure.

        Args:
            item_id: Unique identifier for the item

        Returns:
            True if item exists and has exceeded max_retries
        """
        entry = self._entries.get(item_id)
        return entry is not None and entry.failure_count >= self._max_retries

    def generate_report(self) -> dict[str, Any]:
        """Generate a failure report.

        Returns:
            Dictionary with failure statistics and entries
        """
        permanent = self.get_permanent_failures()
        retryable = self.get_retryable_items()

        return {
            "generated_at": time.time(),
            "max_retries": self._max_retries,
            "total_entries": len(self._entries),
            "permanent_failures": len(permanent),
            "retryable": len(retryable),
            "entries": [entry.to_dict() for entry in self._entries.values()],
        }

    def export_report(self, output_path: Path) -> int:
        """Export failure report to a JSON file.

        Args:
            output_path: Path for the JSON report file

        Returns:
            Number of entries exported
        """
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

        log.info(
            "DLQ report exported",
            path=str(output_path),
            entries=len(self._entries),
        )
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries from the DLQ."""
        self._entries.clear()
        self._save()
        log.info("DLQ cleared")

    def __len__(self) -> int:
        """Get number of entries in the DLQ."""
        return len(self._entries)

    def __contains__(self, item_id: str) -> bool:
        """Check if an item is in the DLQ."""
        return item_id in self._entries

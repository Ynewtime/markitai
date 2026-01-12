"""Tests for flow control utilities (BoundedQueue, DeadLetterQueue)."""

import json
import tempfile
from pathlib import Path

import pytest

from markit.utils.flow_control import BoundedQueue, DeadLetterQueue, DLQEntry, QueueStats


class TestBoundedQueue:
    """Tests for BoundedQueue."""

    @pytest.mark.asyncio
    async def test_basic_put_get(self) -> None:
        """Test basic put and get operations."""
        queue: BoundedQueue[str] = BoundedQueue(max_size=10)

        await queue.put("item1")
        await queue.put("item2")

        assert queue.size == 2
        assert not queue.empty
        assert not queue.full

        item = await queue.get()
        assert item == "item1"
        assert queue.size == 1

    @pytest.mark.asyncio
    async def test_backpressure_blocks(self) -> None:
        """Test that put blocks when queue is full."""
        queue: BoundedQueue[int] = BoundedQueue(max_size=2, put_timeout=0.1)

        await queue.put(1)
        await queue.put(2)
        assert queue.full

        # This should timeout and return False
        result = await queue.put(3)
        assert result is False
        assert queue.stats.total_dropped == 1

    @pytest.mark.asyncio
    async def test_put_nowait(self) -> None:
        """Test put_nowait operation."""
        queue: BoundedQueue[str] = BoundedQueue(max_size=2)

        assert queue.put_nowait("a") is True
        assert queue.put_nowait("b") is True
        assert queue.put_nowait("c") is False  # Queue full

        assert queue.stats.total_dropped == 1

    @pytest.mark.asyncio
    async def test_task_done_and_join(self) -> None:
        """Test task_done and join operations."""
        queue: BoundedQueue[str] = BoundedQueue(max_size=10)

        await queue.put("task1")
        await queue.put("task2")

        async def consumer() -> None:
            while not queue.empty:
                await queue.get()
                queue.task_done()

        await consumer()
        await queue.join()  # Should return immediately

        assert queue.stats.total_completed == 2

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        queue: BoundedQueue[int] = BoundedQueue(max_size=5)

        for i in range(3):
            await queue.put(i)

        assert queue.stats.total_enqueued == 3
        assert queue.stats.current_size == 3

        await queue.get()
        queue.task_done()

        assert queue.stats.total_dequeued == 1
        assert queue.stats.total_completed == 1
        assert queue.stats.current_size == 2

    def test_queue_stats_properties(self) -> None:
        """Test QueueStats computed properties."""
        stats = QueueStats(
            current_size=5,
            max_size=10,
            total_enqueued=100,
            total_dequeued=90,
            total_dropped=5,
            total_completed=85,
        )

        assert stats.throughput_ratio == 0.85
        assert stats.drop_rate == 5.0

    def test_queue_stats_edge_cases(self) -> None:
        """Test QueueStats with zero values."""
        stats = QueueStats(current_size=0, max_size=10)

        assert stats.throughput_ratio == 0.0
        assert stats.drop_rate == 0.0


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    def test_record_failure_increments_count(self) -> None:
        """Test that failure count increments."""
        dlq = DeadLetterQueue(max_retries=3)

        result = dlq.record_failure("item1", "error1")
        assert result is True  # Should retry
        entry1 = dlq.get_entry("item1")
        assert entry1 is not None
        assert entry1.failure_count == 1

        result = dlq.record_failure("item1", "error2")
        assert result is True  # Should still retry
        entry2 = dlq.get_entry("item1")
        assert entry2 is not None
        assert entry2.failure_count == 2

    def test_permanent_failure_after_max_retries(self) -> None:
        """Test that item becomes permanent failure after max retries."""
        dlq = DeadLetterQueue(max_retries=2)

        dlq.record_failure("item1", "error1")
        result = dlq.record_failure("item1", "error2")

        assert result is False  # Should NOT retry
        assert dlq.is_permanent_failure("item1")
        assert len(dlq.get_permanent_failures()) == 1

    def test_record_success_removes_entry(self) -> None:
        """Test that success removes item from DLQ."""
        dlq = DeadLetterQueue(max_retries=3)

        dlq.record_failure("item1", "error1")
        assert "item1" in dlq

        result = dlq.record_success("item1")
        assert result is True
        assert "item1" not in dlq

    def test_record_success_nonexistent(self) -> None:
        """Test record_success for item not in DLQ."""
        dlq = DeadLetterQueue(max_retries=3)

        result = dlq.record_success("nonexistent")
        assert result is False

    def test_metadata_preserved(self) -> None:
        """Test that metadata is preserved and merged."""
        dlq = DeadLetterQueue(max_retries=3)

        dlq.record_failure("item1", "error1", {"provider": "openai"})
        dlq.record_failure("item1", "error2", {"model": "gpt-4"})

        entry = dlq.get_entry("item1")
        assert entry is not None
        assert entry.metadata["provider"] == "openai"
        assert entry.metadata["model"] == "gpt-4"

    def test_get_retryable_items(self) -> None:
        """Test getting retryable items."""
        dlq = DeadLetterQueue(max_retries=3)

        dlq.record_failure("item1", "error")  # 1 failure
        dlq.record_failure("item2", "error")
        dlq.record_failure("item2", "error")
        dlq.record_failure("item2", "error")  # 3 failures = permanent

        retryable = dlq.get_retryable_items()
        assert len(retryable) == 1
        assert retryable[0].item_id == "item1"

    def test_generate_report(self) -> None:
        """Test report generation."""
        dlq = DeadLetterQueue(max_retries=2)

        dlq.record_failure("item1", "error1")
        dlq.record_failure("item2", "error2")
        dlq.record_failure("item2", "error3")  # permanent

        report = dlq.generate_report()

        assert report["max_retries"] == 2
        assert report["total_entries"] == 2
        assert report["permanent_failures"] == 1
        assert report["retryable"] == 1
        assert len(report["entries"]) == 2

    def test_persistence(self) -> None:
        """Test persistence to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "dlq.json"

            # Create and populate DLQ
            dlq1 = DeadLetterQueue(storage_path=storage_path, max_retries=3)
            dlq1.record_failure("item1", "error1", {"key": "value"})
            dlq1.record_failure("item2", "error2")

            # Load in new instance
            dlq2 = DeadLetterQueue(storage_path=storage_path, max_retries=3)

            assert len(dlq2) == 2
            entry = dlq2.get_entry("item1")
            assert entry is not None
            assert entry.failure_count == 1
            assert entry.metadata["key"] == "value"

    def test_export_report(self) -> None:
        """Test exporting report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(max_retries=3)
            dlq.record_failure("item1", "error1")

            output_path = Path(tmpdir) / "report.json"
            count = dlq.export_report(output_path)

            assert count == 1
            assert output_path.exists()

            report = json.loads(output_path.read_text())
            assert report["total_entries"] == 1

    def test_clear(self) -> None:
        """Test clearing all entries."""
        dlq = DeadLetterQueue(max_retries=3)
        dlq.record_failure("item1", "error")
        dlq.record_failure("item2", "error")

        assert len(dlq) == 2

        dlq.clear()

        assert len(dlq) == 0

    def test_dlq_entry_serialization(self) -> None:
        """Test DLQEntry serialization."""
        entry = DLQEntry(
            item_id="test",
            error="test error",
            failure_count=2,
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )

        data = entry.to_dict()
        restored = DLQEntry.from_dict(data)

        assert restored.item_id == entry.item_id
        assert restored.error == entry.error
        assert restored.failure_count == entry.failure_count
        assert restored.timestamp == entry.timestamp
        assert restored.metadata == entry.metadata


class TestAdaptiveRateLimiterRecordError:
    """Tests for AdaptiveRateLimiter.record_error() unified interface."""

    @pytest.mark.asyncio
    async def test_record_error_rate_limit(self) -> None:
        """Test record_error with is_rate_limit=True."""
        from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig

        config = AIMDConfig(initial_concurrency=10, cooldown_seconds=0)
        limiter = AdaptiveRateLimiter(config)

        await limiter.record_error(is_rate_limit=True)

        assert limiter.stats.rate_limit_hits == 1
        assert limiter.current_concurrency == 5  # 10 * 0.5

    @pytest.mark.asyncio
    async def test_record_error_other(self) -> None:
        """Test record_error with is_rate_limit=False."""
        from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig

        config = AIMDConfig(initial_concurrency=10)
        limiter = AdaptiveRateLimiter(config)

        await limiter.record_error(is_rate_limit=False)

        assert limiter.stats.total_failures == 1
        assert limiter.current_concurrency == 10  # No change


class TestStateManagerRecordSuccess:
    """Tests for StateManager.record_success() DLQ cleanup."""

    def test_record_success_clears_failure_state(self, tmp_path: Path) -> None:
        """Test that record_success clears failure state."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        # Create a batch with one file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        manager.create_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            files=[test_file],
        )

        # Record some failures
        manager.record_failure(test_file, "error1")
        manager.record_failure(test_file, "error2")

        state = manager.get_state()
        assert state is not None
        file_state = state.files["test.txt"]
        assert file_state.failure_count == 2

        # Record success
        manager.record_success(test_file)

        # Failure state should be cleared
        file_state = state.files["test.txt"]
        assert file_state.failure_count == 0
        assert file_state.last_error is None

    def test_record_success_no_previous_failure(self, tmp_path: Path) -> None:
        """Test record_success when there was no failure."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        # Create a batch
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        manager.create_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            files=[test_file],
        )

        # Record success without any failure - should not raise
        manager.record_success(test_file)

        state = manager.get_state()
        assert state is not None
        file_state = state.files["test.txt"]
        assert file_state.failure_count == 0

    def test_record_success_file_not_in_batch(self, tmp_path: Path) -> None:
        """Test record_success for file not in batch."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        # Create an empty batch
        manager.create_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            files=[],
        )

        # Record success for non-existent file - should not raise
        manager.record_success(tmp_path / "nonexistent.txt")

"""Tests for batch processing thread safety."""

import threading
import time
from unittest.mock import MagicMock, patch


class TestBatchStateSaving:
    """Test thread-safe state saving in batch processor."""

    def test_save_state_uses_timeout_lock(self):
        """Verify save_state uses timeout-based lock acquisition."""
        from markitai.batch import BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor._save_lock = threading.Lock()
        processor.state = MagicMock()
        processor.state.to_minimal_dict.return_value = {
            "version": "1.0",
            "documents": {},
        }
        processor.config = MagicMock()
        processor.config.state_flush_interval_seconds = 5
        processor.state_file = MagicMock()
        processor.state_file.parent = MagicMock()
        processor._last_state_save = None

        # Should succeed without error
        with patch("markitai.batch.atomic_write_json"):
            processor.save_state(force=True)

    def test_save_state_double_checked_locking(self):
        """Verify interval is re-checked after acquiring lock."""
        from datetime import datetime

        from markitai.batch import BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor._save_lock = threading.Lock()
        processor.state = MagicMock()
        processor.state.to_minimal_dict.return_value = {
            "version": "1.0",
            "documents": {},
        }
        processor.config = MagicMock()
        processor.config.state_flush_interval_seconds = 5
        processor.state_file = MagicMock()
        processor.state_file.parent = MagicMock()

        # Set last save to just now (within interval)
        processor._last_state_save = datetime.now().astimezone()

        with patch("markitai.batch.atomic_write_json") as mock_write:
            # Non-forced save should skip due to interval
            processor.save_state(force=False)
            mock_write.assert_not_called()

            # Forced save should proceed regardless
            processor.save_state(force=True)
            mock_write.assert_called_once()

    def test_concurrent_forced_saves_both_complete(self):
        """Verify concurrent forced saves don't silently skip."""
        from markitai.batch import BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor._save_lock = threading.Lock()
        processor.state = MagicMock()
        processor.state.to_minimal_dict.return_value = {
            "version": "1.0",
            "documents": {},
        }
        processor.config = MagicMock()
        processor.config.state_flush_interval_seconds = 0
        processor.state_file = MagicMock()
        processor.state_file.parent = MagicMock()
        processor._last_state_save = None

        save_count = {"count": 0}

        def counting_write(*args, **kwargs):
            save_count["count"] += 1
            time.sleep(0.05)  # Simulate I/O

        with patch("markitai.batch.atomic_write_json", side_effect=counting_write):
            threads = [
                threading.Thread(target=processor.save_state, kwargs={"force": True})
                for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # All forced saves should eventually complete (blocking lock with timeout)
        assert save_count["count"] == 5


class TestBatchStateIntegrity:
    """Test that batch state updates maintain data integrity."""

    def test_concurrent_state_updates_no_data_loss(self):
        """Verify concurrent state updates don't lose data."""
        from markitai.batch import BatchState, FileState, FileStatus

        state = BatchState()

        def update_state(file_id: int):
            key = f"file_{file_id}.pdf"
            state.files[key] = FileState(
                path=key,
                status=FileStatus.COMPLETED,
                output=f"/output/{key}.md",
            )

        threads = [threading.Thread(target=update_state, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 files should be recorded
        assert len(state.files) == 20

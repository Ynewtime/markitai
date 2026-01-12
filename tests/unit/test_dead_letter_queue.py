"""Test Dead Letter Queue (DLQ) for failure tracking.

DLQ tracks failed items with retry count, metadata, and automatic
cleanup on success. Items become "permanent failures" after max retries.
"""

import tempfile
from pathlib import Path

from markit.utils.flow_control import DeadLetterQueue


def test_dlq_retry_tracking() -> None:
    """Test DLQ retry tracking and permanent failure detection."""
    dlq = DeadLetterQueue(max_retries=3)

    # Simulate failures
    for i in range(4):
        dlq.record_failure(
            "request_001",
            f"Error attempt {i + 1}",
            metadata={"provider": "openai", "attempt": i + 1},
        )

    # Check permanent failure
    assert dlq.is_permanent_failure("request_001")

    # Test metadata preservation
    entry = dlq.get_entry("request_001")
    assert entry is not None
    assert entry.metadata.get("provider") == "openai"


def test_dlq_success_cleanup() -> None:
    """Test that successful retry removes item from DLQ."""
    dlq = DeadLetterQueue(max_retries=3)

    # Add failures
    dlq.record_failure("request_a", "timeout")
    dlq.record_failure("request_b", "rate limit")
    dlq.record_failure("request_b", "rate limit again")

    assert len(dlq) == 2

    # Simulate successful retry
    removed = dlq.record_success("request_a")
    assert removed is True
    assert "request_a" not in dlq
    assert "request_b" in dlq


def test_dlq_persistence() -> None:
    """Test DLQ persistence to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "dlq_state.json"

        # Create and populate DLQ
        dlq1 = DeadLetterQueue(storage_path=storage_path, max_retries=3)
        dlq1.record_failure("file_001.pdf", "Parse error", {"size": 1024})
        dlq1.record_failure("file_002.docx", "LLM timeout", {"provider": "anthropic"})

        # Simulate restart
        dlq2 = DeadLetterQueue(storage_path=storage_path, max_retries=3)
        assert len(dlq2) == 2

        entry = dlq2.get_entry("file_001.pdf")
        assert entry is not None
        assert entry.metadata.get("size") == 1024


def test_dlq_report() -> None:
    """Test DLQ report generation."""
    dlq = DeadLetterQueue(max_retries=2)

    # Add various failures
    dlq.record_failure("retry_1", "error")
    dlq.record_failure("retry_2", "error")
    dlq.record_failure("permanent_1", "error")
    dlq.record_failure("permanent_1", "error")
    dlq.record_failure("permanent_2", "error")
    dlq.record_failure("permanent_2", "error")

    report = dlq.generate_report()
    assert report["total_entries"] == 4
    assert report["permanent_failures"] == 2
    assert report["retryable"] == 2

    # Test export
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "dlq_report.json"
        count = dlq.export_report(export_path)
        assert export_path.exists()
        assert count == 4


def test_state_manager_integration() -> None:
    """Test StateManager.record_success() DLQ cleanup integration."""
    from markit.core.state import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        state_file = tmpdir_path / "state.json"
        input_dir = tmpdir_path / "input"
        output_dir = tmpdir_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test file
        test_file = input_dir / "test.txt"
        test_file.write_text("Test content")

        manager = StateManager(state_file, max_retries=3)
        manager.create_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            files=[test_file],
        )

        # Record failures
        manager.record_failure(test_file, "Error 1")
        manager.record_failure(test_file, "Error 2")

        state = manager.get_state()
        assert state is not None
        file_state = state.files["test.txt"]
        assert file_state.failure_count == 2

        # Record success (should clear failure state)
        manager.record_success(test_file)

        state = manager.get_state()
        assert state is not None
        file_state = state.files["test.txt"]
        assert file_state.failure_count == 0
        assert file_state.last_error is None

"""Tests for StateManager DLQ (Dead Letter Queue) functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from markit.core.state import FileState, StateManager
from markit.exceptions import StateError


class TestFileStateDLQ:
    """Tests for FileState DLQ fields."""

    def test_default_dlq_fields(self):
        """Test default DLQ field values."""
        state = FileState(path="/test/file.md", status="pending")
        assert state.failure_count == 0
        assert state.last_error is None
        assert state.permanent_failure is False

    def test_dlq_fields_serialization(self):
        """Test DLQ fields are included in to_dict."""
        state = FileState(
            path="/test/file.md",
            status="failed",
            failure_count=3,
            last_error="Connection timeout",
            permanent_failure=True,
        )
        data = state.to_dict()
        assert data["failure_count"] == 3
        assert data["last_error"] == "Connection timeout"
        assert data["permanent_failure"] is True

    def test_dlq_fields_deserialization(self):
        """Test DLQ fields are restored from dict."""
        data = {
            "path": "/test/file.md",
            "status": "permanent_failure",
            "failure_count": 3,
            "last_error": "Rate limit exceeded",
            "permanent_failure": True,
        }
        state = FileState.from_dict(data)
        assert state.failure_count == 3
        assert state.last_error == "Rate limit exceeded"
        assert state.permanent_failure is True

    def test_backward_compatibility(self):
        """Test old state files without DLQ fields still work."""
        old_data = {
            "path": "/test/file.md",
            "status": "pending",
            # No DLQ fields
        }
        state = FileState.from_dict(old_data)
        assert state.failure_count == 0
        assert state.last_error is None
        assert state.permanent_failure is False


class TestStateManagerDLQ:
    """Tests for StateManager DLQ methods."""

    @pytest.fixture
    def temp_state_file(self):
        """Create a temporary state file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "state.json"

    @pytest.fixture
    def state_manager(self, temp_state_file):
        """Create a StateManager with test configuration."""
        return StateManager(temp_state_file, max_retries=3)

    @pytest.fixture
    def populated_state(self, state_manager):
        """Create a StateManager with an existing batch."""
        input_dir = Path("/test/input")
        output_dir = Path("/test/output")
        files = [
            Path("/test/input/file1.md"),
            Path("/test/input/file2.md"),
            Path("/test/input/file3.md"),
        ]

        # Create batch with mock file hashes
        state_manager.create_batch(input_dir, output_dir, files)
        return state_manager

    def test_record_failure_increments_count(self, populated_state):
        """Test that record_failure increments failure count."""
        sm = populated_state
        file_path = Path("/test/input/file1.md")

        result = sm.record_failure(file_path, "Network error")

        state = sm.get_state()
        file_state = state.files["file1.md"]
        assert file_state.failure_count == 1
        assert file_state.last_error == "Network error"
        assert result is False  # Not permanent yet

    def test_record_failure_marks_permanent(self, populated_state):
        """Test that record_failure marks as permanent after max_retries."""
        sm = populated_state
        file_path = Path("/test/input/file1.md")

        # Record failures up to max_retries
        sm.record_failure(file_path, "Error 1")
        sm.record_failure(file_path, "Error 2")
        result = sm.record_failure(file_path, "Error 3")

        assert result is True  # Now permanent

        state = sm.get_state()
        file_state = state.files["file1.md"]
        assert file_state.failure_count == 3
        assert file_state.permanent_failure is True
        assert file_state.status == "permanent_failure"
        assert "Permanent failure after 3 attempts" in file_state.error

    def test_record_failure_no_batch(self, state_manager):
        """Test record_failure raises error without batch."""
        with pytest.raises(StateError, match="No batch state loaded"):
            state_manager.record_failure(Path("/test/file.md"), "Error")

    def test_record_failure_unknown_file(self, populated_state):
        """Test record_failure raises error for unknown file."""
        with pytest.raises(StateError, match="File not found in batch"):
            populated_state.record_failure(Path("/unknown/file.md"), "Error")

    def test_get_permanent_failures(self, populated_state):
        """Test get_permanent_failures returns correct list."""
        sm = populated_state

        # Create some permanent failures
        for _ in range(3):
            sm.record_failure(Path("/test/input/file1.md"), "Error 1")
        for _ in range(3):
            sm.record_failure(Path("/test/input/file2.md"), "Error 2")

        failures = sm.get_permanent_failures()

        assert len(failures) == 2
        paths = [f[0] for f in failures]
        assert "/test/input/file1.md" in paths
        assert "/test/input/file2.md" in paths

    def test_get_permanent_failures_empty(self, populated_state):
        """Test get_permanent_failures with no failures."""
        failures = populated_state.get_permanent_failures()
        assert failures == []

    def test_get_retriable_failures(self, populated_state):
        """Test get_retriable_failures returns only non-permanent failures."""
        sm = populated_state

        # Create one permanent failure
        for _ in range(3):
            sm.record_failure(Path("/test/input/file1.md"), "Error 1")

        # Create one retriable failure
        sm.record_failure(Path("/test/input/file2.md"), "Error 2")

        retriable = sm.get_retriable_failures()

        assert len(retriable) == 1
        assert retriable[0] == Path("/test/input/file2.md")

    def test_export_dlq_report(self, populated_state, temp_state_file):
        """Test exporting DLQ report."""
        sm = populated_state
        report_path = temp_state_file.parent / "dlq_report.json"

        # Create some permanent failures
        for _ in range(3):
            sm.record_failure(Path("/test/input/file1.md"), "Connection timeout")
        for _ in range(3):
            sm.record_failure(Path("/test/input/file2.md"), "API error")

        count = sm.export_dlq_report(report_path)

        assert count == 2
        assert report_path.exists()

        # Verify report content
        with open(report_path) as f:
            report = json.load(f)

        assert "batch_id" in report
        assert "exported_at" in report
        assert report["total_permanent_failures"] == 2
        assert len(report["permanent_failures"]) == 2

    def test_reset_file_failures(self, populated_state):
        """Test resetting file failures."""
        sm = populated_state
        file_path = Path("/test/input/file1.md")

        # Create a permanent failure
        for _ in range(3):
            sm.record_failure(file_path, "Error")

        # Verify it's permanent
        state = sm.get_state()
        assert state.files["file1.md"].permanent_failure is True

        # Reset
        sm.reset_file_failures(file_path)

        # Verify reset
        state = sm.get_state()
        file_state = state.files["file1.md"]
        assert file_state.failure_count == 0
        assert file_state.last_error is None
        assert file_state.permanent_failure is False
        assert file_state.status == "pending"
        assert file_state.error is None

    def test_reset_file_failures_no_batch(self, state_manager):
        """Test reset_file_failures raises error without batch."""
        with pytest.raises(StateError, match="No batch state loaded"):
            state_manager.reset_file_failures(Path("/test/file.md"))

    def test_dlq_state_persistence(self, temp_state_file):
        """Test DLQ state persists across reload."""
        input_dir = Path("/test/input")
        output_dir = Path("/test/output")
        files = [Path("/test/input/file1.md")]

        # Create batch and add failures
        sm1 = StateManager(temp_state_file, max_retries=3)
        sm1.create_batch(input_dir, output_dir, files)
        for _ in range(2):
            sm1.record_failure(Path("/test/input/file1.md"), "Error")

        # Load in new manager
        sm2 = StateManager(temp_state_file, max_retries=3)
        sm2.load_batch()

        state = sm2.get_state()
        assert state.files["file1.md"].failure_count == 2

    def test_failed_files_count_updated(self, populated_state):
        """Test that failed_files count is updated on permanent failure."""
        sm = populated_state
        initial_failed = sm.get_state().failed_files

        # Create permanent failure
        for _ in range(3):
            sm.record_failure(Path("/test/input/file1.md"), "Error")

        assert sm.get_state().failed_files == initial_failed + 1


class TestStateManagerMaxRetries:
    """Tests for StateManager max_retries configuration."""

    def test_custom_max_retries(self):
        """Test custom max_retries value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateManager(state_file, max_retries=5)
            assert sm.max_retries == 5

    def test_default_max_retries(self):
        """Test default max_retries value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            sm = StateManager(state_file)
            assert sm.max_retries == 3  # DEFAULT_MAX_RETRIES

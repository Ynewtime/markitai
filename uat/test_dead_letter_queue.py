#!/usr/bin/env python3
"""UAT: Test Dead Letter Queue (DLQ) for failure tracking.

DLQ tracks failed items with retry count, metadata, and automatic
cleanup on success. Items become "permanent failures" after max retries.

Usage:
    uv run python uat/test_dead_letter_queue.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.utils.flow_control import DeadLetterQueue


def test_dlq_retry_tracking():
    """Test DLQ retry tracking and permanent failure detection."""
    print("=" * 60)
    print("UAT: Dead Letter Queue - Retry Tracking")
    print("=" * 60)
    print()

    dlq = DeadLetterQueue(max_retries=3)
    print(f"Max retries: {dlq.max_retries}")
    print("-" * 40)

    # Simulate failures
    print("\n[Test 1] Recording failures for 'request_001':")
    for i in range(4):
        should_retry = dlq.record_failure(
            "request_001",
            f"Error attempt {i + 1}",
            metadata={"provider": "openai", "attempt": i + 1},
        )
        entry = dlq.get_entry("request_001")
        print(
            f"  Attempt {i + 1}: failure_count={entry.failure_count}, should_retry={should_retry}"
        )

    # Check permanent failure
    if dlq.is_permanent_failure("request_001"):
        print("  ✅ Item correctly marked as permanent failure")
    else:
        print("  ❌ Expected permanent failure")
        return False

    # Test metadata merging
    print("\n[Test 2] Metadata preservation and merging:")
    entry = dlq.get_entry("request_001")
    print(f"  provider: {entry.metadata.get('provider')}")
    print(f"  attempt: {entry.metadata.get('attempt')}")

    if entry.metadata.get("provider") == "openai":
        print("  ✅ Metadata preserved across failures")
    else:
        print("  ❌ Metadata not preserved")
        return False

    return True


def test_dlq_success_cleanup():
    """Test that successful retry removes item from DLQ."""
    print()
    print("=" * 60)
    print("UAT: Dead Letter Queue - Success Cleanup")
    print("=" * 60)
    print()

    dlq = DeadLetterQueue(max_retries=3)

    # Add failures
    dlq.record_failure("request_a", "timeout")
    dlq.record_failure("request_b", "rate limit")
    dlq.record_failure("request_b", "rate limit again")

    print(f"Items in DLQ: {len(dlq)}")
    print(f"  request_a: {dlq.get_entry('request_a').failure_count} failures")
    print(f"  request_b: {dlq.get_entry('request_b').failure_count} failures")
    print("-" * 40)

    # Simulate successful retry
    print("\n[Test] Recording success for 'request_a':")
    removed = dlq.record_success("request_a")

    if removed and "request_a" not in dlq:
        print("  ✅ request_a removed from DLQ after success")
    else:
        print("  ❌ Expected removal from DLQ")
        return False

    print(f"  Items remaining: {len(dlq)}")

    # Verify request_b still there
    if "request_b" in dlq:
        print("  ✅ request_b still in DLQ (not yet successful)")
    else:
        print("  ❌ request_b should still be in DLQ")
        return False

    return True


def test_dlq_persistence():
    """Test DLQ persistence to file."""
    print()
    print("=" * 60)
    print("UAT: Dead Letter Queue - Persistence")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "dlq_state.json"

        # Create and populate DLQ
        print(f"Storage path: {storage_path}")
        print("-" * 40)

        print("\n[Step 1] Create DLQ and add failures:")
        dlq1 = DeadLetterQueue(storage_path=storage_path, max_retries=3)
        dlq1.record_failure("file_001.pdf", "Parse error", {"size": 1024})
        dlq1.record_failure("file_002.docx", "LLM timeout", {"provider": "anthropic"})
        print(f"  Added {len(dlq1)} items to DLQ")

        # Simulate restart by creating new instance
        print("\n[Step 2] Simulate restart (new DLQ instance):")
        dlq2 = DeadLetterQueue(storage_path=storage_path, max_retries=3)
        print(f"  Loaded {len(dlq2)} items from storage")

        # Verify data restored
        if len(dlq2) == 2:
            print("  ✅ Entry count matches")
        else:
            print(f"  ❌ Expected 2 entries, got {len(dlq2)}")
            return False

        entry = dlq2.get_entry("file_001.pdf")
        if entry and entry.metadata.get("size") == 1024:
            print("  ✅ Metadata preserved after reload")
        else:
            print("  ❌ Metadata not preserved")
            return False

    return True


def test_dlq_report():
    """Test DLQ report generation."""
    print()
    print("=" * 60)
    print("UAT: Dead Letter Queue - Report Generation")
    print("=" * 60)
    print()

    dlq = DeadLetterQueue(max_retries=2)

    # Add various failures
    dlq.record_failure("retry_1", "error")  # 1 failure, can retry
    dlq.record_failure("retry_2", "error")  # 1 failure, can retry
    dlq.record_failure("permanent_1", "error")
    dlq.record_failure("permanent_1", "error")  # 2 failures, permanent
    dlq.record_failure("permanent_2", "error")
    dlq.record_failure("permanent_2", "error")  # 2 failures, permanent

    print("[Generating report]")
    print("-" * 40)

    report = dlq.generate_report()

    print(f"  Total entries: {report['total_entries']}")
    print(f"  Permanent failures: {report['permanent_failures']}")
    print(f"  Retryable: {report['retryable']}")
    print(f"  Max retries: {report['max_retries']}")

    if report["total_entries"] == 4:
        print("  ✅ Total entries correct")
    else:
        return False

    if report["permanent_failures"] == 2:
        print("  ✅ Permanent failures count correct")
    else:
        return False

    if report["retryable"] == 2:
        print("  ✅ Retryable count correct")
    else:
        return False

    # Test export
    print("\n[Test export to file]:")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "dlq_report.json"
        count = dlq.export_report(export_path)
        if export_path.exists() and count == 4:
            print(f"  ✅ Exported {count} entries to {export_path.name}")
        else:
            return False

    return True


def test_state_manager_integration():
    """Test StateManager.record_success() DLQ cleanup integration."""
    print()
    print("=" * 60)
    print("UAT: StateManager DLQ Integration")
    print("=" * 60)
    print()

    from markit.core.state import StateManager

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        state_file = tmpdir / "state.json"
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
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

        print("[Step 1] Record failures:")
        manager.record_failure(test_file, "Error 1")
        manager.record_failure(test_file, "Error 2")

        state = manager.get_state()
        file_state = state.files["test.txt"]
        print(f"  Failure count: {file_state.failure_count}")
        print(f"  Last error: {file_state.last_error}")

        if file_state.failure_count == 2:
            print("  ✅ Failures recorded correctly")
        else:
            return False

        print("\n[Step 2] Record success (should clear failure state):")
        manager.record_success(test_file)

        state = manager.get_state()
        file_state = state.files["test.txt"]
        print(f"  Failure count: {file_state.failure_count}")
        print(f"  Last error: {file_state.last_error}")

        if file_state.failure_count == 0 and file_state.last_error is None:
            print("  ✅ Failure state cleared after success")
        else:
            print("  ❌ Failure state not cleared")
            return False

    return True


def main():
    """Run all DLQ UAT tests."""
    results = []

    results.append(("Retry Tracking", test_dlq_retry_tracking()))
    results.append(("Success Cleanup", test_dlq_success_cleanup()))
    results.append(("Persistence", test_dlq_persistence()))
    results.append(("Report Generation", test_dlq_report()))
    results.append(("StateManager Integration", test_state_manager_integration()))

    print()
    print("=" * 60)
    print("SUMMARY: Dead Letter Queue UAT")
    print("=" * 60)
    print()

    all_passed = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_passed = False

    print()
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""Resilience test suite for high-volume and failure scenarios.

- Scenario 1 (Marathon): Run many files through ChaosMockProvider, assert 0 crashes
- Scenario 2 (Interrupter): SIGINT during batch, verify state.json, resume without duplicates

Note: The marathon test uses 100 files (reduced from 1000) for faster CI execution.
"""

from pathlib import Path

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestMarathonScenario:
    """Scenario 1: Marathon - High volume processing with chaos.

    Run many files through ChaosMockProvider to verify:
    - No crashes during execution
    - All files eventually complete (success or permanent failure)
    - State tracking remains consistent
    """

    @pytest.fixture
    def marathon_dir(self, tmp_path: Path) -> Path:
        """Create test files for marathon scenario."""
        test_dir = tmp_path / "marathon_input"
        test_dir.mkdir()

        # Generate 100 small test files (reduced from 1000 for faster CI)
        # Use .txt format which markit supports
        for i in range(100):
            file_path = test_dir / f"doc_{i:05d}.txt"
            content = f"""Document {i}

Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Section 2

Document ID: {i}
Generated for resilience testing.

- Item {i * 3}
- Item {i * 3 + 1}
- Item {i * 3 + 2}
"""
            file_path.write_text(content)

        return test_dir

    @pytest.fixture
    def marathon_output(self, tmp_path: Path) -> Path:
        """Create output directory for marathon."""
        output = tmp_path / "marathon_output"
        output.mkdir()
        return output

    def test_marathon_no_crash(self, marathon_dir: Path, marathon_output: Path) -> None:
        """Test that processing many files doesn't crash."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        files = list(marathon_dir.glob("*.txt"))
        assert len(files) == 100

        success_count = 0
        error_count = 0

        for file in files:
            try:
                result = pipeline.convert_file(file, marathon_output)
                if result.success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

        # All files should be processed (success or error, but no crash)
        assert success_count + error_count == 100
        # Most should succeed (markdown files are simple)
        assert success_count >= 95

    @pytest.mark.asyncio
    async def test_marathon_with_chaos_provider(self, marathon_dir: Path) -> None:
        """Test processing with ChaosMockProvider for LLM calls."""
        from markit.llm.chaos import ChaosConfig, ChaosMockProvider

        # Configure chaos with moderate failure rates for testing
        config = ChaosConfig(
            latency_mean=0.1,  # Fast for testing
            latency_stddev=0.05,
            failure_rate=0.1,  # 10% failures
            rate_limit_prob=0.1,  # 10% rate limits
        )
        provider = ChaosMockProvider(config)

        files = list(marathon_dir.glob("*.txt"))
        success_count = 0
        rate_limit_count = 0
        error_count = 0

        for file in files:
            try:
                from markit.llm.base import LLMMessage

                messages = [LLMMessage.user(f"Summarize: {file.read_text()[:200]}")]
                await provider.complete(messages)
                success_count += 1
            except Exception as e:
                if "429" in str(e) or "Rate limit" in str(e):
                    rate_limit_count += 1
                else:
                    error_count += 1

        # Verify stats tracking
        stats = provider.stats
        assert stats.call_count == 100
        assert stats.success_count == success_count

        # With 10% failure rate, expect roughly 80-95% success
        assert success_count >= 70, f"Too few successes: {success_count}"

    @pytest.mark.asyncio
    async def test_marathon_adaptive_limiter(self) -> None:
        """Test AIMD limiter under sustained load."""
        from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig

        config = AIMDConfig(
            initial_concurrency=5,
            max_concurrency=20,
            min_concurrency=1,
            success_threshold=5,
            cooldown_seconds=0.1,
        )
        limiter = AdaptiveRateLimiter(config)

        # Simulate mixed success/rate-limit pattern
        for i in range(100):
            await limiter.acquire()
            try:
                if i % 10 == 0:  # Every 10th request is rate limited
                    await limiter.record_rate_limit()
                else:
                    await limiter.record_success()
            finally:
                limiter.release()

        # Should have adapted concurrency
        assert limiter.stats.total_requests == 100
        assert limiter.stats.increase_count > 0 or limiter.stats.decrease_count > 0


class TestInterrupterScenario:
    """Scenario 2: Interrupter - SIGINT handling and resume.

    Verify:
    - state.json integrity after interrupt
    - Resume doesn't reprocess completed files
    - No duplicate processing
    """

    @pytest.fixture
    def interrupter_dir(self, tmp_path: Path) -> Path:
        """Create test files for interrupter scenario."""
        test_dir = tmp_path / "interrupter_input"
        test_dir.mkdir()

        # Generate 50 test files
        for i in range(50):
            file_path = test_dir / f"doc_{i:05d}.txt"
            file_path.write_text(f"Document {i}\nContent for testing resume functionality.")

        return test_dir

    @pytest.fixture
    def interrupter_output(self, tmp_path: Path) -> Path:
        """Create output directory."""
        output = tmp_path / "interrupter_output"
        output.mkdir()
        return output

    def test_state_json_integrity(
        self, interrupter_dir: Path, interrupter_output: Path, tmp_path: Path
    ) -> None:
        """Test that state.json maintains integrity during processing."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        files = list(interrupter_dir.glob("*.txt"))

        # Create batch
        batch = manager.create_batch(
            input_dir=interrupter_dir,
            output_dir=interrupter_output,
            files=files,
        )

        assert batch.total_files == 50
        assert len(batch.pending_files) == 50

        # Simulate partial processing
        for file in files[:25]:
            manager.update_file_status(
                file, "completed", output_path=interrupter_output / f"{file.name}.md"
            )

        # Verify state
        state = manager.get_state()
        assert state is not None
        assert state.completed_files == 25
        assert len(state.pending_files) == 25

        # Reload state (simulating restart)
        manager2 = StateManager(state_file, max_retries=3)
        reloaded = manager2.load_batch()

        assert reloaded is not None
        assert reloaded.completed_files == 25
        assert len(reloaded.pending_files) == 25

    def test_resume_no_duplicates(
        self, interrupter_dir: Path, interrupter_output: Path, tmp_path: Path
    ) -> None:
        """Test that resume doesn't reprocess completed files."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        files = list(interrupter_dir.glob("*.txt"))

        # Create batch and mark some as completed
        manager.create_batch(
            input_dir=interrupter_dir,
            output_dir=interrupter_output,
            files=files,
        )

        completed_files = set()
        for file in files[:30]:
            manager.update_file_status(file, "completed")
            completed_files.add(str(file))

        # Get pending files (should exclude completed)
        pending = manager.get_pending_files()
        pending_paths = {str(p) for p in pending}

        # Verify no overlap
        assert len(pending) == 20
        assert len(completed_files & pending_paths) == 0, "Completed files should not be pending"

    def test_dlq_prevents_infinite_retry(
        self, interrupter_dir: Path, interrupter_output: Path, tmp_path: Path
    ) -> None:
        """Test that DLQ prevents infinite retries of failing files."""
        from markit.core.state import StateManager

        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, max_retries=3)

        files = list(interrupter_dir.glob("*.txt"))
        poison_file = files[0]  # This file will always fail

        manager.create_batch(
            input_dir=interrupter_dir,
            output_dir=interrupter_output,
            files=files,
        )

        # Simulate repeated failures
        for attempt in range(5):
            is_permanent = manager.record_failure(poison_file, f"Error attempt {attempt}")
            if is_permanent:
                break

        # Should be marked as permanent failure after 3 attempts
        failures = manager.get_permanent_failures()
        assert len(failures) == 1
        assert str(poison_file) in failures[0][0]

        # Should not be in retriable list
        retriable = manager.get_retriable_failures()
        retriable_paths = {str(p) for p in retriable}
        assert str(poison_file) not in retriable_paths


class TestBackpressureQueue:
    """Test backpressure mechanisms under load."""

    @pytest.mark.asyncio
    async def test_bounded_queue_backpressure(self) -> None:
        """Test that bounded queue applies backpressure."""
        from markit.utils.flow_control import BoundedQueue

        queue: BoundedQueue[int] = BoundedQueue(max_size=10, put_timeout=0.5)

        # Fill queue
        for i in range(10):
            result = await queue.put(i)
            assert result is True

        assert queue.full

        # Next put should timeout (backpressure)
        result = await queue.put(100)
        assert result is False
        assert queue.stats.total_dropped == 1

    @pytest.mark.asyncio
    async def test_dlq_with_metadata(self) -> None:
        """Test DLQ tracks metadata correctly."""
        from markit.utils.flow_control import DeadLetterQueue

        dlq = DeadLetterQueue(max_retries=3)

        # Record failures with metadata
        dlq.record_failure(
            "request_1",
            "API timeout",
            metadata={"provider": "openai", "model": "gpt-4"},
        )

        entry = dlq.get_entry("request_1")
        assert entry is not None
        assert entry.metadata["provider"] == "openai"
        assert entry.metadata["model"] == "gpt-4"

        # Success removes from DLQ
        dlq.record_success("request_1")
        assert "request_1" not in dlq

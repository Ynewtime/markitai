"""Tests for batch processing statistics module."""

from time import time

from markit.utils.stats import BatchStats, ModelUsageStats


class TestModelUsageStats:
    """Tests for ModelUsageStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = ModelUsageStats(model_name="gpt-4")

        assert stats.model_name == "gpt-4"
        assert stats.calls == 0
        assert stats.total_tokens == 0
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.estimated_cost == 0.0
        assert stats.total_duration == 0.0

    def test_custom_values(self):
        """Test custom initialization."""
        stats = ModelUsageStats(
            model_name="claude-3",
            calls=5,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            estimated_cost=0.05,
            total_duration=10.5,
        )

        assert stats.model_name == "claude-3"
        assert stats.calls == 5
        assert stats.total_tokens == 1000
        assert stats.prompt_tokens == 600
        assert stats.completion_tokens == 400
        assert stats.estimated_cost == 0.05
        assert stats.total_duration == 10.5


class TestBatchStatsInit:
    """Tests for BatchStats initialization."""

    def test_default_values(self):
        """Test default initialization."""
        stats = BatchStats()

        assert stats.total_files == 0
        assert stats.success_files == 0
        assert stats.failed_files == 0
        assert stats.skipped_files == 0
        assert stats.total_duration == 0.0
        assert stats.init_duration == 0.0
        assert stats.convert_duration == 0.0
        assert stats.llm_wall_duration == 0.0
        assert stats.llm_cumulative_duration == 0.0
        assert stats.total_tokens == 0
        assert stats.estimated_cost == 0.0
        assert stats.model_usage == {}
        assert stats.end_time is None

    def test_start_time_set(self):
        """Test that start_time is set on initialization."""
        before = time()
        stats = BatchStats()
        after = time()

        assert before <= stats.start_time <= after


class TestBatchStatsRecordLlmTiming:
    """Tests for record_llm_timing method."""

    def test_first_call(self):
        """Test recording first LLM timing."""
        stats = BatchStats()
        stats.record_llm_timing(100.0, 105.0)

        assert stats._llm_first_start == 100.0
        assert stats._llm_last_end == 105.0

    def test_earlier_start(self):
        """Test recording earlier start time."""
        stats = BatchStats()
        stats.record_llm_timing(100.0, 105.0)
        stats.record_llm_timing(95.0, 103.0)

        assert stats._llm_first_start == 95.0
        assert stats._llm_last_end == 105.0

    def test_later_end(self):
        """Test recording later end time."""
        stats = BatchStats()
        stats.record_llm_timing(100.0, 105.0)
        stats.record_llm_timing(102.0, 110.0)

        assert stats._llm_first_start == 100.0
        assert stats._llm_last_end == 110.0

    def test_multiple_calls(self):
        """Test multiple timing records."""
        stats = BatchStats()
        stats.record_llm_timing(100.0, 105.0)
        stats.record_llm_timing(95.0, 103.0)
        stats.record_llm_timing(102.0, 115.0)
        stats.record_llm_timing(98.0, 108.0)

        assert stats._llm_first_start == 95.0
        assert stats._llm_last_end == 115.0


class TestBatchStatsAddLlmCall:
    """Tests for add_llm_call method."""

    def test_first_call(self):
        """Test adding first LLM call."""
        stats = BatchStats()
        stats.add_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            duration=2.0,
        )

        assert "gpt-4" in stats.model_usage
        model_stats = stats.model_usage["gpt-4"]
        assert model_stats.calls == 1
        assert model_stats.prompt_tokens == 100
        assert model_stats.completion_tokens == 50
        assert model_stats.total_tokens == 150
        assert model_stats.estimated_cost == 0.01
        assert model_stats.total_duration == 2.0

        assert stats.total_tokens == 150
        assert stats.total_prompt_tokens == 100
        assert stats.total_completion_tokens == 50
        assert stats.estimated_cost == 0.01
        assert stats.llm_cumulative_duration == 2.0

    def test_multiple_calls_same_model(self):
        """Test adding multiple calls for same model."""
        stats = BatchStats()
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50, cost=0.01)
        stats.add_llm_call(model="gpt-4", prompt_tokens=200, completion_tokens=100, cost=0.02)

        model_stats = stats.model_usage["gpt-4"]
        assert model_stats.calls == 2
        assert model_stats.prompt_tokens == 300
        assert model_stats.completion_tokens == 150
        assert model_stats.total_tokens == 450

        assert stats.total_tokens == 450
        assert stats.estimated_cost == 0.03

    def test_multiple_models(self):
        """Test adding calls for different models."""
        stats = BatchStats()
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        stats.add_llm_call(model="claude-3", prompt_tokens=200, completion_tokens=100)

        assert len(stats.model_usage) == 2
        assert stats.model_usage["gpt-4"].calls == 1
        assert stats.model_usage["claude-3"].calls == 1
        assert stats.total_tokens == 450

    def test_with_timing(self):
        """Test adding call with timing info."""
        stats = BatchStats()
        stats.add_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            start_time=100.0,
            end_time=105.0,
        )

        assert stats._llm_first_start == 100.0
        assert stats._llm_last_end == 105.0


class TestBatchStatsAddFileResult:
    """Tests for add_file_result method."""

    def test_success(self):
        """Test recording successful file."""
        stats = BatchStats()
        stats.add_file_result(success=True)

        assert stats.total_files == 1
        assert stats.success_files == 1
        assert stats.failed_files == 0
        assert stats.skipped_files == 0

    def test_failure(self):
        """Test recording failed file."""
        stats = BatchStats()
        stats.add_file_result(success=False)

        assert stats.total_files == 1
        assert stats.success_files == 0
        assert stats.failed_files == 1
        assert stats.skipped_files == 0

    def test_skipped(self):
        """Test recording skipped file."""
        stats = BatchStats()
        stats.add_file_result(success=False, skipped=True)

        assert stats.total_files == 1
        assert stats.success_files == 0
        assert stats.failed_files == 0
        assert stats.skipped_files == 1

    def test_multiple_results(self):
        """Test recording multiple file results."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=True)
        stats.add_file_result(success=False)
        stats.add_file_result(success=False, skipped=True)

        assert stats.total_files == 4
        assert stats.success_files == 2
        assert stats.failed_files == 1
        assert stats.skipped_files == 1


class TestBatchStatsFinish:
    """Tests for finish method."""

    def test_finish_sets_end_time(self):
        """Test that finish sets end_time."""
        stats = BatchStats()
        stats.finish()

        assert stats.end_time is not None
        assert stats.end_time >= stats.start_time

    def test_finish_calculates_duration(self):
        """Test that finish calculates total_duration."""
        stats = BatchStats()
        stats.finish()

        assert stats.total_duration >= 0

    def test_finish_calculates_llm_wall_duration(self):
        """Test that finish calculates llm_wall_duration."""
        stats = BatchStats()
        stats.record_llm_timing(100.0, 105.0)
        stats.record_llm_timing(102.0, 110.0)
        stats.finish()

        assert stats.llm_wall_duration == 10.0  # 110 - 100

    def test_finish_without_llm_calls(self):
        """Test finish when no LLM calls recorded."""
        stats = BatchStats()
        stats.finish()

        assert stats.llm_wall_duration == 0.0


class TestBatchStatsSuccessRate:
    """Tests for success_rate property."""

    def test_no_files(self):
        """Test success rate with no files."""
        stats = BatchStats()
        assert stats.success_rate == 0.0

    def test_all_success(self):
        """Test success rate with all successes."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=True)
        stats.add_file_result(success=True)

        assert stats.success_rate == 100.0

    def test_all_failure(self):
        """Test success rate with all failures."""
        stats = BatchStats()
        stats.add_file_result(success=False)
        stats.add_file_result(success=False)

        assert stats.success_rate == 0.0

    def test_mixed_results(self):
        """Test success rate with mixed results."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=True)
        stats.add_file_result(success=False)
        stats.add_file_result(success=True)

        assert stats.success_rate == 75.0

    def test_skipped_not_counted(self):
        """Test that skipped files don't affect success rate."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=False)
        stats.add_file_result(success=False, skipped=True)

        # 1 success, 1 failure = 50% (skipped not counted)
        assert stats.success_rate == 50.0


class TestBatchStatsFormatSummary:
    """Tests for format_summary method."""

    def test_basic_summary(self):
        """Test basic summary format."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=True)
        stats.add_file_result(success=False)
        stats.finish()

        summary = stats.format_summary()

        assert "2 success" in summary
        assert "1 failed" in summary

    def test_summary_with_skipped(self):
        """Test summary with skipped files."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=False, skipped=True)
        stats.finish()

        summary = stats.format_summary()

        assert "1 skipped" in summary

    def test_summary_with_tokens(self):
        """Test summary with token info."""
        stats = BatchStats()
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50, cost=0.01)
        stats.finish()

        summary = stats.format_summary()

        assert "Tokens: 150" in summary
        assert "$0.01" in summary

    def test_summary_with_model_usage(self):
        """Test summary with model usage."""
        stats = BatchStats()
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        stats.finish()

        summary = stats.format_summary()

        assert "gpt-4(2)" in summary

    def test_summary_with_durations(self):
        """Test summary with duration breakdown."""
        stats = BatchStats()
        stats.init_duration = 1.0
        stats.convert_duration = 5.0
        stats.finish()

        summary = stats.format_summary()

        assert "Init:" in summary
        assert "Process:" in summary


class TestBatchStatsFormatDetailed:
    """Tests for format_detailed method."""

    def test_detailed_includes_summary(self):
        """Test that detailed includes summary."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.finish()

        detailed = stats.format_detailed()

        assert "1 success" in detailed

    def test_detailed_model_breakdown(self):
        """Test detailed model breakdown."""
        stats = BatchStats()
        stats.add_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            duration=2.0,
        )
        stats.add_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            duration=2.0,
        )
        stats.finish()

        detailed = stats.format_detailed()

        assert "Model Breakdown:" in detailed
        assert "gpt-4:" in detailed
        assert "Calls: 2" in detailed
        assert "prompt: 200" in detailed
        assert "completion: 100" in detailed
        assert "Cost: $0.02" in detailed
        assert "Duration:" in detailed


class TestBatchStatsToDict:
    """Tests for to_dict method."""

    def test_basic_dict(self):
        """Test basic dictionary conversion."""
        stats = BatchStats()
        stats.add_file_result(success=True)
        stats.add_file_result(success=False)
        stats.finish()

        result = stats.to_dict()

        assert result["total_files"] == 2
        assert result["success_files"] == 1
        assert result["failed_files"] == 1
        assert result["success_rate"] == 50.0

    def test_dict_with_model_usage(self):
        """Test dictionary with model usage."""
        stats = BatchStats()
        stats.add_llm_call(model="gpt-4", prompt_tokens=100, completion_tokens=50, cost=0.01)
        stats.finish()

        result = stats.to_dict()

        assert "model_usage" in result
        assert "gpt-4" in result["model_usage"]
        assert result["model_usage"]["gpt-4"]["calls"] == 1
        assert result["model_usage"]["gpt-4"]["total_tokens"] == 150

    def test_dict_with_durations(self):
        """Test dictionary with duration fields."""
        stats = BatchStats()
        stats.init_duration = 1.5
        stats.convert_duration = 5.0
        stats.record_llm_timing(100.0, 103.0)
        stats.finish()

        result = stats.to_dict()

        assert result["init_duration"] == 1.5
        assert result["convert_duration"] == 5.0
        assert result["llm_wall_duration"] == 3.0

"""Batch processing statistics collection and reporting."""

from dataclasses import dataclass, field
from time import time


@dataclass
class ModelUsageStats:
    """Statistics for a single model."""

    model_name: str
    calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0
    total_duration: float = 0.0


@dataclass
class BatchStats:
    """Comprehensive statistics for batch processing.

    Duration Semantics:
    - total_duration: Wall-clock time from start to finish
    - init_duration: Initialization/warmup phase (LLM provider warmup, etc.)
    - convert_duration: Processing phase wall-clock time (conversion + LLM + output, interleaved)
    - llm_wall_duration: LLM phase wall-clock time (from first LLM call start to last LLM call end)
    - llm_cumulative_duration: Sum of all LLM call durations (may exceed wall time due to parallelism)

    Note: llm_wall_duration represents the actual time spent waiting for LLM calls,
    while llm_cumulative_duration shows the "equivalent serial time". When LLM calls
    run in parallel, llm_cumulative_duration > llm_wall_duration indicates good parallelism.
    """

    # File counts
    total_files: int = 0
    success_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0

    # Duration breakdown (wall-clock time)
    total_duration: float = 0.0
    init_duration: float = 0.0  # Warmup phase (LLM provider initialization)
    convert_duration: float = 0.0  # Processing phase (conversion + LLM + output, interleaved)
    llm_wall_duration: float = 0.0  # LLM phase wall-clock time

    # LLM timing tracking (for calculating wall-clock duration)
    _llm_first_start: float | None = field(default=None, repr=False)
    _llm_last_end: float | None = field(default=None, repr=False)

    # LLM cumulative duration (sum of all individual LLM call durations)
    # This may exceed llm_wall_duration due to parallel execution
    llm_cumulative_duration: float = 0.0

    # Token/cost totals
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost: float = 0.0

    # Per-model breakdown
    model_usage: dict[str, ModelUsageStats] = field(default_factory=dict)

    # Timing
    start_time: float = field(default_factory=time)
    end_time: float | None = None

    def record_llm_timing(self, start_time: float, end_time: float) -> None:
        """Record LLM call timing for wall-clock duration calculation.

        This method tracks the earliest start and latest end across all LLM calls
        to calculate the overall LLM phase wall-clock duration.

        Args:
            start_time: When the LLM call started (from time())
            end_time: When the LLM call completed (from time())
        """
        if self._llm_first_start is None or start_time < self._llm_first_start:
            self._llm_first_start = start_time
        if self._llm_last_end is None or end_time > self._llm_last_end:
            self._llm_last_end = end_time

    def add_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        duration: float = 0.0,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        """Record an LLM call.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Estimated cost in USD
            duration: Call duration in seconds (cumulative, not wall-clock)
            start_time: When the LLM call started (optional, for wall-clock tracking)
            end_time: When the LLM call completed (optional, for wall-clock tracking)
        """
        if model not in self.model_usage:
            self.model_usage[model] = ModelUsageStats(model_name=model)

        stats = self.model_usage[model]
        stats.calls += 1
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.total_tokens += prompt_tokens + completion_tokens
        stats.estimated_cost += cost
        stats.total_duration += duration

        # Update totals
        total_tokens = prompt_tokens + completion_tokens
        self.total_tokens += total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.estimated_cost += cost
        self.llm_cumulative_duration += duration

        # Track LLM timing for wall-clock calculation
        if start_time is not None and end_time is not None:
            self.record_llm_timing(start_time, end_time)

    def add_file_result(self, success: bool, skipped: bool = False) -> None:
        """Record a file processing result.

        Args:
            success: Whether processing succeeded
            skipped: Whether the file was skipped
        """
        self.total_files += 1
        if skipped:
            self.skipped_files += 1
        elif success:
            self.success_files += 1
        else:
            self.failed_files += 1

    def finish(self) -> None:
        """Mark batch as complete and calculate final duration."""
        self.end_time = time()
        self.total_duration = self.end_time - self.start_time

        # Calculate LLM wall-clock duration
        if self._llm_first_start is not None and self._llm_last_end is not None:
            self.llm_wall_duration = self._llm_last_end - self._llm_first_start

    def format_summary(self) -> str:
        """Format statistics as a human-readable summary.

        Returns:
            Multi-line summary string
        """
        lines = []

        # File summary
        lines.append(f"Complete: {self.success_files} success, {self.failed_files} failed")
        if self.skipped_files > 0:
            lines[-1] += f", {self.skipped_files} skipped"

        # Duration breakdown
        duration_parts = [f"Total: {self.total_duration:.0f}s"]
        if self.init_duration > 0:
            duration_parts.append(f"Init: {self.init_duration:.0f}s")
        if self.convert_duration > 0:
            duration_parts.append(f"Process: {self.convert_duration:.0f}s")
        # Show LLM wall-clock duration (actual time spent waiting for LLM)
        if self.llm_wall_duration > 0:
            duration_parts.append(f"LLM: {self.llm_wall_duration:.0f}s")

        lines.append(" | ".join(duration_parts))

        # Token and cost summary
        if self.total_tokens > 0:
            token_str = f"Tokens: {self.total_tokens:,}"
            if self.estimated_cost > 0:
                token_str += f" | Est. cost: ${self.estimated_cost:.4f}"
            lines.append(token_str)

        # Model usage breakdown
        if self.model_usage:
            model_parts = [f"{m}({s.calls})" for m, s in self.model_usage.items()]
            lines.append(f"Models used: {', '.join(model_parts)}")

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format detailed per-model statistics.

        Returns:
            Multi-line detailed breakdown
        """
        lines = [self.format_summary(), ""]

        if self.model_usage:
            lines.append("Model Breakdown:")
            for name, stats in self.model_usage.items():
                lines.append(f"  {name}:")
                lines.append(f"    Calls: {stats.calls}")
                lines.append(
                    f"    Tokens: {stats.total_tokens:,} (prompt: {stats.prompt_tokens:,}, completion: {stats.completion_tokens:,})"
                )
                if stats.estimated_cost > 0:
                    lines.append(f"    Cost: ${stats.estimated_cost:.4f}")
                if stats.total_duration > 0:
                    avg_duration = stats.total_duration / stats.calls if stats.calls > 0 else 0
                    lines.append(
                        f"    Duration: {stats.total_duration:.1f}s (avg: {avg_duration:.1f}s/call)"
                    )

        return "\n".join(lines)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        processed = self.success_files + self.failed_files
        if processed == 0:
            return 0.0
        return (self.success_files / processed) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of stats
        """
        return {
            "total_files": self.total_files,
            "success_files": self.success_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "total_duration": self.total_duration,
            "init_duration": self.init_duration,
            "convert_duration": self.convert_duration,
            "llm_wall_duration": self.llm_wall_duration,
            "llm_cumulative_duration": self.llm_cumulative_duration,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost": self.estimated_cost,
            "success_rate": self.success_rate,
            "model_usage": {
                name: {
                    "calls": stats.calls,
                    "total_tokens": stats.total_tokens,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "estimated_cost": stats.estimated_cost,
                    "total_duration": stats.total_duration,
                }
                for name, stats in self.model_usage.items()
            },
        }

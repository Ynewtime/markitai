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
    """Comprehensive statistics for batch processing."""

    # File counts
    total_files: int = 0
    success_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0

    # Duration breakdown
    total_duration: float = 0.0
    llm_duration: float = 0.0
    convert_duration: float = 0.0
    init_duration: float = 0.0

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

    def add_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """Record an LLM call.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Estimated cost in USD
            duration: Call duration in seconds
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
        self.llm_duration += duration

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
        other_duration = self.total_duration - self.llm_duration - self.convert_duration - self.init_duration
        duration_parts = [f"Total: {self.total_duration:.0f}s"]
        if self.llm_duration > 0:
            duration_parts.append(f"LLM: {self.llm_duration:.0f}s")
        if self.convert_duration > 0:
            duration_parts.append(f"Convert: {self.convert_duration:.0f}s")
        if self.init_duration > 0:
            duration_parts.append(f"Init: {self.init_duration:.0f}s")
        if other_duration > 1:
            duration_parts.append(f"Other: {other_duration:.0f}s")
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
                lines.append(f"    Tokens: {stats.total_tokens:,} (prompt: {stats.prompt_tokens:,}, completion: {stats.completion_tokens:,})")
                if stats.estimated_cost > 0:
                    lines.append(f"    Cost: ${stats.estimated_cost:.4f}")
                if stats.total_duration > 0:
                    avg_duration = stats.total_duration / stats.calls if stats.calls > 0 else 0
                    lines.append(f"    Duration: {stats.total_duration:.1f}s (avg: {avg_duration:.1f}s/call)")

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
            "llm_duration": self.llm_duration,
            "convert_duration": self.convert_duration,
            "init_duration": self.init_duration,
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

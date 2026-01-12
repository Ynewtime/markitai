"""Chaos Mock Provider for resilience testing.

This provider simulates various failure conditions for testing:
- Random latency (configurable distribution)
- Random 500 errors (configurable failure rate)
- Rate limit (429) simulation
- Timeout simulation
- OOM simulation

Example usage:
    ```python
    from markit.llm.chaos import ChaosMockProvider, ChaosConfig

    # Create a chaos provider with custom settings
    config = ChaosConfig(
        latency_mean=2.0,
        latency_stddev=1.0,
        failure_rate=0.1,
        rate_limit_prob=0.2,
    )
    provider = ChaosMockProvider(config)

    # Use in tests or resilience scenarios
    response = await provider.complete([LLMMessage.user("Hello")])
    ```
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, TokenUsage
from markit.utils.logging import generate_request_id, get_logger

log = get_logger(__name__)


@dataclass
class ChaosConfig:
    """Configuration for chaos behavior.

    Attributes:
        latency_mean: Mean latency in seconds (default: 2.0)
        latency_stddev: Standard deviation for latency (default: 1.0)
        failure_rate: Probability of random 500 error (default: 0.1 = 10%)
        rate_limit_prob: Probability of 429 rate limit (default: 0.2 = 20%)
        timeout_prob: Probability of timeout (default: 0.05 = 5%)
        timeout_duration: Duration for simulated timeout in seconds (default: 120)
        oom_trigger: If True, simulate OOM conditions (default: False)
        response_text: Default response text (default: "Chaos mock response")
        model_name: Simulated model name (default: "chaos-mock-v1")
    """

    latency_mean: float = 2.0
    latency_stddev: float = 1.0
    failure_rate: float = 0.1
    rate_limit_prob: float = 0.2
    timeout_prob: float = 0.05
    timeout_duration: float = 120.0
    oom_trigger: bool = False
    response_text: str = "Chaos mock response"
    model_name: str = "chaos-mock-v1"


@dataclass
class ChaosStats:
    """Statistics for chaos provider operations.

    Attributes:
        call_count: Total number of calls
        success_count: Successful calls
        failure_count: Failed calls (500 errors)
        rate_limit_count: Rate limit (429) errors
        timeout_count: Timeout errors
        total_latency_ms: Cumulative latency in milliseconds
    """

    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    rate_limit_count: int = 0
    timeout_count: int = 0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def total_requests(self) -> int:
        """Total number of completed requests (success + failure + rate_limit)."""
        return self.success_count + self.failure_count + self.rate_limit_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0  # No requests = 100% success (consistent with AIMDStats)
        return (self.success_count / self.total_requests) * 100

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.rate_limit_count / self.total_requests) * 100


class ChaosMockProvider(BaseLLMProvider):
    """Mock LLM provider with configurable chaos for testing resilience.

    This provider is designed for:
    - Unit testing error handling and retry logic
    - Integration testing of fallback mechanisms
    - Load testing with realistic failure patterns
    - Chaos engineering experiments

    The provider tracks statistics about calls and failures, which can be
    accessed via the `stats` property for assertions in tests.
    """

    name = "chaos"

    def __init__(self, config: ChaosConfig | None = None) -> None:
        """Initialize chaos provider.

        Args:
            config: Chaos configuration. If None, uses defaults.
        """
        self.config = config or ChaosConfig()
        self._stats = ChaosStats()
        self._call_history: list[dict[str, Any]] = []

    async def _apply_chaos(self, operation: str) -> tuple[str, int]:
        """Apply chaos effects and return the outcome.

        This method:
        1. Applies random latency (Gaussian distribution)
        2. Randomly triggers rate limits (429)
        3. Randomly triggers failures (500)
        4. Randomly triggers timeouts

        Args:
            operation: Description of the operation (for logging)

        Returns:
            Tuple of (outcome_type, latency_ms) where outcome_type is one of:
            "success", "rate_limit", "failure", "timeout"

        Raises:
            RateLimitError: If rate limit is triggered
            LLMError: If random failure is triggered
            asyncio.TimeoutError: If timeout is triggered
        """
        request_id = generate_request_id()
        start_time = time.perf_counter()

        self._stats.call_count += 1

        # Apply random latency (Gaussian distribution, clamped to non-negative)
        latency = max(0.0, random.gauss(self.config.latency_mean, self.config.latency_stddev))
        await asyncio.sleep(latency)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        self._stats.total_latency_ms += latency_ms

        # Record call in history
        call_record = {
            "request_id": request_id,
            "operation": operation,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }

        # Check for timeout (simulate by raising TimeoutError)
        if random.random() < self.config.timeout_prob:
            self._stats.timeout_count += 1
            call_record["outcome"] = "timeout"
            self._call_history.append(call_record)

            log.debug(
                "Chaos: Timeout triggered",
                request_id=request_id,
                operation=operation,
                latency_ms=latency_ms,
            )
            raise TimeoutError(f"Chaos: Simulated timeout after {latency_ms}ms")

        # Check for rate limit (429)
        if random.random() < self.config.rate_limit_prob:
            self._stats.rate_limit_count += 1
            call_record["outcome"] = "rate_limit"
            self._call_history.append(call_record)

            log.debug(
                "Chaos: Rate limit triggered",
                request_id=request_id,
                operation=operation,
                latency_ms=latency_ms,
            )
            raise RateLimitError(retry_after=5)

        # Check for random failure (500)
        if random.random() < self.config.failure_rate:
            self._stats.failure_count += 1
            call_record["outcome"] = "failure"
            self._call_history.append(call_record)

            log.debug(
                "Chaos: Random failure triggered",
                request_id=request_id,
                operation=operation,
                latency_ms=latency_ms,
            )
            raise LLMError("Chaos: Simulated random server error (500)")

        # Success path
        self._stats.success_count += 1
        call_record["outcome"] = "success"
        self._call_history.append(call_record)

        log.debug(
            "Chaos: Request succeeded",
            request_id=request_id,
            operation=operation,
            latency_ms=latency_ms,
        )

        return "success", latency_ms

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> LLMResponse:
        """Generate mock completion with chaos effects.

        Args:
            messages: List of messages (used for token estimation)
            temperature: Sampling temperature (ignored)
            max_tokens: Maximum tokens (used for output estimation)
            **kwargs: Additional arguments (ignored)

        Returns:
            Mock LLM response

        Raises:
            RateLimitError: If rate limit chaos is triggered
            LLMError: If failure chaos is triggered
        """
        outcome, latency_ms = await self._apply_chaos("complete")

        # Estimate tokens based on message content
        input_text = " ".join(
            msg.content if isinstance(msg.content, str) else str(msg.content) for msg in messages
        )
        prompt_tokens = len(input_text.split()) * 2  # Rough estimate

        # Generate response tokens based on max_tokens or default
        output_tokens = min(max_tokens or 100, len(self.config.response_text.split()) * 2)

        return LLMResponse(
            content=self.config.response_text,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=output_tokens),
            model=self.config.model_name,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[LLMMessage],  # noqa: ARG002
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[str]:
        """Stream mock response with chaos effects.

        Args:
            messages: List of messages
            temperature: Sampling temperature (ignored)
            max_tokens: Maximum tokens (ignored)
            **kwargs: Additional arguments (ignored)

        Yields:
            Words from the mock response text
        """
        await self._apply_chaos("stream")

        # Stream words with small delays
        for word in self.config.response_text.split():
            await asyncio.sleep(0.05)  # 50ms per word
            yield word + " "

    async def analyze_image(
        self,
        image_data: bytes,  # noqa: ARG002
        prompt: str,  # noqa: ARG002
        image_format: str = "png",  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> LLMResponse:
        """Analyze image with chaos effects.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for analysis
            image_format: Image format (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Mock image analysis response in JSON format
        """
        await self._apply_chaos("analyze_image")

        # Return structured JSON response for image analysis
        json_response = """{
    "alt_text": "chaos mock image",
    "detailed_description": "A mock image analysis for testing purposes",
    "detected_text": null,
    "image_type": "other",
    "knowledge_meta": null
}"""

        return LLMResponse(
            content=json_response,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=50),
            model=self.config.model_name,
            finish_reason="stop",
        )

    async def validate(self) -> bool:
        """Validate the provider (always succeeds).

        Returns:
            True always (chaos provider is always "valid")
        """
        return True

    @property
    def stats(self) -> ChaosStats:
        """Get chaos statistics.

        Returns:
            Statistics about calls, failures, and latencies
        """
        return self._stats

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Get call history for debugging.

        Returns:
            List of call records with request_id, operation, outcome, etc.
        """
        return self._call_history.copy()

    def reset_stats(self) -> None:
        """Reset all statistics and call history."""
        self._stats = ChaosStats()
        self._call_history.clear()

    def configure(self, **kwargs: Any) -> None:
        """Update configuration at runtime.

        Args:
            **kwargs: Configuration fields to update (e.g., failure_rate=0.5)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                log.warning(f"Unknown config key: {key}")

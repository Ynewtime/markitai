"""AIMD (Additive Increase Multiplicative Decrease) adaptive rate limiter.

This module implements an adaptive concurrency control mechanism based on
the AIMD algorithm, commonly used in TCP congestion control.

AIMD Algorithm:
- Additive Increase: After N consecutive successes, increase concurrency by 1
- Multiplicative Decrease: On rate limit (429), multiply concurrency by 0.5

Features:
- Dynamic semaphore management for concurrency control
- Cooldown period to prevent oscillation
- Statistics tracking for monitoring
- Callback support for external notifications

Example usage:
    ```python
    from markit.utils.adaptive_limiter import AdaptiveRateLimiter, AIMDConfig

    # Create limiter with custom config
    config = AIMDConfig(
        initial_concurrency=10,
        max_concurrency=50,
        success_threshold=20,
    )
    limiter = AdaptiveRateLimiter(config)

    # Use in async code
    async with limiter:
        await do_work()

    # Or manually acquire/release
    await limiter.acquire()
    try:
        result = await do_work()
        await limiter.record_success()
    except RateLimitError:
        await limiter.record_rate_limit()
    finally:
        limiter.release()
    ```
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AIMDConfig:
    """Configuration for AIMD rate limiter.

    Attributes:
        initial_concurrency: Starting concurrency level (default: 5)
        min_concurrency: Minimum concurrency floor (default: 1)
        max_concurrency: Maximum concurrency ceiling (default: 50)
        success_threshold: Number of successes before increase (default: 10)
        additive_increase: Amount to add on success streak (default: 1)
        multiplicative_decrease: Multiplier on rate limit (default: 0.5)
        cooldown_seconds: Cooldown after decrease before next decrease (default: 5.0)
    """

    initial_concurrency: int = 5
    min_concurrency: int = 1
    max_concurrency: int = 50
    success_threshold: int = 10
    additive_increase: int = 1
    multiplicative_decrease: float = 0.5
    cooldown_seconds: float = 5.0


@dataclass
class AIMDStats:
    """Statistics for AIMD limiter.

    Attributes:
        current_concurrency: Current concurrency level
        total_successes: Total successful requests
        total_failures: Total failed requests (non-rate-limit)
        rate_limit_hits: Total rate limit (429) errors
        increase_count: Number of concurrency increases
        decrease_count: Number of concurrency decreases
        peak_concurrency: Highest concurrency reached
        trough_concurrency: Lowest concurrency reached
    """

    current_concurrency: int = 5
    total_successes: int = 0
    total_failures: int = 0
    rate_limit_hits: int = 0
    increase_count: int = 0
    decrease_count: int = 0
    peak_concurrency: int = 5
    trough_concurrency: int = 5

    @property
    def total_requests(self) -> int:
        """Total number of requests processed."""
        return self.total_successes + self.total_failures + self.rate_limit_hits

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.total_successes / self.total_requests) * 100

    @property
    def rate_limit_rate(self) -> float:
        """Rate limit rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.rate_limit_hits / self.total_requests) * 100


class AdaptiveRateLimiter:
    """AIMD-based adaptive rate limiter.

    This limiter dynamically adjusts concurrency based on success/failure patterns:

    - After `success_threshold` consecutive successes, concurrency increases by
      `additive_increase` (capped at `max_concurrency`)
    - On rate limit (429), concurrency is multiplied by `multiplicative_decrease`
      (floored at `min_concurrency`)
    - A cooldown period prevents rapid oscillation after decreases

    The limiter can be used as an async context manager or via explicit
    acquire/release calls.
    """

    def __init__(
        self,
        config: AIMDConfig | None = None,
        on_adjust: Callable[[int, str], None] | None = None,
    ) -> None:
        """Initialize the adaptive rate limiter.

        Args:
            config: AIMD configuration (uses defaults if None)
            on_adjust: Optional callback when concurrency changes.
                       Called with (new_concurrency, direction) where direction
                       is "increase" or "decrease"
        """
        self.config = config or AIMDConfig()
        self._on_adjust = on_adjust

        self._current_concurrency = self.config.initial_concurrency
        self._semaphore = asyncio.Semaphore(self._current_concurrency)
        self._success_streak = 0
        self._last_decrease_time: float = 0.0
        self._lock = asyncio.Lock()

        self._stats = AIMDStats(
            current_concurrency=self._current_concurrency,
            peak_concurrency=self._current_concurrency,
            trough_concurrency=self._current_concurrency,
        )

    async def acquire(self) -> None:
        """Acquire a slot for execution.

        This method blocks until a slot is available.
        """
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a slot after execution."""
        self._semaphore.release()

    async def __aenter__(self) -> "AdaptiveRateLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.release()

    async def record_success(self) -> None:
        """Record a successful request.

        This increments the success streak. After `success_threshold`
        consecutive successes, concurrency is increased.
        """
        async with self._lock:
            self._success_streak += 1
            self._stats.total_successes += 1

            if self._success_streak >= self.config.success_threshold:
                await self._increase_concurrency()
                self._success_streak = 0

    async def record_rate_limit(self) -> None:
        """Record a rate limit (429) error.

        This triggers multiplicative decrease of concurrency, subject to
        the cooldown period.
        """
        async with self._lock:
            self._success_streak = 0
            self._stats.rate_limit_hits += 1

            # Check cooldown
            now = time.time()
            if now - self._last_decrease_time < self.config.cooldown_seconds:
                remaining = self.config.cooldown_seconds - (now - self._last_decrease_time)
                log.debug(
                    "Rate limit in cooldown, skipping decrease",
                    cooldown_remaining_seconds=round(remaining, 2),
                    current_concurrency=self._current_concurrency,
                )
                return

            await self._decrease_concurrency()
            self._last_decrease_time = now

    async def record_failure(self) -> None:
        """Record a non-rate-limit failure.

        This resets the success streak but does not trigger a decrease.
        """
        async with self._lock:
            self._success_streak = 0
            self._stats.total_failures += 1

    async def record_error(self, is_rate_limit: bool = False) -> None:
        """Record an error with unified interface.

        This is a convenience method that routes to either record_rate_limit()
        or record_failure() based on the error type.

        Args:
            is_rate_limit: True if this was a 429 rate limit error,
                          False for other errors (500, timeout, etc.)
        """
        if is_rate_limit:
            await self.record_rate_limit()
        else:
            await self.record_failure()

    async def _increase_concurrency(self) -> None:
        """Additive increase of concurrency."""
        old = self._current_concurrency
        new = min(
            self._current_concurrency + self.config.additive_increase,
            self.config.max_concurrency,
        )

        if new > old:
            self._current_concurrency = new
            self._stats.current_concurrency = new
            self._stats.increase_count += 1

            # Track peak
            if new > self._stats.peak_concurrency:
                self._stats.peak_concurrency = new

            # Recreate semaphore with new limit
            self._semaphore = asyncio.Semaphore(new)

            log.info(
                "Concurrency increased (additive)",
                old_concurrency=old,
                new_concurrency=new,
                success_streak=self.config.success_threshold,
                total_successes=self._stats.total_successes,
            )

            if self._on_adjust:
                self._on_adjust(new, "increase")

    async def _decrease_concurrency(self) -> None:
        """Multiplicative decrease of concurrency."""
        old = self._current_concurrency
        new = max(
            int(self._current_concurrency * self.config.multiplicative_decrease),
            self.config.min_concurrency,
        )

        if new < old:
            self._current_concurrency = new
            self._stats.current_concurrency = new
            self._stats.decrease_count += 1

            # Track trough
            if new < self._stats.trough_concurrency:
                self._stats.trough_concurrency = new

            # Recreate semaphore with new limit
            self._semaphore = asyncio.Semaphore(new)

            log.warning(
                "Concurrency decreased (multiplicative)",
                old_concurrency=old,
                new_concurrency=new,
                rate_limit_hits=self._stats.rate_limit_hits,
                multiplier=self.config.multiplicative_decrease,
            )

            if self._on_adjust:
                self._on_adjust(new, "decrease")

    @property
    def current_concurrency(self) -> int:
        """Get current concurrency level."""
        return self._current_concurrency

    @property
    def stats(self) -> AIMDStats:
        """Get AIMD statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset limiter to initial state.

        This resets concurrency to initial value and clears all statistics.
        """
        self._current_concurrency = self.config.initial_concurrency
        self._semaphore = asyncio.Semaphore(self._current_concurrency)
        self._success_streak = 0
        self._last_decrease_time = 0.0
        self._stats = AIMDStats(
            current_concurrency=self._current_concurrency,
            peak_concurrency=self._current_concurrency,
            trough_concurrency=self._current_concurrency,
        )

        log.debug(
            "AIMD limiter reset",
            initial_concurrency=self.config.initial_concurrency,
        )

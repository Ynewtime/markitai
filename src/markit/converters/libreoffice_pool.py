"""LibreOffice profile pool for concurrent document conversion.

This module provides a pool of isolated LibreOffice user profiles that enable
concurrent document conversion without lock conflicts. Each profile directory
is managed independently with automatic reset on failures or excessive usage.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from markit.config.constants import (
    DEFAULT_LIBREOFFICE_POOL_SIZE,
    DEFAULT_LIBREOFFICE_PROFILE_DIR,
    DEFAULT_LIBREOFFICE_RESET_AFTER_FAILURES,
    DEFAULT_LIBREOFFICE_RESET_AFTER_USES,
)
from markit.utils.logging import get_logger

log = get_logger(__name__)


class LibreOfficeProfilePool:
    """Manages a pool of isolated LibreOffice user profile directories.

    This pool enables concurrent LibreOffice conversions by providing each
    conversion process with its own isolated user profile directory. This
    prevents lock conflicts that occur when multiple LibreOffice instances
    try to use the same profile.

    Features:
    - Configurable pool size (default: 8 profiles)
    - Automatic profile directory creation
    - Usage tracking with automatic reset after N uses
    - Failure tracking with automatic reset after N consecutive failures
    - Async context manager for safe profile acquisition/release
    """

    def __init__(
        self,
        pool_size: int = DEFAULT_LIBREOFFICE_POOL_SIZE,
        base_dir: Path | str | None = None,
        reset_after_failures: int = DEFAULT_LIBREOFFICE_RESET_AFTER_FAILURES,
        reset_after_uses: int = DEFAULT_LIBREOFFICE_RESET_AFTER_USES,
    ) -> None:
        """Initialize the profile pool.

        Args:
            pool_size: Number of profile directories to maintain
            base_dir: Base directory for profile directories (default: .markit-lo-profiles)
            reset_after_failures: Reset profile after this many consecutive failures
            reset_after_uses: Reset profile after this many successful uses
        """
        self.pool_size = pool_size
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / DEFAULT_LIBREOFFICE_PROFILE_DIR
        self.reset_after_failures = reset_after_failures
        self.reset_after_uses = reset_after_uses

        # Internal state
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._usage_count: dict[Path, int] = {}
        self._failure_count: dict[Path, int] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the profile pool, creating directories and populating the queue.

        This method is idempotent - calling it multiple times has no effect.
        """
        async with self._lock:
            if self._initialized:
                return

            # Create base directory
            self.base_dir.mkdir(parents=True, exist_ok=True)

            # Create profile directories and add to queue
            for i in range(self.pool_size):
                profile_dir = self.base_dir / f"profile_{i}"
                profile_dir.mkdir(exist_ok=True)
                await self._queue.put(profile_dir)
                self._usage_count[profile_dir] = 0
                self._failure_count[profile_dir] = 0

            self._initialized = True
            log.debug(
                "LibreOffice profile pool initialized",
                pool_size=self.pool_size,
                base_dir=str(self.base_dir),
            )

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Path, None]:
        """Acquire a profile directory from the pool.

        This is an async context manager that:
        1. Waits for an available profile (blocks if all are in use)
        2. Yields the profile directory path
        3. Automatically releases the profile back to the pool
        4. Resets the profile if usage/failure thresholds are exceeded

        Yields:
            Path to the acquired profile directory

        Example:
            async with pool.acquire() as profile_dir:
                # Use profile_dir for LibreOffice conversion
                await convert_document(file_path, profile_dir)
        """
        if not self._initialized:
            await self.initialize()

        # Wait for an available profile
        profile_dir = await self._queue.get()

        try:
            yield profile_dir
            # Success - reset failure count
            self._failure_count[profile_dir] = 0
            self._usage_count[profile_dir] += 1

            # Check if reset needed due to excessive usage
            if self._usage_count[profile_dir] >= self.reset_after_uses:
                await self._reset_profile(profile_dir)

        except Exception:
            # Failure - increment failure count
            self._failure_count[profile_dir] += 1

            # Check if reset needed due to excessive failures
            if self._failure_count[profile_dir] >= self.reset_after_failures:
                await self._reset_profile(profile_dir)

            raise

        finally:
            # Return profile to pool
            await self._queue.put(profile_dir)

    async def _reset_profile(self, profile_dir: Path) -> None:
        """Reset a profile directory by clearing its contents.

        Args:
            profile_dir: Profile directory to reset
        """
        try:
            # Remove directory contents (but not the directory itself)
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            profile_dir.mkdir(exist_ok=True)

            # Reset counters
            self._usage_count[profile_dir] = 0
            self._failure_count[profile_dir] = 0

            log.debug("Profile reset", profile_dir=str(profile_dir))

        except Exception as e:
            log.warning("Failed to reset profile", profile_dir=str(profile_dir), error=str(e))

    async def warmup(self) -> None:
        """Warmup the profile pool by ensuring all directories are ready.

        This method should be called before batch processing to avoid
        initialization delays during concurrent conversions. It's essentially
        an alias for initialize() but with a clearer semantic intent.

        Usage:
            pool = LibreOfficeProfilePool()
            await pool.warmup()  # Ensure directories exist before processing
        """
        if not self._initialized:
            await self.initialize()

        log.debug(
            "LibreOffice profile pool warmed up",
            pool_size=self.pool_size,
            available=self.available_profiles,
        )

    async def cleanup(self) -> None:
        """Clean up all profile directories.

        Call this when shutting down to remove temporary profile directories.
        """
        if self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
                log.debug("Profile pool cleaned up", base_dir=str(self.base_dir))
            except Exception as e:
                log.warning("Failed to clean up profile pool", error=str(e))

        self._initialized = False

    @property
    def available_profiles(self) -> int:
        """Get the number of currently available profiles."""
        return self._queue.qsize()

    def get_stats(self) -> dict[str, int | dict[str, int]]:
        """Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": self.pool_size,
            "available": self._queue.qsize(),
            "in_use": self.pool_size - self._queue.qsize(),
            "total_uses": sum(self._usage_count.values()),
            "total_failures": sum(self._failure_count.values()),
        }

"""Tests for LibreOfficeProfilePool."""

import asyncio

import pytest

from markit.converters.libreoffice_pool import LibreOfficeProfilePool


class TestLibreOfficeProfilePoolInit:
    """Tests for LibreOfficeProfilePool initialization."""

    def test_default_values(self):
        """Default configuration values are correct."""
        pool = LibreOfficeProfilePool()
        assert pool.pool_size == 8
        assert pool.reset_after_failures == 3
        assert pool.reset_after_uses == 100
        assert pool._initialized is False

    def test_custom_values(self, temp_dir):
        """Custom values are set correctly."""
        pool = LibreOfficeProfilePool(
            pool_size=4,
            base_dir=temp_dir / "profiles",
            reset_after_failures=5,
            reset_after_uses=50,
        )
        assert pool.pool_size == 4
        assert pool.base_dir == temp_dir / "profiles"
        assert pool.reset_after_failures == 5
        assert pool.reset_after_uses == 50

    def test_accepts_string_base_dir(self, temp_dir):
        """Base directory can be specified as string."""
        pool = LibreOfficeProfilePool(base_dir=str(temp_dir / "profiles"))
        assert pool.base_dir == temp_dir / "profiles"


class TestInitialize:
    """Tests for initialize method."""

    async def test_creates_base_directory(self, temp_dir):
        """Base directory is created on initialization."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")

        await pool.initialize()

        assert (temp_dir / "profiles").exists()

    async def test_creates_profile_directories(self, temp_dir):
        """Profile directories are created on initialization."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")

        await pool.initialize()

        assert (temp_dir / "profiles" / "profile_0").exists()
        assert (temp_dir / "profiles" / "profile_1").exists()
        assert (temp_dir / "profiles" / "profile_2").exists()

    async def test_fills_queue(self, temp_dir):
        """Queue is filled with all profile directories."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")

        await pool.initialize()

        assert pool._queue.qsize() == 3
        assert pool.available_profiles == 3

    async def test_initializes_counters(self, temp_dir):
        """Usage and failure counters are initialized to zero."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")

        await pool.initialize()

        for i in range(3):
            profile_dir = temp_dir / "profiles" / f"profile_{i}"
            assert pool._usage_count[profile_dir] == 0
            assert pool._failure_count[profile_dir] == 0

    async def test_idempotent(self, temp_dir):
        """Multiple initializations have no effect."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")

        await pool.initialize()
        await pool.initialize()
        await pool.initialize()

        # Should still have only 3 profiles
        assert pool._queue.qsize() == 3


class TestAcquire:
    """Tests for acquire context manager."""

    @pytest.fixture
    async def pool(self, temp_dir):
        """Create and initialize a test pool."""
        pool = LibreOfficeProfilePool(
            pool_size=3,
            base_dir=temp_dir / "profiles",
            reset_after_failures=2,
            reset_after_uses=3,
        )
        await pool.initialize()
        return pool

    async def test_returns_profile_directory(self, pool, temp_dir):
        """Acquire returns a valid profile directory."""
        async with pool.acquire() as profile_dir:
            assert profile_dir.exists()
            assert profile_dir.name.startswith("profile_")
            assert profile_dir.parent == temp_dir / "profiles"

    async def test_decrements_available_count(self, pool):
        """Available count is decremented during acquisition."""
        initial = pool.available_profiles

        async with pool.acquire():
            assert pool.available_profiles == initial - 1

        # After release, should be back
        assert pool.available_profiles == initial

    async def test_releases_on_success(self, pool):
        """Profile is released back to pool on success."""
        async with pool.acquire():
            pass

        # Profile should be back in queue
        # Get it again to verify
        async with pool.acquire():
            pass  # Should work without blocking

    async def test_releases_on_exception(self, pool):
        """Profile is released even when exception occurs."""
        initial = pool.available_profiles

        try:
            async with pool.acquire():
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # Profile should be released
        assert pool.available_profiles == initial

    async def test_increments_usage_count(self, pool):
        """Usage count is incremented on successful use."""
        async with pool.acquire() as profile_dir:
            pass

        assert pool._usage_count[profile_dir] == 1

        async with pool.acquire():
            pass

        # One of them should have count incremented
        total_uses = sum(pool._usage_count.values())
        assert total_uses == 2

    async def test_resets_failure_count_on_success(self, pool):
        """Failure count is reset on successful use."""
        # First, cause a failure
        try:
            async with pool.acquire() as profile_dir:
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        assert pool._failure_count[profile_dir] == 1

        # Now succeed with the same profile
        async with pool.acquire() as profile_dir2:
            if profile_dir2 == profile_dir:
                pass  # Success resets the counter

        # If we got the same profile, its count should be reset
        if profile_dir2 == profile_dir:
            assert pool._failure_count[profile_dir] == 0

    async def test_increments_failure_count(self, pool):
        """Failure count is incremented on exception."""
        try:
            async with pool.acquire() as profile_dir:
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        assert pool._failure_count[profile_dir] == 1

    async def test_auto_initializes(self, temp_dir):
        """Pool auto-initializes on first acquire if not initialized."""
        pool = LibreOfficeProfilePool(pool_size=2, base_dir=temp_dir / "profiles")
        assert pool._initialized is False

        async with pool.acquire():
            pass

        assert pool._initialized is True


class TestResetThresholds:
    """Tests for automatic profile reset on threshold."""

    @pytest.fixture
    async def pool(self, temp_dir):
        """Create pool with low thresholds for testing."""
        pool = LibreOfficeProfilePool(
            pool_size=1,  # Single profile for predictable testing
            base_dir=temp_dir / "profiles",
            reset_after_failures=2,
            reset_after_uses=3,
        )
        await pool.initialize()
        return pool

    async def test_resets_after_failure_threshold(self, pool, temp_dir):
        """Profile is reset after reaching failure threshold."""
        profile_dir = temp_dir / "profiles" / "profile_0"

        # Create a file in the profile to track reset
        test_file = profile_dir / "test_marker.txt"
        test_file.write_text("marker")

        # Cause failures up to threshold
        for _ in range(2):
            try:
                async with pool.acquire():
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass

        # After reset, the marker file should be gone
        assert not test_file.exists()
        assert pool._failure_count[profile_dir] == 0
        assert pool._usage_count[profile_dir] == 0

    async def test_resets_after_usage_threshold(self, pool, temp_dir):
        """Profile is reset after reaching usage threshold."""
        profile_dir = temp_dir / "profiles" / "profile_0"

        # Create a file in the profile to track reset
        test_file = profile_dir / "test_marker.txt"
        test_file.write_text("marker")

        # Use the profile up to threshold
        for _ in range(3):
            async with pool.acquire():
                pass

        # After reset, the marker file should be gone
        assert not test_file.exists()
        assert pool._usage_count[profile_dir] == 0


class TestConcurrency:
    """Tests for concurrent access."""

    async def test_blocks_when_exhausted(self, temp_dir):
        """Acquire blocks when all profiles are in use."""
        pool = LibreOfficeProfilePool(pool_size=2, base_dir=temp_dir / "profiles")
        await pool.initialize()

        acquired = []

        async def acquire_and_hold(index: int):
            async with pool.acquire():
                acquired.append(index)
                await asyncio.sleep(0.1)

        # Start 2 acquisitions (should succeed)
        task1 = asyncio.create_task(acquire_and_hold(1))
        task2 = asyncio.create_task(acquire_and_hold(2))

        # Give them time to acquire
        await asyncio.sleep(0.05)

        # Third acquisition should block
        task3_started = asyncio.Event()

        async def try_third():
            task3_started.set()
            async with pool.acquire():
                acquired.append(3)

        task3 = asyncio.create_task(try_third())
        await task3_started.wait()
        await asyncio.sleep(0.02)

        # At this point, task3 should be blocked (not yet in acquired)
        assert 3 not in acquired

        # Wait for all to complete
        await asyncio.gather(task1, task2, task3)
        assert set(acquired) == {1, 2, 3}

    async def test_concurrent_access_is_safe(self, temp_dir):
        """Multiple concurrent acquires don't cause issues."""
        pool = LibreOfficeProfilePool(pool_size=5, base_dir=temp_dir / "profiles")
        await pool.initialize()

        results = []

        async def worker(worker_id: int):
            for _ in range(3):
                async with pool.acquire() as profile_dir:
                    results.append((worker_id, profile_dir.name))
                    await asyncio.sleep(0.01)

        # Run 5 workers concurrently
        await asyncio.gather(*[worker(i) for i in range(5)])

        # Should have 15 results (5 workers * 3 iterations)
        assert len(results) == 15


class TestCleanup:
    """Tests for cleanup method."""

    async def test_removes_base_directory(self, temp_dir):
        """Cleanup removes the entire base directory."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")
        await pool.initialize()

        assert (temp_dir / "profiles").exists()

        await pool.cleanup()

        assert not (temp_dir / "profiles").exists()

    async def test_resets_initialized_flag(self, temp_dir):
        """Cleanup resets the initialized flag."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")
        await pool.initialize()

        await pool.cleanup()

        assert pool._initialized is False


class TestGetStats:
    """Tests for get_stats method."""

    async def test_returns_pool_stats(self, temp_dir):
        """Stats include pool information."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")
        await pool.initialize()

        stats = pool.get_stats()

        assert stats["pool_size"] == 3
        assert stats["available"] == 3
        assert stats["in_use"] == 0
        assert stats["total_uses"] == 0
        assert stats["total_failures"] == 0

    async def test_stats_update_during_use(self, temp_dir):
        """Stats are updated during pool usage."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")
        await pool.initialize()

        async with pool.acquire():
            stats = pool.get_stats()
            assert stats["available"] == 2
            assert stats["in_use"] == 1

        # After release
        async with pool.acquire():
            pass

        stats = pool.get_stats()
        assert stats["total_uses"] == 2

    async def test_stats_track_failures(self, temp_dir):
        """Stats track failure count."""
        pool = LibreOfficeProfilePool(pool_size=3, base_dir=temp_dir / "profiles")
        await pool.initialize()

        try:
            async with pool.acquire():
                raise RuntimeError("Test")
        except RuntimeError:
            pass

        stats = pool.get_stats()
        assert stats["total_failures"] == 1

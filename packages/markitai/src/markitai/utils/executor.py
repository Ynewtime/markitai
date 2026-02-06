"""Shared ThreadPoolExecutor for CPU-bound converter operations."""

from __future__ import annotations

import asyncio
import os
import platform
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def _get_optimal_workers() -> int:
    """Get optimal thread pool size based on platform.

    Windows has higher thread context switch overhead (2-8 μs vs 1-3 μs on Linux),
    so we use a lower default to reduce switching costs.
    """
    cpu_count = os.cpu_count() or 4
    if platform.system() == "Windows":
        # Windows: lower default to reduce thread switch overhead
        return min(cpu_count, 4)
    else:
        # Linux/macOS: can use higher concurrency
        return min(cpu_count, 8)


# Global converter thread pool executor with thread-safe initialization
_CONVERTER_EXECUTOR: ThreadPoolExecutor | None = None
_CONVERTER_MAX_WORKERS = _get_optimal_workers()
_EXECUTOR_LOCK = threading.Lock()


def get_converter_executor() -> ThreadPoolExecutor:
    """Get or create the shared converter thread pool executor.

    Uses double-checked locking for thread-safe lazy initialization.

    Returns:
        Shared ThreadPoolExecutor instance for converter operations
    """
    global _CONVERTER_EXECUTOR
    if _CONVERTER_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            # Double-check after acquiring lock
            if _CONVERTER_EXECUTOR is None:
                _CONVERTER_EXECUTOR = ThreadPoolExecutor(
                    max_workers=_CONVERTER_MAX_WORKERS,
                    thread_name_prefix="markitai-converter",
                )
    return _CONVERTER_EXECUTOR


# Global heavy task semaphore to prevent OOM from LibreOffice/Playwright/etc.
_HEAVY_TASK_SEMAPHORE: asyncio.Semaphore | None = None
_HEAVY_TASK_CONFIGURED_LIMIT: int = 0  # 0 = auto-detect


def _get_total_ram_bytes() -> int | None:
    """Get total system RAM in bytes.

    Tries multiple methods:
    1. /proc/meminfo (Linux)
    2. os.sysconf (POSIX cross-platform)

    Returns:
        Total RAM in bytes, or None if detection fails.
    """
    # Method 1: /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024  # Convert kB to bytes
    except (OSError, ValueError, IndexError):
        pass

    # Method 2: os.sysconf (POSIX)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        pass

    return None


def _detect_heavy_task_limit() -> int:
    """Auto-detect heavy task semaphore limit based on system RAM.

    Returns:
        Optimal concurrent heavy task limit for available memory.
    """
    total_ram = _get_total_ram_bytes()
    if total_ram is None:
        return 2  # Conservative fallback

    gb = total_ram / (1024**3)
    if gb >= 32:
        return 8
    elif gb >= 16:
        return 4
    elif gb >= 8:
        return 3
    else:
        return 2


def get_heavy_task_semaphore(limit: int = 0) -> asyncio.Semaphore:
    """Get the global semaphore for heavyweight tasks (LibreOffice, etc.).

    Args:
        limit: Maximum concurrent heavy tasks. 0 = auto-detect from system RAM.
               Only used on first call; subsequent calls return the existing semaphore.

    Returns:
        Shared asyncio.Semaphore instance.
    """
    global _HEAVY_TASK_SEMAPHORE, _HEAVY_TASK_CONFIGURED_LIMIT
    if _HEAVY_TASK_SEMAPHORE is None:
        effective_limit = limit if limit > 0 else _detect_heavy_task_limit()
        _HEAVY_TASK_CONFIGURED_LIMIT = effective_limit
        _HEAVY_TASK_SEMAPHORE = asyncio.Semaphore(effective_limit)
    return _HEAVY_TASK_SEMAPHORE


async def run_in_converter_thread(
    func: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Run a function in the shared converter thread pool.

    This is used for CPU-bound converter operations (PDF parsing,
    document conversion, etc.) to avoid blocking the event loop.

    Args:
        func: Function to run in thread pool
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)
    """
    loop = asyncio.get_running_loop()
    executor = get_converter_executor()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def shutdown_converter_executor() -> None:
    """Shutdown the shared converter executor.

    Call this during application cleanup to ensure clean shutdown.
    """
    global _CONVERTER_EXECUTOR
    if _CONVERTER_EXECUTOR is not None:
        _CONVERTER_EXECUTOR.shutdown(wait=True)
        _CONVERTER_EXECUTOR = None

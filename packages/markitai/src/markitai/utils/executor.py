"""Shared ThreadPoolExecutor for CPU-bound converter operations."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

# Global converter thread pool executor with thread-safe initialization
_CONVERTER_EXECUTOR: ThreadPoolExecutor | None = None
_CONVERTER_MAX_WORKERS = min(os.cpu_count() or 4, 8)
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

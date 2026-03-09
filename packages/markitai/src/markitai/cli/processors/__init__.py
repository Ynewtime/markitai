"""CLI processors package.

This package contains processing functions for different input types:
- file: Single file processing
- url: URL and batch URL processing
- llm: LLM-based processing helpers
- validators: Validation helpers
- batch: Batch processing orchestration
"""

from __future__ import annotations

import asyncio
from typing import Any


async def run_parallel_llm_tasks(
    document_coro: Any,
    image_coro: Any,
    llm_ready_event: asyncio.Event,
) -> tuple[Any, Any]:
    """Run document and image LLM tasks together without leaking pending work.

    On failure, signals the event, cancels outstanding tasks, and re-raises.
    """
    document_task = asyncio.create_task(document_coro)
    image_task = asyncio.create_task(image_coro)

    try:
        return await asyncio.gather(document_task, image_task)
    except Exception:
        llm_ready_event.set()

        pending_tasks = [
            task for task in (document_task, image_task) if not task.done()
        ]
        for task in pending_tasks:
            task.cancel()

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        raise


from markitai.cli.processors.batch import process_batch
from markitai.cli.processors.file import process_single_file
from markitai.cli.processors.llm import (
    analyze_images_with_llm,
    enhance_document_with_vision,
    format_standalone_image_markdown,
    process_with_llm,
)
from markitai.cli.processors.url import process_url, process_url_batch
from markitai.cli.processors.validators import (
    check_playwright_for_urls,
    check_vision_model_config,
    warn_case_sensitivity_mismatches,
)

__all__ = [
    # File processing
    "process_single_file",
    # URL processing
    "process_url",
    "process_url_batch",
    # LLM processing
    "process_with_llm",
    "analyze_images_with_llm",
    "enhance_document_with_vision",
    "format_standalone_image_markdown",
    # Validators
    "check_vision_model_config",
    "check_playwright_for_urls",
    "warn_case_sensitivity_mismatches",
    # Batch processing
    "process_batch",
]

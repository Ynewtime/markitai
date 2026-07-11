"""Result types shared by the processing paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class Outcome:
    """One work item's result — the union of what the four processing paths
    (single file, single URL, URL batch, directory batch) track.

    Attributes:
        kind: Whether the work item is a local file or a URL.
        source: File name/path or URL identifying the item.
        status: Final item status as written to reports.
        output_path: Path to the produced output file (``.llm.md`` when LLM
            enhancement is enabled), or None when nothing was written.
        error: Error message when status is "failed".
        skip_reason: Why the item was skipped (file paths only, e.g.
            "exists", "image_only").
        images: Count of images extracted/downloaded for the item.
        screenshots: Count of screenshots rendered/captured for the item.
        cost_usd: Total LLM API cost for the item.
        llm_usage: Per-model usage stats
            ``{model: {requests, input_tokens, output_tokens, cost_usd}}``.
        cache_hit: Whether LLM results were served from cache
            (directory-batch tracking).
        fetch_cache_hit: Whether the URL fetch was served from cache
            (URL paths only).
        llm_cache_hit: Whether LLM was enabled but made no requests
            (URL paths only).
        fetch_strategy: The fetch strategy actually used (URL paths only).
        source_file: The .urls list the URL came from, or "cli" for
            single-URL runs (URL paths only).
        duration: Processing time in seconds.
    """

    kind: Literal["file", "url"]
    source: str
    status: Literal["completed", "failed", "skipped"]
    output_path: Path | None = None
    error: str | None = None
    skip_reason: str | None = None
    images: int = 0
    screenshots: int = 0
    cost_usd: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hit: bool = False
    fetch_cache_hit: bool = False
    llm_cache_hit: bool = False
    fetch_strategy: str | None = None
    source_file: str | None = None
    duration: float | None = None

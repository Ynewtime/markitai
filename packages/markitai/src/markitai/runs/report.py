"""Single-sourced report schema and exit-code matrix.

The report dict layouts here were extracted verbatim from the three CLI
call sites that used to build them inline (single file, single URL, and
URL batch in ``markitai.cli.processors``). The dicts produced here must
stay key-for-key and value-for-value identical to those originals
(dynamic timestamps aside); ``markitai.json_order.order_report`` applies
the final field ordering at write time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from markitai.runs.types import Outcome

REPORT_VERSION = "1.0"


def _llm_usage_totals(models: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Sum requests/input_tokens/output_tokens across per-model usage stats."""
    return {
        "requests": sum(u.get("requests", 0) for u in models.values()),
        "input_tokens": sum(u.get("input_tokens", 0) for u in models.values()),
        "output_tokens": sum(u.get("output_tokens", 0) for u in models.values()),
    }


def build_single_report(
    outcome: Outcome,
    *,
    log_file_path: Path | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the success report dict for a single-item (file or URL) run.

    Args:
        outcome: The completed item's result.
        log_file_path: Path to the run's log file, if any.
        options: Optional top-level ``options`` block (single-URL runs
            emit one; single-file runs do not).

    Returns:
        Report dict ready for ``atomic_write_json(..., order_func=order_report)``.
    """
    totals = _llm_usage_totals(outcome.llm_usage)

    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_file": str(log_file_path) if log_file_path else None,
    }
    if options is not None:
        report["options"] = options

    llm_usage_block = {
        "models": outcome.llm_usage,
        **totals,
        "cost_usd": outcome.cost_usd,
    }

    if outcome.kind == "file":
        report["summary"] = {
            "total_documents": 1,
            "completed_documents": 1,
            "failed_documents": 0,
            "duration": outcome.duration,
        }
        report["llm_usage"] = llm_usage_block
        report["documents"] = {
            outcome.source: {
                "status": outcome.status,
                "error": outcome.error,
                "output": str(outcome.output_path),
                "images": outcome.images,
                "screenshots": outcome.screenshots,
                "duration": outcome.duration,
                "llm_usage": {
                    "input_tokens": totals["input_tokens"],
                    "output_tokens": totals["output_tokens"],
                    "cost_usd": outcome.cost_usd,
                },
            }
        }
    else:
        report["summary"] = {
            "total_documents": 0,
            "completed_documents": 0,
            "failed_documents": 0,
            "total_urls": 1,
            "completed_urls": 1,
            "failed_urls": 0,
            "duration": outcome.duration,
        }
        report["llm_usage"] = llm_usage_block
        report["urls"] = {
            outcome.source: {
                "status": outcome.status,
                "source_file": outcome.source_file,
                "error": outcome.error,
                "output": str(outcome.output_path),
                "fetch_strategy": outcome.fetch_strategy,
                "fetch_cache_hit": outcome.fetch_cache_hit,
                "llm_cache_hit": outcome.llm_cache_hit,
                "images": outcome.images,
                "screenshots": outcome.screenshots,
                "duration": outcome.duration,
                "llm_usage": outcome.llm_usage,
            }
        }

    return report


def build_url_batch_report(
    results: dict[str, dict[str, Any]],
    *,
    total_urls: int,
    completed_urls: int,
    failed_urls: int,
    duration: float,
    llm_usage: dict[str, dict[str, Any]],
    cost_usd: float,
    log_file_path: Path | None = None,
) -> dict[str, Any]:
    """Build the report dict for a URL-batch run.

    Args:
        results: Per-URL result entries keyed by URL, as tracked by the
            batch workers (status/error/output/fetch_strategy/...).
        total_urls: Number of URLs in the batch.
        completed_urls: Number of URLs that completed.
        failed_urls: Number of URLs that failed.
        duration: Wall-clock batch duration in seconds.
        llm_usage: Aggregated per-model usage stats for the whole batch.
        cost_usd: Total LLM API cost for the whole batch.
        log_file_path: Path to the run's log file, if any.

    Returns:
        Report dict ready for ``atomic_write_json(..., order_func=order_report)``.
    """
    totals = _llm_usage_totals(llm_usage)
    return {
        "version": REPORT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_file": str(log_file_path) if log_file_path else None,
        "summary": {
            "total_documents": 0,
            "completed_documents": 0,
            "failed_documents": 0,
            "total_urls": total_urls,
            "completed_urls": completed_urls,
            "failed_urls": failed_urls,
            "duration": duration,
        },
        "llm_usage": {
            "models": llm_usage,
            **totals,
            "cost_usd": cost_usd,
        },
        "urls": results,
    }


def resolve_exit_code(failed: int, *, batch: bool) -> int:
    """Single source of the CLI exit-code matrix.

    Matrix:
    - batch run (directory batch / URL batch) with failures -> 10
      (partial failure)
    - single-item run (file or URL) with a failure -> 1
    - no failures -> 0 (dry runs also exit 0)

    Args:
        failed: Number of failed work items.
        batch: True for batch runs, False for single-item runs.

    Returns:
        The process exit code. Callers raise ``SystemExit`` only when the
        result is non-zero, preserving fall-through success paths.
    """
    if failed <= 0:
        return 0
    return 10 if batch else 1

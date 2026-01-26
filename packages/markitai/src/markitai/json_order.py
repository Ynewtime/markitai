"""JSON field ordering definitions and utilities.

This module provides standardized field ordering for JSON output files
(report.json, state.json, assets.json) to ensure consistent, readable output.

It also handles:
- Duration formatting (seconds -> human-readable)
- Cache details merging (fetch_cache_hit + llm_cache_hit -> cache_details)
- URL hierarchy transformation (flat urls -> grouped url_files)
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Field Order Definitions
# =============================================================================

# report.json top-level fields
REPORT_FIELD_ORDER = [
    "version",
    "generated_at",
    "started_at",
    "updated_at",
    "log_file",
    "options",
    "summary",
    "llm_usage",
    "documents",
    "url_sources",
]

# state.json top-level fields (minimal for resume)
STATE_FIELD_ORDER = [
    "version",
    "options",
    "documents",
    "urls",
]

# images.json top-level fields (formerly assets.json)
IMAGES_FIELD_ORDER = [
    "version",
    "created",
    "updated",
    "images",
]

# options fields (used in both report and state)
OPTIONS_FIELD_ORDER = [
    "concurrency",
    "llm",
    "cache",
    "ocr",
    "screenshot",
    "alt",
    "desc",
    "fetch_strategy",
    "models",
    "input_dir",
    "output_dir",
]

# summary fields
SUMMARY_FIELD_ORDER = [
    "total_documents",
    "completed_documents",
    "failed_documents",
    "pending_documents",
    "total_urls",
    "completed_urls",
    "failed_urls",
    "pending_urls",
    "url_cache_hits",
    "url_sources",
    "duration",
    "processing_time",
]

# llm_usage fields
LLM_USAGE_FIELD_ORDER = [
    "models",
    "requests",
    "input_tokens",
    "output_tokens",
    "cost_usd",
]

# llm_usage.models.{model} fields
LLM_MODEL_USAGE_FIELD_ORDER = [
    "requests",
    "input_tokens",
    "output_tokens",
    "cost_usd",
]

# documents.{path} fields (document entry)
FILE_ENTRY_FIELD_ORDER = [
    "status",
    "cache_hit",
    "output",
    "error",
    "started_at",
    "completed_at",
    "duration",
    "images",
    "screenshots",
    "cost_usd",
    "llm_usage",
]

# url_sources.{file}.urls.{url} fields (URL entry)
URL_ENTRY_FIELD_ORDER = [
    "status",
    "cache_hit",
    "cache_details",
    "output",
    "error",
    "fetch_strategy",
    "started_at",
    "completed_at",
    "duration",
    "images",
    "screenshots",
    "cost_usd",
    "llm_usage",
]

# url_sources.{file} fields (URL source file entry)
URL_FILE_ENTRY_FIELD_ORDER = [
    "total",
    "completed",
    "failed",
    "urls",
]

# cache_details fields
CACHE_DETAILS_FIELD_ORDER = [
    "fetch",
    "llm",
]

# images[].{item} fields (formerly assets[])
# Note: llm_usage is intentionally excluded (internal tracking only)
IMAGE_ENTRY_FIELD_ORDER = [
    "path",
    "alt",
    "desc",
    "text",
    "created",
    "source",
]


# =============================================================================
# Helper Functions
# =============================================================================


def _format_duration(seconds: float | None) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "00:03:06" or "32.5s" for short durations
    """
    if seconds is None:
        return "0s"

    if seconds < 60:
        return f"{seconds:.1f}s"

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# Ordering Functions
# =============================================================================


def order_dict(d: dict[str, Any], field_order: list[str]) -> dict[str, Any]:
    """Reorder dict keys according to field_order.

    Fields in field_order come first (in that order), followed by
    any remaining fields in their original order.

    Args:
        d: Dictionary to reorder
        field_order: List of field names in desired order

    Returns:
        New dict with reordered keys
    """
    if not isinstance(d, dict):
        return d

    ordered: dict[str, Any] = {}

    # First, add fields in the specified order
    for key in field_order:
        if key in d:
            ordered[key] = d[key]

    # Then, add any remaining fields not in the order list
    for key in d:
        if key not in ordered:
            ordered[key] = d[key]

    return ordered


def order_dict_keys_sorted(d: dict[str, Any]) -> dict[str, Any]:
    """Reorder dict keys alphabetically.

    Args:
        d: Dictionary to reorder

    Returns:
        New dict with alphabetically sorted keys
    """
    if not isinstance(d, dict):
        return d

    return {k: d[k] for k in sorted(d.keys())}


def _order_llm_usage(llm_usage: dict[str, Any]) -> dict[str, Any]:
    """Order llm_usage structure.

    Orders top-level fields and nested model usage fields.
    """
    if not llm_usage:
        return llm_usage

    result = order_dict(llm_usage, LLM_USAGE_FIELD_ORDER)

    # Order models sub-dict
    if "models" in result and isinstance(result["models"], dict):
        ordered_models = {}
        for model in sorted(result["models"].keys()):
            ordered_models[model] = order_dict(
                result["models"][model], LLM_MODEL_USAGE_FIELD_ORDER
            )
        result["models"] = ordered_models

    return result


def _transform_file_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Transform and order a file entry (local file).

    Converts duration to human-readable format and orders fields.
    """
    result = dict(entry)  # Copy

    # Convert duration to human-readable format
    if "duration" in result and isinstance(result["duration"], (int, float)):
        result["duration"] = _format_duration(result["duration"])

    # Order the result
    result = order_dict(result, FILE_ENTRY_FIELD_ORDER)

    # Order nested llm_usage
    if "llm_usage" in result and isinstance(result["llm_usage"], dict):
        ordered_usage = {}
        for model in sorted(result["llm_usage"].keys()):
            ordered_usage[model] = order_dict(
                result["llm_usage"][model], LLM_MODEL_USAGE_FIELD_ORDER
            )
        result["llm_usage"] = ordered_usage

    return result


def _transform_url_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Transform and order a URL entry.

    Builds cache_details from separate cache fields and converts duration.
    """
    result = dict(entry)  # Copy

    # Build cache_details from separate cache fields (if present)
    fetch_hit = result.pop("fetch_cache_hit", None)
    llm_hit = result.pop("llm_cache_hit", None)

    if fetch_hit is not None or llm_hit is not None:
        cache_hit = bool(fetch_hit) or bool(llm_hit)
        result["cache_hit"] = cache_hit
        result["cache_details"] = order_dict(
            {
                "fetch": bool(fetch_hit) if fetch_hit is not None else False,
                "llm": bool(llm_hit) if llm_hit is not None else False,
            },
            CACHE_DETAILS_FIELD_ORDER,
        )

    # Convert duration to human-readable format
    if "duration" in result and isinstance(result["duration"], (int, float)):
        result["duration"] = _format_duration(result["duration"])

    # Order the result
    result = order_dict(result, URL_ENTRY_FIELD_ORDER)

    # Order nested llm_usage
    if "llm_usage" in result and isinstance(result["llm_usage"], dict):
        ordered_usage = {}
        for model in sorted(result["llm_usage"].keys()):
            ordered_usage[model] = order_dict(
                result["llm_usage"][model], LLM_MODEL_USAGE_FIELD_ORDER
            )
        result["llm_usage"] = ordered_usage

    return result


def _order_image_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Order an image entry (formerly asset entry)."""
    result = order_dict(entry, IMAGE_ENTRY_FIELD_ORDER)

    # Order nested llm_usage
    if "llm_usage" in result and isinstance(result["llm_usage"], dict):
        ordered_usage = {}
        for model in sorted(result["llm_usage"].keys()):
            ordered_usage[model] = order_dict(
                result["llm_usage"][model], LLM_MODEL_USAGE_FIELD_ORDER
            )
        result["llm_usage"] = ordered_usage

    return result


def _transform_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Transform and order summary structure.

    Converts duration fields to human-readable format.
    """
    result = dict(summary)

    # Convert duration to human-readable format
    if "duration" in result and isinstance(result["duration"], (int, float)):
        result["duration"] = _format_duration(result["duration"])

    # Convert processing_time to human-readable format
    if "processing_time" in result and isinstance(
        result["processing_time"], (int, float)
    ):
        result["processing_time"] = _format_duration(result["processing_time"])

    return order_dict(result, SUMMARY_FIELD_ORDER)


def order_report(report: dict[str, Any]) -> dict[str, Any]:
    """Order and transform all fields in a report.json structure.

    This function:
    1. Converts durations to human-readable format
    2. Builds cache_details for URL entries
    3. Transforms flat urls to hierarchical url_files
    4. Orders all fields according to specification

    Args:
        report: Report dictionary

    Returns:
        New dict with all fields properly transformed and ordered
    """
    result = dict(report)  # Copy

    # Order options
    if "options" in result and isinstance(result["options"], dict):
        result["options"] = order_dict(result["options"], OPTIONS_FIELD_ORDER)

    # Transform summary (duration formatting)
    if "summary" in result and isinstance(result["summary"], dict):
        result["summary"] = _transform_summary(result["summary"])

    # Order llm_usage
    if "llm_usage" in result and isinstance(result["llm_usage"], dict):
        result["llm_usage"] = _order_llm_usage(result["llm_usage"])

    # Order documents
    if "documents" in result and isinstance(result["documents"], dict):
        ordered_files = {}
        for path in sorted(result["documents"].keys()):
            ordered_files[path] = _transform_file_entry(result["documents"][path])
        result["documents"] = ordered_files

    # Transform urls (flat dict) -> url_sources (hierarchical by source_file)
    if "urls" in result and isinstance(result["urls"], dict):
        url_sources: dict[str, dict[str, Any]] = {}

        for url, url_data in result["urls"].items():
            source_file = url_data.get("source_file", "unknown.urls")
            if source_file not in url_sources:
                url_sources[source_file] = {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "urls": {},
                }

            # Count status
            url_sources[source_file]["total"] += 1
            status = url_data.get("status", "pending")
            if status == "completed":
                url_sources[source_file]["completed"] += 1
            elif status == "failed":
                url_sources[source_file]["failed"] += 1

            # Add URL entry (without source_file - redundant in this structure)
            entry_data = {k: v for k, v in url_data.items() if k != "source_file"}
            url_sources[source_file]["urls"][url] = entry_data

        # Transform each file entry
        ordered_url_sources = {}
        for file_name in sorted(url_sources.keys()):
            file_entry = url_sources[file_name]
            ordered_entry = order_dict(file_entry, URL_FILE_ENTRY_FIELD_ORDER)

            # Transform urls within the file
            if "urls" in ordered_entry and isinstance(ordered_entry["urls"], dict):
                ordered_urls = {}
                for url, url_data in ordered_entry["urls"].items():
                    ordered_urls[url] = _transform_url_entry(url_data)
                ordered_entry["urls"] = ordered_urls

            ordered_url_sources[file_name] = ordered_entry

        result["url_sources"] = ordered_url_sources
        del result["urls"]

    # Transform url_sources if already in hierarchical format
    elif "url_sources" in result and isinstance(result["url_sources"], dict):
        ordered_url_sources = {}
        for file_name in sorted(result["url_sources"].keys()):
            file_entry = result["url_sources"][file_name]
            ordered_entry = order_dict(file_entry, URL_FILE_ENTRY_FIELD_ORDER)

            if "urls" in ordered_entry and isinstance(ordered_entry["urls"], dict):
                ordered_urls = {}
                for url, url_data in ordered_entry["urls"].items():
                    ordered_urls[url] = _transform_url_entry(url_data)
                ordered_entry["urls"] = ordered_urls

            ordered_url_sources[file_name] = ordered_entry
        result["url_sources"] = ordered_url_sources

    # Order top-level fields
    return order_dict(result, REPORT_FIELD_ORDER)


def order_state(state: dict[str, Any]) -> dict[str, Any]:
    """Order all fields in a state.json structure.

    Note: state.json keeps field values as-is (no duration formatting)
    for resume compatibility. Only ordering is applied.

    Args:
        state: State dictionary

    Returns:
        New dict with all fields properly ordered
    """
    result = order_dict(dict(state), STATE_FIELD_ORDER)

    # Order options
    if "options" in result and isinstance(result["options"], dict):
        result["options"] = order_dict(result["options"], OPTIONS_FIELD_ORDER)

    # Order documents (alphabetically by path)
    if "documents" in result and isinstance(result["documents"], dict):
        ordered_docs = {}
        for path in sorted(result["documents"].keys()):
            ordered_docs[path] = order_dict(
                result["documents"][path], FILE_ENTRY_FIELD_ORDER
            )
            # Order nested llm_usage
            if "llm_usage" in ordered_docs[path]:
                llm_usage = ordered_docs[path]["llm_usage"]
                if isinstance(llm_usage, dict):
                    ordered_usage = {}
                    for model in sorted(llm_usage.keys()):
                        ordered_usage[model] = order_dict(
                            llm_usage[model], LLM_MODEL_USAGE_FIELD_ORDER
                        )
                    ordered_docs[path]["llm_usage"] = ordered_usage
        result["documents"] = ordered_docs

    # Order urls (preserve original order for resume compatibility)
    if "urls" in result and isinstance(result["urls"], dict):
        ordered_urls = {}
        for url, url_data in result["urls"].items():
            ordered_urls[url] = order_dict(url_data, URL_ENTRY_FIELD_ORDER)
            # Order nested llm_usage
            if "llm_usage" in ordered_urls[url]:
                llm_usage = ordered_urls[url]["llm_usage"]
                if isinstance(llm_usage, dict):
                    ordered_usage = {}
                    for model in sorted(llm_usage.keys()):
                        ordered_usage[model] = order_dict(
                            llm_usage[model], LLM_MODEL_USAGE_FIELD_ORDER
                        )
                    ordered_urls[url]["llm_usage"] = ordered_usage
        result["urls"] = ordered_urls

    return result


def order_images(images: dict[str, Any]) -> dict[str, Any]:
    """Order all fields in an images.json structure (formerly assets.json).

    Also handles field name migration:
    - assets -> images
    - asset -> path (within each entry)

    Args:
        images: Images dictionary

    Returns:
        New dict with all fields properly ordered
    """
    result = order_dict(dict(images), IMAGES_FIELD_ORDER)

    # Order each image entry
    if "images" in result and isinstance(result["images"], list):
        ordered_images = []
        for image in result["images"]:
            ordered_image = order_dict(image, IMAGE_ENTRY_FIELD_ORDER)
            # Order nested llm_usage
            if "llm_usage" in ordered_image and isinstance(
                ordered_image["llm_usage"], dict
            ):
                ordered_usage = {}
                for model in sorted(ordered_image["llm_usage"].keys()):
                    ordered_usage[model] = order_dict(
                        ordered_image["llm_usage"][model], LLM_MODEL_USAGE_FIELD_ORDER
                    )
                ordered_image["llm_usage"] = ordered_usage
            ordered_images.append(ordered_image)
        result["images"] = ordered_images

    return result

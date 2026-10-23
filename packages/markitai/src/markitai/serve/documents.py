"""The document JSON of each finished item (contract v1.19/v1.20).

The document is built when the item finishes, from what its conversion knew
(warnings, model usage, picture analyses) beside its published Markdown, and
kept in the job folder outside ``out/`` (``documents/<item_id>.json``): the
result files and their archives stay what they were. ``GET
…/items/{item_id}/document`` serves it whatever ``format`` the job asked for.
A job with ``format: "json"`` also gets ``<base>.json`` beside the item's
Markdown in ``out/``, which is one of the item's files (artifacts, archives,
deletion). A rerun that publishes a new result replaces both; one that fails
keeps the previous result and so its documents.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.serve.jobs import Job, JobItem

_ITEM_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def document_path(job: Job, item_id: str) -> Path | None:
    """Where the item's document is kept; None for an ID that names no file."""
    if not _ITEM_ID.match(item_id):
        return None
    return job.job_dir / "documents" / f"{item_id}.json"


def beside_path(job: Job, item: JobItem) -> Path | None:
    """The item's ``<base>.json`` in ``out/`` (present only for ``format:
    "json"``)."""
    from markitai.serve.jobs import item_base_name

    base_name = item_base_name(item)
    if not base_name:
        return None
    out_dir = job.out_dir.resolve()
    path = (out_dir / f"{base_name}.json").resolve()
    return path if path.parent == out_dir else None


def keep(job: Job, item: JobItem, parts: Any, cfg: MarkitaiConfig) -> None:
    """Keep the document of the item's new result; *parts* is what its
    conversion returned (``ProcessResult.document``). A failure to keep it
    only means the document is not available."""
    from markitai.document_json import build_document, render
    from markitai.security import atomic_write_text

    parts = replace(
        parts,
        warnings=list(item.warnings),
        duration_s=(item.duration_ms or 0) / 1000,
        # A workbench item is its upload name, not the server's path
        source_label=item.name if item.kind == "file" else None,
    )
    target = document_path(job, item.item_id)
    if target is not None:
        try:
            target.parent.mkdir(mode=0o700, exist_ok=True)
            text = render(build_document(parts, job.out_dir.resolve()))
            atomic_write_text(target, text, private=True)
        except Exception as e:  # never fail the item over its document
            logger.warning(
                "[Serve] Could not keep the document of {}: {}", item.name, e
            )
            if target.exists():
                target.unlink(missing_ok=True)
    beside = beside_path(job, item)
    if beside is None:
        return
    try:
        if cfg.output.format == "json":
            atomic_write_text(beside, render(build_document(parts, beside.parent)))
        else:
            # A result without JSON: the previous attempt's goes
            beside.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[Serve] Could not write {}: {}", beside.name, e)


def forget(job: Job, item: JobItem) -> None:
    """Forget the kept document of a deleted item."""
    target = document_path(job, item.item_id)
    if target is not None:
        target.unlink(missing_ok=True)


def kept_document_files(job: Job, item: JobItem) -> list[Path]:
    """The item's kept document, for deleting it with the item."""
    target = document_path(job, item.item_id)
    return [target] if target is not None and target.is_file() else []

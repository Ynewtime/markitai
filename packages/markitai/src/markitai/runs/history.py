"""Record local CLI conversions into the ``markitai serve`` job history.

Serve jobs live under ``~/.markitai/serve/jobs/<job_id>/`` with a terminal
``meta.json`` snapshot; the server's rehydrate pass registers every
directory holding a valid terminal meta as archived history. This module
lets local CLI conversions (``markitai <path-or-url>``) opt into the same
history by writing a job-shaped directory with the same meta shape, so a
CLI run shows up next to web conversions in the history page.

Layering: deliberately self-contained — stdlib plus ``markitai.constants``
and ``markitai.security`` only. It must NOT import ``markitai.serve``
(the reader side): that would couple the layers and drag FastAPI into base
CLI installs. ``markitai.serve`` imports nothing from here either, except
the shared jobs-root constant below.

Crash safety: the job directory is assembled under a hidden ``.tmp-*``
sibling and renamed into place only after ``meta.json`` is written, so a
crash never leaves a meta-less half-written job dir behind (meta-less dirs
are skipped by rehydrate and reaped by the serve TTL cleanup anyway;
writing meta last is the key invariant).
"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.constants import ASSETS_REL_PATH, SCREENSHOTS_REL_PATH
from markitai.runs.types import Outcome
from markitai.security import atomic_write_json

# Single source of truth for the serve jobs root; ``markitai.serve.app``
# re-exports this as DEFAULT_JOBS_ROOT so the CLI writes where the server
# reads (serve sits above runs in the import-linter layering).
DEFAULT_SERVE_JOBS_ROOT = Path.home() / ".markitai" / "serve" / "jobs"

_META_FILENAME = "meta.json"
_JOB_ID_LENGTH = 12  # matches serve's uuid4().hex[:12] job ids


def _now_iso() -> str:
    """Browser-portable RFC 3339 timestamp (same format serve persists)."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _dir_size_bytes(path: Path) -> int:
    """Total size in bytes of all files under *path* (best effort)."""
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            continue
    return total


def _split_markdown_suffix(name: str) -> tuple[str, str]:
    """Split 'a.pdf.llm.md' -> ('a.pdf', '.llm.md'), 'a.md' -> ('a', '.md')."""
    if name.endswith(".llm.md"):
        return name[: -len(".llm.md")], ".llm.md"
    return os.path.splitext(name)


def _dedupe_output_name(name: str, used: set[str]) -> str:
    """Return *name* or a ' (N)'-suffixed variant not yet in *used*."""
    if name.casefold() not in used:
        return name
    stem, suffix = _split_markdown_suffix(name)
    counter = 2
    while f"{stem} ({counter}){suffix}".casefold() in used:
        counter += 1
    return f"{stem} ({counter}){suffix}"


def _find_meta_dirs(output_refs: list[tuple[Path, int]]) -> list[Path]:
    """Locate ``.markitai`` asset dirs referenced by the given outputs.

    Each entry is an output path plus how many directory levels its assets
    may live above it: flat layouts keep the ``.markitai`` dir next to the
    output, while nested directory batches keep it at the batch output root
    (the source relpath's depth). The walk stops at the first level that
    has a ``.markitai`` dir, so an unrelated ancestor's dir (e.g. a global
    ``~/.markitai``) is never picked up.
    """
    found: list[Path] = []
    for output_path, ascend in output_refs:
        level = output_path.parent
        for _ in range(ascend + 1):
            meta_dir = level / ".markitai"
            if meta_dir.is_dir():
                if meta_dir not in found:
                    found.append(meta_dir)
                break
            if level.parent == level:
                break
            level = level.parent
    return found


def _copy_item_output(item: Outcome, out_dir: Path, used_names: set[str]) -> str | None:
    """Copy one item's output file into the job out dir.

    Returns the output's relpath inside the out dir, or None when there is
    nothing (usable) to copy — the history item is still recorded, just
    without a downloadable output.
    """
    output_path = item.output_path
    if output_path is None:
        return None
    try:
        if not output_path.is_file():
            return None
        name = _dedupe_output_name(output_path.name, used_names)
        used_names.add(name.casefold())
        shutil.copy2(output_path, out_dir / name)
        return name
    except OSError as e:
        logger.warning(
            "[History] Could not copy output {} into the history job: {}",
            output_path,
            e,
        )
        return None


def _item_to_payload(
    item: Outcome, index: int, output: str | None, finished_at: str
) -> dict[str, Any]:
    """Map one CLI :class:`Outcome` onto serve's JobItem meta.json payload."""
    skipped = item.status == "skipped" or item.skip_reason is not None
    failed = item.status == "failed"
    return {
        "item_id": f"i{index}",
        "name": item.source,
        "kind": item.kind,
        "status": "error" if failed else "done",
        "error": item.error if failed else None,
        "output": output,
        "output_name": output,
        "duration_ms": (
            max(0, round(item.duration * 1000)) if item.duration is not None else None
        ),
        "finished_at": finished_at,
        "cost_usd": item.cost_usd,
        "llm_enhanced": bool(
            not skipped and output is not None and output.endswith(".llm.md")
        ),
        "operation": "convert",
        "skipped": skipped,
        "skip_reason": item.skip_reason,
    }


def record_cli_job(
    items: list[Outcome],
    *,
    options: dict[str, Any],
    jobs_root: Path,
    started_at: datetime | None = None,
) -> Path | None:
    """Record one finished CLI run as a serve history job directory.

    Args:
        items: Per-item results of the run (files and/or URLs, successes,
            skips and failures alike).
        options: Job options persisted to meta.json; must include
            ``origin: "cli"`` plus the preset/llm/ocr keys serve normalizes.
        jobs_root: Serve jobs root (``~/.markitai/serve/jobs`` by default).
        started_at: When the run started (drives ``created_at``).

    Returns:
        The final job directory, or None when there is nothing to record or
        recording failed. Recording is strictly best effort: any error is
        logged as a warning and swallowed so it can never break a
        conversion.
    """
    if not items:
        return None
    try:
        return _record_cli_job(
            items, options=options, jobs_root=jobs_root, started_at=started_at
        )
    except Exception as e:
        logger.warning("[History] Failed to record CLI run in serve history: {}", e)
        return None


def _record_cli_job(
    items: list[Outcome],
    *,
    options: dict[str, Any],
    jobs_root: Path,
    started_at: datetime | None,
) -> Path:
    """Assemble and publish the job directory (see record_cli_job)."""
    jobs_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:_JOB_ID_LENGTH]
    while (jobs_root / job_id).exists():
        job_id = uuid.uuid4().hex[:_JOB_ID_LENGTH]

    tmp_dir = jobs_root / f".tmp-{uuid.uuid4().hex}"
    out_dir = tmp_dir / "out"
    out_dir.mkdir(parents=True)
    try:
        finished_at = _now_iso()
        created_at = (
            started_at.astimezone().isoformat(timespec="milliseconds")
            if started_at is not None
            else finished_at
        )

        used_names: set[str] = set()
        outputs: list[str | None] = []
        for item in items:
            outputs.append(_copy_item_output(item, out_dir, used_names))

        # Preserve referenced image/screenshot assets so history downloads
        # and archives keep working links. Only the asset subdirs are
        # copied — reports/ and states/ are batch bookkeeping, not output.
        # The source relpath depth anchors where a nested batch's asset
        # root sits above the output; URL outputs are always flat.
        output_refs = [
            (
                item.output_path,
                len(Path(item.source).parent.parts) if item.kind == "file" else 0,
            )
            for item in items
            if item.output_path is not None
        ]
        for meta_dir in _find_meta_dirs(output_refs):
            for rel in (ASSETS_REL_PATH, SCREENSHOTS_REL_PATH):
                assets = meta_dir.parent / rel
                if assets.is_dir():
                    shutil.copytree(assets, out_dir / rel, dirs_exist_ok=True)

        meta = {
            "job_id": job_id,
            "created_at": created_at,
            "finished_at": finished_at,
            "status": "done",
            "options": options,
            "dir_size_bytes": _dir_size_bytes(tmp_dir),
            "items": [
                _item_to_payload(item, index, output, finished_at)
                for index, (item, output) in enumerate(
                    zip(items, outputs, strict=True), start=1
                )
            ],
        }
        # Meta goes last: rehydrate only registers dirs with a valid
        # terminal meta, so the rename below publishes the job atomically.
        atomic_write_json(tmp_dir / _META_FILENAME, meta)
        os.replace(tmp_dir, jobs_root / job_id)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    logger.debug("[History] Recorded CLI run as job {}", job_id)
    return jobs_root / job_id

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

import filecmp
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from loguru import logger

from markitai.constants import (
    ASSETS_REL_PATH,
    MARKITAI_META_DIR,
    SCREENSHOTS_REL_PATH,
    VISIBLE_ASSETS_REL_PATH,
)
from markitai.runs.types import Outcome
from markitai.security import atomic_write_json
from markitai.utils.clock import now_iso

# Single source of truth for the serve jobs root; ``markitai.serve.app``
# re-exports this as DEFAULT_JOBS_ROOT so the CLI writes where the server
# reads (serve sits above runs in the import-linter layering).
DEFAULT_SERVE_JOBS_ROOT = Path.home() / ".markitai" / "serve" / "jobs"

_META_FILENAME = "meta.json"
_JOB_ID_LENGTH = 12  # matches serve's uuid4().hex[:12] job ids

#: ``skip_reason`` of an item handed off to a still-running ``--llm-batch``.
PENDING_BATCH_SKIP_REASON = "pending_batch"
_PENDING_BATCH_NOTE = (
    "LLM enhancement is still running as a provider batch; only the base "
    "Markdown is here so far. Finish it with the `markitai --llm-batch-collect "
    "<batch-id> -o <output-dir>` command the CLI run printed, instead of "
    "enhancing again (that would pay twice)."
)


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


def _find_asset_root(output_path: Path, ascend: int) -> Path | None:
    """Locate the hidden or visible asset root an output's references use.

    *ascend* is how many directory levels the assets may live above the
    output: flat layouts keep the ``.markitai`` dir next to the output,
    while nested directory batches may keep it higher up (at most the
    source relpath's depth). The walk stops at the first level that has a
    ``.markitai`` or ``assets`` dir, so an unrelated ancestor's dir (e.g. a
    global ``~/.markitai``) is never picked up.
    """
    level = output_path.parent
    for _ in range(ascend + 1):
        meta_dir = level / ".markitai"
        if meta_dir.is_dir() or (level / VISIBLE_ASSETS_REL_PATH).is_dir():
            return level
        if level.parent == level:
            break
        level = level.parent
    return None


def _unique_asset_name(name: str, taken: set[str]) -> str:
    """Return *name* or a ``-N``-suffixed variant not yet in *taken*."""
    stem, suffix = os.path.splitext(name)
    counter = 2
    while f"{stem}-{counter}{suffix}".casefold() in taken:
        counter += 1
    return f"{stem}-{counter}{suffix}"


def _merge_asset_root(
    asset_root: Path, out_dir: Path, taken: dict[str, set[str]]
) -> dict[tuple[str, str], str]:
    """Copy one asset root's asset dirs into the job's flat out dir.

    Nested directory batches keep one asset root per output subdirectory,
    and the job flattens them into a single ``out/`` — so two roots can hold
    different files under one name (``a/report.pdf`` and ``b/report.pdf``
    both extract ``report.pdf-0001-01.jpg``). Identical files are copied
    once; a differing one gets a unique name instead of overwriting the
    first, and the rename is returned so the outputs of that root can have
    their references rewritten.

    Returns:
        ``{(asset rel dir, original name): new name}`` for renamed files.
    """
    renames: dict[tuple[str, str], str] = {}
    for rel in (ASSETS_REL_PATH, SCREENSHOTS_REL_PATH, VISIBLE_ASSETS_REL_PATH):
        src_dir = asset_root / rel
        if not src_dir.is_dir():
            continue
        dst_dir = out_dir / rel
        taken_here = taken.setdefault(rel, set())
        for src in sorted(src_dir.rglob("*")):
            if not src.is_file():
                continue
            rel_name = src.relative_to(src_dir).as_posix()
            dst = dst_dir / rel_name
            if dst.is_file():
                if filecmp.cmp(src, dst, shallow=False):
                    continue
                new_name = _unique_asset_name(rel_name, taken_here)
                renames[(rel, rel_name)] = new_name
                dst = dst_dir / new_name
                rel_name = new_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            taken_here.add(rel_name.casefold())
    return renames


def _rewrite_asset_refs(md_file: Path, renames: dict[tuple[str, str], str]) -> None:
    """Point a copied output's asset references at their renamed copies.

    Covers markdown links (raw or percent-encoded), wikilinks and
    frontmatter paths alike: every spelling is ``<asset dir>/<name>``
    followed by a delimiter, never a longer name.
    """
    text = md_file.read_text(encoding="utf-8")
    updated = text
    for (rel, old), new in renames.items():
        for old_spelling, new_spelling in {
            (f"{rel}/{old}", f"{rel}/{new}"),
            (quote(f"{rel}/{old}", safe="/._~-"), quote(f"{rel}/{new}", safe="/._~-")),
        }:
            updated = re.sub(
                # Not a longer name, and visible ``assets/`` never matches
                # inside a hidden ``.markitai/assets/`` path
                rf"(?<![\w.%-])(?<!{re.escape(MARKITAI_META_DIR)}/)"
                rf"{re.escape(old_spelling)}(?![\w.%-])",
                lambda _m, repl=new_spelling: repl,
                updated,
            )
    if updated != text:
        md_file.write_text(updated, encoding="utf-8")


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
    """Map one CLI :class:`Outcome` onto serve's JobItem meta.json payload.

    ``output`` is the file the item produced (``x.llm.md`` for an LLM run);
    ``output_name`` follows serve's meaning, the item's *base* ``x.md`` name
    the ``.llm.md`` sibling derives from, so a serve retry of a recorded URL
    writes ``x.md``/``x.llm.md`` rather than ``x.llm.md``/``x.llm.llm.md``.
    File items are recorded as not retryable: the run keeps no copy of the
    original for serve to convert again.

    A ``"pending"`` item (``--llm-batch`` handed off to a batch that was
    still running) is not a plain success: it only has its base ``.md`` and
    the enhancement is already paid for. It is recorded as a skip with
    reason ``pending_batch`` and a note naming the collect command, so the
    history never shows it as done-without-LLM inviting a second Enhance.
    """
    pending = item.status == "pending"
    skipped = pending or item.status == "skipped" or item.skip_reason is not None
    failed = item.status == "failed"
    output_name = None
    if output is not None:
        stem, suffix = _split_markdown_suffix(output)
        output_name = f"{stem}.md" if suffix in (".md", ".llm.md") else output
    return {
        "item_id": f"i{index}",
        "name": item.source,
        "kind": item.kind,
        "status": "error" if failed else "done",
        "error": item.error if failed else (_PENDING_BATCH_NOTE if pending else None),
        "output": output,
        "output_name": output_name,
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
        "skip_reason": PENDING_BATCH_SKIP_REASON if pending else item.skip_reason,
        "retryable": item.kind != "file",
        "warnings": list(dict.fromkeys(item.warnings)),
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
        finished_at = now_iso()
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
        # The source relpath depth bounds how far above the output a nested
        # batch's asset root may sit; URL outputs are always flat. Roots are
        # merged into the one out dir with clashing names made unique, and
        # each copied output's references follow its own root's renames.
        root_outputs: dict[Path, list[str]] = {}
        for item, output in zip(items, outputs, strict=True):
            if item.output_path is None:
                continue
            ascend = len(Path(item.source).parent.parts) if item.kind == "file" else 0
            asset_root = _find_asset_root(item.output_path, ascend)
            if asset_root is None:
                continue
            copied = root_outputs.setdefault(asset_root, [])
            if output is not None:
                copied.append(output)
        taken_assets: dict[str, set[str]] = {}
        for asset_root, copied_outputs in root_outputs.items():
            renames = _merge_asset_root(asset_root, out_dir, taken_assets)
            if renames:
                for output in copied_outputs:
                    _rewrite_asset_refs(out_dir / output, renames)

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

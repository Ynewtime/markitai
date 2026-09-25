"""In-memory job registry and runner driving the conversion core.

Jobs live in memory (the registry is lost on restart); their uploads and
outputs live on disk under ``<jobs_root>/<job_id>/``. Each job runs as one
asyncio task that fans item progress events out to SSE subscribers via
per-subscriber queues.

Layering note: this module deliberately builds on the UI-free core
(``workflow.core.convert_document_core`` for files, ``fetch.fetch_url`` plus
``workflow.helpers``/``LLMProcessor`` for URLs) instead of importing the
``markitai.cli.processors`` closures — ``markitai.serve`` sits below the CLI
in the import-linter contracts and must not import ``markitai.cli``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from markitai.serve.schemas import JobOptions
from markitai.utils.clock import now_iso

if TYPE_CHECKING:
    from markitai.batch import ProcessResult
    from markitai.config import MarkitaiConfig
    from markitai.llm import LLMProcessor

JOB_TTL_HOURS = 7 * 24.0  # conversion history is kept for 7 days
META_FILENAME = "meta.json"
#: Queue sentinel telling an SSE stream the server is shutting down. Never
#: sent to clients: the stream just ends.
SHUTDOWN_EVENT = "__shutdown__"


def normalize_job_options(options: dict[str, Any]) -> dict[str, Any]:
    """Read current and legacy persisted settings through the request schema."""
    values = {
        key: value for key, value in options.items() if key in JobOptions.model_fields
    }
    try:
        normalized = JobOptions.model_validate(values).model_dump()
    except ValidationError:
        logger.warning("[Serve] Invalid persisted job options; using defaults")
        normalized = JobOptions().model_dump()
    if "origin" in options:
        normalized["origin"] = options["origin"]
    return normalized


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JobItem:
    """One unit of work inside a job (an uploaded file or a URL)."""

    item_id: str
    name: str
    kind: str  # "file" | "url"
    source: Any  # Path (uploaded file) or str (URL)
    status: str = "queued"  # queued | running | done | error
    error: str | None = None
    output: str | None = None  # relpath inside the job out dir
    output_name: str | None = None  # pre-assigned unique output (url items)
    duration_ms: int | None = None
    finished_at: str | None = None
    cost_usd: float | None = None
    llm_enhanced: bool = False  # selected output is an .llm.md variant
    operation: str = "convert"  # convert | retry | enhance
    skipped: bool = False  # completed as a skip (status stays "done")
    skip_reason: str | None = None  # e.g. "exists", "image_only"
    options: dict[str, Any] | None = None  # persisted per-item retry settings
    # False for items whose source cannot be re-run from the web UI: a file
    # recorded from a CLI run (--record-history) has no uploaded original.
    retryable: bool = True
    # Actionable notices raised while this item converted ("pages look
    # scanned", "hidden text detected", ...): the web user has no console.
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Return the item event payload (contract: ``event: item``)."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "output": self.output,
            # persisted so a rehydrated URL item retries into the same
            # de-conflicted output name instead of recomputing (and colliding)
            "output_name": self.output_name,
            "duration_ms": self.duration_ms,
            "finished_at": self.finished_at,
            "cost_usd": self.cost_usd,
            "llm_enhanced": self.llm_enhanced,
            "operation": self.operation,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "retryable": self.retryable,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class RetryWork:
    """One in-place item retry waiting on a job's serial worker.

    ``prior_*`` snapshots the item's previous successful (non-skipped) result
    so a failed rerun can be rolled back to it — enhancing or retrying a done
    item must never destroy the base output it already had.
    """

    item_id: str
    cfg: MarkitaiConfig
    operation: str = "retry"
    prior_output: str | None = None
    prior_cost_usd: float | None = None
    prior_duration_ms: int | None = None
    prior_finished_at: str | None = None
    prior_operation: str = "convert"
    prior_llm_enhanced: bool = False
    prior_warnings: list[str] = field(default_factory=list)


@dataclass
class Job:
    """A conversion job: options, per-job config, items and subscribers."""

    job_id: str
    job_dir: Path
    created_at: str
    options: dict[str, Any]
    cfg: MarkitaiConfig
    items: list[JobItem] = field(default_factory=list)
    status: str = "running"  # running | done
    finished_at: str | None = None  # set when the job reaches its terminal state
    dir_size_bytes: int | None = None  # cached at terminal state (see meta.json)
    task: asyncio.Task[None] | None = None  # the initial run
    retry_task: asyncio.Task[None] | None = None  # the retry-queue drainer
    # Count of live runners (initial run + retry drainer). The job is only
    # finalized when this reaches zero AND no retry work is outstanding, so a
    # retry queued while the initial run is still active is never stranded and
    # the two never double- or prematurely-finalize.
    runners: int = 0
    generation: int = 0  # invalidates a finalizer when new work is accepted
    public_network_only: bool = False
    subscribers: list[asyncio.Queue[tuple[str, dict[str, Any]]]] = field(
        default_factory=list
    )
    archive_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Serializes item / history deletion: each awaits file work off-thread,
    # and two of them interleaving would remove one row twice or leave an
    # empty job behind in history.
    delete_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    retry_queue: asyncio.Queue[RetryWork] = field(
        default_factory=asyncio.Queue, repr=False
    )
    retry_pending: set[str] = field(default_factory=set, repr=False)

    @property
    def uploads_dir(self) -> Path:
        """Directory holding the uploaded originals."""
        return self.job_dir / "uploads"

    @property
    def out_dir(self) -> Path:
        """Output directory passed to the conversion core."""
        return self.job_dir / "out"

    @property
    def done_count(self) -> int:
        """Number of items that finished successfully (including skips)."""
        return sum(1 for i in self.items if i.status == "done")

    @property
    def failed_count(self) -> int:
        """Number of items that failed."""
        return sum(1 for i in self.items if i.status == "error")

    @property
    def skipped_count(self) -> int:
        """Number of items that completed as a skip."""
        return sum(1 for i in self.items if i.skipped)

    def get_item(self, item_id: str) -> JobItem | None:
        """Return the item with *item_id*, or None."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def progress_payload(self) -> dict[str, Any]:
        """Return the job event payload (contract: ``event: job``)."""
        return {
            "status": self.status,
            "done": self.done_count,
            "failed": self.failed_count,
            "total": len(self.items),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the full job JSON (contract: ``GET /api/jobs/{id}``)."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "options": self.options,
            "done": self.done_count,
            "failed": self.failed_count,
            "total": len(self.items),
            "items": [i.to_payload() for i in self.items],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class JobRegistry:
    """In-memory job store with SSE event fan-out."""

    def __init__(self, jobs_root: Path) -> None:
        self.jobs_root = jobs_root
        self.jobs: dict[str, Job] = {}
        self.archive_lock = asyncio.Lock()
        # Set once the server starts shutting down: SSE streams end instead
        # of waiting for their job, so they never hold the shutdown open.
        self.closing = False

    def create_job(self, options: dict[str, Any], cfg: MarkitaiConfig) -> Job:
        """Create a job with fresh uploads/out directories on disk."""
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / job_id
        job = Job(
            job_id=job_id,
            job_dir=job_dir,
            created_at=now_iso(),
            options=options,
            cfg=cfg,
        )
        job.uploads_dir.mkdir(parents=True, exist_ok=True)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job_id] = job
        return job

    def discard_job(self, job: Job) -> None:
        """Remove a job and its on-disk directory.

        Used both for creation-time rollback and for history deletion
        (``DELETE /api/history/{job_id}``).
        """
        self.jobs.pop(job.job_id, None)
        shutil.rmtree(job.job_dir, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        """Return the job with *job_id*, or None."""
        return self.jobs.get(job_id)

    def subscribe(self, job: Job) -> asyncio.Queue[tuple[str, dict[str, Any]]]:
        """Register a new SSE subscriber queue for *job*."""
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        job.subscribers.append(queue)
        return queue

    def unsubscribe(
        self, job: Job, queue: asyncio.Queue[tuple[str, dict[str, Any]]]
    ) -> None:
        """Remove an SSE subscriber queue from *job*."""
        if queue in job.subscribers:
            job.subscribers.remove(queue)

    def publish(self, job: Job, event: str, data: dict[str, Any]) -> None:
        """Fan one event out to every subscriber of *job*."""
        for queue in list(job.subscribers):
            queue.put_nowait((event, data))

    def publish_item(self, job: Job, item: JobItem) -> None:
        """Publish an ``item`` event for *item*."""
        self.publish(job, "item", item.to_payload())

    def publish_job(self, job: Job) -> None:
        """Publish a ``job`` progress event."""
        self.publish(job, "job", job.progress_payload())

    def begin_shutdown(self) -> None:
        """End every open SSE stream (and refuse to keep new ones open).

        The ASGI server waits for open responses before it runs the lifespan
        shutdown; an SSE stream following a long job would otherwise hold it
        open until the job finishes, and a second Ctrl-C then skips
        :meth:`shutdown` entirely (no meta.json, the job vanishes from
        history). Idempotent; must run on the event loop thread.
        """
        if self.closing:
            return
        self.closing = True
        for job in self.jobs.values():
            self.publish(job, SHUTDOWN_EVENT, {})

    async def shutdown(self) -> None:
        """Cancel all still-running job tasks (server shutdown).

        Their items end as ``cancelled (server shutdown)`` and each job is
        finalized, so its meta.json is written and it rehydrates as history.
        """
        self.begin_shutdown()
        tasks = [
            task
            for job in self.jobs.values()
            for task in (job.task, job.retry_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                pass


def cleanup_stale_jobs(jobs_root: Path, ttl_hours: float = JOB_TTL_HOURS) -> int:
    """Delete job directories whose mtime is older than *ttl_hours*.

    Returns:
        Number of removed job directories.
    """
    if not jobs_root.is_dir():
        return 0
    # A history archive left behind by an interrupted download (its normal
    # unlink is a response background task) lives at the jobs root, not inside
    # any job dir, so nothing else reclaims it.
    (jobs_root / "archive.zip").unlink(missing_ok=True)
    cutoff = time.time() - ttl_hours * 3600
    for archive in jobs_root.glob("archive-*.zip"):
        try:
            if archive.stat().st_mtime < cutoff:
                archive.unlink(missing_ok=True)
        except OSError:
            continue
    removed = 0
    for entry in jobs_root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Conversion history (meta.json persistence + startup rehydrate)
# ---------------------------------------------------------------------------


def write_job_meta(job: Job, *, refresh_size: bool = True) -> None:
    """Persist the job's terminal snapshot to ``<job_dir>/meta.json``.

    Written once when the job reaches its terminal state; startup rehydrate
    reads it back to register the job as archived history. Every caller is a
    terminal transition (finalize, item deletion), so the cached directory
    size is refreshed here — /api/history must not rglob per refresh.

    ``refresh_size=False`` skips the rglob and trusts ``job.dir_size_bytes``;
    finalize_job precomputes it off-thread so the loop never rglobs.
    """
    from markitai.security import atomic_write_json

    if refresh_size:
        job.dir_size_bytes = job_dir_size(job.job_dir)
    atomic_write_json(
        job.job_dir / META_FILENAME,
        {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "status": job.status,
            "options": job.options,
            "dir_size_bytes": job.dir_size_bytes,
            "version": 2,
            "items": [
                {**item.to_payload(), "options": item.options or job.options}
                for item in job.items
            ],
        },
    )


def _base_output_name(value: Any) -> str | None:
    """Normalize a persisted ``output_name`` to the base ``.md`` name.

    ``output_name`` is the item's base output (the ``.llm.md`` sibling is
    derived from it). CLI history recorded before that was pinned down
    stored the actual ``x.llm.md`` output there; a retry would then write
    ``x.llm.llm.md`` and put unenhanced markdown into ``x.llm.md``.
    """
    if not isinstance(value, str) or not value:
        return None
    if value.endswith(".llm.md"):
        return f"{value.removesuffix('.llm.md')}.md"
    return value


def _item_from_payload(
    raw: dict[str, Any],
    index: int,
    fallback_finished_at: str | None = None,
    *,
    origin: str | None = None,
) -> JobItem:
    """Rebuild a JobItem from its meta.json payload dict."""
    kind = str(raw.get("kind") or "file")
    raw_warnings = raw.get("warnings")
    warnings = (
        [str(w) for w in raw_warnings if isinstance(w, str)]
        if isinstance(raw_warnings, list)
        else []
    )
    retryable = raw.get("retryable")
    if not isinstance(retryable, bool):
        # CLI-recorded files keep no original to convert again.
        retryable = not (origin == "cli" and kind == "file")
    return JobItem(
        item_id=str(raw.get("item_id") or f"i{index}"),
        name=str(raw.get("name") or ""),
        kind=kind,
        source=None,  # original upload path / URL is not needed for archives
        status=str(raw.get("status") or "done"),
        error=raw.get("error"),
        output=raw.get("output"),
        output_name=_base_output_name(raw.get("output_name")),
        duration_ms=raw.get("duration_ms"),
        finished_at=raw.get("finished_at") or fallback_finished_at,
        cost_usd=raw.get("cost_usd"),
        llm_enhanced=bool(
            raw.get("llm_enhanced", str(raw.get("output") or "").endswith(".llm.md"))
        ),
        operation=str(raw.get("operation") or "convert"),
        skipped=bool(raw.get("skipped", False)),
        skip_reason=raw.get("skip_reason"),
        options=normalize_job_options(raw["options"])
        if isinstance(raw.get("options"), dict)
        else None,
        retryable=retryable,
        warnings=warnings,
    )


def rehydrate_jobs(registry: JobRegistry, cfg: MarkitaiConfig) -> int:
    """Register terminal jobs found on disk as archived registry entries.

    Scans the jobs root for directories with a valid terminal ``meta.json``
    and registers them so the snapshot/result/files/archive/history endpoints
    keep serving them across restarts. Directories with missing or malformed
    meta files (e.g. a job interrupted by a crash) are skipped.

    Args:
        registry: The registry to populate.
        cfg: Base config attached to rehydrated jobs (never used to run them).

    Returns:
        Number of rehydrated jobs.
    """
    if not registry.jobs_root.is_dir():
        return 0
    count = 0
    for entry in sorted(registry.jobs_root.iterdir()):
        meta_path = entry / META_FILENAME
        if not entry.is_dir() or entry.name in registry.jobs:
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("[Serve] Skipping unreadable {}: {}", meta_path, e)
            continue
        if not isinstance(meta, dict) or meta.get("status") != "done":
            logger.warning("[Serve] Skipping non-terminal meta {}", meta_path)
            continue
        raw_items = meta.get("items")
        if not isinstance(raw_items, list):
            raw_items = []
        raw_options = meta.get("options")
        options = raw_options if isinstance(raw_options, dict) else {}
        normalized_options = normalize_job_options(options)
        raw_size = meta.get("dir_size_bytes")
        job = Job(
            job_id=entry.name,
            job_dir=entry,
            created_at=str(meta.get("created_at") or ""),
            options=normalized_options,
            cfg=cfg,
            items=[
                _item_from_payload(
                    raw, index, meta.get("finished_at"), origin=options.get("origin")
                )
                for index, raw in enumerate(raw_items, start=1)
                if isinstance(raw, dict)
            ],
            status="done",
            finished_at=meta.get("finished_at"),
            dir_size_bytes=raw_size if isinstance(raw_size, int) else None,
        )
        registry.jobs[job.job_id] = job
        count += 1
    return count


def job_duration_ms(job: Job) -> int | None:
    """Duration of the latest conversion pass, with a legacy fallback.

    Retrying or enhancing an old job updates ``finished_at`` but intentionally
    preserves ``created_at``. Their wall-clock difference can therefore span
    hours; rerun item durations describe the actual latest pass and keep
    history meaningful. Initial jobs retain their accurate wall-clock timing;
    the longest item is only the fallback for incomplete legacy timestamps.
    """
    item_durations = [
        item.duration_ms for item in job.items if item.duration_ms is not None
    ]
    rerun_durations = [
        item.duration_ms
        for item in job.items
        if item.operation != "convert" and item.duration_ms is not None
    ]
    if rerun_durations:
        return max(rerun_durations)

    if job.created_at and job.finished_at:
        try:
            created = datetime.fromisoformat(job.created_at)
            finished = datetime.fromisoformat(job.finished_at)
            return max(0, int((finished - created).total_seconds() * 1000))
        except (TypeError, ValueError):
            # Mixed offset-aware/naive or edited legacy timestamps.
            pass
    return max(item_durations) if item_durations else None


def job_dir_size(job_dir: Path) -> int:
    """Total size in bytes of all files under *job_dir* (best effort)."""
    total = 0
    try:
        for file in job_dir.rglob("*"):
            try:
                if file.is_file():
                    total += file.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _backfill_meta_dir_size(job: Job) -> None:
    """Write the freshly computed size into a legacy meta.json (best effort)."""
    from markitai.security import atomic_write_json

    meta_path = job.job_dir / META_FILENAME
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(meta, dict):
        return
    meta["dir_size_bytes"] = job.dir_size_bytes
    try:
        atomic_write_json(meta_path, meta)
    except OSError:
        pass


def job_cached_dir_size(job: Job) -> int:
    """Return the job dir size cached at its terminal state.

    Jobs rehydrated from a meta.json that predates ``dir_size_bytes`` get
    the size computed once here and persisted back, so the full rglob never
    runs more than once per job across history refreshes and restarts.
    """
    if job.dir_size_bytes is None:
        job.dir_size_bytes = job_dir_size(job.job_dir)
        _backfill_meta_dir_size(job)
    return job.dir_size_bytes


# ---------------------------------------------------------------------------
# Item processing (UI-free wrappers around the conversion core)
# ---------------------------------------------------------------------------


async def process_file_item(
    file_path: Path,
    cfg: MarkitaiConfig,
    out_dir: Path,
    shared_processor: LLMProcessor | None,
) -> ProcessResult:
    """Convert one uploaded file via ``convert_document_core``.

    Thin serve-side counterpart of the CLI batch worker: the ProcessResult
    mapping (skip semantics, ``.llm.md`` output selection) is the shared
    ``workflow.results.document_process_result``; serve only adds the
    ``images.json`` write.
    """
    from markitai.batch import ProcessResult
    from markitai.constants import MAX_DOCUMENT_SIZE
    from markitai.utils.text import format_error_message
    from markitai.workflow.core import ConversionContext, convert_document_core
    from markitai.workflow.results import document_process_result

    try:
        ctx = ConversionContext(
            input_path=file_path,
            output_dir=out_dir,
            config=cfg,
            shared_processor=shared_processor,
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if (
            result.success
            and result.skip_reason is None
            and cfg.image.desc_enabled
            and ctx.image_analysis is not None
        ):
            from markitai.output_profiles import assets_visible
            from markitai.workflow.helpers import write_images_json

            write_images_json(
                out_dir, [ctx.image_analysis], visible_assets=assets_visible(cfg)
            )

        return document_process_result(ctx, result)
    except Exception as e:
        return ProcessResult(success=False, error=format_error_message(e))


@dataclass
class UrlJobContext:
    """Per-job URL processing context (strategy, cache, screenshot dir)."""

    strategy: Any
    cache: Any | None
    screenshot_dir: Path | None

    @classmethod
    def build(cls, cfg: MarkitaiConfig, out_dir: Path) -> UrlJobContext:
        """Build the shared URL context for one job."""
        from markitai.fetch import FetchStrategy, get_fetch_cache
        from markitai.utils.paths import ensure_screenshots_dir

        cache = None
        if cfg.cache.enabled:
            cache_dir = Path(cfg.cache.global_dir).expanduser()
            cache = get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)
        screenshot_dir = (
            ensure_screenshots_dir(out_dir) if cfg.screenshot.enabled else None
        )
        return cls(
            strategy=FetchStrategy(cfg.fetch.strategy),
            cache=cache,
            screenshot_dir=screenshot_dir,
        )


async def process_url_item(
    url: str,
    cfg: MarkitaiConfig,
    out_dir: Path,
    shared_processor: LLMProcessor | None,
    url_ctx: UrlJobContext,
    output_name: str | None = None,
) -> ProcessResult:
    """Convert one URL via the shared cascade, mapping to a ProcessResult.

    Thin serve wrapper over ``workflow.url.convert_url_cascade`` — the
    cascade owns fetch/images/LLM/frontmatter/profile; this wrapper owns
    job-facing concerns: error mapping and screenshot counting (a failed
    LLM enhancement follows ``llm.on_failure`` inside the cascade).
    ``output_name`` is the per-job pre-assigned
    unique output filename (URLs whose derived filenames collide would
    otherwise clobber each other's ``.llm.md`` in LLM mode).
    """
    from markitai.batch import ProcessResult
    from markitai.fetch import FetchError, JinaRateLimitError
    from markitai.utils.text import format_error_message
    from markitai.workflow.url import convert_url_cascade

    try:
        result = await convert_url_cascade(
            url,
            cfg,
            out_dir,
            processor=shared_processor,
            cache=url_ctx.cache,
            screenshot_dir=url_ctx.screenshot_dir,
            output_name=output_name,
        )
    except JinaRateLimitError:
        return ProcessResult(
            success=False, error="Jina Reader rate limit exceeded (20 RPM)"
        )
    except FetchError as e:
        return ProcessResult(success=False, error=format_error_message(e))
    except Exception as e:
        # ConversionError (no content) and anything unexpected.
        return ProcessResult(success=False, error=format_error_message(e))

    # A failed LLM enhancement follows llm.on_failure, as for files: under
    # "fail" the cascade raised (ConversionError, caught above); under
    # "fallback" the base .md is the output and the warning reached the
    # item's notices.

    if result.skipped:
        return ProcessResult(
            success=True,
            output_path=str(result.skip_target),
            error="skipped (exists)",
        )

    screenshots = (
        len(result.screenshot_tiles)
        if result.screenshot_tiles
        else (1 if result.screenshot_path else 0)
    )
    if cfg.screenshot.enabled and not screenshots:
        from markitai.notices import user_notice
        from markitai.utils.url_redaction import redact_url

        # The fetch layer logged why; the job item needs to say it happened.
        user_notice(
            "[URL] Screenshot not captured for {}; the page was converted without it",
            redact_url(url),
        )
    final_output = result.llm_output_path or result.output_path
    return ProcessResult(
        success=True,
        output_path=str(final_output) if final_output is not None else None,
        screenshots=screenshots,
        cost_usd=result.cost_usd,
        llm_usage=result.llm_usage,
        llm_enhanced=result.llm_output_path is not None,
    )


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def _apply_result(job: Job, item: JobItem, result: ProcessResult) -> None:
    """Map a ProcessResult onto the item (skip strings are non-errors)."""
    item.cost_usd = result.cost_usd
    if result.output_path:
        try:
            item.output = (
                Path(result.output_path)
                .resolve()
                .relative_to(job.out_dir.resolve())
                .as_posix()
            )
        except ValueError:
            item.output = None
    if result.success:
        item.status = "done"
        is_skip = result.error is not None and result.error.startswith("skipped (")
        item.error = result.error if is_skip else None
        item.skipped = is_skip
        item.llm_enhanced = bool(not is_skip and item.output and result.llm_enhanced)
        if is_skip and result.error is not None:
            item.skip_reason = result.error.removeprefix("skipped (").removesuffix(")")
    else:
        item.status = "error"
        item.error = result.error or "unknown error"
        item.llm_enhanced = False


async def _run_item(
    registry: JobRegistry,
    job: Job,
    item: JobItem,
    cfg: MarkitaiConfig,
    shared_processor: LLMProcessor | None,
    url_ctx: UrlJobContext | None,
    *,
    require_llm: bool = False,
) -> None:
    """Run one item, emitting running -> done/error events."""
    from markitai.notices import capture_task_notices

    item.status = "running"
    item.warnings = []
    registry.publish_item(job, item)
    start = time.perf_counter()
    try:
        # Per-item capture: items of one job convert concurrently, and each
        # gathered item runs in its own task (its own context copy).
        with capture_task_notices() as notices:
            try:
                if item.kind == "file":
                    result = await process_file_item(
                        Path(item.source), cfg, job.out_dir, shared_processor
                    )
                else:
                    assert url_ctx is not None
                    result = await process_url_item(
                        str(item.source),
                        cfg,
                        job.out_dir,
                        shared_processor,
                        url_ctx,
                        output_name=item.output_name,
                    )
            finally:
                item.warnings = list(notices)
    except asyncio.CancelledError:
        # Only registry.shutdown() cancels job tasks.
        item.status = "error"
        item.error = "cancelled (server shutdown)"
        item.duration_ms = int((time.perf_counter() - start) * 1000)
        item.finished_at = now_iso()
        registry.publish_item(job, item)
        raise
    except Exception as e:  # defensive: the item must reach a terminal state
        from markitai.batch import ProcessResult
        from markitai.utils.text import format_error_message

        result = ProcessResult(success=False, error=format_error_message(e))
    if require_llm and result.success:
        if not (result.output_path and result.llm_enhanced):
            result.success = False
            result.output_path = None
            result.error = "LLM enhancement did not produce an enhanced Markdown result"
    item.duration_ms = int((time.perf_counter() - start) * 1000)
    _apply_result(job, item, result)
    item.finished_at = now_iso()
    registry.publish_item(job, item)
    registry.publish_job(job)


async def run_job(
    registry: JobRegistry,
    job: Job,
    *,
    items: list[JobItem] | None = None,
    cfg: MarkitaiConfig | None = None,
    finalize: bool = True,
    require_llm: bool = False,
) -> None:
    """Run selected items of *job*, optionally leaving finalization to a queue."""
    targets = list(job.items if items is None else items)
    for item in targets:
        if item.options is None:
            item.options = dict(job.options)
    run_cfg = job.cfg if cfg is None else cfg
    if finalize:
        job.runners += 1
    try:
        shared_processor: LLMProcessor | None = None
        if run_cfg.llm.enabled and run_cfg.llm.model_list:
            # The LLM stack (LiteLLM and co.) takes most of a second to
            # import: on a worker thread, not the loop serving everyone else
            import importlib

            await asyncio.to_thread(importlib.import_module, "markitai.llm")
            from markitai.llm import LLMRuntime
            from markitai.workflow.helpers import create_llm_processor

            shared_processor = create_llm_processor(
                run_cfg,
                runtime=LLMRuntime(concurrency=run_cfg.llm.concurrency),
            )

        url_ctx = (
            UrlJobContext.build(run_cfg, job.out_dir)
            if any(i.kind == "url" for i in targets)
            else None
        )
        pdf_items = sum(
            1
            for i in targets
            if i.kind == "file"
            and isinstance(i.source, Path)
            and i.source.suffix.lower() == ".pdf"
        )
        if pdf_items > 1:
            # As a CLI batch does: the extraction workers load their models
            # while the job starts (on a thread: it imports the PDF stack)
            from markitai.converter.pdf_parallel import prestart

            await asyncio.to_thread(prestart)

        # File slots are handed on at the LLM step, as in a CLI batch: the
        # next file converts while this one waits on the model
        from contextlib import AbstractAsyncContextManager

        from markitai.workflow.slots import StagedSlots

        file_slots = StagedSlots(
            run_cfg.batch.concurrency,
            max(run_cfg.llm.concurrency, run_cfg.batch.concurrency),
        )
        url_semaphore = asyncio.Semaphore(max(1, run_cfg.batch.url_concurrency))

        async def run_gated(item: JobItem) -> None:
            gate: AbstractAsyncContextManager[object | None] = (
                file_slots.slot(llm=run_cfg.llm.enabled)
                if item.kind == "file"
                else url_semaphore
            )
            async with gate:
                from markitai.fetch_policy import public_network_only

                token = public_network_only.set(job.public_network_only)
                try:
                    await _run_item(
                        registry,
                        job,
                        item,
                        run_cfg,
                        shared_processor,
                        url_ctx,
                        require_llm=require_llm,
                    )
                finally:
                    public_network_only.reset(token)

        await asyncio.gather(*(run_gated(item) for item in targets))
    except asyncio.CancelledError:
        for item in targets:
            if item.status in ("queued", "running"):
                item.status = "error"
                item.error = "cancelled (server shutdown)"
                item.finished_at = now_iso()
        raise
    except Exception as e:  # defensive: the job must reach a terminal state
        logger.exception("[Serve] Job {} crashed: {}", job.job_id, e)
        for item in targets:
            if item.status in ("queued", "running"):
                item.status = "error"
                item.error = f"internal error: {e}"
                item.finished_at = now_iso()
                registry.publish_item(job, item)
    finally:
        if finalize:
            job.runners -= 1
            await finalize_if_idle(registry, job)


async def finalize_if_idle(registry: JobRegistry, job: Job) -> None:
    """Finalize once the initial run and retry drainer are both idle.

    ``retry_pending`` is added to synchronously by the retry endpoint before
    its drainer task is even scheduled, so this guard also covers the gap
    between queueing a retry and the drainer starting.
    """
    if _job_is_idle(job):
        await finalize_job(registry, job)


def _job_is_idle(job: Job) -> bool:
    return job.runners == 0 and job.retry_queue.empty() and not job.retry_pending


async def finalize_job(registry: JobRegistry, job: Job) -> None:
    """Persist and publish a job after its initial run or retry queue drains."""
    finished_at = now_iso()
    generation = job.generation
    try:
        # rglob the out dir off-thread WHILE the job is still "running" — a
        # job with hundreds of files (or a slow disk) must not block the loop
        # and stall other jobs' SSE progress.
        size: int | None = await asyncio.to_thread(job_dir_size, job.job_dir)
    except asyncio.CancelledError:
        # graceful shutdown: persist synchronously so the job still rehydrates
        if generation != job.generation or not _job_is_idle(job):
            raise
        job.finished_at = finished_at
        job.status = "done"
        try:
            write_job_meta(job)
        except OSError:
            pass
        raise
    except OSError:
        size = None
    if generation != job.generation or not _job_is_idle(job) or job.status == "done":
        return
    # No await past this point: the observable status flips to "done" and
    # meta.json lands together, so the job is never seen terminal without it.
    job.finished_at = finished_at
    job.dir_size_bytes = size
    job.status = "done"
    try:
        write_job_meta(job, refresh_size=False)
    except OSError as e:  # history is best effort; the job itself succeeded
        logger.warning(
            "[Serve] Failed to write meta.json for job {}: {}", job.job_id, e
        )
    registry.publish_job(job)
    logger.info(
        "[Serve] Job {} finished: {} done, {} failed, {} total",
        job.job_id,
        job.done_count,
        job.failed_count,
        len(job.items),
    )


def _restore_prior_result(job: Job, item: JobItem, work: RetryWork) -> None:
    """Roll a failed rerun back to the item's previous successful result.

    Enhance/retry reset the row and reuse the job's out_dir with
    ``on_conflict=overwrite``, so the base ``.md`` is rewritten in place and
    stays valid; restoring the reference keeps the working result downloadable
    instead of leaving a done row downgraded to an output-less error.
    """
    item.status = "done"
    item.error = None
    item.output = work.prior_output
    item.cost_usd = work.prior_cost_usd
    item.duration_ms = work.prior_duration_ms
    item.finished_at = work.prior_finished_at
    item.operation = work.prior_operation
    item.llm_enhanced = work.prior_llm_enhanced
    item.skipped = False
    item.skip_reason = None
    item.warnings = list(work.prior_warnings)


def item_base_name(item: JobItem) -> str | None:
    """The item's output base name: its output without the markdown suffix.

    Taken from what the item is known to write, not reverse-engineered from
    its current output: ``split_output_name`` cannot tell the base output of
    an upload named ``notes.llm`` (``notes.llm.md``) from the enhanced output
    of a sibling upload ``notes`` (also ``notes.llm.md``). ``output_name`` is
    the pinned base ``.md`` name (URL items, CLI history); an uploaded file
    writes ``derive_output_name(name)``. Only an output that matches neither
    variant of that base (or an item with no known base) falls back to
    stripping its suffix.
    """
    from markitai.serve.artifacts import split_output_name
    from markitai.utils.paths import derive_output_name

    base: str | None = None
    if item.output_name:
        base = Path(item.output_name).name.removesuffix(".md")
    elif item.kind == "file" and item.name:
        base = derive_output_name(Path(item.name).name).removesuffix(".md")
    if item.output:
        output_name = Path(item.output).name
        if base is None or output_name not in (f"{base}.md", f"{base}.llm.md"):
            return split_output_name(output_name)
    return base or None


def _sibling_markdown(job: Job, item: JobItem) -> set[Path]:
    """Markdown outputs other items of *job* claim (``serve.artifacts`` rule)."""
    from markitai.serve.artifacts import markdown_pair

    out_dir = job.out_dir.resolve()
    claimed: set[Path] = set()
    for sibling in job.items:
        if sibling is item:
            continue
        base_name = item_base_name(sibling)
        if base_name is not None:
            claimed.update(p.resolve() for p in markdown_pair(out_dir, base_name))
    return claimed


def _snapshot_outputs(
    job: Job, item: JobItem, work: RetryWork
) -> dict[Path, str | None]:
    """Capture the item's ``.md``/``.llm.md`` pair before a rerun touches it.

    A rerun writes into the job's own out_dir with ``on_conflict=overwrite``,
    so the snapshot is what makes the rerun provisional: its files only
    stand if it produces a real result, otherwise ``_restore_outputs`` puts
    the previous pair back verbatim (``None`` marks a file that did
    not exist and must not be left behind).

    The pair comes from the item's known base name (``item_base_name``),
    never from stripping ``prior_output``'s suffix, which is ambiguous.
    """
    if work.prior_output is None:
        return {}
    stem = item_base_name(item)
    if stem is None:
        return {}
    out_dir = job.out_dir.resolve()
    snapshot: dict[Path, str | None] = {}
    for name in (f"{stem}.md", f"{stem}.llm.md"):
        path = (out_dir / name).resolve()
        if not path.is_relative_to(out_dir):
            continue
        try:
            snapshot[path] = (
                path.read_text(encoding="utf-8") if path.is_file() else None
            )
        except (OSError, UnicodeDecodeError):
            continue
    return snapshot


def _restore_outputs(
    snapshot: dict[Path, str | None], keep: set[Path] | frozenset[Path] = frozenset()
) -> None:
    """Put back the files captured by ``_snapshot_outputs`` (best effort).

    A file that did not exist before the rerun is removed again, unless it
    is in *keep* — markdown a sibling item claims (two uploads can map to
    one name: ``notes`` enhanced and ``notes.llm`` both write
    ``notes.llm.md``), which a rollback must never delete.
    """
    from markitai.security import atomic_write_text

    for path, content in snapshot.items():
        try:
            if content is None:
                if path not in keep:
                    path.unlink(missing_ok=True)
            else:
                atomic_write_text(path, content)
        except OSError as e:
            logger.warning("[Serve] Could not restore {}: {}", path, e)


def _roll_back_rerun(
    registry: JobRegistry,
    job: Job,
    item: JobItem,
    work: RetryWork,
    snapshot: dict[Path, str | None],
) -> None:
    """Undo a rerun that did not stand: previous files, then previous row."""
    _restore_outputs(snapshot, _sibling_markdown(job, item))
    _restore_prior_result(job, item, work)
    registry.publish_item(job, item)


def _prune_stale_variant(job: Job, item: JobItem) -> None:
    """Delete the opposite-variant sibling a successful rerun made stale.

    A non-LLM retry of a previously enhanced item leaves its ``.llm.md`` on
    disk; without this the result endpoint (and diff) could still surface the
    outdated enhanced markdown.
    """
    if item.output is None or item.llm_enhanced:
        return
    if not item.output.endswith(".md") or item.output.endswith(".llm.md"):
        return
    stale = (job.out_dir / f"{item.output.removesuffix('.md')}.llm.md").resolve()
    if stale.is_relative_to(job.out_dir.resolve()):
        stale.unlink(missing_ok=True)


async def run_retry_queue(registry: JobRegistry, job: Job) -> None:
    """Run queued item retries serially and keep their original ledger rows."""
    job.runners += 1
    cancelled = False
    try:
        while not job.retry_queue.empty():
            work = job.retry_queue.get_nowait()
            item = job.get_item(work.item_id)
            snapshot: dict[Path, str | None] = {}
            try:
                if item is not None:
                    snapshot = _snapshot_outputs(job, item, work)
                    await run_job(
                        registry,
                        job,
                        items=[item],
                        cfg=work.cfg,
                        finalize=False,
                        require_llm=work.operation == "enhance",
                    )
                    if item.status == "error" and work.prior_output is not None:
                        # A failed rerun (an enhance whose LLM failed
                        # included) leaves the previous files and row intact
                        _roll_back_rerun(registry, job, item, work, snapshot)
                    elif item.status == "done":
                        _prune_stale_variant(job, item)
            except asyncio.CancelledError:
                cancelled = True
                if item is not None and work.prior_output is not None:
                    # Shut down mid-rerun: the interrupted rerun never
                    # produced a result, so the done row and its files go
                    # back rather than being persisted as a cancelled error.
                    _roll_back_rerun(registry, job, item, work, snapshot)
                raise
            finally:
                job.retry_pending.discard(work.item_id)
                job.retry_queue.task_done()
    finally:
        if cancelled:
            while True:
                try:
                    work = job.retry_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                item = job.get_item(work.item_id)
                if item is not None and item.status == "queued":
                    if work.prior_output is not None:
                        # Never started: its files are untouched, only the
                        # row was reset when the rerun was queued.
                        _restore_prior_result(job, item, work)
                    else:
                        item.status = "error"
                        item.error = "cancelled (server shutdown)"
                        item.finished_at = now_iso()
                    registry.publish_item(job, item)
                job.retry_pending.discard(work.item_id)
                job.retry_queue.task_done()
        job.runners -= 1
        await finalize_if_idle(registry, job)

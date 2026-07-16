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

if TYPE_CHECKING:
    from markitai.batch import ProcessResult
    from markitai.config import MarkitaiConfig
    from markitai.llm import LLMProcessor

JOB_TTL_HOURS = 7 * 24.0  # conversion history is kept for 7 days
META_FILENAME = "meta.json"


def _now_iso() -> str:
    """Browser-portable RFC 3339 timestamp with millisecond precision."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


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

    def to_payload(self) -> dict[str, Any]:
        """Return the item event payload (contract: ``event: item``)."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "finished_at": self.finished_at,
            "cost_usd": self.cost_usd,
            "llm_enhanced": self.llm_enhanced,
            "operation": self.operation,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass(slots=True)
class RetryWork:
    """One in-place item retry waiting on a job's serial worker."""

    item_id: str
    cfg: MarkitaiConfig
    operation: str = "retry"


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
    task: asyncio.Task[None] | None = None
    subscribers: list[asyncio.Queue[tuple[str, dict[str, Any]]]] = field(
        default_factory=list
    )
    archive_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
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

    def create_job(self, options: dict[str, Any], cfg: MarkitaiConfig) -> Job:
        """Create a job with fresh uploads/out directories on disk."""
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / job_id
        job = Job(
            job_id=job_id,
            job_dir=job_dir,
            created_at=_now_iso(),
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

    async def shutdown(self) -> None:
        """Cancel all still-running job tasks (server shutdown)."""
        tasks = [
            job.task
            for job in self.jobs.values()
            if job.task is not None and not job.task.done()
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
    cutoff = time.time() - ttl_hours * 3600
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


def write_job_meta(job: Job) -> None:
    """Persist the job's terminal snapshot to ``<job_dir>/meta.json``.

    Written once when the job reaches its terminal state; startup rehydrate
    reads it back to register the job as archived history.
    """
    from markitai.security import atomic_write_json

    atomic_write_json(
        job.job_dir / META_FILENAME,
        {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "status": job.status,
            "options": job.options,
            "items": [item.to_payload() for item in job.items],
        },
    )


def _item_from_payload(
    raw: dict[str, Any], index: int, fallback_finished_at: str | None = None
) -> JobItem:
    """Rebuild a JobItem from its meta.json payload dict."""
    return JobItem(
        item_id=str(raw.get("item_id") or f"i{index}"),
        name=str(raw.get("name") or ""),
        kind=str(raw.get("kind") or "file"),
        source=None,  # original upload path / URL is not needed for archives
        status=str(raw.get("status") or "done"),
        error=raw.get("error"),
        output=raw.get("output"),
        duration_ms=raw.get("duration_ms"),
        finished_at=raw.get("finished_at") or fallback_finished_at,
        cost_usd=raw.get("cost_usd"),
        llm_enhanced=bool(
            raw.get("llm_enhanced", str(raw.get("output") or "").endswith(".llm.md"))
        ),
        operation=str(raw.get("operation") or "convert"),
        skipped=bool(raw.get("skipped", False)),
        skip_reason=raw.get("skip_reason"),
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
        normalized_options = {
            key: options.get(key) for key in ("preset", "llm", "ocr")
        }
        job = Job(
            job_id=entry.name,
            job_dir=entry,
            created_at=str(meta.get("created_at") or ""),
            options=normalized_options,
            cfg=cfg,
            items=[
                _item_from_payload(raw, index, meta.get("finished_at"))
                for index, raw in enumerate(raw_items, start=1)
                if isinstance(raw, dict)
            ],
            status="done",
            finished_at=meta.get("finished_at"),
        )
        registry.jobs[job.job_id] = job
        count += 1
    return count


def job_duration_ms(job: Job) -> int | None:
    """Wall-clock duration of a terminal job, when timestamps are valid."""
    if not job.created_at or not job.finished_at:
        return None
    try:
        created = datetime.fromisoformat(job.created_at)
        finished = datetime.fromisoformat(job.finished_at)
    except ValueError:
        return None
    try:
        return max(0, int((finished - created).total_seconds() * 1000))
    except TypeError:  # mixed offset-aware/naive timestamps in old or edited meta
        return None


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

    Thin serve-side counterpart of the CLI batch worker: same skip semantics
    (``skipped (...)`` error strings are non-errors), same ``.llm.md`` output
    selection when LLM is enabled.
    """
    from markitai.batch import ProcessResult
    from markitai.constants import MAX_DOCUMENT_SIZE
    from markitai.utils.paths import derive_output_name
    from markitai.utils.text import format_error_message
    from markitai.workflow.core import ConversionContext, convert_document_core

    try:
        ctx = ConversionContext(
            input_path=file_path,
            output_dir=out_dir,
            config=cfg,
            shared_processor=shared_processor,
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if not result.success:
            return ProcessResult(success=False, error=result.error)

        if result.skip_reason == "exists":
            skipped_output = out_dir / derive_output_name(file_path.name)
            return ProcessResult(
                success=True,
                output_path=str(skipped_output),
                error="skipped (exists)",
            )
        if result.skip_reason == "image_only":
            return ProcessResult(success=True, error="skipped (image_only)")

        if cfg.image.desc_enabled and ctx.image_analysis is not None:
            from markitai.workflow.helpers import write_images_json

            write_images_json(out_dir, [ctx.image_analysis])

        output_file = ctx.output_file
        if cfg.llm.enabled and output_file is not None:
            output_file = output_file.with_suffix(".llm.md")
        return ProcessResult(
            success=True,
            output_path=str(output_file) if output_file else None,
            images=ctx.embedded_images_count,
            screenshots=ctx.screenshots_count,
            cost_usd=ctx.llm_cost,
            llm_usage=ctx.llm_usage,
        )
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
    """Convert one URL: fetch -> localize images -> base .md -> LLM .llm.md.

    Minimal serve-side cascade following the documented programmatic recipe
    (``fetch_url`` + ``add_basic_frontmatter`` + ``LLMProcessor``). The CLI's
    vision/screenshot-only URL branches are intentionally not replicated.
    ``output_name`` is the per-job pre-assigned unique output filename (URLs
    whose derived filenames collide would otherwise clobber each other's
    ``.llm.md`` in LLM mode); falls back to ``url_to_filename(url)``.
    """
    from markitai import fetch as fetch_module
    from markitai.batch import ProcessResult
    from markitai.fetch import FetchError, JinaRateLimitError
    from markitai.security import atomic_write_text
    from markitai.utils.cli_helpers import url_to_filename
    from markitai.utils.output import resolve_output_path
    from markitai.utils.text import format_error_message
    from markitai.workflow.helpers import add_basic_frontmatter

    try:
        fetch_result = await fetch_module.fetch_url(
            url,
            url_ctx.strategy,
            cfg.fetch,
            cache=url_ctx.cache,
            skip_read_cache=cfg.cache.no_cache,
            screenshot=cfg.screenshot.enabled,
            screenshot_dir=url_ctx.screenshot_dir,
            screenshot_config=cfg.screenshot if cfg.screenshot.enabled else None,
        )
    except JinaRateLimitError:
        return ProcessResult(
            success=False, error="Jina Reader rate limit exceeded (20 RPM)"
        )
    except FetchError as e:
        return ProcessResult(success=False, error=format_error_message(e))

    markdown = fetch_result.content
    if not markdown.strip():
        return ProcessResult(success=False, error="No content extracted")

    filename = output_name or url_to_filename(url)

    # Remote images are inputs to LLM image analysis. Preset image flags can
    # remain set after an explicit --no-llm override; downloading in that case
    # adds no analysis value and can turn a fast fetch into minutes of waits.
    if cfg.llm.enabled and (cfg.image.alt_enabled or cfg.image.desc_enabled):
        from markitai.image import download_url_images

        download_result = await download_url_images(
            markdown=markdown,
            output_dir=out_dir,
            base_url=url,
            config=cfg.image,
            source_name=filename.removesuffix(".md"),
        )
        markdown = download_result.updated_markdown

    output_file = resolve_output_path(out_dir / filename, cfg.output.on_conflict)
    if output_file is None:
        return ProcessResult(
            success=True,
            output_path=str(out_dir / filename),
            error="skipped (exists)",
        )

    title = fetch_result.title
    extra_meta = fetch_result.metadata.get("source_frontmatter")
    screenshots = 1 if fetch_result.screenshot_path else 0

    cost_usd = 0.0
    llm_usage: dict[str, dict[str, Any]] = {}
    llm_written = False
    if cfg.llm.enabled and shared_processor is not None:
        try:
            if cfg.llm.pure:
                content = await shared_processor.clean_document_pure(markdown, url)
            else:
                cleaned, frontmatter = await shared_processor.process_document(
                    markdown,
                    url,
                    fetch_strategy=fetch_result.strategy_used,
                    extra_meta=extra_meta,
                    title=title,
                )
                content = shared_processor.format_llm_output(cleaned, frontmatter)
            atomic_write_text(output_file.with_suffix(".llm.md"), content)
            llm_written = True
        except Exception as e:
            logger.error(
                "[Serve] URL LLM processing failed for {}: {}",
                url,
                format_error_message(e),
            )
        finally:
            cost_usd = shared_processor.get_context_cost(url)
            llm_usage = shared_processor.get_context_usage(url)
            shared_processor.clear_context_usage(url)

    # Base .md: always without LLM; with LLM only for keep_base or as fallback.
    if not llm_written or cfg.llm.keep_base:
        base_content = add_basic_frontmatter(
            markdown,
            url,
            fetch_strategy=fetch_result.strategy_used,
            screenshot_path=fetch_result.screenshot_path,
            output_dir=out_dir,
            title=title,
            extra_meta=extra_meta,
        )
        atomic_write_text(output_file, base_content)

    final_output = output_file.with_suffix(".llm.md") if llm_written else output_file
    return ProcessResult(
        success=True,
        output_path=str(final_output),
        screenshots=screenshots,
        cost_usd=cost_usd,
        llm_usage=llm_usage,
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
        item.llm_enhanced = bool(
            not is_skip and item.output and item.output.endswith(".llm.md")
        )
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
    item.status = "running"
    registry.publish_item(job, item)
    start = time.perf_counter()
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
    except asyncio.CancelledError:
        item.status = "error"
        item.error = "cancelled"
        item.duration_ms = int((time.perf_counter() - start) * 1000)
        item.finished_at = _now_iso()
        registry.publish_item(job, item)
        raise
    except Exception as e:  # defensive: the item must reach a terminal state
        from markitai.batch import ProcessResult
        from markitai.utils.text import format_error_message

        result = ProcessResult(success=False, error=format_error_message(e))
    if require_llm and result.success:
        enhanced_output = bool(
            result.output_path and result.output_path.endswith(".llm.md")
        )
        if not enhanced_output:
            result.success = False
            result.output_path = None
            result.error = "LLM enhancement did not produce an enhanced Markdown result"
    item.duration_ms = int((time.perf_counter() - start) * 1000)
    _apply_result(job, item, result)
    item.finished_at = _now_iso()
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
    run_cfg = job.cfg if cfg is None else cfg
    try:
        shared_processor: LLMProcessor | None = None
        if run_cfg.llm.enabled and run_cfg.llm.model_list:
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

        file_semaphore = asyncio.Semaphore(max(1, run_cfg.batch.concurrency))
        url_semaphore = asyncio.Semaphore(max(1, run_cfg.batch.url_concurrency))

        async def run_gated(item: JobItem) -> None:
            semaphore = file_semaphore if item.kind == "file" else url_semaphore
            async with semaphore:
                await _run_item(
                    registry,
                    job,
                    item,
                    run_cfg,
                    shared_processor,
                    url_ctx,
                    require_llm=require_llm,
                )

        await asyncio.gather(*(run_gated(item) for item in targets))
    except asyncio.CancelledError:
        for item in targets:
            if item.status in ("queued", "running"):
                item.status = "error"
                item.error = "cancelled (server shutdown)"
                item.finished_at = _now_iso()
        raise
    except Exception as e:  # defensive: the job must reach a terminal state
        logger.exception("[Serve] Job {} crashed: {}", job.job_id, e)
        for item in targets:
            if item.status in ("queued", "running"):
                item.status = "error"
                item.error = f"internal error: {e}"
                item.finished_at = _now_iso()
                registry.publish_item(job, item)
    finally:
        if finalize:
            finalize_job(registry, job)


def finalize_job(registry: JobRegistry, job: Job) -> None:
    """Persist and publish a job after its initial run or retry queue drains."""
    job.status = "done"
    job.finished_at = _now_iso()
    try:
        write_job_meta(job)
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


async def run_retry_queue(registry: JobRegistry, job: Job) -> None:
    """Run queued item retries serially and keep their original ledger rows."""
    cancelled = False
    try:
        while True:
            try:
                work = job.retry_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                item = job.get_item(work.item_id)
                if item is not None:
                    await run_job(
                        registry,
                        job,
                        items=[item],
                        cfg=work.cfg,
                        finalize=False,
                        require_llm=work.operation == "enhance",
                    )
            except asyncio.CancelledError:
                cancelled = True
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
                    item.status = "error"
                    item.error = "cancelled (server shutdown)"
                    item.finished_at = _now_iso()
                    registry.publish_item(job, item)
                job.retry_pending.discard(work.item_id)
                job.retry_queue.task_done()
        finalize_job(registry, job)

"""Batch processing module with resume capability."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.json_order import order_report, order_state
from markitai.security import atomic_write_json
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from markitai.config import BatchConfig
    from markitai.workflow.single import ImageAnalysisResult


class FileStatus(str, Enum):
    """Status of a file in batch processing.

    State transitions:
        PENDING -> IN_PROGRESS -> COMPLETED
                               -> FAILED

    On resume: IN_PROGRESS files are treated as FAILED (re-processed).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileState:
    """State of a single file in batch processing.

    Attributes:
        path: Relative path to source file from input_dir
        status: Current processing status
        output: Relative path to output .md file from output_dir
        error: Error message if status is FAILED
        started_at: ISO timestamp when processing started
        completed_at: ISO timestamp when processing completed
        duration: Total processing time in seconds
        images: Count of embedded images extracted from document content
        screenshots: Count of page/slide screenshots rendered for OCR/LLM
        cost_usd: Total LLM API cost for this file
        llm_usage: Per-model usage stats {model: {requests, input_tokens, output_tokens, cost_usd}}
        cache_hit: Whether LLM results were served from cache (no API calls made)
    """

    path: str
    status: FileStatus = FileStatus.PENDING
    output: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    images: int = 0
    screenshots: int = 0
    cost_usd: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hit: bool = False


@dataclass
class UrlState:
    """State of a single URL in batch processing.

    Attributes:
        url: The URL being processed
        source_file: Path to the .urls file containing this URL
        status: Current processing status
        output: Relative path to output .md file from output_dir
        error: Error message if status is FAILED
        fetch_strategy: The fetch strategy that was used (static/browser/jina)
        images: Count of images downloaded from the URL
        started_at: ISO timestamp when processing started
        completed_at: ISO timestamp when processing completed
        duration: Total processing time in seconds
        cost_usd: Total LLM API cost for this URL
        llm_usage: Per-model usage stats {model: {requests, input_tokens, output_tokens, cost_usd}}
        cache_hit: Whether LLM results were served from cache (no API calls made)
    """

    url: str
    source_file: str
    status: FileStatus = FileStatus.PENDING
    output: str | None = None
    error: str | None = None
    fetch_strategy: str | None = None
    images: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    cost_usd: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hit: bool = False


@dataclass
class BatchState:
    """State of batch processing for resume capability."""

    version: str = "1.0"
    started_at: str = ""
    updated_at: str = ""
    input_dir: str = ""
    output_dir: str = ""
    log_file: str | None = None  # Path to log file for this run
    options: dict = field(default_factory=dict)
    files: dict[str, FileState] = field(default_factory=dict)
    urls: dict[str, UrlState] = field(default_factory=dict)  # key: URL string
    url_sources: list[str] = field(default_factory=list)  # .urls file paths

    @property
    def total(self) -> int:
        """Total number of files."""
        return len(self.files)

    @property
    def total_urls(self) -> int:
        """Total number of URLs."""
        return len(self.urls)

    @property
    def completed_count(self) -> int:
        """Number of completed files."""
        return sum(1 for f in self.files.values() if f.status == FileStatus.COMPLETED)

    @property
    def completed_urls_count(self) -> int:
        """Number of completed URLs."""
        return sum(1 for u in self.urls.values() if u.status == FileStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Number of failed files."""
        return sum(1 for f in self.files.values() if f.status == FileStatus.FAILED)

    @property
    def failed_urls_count(self) -> int:
        """Number of failed URLs."""
        return sum(1 for u in self.urls.values() if u.status == FileStatus.FAILED)

    @property
    def pending_count(self) -> int:
        """Number of pending files."""
        return sum(
            1
            for f in self.files.values()
            if f.status in (FileStatus.PENDING, FileStatus.FAILED)
        )

    @property
    def pending_urls_count(self) -> int:
        """Number of pending URLs."""
        return sum(
            1
            for u in self.urls.values()
            if u.status in (FileStatus.PENDING, FileStatus.FAILED)
        )

    def get_pending_files(self) -> list[Path]:
        """Get list of files that need processing."""
        return [
            Path(f.path)
            for f in self.files.values()
            if f.status in (FileStatus.PENDING, FileStatus.FAILED)
        ]

    def get_pending_urls(self) -> list[str]:
        """Get list of URLs that need processing."""
        return [
            u.url
            for u in self.urls.values()
            if u.status in (FileStatus.PENDING, FileStatus.FAILED)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Note: input_dir/output_dir are stored in options with absolute paths.
        Files keys are stored as relative paths (relative to input_dir).
        """
        # Convert log_file to absolute path if it exists
        log_file_abs = None
        if self.log_file:
            log_path = Path(self.log_file)
            log_file_abs = (
                str(log_path.resolve()) if log_path.exists() else self.log_file
            )

        # Convert files keys to relative paths (relative to input_dir)
        input_dir_path = Path(self.input_dir).resolve()
        files_dict = {}
        for path, state in self.files.items():
            file_path = Path(path).resolve()
            try:
                rel_path = str(file_path.relative_to(input_dir_path))
            except ValueError:
                # File is not under input_dir, use filename only
                rel_path = file_path.name
            files_dict[rel_path] = {
                "status": state.status.value,
                "output": state.output,
                "error": state.error,
                "started_at": state.started_at,
                "completed_at": state.completed_at,
                "duration": state.duration,
                "images": state.images,
                "screenshots": state.screenshots,
                "cost_usd": state.cost_usd,
                "llm_usage": state.llm_usage,
                "cache_hit": state.cache_hit,
            }

        # Convert URLs to dict
        urls_dict = {}
        for url, state in self.urls.items():
            urls_dict[url] = {
                "source_file": state.source_file,
                "status": state.status.value,
                "output": state.output,
                "error": state.error,
                "fetch_strategy": state.fetch_strategy,
                "images": state.images,
                "started_at": state.started_at,
                "completed_at": state.completed_at,
                "duration": state.duration,
                "cost_usd": state.cost_usd,
                "llm_usage": state.llm_usage,
                "cache_hit": state.cache_hit,
            }

        return {
            "version": self.version,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "log_file": log_file_abs,
            "options": self.options,
            "documents": files_dict,
            "urls": urls_dict,
            "url_sources": self.url_sources,
        }

    def to_minimal_dict(self) -> dict[str, Any]:
        """Convert to minimal dictionary for state file (resume capability).

        Only includes fields necessary for determining what needs to be reprocessed:
        - version: For compatibility checking
        - options: input_dir/output_dir needed to resolve paths
        - documents: status + output (completed) or error (failed)
        - urls: status + output + source_file
        """
        # Convert files keys to relative paths (relative to input_dir)
        input_dir_path = Path(self.input_dir).resolve()
        files_dict = {}
        for path, state in self.files.items():
            file_path = Path(path).resolve()
            try:
                rel_path = str(file_path.relative_to(input_dir_path))
            except ValueError:
                rel_path = file_path.name

            # Minimal state: only what's needed for resume
            entry: dict[str, Any] = {"status": state.status.value}
            if state.status == FileStatus.COMPLETED and state.output:
                entry["output"] = state.output
            elif state.status == FileStatus.FAILED and state.error:
                entry["error"] = state.error
            files_dict[rel_path] = entry

        # Convert URLs to minimal dict
        urls_dict = {}
        for url, state in self.urls.items():
            entry: dict[str, Any] = {
                "status": state.status.value,
                "source_file": state.source_file,
            }
            if state.status == FileStatus.COMPLETED and state.output:
                entry["output"] = state.output
            elif state.status == FileStatus.FAILED and state.error:
                entry["error"] = state.error
            urls_dict[url] = entry

        return {
            "version": self.version,
            "options": self.options,
            "documents": files_dict,
            "urls": urls_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchState:
        """Create from dictionary."""
        options = data.get("options", {})

        # Get input_dir/output_dir from options
        input_dir = options.get("input_dir", "")
        output_dir = options.get("output_dir", "")

        state = cls(
            version=data.get("version", "1.0"),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=data.get("log_file"),
            options=options,
            url_sources=data.get("url_sources", []),
        )

        documents_data = data.get("documents", {})

        # Reconstruct absolute file paths from relative paths
        input_dir_path = Path(input_dir) if input_dir else Path(".")
        for path, file_data in documents_data.items():
            # If path is relative, make it absolute relative to input_dir
            file_path = Path(path)
            if not file_path.is_absolute():
                abs_path = str(input_dir_path / path)
            else:
                abs_path = path

            state.files[abs_path] = FileState(
                path=abs_path,
                status=FileStatus(file_data.get("status", "pending")),
                output=file_data.get("output"),
                error=file_data.get("error"),
                started_at=file_data.get("started_at"),
                completed_at=file_data.get("completed_at"),
                duration=file_data.get("duration"),
                images=file_data.get("images", 0),
                screenshots=file_data.get("screenshots", 0),
                cost_usd=file_data.get("cost_usd", 0.0),
                llm_usage=file_data.get("llm_usage", {}),
                cache_hit=file_data.get("cache_hit", False),
            )

        # Reconstruct URL states
        for url, url_data in data.get("urls", {}).items():
            state.urls[url] = UrlState(
                url=url,
                source_file=url_data.get("source_file", ""),
                status=FileStatus(url_data.get("status", "pending")),
                output=url_data.get("output"),
                error=url_data.get("error"),
                fetch_strategy=url_data.get("fetch_strategy"),
                images=url_data.get("images", 0),
                started_at=url_data.get("started_at"),
                completed_at=url_data.get("completed_at"),
                duration=url_data.get("duration"),
                cost_usd=url_data.get("cost_usd", 0.0),
                llm_usage=url_data.get("llm_usage", {}),
                cache_hit=url_data.get("cache_hit", False),
            )

        return state


@dataclass
class ProcessResult:
    """Result of processing a single file.

    Attributes:
        success: Whether processing completed without errors
        output_path: Path to generated .md file (None if failed)
        error: Error message if success is False
        images: Count of embedded images extracted from document
        screenshots: Count of page/slide screenshots for OCR/LLM
        cost_usd: Total LLM API cost for this file
        llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}
        image_analysis_result: Aggregated image analysis for JSON output (None if disabled)
        cache_hit: Whether LLM results were served entirely from cache
    """

    success: bool
    output_path: str | None = None
    error: str | None = None
    images: int = 0
    screenshots: int = 0
    cost_usd: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_analysis_result: ImageAnalysisResult | None = None
    cache_hit: bool = False


# Type alias for process function
ProcessFunc = Callable[[Path], Coroutine[Any, Any, ProcessResult]]


class BatchProcessor:
    """Batch processor with concurrent execution and progress display."""

    def __init__(
        self,
        config: BatchConfig,
        output_dir: Path,
        input_path: Path | None = None,
        log_file: Path | str | None = None,
        on_conflict: str = "rename",
        task_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
            output_dir: Output directory
            input_path: Input file or directory (used for report file naming)
            log_file: Path to the log file for this run
            on_conflict: Conflict resolution strategy ("skip", "overwrite", "rename")
            task_options: Task options dict (used for computing task hash)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.input_path = Path(input_path) if input_path else None
        self.log_file = str(log_file) if log_file else None
        self.on_conflict = on_conflict
        self.task_options = task_options or {}
        self.task_hash = self._compute_task_hash()
        self.state_file = self._get_state_file_path()
        self.report_file = self._get_report_file_path()
        self.state: BatchState | None = None
        self.console = get_console()
        # Collect image analysis results for JSON aggregation
        self.image_analysis_results: list[ImageAnalysisResult] = []

        # Optimization: Lock for state saving to prevent IO congestion
        import threading

        self._save_lock = threading.Lock()

        # Live display state (managed by start_live_display/stop_live_display)
        self._live: Live | None = None
        self._console_handler_id: int | None = None
        self._verbose: bool = False
        self._progress: Progress | None = None
        self._overall_task_id: TaskID | None = None
        self._url_task_id: TaskID | None = None
        self._total_urls: int = 0
        self._total_files: int = 0
        self._completed_urls: int = 0
        self._completed_files: int = 0

    def _compute_task_hash(self) -> str:
        """Compute hash from task input parameters.

        Hash is based on:
        - input_path (resolved)
        - output_dir (resolved)
        - key task options (llm_enabled, ocr_enabled, etc.)

        This ensures different parameter combinations produce different hashes,
        so resuming with different options creates a new state file.
        """
        import hashlib

        # Extract key options that affect output (exclude paths, they're added separately)
        key_options = {
            k: v
            for k, v in self.task_options.items()
            if k
            in (
                "llm_enabled",
                "ocr_enabled",
                "screenshot_enabled",
                "image_alt_enabled",
                "image_desc_enabled",
            )
        }

        hash_params = {
            "input": str(self.input_path.resolve()) if self.input_path else "",
            "output": str(self.output_dir.resolve()),
            "options": key_options,
        }
        hash_str = json.dumps(hash_params, sort_keys=True)
        return hashlib.md5(hash_str.encode(), usedforsecurity=False).hexdigest()[:6]

    def _get_state_file_path(self) -> Path:
        """Generate state file path for resume capability.

        Format: states/markitai.<hash>.state.json
        """
        states_dir = self.output_dir / "states"
        return states_dir / f"markitai.{self.task_hash}.state.json"

    def _get_report_file_path(self) -> Path:
        """Generate report file path based on task hash.

        Format: reports/markitai.<hash>.report.json
        Respects on_conflict strategy for rename.
        """
        reports_dir = self.output_dir / "reports"
        base_path = reports_dir / f"markitai.{self.task_hash}.report.json"

        if not base_path.exists():
            return base_path

        if self.on_conflict == "skip":
            return base_path  # Will be handled by caller
        elif self.on_conflict == "overwrite":
            return base_path
        else:  # rename
            seq = 2
            max_seq = 9999  # Safety limit to prevent infinite loop
            while seq <= max_seq:
                new_path = reports_dir / f"markitai.{self.task_hash}.v{seq}.report.json"
                if not new_path.exists():
                    return new_path
                seq += 1
            # Fallback: use timestamp if too many versions exist
            import time

            ts = int(time.time())
            return reports_dir / f"markitai.{self.task_hash}.{ts}.report.json"

    def start_live_display(
        self,
        verbose: bool = False,
        console_handler_id: int | None = None,
        total_files: int = 0,
        total_urls: int = 0,
    ) -> None:
        """Start Live display with progress bar.

        Call this before any processing (including pre-conversion).
        In verbose mode, logs are printed directly to console with │ prefix.

        Args:
            verbose: Whether to show verbose log output
            console_handler_id: Loguru console handler ID to disable
            total_files: Total number of files (for progress bar)
            total_urls: Total number of URLs to process
        """

        self._verbose = verbose
        self._console_handler_id = console_handler_id

        # Create progress display
        # Format: "Files  ━━━━━━━━━━━━━  5/7  (example.pdf)"
        self._progress = Progress(
            TextColumn("[progress.description]{task.description: <6}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[current]}"),
            console=self.console,
            transient=True,  # Clear progress when done
        )

        # Store totals for progress display
        self._total_urls = total_urls
        self._total_files = total_files
        self._completed_urls = 0
        self._completed_files = 0

        # Add URL progress task if there are URLs to process
        if total_urls > 0:
            self._url_task_id = self._progress.add_task(
                "URLs",
                total=total_urls,
                current="",
            )

        # Add file progress task (or overall if no URLs)
        self._overall_task_id = self._progress.add_task(
            "Files",
            total=total_files,
            current="",
        )

        # Disable console handler to avoid conflict with progress bar
        if console_handler_id is not None:
            try:
                logger.remove(console_handler_id)
            except ValueError:
                pass  # Handler already removed

        # Start Live display (progress bar only, no panel)
        self._live = Live(self._progress, console=self.console, refresh_per_second=4)
        self._live.start()

    def stop_live_display(self) -> None:
        """Stop Live display and restore console handler."""
        import sys

        # Stop Live display
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Re-add console handler (restore original state)
        if self._console_handler_id is not None:
            new_handler_id = logger.add(
                sys.stderr,
                level="DEBUG" if self._verbose else "INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            )
            self._restored_console_handler_id = new_handler_id
            self._console_handler_id = None

    def update_progress_total(self, total: int) -> None:
        """Update progress bar total after file discovery."""
        self._total_files = total
        if self._progress is not None and self._overall_task_id is not None:
            self._progress.update(self._overall_task_id, total=total)

    def advance_progress(self, clear_current: bool = True) -> None:
        """Advance progress bar by one.

        Args:
            clear_current: Whether to clear the current file field after advancing
        """
        if self._progress is not None and self._overall_task_id is not None:
            self._completed_files += 1
            self._progress.advance(self._overall_task_id)
            if clear_current:
                self._progress.update(self._overall_task_id, current="")

    def set_current_file(self, filename: str) -> None:
        """Set the current file being processed in progress display.

        Args:
            filename: Name of the file being processed (displayed in progress bar)
        """
        if self._progress is not None and self._overall_task_id is not None:
            self._progress.update(self._overall_task_id, current=f"({filename})")

    def update_url_status(self, url: str, completed: bool = False) -> None:
        """Update URL processing status in progress display.

        Args:
            url: The URL being processed (displayed in progress bar)
            completed: If True, advance the URL progress counter and clear current
        """
        from urllib.parse import urlparse

        if self._progress is not None and self._url_task_id is not None:
            if completed:
                self._completed_urls += 1
                self._progress.advance(self._url_task_id)
                # Clear current URL after completion
                self._progress.update(self._url_task_id, current="")
            else:
                # Show domain of current URL being processed
                domain = urlparse(url).netloc
                self._progress.update(self._url_task_id, current=f"({domain})")

    def finish_url_processing(self, completed: int, failed: int) -> None:
        """Mark URL processing as complete.

        Args:
            completed: Number of URLs successfully processed
            failed: Number of URLs that failed
        """
        if self._progress is not None and self._url_task_id is not None:
            # Final status already shows count from update_url_status
            pass

    def discover_files(
        self,
        input_path: Path,
        extensions: set[str],
    ) -> list[Path]:
        """
        Discover files to process.

        Args:
            input_path: Input file or directory
            extensions: Set of valid file extensions (e.g., {".docx", ".pdf"})

        Returns:
            List of file paths

        Raises:
            ValueError: If any discovered file is outside the input directory
        """
        from markitai.security import validate_path_within_base

        if input_path.is_file():
            return [input_path]

        input_resolved = input_path.resolve()
        files: list[Path] = []
        max_depth = max(0, self.config.scan_max_depth)
        max_files = max(1, self.config.scan_max_files)

        def should_include(path: Path) -> bool:
            try:
                validate_path_within_base(path, input_resolved)
            except ValueError:
                logger.warning(f"Skipping file outside input directory: {path}")
                return False
            return path.is_file()

        for ext in extensions:
            # Search both lowercase and uppercase variants (Linux glob is case-sensitive)
            ext_variants = [ext, ext.upper()]
            candidates = []

            for ext_variant in ext_variants:
                if max_depth == 0:
                    candidates.extend(input_path.glob(f"*{ext_variant}"))
                else:
                    # Use rglob for recursive search, then filter by depth
                    for f in input_path.rglob(f"*{ext_variant}"):
                        # Calculate relative depth
                        try:
                            rel_path = f.relative_to(input_path)
                            depth = len(rel_path.parts) - 1  # -1 for filename itself
                            if depth <= max_depth:
                                candidates.append(f)
                        except ValueError:
                            continue

            for f in candidates:
                if len(files) >= max_files:
                    logger.warning(
                        f"Reached scan_max_files={max_files}, stopping file discovery"
                    )
                    return sorted(set(files))
                if should_include(f):
                    files.append(f)

        return sorted(set(files))

    def load_state(self) -> BatchState | None:
        """Load state from state file if exists (for resume capability)."""
        from markitai.constants import MAX_STATE_FILE_SIZE
        from markitai.security import validate_file_size

        if not self.state_file.exists():
            return None

        try:
            # Validate file size to prevent DoS
            validate_file_size(self.state_file, MAX_STATE_FILE_SIZE)
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return BatchState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load state file for resume: {e}")
            return None

    def save_state(self, force: bool = False, log: bool = False) -> None:
        """Save current state to state file for resume capability.

        State file is saved to: states/markitai.<hash>.state.json

        Optimized with interval-based throttling:
        - Checks interval BEFORE serialization to avoid unnecessary work
        - Uses minimal serialization when possible
        - Uses thread lock to prevent concurrent disk writes

        Args:
            force: Force save even if interval hasn't passed
            log: Whether to log the save operation
        """
        if self.state is None:
            return

        now = datetime.now().astimezone()
        # Default to 5 seconds if not specified in config to prevent $O(N^2)$ IO
        interval = getattr(self.config, "state_flush_interval_seconds", 5) or 5

        # Check interval BEFORE any serialization work (optimization)
        if not force:
            last_saved = getattr(self, "_last_state_save", None)
            if last_saved and (now - last_saved).total_seconds() < interval:
                return  # Skip: interval not passed, no work done

        # Ensure only one thread is writing at a time
        if not self._save_lock.acquire(blocking=force):
            return  # Skip if another thread is already saving, unless forced

        try:
            self.state.updated_at = now.isoformat()

            # Build minimal state document (only what's needed for resume)
            state_data = self.state.to_minimal_dict()

            # Ensure states directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            atomic_write_json(self.state_file, state_data, order_func=order_state)
            self._last_state_save = now

            if log:
                logger.info(f"State file saved: {self.state_file.resolve()}")
        finally:
            self._save_lock.release()

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics for report."""
        if self.state is None:
            return {}

        # Calculate wall-clock duration
        wall_duration = 0.0
        if self.state.started_at and self.state.updated_at:
            try:
                start = datetime.fromisoformat(self.state.started_at)
                end = datetime.fromisoformat(self.state.updated_at)
                wall_duration = (end - start).total_seconds()
            except ValueError:
                wall_duration = 0.0

        # Calculate cumulative processing time (files + URLs)
        file_duration = sum(f.duration or 0 for f in self.state.files.values())
        url_duration = sum(u.duration or 0 for u in self.state.urls.values())
        processing_time = file_duration + url_duration

        # URL cache hits count
        url_cache_hits = sum(
            1
            for u in self.state.urls.values()
            if u.status == FileStatus.COMPLETED and u.cache_hit
        )

        return {
            "total_documents": self.state.total,
            "completed_documents": self.state.completed_count,
            "failed_documents": self.state.failed_count,
            "pending_documents": self.state.pending_count,
            "total_urls": self.state.total_urls,
            "completed_urls": self.state.completed_urls_count,
            "failed_urls": self.state.failed_urls_count,
            "pending_urls": self.state.pending_urls_count,
            "url_cache_hits": url_cache_hits,
            "url_sources": len(self.state.url_sources),
            "duration": wall_duration,
            "processing_time": processing_time,
        }

    def _compute_llm_usage(self) -> dict[str, Any]:
        """Compute aggregated LLM usage statistics for report."""
        if self.state is None:
            return {}

        # Aggregate LLM usage by model (from both files and URLs)
        models_usage: dict[str, dict[str, Any]] = {}

        # Aggregate from files
        for f in self.state.files.values():
            for model, usage in f.llm_usage.items():
                if model not in models_usage:
                    models_usage[model] = {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                models_usage[model]["requests"] += usage.get("requests", 0)
                models_usage[model]["input_tokens"] += usage.get("input_tokens", 0)
                models_usage[model]["output_tokens"] += usage.get("output_tokens", 0)
                models_usage[model]["cost_usd"] += usage.get("cost_usd", 0.0)

        # Aggregate from URLs
        for u in self.state.urls.values():
            for model, usage in u.llm_usage.items():
                if model not in models_usage:
                    models_usage[model] = {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                models_usage[model]["requests"] += usage.get("requests", 0)
                models_usage[model]["input_tokens"] += usage.get("input_tokens", 0)
                models_usage[model]["output_tokens"] += usage.get("output_tokens", 0)
                models_usage[model]["cost_usd"] += usage.get("cost_usd", 0.0)

        # Calculate totals (files + URLs)
        total_cost = sum(f.cost_usd for f in self.state.files.values()) + sum(
            u.cost_usd for u in self.state.urls.values()
        )
        input_tokens = sum(m["input_tokens"] for m in models_usage.values())
        output_tokens = sum(m["output_tokens"] for m in models_usage.values())
        requests = sum(m["requests"] for m in models_usage.values())

        return {
            "models": models_usage,
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": total_cost,
        }

    def init_state(
        self,
        input_dir: Path,
        files: list[Path],
        options: dict[str, Any],
        started_at: str | None = None,
    ) -> BatchState:
        """
        Initialize a new batch state.

        Args:
            input_dir: Input directory
            files: List of files to process
            options: Processing options (will be updated with absolute paths)
            started_at: ISO timestamp when processing started (defaults to now)

        Returns:
            New BatchState
        """
        # Resolve absolute paths
        abs_input_dir = str(input_dir.resolve())
        abs_output_dir = str(self.output_dir.resolve())
        abs_log_file = None
        if self.log_file:
            log_path = Path(self.log_file)
            abs_log_file = (
                str(log_path.resolve()) if log_path.exists() else self.log_file
            )

        # Update options with absolute paths
        options["input_dir"] = abs_input_dir
        options["output_dir"] = abs_output_dir

        now = datetime.now().astimezone().isoformat()
        state = BatchState(
            started_at=started_at or now,
            updated_at=now,
            input_dir=abs_input_dir,
            output_dir=abs_output_dir,
            log_file=abs_log_file,
            options=options,
        )

        for file_path in files:
            state.files[str(file_path)] = FileState(path=str(file_path))

        return state

    async def process_batch(
        self,
        files: list[Path],
        process_func: ProcessFunc,
        resume: bool = False,
        options: dict[str, Any] | None = None,
        verbose: bool = False,
        console_handler_id: int | None = None,
        started_at: str | None = None,
    ) -> BatchState:
        """
        Process files in batch with concurrency control.

        Args:
            files: List of files to process
            process_func: Async function to process each file
            resume: Whether to resume from previous state
            options: Task options to record in report
            verbose: Whether to show log panel during processing
            console_handler_id: Loguru console handler ID for temporary disable
                               (ignored if start_live_display was already called)
            started_at: ISO timestamp when processing started (for accurate duration)

        Returns:
            Final batch state
        """
        # Use provided started_at or default to now
        actual_started_at = started_at or datetime.now().astimezone().isoformat()

        # Initialize or load state
        if resume:
            self.state = self.load_state()
            if self.state:
                files = self.state.get_pending_files()
                logger.info(
                    f"Resuming batch: {self.state.completed_count} completed, "
                    f"{len(files)} remaining"
                )
                # Reset started_at for accurate duration calculation in this session
                self.state.started_at = actual_started_at

        if self.state is None:
            self.state = self.init_state(
                input_dir=files[0].parent if files else Path("."),
                files=files,
                options=options or {},
                started_at=actual_started_at,
            )
            self.save_state(force=True)

        if not files:
            logger.info("No files to process")
            self.save_state(force=True)
            return self.state

        # Preheat OCR engine if OCR is enabled to eliminate cold start delay
        if options and options.get("ocr_enabled"):
            try:
                from markitai.ocr import OCRProcessor

                OCRProcessor.preheat()
            except ImportError:
                logger.debug("OCR preheat skipped: RapidOCR not installed")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrency)

        # Check if Live display was already started by caller
        live_already_started = self._live is not None

        # Use existing progress or create new one
        if live_already_started and self._progress is not None:
            progress = self._progress
            overall_task = self._overall_task_id
            assert overall_task is not None  # Guaranteed when _progress is set
            # Update total in case it changed
            progress.update(overall_task, total=len(files))
        else:
            # Create progress display (legacy path for backwards compatibility)
            # Format: "Files  ━━━━━━━━━━━━━  5/7  (example.pdf)"
            progress = Progress(
                TextColumn("[progress.description]{task.description: <6}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[current]}"),
                console=self.console,
                transient=True,  # Clear progress when done
            )
            overall_task = progress.add_task(
                "Files",
                total=len(files),
                current="",
            )

        async def process_with_limit(file_path: Path) -> None:
            """Process a file with semaphore limit.

            State saving is performed outside the semaphore to avoid blocking
            concurrent file processing.
            """
            assert self.state is not None  # Guaranteed by _init_state() above
            file_key = str(file_path)
            file_state = self.state.files.get(file_key)

            if file_state is None:
                file_state = FileState(path=file_key)
                self.state.files[file_key] = file_state

            # Update state to in_progress
            file_state.status = FileStatus.IN_PROGRESS
            file_state.started_at = datetime.now().astimezone().isoformat()

            start_time = asyncio.get_event_loop().time()

            try:
                # Process file within semaphore
                async with semaphore:
                    # Show current file being processed
                    progress.update(overall_task, current=f"({file_path.name})")
                    result = await process_func(file_path)

                if result.success:
                    file_state.status = FileStatus.COMPLETED
                    file_state.output = result.output_path
                    file_state.images = result.images
                    file_state.screenshots = result.screenshots
                    file_state.cost_usd = result.cost_usd
                    file_state.llm_usage = result.llm_usage
                    file_state.cache_hit = result.cache_hit
                    # Collect image analysis result for JSON aggregation
                    if result.image_analysis_result is not None:
                        self.image_analysis_results.append(result.image_analysis_result)
                else:
                    file_state.status = FileStatus.FAILED
                    file_state.error = result.error

            except Exception as e:
                file_state.status = FileStatus.FAILED
                file_state.error = format_error_message(e)
                logger.error(
                    f"Failed to process {file_path.name}: {format_error_message(e)}"
                )

            finally:
                end_time = asyncio.get_event_loop().time()
                file_state.completed_at = datetime.now().astimezone().isoformat()
                file_state.duration = end_time - start_time

                # Update progress and clear current file
                progress.advance(overall_task)
                progress.update(overall_task, current="")

            # Save state outside semaphore (non-blocking, throttled)
            # Use asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(self.save_state)

        # If Live display was already started, just run the tasks without creating new Live
        if live_already_started:
            tasks = [process_with_limit(f) for f in files]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # No external Live display provided - create one here
            # Disable console handler to avoid conflict with progress bar
            if console_handler_id is not None:
                try:
                    logger.remove(console_handler_id)
                except ValueError:
                    pass  # Handler already removed

            try:
                # Progress bar only (verbose logs handled by loguru)
                with Live(progress, console=self.console, refresh_per_second=4):
                    tasks = [process_with_limit(f) for f in files]
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # Re-add console handler (restore original state)
                if console_handler_id is not None:
                    import sys

                    new_handler_id = logger.add(
                        sys.stderr,
                        level="DEBUG" if verbose else "INFO",
                        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
                    )
                    self._restored_console_handler_id = new_handler_id

        # Final save
        self.save_state(force=True)

        return self.state

    def generate_report(self) -> dict[str, Any]:
        """
        Generate final processing report.

        Returns:
            Report dictionary (same structure as saved report file)
        """
        if self.state is None:
            return {}

        report = self.state.to_dict()
        report["summary"] = self._compute_summary()
        report["llm_usage"] = self._compute_llm_usage()
        report["generated_at"] = datetime.now().astimezone().isoformat()

        return report

    def save_report(self) -> Path:
        """Finalize and save report to file.

        Report file is saved to: reports/markitai.<hash>.report.json
        Respects on_conflict strategy (skip/overwrite/rename).

        Returns:
            Path to the report file
        """
        # First, ensure state is saved for resume capability
        self.save_state(force=True, log=True)

        # Generate and save the report
        report = self.generate_report()

        # Ensure reports directory exists
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(self.report_file, report, order_func=order_report)
        logger.info(f"Report saved: {self.report_file.resolve()}")

        return self.report_file

    def print_summary(
        self,
        url_completed: int = 0,
        url_failed: int = 0,
        url_cache_hits: int = 0,
        url_sources: int = 0,
    ) -> None:
        """Print summary to console.

        This displays:
        1. Completion results (files/URLs completed with duration)
        2. Warnings for failed items
        3. Final summary with output location

        Args:
            url_completed: Number of URLs successfully processed
            url_failed: Number of URLs that failed
            url_cache_hits: Number of URLs that hit LLM cache
            url_sources: Number of .urls source files processed
        """
        if self.state is None:
            return

        # Helper function to format duration as human-readable (m:ss format)
        def format_duration(seconds: float) -> str:
            if seconds < 60:
                return f"0:{int(seconds):02d}"
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"

        # Calculate durations for files and URLs separately
        files_duration = sum(f.duration or 0 for f in self.state.files.values())
        urls_duration = sum(u.duration or 0 for u in self.state.urls.values())

        # Calculate wall-clock duration from started_at to updated_at
        wall_duration = 0.0
        if self.state.started_at and self.state.updated_at:
            try:
                start = datetime.fromisoformat(self.state.started_at)
                end = datetime.fromisoformat(self.state.updated_at)
                wall_duration = (end - start).total_seconds()
            except ValueError:
                # Fallback to sum of individual durations
                wall_duration = files_duration + urls_duration

        # LLM cost (from both files and URLs)
        total_cost = sum(f.cost_usd for f in self.state.files.values()) + sum(
            u.cost_usd for u in self.state.urls.values()
        )

        # Collect warnings (failed files and URLs)
        warnings: list[str] = []
        for f in self.state.files.values():
            if f.status == FileStatus.FAILED and f.error:
                warnings.append(f"{Path(f.path).name}: {f.error}")
        for u in self.state.urls.values():
            if u.status == FileStatus.FAILED and u.error:
                # Extract domain for display
                from urllib.parse import urlparse

                domain = urlparse(u.url).netloc
                warnings.append(f"{domain}: {u.error}")

        # Print completion display (title + results)
        self.console.print()
        ui.title("Converting", console=self.console)

        # Files result
        files_completed = self.state.completed_count
        if files_completed > 0:
            files_duration_str = format_duration(files_duration)
            self.console.print(
                f"  [green]{ui.MARK_SUCCESS}[/] {files_completed} files completed "
                f"({files_duration_str})"
            )

        # URLs result
        if url_completed > 0:
            urls_duration_str = format_duration(urls_duration)
            self.console.print(
                f"  [green]{ui.MARK_SUCCESS}[/] {url_completed} URLs completed "
                f"({urls_duration_str})"
            )

        # Warnings (failed items)
        if warnings:
            self.console.print()
            warning_count = len(warnings)
            if warning_count == 1:
                self.console.print(
                    f"  [yellow]{ui.MARK_WARNING}[/] 1 warning: {warnings[0]}"
                )
            else:
                self.console.print(
                    f"  [yellow]{ui.MARK_WARNING}[/] {warning_count} warnings:"
                )
                for warning in warnings:
                    self.console.print(f"    {ui.MARK_LINE} {warning}")

        # Final summary line
        self.console.print()
        cost_str = f", ${total_cost:.3f}" if total_cost > 0 else ""
        total_items = files_completed + url_completed
        total_str = f"{total_items} items" if total_items > 1 else f"{total_items} item"

        # Format wall duration for summary (human readable)
        if wall_duration < 60:
            wall_str = f"{wall_duration:.1f}s"
        else:
            mins = int(wall_duration // 60)
            secs = int(wall_duration % 60)
            wall_str = f"{mins}m {secs}s"

        self.console.print(f"  Done: {total_str} in {wall_str}{cost_str}")
        self.console.print(f"  Output: {self.output_dir}/")

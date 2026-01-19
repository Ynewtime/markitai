"""Batch processing module with resume capability."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from markit.security import atomic_write_json

if TYPE_CHECKING:
    from markit.config import BatchConfig
    from markit.workflow.single import ImageAnalysisResult


class FileStatus(str, Enum):
    """Status of a file in batch processing."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileState:
    """State of a single file in batch processing."""

    path: str
    status: FileStatus = FileStatus.PENDING
    output: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    images_extracted: int = 0  # Embedded images extracted from document content
    screenshots: int = 0  # Page/slide screenshots for OCR/LLM processing
    llm_cost_usd: float = 0.0
    # LLM usage per model: {"model_name": {"requests": N, "input_tokens": N, "output_tokens": N, "cost_usd": F}}
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)


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

    @property
    def total(self) -> int:
        """Total number of files."""
        return len(self.files)

    @property
    def completed_count(self) -> int:
        """Number of completed files."""
        return sum(1 for f in self.files.values() if f.status == FileStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Number of failed files."""
        return sum(1 for f in self.files.values() if f.status == FileStatus.FAILED)

    @property
    def pending_count(self) -> int:
        """Number of pending files."""
        return sum(
            1
            for f in self.files.values()
            if f.status in (FileStatus.PENDING, FileStatus.FAILED)
        )

    def get_pending_files(self) -> list[Path]:
        """Get list of files that need processing."""
        return [
            Path(f.path)
            for f in self.files.values()
            if f.status in (FileStatus.PENDING, FileStatus.FAILED)
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
                "duration_seconds": state.duration_seconds,
                "images_extracted": state.images_extracted,
                "screenshots": state.screenshots,
                "llm_cost_usd": state.llm_cost_usd,
                "llm_usage": state.llm_usage,
            }

        return {
            "version": self.version,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "log_file": log_file_abs,
            "options": self.options,
            "files": files_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchState:
        """Create from dictionary.

        Supports both old format (root-level input_dir/output_dir) and
        new format (input_dir/output_dir in options with absolute paths).
        """
        options = data.get("options", {})

        # Get input_dir/output_dir from options (new format) or root level (old format)
        input_dir = options.get("input_dir") or data.get("input_dir", "")
        output_dir = options.get("output_dir") or data.get("output_dir", "")

        state = cls(
            version=data.get("version", "1.0"),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=data.get("log_file"),
            options=options,
        )

        # Reconstruct absolute file paths from relative paths
        input_dir_path = Path(input_dir) if input_dir else Path(".")
        for path, file_data in data.get("files", {}).items():
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
                duration_seconds=file_data.get("duration_seconds"),
                images_extracted=file_data.get("images_extracted", 0),
                screenshots=file_data.get("screenshots", 0),
                llm_cost_usd=file_data.get("llm_cost_usd", 0.0),
                llm_usage=file_data.get("llm_usage", {}),
            )

        return state


@dataclass
class ProcessResult:
    """Result of processing a single file."""

    success: bool
    output_path: str | None = None
    error: str | None = None
    images_extracted: int = 0  # Embedded images extracted from document content
    screenshots: int = 0  # Page/slide screenshots for OCR/LLM processing
    llm_cost_usd: float = 0.0
    # LLM usage per model: {"model_name": {"requests": N, "input_tokens": N, "output_tokens": N, "cost_usd": F}}
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Image analysis result for JSON aggregation
    image_analysis_result: ImageAnalysisResult | None = None


# Type alias for process function
ProcessFunc = Callable[[Path], Coroutine[Any, Any, ProcessResult]]


class LogPanel:
    """Log panel for verbose mode, displays scrolling log messages."""

    def __init__(self, max_lines: int = 8):
        self.logs: deque[str] = deque(maxlen=max_lines)

    def add(self, message: str) -> None:
        """Add a log message to the panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"{timestamp} | {message}")

    def __rich__(self) -> Panel:
        """Render the log panel."""
        content = "\n".join(self.logs) if self.logs else "(waiting for logs...)"
        return Panel(content, title="Logs", border_style="dim")


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
        self.console = Console()
        # Collect image analysis results for JSON aggregation
        self.image_analysis_results: list[ImageAnalysisResult] = []

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
        return hashlib.md5(hash_str.encode()).hexdigest()[:6]

    def _get_state_file_path(self) -> Path:
        """Generate state file path for resume capability.

        Format: states/markit.<hash>.state.json
        """
        states_dir = self.output_dir / "states"
        return states_dir / f"markit.{self.task_hash}.state.json"

    def _get_report_file_path(self) -> Path:
        """Generate report file path based on task hash.

        Format: reports/markit.<hash>.report.json
        Respects on_conflict strategy for rename.
        """
        reports_dir = self.output_dir / "reports"
        base_path = reports_dir / f"markit.{self.task_hash}.report.json"

        if not base_path.exists():
            return base_path

        if self.on_conflict == "skip":
            return base_path  # Will be handled by caller
        elif self.on_conflict == "overwrite":
            return base_path
        else:  # rename
            seq = 2
            while True:
                new_path = reports_dir / f"markit.{self.task_hash}.v{seq}.report.json"
                if not new_path.exists():
                    return new_path
                seq += 1

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
        from markit.security import validate_path_within_base

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
        from markit.security import MAX_STATE_FILE_SIZE, validate_file_size

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

        State file is saved to: states/markit.<hash>.state.json

        Args:
            force: Force save even if interval hasn't passed
            log: Whether to log the save operation
        """
        if self.state is None:
            return

        now = datetime.now().astimezone()
        interval = getattr(self.config, "state_flush_interval_seconds", 0) or 0
        if not force and interval > 0:
            last_saved = getattr(self, "_last_state_save", None)
            if last_saved and (now - last_saved).total_seconds() < interval:
                return

        self.state.updated_at = now.isoformat()

        # Build state document (minimal, for resume)
        state_data = self.state.to_dict()

        # Ensure states directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(self.state_file, state_data)
        self._last_state_save = now

        if log:
            logger.info(f"State file saved: {self.state_file.resolve()}")

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

        # Calculate cumulative processing time
        cumulative_duration = sum(
            f.duration_seconds or 0 for f in self.state.files.values()
        )

        # Format duration
        hours, remainder = divmod(int(wall_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_human = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            "total": self.state.total,
            "completed": self.state.completed_count,
            "failed": self.state.failed_count,
            "pending": self.state.pending_count,
            "duration_seconds": wall_duration,
            "duration_human": duration_human,
            "cumulative_processing_seconds": cumulative_duration,
        }

    def _compute_llm_usage(self) -> dict[str, Any]:
        """Compute aggregated LLM usage statistics for report."""
        if self.state is None:
            return {}

        # Aggregate LLM usage by model
        models_usage: dict[str, dict[str, Any]] = {}
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

        # Calculate totals
        total_llm_cost = sum(f.llm_cost_usd for f in self.state.files.values())
        total_input_tokens = sum(m["input_tokens"] for m in models_usage.values())
        total_output_tokens = sum(m["output_tokens"] for m in models_usage.values())
        total_requests = sum(m["requests"] for m in models_usage.values())
        token_status = "estimated" if models_usage else "unknown"

        return {
            "models": models_usage,
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": total_llm_cost,
            "token_status": token_status,
        }

    def init_state(
        self,
        input_dir: Path,
        files: list[Path],
        options: dict,
    ) -> BatchState:
        """
        Initialize new batch state.

        Args:
            input_dir: Input directory
            files: List of files to process
            options: Processing options (will be updated with absolute paths)

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

        state = BatchState(
            started_at=datetime.now().astimezone().isoformat(),
            updated_at=datetime.now().astimezone().isoformat(),
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

        Returns:
            Final batch state
        """
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
                self.state.started_at = datetime.now().astimezone().isoformat()

        if self.state is None:
            self.state = self.init_state(
                input_dir=files[0].parent if files else Path("."),
                files=files,
                options=options or {},
            )
            self.save_state(force=True)

        if not files:
            logger.info("No files to process")
            self.save_state(force=True)
            return self.state

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrency)

        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[filename]:<30}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )

        overall_task = progress.add_task(
            "Overall",
            total=len(files),
            filename="[Overall Progress]",
        )

        # Create log panel for verbose mode
        log_panel: LogPanel | None = None
        panel_handler_id: int | None = None

        if verbose:
            log_panel = LogPanel(max_lines=8)

            def panel_sink(message: Any) -> None:
                """Sink function to write logs to the panel."""
                if log_panel is not None:
                    log_panel.add(message.record["message"])

            # Add a handler that writes to the log panel
            panel_handler_id = logger.add(
                panel_sink,
                level="INFO",
                format="{message}",
                filter=lambda record: record["level"].no >= 20,  # INFO and above
            )

        async def process_with_limit(file_path: Path) -> None:
            """Process a file with semaphore limit."""
            async with semaphore:
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
                    result = await process_func(file_path)

                    if result.success:
                        file_state.status = FileStatus.COMPLETED
                        file_state.output = result.output_path
                        file_state.images_extracted = result.images_extracted
                        file_state.screenshots = result.screenshots
                        file_state.llm_cost_usd = result.llm_cost_usd
                        file_state.llm_usage = result.llm_usage
                        # Collect image analysis result for JSON aggregation
                        if result.image_analysis_result is not None:
                            self.image_analysis_results.append(
                                result.image_analysis_result
                            )
                    else:
                        file_state.status = FileStatus.FAILED
                        file_state.error = result.error

                except Exception as e:
                    file_state.status = FileStatus.FAILED
                    file_state.error = str(e)
                    logger.error(f"Failed to process {file_path.name}: {e}")

                finally:
                    end_time = asyncio.get_event_loop().time()
                    file_state.completed_at = datetime.now().astimezone().isoformat()
                    file_state.duration_seconds = end_time - start_time

                    # Save state after each file (throttled)
                    self.save_state()

                    # Update progress
                    progress.advance(overall_task)

        # Run with live progress display
        # Disable console handler to avoid conflict with progress bar
        # (file logging continues to work)
        if console_handler_id is not None:
            try:
                logger.remove(console_handler_id)
            except ValueError:
                pass  # Handler already removed

        try:
            if verbose and log_panel is not None:
                # Verbose mode: show progress + log panel
                display = Group(progress, log_panel)
                with Live(display, console=self.console, refresh_per_second=4):
                    tasks = [process_with_limit(f) for f in files]
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Normal mode: progress bar only
                with Live(progress, console=self.console, refresh_per_second=4):
                    tasks = [process_with_limit(f) for f in files]
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Remove panel handler if added
            if panel_handler_id is not None:
                try:
                    logger.remove(panel_handler_id)
                except ValueError:
                    pass

            # Re-add console handler (restore original state)
            if console_handler_id is not None:
                import sys

                logger.add(
                    sys.stderr,
                    level="DEBUG" if verbose else "INFO",
                    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
                )

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

        Report file is saved to: reports/markit.<hash>.report.json
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

        atomic_write_json(self.report_file, report)
        logger.info(f"Report saved: {self.report_file.resolve()}")

        return self.report_file

    def print_summary(self) -> None:
        """Print summary to console."""
        if self.state is None:
            return

        table = Table(title="Batch Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Files", str(self.state.total))
        table.add_row("Completed", str(self.state.completed_count))
        table.add_row("Failed", str(self.state.failed_count))

        # Calculate wall-clock duration from started_at to updated_at
        wall_duration = 0.0
        if self.state.started_at and self.state.updated_at:
            try:
                start = datetime.fromisoformat(self.state.started_at)
                end = datetime.fromisoformat(self.state.updated_at)
                wall_duration = (end - start).total_seconds()
            except ValueError:
                # Fallback to sum of individual durations
                wall_duration = sum(
                    f.duration_seconds or 0 for f in self.state.files.values()
                )
        table.add_row("Duration", f"{wall_duration:.1f}s")

        # LLM cost
        total_cost = sum(f.llm_cost_usd for f in self.state.files.values())
        if total_cost > 0:
            table.add_row("LLM Cost", f"${total_cost:.4f}")

        self.console.print(table)

        # Print failed files if any
        failed = [f for f in self.state.files.values() if f.status == FileStatus.FAILED]
        if failed:
            self.console.print("\n[red]Failed files:[/red]")
            for f in failed:
                self.console.print(f"  - {Path(f.path).name}: {f.error}")

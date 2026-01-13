"""Shared execution logic for conversion commands."""

import asyncio
import signal
import sys
from collections.abc import Callable
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from markit.cli.shared.context import ConversionContext
from markit.utils.logging import get_logger, set_log_output, setup_logging

if TYPE_CHECKING:
    from markit.core.pipeline import ConversionPipeline, PipelineResult

log = get_logger(__name__)


class SignalHandler:
    """Context manager for handling interrupt signals during conversion."""

    def __init__(
        self,
        console: Console,
        context_info: dict | None = None,
    ):
        self.console = console
        self.context_info = context_info or {}
        self.interrupted = False
        self._original_sigint = None
        self._original_sigterm = None

    def __enter__(self) -> "SignalHandler":
        """Install signal handlers."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handler)
        if hasattr(signal, "SIGTERM"):
            self._original_sigterm = signal.signal(signal.SIGTERM, self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        return False

    def _handler(self, signum: int, frame) -> None:  # noqa: ARG002
        """Handle interrupt signals."""
        if self.interrupted:
            sys.exit(130)

        self.interrupted = True
        sig_name = signal.Signals(signum).name

        log.warning(
            "Task Interrupted",
            signal=sig_name,
            interrupted_at=dt.now().isoformat(),
            **self.context_info,
        )

        self.console.print(f"\n[yellow]Interrupted by {sig_name}. Exiting...[/yellow]")
        raise typer.Exit(130)


def execute_single_file(
    input_file: Path,
    ctx: ConversionContext,
    on_dry_run: Callable[[], None] | None = None,
) -> None:
    """Execute single file conversion with progress display.

    Args:
        input_file: Input file to convert
        ctx: Conversion context
        on_dry_run: Optional callback for dry-run display
    """
    ctx.log_start(input_file=str(input_file))

    if ctx.options.dry_run:
        if on_dry_run:
            on_dry_run()
        return

    with SignalHandler(ctx.console, {"input_file": str(input_file)}):
        try:
            _run_conversion(input_file, ctx)
        except KeyboardInterrupt:
            log.warning(
                "Task Interrupted by KeyboardInterrupt",
                input_file=str(input_file),
                interrupted_at=dt.now().isoformat(),
            )
            ctx.console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
            raise typer.Exit(130) from None
        except Exception as e:
            log.error("Conversion failed", error=str(e), exc_info=True)
            ctx.console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e


def _run_conversion(input_file: Path, ctx: ConversionContext) -> None:
    """Run the actual conversion logic."""

    # Use concurrent fallback setting from config (enables backup model on timeout)
    use_concurrent_fallback = ctx.settings.llm.concurrent_fallback_enabled
    pipeline = ctx.create_pipeline(use_concurrent_fallback=use_concurrent_fallback)

    if ctx.options.verbose:
        result = pipeline.convert_file(input_file, ctx.output_dir)
    elif ctx.options.use_phased_pipeline:
        result = asyncio.run(
            execute_phased_conversion(input_file, ctx.output_dir, pipeline, ctx.console)
        )
    else:
        result = _run_simple_conversion(input_file, ctx, pipeline)

    _display_result(result, input_file, ctx.console)


def _run_simple_conversion(
    input_file: Path,
    ctx: ConversionContext,
    pipeline: "ConversionPipeline",
) -> "PipelineResult":
    """Run simple conversion with spinner."""
    import logging

    root_logger = logging.getLogger()
    console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    original_levels = [(h, h.level) for h in console_handlers]

    for handler in console_handlers:
        handler.setLevel(logging.CRITICAL + 1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=ctx.console,
            transient=True,
        ) as progress:
            progress.add_task("Converting...", total=None)
            result = pipeline.convert_file(input_file, ctx.output_dir)
    finally:
        for handler, level in original_levels:
            handler.setLevel(level)

    return result


def _display_result(
    result: "PipelineResult",
    input_file: Path,
    console: Console,
) -> None:
    """Display conversion result."""
    if result.success:
        log.info(
            "Task Completed Successfully",
            output_path=str(result.output_path),
            images_count=result.images_count,
            metadata=result.metadata,
        )
        images_info = f" ({result.images_count} images)" if result.images_count > 0 else ""
        console.print(f"  [green]✓[/green] {input_file.name}{images_info}")
        console.print(f"  Output: {result.output_path}")
    else:
        log.error("Task Failed", error=result.error)
        console.print(f"  [red]✗[/red] {input_file.name}")
        console.print(f"  [red]Error:[/red] {result.error}")
        raise typer.Exit(1)


async def execute_phased_conversion(
    input_file: Path,
    output_dir: Path,
    pipeline: "ConversionPipeline",
    console: Console,
) -> "PipelineResult":
    """Execute conversion with phased progress display.

    This is the shared implementation for phased conversion that
    shows progress bars for each phase:
    1. Document Conversion
    2. LLM Processing
    3. Finalization

    Args:
        input_file: Input file to convert
        output_dir: Output directory
        pipeline: Initialized ConversionPipeline
        console: Rich console for output

    Returns:
        PipelineResult from the conversion
    """
    import os

    from markit.core.pipeline import PipelineResult
    from markit.image.analyzer import ImageAnalysis

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        original_stderr = sys.stderr

        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            setup_logging(level="WARNING", console=progress.console)

            # Phase 1: Document Conversion
            task_convert = progress.add_task("[cyan]Converting document...", total=1)
            doc_result = await pipeline.convert_document_only(input_file, output_dir)
            progress.advance(task_convert)

            if not doc_result.success:
                set_log_output(original_stderr)
                setup_logging(level="WARNING")
                return PipelineResult(success=False, error=doc_result.error)

            # Phase 2: LLM Processing
            llm_tasks = await pipeline.create_llm_tasks(doc_result)

            if llm_tasks:
                task_llm = progress.add_task("[cyan]LLM Processing...", total=len(llm_tasks))
                llm_results = []
                for coro in llm_tasks:
                    result = await coro
                    llm_results.append(result)
                    progress.advance(task_llm)
            else:
                llm_results = []

            # Phase 3: Finalization
            task_final = progress.add_task("[cyan]Finalizing...", total=1)

            image_analyses: list[ImageAnalysis] = []
            enhanced_markdown: str | None = None

            for llm_result in llm_results:
                if llm_result is not None:
                    if isinstance(llm_result, ImageAnalysis):
                        image_analyses.append(llm_result)
                    elif isinstance(llm_result, str):
                        enhanced_markdown = llm_result

            pipeline_result = await pipeline.finalize_output(
                doc_result,
                image_analyses=image_analyses if image_analyses else None,
                enhanced_markdown=enhanced_markdown,
            )
            progress.advance(task_final)

        set_log_output(original_stderr)
        setup_logging(level="WARNING")

    return pipeline_result

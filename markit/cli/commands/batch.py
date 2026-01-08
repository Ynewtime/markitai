"""Batch command for directory conversion."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from markit.config import get_settings
from markit.config.constants import LLM_PROVIDERS, PDF_ENGINES, SUPPORTED_EXTENSIONS
from markit.core.pipeline import ConversionPipeline, DocumentConversionResult, PipelineResult
from markit.core.state import StateManager
from markit.llm.queue import LLMTask, LLMTaskQueue
from markit.utils.concurrency import ConcurrencyManager, TaskResult
from markit.utils.logging import get_console, get_logger, setup_logging

console = get_console()
log = get_logger(__name__)


def batch(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Input directory containing documents.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for converted files.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Recursively process subdirectories.",
        ),
    ] = False,
    include: Annotated[
        str | None,
        typer.Option(
            "--include",
            help="File pattern to include (glob syntax).",
        ),
    ] = None,
    exclude: Annotated[
        str | None,
        typer.Option(
            "--exclude",
            help="File pattern to exclude (glob syntax).",
        ),
    ] = None,
    file_concurrency: Annotated[
        int,
        typer.Option(
            "--file-concurrency",
            help="Number of files to process concurrently.",
        ),
    ] = 8,
    image_concurrency: Annotated[
        int,
        typer.Option(
            "--image-concurrency",
            help="Number of images to process concurrently.",
        ),
    ] = 16,
    llm_concurrency: Annotated[
        int,
        typer.Option(
            "--llm-concurrency",
            help="Number of LLM requests concurrently.",
        ),
    ] = 10,
    on_conflict: Annotated[
        str,
        typer.Option(
            "--on-conflict",
            help="How to handle output file conflicts: skip, overwrite, rename.",
        ),
    ] = "rename",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last interrupted batch.",
        ),
    ] = False,
    state_file: Annotated[
        Path | None,
        typer.Option(
            "--state-file",
            help="State file path for tracking progress.",
        ),
    ] = None,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help="Enable LLM Markdown format optimization.",
        ),
    ] = False,
    analyze_image: Annotated[
        bool,
        typer.Option(
            "--analyze-image",
            help="Enable LLM image analysis (generates alt text only).",
        ),
    ] = False,
    analyze_image_with_md: Annotated[
        bool,
        typer.Option(
            "--analyze-image-with-md",
            help="Enable LLM image analysis with detailed .md description files.",
        ),
    ] = False,
    no_compress: Annotated[
        bool,
        typer.Option(
            "--no-compress",
            help="Disable image compression.",
        ),
    ] = False,
    pdf_engine: Annotated[
        str | None,
        typer.Option(
            "--pdf-engine",
            help=f"PDF processing engine. Options: {', '.join(PDF_ENGINES)}",
        ),
    ] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider",
            help=f"LLM provider to use. Options: {', '.join(LLM_PROVIDERS)}",
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="LLM model name to use.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show batch plan without executing.",
        ),
    ] = False,
) -> None:
    """Batch convert documents in a directory.

    Examples:
        markit batch ./documents
        markit batch ./documents -o ./output -r
        markit batch ./documents --include "*.docx" --llm
        markit batch ./documents --resume --state-file ./state.json
    """
    # Setup logging
    # In non-verbose mode, only show warnings/errors on console to avoid clutter with progress bar
    # In verbose mode, show all logs (DEBUG level) and no progress bar
    settings = get_settings()
    if verbose:
        log_level = "DEBUG"
    else:
        # Non-verbose: suppress console logs below WARNING, but file logs use configured level
        log_level = "WARNING"
    setup_logging(level=log_level, log_file=settings.log_file)

    # Log file behavior hint (only in verbose mode)
    if verbose:
        if settings.log_file:
            log.info("Logs will be saved to", log_file=settings.log_file)
        else:
            log.info(
                "Logs output to console only. Set log_file in markit.toml to save logs to file."
            )

    # Validate options
    if on_conflict not in ("skip", "overwrite", "rename"):
        console.print(
            f"[red]Error:[/red] Invalid conflict strategy '{on_conflict}'. "
            "Options: skip, overwrite, rename"
        )
        raise typer.Exit(1)

    if pdf_engine and pdf_engine not in PDF_ENGINES:
        console.print(
            f"[red]Error:[/red] Invalid PDF engine '{pdf_engine}'. "
            f"Options: {', '.join(PDF_ENGINES)}"
        )
        raise typer.Exit(1)

    if llm_provider and llm_provider not in LLM_PROVIDERS:
        console.print(
            f"[red]Error:[/red] Invalid LLM provider '{llm_provider}'. "
            f"Options: {', '.join(LLM_PROVIDERS)}"
        )
        raise typer.Exit(1)

    # Determine output directory (use cwd to be consistent with convert command)
    output_dir = output or Path.cwd() / settings.output.default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine state file
    state_path = state_file or Path(settings.state_file)

    # Effective analyze_image: either flag enables analysis, with_md adds description files
    effective_analyze_image = analyze_image or analyze_image_with_md

    log.info(
        "Starting batch conversion",
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        recursive=recursive,
        llm_enabled=llm,
        analyze_image=effective_analyze_image,
        analyze_image_with_md=analyze_image_with_md,
    )

    if dry_run:
        _show_dry_run(
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=recursive,
            include=include,
            exclude=exclude,
        )
        return

    # Execute batch conversion
    try:
        asyncio.run(
            _execute_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=recursive,
                include=include,
                exclude=exclude,
                file_concurrency=file_concurrency,
                image_concurrency=image_concurrency,
                llm_concurrency=llm_concurrency,
                on_conflict=on_conflict,
                resume=resume,
                state_path=state_path,
                llm_enabled=llm,
                analyze_image=effective_analyze_image,
                analyze_image_with_md=analyze_image_with_md,
                compress_images=not no_compress,
                pdf_engine=pdf_engine,
                llm_provider=llm_provider,
                llm_model=llm_model,
                settings=settings,
                verbose=verbose,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch interrupted. Use --resume to continue.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        log.error("Batch conversion failed", error=str(e), exc_info=True)
        raise typer.Exit(1) from e


async def _execute_batch(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    include: str | None,
    exclude: str | None,
    file_concurrency: int,
    image_concurrency: int,
    llm_concurrency: int,
    on_conflict: str,
    resume: bool,
    state_path: Path,
    llm_enabled: bool,
    analyze_image: bool,
    analyze_image_with_md: bool,
    compress_images: bool,
    pdf_engine: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    settings,
    verbose: bool = False,
) -> None:
    """Execute the batch conversion asynchronously."""
    # Initialize state manager
    state_manager = StateManager(state_path)

    # Discover files or load from state
    if resume:
        state = state_manager.load_batch()
        if state is None:
            console.print("[yellow]No previous batch state found. Starting fresh.[/yellow]")
            files = _discover_files(input_dir, recursive, include, exclude)
        else:
            files = state_manager.get_pending_files()
            console.print(
                f"[green]Resuming batch:[/green] {len(files)} files remaining "
                f"(completed: {state.completed_files}, failed: {state.failed_files})"
            )
    else:
        files = _discover_files(input_dir, recursive, include, exclude)
        state_manager.clear()  # Clear any previous state

    if not files:
        console.print("[yellow]No files to process.[/yellow]")
        return

    # Create batch state if not resuming
    if not resume or state_manager.get_state() is None:
        state_manager.create_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            files=files,
            options={
                "recursive": recursive,
                "include": include,
                "exclude": exclude,
                "llm_enabled": llm_enabled,
                "analyze_image": analyze_image,
                "compress_images": compress_images,
                "on_conflict": on_conflict,
            },
        )

    # Create pipeline and concurrency manager
    pipeline = ConversionPipeline(
        settings=settings,
        llm_enabled=llm_enabled,
        analyze_image=analyze_image,
        analyze_image_with_md=analyze_image_with_md,
        compress_images=compress_images,
        pdf_engine=pdf_engine,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    concurrency = ConcurrencyManager(
        file_workers=file_concurrency,
        image_workers=image_concurrency,
        llm_workers=llm_concurrency,
    )

    # Update conflict strategy in settings
    settings.output.on_conflict = on_conflict

    # Choose processing mode based on LLM usage
    # When LLM is enabled, use phased pipeline for better parallelism
    # When LLM is disabled, use original pipeline (faster for simple conversions)
    use_phased_pipeline = llm_enabled or analyze_image

    if use_phased_pipeline:
        # Phased pipeline: maximize parallelism for LLM-heavy workloads
        if verbose:
            results = await _process_files_phased_verbose(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
                llm_concurrency=llm_concurrency,
            )
        else:
            results = await _process_files_phased_with_progress(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
                llm_concurrency=llm_concurrency,
            )
    else:
        # Original pipeline: simpler flow when no LLM processing needed
        if verbose:
            results = await _process_files_verbose(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
            )
        else:
            results = await _process_files_with_progress(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
            )

    # Mark batch complete
    state_manager.mark_batch_complete()

    # Display summary
    _display_summary(results, state_manager)


def _discover_files(
    input_dir: Path,
    recursive: bool,
    include: str | None,
    exclude: str | None,
) -> list[Path]:
    """Discover files to process."""
    pattern = "**/*" if recursive else "*"
    files = []

    for ext in SUPPORTED_EXTENSIONS:
        if include:
            # Use custom include pattern
            for f in input_dir.glob(f"{pattern}{ext}"):
                if Path(f.name).match(include):
                    files.append(f)
        else:
            files.extend(input_dir.glob(f"{pattern}{ext}"))

    # Apply exclude pattern
    if exclude:
        files = [f for f in files if not Path(f.name).match(exclude)]

    # Sort by name for consistent ordering
    files.sort(key=lambda p: str(p))

    return files


async def _process_files_verbose(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
) -> list[TaskResult[Path]]:
    """Process files with verbose logging (no progress bar).

    Args:
        files: List of files to process
        pipeline: Conversion pipeline
        concurrency: Concurrency manager
        output_dir: Output directory
        state_manager: State manager
        input_dir: Input directory for relative path display
    """
    results: list[TaskResult[Path]] = []

    log.info("Processing files", total=len(files))

    async def process_file(file_path: Path) -> PipelineResult:
        """Process a single file."""
        rel_path = file_path.relative_to(input_dir)
        log.info("Converting file", file=str(rel_path))

        # Update state to processing
        state_manager.update_file_status(file_path, "processing")

        try:
            result = await pipeline.convert_file_async(file_path, output_dir)

            if result.success:
                state_manager.update_file_status(
                    file_path,
                    "completed",
                    output_path=result.output_path,
                )
                log.info(
                    "File converted successfully",
                    file=str(rel_path),
                    output=str(result.output_path),
                    images=result.images_count,
                )
            else:
                state_manager.update_file_status(
                    file_path,
                    "failed",
                    error=result.error,
                )
                log.error("File conversion failed", file=str(rel_path), error=result.error)

            return result
        except Exception as e:
            state_manager.update_file_status(
                file_path,
                "failed",
                error=str(e),
            )
            log.error("File conversion error", file=str(rel_path), error=str(e))
            raise

    def on_progress(
        item: Path,
        result: PipelineResult | None,
        error: Exception | None,
    ) -> None:
        """Progress callback - just for tracking, no console output."""
        pass  # Logging is done in process_file

    # Process with concurrency
    task_results = await concurrency.map_file_tasks(
        items=files,
        func=process_file,
        on_progress=on_progress,
    )

    results.extend(task_results)
    return results


async def _process_files_with_progress(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    quiet: bool = False,
) -> list[TaskResult[Path]]:
    """Process files with Rich progress display.

    Args:
        files: List of files to process
        pipeline: Conversion pipeline
        concurrency: Concurrency manager
        output_dir: Output directory
        state_manager: State manager
        input_dir: Input directory for relative path display
        quiet: If True, suppress per-file status output (only show progress bar)
    """
    from markit.utils.logging import set_log_output, setup_logging

    results: list[TaskResult[Path]] = []

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
        # Suppress structlog output during progress to avoid visual clutter
        # Logs will still go to file if configured
        import os
        import sys

        # Suppress console logs during progress (they interfere with the display)
        original_stderr = sys.stderr
        settings = get_settings()

        # Use context manager for devnull to ensure proper cleanup
        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            # Set log level to WARNING to reduce noise
            setup_logging(level="WARNING", console=progress.console)

            task_id = progress.add_task(
                "[cyan]Converting files...",
                total=len(files),
            )

            async def process_file(file_path: Path) -> PipelineResult:
                """Process a single file."""
                # Update state to processing
                state_manager.update_file_status(file_path, "processing")

                try:
                    result = await pipeline.convert_file_async(file_path, output_dir)

                    if result.success:
                        state_manager.update_file_status(
                            file_path,
                            "completed",
                            output_path=result.output_path,
                        )
                    else:
                        state_manager.update_file_status(
                            file_path,
                            "failed",
                            error=result.error,
                        )

                    return result
                except Exception as e:
                    state_manager.update_file_status(
                        file_path,
                        "failed",
                        error=str(e),
                    )
                    raise

            def on_progress(
                item: Path,
                result: PipelineResult | None,
                error: Exception | None,
            ) -> None:
                """Progress callback."""
                progress.advance(task_id)

                # In quiet mode, only show progress bar (no per-file output)
                # But always show errors even in quiet mode
                rel_path = item.relative_to(input_dir)
                if error:
                    error_msg = _simplify_error(str(error))
                    progress.console.print(f"  [red]x[/red] {rel_path}")
                    progress.console.print(f"    [dim]{error_msg}[/dim]")
                elif result and not result.success:
                    error_msg = _simplify_error(result.error or "Unknown error")
                    progress.console.print(f"  [red]x[/red] {rel_path}")
                    progress.console.print(f"    [dim]{error_msg}[/dim]")
                elif result and result.success and not quiet:
                    # Only show success messages when not in quiet mode
                    images_info = ""
                    if result.images_count > 0:
                        images_info = f" [dim]({result.images_count} images)[/dim]"
                    progress.console.print(f"  [green]✓[/green] {rel_path}{images_info}")

            # Process with concurrency
            task_results = await concurrency.map_file_tasks(
                items=files,
                func=process_file,
                on_progress=on_progress,
            )

            results.extend(task_results)

        # Restore normal log output (keep WARNING level for non-verbose mode)
        set_log_output(original_stderr)
        # Keep WARNING level to avoid log clutter after progress bar
        setup_logging(level="WARNING", log_file=settings.log_file)

    return results


# ============================================================================
# Phased Pipeline Processing Functions (for LLM-enabled batch processing)
# ============================================================================


async def _process_files_phased_verbose(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    llm_concurrency: int = 10,
) -> list[TaskResult[Path]]:
    """Process files using phased pipeline with verbose logging.

    Three-phase processing for maximum parallelism:
    - Phase 1: Document conversion (file_semaphore, releases early)
    - Phase 2: LLM processing (global queue with llm_semaphore)
    - Phase 3: Output finalization

    Args:
        files: List of files to process
        pipeline: Conversion pipeline
        concurrency: Concurrency manager
        output_dir: Output directory
        state_manager: State manager
        input_dir: Input directory for relative path display
        llm_concurrency: Maximum concurrent LLM requests
    """
    from markit.image.analyzer import ImageAnalysis

    results: list[TaskResult[Path]] = []
    llm_queue = LLMTaskQueue(max_concurrent=llm_concurrency, max_pending=100)

    log.info("Starting phased batch processing", total=len(files), llm_concurrency=llm_concurrency)

    # ========================================================================
    # Phase 1: Document Conversion (parallel with file_semaphore)
    # ========================================================================
    log.info("Phase 1: Document conversion")

    doc_results: dict[Path, DocumentConversionResult] = {}
    conversion_errors: dict[Path, str] = {}

    async def convert_document(file_path: Path) -> DocumentConversionResult:
        """Convert a single document (Phase 1)."""
        rel_path = file_path.relative_to(input_dir)
        log.info("Converting document", file=str(rel_path))
        state_manager.update_file_status(file_path, "processing")

        try:
            result = await pipeline.convert_document_only(file_path, output_dir)
            if result.success:
                log.info(
                    "Document converted",
                    file=str(rel_path),
                    images=len(result.processed_images),
                )
            else:
                log.error("Document conversion failed", file=str(rel_path), error=result.error)
            return result
        except Exception as e:
            log.error("Document conversion error", file=str(rel_path), error=str(e))
            raise

    # Run document conversions with file concurrency limit
    conversion_task_results = await concurrency.map_file_tasks(
        items=files,
        func=convert_document,
    )

    # Collect results
    for task_result in conversion_task_results:
        file_path = task_result.item
        if task_result.success and task_result.result:
            doc_results[file_path] = task_result.result
        else:
            conversion_errors[file_path] = task_result.error or "Unknown error"
            state_manager.update_file_status(file_path, "failed", error=task_result.error)

    log.info(
        "Phase 1 complete",
        succeeded=len(doc_results),
        failed=len(conversion_errors),
    )

    # ========================================================================
    # Phase 2: LLM Processing (global queue with rate limiting)
    # ========================================================================
    if doc_results:
        log.info("Phase 2: LLM processing")

        # Submit all LLM tasks to the global queue
        for file_path, doc_result in doc_results.items():
            if not doc_result.success:
                continue

            # Create LLM tasks for this file
            llm_tasks = pipeline.create_llm_tasks(doc_result)

            # Submit image analysis tasks
            for i, coro in enumerate(llm_tasks):
                # Determine task type based on position
                # Image analysis tasks come first, then enhancement task
                num_images = len(doc_result.images_for_analysis)
                if i < num_images:
                    task_type = "image_analysis"
                    task_id = doc_result.images_for_analysis[i].filename
                else:
                    task_type = "chunk_enhancement"
                    task_id = "markdown"

                await llm_queue.submit(
                    LLMTask(
                        source_file=file_path,
                        task_type=task_type,  # type: ignore[arg-type]
                        task_id=task_id,
                        coro=coro,
                    )
                )

        # Wait for all LLM tasks to complete
        llm_results = await llm_queue.wait_all()

        log.info(
            "Phase 2 complete",
            total_tasks=len(llm_results),
            succeeded=sum(1 for r in llm_results if r.success),
        )

    else:
        llm_results = []

    # ========================================================================
    # Phase 3: Output Finalization
    # ========================================================================
    log.info("Phase 3: Output finalization")

    for file_path, doc_result in doc_results.items():
        rel_path = file_path.relative_to(input_dir)

        try:
            # Gather LLM results for this file
            file_llm_results = llm_queue.get_results_for_file(llm_results, file_path)

            # Separate image analyses and enhancement result
            image_analyses: list[ImageAnalysis] = []
            enhanced_markdown: str | None = None

            for llm_result in file_llm_results:
                if llm_result.success:
                    if llm_result.task_type == "image_analysis":
                        image_analyses.append(llm_result.result)
                    elif llm_result.task_type == "chunk_enhancement":
                        enhanced_markdown = llm_result.result

            # Finalize output
            pipeline_result = await pipeline.finalize_output(
                doc_result,
                image_analyses=image_analyses if image_analyses else None,
                enhanced_markdown=enhanced_markdown,
            )

            if pipeline_result.success:
                state_manager.update_file_status(
                    file_path,
                    "completed",
                    output_path=pipeline_result.output_path,
                )
                log.info(
                    "File completed",
                    file=str(rel_path),
                    output=str(pipeline_result.output_path),
                    images=pipeline_result.images_count,
                )
            else:
                state_manager.update_file_status(
                    file_path,
                    "failed",
                    error=pipeline_result.error,
                )
                log.error("Finalization failed", file=str(rel_path), error=pipeline_result.error)

            results.append(
                TaskResult(
                    item=file_path,
                    success=pipeline_result.success,
                    result=pipeline_result,
                    error=pipeline_result.error,
                )
            )

        except Exception as e:
            state_manager.update_file_status(file_path, "failed", error=str(e))
            log.error("Finalization error", file=str(rel_path), error=str(e))
            results.append(
                TaskResult(
                    item=file_path,
                    success=False,
                    error=str(e),
                )
            )

    # Add failed conversions to results
    for file_path, error in conversion_errors.items():
        results.append(
            TaskResult(
                item=file_path,
                success=False,
                error=error,
            )
        )

    log.info("Phased batch processing complete", total=len(results))
    return results


async def _process_files_phased_with_progress(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    llm_concurrency: int = 10,
) -> list[TaskResult[Path]]:
    """Process files using phased pipeline with progress display.

    Three-phase processing for maximum parallelism with Rich progress bar.

    Args:
        files: List of files to process
        pipeline: Conversion pipeline
        concurrency: Concurrency manager
        output_dir: Output directory
        state_manager: State manager
        input_dir: Input directory for relative path display
        llm_concurrency: Maximum concurrent LLM requests
    """
    from markit.image.analyzer import ImageAnalysis
    from markit.utils.logging import set_log_output, setup_logging

    results: list[TaskResult[Path]] = []
    llm_queue = LLMTaskQueue(max_concurrent=llm_concurrency, max_pending=100)

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
        import os
        import sys

        original_stderr = sys.stderr
        settings = get_settings()

        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            setup_logging(level="WARNING", console=progress.console)

            # ================================================================
            # Phase 1: Document Conversion
            # ================================================================
            phase1_task = progress.add_task(
                "[cyan]Phase 1: Converting documents...",
                total=len(files),
            )

            doc_results: dict[Path, DocumentConversionResult] = {}
            conversion_errors: dict[Path, str] = {}

            async def convert_document(file_path: Path) -> DocumentConversionResult:
                """Convert a single document."""
                state_manager.update_file_status(file_path, "processing")
                result = await pipeline.convert_document_only(file_path, output_dir)
                return result

            def on_conversion_progress(
                item: Path,
                result: DocumentConversionResult | None,
                error: Exception | None,
            ) -> None:
                """Progress callback for Phase 1."""
                progress.advance(phase1_task)
                rel_path = item.relative_to(input_dir)

                if error or result and not result.success:
                    progress.console.print(f"  [red]x[/red] {rel_path} (conversion failed)")

            conversion_task_results = await concurrency.map_file_tasks(
                items=files,
                func=convert_document,
                on_progress=on_conversion_progress,
            )

            for task_result in conversion_task_results:
                file_path = task_result.item
                if task_result.success and task_result.result and task_result.result.success:
                    doc_results[file_path] = task_result.result
                else:
                    error_msg = task_result.error or (
                        task_result.result.error if task_result.result else "Unknown error"
                    )
                    conversion_errors[file_path] = error_msg
                    state_manager.update_file_status(file_path, "failed", error=error_msg)

            progress.update(phase1_task, description="[green]Phase 1: Documents converted")

            # ================================================================
            # Phase 2: LLM Processing
            # ================================================================
            total_llm_tasks = 0
            if doc_results:
                # Count total LLM tasks
                for doc_result in doc_results.values():
                    if doc_result.success:
                        total_llm_tasks += len(doc_result.images_for_analysis)
                        if pipeline.llm_enabled:
                            total_llm_tasks += 1  # Enhancement task

            if total_llm_tasks > 0:
                phase2_task = progress.add_task(
                    "[cyan]Phase 2: LLM processing...",
                    total=total_llm_tasks,
                )

                # Submit all LLM tasks
                for file_path, doc_result in doc_results.items():
                    if not doc_result.success:
                        continue

                    llm_tasks = pipeline.create_llm_tasks(doc_result)
                    num_images = len(doc_result.images_for_analysis)

                    for i, coro in enumerate(llm_tasks):
                        if i < num_images:
                            task_type = "image_analysis"
                            task_id = doc_result.images_for_analysis[i].filename
                        else:
                            task_type = "chunk_enhancement"
                            task_id = "markdown"

                        await llm_queue.submit(
                            LLMTask(
                                source_file=file_path,
                                task_type=task_type,  # type: ignore[arg-type]
                                task_id=task_id,
                                coro=coro,
                            )
                        )

                # Wait for LLM tasks with progress updates
                llm_results = await llm_queue.wait_all()

                # Update progress (all at once since we waited)
                progress.update(phase2_task, completed=total_llm_tasks)
                progress.update(phase2_task, description="[green]Phase 2: LLM processing done")
            else:
                llm_results = []

            # ================================================================
            # Phase 3: Output Finalization
            # ================================================================
            phase3_task = progress.add_task(
                "[cyan]Phase 3: Finalizing outputs...",
                total=len(doc_results),
            )

            for file_path, doc_result in doc_results.items():
                rel_path = file_path.relative_to(input_dir)

                try:
                    file_llm_results = llm_queue.get_results_for_file(llm_results, file_path)

                    image_analyses: list[ImageAnalysis] = []
                    enhanced_markdown: str | None = None

                    for llm_result in file_llm_results:
                        if llm_result.success:
                            if llm_result.task_type == "image_analysis":
                                image_analyses.append(llm_result.result)
                            elif llm_result.task_type == "chunk_enhancement":
                                enhanced_markdown = llm_result.result

                    pipeline_result = await pipeline.finalize_output(
                        doc_result,
                        image_analyses=image_analyses if image_analyses else None,
                        enhanced_markdown=enhanced_markdown,
                    )

                    progress.advance(phase3_task)

                    if pipeline_result.success:
                        state_manager.update_file_status(
                            file_path,
                            "completed",
                            output_path=pipeline_result.output_path,
                        )
                        images_info = ""
                        if pipeline_result.images_count > 0:
                            images_info = f" [dim]({pipeline_result.images_count} images)[/dim]"
                        progress.console.print(f"  [green]✓[/green] {rel_path}{images_info}")
                    else:
                        state_manager.update_file_status(
                            file_path, "failed", error=pipeline_result.error
                        )
                        error_msg = _simplify_error(pipeline_result.error or "Unknown error")
                        progress.console.print(f"  [red]x[/red] {rel_path}")
                        progress.console.print(f"    [dim]{error_msg}[/dim]")

                    results.append(
                        TaskResult(
                            item=file_path,
                            success=pipeline_result.success,
                            result=pipeline_result,
                            error=pipeline_result.error,
                        )
                    )

                except Exception as e:
                    progress.advance(phase3_task)
                    state_manager.update_file_status(file_path, "failed", error=str(e))
                    error_msg = _simplify_error(str(e))
                    progress.console.print(f"  [red]x[/red] {rel_path}")
                    progress.console.print(f"    [dim]{error_msg}[/dim]")
                    results.append(
                        TaskResult(
                            item=file_path,
                            success=False,
                            error=str(e),
                        )
                    )

            progress.update(phase3_task, description="[green]Phase 3: Outputs finalized")

            # Add failed conversions to results
            for file_path, error in conversion_errors.items():
                results.append(
                    TaskResult(
                        item=file_path,
                        success=False,
                        error=error,
                    )
                )

        set_log_output(original_stderr)
        setup_logging(level="WARNING", log_file=settings.log_file)

    return results


def _display_summary(
    _results: list[TaskResult[Path]],  # Reserved for future detailed result analysis
    state_manager: StateManager,
) -> None:
    """Display batch processing summary."""
    state = state_manager.get_state()
    if state is None:
        return

    console.print()

    # Create summary table
    table = Table(title="Batch Summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total Files", str(state.total_files))
    table.add_row("Completed", f"[green]{state.completed_files}[/green]")
    table.add_row("Failed", f"[red]{state.failed_files}[/red]")
    table.add_row("Skipped", f"[yellow]{state.skipped_files}[/yellow]")

    if state.completed_files > 0 and state.total_files > 0:
        success_rate = (state.completed_files / state.total_files) * 100
        table.add_row("Success Rate", f"{success_rate:.1f}%")

    console.print(table)

    # Show failed files with simplified error messages
    failed_files = [(path, fs.error) for path, fs in state.files.items() if fs.status == "failed"]

    if failed_files:
        console.print()
        console.print("[bold red]Failed Files:[/bold red]")
        for path, error in failed_files[:10]:
            # Extract just the filename from path
            filename = Path(path).name
            # Simplify error message - extract the core reason
            error_summary = _simplify_error(error or "Unknown error")
            console.print(f"  [dim]-[/dim] {filename}")
            console.print(f"    [dim]{error_summary}[/dim]")
        if len(failed_files) > 10:
            console.print(f"  [dim]... and {len(failed_files) - 10} more[/dim]")

    console.print()


def _simplify_error(error: str) -> str:
    """Simplify error message for display."""
    # Common patterns to simplify
    if "MissingDependencyException" in error:
        # Extract the dependency hint
        if "docx" in error.lower():
            return "Missing docx support - run: pip install markitdown[docx]"
        elif "xlsx" in error.lower():
            return "Missing xlsx support - run: pip install markitdown[xlsx]"
        elif "pptx" in error.lower():
            return "Missing pptx support - run: pip install markitdown[pptx]"
        elif "pdf" in error.lower():
            return "Missing pdf support - run: pip install markitdown[pdf]"
        return "Missing dependency - run: pip install markitdown[all]"

    if "All conversion attempts failed" in error:
        return "All converters failed for this file type"

    if "ExtractedImage" in error:
        return "Image processing error"

    if "Pandoc error" in error:
        if "Unknown input format" in error:
            return "Unsupported format for Pandoc fallback"
        return "Pandoc conversion failed"

    # Truncate long messages
    if len(error) > 100:
        return error[:97] + "..."

    return error


def _show_dry_run(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    include: str | None,
    exclude: str | None,
) -> None:
    """Display the batch plan without executing."""
    console.print("\n[bold blue]Batch Plan (Dry Run)[/bold blue]\n")
    console.print(f"  [bold]Input Directory:[/bold] {input_dir}")
    console.print(f"  [bold]Output Directory:[/bold] {output_dir}")
    console.print(f"  [bold]Recursive:[/bold] {recursive}")
    if include:
        console.print(f"  [bold]Include Pattern:[/bold] {include}")
    if exclude:
        console.print(f"  [bold]Exclude Pattern:[/bold] {exclude}")

    # Scan for files
    files = _discover_files(input_dir, recursive, include, exclude)

    console.print()
    console.print(f"[bold]Files Found:[/bold] {len(files)}")

    if files:
        # Group by extension
        by_ext: dict[str, list[Path]] = {}
        for f in files:
            ext = f.suffix.lower()
            by_ext.setdefault(ext, []).append(f)

        console.print()
        console.print("[bold]By Type:[/bold]")
        for ext, ext_files in sorted(by_ext.items()):
            console.print(f"  {ext}: {len(ext_files)}")

        console.print()
        console.print("[bold]Files:[/bold]")
        for f in files[:10]:
            console.print(f"  - {f.relative_to(input_dir)}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more")

    console.print()

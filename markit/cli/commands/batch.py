"""Batch command for directory conversion."""

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from markit.config import get_settings
from markit.config.constants import LLM_PROVIDERS, PDF_ENGINES, SUPPORTED_EXTENSIONS
from markit.core.pipeline import ConversionPipeline, DocumentConversionResult, PipelineResult
from markit.core.state import StateManager
from markit.llm.queue import LLMTask, LLMTaskQueue
from markit.utils.concurrency import ConcurrencyManager, TaskResult
from markit.utils.logging import (
    get_console,
    get_logger,
    set_log_output,
    setup_logging,
    setup_task_logging,
)

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
    settings = get_settings()

    # Setup task-level logging with unified behavior
    # Priority: user config (settings.log_dir) > default (.logs)
    task_id, log_path = setup_task_logging(
        log_dir=settings.log_dir,
        prefix="batch",
        verbose=verbose,
    )

    # Log file behavior hint (only in verbose mode)
    if verbose:
        log.info("Logs will be saved to", log_file=str(log_path))

    # Log detailed configuration at the beginning
    # Mask API keys before logging
    config_dump = settings.model_dump()
    if "llm" in config_dump and "providers" in config_dump["llm"]:
        for provider in config_dump["llm"]["providers"]:
            if "api_key" in provider and provider["api_key"]:
                provider["api_key"] = "***"

    log.info("Task Configuration", task_id=task_id, config=config_dump)

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

        # Use context manager for devnull to ensure proper cleanup
        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            # Set log level to WARNING to reduce noise
            setup_logging(level="WARNING", console=progress.console)

            progress_task_id = progress.add_task(
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
                progress.advance(progress_task_id)

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
        # Note: File handler is already set up from initial setup_task_logging
        setup_logging(level="WARNING")

    return results


# ============================================================================
# Phased Pipeline Processing Functions (for LLM-enabled batch processing)
# ============================================================================


async def _process_file_pipeline(
    file_path: Path,
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    llm_queue: LLMTaskQueue,
    output_dir: Path,
    input_dir: Path,
    state_manager: StateManager,
    callbacks: dict[str, Any],
) -> TaskResult[Path]:
    """Process a single file through the full pipeline (streaming)."""
    from markit.image.analyzer import ImageAnalysis

    # Phase 1: Document Conversion (Semaphore controlled)
    callbacks.get("on_phase1_start", lambda x: None)(file_path)
    state_manager.update_file_status(file_path, "processing")

    try:
        # Run conversion with file worker limit
        doc_result = await concurrency.run_file_task(
            pipeline.convert_document_only(file_path, output_dir)
        )

        if not doc_result.success:
            callbacks.get("on_phase1_error", lambda x, y: None)(file_path, doc_result.error)
            state_manager.update_file_status(file_path, "failed", error=doc_result.error)
            return TaskResult(file_path, False, error=doc_result.error)

        callbacks.get("on_phase1_complete", lambda x, y: None)(file_path, doc_result)

        # Phase 2: LLM Tasks
        llm_tasks = await pipeline.create_llm_tasks(doc_result)
        submitted_tasks = []

        # Only proceed to Phase 2 if there are tasks or if we just need to finalize
        if llm_tasks:
            callbacks.get("on_phase2_start", lambda x, y: None)(file_path, len(llm_tasks))

            for i, coro in enumerate(llm_tasks):
                # Determine task metadata
                num_images = len(doc_result.images_for_analysis)
                if i < num_images:
                    task_type = "image_analysis"
                    task_id = doc_result.images_for_analysis[i].filename
                else:
                    task_type = "chunk_enhancement"
                    task_id = "markdown"

                # Submit to global queue (returns Task object)
                # We await submit() which handles rate limiting (backpressure)
                task_obj = await llm_queue.submit(
                    LLMTask(
                        source_file=file_path,
                        task_type=task_type,  # type: ignore[arg-type]
                        task_id=task_id,
                        coro=coro,
                    )
                )
                submitted_tasks.append(task_obj)

            # Wait for THIS file's tasks to complete
            llm_results = await asyncio.gather(*submitted_tasks)
            callbacks.get("on_phase2_complete", lambda x, y: None)(file_path, llm_results)
        else:
            llm_results = []

        # Phase 3: Finalize
        callbacks.get("on_phase3_start", lambda x: None)(file_path)

        # Extract results
        image_analyses: list[ImageAnalysis] = []
        enhanced_markdown: str | None = None

        for llm_result in llm_results:
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

        if pipeline_result.success:
            state_manager.update_file_status(
                file_path, "completed", output_path=pipeline_result.output_path
            )
            callbacks.get("on_phase3_complete", lambda x, y: None)(file_path, pipeline_result)
            return TaskResult(file_path, True, result=pipeline_result)
        else:
            state_manager.update_file_status(file_path, "failed", error=pipeline_result.error)
            callbacks.get("on_phase3_error", lambda x, y: None)(file_path, pipeline_result.error)
            return TaskResult(file_path, False, error=pipeline_result.error)

    except Exception as e:
        callbacks.get("on_error", lambda x, y: None)(file_path, str(e))
        state_manager.update_file_status(file_path, "failed", error=str(e))
        return TaskResult(file_path, False, error=str(e))


async def _process_files_phased_verbose(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    llm_concurrency: int = 10,
) -> list[TaskResult[Path]]:
    """Process files using phased pipeline with verbose logging (streaming)."""
    llm_queue = LLMTaskQueue(max_concurrent=llm_concurrency, max_pending=100)
    log.info(
        "Starting streaming batch processing", total=len(files), llm_concurrency=llm_concurrency
    )

    # Define callbacks for verbose logging
    def on_phase1_start(file_path: Path) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.info("Converting document", file=str(rel_path))

    def on_phase1_complete(file_path: Path, result: DocumentConversionResult) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.info("Document converted", file=str(rel_path), images=len(result.processed_images))

    def on_phase1_error(file_path: Path, error: str) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.error("Document conversion failed", file=str(rel_path), error=error)

    def on_phase2_start(file_path: Path, count: int) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.info("Submitting LLM tasks", file=str(rel_path), count=count)

    def on_phase2_complete(file_path: Path, results: list[Any]) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.info("LLM tasks completed", file=str(rel_path), count=len(results))

    def on_phase3_complete(file_path: Path, result: PipelineResult) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.info(
            "File completed",
            file=str(rel_path),
            output=str(result.output_path),
            images=result.images_count,
        )

    def on_phase3_error(file_path: Path, error: str) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.error("Finalization failed", file=str(rel_path), error=error)

    def on_error(file_path: Path, error: str) -> None:
        rel_path = file_path.relative_to(input_dir)
        log.error("Processing error", file=str(rel_path), error=error)

    callbacks = {
        "on_phase1_start": on_phase1_start,
        "on_phase1_complete": on_phase1_complete,
        "on_phase1_error": on_phase1_error,
        "on_phase2_start": on_phase2_start,
        "on_phase2_complete": on_phase2_complete,
        "on_phase3_complete": on_phase3_complete,
        "on_phase3_error": on_phase3_error,
        "on_error": on_error,
    }

    # Start all pipelines concurrently
    tasks = [
        _process_file_pipeline(
            f, pipeline, concurrency, llm_queue, output_dir, input_dir, state_manager, callbacks
        )
        for f in files
    ]

    results = await asyncio.gather(*tasks)
    log.info("Batch processing complete", total=len(results))
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
    """Process files using phased pipeline with progress display (streaming)."""
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

        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            setup_logging(level="WARNING", console=progress.console)

            # Define tasks
            # 1. Document Conversion (Files)
            task_docs = progress.add_task(
                "[cyan]Converting documents...",
                total=len(files),
            )

            # 2. LLM Processing (Dynamic total)
            # Initial total 1 to show 0/1 instead of 0/0, or 0 is fine
            task_llm = progress.add_task(
                "[cyan]LLM Processing...",
                total=0,
                visible=False,  # Hide until tasks are added
            )

            # 3. Finalization (Files)
            task_final = progress.add_task(
                "[cyan]Finalizing...",
                total=len(files),
            )

            # Callbacks for progress updates
            def on_phase1_start(file_path: Path) -> None:
                pass  # Already counted in total

            def on_phase1_complete(file_path: Path, result: DocumentConversionResult) -> None:
                progress.advance(task_docs)

            def on_phase1_error(file_path: Path, error: str) -> None:
                progress.advance(task_docs)
                rel_path = file_path.relative_to(input_dir)
                progress.console.print(f"  [red]x[/red] {rel_path} (conversion failed)")

            def on_phase2_start(file_path: Path, count: int) -> None:
                # Make LLM task visible if not already
                if not progress.tasks[task_llm].visible:
                    progress.update(task_llm, visible=True)

                # Add new tasks to total
                new_total = (progress.tasks[task_llm].total or 0) + count
                progress.update(task_llm, total=new_total)

            def on_phase2_complete(file_path: Path, results: list[Any]) -> None:
                progress.advance(task_llm, advance=len(results))

            def on_phase3_complete(file_path: Path, result: PipelineResult) -> None:
                progress.advance(task_final)
                rel_path = file_path.relative_to(input_dir)
                images_info = ""
                if result.images_count > 0:
                    images_info = f" [dim]({result.images_count} images)[/dim]"
                progress.console.print(f"  [green]✓[/green] {rel_path}{images_info}")

            def on_phase3_error(file_path: Path, error: str) -> None:
                progress.advance(task_final)
                rel_path = file_path.relative_to(input_dir)
                error_msg = _simplify_error(error)
                progress.console.print(f"  [red]x[/red] {rel_path}")
                progress.console.print(f"    [dim]{error_msg}[/dim]")

            def on_error(file_path: Path, error: str) -> None:
                # If total failure, ensure we advance all bars to not hang
                # (though strict accounting is hard here, mainly need to ensure completion)
                pass

            callbacks = {
                "on_phase1_start": on_phase1_start,
                "on_phase1_complete": on_phase1_complete,
                "on_phase1_error": on_phase1_error,
                "on_phase2_start": on_phase2_start,
                "on_phase2_complete": on_phase2_complete,
                "on_phase3_complete": on_phase3_complete,
                "on_phase3_error": on_phase3_error,
                "on_error": on_error,
            }

            # Start all pipelines concurrently
            tasks = [
                _process_file_pipeline(
                    f,
                    pipeline,
                    concurrency,
                    llm_queue,
                    output_dir,
                    input_dir,
                    state_manager,
                    callbacks,
                )
                for f in files
            ]

            results = await asyncio.gather(*tasks)

        set_log_output(original_stderr)
        # Restore WARNING level to avoid log clutter after progress bar
        # Note: File handler is already set up from initial setup_task_logging
        setup_logging(level="WARNING")

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

        config_suggestions = []

        for path, error in failed_files[:10]:
            # Extract just the filename from path
            filename = Path(path).name
            # Simplify error message - extract the core reason
            error_msg = error or "Unknown error"
            error_summary = _simplify_error(error_msg)

            # Check for configuration hints in error
            if "capabilities=['text']" in error_msg:
                # Extract provider name if possible or just generic hint
                suggestion = "  - Image analysis failed. Suggestion: Check 'capabilities' in markit.toml. Add capabilities=['text'] for text-only models."
                if suggestion not in config_suggestions:
                    config_suggestions.append(suggestion)

            console.print(f"  [dim]-[/dim] {filename}")
            console.print(f"    [dim]{error_summary}[/dim]")
        if len(failed_files) > 10:
            console.print(f"  [dim]... and {len(failed_files) - 10} more[/dim]")

    # Display configuration suggestions if any
    if failed_files and any("capabilities=['text']" in (f[1] or "") for f in failed_files):
        console.print()
        console.print("[bold yellow]Configuration Suggestions:[/bold yellow]")
        console.print(
            "  - Some image analysis tasks failed. If using text-only models (like DeepSeek),"
        )
        console.print(
            "    please explicitly set [bold]capabilities=['text'][/bold] in your markit.toml"
        )
        console.print("    to skip image analysis for those providers.")

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

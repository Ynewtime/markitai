"""Batch command for directory conversion."""

import asyncio
from pathlib import Path
from time import time
from typing import Annotated, Any, Literal

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from markit.cli.shared import ConversionContext, ConversionOptions
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
)
from markit.utils.stats import BatchStats

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
    fast: Annotated[
        bool,
        typer.Option(
            "--fast",
            help="Fast mode: skip validation, minimal retries, reduced logging.",
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
    # Pack shared options into dataclass
    options = ConversionOptions(
        output_dir=output,
        llm=llm,
        analyze_image=analyze_image,
        analyze_image_with_md=analyze_image_with_md,
        no_compress=no_compress,
        pdf_engine=pdf_engine,
        llm_provider=llm_provider,
        llm_model=llm_model,
        verbose=verbose,
        fast=fast,
        dry_run=dry_run,
    )

    # Create context with shared initialization (use cwd as base for relative output)
    ctx = ConversionContext.create(
        options, command_prefix="batch", console=console, base_path=Path.cwd()
    )

    # Batch-specific validation
    if on_conflict not in ("skip", "overwrite", "rename"):
        console.print(
            f"[red]Error:[/red] Invalid conflict strategy '{on_conflict}'. "
            "Options: skip, overwrite, rename"
        )
        raise typer.Exit(1)

    # Determine state file
    state_path = state_file or Path(ctx.settings.state_file)

    ctx.log_start(
        input_dir=str(input_dir),
        recursive=recursive,
    )

    if dry_run:
        _show_dry_run(
            input_dir=input_dir,
            output_dir=ctx.output_dir,
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
                output_dir=ctx.output_dir,
                recursive=recursive,
                include=include,
                exclude=exclude,
                file_concurrency=file_concurrency,
                image_concurrency=image_concurrency,
                llm_concurrency=llm_concurrency,
                on_conflict=on_conflict,
                resume=resume,
                state_path=state_path,
                ctx=ctx,
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
    on_conflict: Literal["skip", "overwrite", "rename"],
    resume: bool,
    state_path: Path,
    ctx: ConversionContext,
) -> None:
    """Execute the batch conversion asynchronously."""
    # Initialize batch statistics
    stats = BatchStats()
    init_start = time()

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
        state_manager.clear()

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
                "llm_enabled": ctx.options.llm,
                "analyze_image": ctx.options.effective_analyze_image,
                "compress_images": ctx.options.compress_images,
                "on_conflict": on_conflict,
            },
        )

    # Create pipeline and concurrency manager
    # Enable concurrent fallback in default mode (not fast mode) for better resilience
    use_concurrent_fallback = not ctx.options.fast
    pipeline = ctx.create_pipeline(use_concurrent_fallback=use_concurrent_fallback)

    concurrency = ConcurrencyManager(
        file_workers=file_concurrency,
        image_workers=image_concurrency,
        llm_workers=llm_concurrency,
    )

    # Update conflict strategy in settings
    ctx.settings.output.on_conflict = on_conflict

    # Warmup LLM providers if enabled (fail fast and avoid concurrency race)
    if ctx.options.llm or ctx.options.effective_analyze_image:
        if ctx.options.verbose:
            # In verbose mode, skip spinner to avoid mixing with log output
            await pipeline.warmup()
        else:
            with console.status("[cyan]Initializing LLM providers...[/cyan]"):
                await pipeline.warmup()

    # Track initialization time
    stats.init_duration = time() - init_start
    stats.total_files = len(files)

    # Choose processing mode based on LLM usage
    convert_start = time()
    if ctx.options.use_phased_pipeline:
        if ctx.options.verbose:
            results = await _process_files_phased_verbose(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
                llm_concurrency=llm_concurrency,
                stats=stats,
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
                stats=stats,
            )
    else:
        if ctx.options.verbose:
            results = await _process_files_verbose(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
                stats=stats,
            )
        else:
            results = await _process_files_with_progress(
                files=files,
                pipeline=pipeline,
                concurrency=concurrency,
                output_dir=output_dir,
                state_manager=state_manager,
                input_dir=input_dir,
                stats=stats,
            )

    # Track conversion time and finalize stats
    stats.convert_duration = time() - convert_start
    stats.finish()

    # Mark batch complete
    state_manager.mark_batch_complete()

    # Display summary
    _display_summary(results, state_manager, stats)


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
            for f in input_dir.glob(f"{pattern}{ext}"):
                if Path(f.name).match(include):
                    files.append(f)
        else:
            files.extend(input_dir.glob(f"{pattern}{ext}"))

    if exclude:
        files = [f for f in files if not Path(f.name).match(exclude)]

    files.sort(key=lambda p: str(p))
    return files


async def _process_files_verbose(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    stats: BatchStats | None = None,
) -> list[TaskResult[Path]]:
    """Process files with verbose logging (no progress bar)."""
    results: list[TaskResult[Path]] = []
    log.info("Processing files", total=len(files))

    async def process_file(file_path: Path) -> PipelineResult:
        rel_path = file_path.relative_to(input_dir)
        log.info("Converting file", file=str(rel_path))
        state_manager.update_file_status(file_path, "processing")

        try:
            result = await pipeline.convert_file_async(file_path, output_dir)

            if result.success:
                state_manager.update_file_status(
                    file_path, "completed", output_path=result.output_path
                )
                log.info(
                    "File converted successfully",
                    file=str(rel_path),
                    output=str(result.output_path),
                    images=result.images_count,
                )
                if stats:
                    stats.add_file_result(success=True)
            else:
                state_manager.update_file_status(file_path, "failed", error=result.error)
                log.error("File conversion failed", file=str(rel_path), error=result.error)
                if stats:
                    stats.add_file_result(success=False)

            return result
        except Exception as e:
            state_manager.update_file_status(file_path, "failed", error=str(e))
            log.error("File conversion error", file=str(rel_path), error=str(e))
            if stats:
                stats.add_file_result(success=False)
            raise

    def on_progress(
        item: Path,  # noqa: ARG001
        result: PipelineResult | None,  # noqa: ARG001
        error: Exception | None,  # noqa: ARG001
    ) -> None:
        pass

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
    stats: BatchStats | None = None,
) -> list[TaskResult[Path]]:
    """Process files with Rich progress display."""
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
        import os
        import sys

        original_stderr = sys.stderr

        with open(os.devnull, "w") as devnull:
            set_log_output(devnull)
            setup_logging(level="WARNING", console=progress.console)

            progress_task_id = progress.add_task("[cyan]Converting files...", total=len(files))

            async def process_file(file_path: Path) -> PipelineResult:
                state_manager.update_file_status(file_path, "processing")

                try:
                    result = await pipeline.convert_file_async(file_path, output_dir)

                    if result.success:
                        state_manager.update_file_status(
                            file_path, "completed", output_path=result.output_path
                        )
                        if stats:
                            stats.add_file_result(success=True)
                    else:
                        state_manager.update_file_status(file_path, "failed", error=result.error)
                        if stats:
                            stats.add_file_result(success=False)

                    return result
                except Exception as e:
                    state_manager.update_file_status(file_path, "failed", error=str(e))
                    if stats:
                        stats.add_file_result(success=False)
                    raise

            def on_progress(
                item: Path,
                result: PipelineResult | None,
                error: Exception | None,
            ) -> None:
                progress.advance(progress_task_id)
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
                    images_info = ""
                    if result.images_count > 0:
                        images_info = f" [dim]({result.images_count} images)[/dim]"
                    progress.console.print(f"  [green]✓[/green] {rel_path}{images_info}")

            task_results = await concurrency.map_file_tasks(
                items=files,
                func=process_file,
                on_progress=on_progress,
            )

            results.extend(task_results)

        set_log_output(original_stderr)
        setup_logging(level="WARNING")

    return results


async def _process_file_pipeline(
    file_path: Path,
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    llm_queue: LLMTaskQueue,
    output_dir: Path,
    input_dir: Path,  # noqa: ARG001
    state_manager: StateManager,
    callbacks: dict[str, Any],
    stats: BatchStats | None = None,
) -> TaskResult[Path]:
    """Process a single file through the full pipeline (streaming)."""
    from markit.image.analyzer import ImageAnalysis

    callbacks.get("on_phase1_start", lambda _: None)(file_path)
    state_manager.update_file_status(file_path, "processing")

    try:
        doc_result = await concurrency.run_file_task(
            pipeline.convert_document_only(file_path, output_dir)
        )

        if not doc_result.success:
            callbacks.get("on_phase1_error", lambda _p, _e: None)(file_path, doc_result.error)
            state_manager.update_file_status(file_path, "failed", error=doc_result.error)
            if stats:
                stats.add_file_result(success=False)
            return TaskResult(file_path, False, error=doc_result.error)

        callbacks.get("on_phase1_complete", lambda _p, _r: None)(file_path, doc_result)

        llm_tasks = await pipeline.create_llm_tasks(doc_result)
        submitted_tasks = []

        if llm_tasks:
            callbacks.get("on_phase2_start", lambda _p, _c: None)(file_path, len(llm_tasks))

            for i, coro in enumerate(llm_tasks):
                num_images = len(doc_result.images_for_analysis)
                if i < num_images:
                    task_type = "image_analysis"
                    task_id = doc_result.images_for_analysis[i].filename
                else:
                    task_type = "chunk_enhancement"
                    task_id = "markdown"

                task_obj = await llm_queue.submit(
                    LLMTask(
                        source_file=file_path,
                        task_type=task_type,  # type: ignore[arg-type]
                        task_id=task_id,
                        coro=coro,
                    )
                )
                submitted_tasks.append(task_obj)

            llm_results = await asyncio.gather(*submitted_tasks)
            callbacks.get("on_phase2_complete", lambda _p, _r: None)(file_path, llm_results)
        else:
            llm_results = []

        callbacks.get("on_phase3_start", lambda _: None)(file_path)

        image_analyses: list[ImageAnalysis] = []
        enhanced_markdown: str | None = None

        for llm_result in llm_results:
            # Collect LLM statistics
            if stats and llm_result.model:
                stats.add_llm_call(
                    model=llm_result.model,
                    prompt_tokens=llm_result.prompt_tokens,
                    completion_tokens=llm_result.completion_tokens,
                    cost=llm_result.estimated_cost,
                    duration=llm_result.duration,
                    start_time=llm_result.start_time,
                    end_time=llm_result.end_time,
                )

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
            callbacks.get("on_phase3_complete", lambda _p, _r: None)(file_path, pipeline_result)
            if stats:
                stats.add_file_result(success=True)
            return TaskResult(file_path, True, result=pipeline_result)
        else:
            state_manager.update_file_status(file_path, "failed", error=pipeline_result.error)
            callbacks.get("on_phase3_error", lambda _p, _e: None)(file_path, pipeline_result.error)
            if stats:
                stats.add_file_result(success=False)
            return TaskResult(file_path, False, error=pipeline_result.error)

    except Exception as e:
        callbacks.get("on_error", lambda _p, _e: None)(file_path, str(e))
        state_manager.update_file_status(file_path, "failed", error=str(e))
        if stats:
            stats.add_file_result(success=False)
        return TaskResult(file_path, False, error=str(e))


async def _process_files_phased_verbose(
    files: list[Path],
    pipeline: ConversionPipeline,
    concurrency: ConcurrencyManager,
    output_dir: Path,
    state_manager: StateManager,
    input_dir: Path,
    llm_concurrency: int = 10,
    stats: BatchStats | None = None,
) -> list[TaskResult[Path]]:
    """Process files using phased pipeline with verbose logging (streaming)."""
    llm_queue = LLMTaskQueue(max_concurrent=llm_concurrency, max_pending=100)
    log.info(
        "Starting streaming batch processing", total=len(files), llm_concurrency=llm_concurrency
    )

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
            stats=stats,
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
    stats: BatchStats | None = None,
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

            task_docs = progress.add_task("[cyan]Converting documents...", total=len(files))
            task_llm = progress.add_task("[cyan]LLM Processing...", total=0, visible=False)
            task_final = progress.add_task("[cyan]Finalizing...", total=len(files))

            def on_phase1_start(_file_path: Path) -> None:
                pass

            def on_phase1_complete(
                _file_path: Path,
                _result: DocumentConversionResult,  # noqa: ARG001
            ) -> None:
                progress.advance(task_docs)

            def on_phase1_error(file_path: Path, _error: str) -> None:  # noqa: ARG001
                progress.advance(task_docs)
                rel_path = file_path.relative_to(input_dir)
                progress.console.print(f"  [red]x[/red] {rel_path} (conversion failed)")

            def on_phase2_start(_file_path: Path, count: int) -> None:  # noqa: ARG001
                if not progress.tasks[task_llm].visible:
                    progress.update(task_llm, visible=True)
                new_total = (progress.tasks[task_llm].total or 0) + count
                progress.update(task_llm, total=new_total)

            def on_phase2_complete(_file_path: Path, results: list[Any]) -> None:  # noqa: ARG001
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

            def on_error(file_path: Path, error: str) -> None:  # noqa: ARG001
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
                    stats=stats,
                )
                for f in files
            ]

            results = await asyncio.gather(*tasks)

        set_log_output(original_stderr)
        setup_logging(level="WARNING")

    return results


def _display_summary(
    _results: list[TaskResult[Path]],
    state_manager: StateManager,
    stats: BatchStats | None = None,
) -> None:
    """Display batch processing summary."""
    state = state_manager.get_state()
    if state is None:
        return

    console.print()

    # Display enhanced stats if available
    if stats:
        console.print("[bold]Batch Statistics:[/bold]")
        console.print(stats.format_summary())
        console.print()
    else:
        # Fallback to basic display
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

    failed_files = [(path, fs.error) for path, fs in state.files.items() if fs.status == "failed"]

    if failed_files:
        console.print()
        console.print("[bold red]Failed Files:[/bold red]")

        for path, error in failed_files[:10]:
            filename = Path(path).name
            error_msg = error or "Unknown error"
            error_summary = _simplify_error(error_msg)

            console.print(f"  [dim]-[/dim] {filename}")
            console.print(f"    [dim]{error_summary}[/dim]")

        if len(failed_files) > 10:
            console.print(f"  [dim]... and {len(failed_files) - 10} more[/dim]")

    if failed_files and any("capabilities=['text']" in (f[1] or "") for f in failed_files):
        console.print()
        console.print("[bold yellow]Configuration Suggestions:[/bold yellow]")
        console.print(
            "  - Some image analysis tasks failed. If using text-only models (like DeepSeek),"
        )
        console.print(
            "    please explicitly set [bold]capabilities=['text'][/bold] in your markit.yaml"
        )
        console.print("    to skip image analysis for those providers.")

    console.print()


def _simplify_error(error: str) -> str:
    """Simplify error message for display."""
    if "MissingDependencyException" in error:
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

    files = _discover_files(input_dir, recursive, include, exclude)

    console.print()
    console.print(f"[bold]Files Found:[/bold] {len(files)}")

    if files:
        import contextlib

        by_ext: dict[str, list[Path]] = {}
        total_size = 0
        for f in files:
            ext = f.suffix.lower()
            by_ext.setdefault(ext, []).append(f)
            with contextlib.suppress(OSError):
                total_size += f.stat().st_size

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

        # Token and cost estimation
        estimates = _estimate_tokens_and_cost(files, by_ext, total_size)
        _display_token_estimates(estimates)

    console.print()


def _estimate_tokens_and_cost(
    files: list[Path],
    by_ext: dict[str, list[Path]],
    total_size: int,
) -> dict[str, Any]:
    """Estimate tokens and costs for dry-run display.

    Token estimation heuristics:
    - Text content: ~1 token per 4 bytes (UTF-8 mixed content)
    - System prompts: Fixed overhead per LLM call
    - Image analysis: ~1,500 tokens per image (vision model)
    - PDF images: Estimate 2-5 images per PDF file

    Args:
        files: List of files to process
        by_ext: Files grouped by extension
        total_size: Total file size in bytes

    Returns:
        Dictionary with estimation data
    """
    # Constants for estimation
    BYTES_PER_TOKEN = 4  # Average for mixed UTF-8 content
    ENHANCEMENT_PROMPT_TOKENS = 500  # System prompt overhead
    IMAGE_ANALYSIS_TOKENS = 1500  # Per image (vision model)
    IMAGES_PER_PDF = 3  # Average images per PDF

    # Default cost rates (USD per 1M tokens) - conservative estimates
    DEFAULT_INPUT_COST = 1.0  # $1 per 1M input tokens
    DEFAULT_OUTPUT_COST = 3.0  # $3 per 1M output tokens
    VISION_INPUT_COST = 2.5  # Vision models typically cost more

    # Count files by category
    pdf_count = len(by_ext.get(".pdf", []))
    doc_count = sum(
        len(by_ext.get(ext, [])) for ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]
    )
    text_count = len(files) - pdf_count

    # Estimate images (PDFs contain images)
    estimated_images = pdf_count * IMAGES_PER_PDF

    # Estimate tokens
    # Base content tokens (from file size)
    content_tokens = total_size // BYTES_PER_TOKEN

    # LLM enhancement tokens (content + prompts + output)
    enhancement_input = content_tokens + (len(files) * ENHANCEMENT_PROMPT_TOKENS)
    enhancement_output = content_tokens // 2  # Output is typically smaller

    # Image analysis tokens
    image_input = estimated_images * IMAGE_ANALYSIS_TOKENS
    image_output = estimated_images * 200  # ~200 tokens per image description

    # Calculate costs
    def calc_cost(
        input_tokens: int,
        output_tokens: int,
        input_rate: float = DEFAULT_INPUT_COST,
        output_rate: float = DEFAULT_OUTPUT_COST,
    ) -> float:
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    return {
        "total_size": total_size,
        "file_count": len(files),
        "pdf_count": pdf_count,
        "doc_count": doc_count,
        "text_count": text_count,
        "estimated_images": estimated_images,
        "content_tokens": content_tokens,
        "scenarios": {
            "convert_only": {
                "tokens": 0,
                "cost": 0.0,
                "description": "No LLM",
            },
            "llm_enhance": {
                "tokens": enhancement_input + enhancement_output,
                "input_tokens": enhancement_input,
                "output_tokens": enhancement_output,
                "cost": calc_cost(enhancement_input, enhancement_output),
                "description": "--llm",
            },
            "full_analysis": {
                "tokens": enhancement_input + enhancement_output + image_input + image_output,
                "input_tokens": enhancement_input + image_input,
                "output_tokens": enhancement_output + image_output,
                "cost": calc_cost(enhancement_input, enhancement_output)
                + calc_cost(image_input, image_output, VISION_INPUT_COST, DEFAULT_OUTPUT_COST),
                "description": "--llm --analyze-image",
            },
        },
    }


def _display_token_estimates(estimates: dict[str, Any]) -> None:
    """Display token and cost estimates."""
    console.print()
    console.print("[bold]Estimates:[/bold]")

    # File size
    size_mb = estimates["total_size"] / (1024 * 1024)
    console.print(f"  Total Size: {size_mb:.2f} MB")

    if estimates["estimated_images"] > 0:
        console.print(f"  Est. Images: ~{estimates['estimated_images']} (from PDFs)")

    # Token estimates by scenario
    console.print()
    console.print("[bold]Token & Cost Estimates:[/bold]")
    console.print("  [dim](Based on default rates: $1/1M input, $3/1M output)[/dim]")
    console.print()

    scenarios = estimates["scenarios"]

    # Convert only
    console.print(f"  [cyan]Convert Only[/cyan] ({scenarios['convert_only']['description']})")
    console.print("    Tokens: 0 | Cost: $0.00")

    # LLM enhance
    llm = scenarios["llm_enhance"]
    console.print(f"  [cyan]LLM Enhance[/cyan] ({llm['description']})")
    console.print(f"    Tokens: ~{llm['tokens']:,} | Est. Cost: ${llm['cost']:.2f}")

    # Full analysis
    if estimates["estimated_images"] > 0:
        full = scenarios["full_analysis"]
        console.print(f"  [cyan]Full Analysis[/cyan] ({full['description']})")
        console.print(f"    Tokens: ~{full['tokens']:,} | Est. Cost: ${full['cost']:.2f}")

    console.print()
    console.print("  [dim]Note: Actual usage depends on content complexity and model choice.[/dim]")

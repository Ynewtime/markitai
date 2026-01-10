"""Convert command for single file conversion."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from markit.config import MarkitSettings, get_settings
from markit.config.constants import LLM_PROVIDERS, PDF_ENGINES
from markit.utils.logging import get_logger, setup_task_logging

# Use a simple Typer instance - the command will be registered in main.py
console = Console()
log = get_logger(__name__)


def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input file path to convert.",
            exists=True,
            file_okay=True,
            dir_okay=False,
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
    llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help="Enable LLM Markdown format optimization (frontmatter, cleanup, GFM).",
        ),
    ] = False,
    analyze_image: Annotated[
        bool,
        typer.Option(
            "--analyze-image",
            help="Enable LLM image analysis for alt text generation.",
        ),
    ] = False,
    analyze_image_with_md: Annotated[
        bool,
        typer.Option(
            "--analyze-image-with-md",
            help="Enable LLM image analysis and generate .md description files for each image.",
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
            help="Show conversion plan without executing.",
        ),
    ] = False,
) -> None:
    """Convert a single document file to Markdown.

    Examples:
        markit convert document.docx
        markit convert document.docx -o ./output
        markit convert document.docx --llm --analyze-image
        markit convert report.pdf --pdf-engine pdfplumber
    """
    settings = get_settings()

    # Setup task-level logging with unified behavior
    # Priority: user config (settings.log_dir) > default (.logs)
    task_id, log_path = setup_task_logging(
        log_dir=settings.log_dir,
        prefix="convert",
        verbose=verbose,
    )

    # Log file behavior hint (only in verbose mode)
    if verbose:
        log.info("Logs will be saved to", log_file=str(log_path))

    # Feature 1: Log detailed configuration at the beginning
    # Mask API keys before logging
    config_dump = settings.model_dump()
    if "llm" in config_dump and "providers" in config_dump["llm"]:
        for provider in config_dump["llm"]["providers"]:
            if "api_key" in provider and provider["api_key"]:
                provider["api_key"] = "***"

    log.info("Task Configuration", task_id=task_id, config=config_dump)

    # Validate PDF engine if specified
    if pdf_engine and pdf_engine not in PDF_ENGINES:
        console.print(
            f"[red]Error:[/red] Invalid PDF engine '{pdf_engine}'. "
            f"Options: {', '.join(PDF_ENGINES)}"
        )
        raise typer.Exit(1)

    # Validate LLM provider if specified
    if llm_provider and llm_provider not in LLM_PROVIDERS:
        console.print(
            f"[red]Error:[/red] Invalid LLM provider '{llm_provider}'. "
            f"Options: {', '.join(LLM_PROVIDERS)}"
        )
        raise typer.Exit(1)

    # Determine output directory
    output_dir = output or Path(settings.output.default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If --analyze-image-with-md is set, it implies analyze_image
    effective_analyze_image = analyze_image or analyze_image_with_md

    log.info(
        "Starting conversion",
        input_file=str(input_file),
        output_dir=str(output_dir),
        llm_enabled=llm,
        analyze_image=effective_analyze_image,
        analyze_image_with_md=analyze_image_with_md,
    )

    if dry_run:
        _show_dry_run(
            input_file=input_file,
            output_dir=output_dir,
            llm=llm,
            analyze_image=effective_analyze_image,
            analyze_image_with_md=analyze_image_with_md,
            no_compress=no_compress,
            pdf_engine=pdf_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        return

    # Execute conversion with signal handling for interrupts
    import signal
    import sys
    from datetime import datetime as dt

    interrupted = False

    def signal_handler(signum: int, frame) -> None:  # noqa: ARG001
        """Handle interrupt signals and log before exit."""
        nonlocal interrupted
        if interrupted:
            # Force exit on second interrupt
            sys.exit(130)
        interrupted = True
        sig_name = signal.Signals(signum).name
        log.warning(
            "Task Interrupted",
            signal=sig_name,
            input_file=str(input_file),
            interrupted_at=dt.now().isoformat(),
        )
        console.print(f"\n[yellow]Interrupted by {sig_name}. Exiting...[/yellow]")
        raise typer.Exit(130)

    # Register signal handlers
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = None
    if hasattr(signal, "SIGTERM"):
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    try:
        _execute_conversion(
            input_file=input_file,
            output_dir=output_dir,
            llm=llm,
            analyze_image=effective_analyze_image,
            analyze_image_with_md=analyze_image_with_md,
            no_compress=no_compress,
            pdf_engine=pdf_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
            settings=settings,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        log.warning(
            "Task Interrupted by KeyboardInterrupt",
            input_file=str(input_file),
            interrupted_at=dt.now().isoformat(),
        )
        console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        log.error("Conversion failed", error=str(e), exc_info=True)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        if original_sigterm is not None:
            signal.signal(signal.SIGTERM, original_sigterm)


def _show_dry_run(
    input_file: Path,
    output_dir: Path,
    llm: bool,
    analyze_image: bool,
    analyze_image_with_md: bool,
    no_compress: bool,
    pdf_engine: str | None,
    llm_provider: str | None,
    llm_model: str | None,
) -> None:
    """Display the conversion plan without executing."""
    import asyncio

    from markit.cli.commands.provider import display_test_results, test_all_providers

    console.print("\n[bold blue]Conversion Plan (Dry Run)[/bold blue]\n")
    console.print(f"  [bold]Input:[/bold] {input_file}")
    console.print(f"  [bold]Output Directory:[/bold] {output_dir}")
    console.print(f"  [bold]File Type:[/bold] {input_file.suffix}")
    console.print()
    console.print("[bold]Options:[/bold]")
    console.print(f"  LLM Enhancement: {'Enabled' if llm else 'Disabled'}")
    console.print(f"  Image Analysis (alt text): {'Enabled' if analyze_image else 'Disabled'}")
    console.print(
        f"  Image Description Files (.md): {'Enabled' if analyze_image_with_md else 'Disabled'}"
    )
    console.print(f"  Image Compression: {'Disabled' if no_compress else 'Enabled'}")
    if pdf_engine:
        console.print(f"  PDF Engine: {pdf_engine}")
    if llm_provider:
        console.print(f"  LLM Provider: {llm_provider}")
    if llm_model:
        console.print(f"  LLM Model: {llm_model}")
    console.print()

    # Test LLM providers if LLM features are enabled
    if llm or analyze_image or analyze_image_with_md:
        console.print("[bold]LLM Provider Connectivity Test:[/bold]")
        results = asyncio.run(test_all_providers(show_progress=True))
        console.print()
        display_test_results(results)

        # Check if any provider failed
        failed = [r for r in results if r.status == "failed"]
        if failed:
            console.print(
                f"\n[yellow]Warning:[/yellow] {len(failed)} provider(s) failed connectivity test."
            )
        console.print()


def _execute_conversion(
    input_file: Path,
    output_dir: Path,
    llm: bool,
    analyze_image: bool,
    analyze_image_with_md: bool,
    no_compress: bool,
    pdf_engine: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    settings: MarkitSettings,
    verbose: bool = False,
) -> None:
    """Execute the actual conversion."""
    import logging

    from markit.core.pipeline import ConversionPipeline

    # Create pipeline
    pipeline = ConversionPipeline(
        settings=settings,
        llm_enabled=llm,
        analyze_image=analyze_image,
        analyze_image_with_md=analyze_image_with_md,
        compress_images=not no_compress,
        pdf_engine=pdf_engine,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    if verbose:
        # In verbose mode, just run conversion without spinner
        # This keeps log output clean and readable
        result = pipeline.convert_file(input_file, output_dir)
    else:
        # In non-verbose mode, show a spinner for user feedback
        # Temporarily disable console logging to avoid interference with spinner
        # File logging (if configured) still works
        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        original_levels = [(h, h.level) for h in console_handlers]

        # Set console handlers to CRITICAL to suppress most output during spinner
        for handler in console_handlers:
            handler.setLevel(logging.CRITICAL + 1)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Converting...", total=None)
                result = pipeline.convert_file(input_file, output_dir)
        finally:
            # Restore original handler levels
            for handler, level in original_levels:
                handler.setLevel(level)

    # Display results
    if result.success:
        log.info(
            "Task Completed Successfully",
            output_path=str(result.output_path),
            images_count=result.images_count,
            metadata=result.metadata,
        )
        console.print("[bold green]Conversion completed![/bold green]")
        console.print(f"  Output: {result.output_path}")
        if result.images_count > 0:
            console.print(f"  Images: {result.images_count}")
    else:
        log.error("Task Failed", error=result.error)
        console.print()
        console.print(f"[bold red]Conversion failed:[/bold red] {result.error}")
        raise typer.Exit(1)

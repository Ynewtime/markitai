"""Convert command for single file conversion."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from markit.cli.shared import ConversionContext, ConversionOptions, execute_single_file
from markit.config.constants import LLM_PROVIDERS, PDF_ENGINES

console = Console()


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
    fast: Annotated[
        bool,
        typer.Option(
            "--fast",
            help="Fast mode: skip validation, minimal retries.",
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

    # Create context with shared initialization
    ctx = ConversionContext.create(options, command_prefix="convert", console=console)

    # Execute with shared logic
    execute_single_file(
        input_file=input_file,
        ctx=ctx,
        on_dry_run=lambda: _show_dry_run(input_file, ctx),
    )


def _show_dry_run(input_file: Path, ctx: ConversionContext) -> None:
    """Display the conversion plan without executing."""
    from markit.cli.commands.provider import display_test_results, test_all_providers

    console.print("\n[bold blue]Conversion Plan (Dry Run)[/bold blue]\n")
    console.print(f"  [bold]Input:[/bold] {input_file}")
    console.print(f"  [bold]Output Directory:[/bold] {ctx.output_dir}")
    console.print(f"  [bold]File Type:[/bold] {input_file.suffix}")
    console.print()
    console.print("[bold]Options:[/bold]")
    console.print(f"  LLM Enhancement: {'Enabled' if ctx.options.llm else 'Disabled'}")
    console.print(
        f"  Image Analysis (alt text): {'Enabled' if ctx.options.effective_analyze_image else 'Disabled'}"
    )
    console.print(
        f"  Image Description Files (.md): {'Enabled' if ctx.options.analyze_image_with_md else 'Disabled'}"
    )
    console.print(f"  Image Compression: {'Disabled' if ctx.options.no_compress else 'Enabled'}")
    if ctx.options.pdf_engine:
        console.print(f"  PDF Engine: {ctx.options.pdf_engine}")
    if ctx.options.llm_provider:
        console.print(f"  LLM Provider: {ctx.options.llm_provider}")
    if ctx.options.llm_model:
        console.print(f"  LLM Model: {ctx.options.llm_model}")
    console.print()

    # Test LLM providers if LLM features are enabled
    if ctx.options.llm or ctx.options.effective_analyze_image:
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

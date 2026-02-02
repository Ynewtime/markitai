"""Single file processing for CLI.

This module contains the process_single_file function for
converting individual documents to Markdown.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.panel import Panel

from markitai.cli.console import get_console
from markitai.config import MarkitaiConfig
from markitai.constants import MAX_DOCUMENT_SIZE
from markitai.converter import FileFormat, detect_format
from markitai.json_order import order_report
from markitai.security import atomic_write_json, validate_file_size
from markitai.utils.cli_helpers import compute_task_hash, get_report_file_path
from markitai.utils.progress import ProgressReporter
from markitai.workflow.helpers import write_images_json

console = get_console()


async def process_single_file(
    input_path: Path,
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    log_file_path: Path | None = None,
    verbose: bool = False,
) -> None:
    """Process a single file using workflow/core pipeline.

    After conversion completes, outputs the final markdown to stdout.
    If LLM is enabled, outputs .llm.md content; otherwise outputs .md content.

    Args:
        input_path: Path to the input file.
        output_dir: Directory to write output files.
        cfg: Configuration object.
        dry_run: If True, only show what would be done.
        log_file_path: Path to log file for report.
        verbose: Enable verbose output.
    """
    from markitai.workflow.core import ConversionContext, convert_document_core

    # Validate file size to prevent DoS
    try:
        validate_file_size(input_path, MAX_DOCUMENT_SIZE)
    except ValueError as e:
        console.print(Panel(f"[red]{e}[/red]", title="Error"))
        raise SystemExit(1)

    # Detect file format for dry-run display
    fmt = detect_format(input_path)
    if fmt == FileFormat.UNKNOWN:
        console.print(
            Panel(
                f"[red]Unsupported file format: {input_path.suffix}[/red]",
                title="Error",
            )
        )
        raise SystemExit(1)

    # Handle dry-run
    if dry_run:
        # Build feature status indicators
        features = []
        if cfg.llm.enabled:
            features.append("[green]LLM[/green]")
        if cfg.image.alt_enabled:
            features.append("[green]alt[/green]")
        if cfg.image.desc_enabled:
            features.append("[green]desc[/green]")
        if cfg.ocr.enabled:
            features.append("[green]OCR[/green]")
        if cfg.screenshot.enabled:
            features.append("[green]screenshot[/green]")

        feature_str = " ".join(features) if features else "[dim]none[/dim]"
        cache_status = "enabled" if cfg.cache.enabled else "disabled"

        dry_run_msg = (
            f"[yellow]Would convert:[/yellow] {input_path}\n"
            f"[yellow]Format:[/yellow] {fmt.value.upper()}\n"
            f"[yellow]Output:[/yellow] {output_dir / (input_path.name + '.md')}\n"
            f"[yellow]Features:[/yellow] {feature_str}\n"
            f"[yellow]Cache:[/yellow] {cache_status}"
        )
        console.print(Panel(dry_run_msg, title="Dry Run"))
        if cfg.cache.enabled:
            console.print(
                "[dim]Tip: Use 'markitai cache stats -v' to view cached entries[/dim]"
            )
        raise SystemExit(0)

    # Progress reporter for non-verbose mode feedback
    progress = ProgressReporter(enabled=not verbose)
    started_at = datetime.now()
    error_msg = None

    try:
        progress.start_spinner(f"Converting {input_path.name}...")

        # Create conversion context
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=output_dir,
            config=cfg,
        )

        # Run core conversion pipeline
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if not result.success:
            if result.error:
                raise RuntimeError(result.error)
            raise RuntimeError("Unknown conversion error")

        if result.skip_reason == "exists":
            progress.stop_spinner()
            base_output_file = output_dir / f"{input_path.name}.md"
            console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # Show conversion complete message
        progress.log(f"Converted: {input_path.name}")

        # Write image descriptions (single file)
        if ctx.image_analysis and cfg.image.desc_enabled:
            write_images_json(output_dir, [ctx.image_analysis])

        # Generate report
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        input_tokens = sum(u.get("input_tokens", 0) for u in ctx.llm_usage.values())
        output_tokens = sum(u.get("output_tokens", 0) for u in ctx.llm_usage.values())
        requests = sum(u.get("requests", 0) for u in ctx.llm_usage.values())

        report = {
            "version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(log_file_path) if log_file_path else None,
            "summary": {
                "total_documents": 1,
                "completed_documents": 1,
                "failed_documents": 0,
                "duration": duration,
            },
            "llm_usage": {
                "models": ctx.llm_usage,
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": ctx.llm_cost,
            },
            "documents": {
                input_path.name: {
                    "status": "completed",
                    "error": None,
                    "output": str(
                        ctx.output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled and ctx.output_file
                        else ctx.output_file
                    ),
                    "images": ctx.embedded_images_count,
                    "screenshots": ctx.screenshots_count,
                    "duration": duration,
                    "llm_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": ctx.llm_cost,
                    },
                }
            },
        }

        # Generate report file path
        task_options = {
            "llm": cfg.llm.enabled,
            "ocr": cfg.ocr.enabled,
            "screenshot": cfg.screenshot.enabled,
            "alt": cfg.image.alt_enabled,
            "desc": cfg.image.desc_enabled,
        }
        task_hash = compute_task_hash(input_path, output_dir, task_options)
        report_path = get_report_file_path(
            output_dir, task_hash, cfg.output.on_conflict
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(report_path, report, order_func=order_report)
        logger.info(f"Report saved: {report_path}")

        # Clear progress output before printing final result
        progress.clear_and_finish()

        # Output final markdown to stdout
        if ctx.output_file:
            final_output_file = (
                ctx.output_file.with_suffix(".llm.md")
                if cfg.llm.enabled
                else ctx.output_file
            )
            # Fallback to .md file if .llm.md doesn't exist (e.g., LLM processing failed)
            if not final_output_file.exists() and cfg.llm.enabled:
                final_output_file = ctx.output_file
            if final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")
                # Use console.print with markup=False to handle Unicode correctly on Windows
                console.print(final_content, markup=False, highlight=False)

    except Exception as e:
        error_msg = str(e)
        console.print(Panel(f"[red]{error_msg}[/red]", title="Error"))
        sys.exit(1)

    finally:
        if error_msg:
            logger.warning(f"Failed to process {input_path.name}: {error_msg}")

"""Single file processing for CLI.

This module contains the process_single_file function for
converting individual documents to Markdown.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.ui import MARK_INFO, MARK_TITLE
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
    output_dir: Path | None,
    cfg: MarkitaiConfig,
    dry_run: bool,
    log_file_path: Path | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Process a single file with layered output.

    Output behavior:
    - No output_dir: Print markdown content to stdout
    - With output_dir: Save file, print path (e.g., "output/file.md")
    - With output_dir + verbose: Show detailed progress
    - With quiet: No output at all

    Args:
        input_path: Path to the input file.
        output_dir: Directory to write output files. If None, output to stdout.
        cfg: Configuration object.
        dry_run: If True, only show what would be done.
        log_file_path: Path to log file for report.
        verbose: Enable verbose output.
        quiet: Suppress all output.
    """
    from markitai.workflow.core import ConversionContext, convert_document_core

    # Validate file size to prevent DoS
    try:
        validate_file_size(input_path, MAX_DOCUMENT_SIZE)
    except ValueError as e:
        if not quiet:
            ui.error(str(e))
        raise SystemExit(1)

    # Detect file format for dry-run display
    fmt = detect_format(input_path)
    if fmt == FileFormat.UNKNOWN:
        if not quiet:
            ui.error(f"Unsupported file format: {input_path.suffix}")
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

        # Determine output location for display
        if output_dir:
            output_display = str(output_dir / (input_path.name + ".md"))
        else:
            output_display = "stdout"

        # Unified UI dry-run display
        console.print(f"[cyan]{MARK_TITLE}[/] [bold]Dry Run[/]\n")
        console.print("  Files (1)")
        console.print(f"    [dim]{MARK_INFO}[/] {input_path.name}")
        console.print()
        console.print("  ─────────────────")
        console.print(f"  Output: {output_display}")
        console.print(f"  Format: {fmt.value.upper()}")
        console.print(f"  Features: {feature_str}")
        console.print(f"  Cache: {cache_status}")
        raise SystemExit(0)

    # Track timing
    start_time = time.time()
    error_msg = None

    # Determine output mode
    # - No output_dir: stdout mode (output content)
    # - With output_dir: file mode (save and show path)
    stdout_mode = output_dir is None

    # For stdout mode, use a temporary directory for conversion
    if stdout_mode:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="markitai_"))
        effective_output_dir = temp_dir
    else:
        effective_output_dir = output_dir

    # Progress spinner: only show when saving to file and not quiet/verbose
    show_spinner = not stdout_mode and not quiet and not verbose
    progress = ProgressReporter(enabled=show_spinner)

    try:
        # Show verbose title
        if verbose and not quiet and not stdout_mode:
            ui.title(f"Converting {input_path.name}")

        progress.start_spinner(f"Converting {input_path.name}...")

        # Create conversion context
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=effective_output_dir,
            config=cfg,
        )

        # Run core conversion pipeline
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if not result.success:
            if result.error:
                raise RuntimeError(result.error)
            raise RuntimeError("Unknown conversion error")

        # Handle skipped files (already exists)
        if result.skip_reason == "exists":
            progress.stop_spinner()
            if not quiet and not stdout_mode:
                assert output_dir is not None
                base_output_file = output_dir / f"{input_path.name}.md"
                console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # Stop spinner before output
        progress.stop_spinner()

        # Write image descriptions (single file)
        if ctx.image_analysis and cfg.image.desc_enabled:
            write_images_json(effective_output_dir, [ctx.image_analysis])

        # Calculate duration
        duration = time.time() - start_time
        finished_at = datetime.now()

        # Generate report (only when saving to file)
        if not stdout_mode:
            assert output_dir is not None
            input_tokens = sum(u.get("input_tokens", 0) for u in ctx.llm_usage.values())
            output_tokens = sum(
                u.get("output_tokens", 0) for u in ctx.llm_usage.values()
            )
            requests = sum(u.get("requests", 0) for u in ctx.llm_usage.values())

            report = {
                "version": "1.0",
                "generated_at": finished_at.astimezone().isoformat(),
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
            logger.debug(f"Report saved: {report_path}")

        # Determine final output file
        final_output_file = None
        if ctx.output_file:
            final_output_file = (
                ctx.output_file.with_suffix(".llm.md")
                if cfg.llm.enabled
                else ctx.output_file
            )
            # Fallback to .md file if .llm.md doesn't exist
            if not final_output_file.exists() and cfg.llm.enabled:
                final_output_file = ctx.output_file

        # Output based on mode
        if quiet:
            # Quiet mode: no output
            pass
        elif stdout_mode:
            # stdout mode: output markdown content
            if final_output_file and final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")
                console.print(final_content, markup=False, highlight=False)
        else:
            # File mode: show output path
            if verbose:
                # Verbose: show detailed steps
                console.print(f"  {ui.MARK_SUCCESS} Parsed ({duration:.1f}s)")
                if ctx.embedded_images_count > 0:
                    console.print(
                        f"  {ui.MARK_SUCCESS} Images: {ctx.embedded_images_count} "
                        f"extracted"
                    )
                if ctx.screenshots_count > 0:
                    console.print(
                        f"  {ui.MARK_SUCCESS} Screenshots: {ctx.screenshots_count} "
                        f"captured"
                    )
                console.print()

            # Always show output path when saving to file
            if final_output_file:
                duration_str = f" ({duration:.1f}s)" if verbose else ""
                ui.success(f"{final_output_file}{duration_str}")

    except Exception as e:
        error_msg = str(e)
        progress.stop_spinner()
        if not quiet:
            ui.error(error_msg)
        sys.exit(1)

    finally:
        if error_msg:
            logger.warning(f"Failed to process {input_path.name}: {error_msg}")
        # Cleanup temp directory for stdout mode
        if stdout_mode:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

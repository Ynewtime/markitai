"""Single file processing for CLI.

This module contains the process_single_file function for
converting individual documents to Markdown.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.cli import ui
from markitai.cli.console import get_console, get_stderr_console
from markitai.cli.processors import split_output_file_target
from markitai.cli.ui import MARK_INFO, MARK_TITLE
from markitai.config import MarkitaiConfig
from markitai.constants import MAX_DOCUMENT_SIZE
from markitai.converter import FileFormat, detect_format
from markitai.json_order import order_report
from markitai.security import atomic_write_json, validate_file_size
from markitai.utils.cli_helpers import compute_task_hash, get_report_file_path
from markitai.utils.paths import derive_output_name
from markitai.utils.text import format_error_message
from markitai.workflow.helpers import write_images_json

console = get_console()

# Pattern matches markdown image references to .markitai/assets/ or .markitai/screenshots/
# Supports both forward slash and backslash for Windows compatibility
# Groups: 1=alt text, 2=subdir (assets|screenshots), 3=filename
_ASSET_REF_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(\.markitai[/\\](assets|screenshots)[/\\]([^)]+)\)"
)


def _warn_ephemeral_links() -> None:
    """Warn on stderr that stdout image links will not outlive the process.

    Used when ``image.stdout_persist`` is explicitly disabled. Writes to
    stderr directly because the stdout-mode console handler only shows
    ERROR+, and stdout itself must stay clean for the markdown content.
    """
    message = (
        "Warning: image.stdout_persist is disabled — extracted images are "
        "deleted at exit, so image links in the output are ephemeral"
    )
    print(message, file=sys.stderr)
    logger.warning(message)


def normalize_temp_asset_refs(markdown: str, temp_dir: Path) -> str:
    """Rewrite absolute temp-dir asset refs to relative ``.markitai/`` refs.

    Some converters emit image refs with absolute paths into the stdout-mode
    temp directory (e.g. pymupdf4llm canonicalizes macOS ``/var/...`` temp
    paths to ``/private/var/...``, defeating the converter's own relative
    rewrite). Normalizing here lets ``resolve_asset_references`` handle them
    instead of leaking dead temp links to stdout.

    Args:
        markdown: Markdown content possibly containing absolute asset refs.
        temp_dir: The stdout-mode temp directory used for conversion.

    Returns:
        Markdown with absolute temp-dir asset refs rewritten to relative form.
    """
    for base in {temp_dir.as_posix(), temp_dir.resolve().as_posix()}:
        markdown = markdown.replace(f"]({base}/.markitai/", "](.markitai/")
    return markdown


def resolve_asset_references(
    markdown: str,
    temp_dir: Path,
    protocol: Any = None,
    asset_store: Any = None,
    source_name: str = "unknown",
) -> str:
    """Resolve .markitai/assets/ and .markitai/screenshots/ image references.

    Priority cascade:
    1. If protocol is set: replace with terminal inline image escape sequence.
    2. If asset_store is set: persist image, replace with absolute-path URI.
    3. Fallback: replace with ``![image: filename]()`` placeholder.

    Args:
        markdown: Markdown content with asset references.
        temp_dir: Path to the temp directory containing .markitai/ assets.
        protocol: Detected terminal image protocol, or None.
        asset_store: Configured asset store, or None.
        source_name: Source document name for asset store grouping.

    Returns:
        Markdown with asset references resolved.
    """

    def _resolve_image_path(subdir: str, filename: str) -> Path:
        """Resolve the actual image file path from captured regex groups."""
        filename_normalized = filename.replace("\\", "/")
        return temp_dir / ".markitai" / subdir / filename_normalized

    def _replace(match: re.Match[str]) -> str:
        subdir = match.group(2)  # "assets" or "screenshots" — captured group
        filename = match.group(3)

        if protocol is not None:
            # Tier 1: terminal inline image
            image_path = _resolve_image_path(subdir, filename)
            if image_path.exists():
                from markitai.utils.terminal_image import render_inline_image

                try:
                    return render_inline_image(image_path, protocol)
                except Exception:
                    pass  # fall through

        if asset_store is not None:
            # Tier 2: persistent asset store
            image_path = _resolve_image_path(subdir, filename)
            if image_path.exists():
                try:
                    ref_path = asset_store.save(image_path, source_name)
                    uri = asset_store.ref_path_to_markdown_uri(ref_path)
                    return f"![{filename}]({uri})"
                except Exception as e:
                    logger.warning(f"Asset store save failed for {filename}: {e}")
                    # fall through to placeholder

        # Tier 3: placeholder fallback
        return f"![image: {filename}]()"

    return _ASSET_REF_PATTERN.sub(_replace, markdown)


def _file_final_stage_text(active_key: str | None, input_name: str) -> str:
    """Completion text for the last active stage of a file conversion."""
    if active_key == "llm":
        return "LLM enhanced"
    if active_key == "images":
        return "Images analyzed"
    return f"Converted {input_name}"


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

    # `-o out.md` (single-file mode) targets an output FILE, not a directory:
    # the parent becomes the output dir and the name overrides derived naming
    output_file_name: str | None = None
    if output_dir is not None:
        output_dir, output_file_name = split_output_file_target(output_dir)

    # Determine output mode early so diagnostics can be routed:
    # in stdout mode the markdown content owns stdout, so warnings/errors
    # must go to stderr to keep piped output clean
    stdout_mode = output_dir is None
    diag_console = get_stderr_console() if stdout_mode else console

    # Validate file size to prevent DoS
    try:
        validate_file_size(input_path, MAX_DOCUMENT_SIZE)
    except ValueError as e:
        if not quiet:
            ui.error(str(e), console=diag_console)
        raise SystemExit(1)

    # Detect file format for dry-run display
    fmt = detect_format(input_path)
    if fmt == FileFormat.UNKNOWN:
        if not quiet:
            ui.error(
                f"Unsupported file format: {input_path.suffix}", console=diag_console
            )
        raise SystemExit(1)

    # Handle dry-run
    if dry_run:
        feature_str = ui.build_feature_str(cfg)
        cache_status = "enabled" if cfg.cache.enabled else "disabled"

        # Determine output location for display
        if output_dir is not None:
            planned_name = output_file_name or derive_output_name(input_path.name)
            output_display = str(output_dir / planned_name)
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

    # Output mode (stdout_mode computed above):
    # - No output_dir: stdout mode (output content)
    # - With output_dir: file mode (save and show path)

    # For stdout mode, use a temporary directory for conversion
    if stdout_mode:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="markitai_"))
        effective_output_dir = temp_dir
    else:
        effective_output_dir = output_dir

    # Live multi-stage progress list (stderr). Enabled in BOTH file and
    # stdout modes; suppressed for --quiet and -v/verbose. The stage text
    # follows conversion logs via the loguru bridge (see cli.ui.StageList) —
    # convert_document_core needs no changes: its "[LLM]"/"Analyzing" logs
    # advance the list automatically.
    stages = ui.StageList(
        enabled=not quiet and not verbose,
        transient=stdout_mode,
    )

    try:
        # Show verbose title
        if verbose and not quiet and not stdout_mode:
            ui.title(f"Converting {input_path.name}")

        stages.start()
        stages.advance("convert", f"Converting {input_path.name}...")

        # Create conversion context (explicit -o file target overrides the
        # derived output name)
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=effective_output_dir,
            config=cfg,
            output_name=output_file_name,
        )

        # Run core conversion pipeline
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if not result.success:
            if result.error:
                raise RuntimeError(result.error)
            raise RuntimeError("Unknown conversion error")

        # Handle skipped files (already exists)
        if result.skip_reason == "exists":
            stages.stop()
            if not quiet and not stdout_mode:
                assert output_dir is not None
                planned_name = output_file_name or derive_output_name(input_path.name)
                base_output_file = output_dir / planned_name
                console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # Handle skipped files (image-only format, no LLM/OCR)
        if result.skip_reason == "image_only":
            stages.stop()
            if not quiet:
                ui.warning(
                    f"Skipped {input_path.name} (image file, no text to extract). "
                    f"Use --llm or --ocr for content extraction.",
                    console=diag_console,
                )
            return

        # Finalize the last active stage (the loguru bridge may have advanced
        # it past "convert") and stop the list before printing the result
        # (transient in stdout mode; rich erases its frame)
        stages.finalize(_file_final_stage_text(stages.active_key, input_path.name))
        stages.stop()

        # Write image descriptions (single file)
        if ctx.image_analysis and cfg.image.desc_enabled:
            write_images_json(effective_output_dir, [ctx.image_analysis])

        # Calculate duration
        duration = time.time() - start_time
        finished_at = datetime.now()

        # Generate report only when explicitly enabled (output.report=true);
        # single-file conversions skip reports by default
        if not stdout_mode and cfg.output.report is True:
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

            # Explicit -o file target: the final content must land exactly at
            # the requested path. In LLM mode (without --keep-base) the final
            # file is `<name>.llm.md`; move it onto the requested `.md` path.
            if (
                output_file_name is not None
                and final_output_file != ctx.output_file
                and final_output_file.exists()
                and not ctx.output_file.exists()
            ):
                final_output_file.replace(ctx.output_file)
                final_output_file = ctx.output_file

        # Output based on mode
        if quiet:
            # Quiet mode: no output
            pass
        elif stdout_mode:
            # stdout mode: output markdown content
            if final_output_file and final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")

                # Rewrite absolute temp-dir refs so they resolve below
                final_content = normalize_temp_asset_refs(final_content, temp_dir)

                # Detect terminal image protocol (only if stdout is a TTY)
                from markitai.utils.terminal_image import detect_protocol

                protocol = detect_protocol()

                # Set up asset store for image persistence (default on so
                # stdout links outlive the temp dir; opt out via
                # image.stdout_persist=false)
                store = None
                if cfg.image.stdout_persist:
                    from markitai.utils.asset_store import AssetStore

                    try:
                        store = AssetStore(Path(cfg.image.stdout_persist_dir))
                    except Exception as e:
                        logger.warning(f"Asset store init failed: {e}")
                elif _ASSET_REF_PATTERN.search(final_content):
                    # Console handler drops WARNING in stdout mode, so write
                    # to stderr directly (stdout stays clean for the content)
                    _warn_ephemeral_links()

                # Resolve image references (protocol > persist > placeholder)
                final_content = resolve_asset_references(
                    final_content,
                    temp_dir=temp_dir,
                    protocol=protocol,
                    asset_store=store,
                    source_name=input_path.name,
                )

                # Always write content raw: Rich's console.print hard-wraps
                # at terminal width, breaking long URLs/lines mid-token
                sys.stdout.write(final_content)
                if not final_content.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
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
        error_msg = format_error_message(e)
        stages.fail()
        stages.stop()
        if not quiet:
            ui.error(error_msg, console=diag_console)
        sys.exit(1)

    finally:
        # Safety net: ensure the stage list is stopped on every exit path
        stages.stop()
        if error_msg:
            logger.warning(f"Failed to process {input_path.name}: {error_msg}")
        # Cleanup temp directory for stdout mode
        if stdout_mode:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

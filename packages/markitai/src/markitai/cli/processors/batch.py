"""Batch processing for CLI.

This module contains functions for batch processing of files and URLs.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.panel import Panel

from markitai.cli.console import get_console
from markitai.config import MarkitaiConfig
from markitai.constants import MAX_DOCUMENT_SIZE
from markitai.converter.base import EXTENSION_MAP
from markitai.security import atomic_write_text
from markitai.utils.cli_helpers import url_to_filename
from markitai.utils.output import resolve_output_path
from markitai.utils.paths import ensure_dir, ensure_screenshots_dir
from markitai.utils.text import format_error_message
from markitai.workflow.helpers import (
    add_basic_frontmatter as _add_basic_frontmatter,
)
from markitai.workflow.helpers import (
    create_llm_processor,
    write_images_json,
)
from markitai.workflow.helpers import (
    merge_llm_usage as _merge_llm_usage,
)

if TYPE_CHECKING:
    from markitai.fetch import FetchStrategy
    from markitai.llm import LLMProcessor

console = get_console()


def create_process_file(
    cfg: MarkitaiConfig,
    input_dir: Path,
    output_dir: Path,
    preconverted_map: dict[Path, Path],
    shared_processor: LLMProcessor | None,
) -> Callable:
    """Create a process_file function using workflow/core pipeline.

    This factory function creates a closure that captures the batch processing
    context for conversion.

    Args:
        cfg: Markitai configuration
        input_dir: Input directory for relative path calculation
        output_dir: Output directory
        preconverted_map: Map of pre-converted legacy Office files
        shared_processor: Shared LLM processor for batch mode

    Returns:
        An async function that processes a single file and returns ProcessResult
    """
    from markitai.batch import ProcessResult
    from markitai.workflow.core import ConversionContext, convert_document_core

    async def process_file(file_path: Path) -> ProcessResult:
        """Process a single file using workflow/core pipeline."""
        import time

        start_time = time.perf_counter()
        logger.info(f"[START] {file_path.name}")

        try:
            # Calculate relative path to preserve directory structure
            try:
                rel_path = file_path.parent.relative_to(input_dir)
                file_output_dir = output_dir / rel_path
            except ValueError:
                file_output_dir = output_dir

            # Create conversion context
            ctx = ConversionContext(
                input_path=file_path,
                output_dir=file_output_dir,
                config=cfg,
                actual_file=preconverted_map.get(file_path),
                shared_processor=shared_processor,
                use_multiprocess_images=True,
                input_base_dir=input_dir,
            )

            # Run core conversion pipeline
            result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

            total_time = time.perf_counter() - start_time

            if not result.success:
                logger.error(
                    f"[FAIL] {file_path.name}: {result.error} ({total_time:.2f}s)"
                )
                return ProcessResult(success=False, error=result.error)

            if result.skip_reason == "exists":
                logger.info(
                    f"[SKIP] Output exists: {file_output_dir / f'{file_path.name}.md'}"
                )
                return ProcessResult(
                    success=True,
                    output_path=str(file_output_dir / f"{file_path.name}.md"),
                    error="skipped (exists)",
                )

            # Determine cache hit
            cache_hit = cfg.llm.enabled and not ctx.llm_usage

            logger.info(
                f"[DONE] {file_path.name}: {total_time:.2f}s "
                f"(images={ctx.embedded_images_count}, screenshots={ctx.screenshots_count}, cost=${ctx.llm_cost:.4f})"
                + (" [cache]" if cache_hit else "")
            )

            return ProcessResult(
                success=True,
                output_path=str(
                    ctx.output_file.with_suffix(".llm.md")
                    if cfg.llm.enabled and ctx.output_file
                    else ctx.output_file
                ),
                images=ctx.embedded_images_count,
                screenshots=ctx.screenshots_count,
                cost_usd=ctx.llm_cost,
                llm_usage=ctx.llm_usage,
                image_analysis_result=ctx.image_analysis,
                cache_hit=cache_hit,
            )

        except Exception as e:
            total_time = time.perf_counter() - start_time
            err_msg = format_error_message(e)
            logger.error(f"[FAIL] {file_path.name}: {err_msg} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=err_msg)

    return process_file


def create_url_processor(
    cfg: MarkitaiConfig,
    output_dir: Path,
    fetch_strategy: FetchStrategy | None,
    explicit_fetch_strategy: bool,
    shared_processor: LLMProcessor | None = None,
    renderer: Any | None = None,
) -> Callable:
    """Create a URL processing function for batch processing.

    Args:
        cfg: Configuration
        output_dir: Output directory
        fetch_strategy: Fetch strategy to use
        explicit_fetch_strategy: Whether strategy was explicitly specified
        shared_processor: Optional shared LLMProcessor
        renderer: Optional shared PlaywrightRenderer

    Returns:
        Async function that processes a single URL and returns ProcessResult
    """
    from markitai import fetch as fetch_module
    from markitai.batch import ProcessResult
    from markitai.cli.processors.llm import (
        analyze_images_with_llm,
        process_with_llm,
    )
    from markitai.cli.processors.url import (
        build_multi_source_content,
        process_url_with_vision,
    )
    from markitai.fetch import (
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        get_fetch_cache,
    )
    from markitai.image import download_url_images

    # Determine fetch strategy (use config default if not specified)
    _fetch_strategy = fetch_strategy
    if _fetch_strategy is None:
        _fetch_strategy = FetchStrategy(cfg.fetch.strategy)

    # Initialize fetch cache for URL processing
    url_fetch_cache = None
    if cfg.cache.enabled:
        url_cache_dir = Path(cfg.cache.global_dir).expanduser()
        url_fetch_cache = get_fetch_cache(url_cache_dir, cfg.cache.max_size_bytes)

    # Prepare screenshot directory if enabled
    url_screenshot_dir = (
        ensure_screenshots_dir(output_dir) if cfg.screenshot.enabled else None
    )

    async def process_url(
        url: str,
        source_file: Path,
        custom_name: str | None = None,
    ) -> tuple[ProcessResult, dict[str, Any]]:
        """Process a single URL.

        Args:
            url: URL to process
            source_file: Path to the .urls file containing this URL
            custom_name: Optional custom output name

        Returns:
            Tuple of (ProcessResult, extra_info dict with fetch_strategy)
        """
        import time

        start_time = time.perf_counter()
        extra_info: dict[str, Any] = {
            "fetch_strategy": "unknown",
        }

        try:
            # Generate filename
            if custom_name:
                filename = f"{custom_name}.md"
            else:
                filename = url_to_filename(url)

            logger.info(f"[URL] Processing: {url} (strategy: {_fetch_strategy.value})")

            # Fetch URL using the configured strategy
            try:
                fetch_result = await fetch_module.fetch_url(
                    url,
                    _fetch_strategy,
                    cfg.fetch,
                    explicit_strategy=explicit_fetch_strategy,
                    cache=url_fetch_cache,
                    skip_read_cache=cfg.cache.no_cache,
                    screenshot=cfg.screenshot.enabled,
                    screenshot_dir=url_screenshot_dir,
                    screenshot_config=cfg.screenshot
                    if cfg.screenshot.enabled
                    else None,
                    renderer=renderer,
                )
                extra_info["fetch_strategy"] = fetch_result.strategy_used
                original_markdown = fetch_result.content
                screenshot_path = fetch_result.screenshot_path
                cache_status = " [cache]" if fetch_result.cache_hit else ""
                logger.debug(
                    f"[URL] Fetched via {fetch_result.strategy_used}{cache_status}: {url}"
                )
            except JinaRateLimitError:
                logger.error(f"[URL] Jina rate limit exceeded for: {url}")
                return ProcessResult(
                    success=False,
                    error="Jina Reader rate limit exceeded (20 RPM)",
                ), extra_info
            except FetchError as e:
                err_msg = format_error_message(e)
                logger.error(f"[URL] Fetch failed {url}: {err_msg}")
                return ProcessResult(success=False, error=err_msg), extra_info

            if not original_markdown.strip():
                logger.warning(f"[URL] No content: {url}")
                return ProcessResult(
                    success=False,
                    error="No content extracted",
                ), extra_info

            markdown_for_llm = original_markdown

            # Check for multi-source content (static + browser + screenshot)
            has_multi_source = (
                fetch_result.static_content is not None
                or fetch_result.browser_content is not None
            )
            has_screenshot = screenshot_path and screenshot_path.exists()

            logger.debug(
                f"[URL] Multi-source check: static={fetch_result.static_content is not None}, "
                f"browser={fetch_result.browser_content is not None}, "
                f"has_multi_source={has_multi_source}, has_screenshot={has_screenshot}"
            )

            # Download images if --alt or --desc is enabled
            images_count = 0
            screenshots_count = 1 if has_screenshot else 0
            downloaded_images: list[Path] = []

            if has_screenshot and screenshot_path:
                logger.debug(f"[URL] Screenshot captured: {screenshot_path.name}")
            if cfg.image.alt_enabled or cfg.image.desc_enabled:
                download_result = await download_url_images(
                    markdown=original_markdown,
                    output_dir=output_dir,
                    base_url=url,
                    config=cfg.image,
                    source_name=filename.replace(".md", ""),
                    concurrency=5,
                    timeout=30,
                )
                markdown_for_llm = download_result.updated_markdown
                downloaded_images = download_result.downloaded_paths
                images_count = len(downloaded_images)

            # Generate output path
            base_output_file = output_dir / filename
            output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

            if output_file is None:
                logger.info(f"[URL] Skipped (exists): {base_output_file}")
                return ProcessResult(
                    success=True,
                    output_path=str(base_output_file),
                    error="skipped (exists)",
                ), extra_info

            # Write base .md file with original content
            base_content = _add_basic_frontmatter(
                original_markdown,
                url,
                fetch_strategy=fetch_result.strategy_used if fetch_result else None,
                screenshot_path=screenshot_path,
                output_dir=output_dir,
                title=fetch_result.title if fetch_result else None,
            )
            atomic_write_text(output_file, base_content)

            # LLM processing uses markdown with local image paths
            url_llm_usage: dict[str, dict[str, Any]] = {}
            llm_cost = 0.0
            img_analysis = None

            if cfg.llm.enabled:
                # Check if image analysis should run
                should_analyze_images = (
                    cfg.image.alt_enabled or cfg.image.desc_enabled
                ) and downloaded_images

                # Check if we should use vision enhancement (multi-source + screenshot)
                use_vision_enhancement = (
                    has_multi_source and has_screenshot and screenshot_path
                )

                if use_vision_enhancement:
                    # Multi-source URL with screenshot: use vision LLM for better content extraction
                    # Build multi-source markdown content for LLM
                    multi_source_content = build_multi_source_content(
                        fetch_result.static_content,
                        fetch_result.browser_content,
                        markdown_for_llm,  # Fallback primary content
                    )

                    logger.info(
                        f"[URL] Using vision enhancement for multi-source URL: {url}"
                    )

                    # Use vision enhancement with screenshot
                    assert (
                        screenshot_path is not None
                    )  # Guaranteed by use_vision_enhancement check
                    _, cost, url_llm_usage = await process_url_with_vision(
                        multi_source_content,
                        screenshot_path,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                        original_title=fetch_result.title if fetch_result else None,
                    )
                    llm_cost = cost

                    # Run image analysis in parallel if needed
                    if should_analyze_images:
                        (
                            _,
                            image_cost,
                            image_usage,
                            img_analysis,
                        ) = await analyze_images_with_llm(
                            downloaded_images,
                            multi_source_content,
                            output_file,
                            cfg,
                            Path(url),
                            concurrency_limit=cfg.llm.concurrency,
                            processor=shared_processor,
                        )
                        _merge_llm_usage(url_llm_usage, image_usage)
                        llm_cost += image_cost
                elif should_analyze_images:
                    # Standard processing with image analysis
                    doc_task = process_with_llm(
                        markdown_for_llm,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                    )
                    img_task = analyze_images_with_llm(
                        downloaded_images,
                        markdown_for_llm,
                        output_file,
                        cfg,
                        Path(url),  # Use URL as source path
                        concurrency_limit=cfg.llm.concurrency,
                        processor=shared_processor,
                    )

                    # Execute in parallel
                    doc_result, img_result = await asyncio.gather(doc_task, img_task)

                    # Unpack results
                    _, cost, url_llm_usage = doc_result
                    _, image_cost, image_usage, img_analysis = img_result

                    _merge_llm_usage(url_llm_usage, image_usage)
                    llm_cost = cost + image_cost
                else:
                    # Only document processing
                    _, cost, url_llm_usage = await process_with_llm(
                        markdown_for_llm,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                    )
                    llm_cost = cost

            # Track cache hit: LLM enabled but no usage means cache hit
            is_cache_hit = cfg.llm.enabled and not url_llm_usage

            total_time = time.perf_counter() - start_time
            logger.info(
                f"[URL] Completed via {extra_info['fetch_strategy']}: {url} "
                f"({total_time:.2f}s)" + (" [cache]" if is_cache_hit else "")
            )

            return ProcessResult(
                success=True,
                output_path=str(
                    output_file.with_suffix(".llm.md")
                    if cfg.llm.enabled
                    else output_file
                ),
                images=images_count,
                screenshots=screenshots_count,
                cost_usd=llm_cost,
                llm_usage=url_llm_usage,
                image_analysis_result=img_analysis,
                cache_hit=is_cache_hit,
            ), extra_info

        except Exception as e:
            total_time = time.perf_counter() - start_time
            err_msg = format_error_message(e)
            logger.error(f"[URL] Failed {url}: {err_msg} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=err_msg), extra_info

    return process_url


async def process_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: MarkitaiConfig,
    resume: bool,
    dry_run: bool,
    verbose: bool = False,
    console_handler_id: int | None = None,
    log_file_path: Path | None = None,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Process directory in batch mode."""
    from datetime import datetime

    from markitai.batch import BatchProcessor, FileStatus, UrlState
    from markitai.cli.processors.validators import (
        check_playwright_for_urls,
        warn_case_sensitivity_mismatches,
    )
    from markitai.security import check_symlink_safety
    from markitai.urls import find_url_list_files, parse_url_list

    # Supported extensions
    extensions = set(EXTENSION_MAP.keys())

    # Build task options for report (before BatchProcessor init for hash calculation)
    # Note: input_dir and output_dir will be converted to absolute paths by init_state()
    task_options: dict[str, Any] = {
        "concurrency": cfg.batch.concurrency,
        "llm": cfg.llm.enabled,
        "ocr": cfg.ocr.enabled,
        "screenshot": cfg.screenshot.enabled,
        "alt": cfg.image.alt_enabled,
        "desc": cfg.image.desc_enabled,
    }
    if cfg.llm.enabled and cfg.llm.model_list:
        task_options["models"] = [m.litellm_params.model for m in cfg.llm.model_list]

    batch = BatchProcessor(
        cfg.batch,
        output_dir,
        input_path=input_dir,
        log_file=log_file_path,
        on_conflict=cfg.output.on_conflict,
        task_options=task_options,
    )
    files = batch.discover_files(input_dir, extensions)

    # Discover .urls files for URL batch processing
    url_list_files = find_url_list_files(input_dir)
    url_entries_from_files: list = []  # List of (source_file, UrlEntry)

    for url_file in url_list_files:
        try:
            entries = parse_url_list(url_file)
            for entry in entries:
                url_entries_from_files.append((url_file, entry))
            if entries:
                logger.info(f"Found {len(entries)} URLs in {url_file.name}")
        except Exception as e:
            logger.warning(f"Failed to parse URL list {url_file}: {e}")

    # Check Playwright availability if URLs will be processed
    if url_entries_from_files:
        check_playwright_for_urls(cfg, console)

    if not files and not url_entries_from_files:
        console.print("[yellow]No supported files or URL lists found.[/yellow]")
        raise SystemExit(0)

    # Warn about potential case-sensitivity mismatches in --no-cache-for patterns
    if cfg.cache.no_cache_patterns:
        warn_case_sensitivity_mismatches(files, input_dir, cfg.cache.no_cache_patterns)

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

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

        # Build dry run message
        dry_run_msg = f"[yellow]Would process {len(files)} files[/yellow]"
        if url_entries_from_files:
            dry_run_msg += f"\n[yellow]Would process {len(url_entries_from_files)} URLs from {len(url_list_files)} .urls files[/yellow]"
        dry_run_msg += f"\n[yellow]Input:[/yellow] {input_dir}\n[yellow]Output:[/yellow] {output_dir}"
        dry_run_msg += f"\n[yellow]Features:[/yellow] {feature_str}"
        dry_run_msg += f"\n[yellow]Cache:[/yellow] {cache_status}"

        console.print(Panel(dry_run_msg, title="Dry Run"))
        for f in files[:10]:
            console.print(f"  - {f.name}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more files")
        if url_entries_from_files:
            console.print("[dim]URL list files:[/dim]")
            for url_file in url_list_files[:5]:
                console.print(f"  - {url_file.name}")
            if len(url_list_files) > 5:
                console.print(f"  ... and {len(url_list_files) - 5} more .urls files")
        if cfg.cache.enabled:
            console.print(
                "[dim]Tip: Use 'markitai cache stats -v' to view cached entries[/dim]"
            )
        raise SystemExit(0)

    # Record batch start time before any processing (including pre-conversion)
    batch_started_at = datetime.now().astimezone().isoformat()

    # Start Live display early to capture all logs (including URL processing)
    # This ensures all INFO+ logs go to the panel instead of console
    batch.start_live_display(
        verbose=verbose,
        console_handler_id=console_handler_id,
        total_files=len(files),
        total_urls=len(url_entries_from_files),
    )

    # Pre-convert legacy Office files using batch COM (Windows only)
    # This reduces overhead by starting each Office app only once
    legacy_suffixes = {".doc", ".ppt", ".xls"}
    legacy_files = [f for f in files if f.suffix.lower() in legacy_suffixes]
    preconverted_map: dict[Path, Path] = {}
    preconvert_temp_dir: tempfile.TemporaryDirectory | None = None

    if legacy_files:
        import platform

        if platform.system() == "Windows":
            from markitai.converter.legacy import batch_convert_legacy_files

            # Create temp directory for pre-converted files
            preconvert_temp_dir = tempfile.TemporaryDirectory(
                prefix="markitai_preconv_"
            )
            preconvert_path = Path(preconvert_temp_dir.name)

            logger.info(f"Pre-converting {len(legacy_files)} legacy files...")
            preconverted_map = batch_convert_legacy_files(legacy_files, preconvert_path)
            if preconverted_map:
                logger.info(
                    f"Pre-converted {len(preconverted_map)}/{len(legacy_files)} files with MS Office COM"
                )

    # Create shared LLM runtime and processor for batch mode
    shared_processor = None
    if cfg.llm.enabled:
        from markitai.llm import LLMRuntime

        runtime = LLMRuntime(concurrency=cfg.llm.concurrency)
        shared_processor = create_llm_processor(cfg, runtime=runtime)
        logger.info(
            f"Created shared LLMProcessor with concurrency={cfg.llm.concurrency}"
        )

    # Create shared Playwright renderer for batch URL processing
    shared_renderer = None
    if url_entries_from_files:
        from markitai.fetch import _detect_proxy, _get_playwright_renderer

        # Only initialize if browser strategy might be needed
        # We initialize it here to reuse across all URLs in the batch
        proxy = _detect_proxy() if getattr(cfg.fetch, "auto_proxy", True) else None
        shared_renderer = await _get_playwright_renderer(proxy=proxy)
        logger.info("Created shared PlaywrightRenderer for batch URL processing")

    # Create process_file using workflow/core implementation
    process_file = create_process_file(
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        preconverted_map=preconverted_map,
        shared_processor=shared_processor,
    )
    logger.debug("Using workflow/core implementation for batch processing")

    # Group URL entries by source file and collect source file list
    url_sources_set: set[str] = set()
    if url_entries_from_files:
        for source_file, _entry in url_entries_from_files:
            url_sources_set.add(str(source_file))

    # Initialize batch state with files
    if files or url_entries_from_files:
        batch.state = batch.init_state(
            input_dir=input_dir,
            files=files,
            options=task_options,
            started_at=batch_started_at,
        )
        # Add URL source files to state
        batch.state.url_sources = list(url_sources_set)

        # Initialize URL states in batch state
        for source_file, entry in url_entries_from_files:
            batch.state.urls[entry.url] = UrlState(
                url=entry.url,
                source_file=str(source_file),
                status=FileStatus.PENDING,
            )

    # Create URL processor function
    url_processor = None
    if url_entries_from_files:
        url_processor = create_url_processor(
            cfg=cfg,
            output_dir=output_dir,
            fetch_strategy=fetch_strategy,
            explicit_fetch_strategy=explicit_fetch_strategy,
            shared_processor=shared_processor,
            renderer=shared_renderer,
        )

    # Create separate semaphores for file and URL processing
    # This allows file processing and URL fetching to run at their own concurrency levels
    file_semaphore = asyncio.Semaphore(cfg.batch.concurrency)
    url_semaphore = asyncio.Semaphore(cfg.batch.url_concurrency)

    async def process_url_with_state(
        url: str,
        source_file: Path,
        custom_name: str | None,
    ) -> None:
        """Process a URL and update batch state."""
        assert batch.state is not None
        assert url_processor is not None

        url_state = batch.state.urls.get(url)
        if url_state is None:
            return

        # Update state to in_progress
        url_state.status = FileStatus.IN_PROGRESS
        url_state.started_at = datetime.now().astimezone().isoformat()

        start_time = asyncio.get_event_loop().time()

        try:
            async with url_semaphore:
                result, extra_info = await url_processor(url, source_file, custom_name)

            if result.success:
                url_state.status = FileStatus.COMPLETED
                url_state.output = result.output_path
                url_state.fetch_strategy = extra_info.get("fetch_strategy")
                url_state.images = result.images
                url_state.cost_usd = result.cost_usd
                url_state.llm_usage = result.llm_usage
                url_state.cache_hit = result.cache_hit
                # Collect image analysis for JSON output
                if result.image_analysis_result is not None:
                    batch.image_analysis_results.append(result.image_analysis_result)
            else:
                url_state.status = FileStatus.FAILED
                url_state.error = result.error

        except Exception as e:
            err_msg = format_error_message(e)
            url_state.status = FileStatus.FAILED
            url_state.error = err_msg
            logger.error(f"[URL] Failed {url}: {err_msg}")

        finally:
            end_time = asyncio.get_event_loop().time()
            url_state.completed_at = datetime.now().astimezone().isoformat()
            url_state.duration = end_time - start_time

            # Update progress
            batch.update_url_status(url, completed=True)

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    async def process_file_with_state(file_path: Path) -> None:
        """Process a file and update batch state."""
        assert batch.state is not None

        file_key = str(file_path)
        file_state = batch.state.files.get(file_key)

        if file_state is None:
            return

        # Update state to in_progress
        file_state.status = FileStatus.IN_PROGRESS
        file_state.started_at = datetime.now().astimezone().isoformat()

        start_time = asyncio.get_event_loop().time()

        try:
            async with file_semaphore:
                result = await process_file(file_path)

            if result.success:
                file_state.status = FileStatus.COMPLETED
                file_state.output = result.output_path
                file_state.images = result.images
                file_state.screenshots = result.screenshots
                file_state.cost_usd = result.cost_usd
                file_state.llm_usage = result.llm_usage
                file_state.cache_hit = result.cache_hit
                # Collect image analysis for JSON output
                if result.image_analysis_result is not None:
                    batch.image_analysis_results.append(result.image_analysis_result)
            else:
                file_state.status = FileStatus.FAILED
                file_state.error = result.error

        except Exception as e:
            file_state.status = FileStatus.FAILED
            err_msg = format_error_message(e)
            file_state.error = err_msg
            logger.error(f"[FAIL] {file_path.name}: {err_msg}")

        finally:
            end_time = asyncio.get_event_loop().time()
            file_state.completed_at = datetime.now().astimezone().isoformat()
            file_state.duration = end_time - start_time

            # Update progress
            batch.advance_progress()

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    # Run all tasks in parallel (URLs + files)
    state = batch.state
    try:
        if files or url_entries_from_files:
            # Build task list
            all_tasks = []

            # Add URL tasks
            for source_file, entry in url_entries_from_files:
                all_tasks.append(
                    process_url_with_state(entry.url, source_file, entry.output_name)
                )

            # Add file tasks
            for file_path in files:
                all_tasks.append(process_file_with_state(file_path))

            if all_tasks:
                logger.info(
                    f"Processing {len(files)} files and {len(url_entries_from_files)} URLs "
                    f"with concurrency {cfg.batch.concurrency}"
                )

                # Run all tasks in parallel
                await asyncio.gather(*all_tasks, return_exceptions=True)

    finally:
        # Stop Live display and restore console handler
        # This must be done before printing summary
        batch.stop_live_display()

        # Clean up pre-conversion temp directory
        if preconvert_temp_dir is not None:
            preconvert_temp_dir.cleanup()

    if state:
        # Update state timestamp
        state.updated_at = datetime.now().astimezone().isoformat()
        batch.save_state(force=True)

        # Print summary (uses state for URL stats)
        batch.print_summary(
            url_completed=state.completed_urls_count,
            url_failed=state.failed_urls_count,
            url_cache_hits=sum(
                1
                for u in state.urls.values()
                if u.status == FileStatus.COMPLETED and u.cache_hit
            ),
            url_sources=len(state.url_sources),
        )

        # Write aggregated image analysis JSON (if any)
        if batch.image_analysis_results and cfg.image.desc_enabled:
            write_images_json(output_dir, batch.image_analysis_results)

        # Save report (logging is done inside save_report)
        batch.save_report()

    # Exit with appropriate code
    total_failed = (state.failed_count if state else 0) + (
        state.failed_urls_count if state else 0
    )
    if total_failed > 0:
        raise SystemExit(10)  # PARTIAL_FAILURE


# Backward compatibility aliases
_create_process_file = create_process_file
_create_url_processor = create_url_processor

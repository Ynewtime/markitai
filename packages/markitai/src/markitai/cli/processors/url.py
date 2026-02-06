"""URL processing for CLI.

This module contains functions for fetching and processing URLs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.config import MarkitaiConfig
from markitai.json_order import order_report
from markitai.security import atomic_write_json, atomic_write_text
from markitai.utils.cli_helpers import (
    compute_task_hash,
    get_report_file_path,
    url_to_filename,
)
from markitai.utils.output import resolve_output_path
from markitai.utils.paths import ensure_dir, ensure_screenshots_dir
from markitai.utils.progress import ProgressReporter
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
from markitai.workflow.single import ImageAnalysisResult

if TYPE_CHECKING:
    from markitai.fetch import FetchCache, FetchStrategy
    from markitai.llm import LLMProcessor

console = get_console()


async def process_url(
    url: str,
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    verbose: bool,
    log_file_path: Path | None = None,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Process a URL and convert to Markdown.

    Supports multiple fetch strategies:
    - auto: Detect JS-required pages and fallback automatically
    - static: Direct HTTP request via markitdown (fastest)
    - playwright: Headless browser via Playwright (for JS-rendered pages)
    - jina: Jina Reader API (cloud-based, no local dependencies)

    Also supports:
    - LLM enhancement via --llm flag for document cleaning and frontmatter
    - Image downloading and analysis via --alt/--desc flags

    Note: --screenshot and --ocr are not supported for URLs.

    Args:
        url: URL to convert (http:// or https://)
        output_dir: Output directory for the markdown file
        cfg: Configuration
        dry_run: If True, only show what would be done
        verbose: If True, print logs before output
        log_file_path: Path to log file (for report)
        fetch_strategy: Strategy to use for fetching URL content
        explicit_fetch_strategy: If True, strategy was explicitly set via CLI flag
    """
    from markitai.cli.processors.llm import (
        analyze_images_with_llm,
        process_with_llm,
    )
    from markitai.fetch import (
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        fetch_url,
    )
    from markitai.image import download_url_images

    # Default to auto strategy if not specified
    if fetch_strategy is None:
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
    # At this point fetch_strategy is guaranteed to be non-None
    assert fetch_strategy is not None  # for type checker

    # Warn about unsupported/ignored options for URL mode
    # Note: --alt and --desc are now supported (images will be downloaded)
    # --screenshot is now supported for URLs (captures full-page screenshot via browser)
    # --ocr is not applicable for URLs
    if cfg.ocr.enabled:
        logger.warning("[URL] --ocr is not supported for URL conversion, ignored")

    # Generate output filename from URL
    filename = url_to_filename(url)

    if dry_run:
        # Build feature status indicators
        features = []
        if cfg.llm.enabled:
            features.append("[green]LLM[/green]")
        if cfg.image.alt_enabled:
            features.append("[green]alt[/green]")
        if cfg.image.desc_enabled:
            features.append("[green]desc[/green]")
        if cfg.screenshot.enabled:
            features.append("[green]screenshot[/green]")

        feature_str = " ".join(features) if features else "[dim]none[/dim]"
        cache_status = "enabled" if cfg.cache.enabled else "disabled"
        fetch_strategy_str = fetch_strategy.value if fetch_strategy else "auto"

        path_max = max(ui.term_width(console) - 10, 20)

        ui.title("Dry Run")
        console.print(f"  URL: {ui.truncate(url, path_max)}")
        console.print(f"  Output: {ui.truncate(str(output_dir / filename), path_max)}")
        console.print(f"  Fetch strategy: {fetch_strategy_str}")
        console.print(f"  Features: {feature_str}")
        console.print(f"  Cache: {cache_status}")
        if cfg.cache.enabled:
            ui.step("Tip: Use 'markitai cache stats -v' to view cached entries")
        raise SystemExit(0)

    # Create output directory
    from markitai.security import check_symlink_safety

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    from datetime import datetime

    started_at = datetime.now()
    llm_cost = 0.0
    llm_usage: dict[str, dict[str, Any]] = {}

    # Progress reporter for non-verbose mode feedback
    progress = ProgressReporter(enabled=not verbose)

    # Track cache hit for reporting
    fetch_cache_hit = False

    # Initialize fetch cache if caching is enabled
    fetch_cache: FetchCache | None = None
    if cfg.cache.enabled:
        from markitai.fetch import get_fetch_cache

        cache_dir = Path(cfg.cache.global_dir).expanduser()
        fetch_cache = get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)

    try:
        logger.info(f"Fetching URL: {url} (strategy: {fetch_strategy.value})")
        progress.start_spinner(f"Fetching {url}...")

        # Fetch URL using the configured strategy
        # Prepare screenshot options if enabled
        screenshot_dir = (
            ensure_screenshots_dir(output_dir) if cfg.screenshot.enabled else None
        )

        try:
            fetch_result = await fetch_url(
                url,
                fetch_strategy,
                cfg.fetch,
                explicit_strategy=explicit_fetch_strategy,
                cache=fetch_cache,
                skip_read_cache=cfg.cache.no_cache,
                screenshot=cfg.screenshot.enabled,
                screenshot_dir=screenshot_dir,
                screenshot_config=cfg.screenshot if cfg.screenshot.enabled else None,
            )
            fetch_cache_hit = fetch_result.cache_hit
            used_strategy = fetch_result.strategy_used
            original_markdown = fetch_result.content
            screenshot_path = fetch_result.screenshot_path
            logger.info(f"Fetched via {used_strategy}: {url}")
        except JinaRateLimitError:
            ui.error("Jina Reader rate limit exceeded (free tier: 20 RPM)")
            ui.step("Try again later or use --playwright for local rendering")
            raise SystemExit(1)
        except FetchError as e:
            ui.error(str(e))
            raise SystemExit(1)

        if not original_markdown.strip():
            ui.error(f"No content extracted from URL: {url}")
            ui.step(
                "The page may be empty, require JavaScript, "
                "or use an unsupported format."
            )
            raise SystemExit(1)

        # Generate output path with conflict resolution
        base_output_file = output_dir / filename
        output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

        if output_file is None:
            logger.info(f"[SKIP] Output exists: {base_output_file}")
            console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # original_markdown was already set from fetch_result.content above
        markdown_for_llm = original_markdown
        progress.log(f"Fetched via {used_strategy}: {url}")

        # Download images from URLs if --alt or --desc is enabled
        # Only update markdown_for_llm, keep original_markdown unchanged
        downloaded_images: list[Path] = []
        images_count = 0
        screenshots_count = 1 if screenshot_path and screenshot_path.exists() else 0
        img_analysis: ImageAnalysisResult | None = None

        # Log screenshot capture if successful
        if screenshot_path and screenshot_path.exists():
            progress.log(f"Screenshot captured: {screenshot_path.name}")
            logger.info(f"Screenshot saved: {screenshot_path}")

        if cfg.image.alt_enabled or cfg.image.desc_enabled:
            progress.start_spinner("Downloading images...")
            download_result = await download_url_images(
                markdown=original_markdown,
                output_dir=output_dir,
                base_url=url,
                config=cfg.image,
                source_name=url_to_filename(url).replace(".md", ""),
                concurrency=5,
                timeout=30,
            )
            markdown_for_llm = download_result.updated_markdown
            downloaded_images = download_result.downloaded_paths
            images_count = len(downloaded_images)

            if download_result.failed_urls:
                for failed_url in download_result.failed_urls:
                    logger.warning(f"Failed to download image: {failed_url}")

            if downloaded_images:
                progress.log(f"Downloaded {len(downloaded_images)} images")
            else:
                progress.log("No images to download")

        # Check for screenshot-only mode without LLM
        # --screenshot-only without --llm: just save screenshot, no .md output
        has_screenshot = screenshot_path is not None and screenshot_path.exists()
        if cfg.screenshot.screenshot_only and not cfg.llm.enabled:
            if has_screenshot and screenshot_path is not None:
                progress.log(f"Screenshot saved: {screenshot_path.name}")
                console.print(f"[green]Screenshot saved:[/green] {screenshot_path}")
            else:
                console.print(
                    "[yellow]Warning: --screenshot-only but no screenshot captured[/yellow]"
                )
            return

        # Write base .md file
        # For --llm --screenshot-only: .md contains just screenshot reference
        # Otherwise: .md contains original markdown content
        if (
            cfg.screenshot.screenshot_only
            and cfg.llm.enabled
            and has_screenshot
            and screenshot_path is not None
        ):
            # .md file just references the screenshot (not as HTML comment)
            screenshot_ref = f"![Screenshot](screenshots/{screenshot_path.name})"
            base_content = _add_basic_frontmatter(
                screenshot_ref,
                url,
                fetch_strategy=used_strategy,
                screenshot_path=None,  # Don't add screenshot again
                output_dir=output_dir,
                title=fetch_result.title,
            )
        else:
            base_content = _add_basic_frontmatter(
                original_markdown,
                url,
                fetch_strategy=used_strategy,
                screenshot_path=screenshot_path,
                output_dir=output_dir,
                title=fetch_result.title,
            )
        atomic_write_text(output_file, base_content)
        logger.info(f"Written output: {output_file}")

        # LLM processing (if enabled) uses markdown with local image paths
        final_content = base_content
        if cfg.llm.enabled:
            logger.info(f"[LLM] Processing URL content: {url}")

            # Check if image analysis should run
            should_analyze_images = (
                cfg.image.alt_enabled or cfg.image.desc_enabled
            ) and downloaded_images

            # Check for screenshot-only mode (extract purely from screenshot)
            # has_screenshot is already defined above
            use_screenshot_only = cfg.screenshot.screenshot_only and has_screenshot

            if use_screenshot_only and screenshot_path:
                # Screenshot-only mode: extract content purely from screenshot
                progress.start_spinner("Extracting content from screenshot...")

                _, doc_cost, doc_usage = await process_url_screenshot_only(
                    screenshot_path,
                    url,
                    cfg,
                    output_file,
                    original_title=fetch_result.title,
                )
                llm_cost += doc_cost
                _merge_llm_usage(llm_usage, doc_usage)

                # Run image analysis if needed
                if should_analyze_images:
                    (
                        _,
                        image_cost,
                        image_usage,
                        img_analysis,
                    ) = await analyze_images_with_llm(
                        downloaded_images,
                        "",  # No source content in screenshot-only mode
                        output_file,
                        cfg,
                        Path(url),
                        concurrency_limit=cfg.llm.concurrency,
                    )
                    llm_cost += image_cost
                    _merge_llm_usage(llm_usage, image_usage)
                progress.log("LLM processing complete (screenshot-only)")

            # Check for multi-source content (static + browser + screenshot)
            elif has_screenshot:
                has_multi_source = (
                    fetch_result.static_content is not None
                    or fetch_result.browser_content is not None
                )
                use_vision_enhancement = has_multi_source

                if use_vision_enhancement and screenshot_path:
                    # Multi-source URL with screenshot: use vision LLM
                    progress.start_spinner(
                        "Processing with Vision LLM (multi-source)..."
                    )
                    multi_source_content = build_multi_source_content(
                        fetch_result.static_content,
                        fetch_result.browser_content,
                        markdown_for_llm,
                    )

                    _, doc_cost, doc_usage = await process_url_with_vision(
                        multi_source_content,
                        screenshot_path,
                        url,
                        cfg,
                        output_file,
                        original_title=fetch_result.title,
                    )
                    llm_cost += doc_cost
                    _merge_llm_usage(llm_usage, doc_usage)

                    # Run image analysis if needed
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
                        )
                        llm_cost += image_cost
                        _merge_llm_usage(llm_usage, image_usage)
                    progress.log("LLM processing complete (vision enhanced)")

            elif should_analyze_images:
                # Standard processing with image analysis (no screenshot/vision)
                progress.start_spinner("Processing document and images with LLM...")

                # Create parallel tasks
                doc_task = process_with_llm(
                    markdown_for_llm,
                    url,  # Use URL as source identifier
                    cfg,
                    output_file,
                )
                img_task = analyze_images_with_llm(
                    downloaded_images,
                    markdown_for_llm,
                    output_file,
                    cfg,
                    Path(url),  # Use URL as source path
                    concurrency_limit=cfg.llm.concurrency,
                )

                # Execute in parallel
                doc_result, img_result = await asyncio.gather(doc_task, img_task)

                # Unpack results
                _, doc_cost, doc_usage = doc_result
                _, image_cost, image_usage, img_analysis = img_result

                llm_cost += doc_cost + image_cost
                _merge_llm_usage(llm_usage, doc_usage)
                _merge_llm_usage(llm_usage, image_usage)
                progress.log("LLM processing complete (document + images)")
            else:
                # Only document processing, no images to analyze, no screenshot
                progress.start_spinner("Processing with LLM...")
                _, doc_cost, doc_usage = await process_with_llm(
                    markdown_for_llm,
                    url,  # Use URL as source identifier
                    cfg,
                    output_file,
                )
                llm_cost += doc_cost
                _merge_llm_usage(llm_usage, doc_usage)
                progress.log("LLM processing complete")

            # Read the LLM-processed content for stdout output
            llm_output_file = output_file.with_suffix(".llm.md")
            if llm_output_file.exists():
                final_content = llm_output_file.read_text(encoding="utf-8")

        # Write image descriptions (if enabled and images were analyzed)
        if img_analysis and cfg.image.desc_enabled:
            write_images_json(output_dir, [img_analysis])

        # Generate report before final output
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        input_tokens = sum(u.get("input_tokens", 0) for u in llm_usage.values())
        output_tokens = sum(u.get("output_tokens", 0) for u in llm_usage.values())
        requests = sum(u.get("requests", 0) for u in llm_usage.values())

        task_options = {
            "llm": cfg.llm.enabled,
            "url": url,
        }
        task_hash = compute_task_hash(output_dir, output_dir, task_options)
        report_path = get_report_file_path(
            output_dir, task_hash, cfg.output.on_conflict
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine cache hit status (LLM was enabled but no tokens used)
        llm_cache_hit = cfg.llm.enabled and requests == 0

        report = {
            "version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(log_file_path) if log_file_path else None,
            "options": {
                "llm": cfg.llm.enabled,
                "cache": cfg.cache.enabled,
                "fetch_strategy": used_strategy,
                "alt": cfg.image.alt_enabled,
                "desc": cfg.image.desc_enabled,
            },
            "summary": {
                "total_documents": 0,
                "completed_documents": 0,
                "failed_documents": 0,
                "total_urls": 1,
                "completed_urls": 1,
                "failed_urls": 0,
                "duration": duration,
            },
            "llm_usage": {
                "models": llm_usage,
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": llm_cost,
            },
            "urls": {
                url: {
                    "status": "completed",
                    "source_file": "cli",
                    "error": None,
                    "output": str(
                        output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled
                        else output_file
                    ),
                    "fetch_strategy": used_strategy,
                    "fetch_cache_hit": fetch_cache_hit,
                    "llm_cache_hit": llm_cache_hit,
                    "images": images_count,
                    "screenshots": screenshots_count,
                    "duration": duration,
                    "llm_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": llm_cost,
                    },
                }
            },
        }

        atomic_write_json(report_path, report, order_func=order_report)
        logger.debug(f"Report saved: {report_path}")

        # Clear progress output before printing final result
        progress.clear_and_finish()

        # Output to stdout (single URL mode behavior, same as single file)
        # Use console.print with markup=False to handle Unicode correctly on Windows
        console.print(final_content, markup=False, highlight=False)

    except SystemExit:
        raise
    except Exception as e:
        ui.error(str(e))
        raise SystemExit(1)


async def process_url_batch(
    url_entries: list,  # list[UrlEntry] but imported dynamically
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    verbose: bool,
    log_file_path: Path | None = None,
    concurrency: int = 3,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Batch process multiple URLs from a URL list file.

    Shows progress bar similar to file batch processing.
    Each URL is processed concurrently up to the concurrency limit.

    Args:
        url_entries: List of UrlEntry objects from parse_url_list()
        output_dir: Output directory for all markdown files
        cfg: Configuration
        dry_run: If True, only show what would be done
        verbose: If True, enable verbose logging
        log_file_path: Path to log file (for report)
        concurrency: Max concurrent URL processing (default 3)
        fetch_strategy: Strategy to use for fetching URL content
        explicit_fetch_strategy: If True, strategy was explicitly set via CLI flag
    """
    from datetime import datetime

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from markitai.cli.processors.llm import process_with_llm
    from markitai.fetch import (
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        fetch_url,
        get_fetch_cache,
    )
    from markitai.image import download_url_images
    from markitai.security import check_symlink_safety

    # Default to auto strategy if not specified
    if fetch_strategy is None:
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
    assert fetch_strategy is not None  # for type checker

    # Dry run: just show what would be done
    if dry_run:
        # Build feature status indicators
        features = []
        if cfg.llm.enabled:
            features.append("[green]LLM[/green]")
        if cfg.image.alt_enabled:
            features.append("[green]alt[/green]")
        if cfg.image.desc_enabled:
            features.append("[green]desc[/green]")
        if cfg.screenshot.enabled:
            features.append("[green]screenshot[/green]")

        feature_str = " ".join(features) if features else "[dim]none[/dim]"
        cache_status = "enabled" if cfg.cache.enabled else "disabled"
        fetch_strategy_str = fetch_strategy.value if fetch_strategy else "auto"

        width = ui.term_width(console)
        path_max = max(width - 10, 20)
        entry_max = max(width - 4, 20)

        ui.title("Dry Run - URL Batch")
        console.print(f"  URLs: {len(url_entries)}")
        console.print(f"  Output directory: {ui.truncate(str(output_dir), path_max)}")
        console.print(f"  Fetch strategy: {fetch_strategy_str}")
        console.print(f"  Features: {feature_str}")
        console.print(f"  Cache: {cache_status}")
        console.print()
        for entry in url_entries[:10]:
            filename = entry.output_name or url_to_filename(entry.url).replace(
                ".md", ""
            )
            line = f"{entry.url} -> {filename}.md"
            console.print(f"  - {ui.truncate(line, entry_max)}")
        if len(url_entries) > 10:
            console.print(f"  ... and {len(url_entries) - 10} more")
        raise SystemExit(0)

    # Create output directory
    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    # Initialize fetch cache if caching is enabled
    fetch_cache = None
    if cfg.cache.enabled:
        cache_dir = Path(cfg.cache.global_dir).expanduser()
        fetch_cache = get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)

    started_at = datetime.now()
    total_llm_cost = 0.0
    total_llm_usage: dict[str, dict[str, Any]] = {}
    completed = 0
    failed = 0
    results: dict[str, dict] = {}

    semaphore = asyncio.Semaphore(concurrency)

    async def process_single_url(entry, progress_task, progress_obj) -> None:
        """Process a single URL."""
        nonlocal completed, failed, total_llm_cost

        url = entry.url
        custom_name = entry.output_name
        url_fetch_strategy = "unknown"

        async with semaphore:
            try:
                # Generate filename
                if custom_name:
                    filename = f"{custom_name}.md"
                else:
                    filename = url_to_filename(url)

                logger.info(f"Processing URL: {url} (strategy: {fetch_strategy.value})")
                desc_max = max(ui.term_width(console) // 3, 15)
                progress_obj.update(
                    progress_task,
                    description=f"[cyan]{ui.truncate(url, desc_max)}",
                )

                # Fetch URL using the configured strategy
                try:
                    fetch_result = await fetch_url(
                        url,
                        fetch_strategy,
                        cfg.fetch,
                        explicit_strategy=explicit_fetch_strategy,
                        cache=fetch_cache,
                        skip_read_cache=cfg.cache.no_cache,
                    )
                    url_fetch_strategy = fetch_result.strategy_used
                    markdown_content = fetch_result.content
                    cache_status = " [cache]" if fetch_result.cache_hit else ""
                    logger.info(
                        f"Fetched via {url_fetch_strategy}{cache_status}: {url}"
                    )
                except JinaRateLimitError:
                    logger.error(f"Jina Reader rate limit exceeded for: {url}")
                    results[url] = {
                        "status": "failed",
                        "error": "Jina Reader rate limit exceeded (20 RPM)",
                    }
                    failed += 1
                    return
                except FetchError as e:
                    err_msg = format_error_message(e)
                    logger.error(f"Failed to fetch {url}: {err_msg}")
                    results[url] = {"status": "failed", "error": err_msg}
                    failed += 1
                    return

                if not markdown_content.strip():
                    logger.warning(f"No content extracted from URL: {url}")
                    results[url] = {
                        "status": "failed",
                        "error": "No content extracted",
                    }
                    failed += 1
                    return

                # Download images if --alt or --desc is enabled
                images_count = 0
                if cfg.image.alt_enabled or cfg.image.desc_enabled:
                    download_result = await download_url_images(
                        markdown=markdown_content,
                        output_dir=output_dir,
                        base_url=url,
                        config=cfg.image,
                        source_name=filename.replace(".md", ""),
                        concurrency=5,
                        timeout=30,
                    )
                    markdown_content = download_result.updated_markdown
                    images_count = len(download_result.downloaded_paths)

                # Generate output path with conflict resolution
                base_output_file = output_dir / filename
                output_file = resolve_output_path(
                    base_output_file, cfg.output.on_conflict
                )

                if output_file is None:
                    logger.info(f"[SKIP] Output exists: {base_output_file}")
                    results[url] = {"status": "skipped", "error": "Output exists"}
                    return

                # Write base .md file with frontmatter
                base_content = _add_basic_frontmatter(
                    markdown_content,
                    url,
                    fetch_strategy=url_fetch_strategy,
                    output_dir=output_dir,
                    title=fetch_result.title,
                )
                atomic_write_text(output_file, base_content)

                llm_cost = 0.0
                llm_usage: dict[str, dict[str, Any]] = {}

                # LLM processing (if enabled)
                if cfg.llm.enabled:
                    _, doc_cost, doc_usage = await process_with_llm(
                        markdown_content,
                        url,
                        cfg,
                        output_file,
                    )
                    llm_cost += doc_cost
                    _merge_llm_usage(llm_usage, doc_usage)

                total_llm_cost += llm_cost
                _merge_llm_usage(total_llm_usage, llm_usage)

                results[url] = {
                    "status": "completed",
                    "error": None,
                    "output": str(
                        output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled
                        else output_file
                    ),
                    "fetch_strategy": url_fetch_strategy,
                    "images": images_count,
                }
                completed += 1
                logger.info(f"Completed via {url_fetch_strategy}: {url}")

            except Exception as e:
                err_msg = format_error_message(e)
                logger.error(f"Failed to process {url}: {err_msg}")
                results[url] = {"status": "failed", "error": err_msg}
                failed += 1

            finally:
                progress_obj.advance(progress_task)

    # Process all URLs with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing URLs...", total=len(url_entries))

        tasks = [process_single_url(entry, task, progress) for entry in url_entries]
        await asyncio.gather(*tasks)

    # Generate report
    finished_at = datetime.now()
    duration = (finished_at - started_at).total_seconds()

    input_tokens = sum(u.get("input_tokens", 0) for u in total_llm_usage.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in total_llm_usage.values())
    requests = sum(u.get("requests", 0) for u in total_llm_usage.values())

    task_options = {
        "llm": cfg.llm.enabled,
        "alt": cfg.image.alt_enabled,
        "desc": cfg.image.desc_enabled,
    }
    task_hash = compute_task_hash(output_dir, output_dir, task_options)
    report_path = get_report_file_path(output_dir, task_hash, cfg.output.on_conflict)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_file": str(log_file_path) if log_file_path else None,
        "summary": {
            "total_documents": 0,
            "completed_documents": 0,
            "failed_documents": 0,
            "total_urls": len(url_entries),
            "completed_urls": completed,
            "failed_urls": failed,
            "duration": duration,
        },
        "llm_usage": {
            "models": total_llm_usage,
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": total_llm_cost,
        },
        "urls": results,
    }

    atomic_write_json(report_path, report, order_func=order_report)
    logger.debug(f"Report saved: {report_path}")

    # Print summary
    ui.summary(f"Done: {completed} URLs ({duration:.1f}s)")
    if failed > 0:
        ui.warning(f"{failed} failed")
    out_max = max(ui.term_width(console) - 10, 20)
    console.print(f"\n  Output: {ui.truncate(str(output_dir) + '/', out_max)}")


def build_multi_source_content(
    static_content: str | None,
    browser_content: str | None,
    fallback_content: str,
) -> str:
    """Build content from URL fetch result (single-source strategy).

    With the static-first + browser-fallback strategy, we only have one
    valid source at a time. This function simply returns the primary content
    without adding any source labels (which would leak into the final output).

    Args:
        static_content: Content from static/jina fetch (may be None)
        browser_content: Content from browser fetch (may be None)
        fallback_content: Primary content from FetchResult.content

    Returns:
        Single-source content without labels
    """
    # With single-source strategy, fallback_content is already the best source
    # No need to merge or add labels - just return the primary content
    return fallback_content.strip() if fallback_content else ""


async def process_url_with_vision(
    content: str,
    screenshot_path: Path,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    processor: LLMProcessor | None = None,
    original_title: str | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Process URL content with vision enhancement using screenshot.

    This provides similar functionality to PDF/PPTX vision enhancement,
    using the page screenshot as visual reference for content extraction.

    Args:
        content: Markdown content (may be multi-source combined)
        screenshot_path: Path to the URL screenshot
        url: Original URL (used as source identifier)
        cfg: Configuration
        output_file: Output file path
        processor: Optional shared LLMProcessor
        original_title: Optional page title from fetch result

    Returns:
        Tuple of (original_content, cost_usd, llm_usage)
    """
    from markitai.cli.processors.llm import process_with_llm

    try:
        if processor is None:
            processor = create_llm_processor(cfg)

        # Use URL-specific vision enhancement (no slide/page marker protection)
        cleaned_content, frontmatter = await processor.enhance_url_with_vision(
            content, screenshot_path, context=url, original_title=original_title
        )

        # Format and write LLM output
        llm_output = output_file.with_suffix(".llm.md")
        llm_content = processor.format_llm_output(
            cleaned_content, frontmatter, source=url
        )

        # Add screenshot reference as comment
        screenshot_comment = (
            f"\n\n<!-- Screenshot for reference -->\n"
            f"<!-- ![Screenshot](screenshots/{screenshot_path.name}) -->"
        )
        llm_content += screenshot_comment

        atomic_write_text(llm_output, llm_content)
        logger.info(f"Written LLM version with vision: {llm_output}")

        # Get usage for this URL
        cost = processor.get_context_cost(url)
        usage = processor.get_context_usage(url)
        return content, cost, usage

    except Exception as e:
        logger.warning(
            f"Vision enhancement failed for {url}: {format_error_message(e)}, "
            "falling back to standard processing"
        )
        # Fallback to standard processing
        result = await process_with_llm(
            content,
            url,
            cfg,
            output_file,
            processor=processor,
        )

        # Add screenshot comment to .llm.md file even in fallback path
        # This ensures the screenshot reference is preserved for future use
        llm_output = output_file.with_suffix(".llm.md")
        screenshot_comment = (
            f"\n\n<!-- Screenshot for reference -->\n"
            f"<!-- ![Screenshot](screenshots/{screenshot_path.name}) -->"
        )

        if llm_output.exists():
            llm_content = llm_output.read_text(encoding="utf-8")
            # Only add if not already present
            if "<!-- Screenshot for reference -->" not in llm_content:
                llm_content += screenshot_comment
                atomic_write_text(llm_output, llm_content)
                logger.debug(
                    f"Added screenshot comment to fallback output: {llm_output}"
                )
        else:
            # If process_with_llm failed completely, create a basic .llm.md file
            # with the content and screenshot reference
            from markitai.workflow.helpers import add_basic_frontmatter

            llm_content = add_basic_frontmatter(content, url)
            llm_content += screenshot_comment
            atomic_write_text(llm_output, llm_content)
            logger.info(f"Created fallback LLM file with screenshot: {llm_output}")

        return result


async def process_url_screenshot_only(
    screenshot_path: Path,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    processor: LLMProcessor | None = None,
    original_title: str | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Process URL using screenshot-only mode (no pre-extracted text).

    This mode relies entirely on Vision LLM to extract content from the
    screenshot, ignoring any pre-extracted text from Playwright/markitdown.

    Args:
        screenshot_path: Path to the URL screenshot
        url: Original URL (used as source identifier)
        cfg: Configuration
        output_file: Output file path
        processor: Optional shared LLMProcessor
        original_title: Optional title from fetch result to preserve

    Returns:
        Tuple of (empty_string, cost_usd, llm_usage)
    """
    try:
        if processor is None:
            processor = create_llm_processor(cfg)

        # Extract content purely from screenshot
        cleaned_content, frontmatter = await processor.extract_from_screenshot(
            screenshot_path, context=url, original_title=original_title
        )

        # Format and write LLM output
        llm_output = output_file.with_suffix(".llm.md")
        llm_content = processor.format_llm_output(
            cleaned_content, frontmatter, source=url
        )

        # Add screenshot reference as comment
        screenshot_comment = (
            f"\n\n<!-- Screenshot for reference -->\n"
            f"<!-- ![Screenshot](screenshots/{screenshot_path.name}) -->"
        )
        llm_content += screenshot_comment

        atomic_write_text(llm_output, llm_content)
        logger.info(f"Written LLM version (screenshot-only): {llm_output}")

        # Get usage for this URL
        cost = processor.get_context_cost(url)
        usage = processor.get_context_usage(url)
        return "", cost, usage

    except Exception as e:
        logger.error(
            f"Screenshot-only extraction failed for {url}: {format_error_message(e)}"
        )
        raise


# Backward compatibility aliases
_build_multi_source_content = build_multi_source_content
_process_url_with_vision = process_url_with_vision

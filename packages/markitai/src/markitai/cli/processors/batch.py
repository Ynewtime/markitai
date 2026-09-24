"""Batch processing for CLI.

This module contains functions for batch processing of files and URLs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from loguru import logger

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.logging_config import restore_console_handler
from markitai.config import MarkitaiConfig
from markitai.constants import MAX_DOCUMENT_SIZE
from markitai.converter.base import EXTENSION_MAP
from markitai.runs import Outcome, resolve_exit_code
from markitai.security import atomic_write_text
from markitai.utils.cli_helpers import sanitize_filename, url_to_filename
from markitai.utils.output import (
    OutputNameReservations,
    output_claim_scope,
    resolve_item_output_path,
    split_markdown_name,
)
from markitai.utils.paths import ensure_dir, ensure_screenshots_dir
from markitai.utils.text import format_error_message
from markitai.utils.url_redaction import redact_url, redact_urls_in_text
from markitai.workflow.helpers import (
    add_basic_frontmatter as _add_basic_frontmatter,
)
from markitai.workflow.helpers import (
    create_llm_processor,
    write_images_json,
)

if TYPE_CHECKING:
    from markitai.batch import FileState, UrlState
    from markitai.fetch import FetchStrategy
    from markitai.llm import LLMProcessor
    from markitai.llm.engine import CacheTally
    from markitai.urls import UrlEntry

console = get_console()


def _file_output_dir(file_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Compute per-file output directory, preserving relative structure."""
    try:
        rel_path = file_path.parent.relative_to(input_dir)
        return output_dir / rel_path
    except ValueError:
        return output_dir


def _file_state_to_outcome(file_state: FileState, input_dir: Path) -> Outcome:
    """Map one batch file state onto a history Outcome.

    Output paths are stored output_dir-prefixed in the state and are used
    as-is (relative ones resolve against the process cwd, which the
    recorder shares).
    """
    from markitai.batch import FileStatus

    path = Path(file_state.path)
    try:
        source = path.relative_to(input_dir).as_posix()
    except ValueError:
        source = path.name
    failed = file_state.status == FileStatus.FAILED
    skipped = not failed and file_state.skip_reason is not None
    return Outcome(
        kind="file",
        source=source,
        status="failed" if failed else ("skipped" if skipped else "completed"),
        output_path=Path(file_state.output) if file_state.output else None,
        error=file_state.error if failed else None,
        skip_reason=file_state.skip_reason,
        images=file_state.images,
        screenshots=file_state.screenshots,
        cost_usd=file_state.cost_usd,
        llm_usage=file_state.llm_usage,
        cache_hit=file_state.cache_hit,
        llm_cache_hit=file_state.cache_hit,
        duration=file_state.duration,
        warnings=list(file_state.warnings),
    )


def _url_state_to_outcome(url_state: UrlState) -> Outcome:
    """Map one batch URL state onto a history Outcome (warnings included)."""
    from markitai.batch import FileStatus

    failed = url_state.status == FileStatus.FAILED
    return Outcome(
        kind="url",
        source=url_state.url,
        status="failed" if failed else "completed",
        output_path=Path(url_state.output) if url_state.output else None,
        error=url_state.error if failed else None,
        images=url_state.images,
        screenshots=url_state.screenshots,
        cost_usd=url_state.cost_usd,
        llm_usage=url_state.llm_usage,
        cache_hit=url_state.cache_hit,
        llm_cache_hit=url_state.cache_hit,
        fetch_strategy=url_state.fetch_strategy,
        duration=url_state.duration,
        warnings=list(url_state.warnings),
    )


def batch_item_claim_scope(
    reservations: OutputNameReservations,
    item_state: FileState | UrlState,
    *,
    requeued: bool,
    on_claimed: Callable[[Path], None],
) -> AbstractContextManager[None]:
    """Open the output-name claim scope for one batch item.

    A resumed item (failed, or interrupted mid-run) redoes its work over
    its own earlier output: the output path it had reserved is reused and
    overwritten, so no ``.v2`` duplicate appears and the state keeps
    pointing at one file. A re-queued item with no recorded target (an
    older state, or a failure before its name was claimed) never wrote a
    file this batch knows of: its default name may belong to someone
    else's file, so it follows the user's ``on_conflict`` like a new item.
    Names held by other items of the batch are always renamed around.
    """
    target = Path(item_state.target) if requeued and item_state.target else None
    return output_claim_scope(
        reservations,
        reuse=target,
        on_conflict="overwrite" if target is not None else None,
        on_claimed=on_claimed,
    )


def drop_duplicate_url_entries(
    entries: list[UrlEntry], seen: set[str], source: Path | None = None
) -> list[UrlEntry]:
    """Drop URL-list entries whose ``(url, output_name)`` was already seen.

    Each entry is one work item with its own batch state (see
    ``url_state_key``); an exact repeat would share that state and only
    produce a ``.v2`` copy of the same page, so it is skipped with a
    warning. The same URL under a different output name is kept. *seen*
    is updated in place (a directory batch shares it across its lists).
    """
    from markitai.batch import url_state_key

    kept: list[UrlEntry] = []
    for entry in entries:
        key = url_state_key(entry.url, entry.output_name)
        if key in seen:
            where = f"{source.name}: " if source is not None else ""
            name = f" ({entry.output_name})" if entry.output_name else ""
            logger.warning(
                f"{where}skipping duplicate URL entry {redact_url(entry.url)}{name}"
            )
            continue
        seen.add(key)
        kept.append(entry)
    return kept


def create_process_file(
    cfg: MarkitaiConfig,
    input_dir: Path,
    output_dir: Path,
    shared_processor: LLMProcessor | None,
) -> Callable:
    """Create a process_file function using workflow/core pipeline.

    This factory function creates a closure that captures the batch processing
    context for conversion.

    Args:
        cfg: Markitai configuration
        input_dir: Input directory for relative path calculation
        output_dir: Output directory
        shared_processor: Shared LLM processor for batch mode

    Returns:
        An async function that processes a single file and returns ProcessResult
    """
    from markitai.batch import ProcessResult
    from markitai.workflow.core import ConversionContext, convert_document_core
    from markitai.workflow.results import SKIPPED_PREFIX, document_process_result

    async def process_file(file_path: Path) -> ProcessResult:
        """Process a single file using workflow/core pipeline."""
        import time

        start_time = time.perf_counter()
        logger.debug(f"[START] {file_path.name}")

        try:
            # Calculate relative path to preserve directory structure
            file_output_dir = _file_output_dir(file_path, input_dir, output_dir)

            # Create conversion context (output name derived in the core
            # pipeline: <input filename>.md)
            ctx = ConversionContext(
                input_path=file_path,
                output_dir=file_output_dir,
                config=cfg,
                shared_processor=shared_processor,
                use_multiprocess_images=True,
                input_base_dir=input_dir,
            )

            # Run core conversion pipeline
            result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

            total_time = time.perf_counter() - start_time
            processed = document_process_result(ctx, result)
            if not processed.success:
                logger.error(
                    f"[FAIL] {file_path.name}: {processed.error} ({total_time:.2f}s)"
                )
            elif processed.error == f"{SKIPPED_PREFIX}exists)":
                logger.debug(f"[SKIP] Output exists: {processed.output_path}")
            elif processed.error == f"{SKIPPED_PREFIX}image_only)":
                logger.info(f"[SKIP] Image file, no LLM/OCR: {file_path.name}")
            else:
                logger.debug(
                    f"[DONE] {file_path.name}: {total_time:.2f}s "
                    f"(images={ctx.embedded_images_count}, "
                    f"screenshots={ctx.screenshots_count}, "
                    f"cost=${ctx.llm_cost:.4f})"
                    + (" [cache]" if processed.cache_hit else "")
                )
            return processed

        except Exception as e:
            total_time = time.perf_counter() - start_time
            err_msg = format_error_message(e)
            logger.error(f"[FAIL] {file_path.name}: {err_msg} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=err_msg)

    return process_file


def _screenshot_reference_markdown(tiles: list[Path]) -> str:
    """Base .md body of a screenshot-only URL: one image link per tile."""
    from markitai.constants import SCREENSHOTS_REL_PATH
    from markitai.utils.text import markdown_image_reference

    return "\n\n".join(
        markdown_image_reference(
            f"Screenshot {i + 1}" if len(tiles) > 1 else "Screenshot",
            f"{SCREENSHOTS_REL_PATH}/{tile.name}",
        )
        for i, tile in enumerate(tiles)
    )


async def _run_url_llm_branches(
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    fetch_result: Any,
    *,
    markdown_for_llm: str,
    downloaded_images: list[Path],
    screenshot_only: bool,
    vision: bool,
    processor: LLMProcessor | None,
) -> tuple[float, dict[str, dict[str, Any]], Any]:
    """Run the batch URL's CLI LLM branch (screenshot-only, vision, images).

    Writes ``<output>.llm.md``; any LLM failure propagates so the caller can
    write the base .md fallback and fail the URL.

    Returns:
        Tuple of (cost, usage stats, image analysis result or None).
    """
    from markitai.cli.processors.url import (
        build_multi_source_content,
        process_url_with_vision,
        run_url_document_llm,
        run_url_llm_with_images,
        run_url_screenshot_only_llm,
    )

    screenshot_path = fetch_result.screenshot_path
    screenshot_tiles = list(fetch_result.screenshot_tiles or [])
    source_extra_meta = fetch_result.metadata.get("source_frontmatter")
    should_analyze_images = bool(
        (cfg.image.alt_enabled or cfg.image.desc_enabled) and downloaded_images
    )

    if screenshot_only:
        # --screenshot-only with LLM (single-URL parity): extract content
        # purely from the screenshot
        assert screenshot_path is not None  # guaranteed by the caller
        return await run_url_screenshot_only_llm(
            screenshot_path,
            url,
            cfg,
            output_file,
            fetch_result,
            screenshot_tiles=screenshot_tiles or None,
            downloaded_images=downloaded_images,
            image_context=markdown_for_llm,
            processor=processor,
        )

    if vision:
        # Multi-source URL with screenshot: vision LLM for better extraction
        assert screenshot_path is not None  # guaranteed by the caller
        multi_source_content = build_multi_source_content(
            fetch_result.static_content,
            fetch_result.browser_content,
            markdown_for_llm,  # Fallback primary content
        )
        logger.debug(
            f"[URL] Using vision enhancement for multi-source URL: {redact_url(url)}"
        )

        async def _vision_task() -> tuple[str, float, dict[str, dict[str, Any]]]:
            return await process_url_with_vision(
                multi_source_content,
                screenshot_path,
                url,
                cfg,
                output_file,
                processor=processor,
                original_title=fetch_result.title,
                fetch_strategy=fetch_result.strategy_used,
                extra_meta=source_extra_meta,
            )

        if not should_analyze_images:
            _, cost, usage = await _vision_task()
            return cost, usage, None
        # Run vision enhancement and image analysis in parallel
        return await run_url_llm_with_images(
            _vision_task,
            downloaded_images=downloaded_images,
            image_context=multi_source_content,
            output_file=output_file,
            cfg=cfg,
            url=url,
            processor=processor,
        )

    async def _doc_task() -> tuple[str, float, dict[str, dict[str, Any]]]:
        return await run_url_document_llm(
            markdown_for_llm,
            url,
            cfg,
            output_file,
            fetch_result,
            screenshot_path=screenshot_path,
            extra_meta=source_extra_meta,
            processor=processor,
        )

    if not should_analyze_images:
        _, cost, usage = await _doc_task()
        return cost, usage, None
    # Standard processing with image analysis, in parallel
    return await run_url_llm_with_images(
        _doc_task,
        downloaded_images=downloaded_images,
        image_context=markdown_for_llm,
        output_file=output_file,
        cfg=cfg,
        url=url,
        processor=processor,
    )


def create_url_processor(
    cfg: MarkitaiConfig,
    output_dir: Path,
    fetch_strategy: FetchStrategy | None,
    explicit_fetch_strategy: bool,
    shared_processor: LLMProcessor | None = None,
    renderer: Any | None = None,
    *,
    localized_base_md: bool = False,
    fetch_error_max_length: int = 200,
) -> Callable:
    """Create a URL processing function for batch processing.

    This is the single shared fetch->images->base .md->LLM cascade for one
    URL; both the directory batch (.urls files discovered in a directory)
    and the URL-list batch (`markitai list.urls`) build their workers on it.

    Args:
        cfg: Configuration
        output_dir: Output directory
        fetch_strategy: Fetch strategy to use
        explicit_fetch_strategy: Whether strategy was explicitly specified
        shared_processor: Optional shared LLMProcessor
        renderer: Optional PlaywrightRenderer used for every URL. Without
            one, each URL takes the fetch session's shared renderer for its
            own proxy: NO_PROXY-exempt URLs get an unproxied browser, and
            renderers of different proxies coexist for the whole batch.

    --screenshot-only is handled like the single-URL path: without LLM the
    screenshot is recorded and no .md is written; with LLM the content is
    extracted purely from the screenshot.
        localized_base_md: Write the base .md from the image-localized
            markdown (URL-list batch behavior) instead of the original
            fetched markdown (directory-batch behavior).
        fetch_error_max_length: Max length for formatted FetchError messages.
            The default matches format_error_message's own 200-char limit;
            the URL-list batch passes 800 to keep actionable multi-line
            error blocks intact in logs/report.

    Returns:
        Async function that processes a single URL and returns ProcessResult
    """
    from markitai import fetch as fetch_module
    from markitai.batch import ProcessResult
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
        source_file: Path | None = None,
        custom_name: str | None = None,
    ) -> tuple[ProcessResult, dict[str, Any]]:
        """Process a single URL.

        Args:
            url: URL to process
            source_file: Path to the .urls file containing this URL
                (caller-side bookkeeping only; unused here)
            custom_name: Optional custom output name

        Returns:
            Tuple of (ProcessResult, extra_info dict with fetch_strategy)
        """
        if not cfg.llm.enabled:
            return await _process_url(url, custom_name, None)

        from markitai.llm.engine import track_cache_hits

        # This URL's own cache lookups: the processor is shared by the batch
        with track_cache_hits() as cache_tally:
            return await _process_url(url, custom_name, cache_tally)

    async def _process_url(
        url: str,
        custom_name: str | None,
        cache_tally: CacheTally | None,
    ) -> tuple[ProcessResult, dict[str, Any]]:
        """Body of ``process_url``; *cache_tally* is None without LLM."""
        import time

        from markitai.cli.processors.url import _reads_screenshot_only

        start_time = time.perf_counter()
        extra_info: dict[str, Any] = {
            "fetch_strategy": "unknown",
        }

        try:
            # Generate filename (sanitize custom names to prevent path traversal)
            if custom_name:
                filename = f"{sanitize_filename(custom_name)}.md"
            else:
                filename = url_to_filename(url)

            logger.debug(
                f"[URL] Processing: {redact_url(url)} "
                f"(strategy: {_fetch_strategy.value})"
            )
            source_extra_meta: dict[str, Any] | None = None

            url_renderer = renderer
            if url_renderer is None:
                # One browser per proxy configuration, shared by the batch.
                # Creating it is free (Chromium launches on first use), and
                # a NO_PROXY-exempt URL never rides a proxied browser.
                url_renderer = await fetch_module._get_playwright_renderer(
                    proxy=fetch_module.get_proxy_for_url(url) or None,
                    config=cfg.fetch,
                )

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
                    renderer=url_renderer,
                )
                extra_info["fetch_strategy"] = fetch_result.strategy_used
                original_markdown = fetch_result.content
                screenshot_path = fetch_result.screenshot_path
                screenshot_tiles = list(fetch_result.screenshot_tiles or [])
                source_extra_meta = fetch_result.metadata.get("source_frontmatter")
                cache_status = " [cache]" if fetch_result.cache_hit else ""
                logger.debug(
                    f"[URL] Fetched via {fetch_result.strategy_used}{cache_status}: "
                    f"{redact_url(url)}"
                )
            except JinaRateLimitError:
                logger.error(f"[URL] Jina rate limit exceeded for: {redact_url(url)}")
                return ProcessResult(
                    success=False,
                    error="Jina Reader rate limit exceeded (20 RPM)",
                ), extra_info
            except FetchError as e:
                err_msg = format_error_message(e, max_length=fetch_error_max_length)
                logger.error(
                    f"[URL] Fetch failed {redact_url(url)}: "
                    f"{redact_urls_in_text(err_msg)}"
                )
                return ProcessResult(success=False, error=err_msg), extra_info

            # --screenshot-only reads the page from its screenshot: an empty
            # text layer (canvas app, image-only page) is not a failure there
            if not original_markdown.strip() and not (
                _reads_screenshot_only(cfg)
                and screenshot_path is not None
                and screenshot_path.exists()
            ):
                logger.error(f"[URL] No content: {redact_url(url)}")
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

            # Download images only when their LLM analysis can run. Presets
            # may leave image flags set after an explicit --no-llm override.
            images_count = 0
            # A long page is several tiles; count what was written
            screenshots_count = (
                len(screenshot_tiles)
                if has_screenshot and screenshot_tiles
                else (1 if has_screenshot else 0)
            )
            downloaded_images: list[Path] = []

            if has_screenshot and screenshot_path:
                logger.debug(f"[URL] Screenshot captured: {screenshot_path.name}")
            elif cfg.screenshot.enabled:
                # Requested but not captured: never silent (the batch prints
                # it and --json carries it in the item's warnings)
                reason = fetch_result.metadata.get("screenshot_error")
                if not isinstance(reason, str) or not reason:
                    reason = "the page was not rendered in a browser"
                extra_info["warnings"] = [f"Screenshot not captured: {reason}"]

            # Same rule as a single URL: --llm --pure reads the text layer
            screenshot_only_run = _reads_screenshot_only(cfg)
            if screenshot_only_run and not has_screenshot:
                # --screenshot-only without a screenshot has nothing to deliver
                warnings = extra_info.pop("warnings", None) or [
                    "Screenshot not captured"
                ]
                logger.error(f"[URL] {warnings[0]}: {redact_url(url)}")
                return ProcessResult(
                    success=False,
                    error=f"--screenshot-only: {warnings[0]}",
                ), extra_info

            # --screenshot-only without LLM (single-URL parity):
            # just record the screenshot, no image download, no .md output
            if screenshot_only_run and not cfg.llm.enabled:
                assert screenshot_path is not None  # has_screenshot above
                logger.debug(f"[URL] Screenshot-only (no LLM): {screenshot_path.name}")
                return ProcessResult(
                    success=True,
                    output_path=str(screenshot_path),
                    screenshots=screenshots_count,
                ), extra_info

            # Generate output path (claimed from the batch's reservation
            # table when the caller opened a claim scope for this URL; the
            # cascade below resolves the same base and gets the same answer).
            # Claimed before the images are downloaded: they are named after
            # the output, so a renamed page.v2.md gets page.v2.0001.jpg
            # instead of overwriting page.md's images.
            base_output_file = output_dir / filename
            output_file = resolve_item_output_path(
                base_output_file, cfg.output.on_conflict
            )

            if output_file is None:
                logger.debug(f"[URL] Skipped (exists): {base_output_file}")
                return ProcessResult(
                    success=True,
                    output_path=str(base_output_file),
                    error="skipped (exists)",
                ), extra_info

            if cfg.llm.enabled and (cfg.image.alt_enabled or cfg.image.desc_enabled):
                download_result = await download_url_images(
                    markdown=original_markdown,
                    output_dir=output_dir,
                    # Relative image paths resolve against the page's
                    # post-redirect URL (/docs -> /docs/, short links)
                    base_url=fetch_result.final_url or url,
                    config=cfg.image,
                    source_name=split_markdown_name(output_file.name)[0],
                    concurrency=5,
                    timeout=30,
                )
                markdown_for_llm = download_result.updated_markdown
                downloaded_images = download_result.downloaded_paths
                images_count = len(downloaded_images)

            # Standard path — no screenshot-only, no vision enhancement,
            # no image analysis, no raw pure base: delegate base+LLM to the
            # shared workflow cascade (the same code serve/api/CLI run).
            url_llm_usage: dict[str, dict[str, Any]] = {}
            llm_cost = 0.0
            img_analysis = None
            should_analyze_images = bool(
                (cfg.image.alt_enabled or cfg.image.desc_enabled) and downloaded_images
            )
            use_vision_enhancement = bool(
                has_multi_source and has_screenshot and screenshot_path
            )
            use_screenshot_only_llm = bool(
                _reads_screenshot_only(cfg)
                and has_screenshot
                and screenshot_path is not None
            )
            use_cli_llm_branches = cfg.llm.enabled and (
                use_screenshot_only_llm
                or use_vision_enhancement
                or should_analyze_images
            )
            use_raw_pure_base = cfg.llm.pure and not cfg.llm.enabled

            if not use_cli_llm_branches and not use_raw_pure_base:
                from markitai.cli.processors.url import cli_document_llm_stage
                from markitai.workflow.url import convert_url_cascade

                # LLM failures raise ConversionError after the base file is
                # on disk; the outer catch-all maps it to a failed result.
                cascade = await convert_url_cascade(
                    url,
                    cfg,
                    output_dir,
                    processor=shared_processor,
                    fetch_result=fetch_result,
                    markdown_override=(
                        markdown_for_llm
                        if cfg.llm.enabled
                        and (cfg.image.alt_enabled or cfg.image.desc_enabled)
                        else None
                    ),
                    base_from_localized=localized_base_md,
                    output_name=filename,
                    llm_error_policy="raise",
                    llm_stage=cli_document_llm_stage,
                )
                assert cascade.target_file is not None  # skip handled above
                output_file = cascade.target_file
                llm_cost = cascade.cost_usd
                url_llm_usage = cascade.llm_usage
            else:
                # Write base .md file (respect --llm, --pure, --keep-base).
                # localized_base_md: URL-list batch writes the base .md from the
                # image-localized markdown; directory batch keeps the original.
                base_source = (
                    markdown_for_llm if localized_base_md else original_markdown
                )
                if cfg.llm.pure and not cfg.llm.enabled:
                    # Pure mode without LLM: write raw markdown, no frontmatter
                    base_content = base_source
                elif use_screenshot_only_llm and screenshot_path is not None:
                    # The page is read from its screenshot(s), so the base
                    # .md references them instead of the text layer
                    base_content = _add_basic_frontmatter(
                        _screenshot_reference_markdown(
                            screenshot_tiles or [screenshot_path]
                        ),
                        url,
                        fetch_strategy=fetch_result.strategy_used,
                        screenshot_path=None,  # referenced above, not twice
                        output_dir=output_dir,
                        title=fetch_result.title,
                        extra_meta=source_extra_meta,
                    )
                else:
                    base_content = _add_basic_frontmatter(
                        base_source,
                        url,
                        fetch_strategy=fetch_result.strategy_used,
                        screenshot_path=screenshot_path,
                        screenshot_tiles=screenshot_tiles or None,
                        output_dir=output_dir,
                        title=fetch_result.title,
                        extra_meta=source_extra_meta,
                    )
                should_write_base = not cfg.llm.enabled or cfg.llm.keep_base
                if should_write_base:
                    atomic_write_text(output_file, base_content)

                if cfg.llm.enabled:
                    try:
                        (
                            llm_cost,
                            url_llm_usage,
                            img_analysis,
                        ) = await _run_url_llm_branches(
                            url,
                            cfg,
                            output_file,
                            fetch_result,
                            markdown_for_llm=markdown_for_llm,
                            downloaded_images=downloaded_images,
                            screenshot_only=use_screenshot_only_llm,
                            vision=use_vision_enhancement,
                            processor=shared_processor,
                        )
                    except Exception as e:
                        # Same policy as the shared cascade and the file
                        # pipeline: an LLM failure fails the URL, with the
                        # base .md on disk as the fallback output.
                        if not should_write_base:
                            atomic_write_text(output_file, base_content)
                        from markitai.utils.errors import ConversionError

                        raise ConversionError(
                            f"LLM processing failed: {format_error_message(e)}"
                        ) from e

            # Output profile post-processing (no-op without a profile)
            if cfg.output.profile is not None:
                from markitai.output_profiles import apply_profile_to_file

                for candidate in (output_file, output_file.with_suffix(".llm.md")):
                    apply_profile_to_file(candidate, output_dir, cfg)

            total_time = time.perf_counter() - start_time

            # Never mark a URL completed with an output that is not on disk
            produced_file = (
                output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
            )
            if not produced_file.is_file():
                error = f"No output was produced for {redact_url(url)}"
                logger.error(f"[URL] {error} ({total_time:.2f}s)")
                return ProcessResult(success=False, error=error), extra_info

            # Served from cache: this URL's tally saw only cache hits (an
            # empty usage dict alone is also what a total LLM failure is)
            is_cache_hit = cache_tally is not None and cache_tally.served_from_cache(
                url_llm_usage
            )

            logger.debug(
                f"[URL] Completed via {extra_info['fetch_strategy']}: "
                f"{redact_url(url)} "
                f"({total_time:.2f}s)" + (" [cache]" if is_cache_hit else "")
            )

            return ProcessResult(
                success=True,
                output_path=str(produced_file),
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
            logger.error(
                f"[URL] Failed {redact_url(url)}: "
                f"{redact_urls_in_text(err_msg)} ({total_time:.2f}s)"
            )
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
    glob_patterns: tuple[str, ...] = (),
    quiet: bool = False,
    history: list[Outcome] | None = None,
) -> None:
    """Process directory in batch mode.

    Args:
        history: Optional list collecting this run's per-item Outcomes for
            serve history recording (append-only; the caller decides
            whether to record). Only items processed in this run are
            collected — a resumed batch does not re-record items that
            completed earlier.
    """
    # Batch output must be a directory; reject file-like -o values early
    if output_dir.suffix == ".md" and not output_dir.is_dir():
        raise click.UsageError(
            "-o looks like a file path but input is a directory; "
            "pass an output directory"
        )

    from datetime import UTC, datetime

    from markitai.batch import (
        BatchProcessor,
        FileState,
        FileStatus,
        UrlState,
        url_state_key,
    )
    from markitai.cli.processors.validators import (
        check_playwright_for_urls,
        warn_case_sensitivity_mismatches,
    )
    from markitai.security import check_symlink_safety
    from markitai.urls import parse_url_list

    # Supported extensions
    extensions = set(EXTENSION_MAP.keys())
    normalized_globs = [pattern.strip() for pattern in glob_patterns if pattern.strip()]

    # Build task options for report (before BatchProcessor init for hash calculation)
    # Note: input_dir and output_dir will be converted to absolute paths by init_state()
    task_options: dict[str, Any] = {
        "concurrency": cfg.batch.concurrency,
        "llm": cfg.llm.enabled,
        "ocr": cfg.ocr.enabled,
        "screenshot": cfg.screenshot.enabled,
        "alt": cfg.image.alt_enabled,
        "desc": cfg.image.desc_enabled,
        "scan_max_depth": cfg.batch.scan_max_depth,
    }
    if normalized_globs:
        task_options["glob_patterns"] = normalized_globs
    if cfg.llm.enabled and cfg.llm.model_list:
        task_options["models"] = [m.litellm_params.model for m in cfg.llm.model_list]

    batch = BatchProcessor(
        cfg.batch,
        output_dir,
        input_path=input_dir,
        log_file=log_file_path,
        on_conflict=cfg.output.on_conflict,
        task_options=task_options,
        console_log_restorer=restore_console_handler,
    )
    from markitai.output_profiles import assets_visible

    files = batch.discover_files(
        input_dir,
        extensions,
        glob_patterns=normalized_globs,
        visible_assets=assets_visible(cfg),
    )

    # Discover .urls files for URL batch processing
    url_list_files = batch.discover_files(
        input_dir,
        {".urls"},
        glob_patterns=normalized_globs,
        visible_assets=assets_visible(cfg),
    )
    url_entries_from_files: list = []  # List of (source_file, UrlEntry)
    seen_url_keys: set[str] = set()

    for url_file in url_list_files:
        try:
            entries = drop_duplicate_url_entries(
                parse_url_list(url_file), seen_url_keys, url_file
            )
            for entry in entries:
                url_entries_from_files.append((url_file, entry))
            if entries:
                logger.debug(f"Found {len(entries)} URLs in {url_file.name}")
        except Exception as e:
            logger.warning(f"Failed to parse URL list {url_file}: {e}")

    # A quiet dry run intentionally produces no preview and has no side
    # effects. Discovery above still validates the input shape.
    if dry_run and quiet:
        raise SystemExit(0)

    # Check Playwright availability if URLs will be processed
    if url_entries_from_files and not quiet:
        check_playwright_for_urls(cfg, console)

    if not files and not url_entries_from_files:
        if not quiet:
            console.print("[yellow]No supported files or URL lists found.[/yellow]")
        raise SystemExit(0)

    # Warn about potential case-sensitivity mismatches in --no-cache-for patterns
    if cfg.cache.no_cache_patterns and not quiet:
        warn_case_sensitivity_mismatches(files, input_dir, cfg.cache.no_cache_patterns)

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    # Collect URL source files (used for state bookkeeping below)
    url_sources_set: set[str] = set()
    for source_file, _entry in url_entries_from_files:
        url_sources_set.add(str(source_file))

    # Resume: load the previous state (same task hash) and keep COMPLETED
    # entries; PENDING/FAILED entries (including interrupted IN_PROGRESS,
    # downgraded by load_state) are re-queued, and files/URLs discovered
    # this run but absent from the state are added as new work.
    resumed_state = batch.load_state() if resume else None
    if resumed_state is not None:
        # Named entries used to share their URL's state; give them their own
        resumed_state.adopt_legacy_url_keys(
            (url_state_key(entry.url, entry.output_name), entry.url)
            for _source_file, entry in url_entries_from_files
        )
    # One output-name reservation table for every file and URL of the run
    reservations = OutputNameReservations()
    # Items re-queued from the previous state (failed or interrupted): they
    # redo their work over their own earlier output (see batch_item_claim_scope)
    requeued_files: set[str] = set()
    requeued_urls: set[str] = set()
    if resumed_state is not None:
        for key, file_state in resumed_state.files.items():
            if file_state.status == FileStatus.FAILED:
                requeued_files.add(key)
            elif file_state.status == FileStatus.COMPLETED and file_state.output:
                reservations.reserve(Path(file_state.output))
        for key, url_state in resumed_state.urls.items():
            if url_state.status == FileStatus.FAILED:
                requeued_urls.add(key)
            elif url_state.status == FileStatus.COMPLETED and url_state.output:
                reservations.reserve(Path(url_state.output))

        known_resolved = {str(Path(k).resolve()) for k in resumed_state.files}
        for f in files:
            if str(f.resolve()) not in known_resolved:
                resumed_state.files[str(f)] = FileState(path=str(f))
        files_to_process = resumed_state.get_pending_files()

        url_entries_to_process = []
        for source_file, entry in url_entries_from_files:
            url_key = url_state_key(entry.url, entry.output_name)
            url_state = resumed_state.urls.get(url_key)
            if url_state is None:
                resumed_state.urls[url_key] = UrlState(
                    url=entry.url,
                    source_file=str(source_file),
                    status=FileStatus.PENDING,
                )
                url_entries_to_process.append((source_file, entry))
            elif url_state.status != FileStatus.COMPLETED:
                url_entries_to_process.append((source_file, entry))
        resumed_state.url_sources = sorted(
            set(resumed_state.url_sources) | url_sources_set
        )

        done_count = resumed_state.completed_count + resumed_state.completed_urls_count
        remaining = len(files_to_process) + len(url_entries_to_process)
        if not quiet:
            console.print(
                f"[dim]Resuming batch: {done_count} completed, "
                f"{remaining} remaining[/dim]"
            )
    else:
        if resume:
            logger.debug("No previous batch state found; starting fresh")
        files_to_process = files
        url_entries_to_process = url_entries_from_files

    if dry_run:
        feature_str = ui.build_feature_str(cfg)
        cache_status = "enabled" if cfg.cache.enabled else "disabled"

        # Display unified dry run output
        width = ui.term_width(console)
        path_max = max(width - 10, 20)
        url_max = max(width - 6, 20)

        ui.title("Dry Run")
        console.print(f"  Input: {ui.truncate(str(input_dir), path_max)}")
        console.print(f"  Output: {ui.truncate(str(output_dir), path_max)}")
        console.print(f"  Features: {feature_str}")
        console.print(f"  Cache: {cache_status}")
        console.print()
        console.print(f"  Files ({len(files_to_process)})")
        for f in files_to_process[:10]:
            console.print(f"    {ui.MARK_INFO} {ui.truncate(f.name, url_max)}")
        if len(files_to_process) > 10:
            console.print(f"    ... and {len(files_to_process) - 10} more files")
        if url_entries_to_process:
            console.print()
            console.print(f"  URLs ({len(url_entries_to_process)})")
            for _source_file, entry in url_entries_to_process[:10]:
                console.print(
                    f"    {ui.MARK_INFO} {ui.truncate(redact_url(entry.url), url_max)}"
                )
            if len(url_entries_to_process) > 10:
                console.print(
                    f"    ... and {len(url_entries_to_process) - 10} more URLs"
                )
        console.print()
        console.print("  " + "\u2500" * 20)
        console.print(
            f"  Total: {len(files_to_process)} files, "
            f"{len(url_entries_to_process)} URLs"
        )
        if cfg.cache.enabled:
            ui.step("Tip: Use 'markitai cache stats -v' to view cached entries")
        raise SystemExit(0)

    # Record batch start time before any processing (including pre-conversion)
    batch_started_at = datetime.now(UTC).astimezone().isoformat()

    # Start Live display early to capture all logs (including URL processing)
    # This ensures all INFO+ logs go to the panel instead of console
    if not quiet:
        batch.start_live_display(
            verbose=verbose,
            console_handler_id=console_handler_id,
            total_files=len(files_to_process),
            total_urls=len(url_entries_to_process),
        )

    # Create shared LLM runtime and processor for batch mode
    shared_processor = None
    if cfg.llm.enabled:
        from markitai.llm import LLMRuntime

        runtime = LLMRuntime(concurrency=cfg.llm.concurrency)
        shared_processor = create_llm_processor(cfg, runtime=runtime)
        logger.debug(
            f"Created shared LLMProcessor with concurrency={cfg.llm.concurrency}"
        )

    # Create process_file using workflow/core implementation
    process_file = create_process_file(
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        shared_processor=shared_processor,
    )
    logger.debug("Using workflow/core implementation for batch processing")

    # Adopt the resumed state, or initialize a fresh one
    if resumed_state is not None:
        batch.state = resumed_state
        batch.state.started_at = batch_started_at
    elif files or url_entries_from_files:
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
            batch.state.urls[url_state_key(entry.url, entry.output_name)] = UrlState(
                url=entry.url,
                source_file=str(source_file),
                status=FileStatus.PENDING,
            )

        # Write the base state before any work starts. Runs after this only
        # append their deltas to a .jsonl sidecar, and load_state() gives up
        # the moment the base file is missing — so without this line an
        # interrupted batch left a sidecar nothing could replay, and --resume
        # silently restarted from zero, re-paying for every LLM call already
        # made. BatchProcessor.process_batch has always done this; the CLI
        # path, which is the one people use, did not.
        batch.save_state(force=True)

    # Create URL processor function
    url_processor = None
    if url_entries_to_process:
        url_processor = create_url_processor(
            cfg=cfg,
            output_dir=output_dir,
            fetch_strategy=fetch_strategy,
            explicit_fetch_strategy=explicit_fetch_strategy,
            shared_processor=shared_processor,
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

        key = url_state_key(url, custom_name)
        url_state = batch.state.urls.get(key)
        if url_state is None:
            return

        def record_target(path: Path) -> None:
            url_state.target = str(path)
            batch._dirty_keys.add(key)

        # Workers outnumber URL slots (the pool serves files too): a URL is
        # in_progress only once it holds a slot, so an interrupt leaves the
        # ones still waiting pending for --resume
        async with url_semaphore:
            url_state.status = FileStatus.IN_PROGRESS
            url_state.started_at = datetime.now(UTC).astimezone().isoformat()
            batch._dirty_keys.add(key)

            start_time = asyncio.get_running_loop().time()

            try:
                batch.update_url_status(url, key=key)
                with batch_item_claim_scope(
                    reservations,
                    url_state,
                    requeued=key in requeued_urls,
                    on_claimed=record_target,
                ):
                    result, extra_info = await url_processor(
                        url, source_file, custom_name
                    )
                # Non-fatal problems (a requested screenshot not captured):
                # printed like the URL-list batch does, and --json carries them
                url_state.warnings = list(extra_info.get("warnings") or [])

                if result.success:
                    for warning in url_state.warnings:
                        logger.warning(f"[URL] {redact_url(url)}: {warning}")
                        if not quiet:
                            ui.warning(
                                f"{redact_url(url)}: {warning}",
                                console=batch.console,
                            )
                    url_state.status = FileStatus.COMPLETED
                    url_state.output = result.output_path
                    url_state.fetch_strategy = extra_info.get("fetch_strategy")
                    url_state.images = result.images
                    url_state.screenshots = result.screenshots
                    url_state.cost_usd = result.cost_usd
                    url_state.llm_usage = result.llm_usage
                    url_state.cache_hit = result.cache_hit
                    batch._dirty_keys.add(key)
                    # Collect image analysis for JSON output
                    if result.image_analysis_result is not None:
                        batch.image_analysis_results.append(
                            result.image_analysis_result
                        )
                else:
                    url_state.status = FileStatus.FAILED
                    url_state.error = result.error
                    batch._dirty_keys.add(key)

            except Exception as e:
                err_msg = format_error_message(e)
                url_state.status = FileStatus.FAILED
                url_state.error = err_msg
                batch._dirty_keys.add(key)
                logger.error(
                    f"[URL] Failed {redact_url(url)}: {redact_urls_in_text(err_msg)}"
                )

            finally:
                end_time = asyncio.get_running_loop().time()
                url_state.completed_at = datetime.now(UTC).astimezone().isoformat()
                url_state.duration = end_time - start_time

                # Update progress
                batch.update_url_status(url, completed=True, key=key)

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    async def process_file_with_state(file_path: Path) -> None:
        """Process a file and update batch state."""
        assert batch.state is not None

        file_key = str(file_path)
        file_state = batch.state.files.get(file_key)

        if file_state is None:
            return

        def record_target(path: Path) -> None:
            file_state.target = str(path)
            batch._dirty_keys.add(file_key)

        # Workers outnumber file slots (the pool serves URLs too): a file is
        # in_progress only once it holds a slot, so an interrupt leaves the
        # ones still waiting pending for --resume
        async with file_semaphore:
            file_state.status = FileStatus.IN_PROGRESS
            file_state.started_at = datetime.now(UTC).astimezone().isoformat()
            batch._dirty_keys.add(file_key)

            start_time = asyncio.get_running_loop().time()

            try:
                display_name = file_path.name
                if batch.input_path is not None:
                    try:
                        display_name = file_path.relative_to(
                            batch.input_path
                        ).as_posix()
                    except ValueError:
                        display_name = file_path.name
                batch.set_current_file(file_key, display_name)
                with batch_item_claim_scope(
                    reservations,
                    file_state,
                    requeued=file_key in requeued_files,
                    on_claimed=record_target,
                ):
                    result = await process_file(file_path)

                if result.success:
                    file_state.status = FileStatus.COMPLETED
                    file_state.output = result.output_path
                    file_state.images = result.images
                    file_state.screenshots = result.screenshots
                    file_state.cost_usd = result.cost_usd
                    file_state.llm_usage = result.llm_usage
                    file_state.cache_hit = result.cache_hit
                    # Non-fatal problems (an image analysis that failed):
                    # logged like a URL's, and --json carries them
                    file_state.warnings = list(result.warnings)
                    for warning in file_state.warnings:
                        logger.warning(f"[File] {display_name}: {warning}")
                        if not quiet:
                            ui.warning(
                                f"{display_name}: {warning}", console=batch.console
                            )
                    # Extract skip reason from ProcessResult error field
                    if result.error and result.error.startswith("skipped ("):
                        file_state.skip_reason = result.error[9:-1]
                    batch._dirty_keys.add(file_key)
                    # Collect image analysis for JSON output
                    if result.image_analysis_result is not None:
                        batch.image_analysis_results.append(
                            result.image_analysis_result
                        )
                else:
                    file_state.status = FileStatus.FAILED
                    file_state.error = result.error
                    batch._dirty_keys.add(file_key)

            except Exception as e:
                file_state.status = FileStatus.FAILED
                err_msg = format_error_message(e)
                file_state.error = err_msg
                batch._dirty_keys.add(file_key)
                logger.error(f"[FAIL] {file_path.name}: {err_msg}")

            finally:
                end_time = asyncio.get_running_loop().time()
                file_state.completed_at = datetime.now(UTC).astimezone().isoformat()
                file_state.duration = end_time - start_time

                # Update progress
                batch.advance_progress(current_item=file_key)

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    # Run all tasks via queue + worker pool (URLs + files)
    state = batch.state
    try:
        if files_to_process or url_entries_to_process:
            logger.debug(
                f"Processing {len(files_to_process)} files and "
                f"{len(url_entries_to_process)} URLs "
                f"with concurrency {cfg.batch.concurrency}"
            )

            items: list[tuple[str, Any]] = []
            for source_file, entry in url_entries_to_process:
                items.append(("url", (entry.url, source_file, entry.output_name)))
            for file_path in files_to_process:
                items.append(("file", file_path))

            if items:
                max_concurrency = max(cfg.batch.concurrency, cfg.batch.url_concurrency)
                queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue(
                    maxsize=max_concurrency * 2
                )

                async def producer() -> None:
                    for item in items:
                        await queue.put(item)
                    for _ in range(max_concurrency):
                        await queue.put(None)

                async def worker() -> None:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        try:
                            item_type, args = item
                            if item_type == "url":
                                url, src_file, custom_name = args
                                await process_url_with_state(url, src_file, custom_name)
                            else:
                                await process_file_with_state(args)
                        except Exception:
                            logger.debug("Unexpected error in worker", exc_info=True)

                producer_task = asyncio.create_task(producer())
                workers = [
                    asyncio.create_task(worker()) for _ in range(max_concurrency)
                ]
                await asyncio.gather(producer_task, *workers)

    except BaseException:
        # Ctrl-C (CancelledError/KeyboardInterrupt) or an unexpected error:
        # the throttled saves may not hold what finished in the last
        # interval, and the compaction below never runs. Save now so
        # --resume picks up exactly where this run stopped.
        batch.persist_state_on_abort()
        raise

    finally:
        # Stop Live display and restore console handler
        # This must be done before printing summary
        batch.stop_live_display()

    if state:
        # Update state timestamp
        state.updated_at = datetime.now(UTC).astimezone().isoformat()
        batch.compact_state()

        # Print summary (uses state for URL stats)
        if not quiet:
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
            write_images_json(
                output_dir,
                batch.image_analysis_results,
                visible_assets=assets_visible(cfg),
            )

        # Save report (default ON for batch runs; output.report=false opts out)
        if cfg.output.report is not False:
            batch.save_report()
        else:
            # Still persist state for resume capability
            batch.save_state(force=True, log=True)
            logger.debug("Report generation disabled (output.report=false)")

    # Collect this run's per-item outcomes for serve history recording.
    # Only files/URLs processed in this run — a resumed batch does not
    # re-record items that completed in an earlier run.
    if history is not None and state is not None:
        for file_path in files_to_process:
            file_state = state.files.get(str(file_path))
            if file_state is not None:
                history.append(_file_state_to_outcome(file_state, input_dir))
        for _source_file, entry in url_entries_to_process:
            url_state = state.urls.get(url_state_key(entry.url, entry.output_name))
            if url_state is not None:
                history.append(_url_state_to_outcome(url_state))

    # Exit with appropriate code
    total_failed = (state.failed_count if state else 0) + (
        state.failed_urls_count if state else 0
    )
    exit_code = resolve_exit_code(total_failed, batch=True)
    if exit_code != 0:
        raise SystemExit(exit_code)  # PARTIAL_FAILURE

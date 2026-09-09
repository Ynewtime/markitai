"""Shared URL conversion cascade: fetch -> localize images -> .md -> .llm.md.

This is the single implementation of the UI-free URL pipeline. The CLI
(``cli/processors/url.py``), the serve job runner (``serve/jobs.py``), and
the public API (``api.py``) each wrap it with their own concerns — progress
UI, job bookkeeping, result shaping — instead of keeping near-identical
copies of the cascade in sync by hand.

The cascade never prints, never raises for expected control flow (output
conflicts, LLM fallback), and returns everything callers need to build
their own result types. Fetch errors (``FetchError`` and friends) propagate
to the caller, whose handling differs per surface.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from markitai.constants import SCREENSHOTS_REL_PATH

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.fetch_cache import FetchCache
    from markitai.fetch_types import FetchResult
    from markitai.llm.processor import LLMProcessor

# Replaceable LLM stage: (markdown, url, cfg, output_file, fetch_result,
# processor) -> (llm_output_path, cost, usage, error_message). The CLI
# plugs its process_with_llm-based variant in; serve/api use the default.
LlmStageFn = Callable[
    [str, str, "MarkitaiConfig", Path, "FetchResult", "LLMProcessor | None"],
    Awaitable[tuple[Path | None, float, dict[str, dict[str, Any]], str | None]],
]


@dataclass
class UrlCascadeResult:
    """Outcome of one URL conversion.

    ``markdown`` is the base markdown after image localization (the same
    text written to ``output_path`` minus frontmatter). Fetch metadata is
    passed through so callers need not re-inspect the FetchResult.
    """

    markdown: str
    output_path: Path | None
    llm_output_path: Path | None
    skipped: bool = False
    skip_target: Path | None = None
    #: Resolved base-.md target path (post on_conflict), set unless skipped.
    #: Differs from ``output_path`` when keep_base is off and LLM succeeded
    #: (the base file was never written but .llm.md derives from it).
    target_file: Path | None = None
    llm_error: str | None = None
    cost_usd: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    title: str | None = None
    fetch_strategy_used: str | None = None
    screenshot_path: Path | None = None
    screenshot_tiles: list[Path] = field(default_factory=list)
    extra_meta: dict[str, Any] | None = None
    cache_hit: bool = False


async def convert_url_cascade(
    url: str,
    cfg: MarkitaiConfig,
    workdir: Path,
    *,
    processor: LLMProcessor | None = None,
    cache: FetchCache | None = None,
    screenshot_dir: Path | None = None,
    output_name: str | None = None,
    llm_error_policy: Literal["raise", "fallback"] = "fallback",
    fetch_result: FetchResult | None = None,
    markdown_override: str | None = None,
    base_from_localized: bool = True,
    llm_stage: LlmStageFn | None = None,
) -> UrlCascadeResult:
    """Convert one URL to markdown file(s) under ``workdir``.

    Stages: fetch (with optional cache/screenshot reuse) -> localize remote
    images when LLM image analysis is enabled -> LLM-enhanced ``.llm.md``
    -> base ``.md`` (always without LLM; with LLM for ``keep_base`` or as
    fallback). Output profiles post-process every written file.

    Args:
        url: http(s) URL to convert.
        cfg: Markitai configuration.
        workdir: Directory outputs and downloaded images are written into.
        processor: Shared LLM processor (serve reuses one per job). When
            ``None`` and LLM is enabled, one is created from ``cfg``.
        cache: Shared fetch cache; built from ``cfg.cache`` when ``None``.
        screenshot_dir: Shared screenshot directory; derived from
            ``workdir`` when screenshots are enabled and this is ``None``.
        output_name: Pre-assigned output filename (serve jobs assign unique
            names so colliding URLs never clobber each other); falls back
            to the URL-derived filename.
        llm_error_policy: ``"fallback"`` records the error on the result
            and still writes the base file; ``"raise"`` additionally raises
            ``ConversionError`` after the base file is on disk.
        fetch_result: Pre-fetched result (callers that branch on fetch
            metadata — the CLI's vision/screenshot-only paths — fetch
            themselves); skips the fetch stage when given.
        markdown_override: Caller-localized markdown; skips the image
            download stage and becomes the LLM input when given.
        base_from_localized: Base ``.md`` source — the image-localized
            markdown (serve/api behavior, default) or the original fetched
            content (CLI single-URL behavior).
        llm_stage: Replacement LLM stage; the default runs
            ``process_document``/``clean_document_pure`` and writes
            ``.llm.md`` itself.

    Returns:
        Cascade outcome; ``skipped`` is set when the output already exists
        and ``output.on_conflict`` resolves to skip.

    Raises:
        ConversionError: No content extracted, or LLM failure under the
            ``"raise"`` policy.
        FetchError: Fetch failures propagate unchanged.
    """
    from markitai import fetch as fetch_module
    from markitai.fetch import FetchStrategy
    from markitai.security import atomic_write_text
    from markitai.utils.cli_helpers import url_to_filename
    from markitai.utils.errors import ConversionError
    from markitai.utils.output import resolve_output_path
    from markitai.utils.paths import ensure_screenshots_dir
    from markitai.workflow.helpers import (
        add_basic_frontmatter,
        create_llm_processor,
        merge_llm_usage,
    )

    if fetch_result is None:
        if cache is None and cfg.cache.enabled:
            cache_dir = Path(cfg.cache.global_dir).expanduser()
            cache = fetch_module.get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)
        if screenshot_dir is None and cfg.screenshot.enabled:
            screenshot_dir = ensure_screenshots_dir(workdir)

        fetch_result = await fetch_module.fetch_url(
            url,
            FetchStrategy(cfg.fetch.strategy),
            cfg.fetch,
            cache=cache,
            skip_read_cache=cfg.cache.no_cache,
            screenshot=cfg.screenshot.enabled,
            screenshot_dir=screenshot_dir,
            screenshot_config=cfg.screenshot if cfg.screenshot.enabled else None,
        )

    markdown = fetch_result.content
    capture_only = cfg.screenshot.screenshot_only and not cfg.llm.enabled
    if capture_only:
        # The web result contract needs a Markdown wrapper, but its content
        # source is still only the capture, never the extracted text layer.
        if (
            fetch_result.screenshot_path is None
            or not fetch_result.screenshot_path.is_file()
        ):
            raise ConversionError(f"No screenshot captured from {url}")
    elif not markdown.strip():
        raise ConversionError(f"No content extracted from {url}")

    filename = output_name or url_to_filename(url)

    # Remote images are inputs to LLM image analysis; without LLM there is
    # nothing to analyze, so skip the downloads entirely.
    downloaded_images: list[Path] = []
    if markdown_override is not None:
        markdown = markdown_override
    elif cfg.llm.enabled and (cfg.image.alt_enabled or cfg.image.desc_enabled):
        from markitai.image import download_url_images

        download_result = await download_url_images(
            markdown=markdown,
            output_dir=workdir,
            base_url=url,
            config=cfg.image,
            source_name=filename.removesuffix(".md"),
        )
        markdown = download_result.updated_markdown
        downloaded_images = download_result.downloaded_paths

    output_file = resolve_output_path(workdir / filename, cfg.output.on_conflict)
    if output_file is None:
        return UrlCascadeResult(
            markdown=markdown,
            output_path=None,
            llm_output_path=None,
            skipped=True,
            skip_target=workdir / filename,
            title=fetch_result.title,
            fetch_strategy_used=fetch_result.strategy_used,
            screenshot_path=fetch_result.screenshot_path,
            screenshot_tiles=list(fetch_result.screenshot_tiles or []),
            extra_meta=fetch_result.metadata.get("source_frontmatter"),
            cache_hit=fetch_result.cache_hit,
        )

    title = fetch_result.title
    extra_meta = fetch_result.metadata.get("source_frontmatter")

    cost_usd = 0.0
    llm_usage: dict[str, dict[str, Any]] = {}
    llm_output_path: Path | None = None
    llm_error: str | None = None
    screenshot_only_mode = uses_screenshot_only(cfg, fetch_result)
    if cfg.llm.enabled:
        from markitai.utils.text import format_error_message

        proc = processor if processor is not None else create_llm_processor(cfg)
        try:
            if llm_stage is not None:
                (
                    llm_output_path,
                    stage_cost,
                    stage_usage,
                    stage_error,
                ) = await llm_stage(markdown, url, cfg, output_file, fetch_result, proc)
                cost_usd = stage_cost
                llm_usage = stage_usage
                llm_error = stage_error
            elif screenshot_only_mode:
                (
                    llm_output_path,
                    cost_usd,
                    llm_usage,
                    llm_error,
                ) = await _screenshot_only_llm_stage(
                    markdown, url, cfg, output_file, fetch_result, proc
                )
            elif uses_vision_enhancement(cfg, fetch_result):
                (
                    llm_output_path,
                    cost_usd,
                    llm_usage,
                    llm_error,
                ) = await _vision_llm_stage(
                    markdown, url, cfg, output_file, fetch_result, proc
                )
            else:
                (
                    llm_output_path,
                    cost_usd,
                    llm_usage,
                    llm_error,
                ) = await _default_document_stage(
                    markdown,
                    url,
                    cfg,
                    output_file,
                    fetch_result,
                    proc,
                    extra_meta=extra_meta,
                    title=title,
                )
        except Exception as e:
            # Same policy as the file pipeline: write the base .md as a
            # fallback below, then surface the failure per the policy.
            llm_error = format_error_message(e)
        finally:
            proc.clear_context_usage(url)

    # Base .md: always without LLM; with LLM for keep_base or as fallback.
    # Screenshot-only mode writes a screenshot-reference base instead of
    # the (empty) extracted text.
    output_path: Path | None = None
    if llm_output_path is None or cfg.llm.keep_base:
        if (
            screenshot_only_mode or capture_only
        ) and fetch_result.screenshot_path is not None:
            from markitai.utils.text import markdown_image_reference

            ref_files = fetch_result.screenshot_tiles or [fetch_result.screenshot_path]
            screenshot_ref = "\n\n".join(
                markdown_image_reference(
                    f"Screenshot {i + 1}" if len(ref_files) > 1 else "Screenshot",
                    f"{SCREENSHOTS_REL_PATH}/{t.name}",
                )
                for i, t in enumerate(ref_files)
            )
            if capture_only:
                markdown = screenshot_ref
            base_content = add_basic_frontmatter(
                screenshot_ref,
                url,
                fetch_strategy=fetch_result.strategy_used,
                screenshot_path=None,  # referenced above, not twice
                output_dir=workdir,
                title=title,
                extra_meta=extra_meta,
            )
        elif cfg.llm.pure and not cfg.llm.enabled:
            base_content = fetch_result.content
        else:
            base_markdown = markdown if base_from_localized else fetch_result.content
            base_content = add_basic_frontmatter(
                base_markdown,
                url,
                fetch_strategy=fetch_result.strategy_used,
                screenshot_path=fetch_result.screenshot_path,
                screenshot_tiles=list(fetch_result.screenshot_tiles or []),
                output_dir=workdir,
                title=title,
                extra_meta=extra_meta,
            )
        atomic_write_text(output_file, base_content)
        _apply_profile(output_file, workdir, cfg)
        output_path = output_file

    # Image analysis (alt/desc) on the written .llm.md — previously a CLI
    # only capability; the cascade runs it serially after the LLM stage.
    if (
        llm_output_path is not None
        and llm_output_path.exists()
        and (cfg.image.alt_enabled or cfg.image.desc_enabled)
        and downloaded_images
        and llm_error is None
    ):
        try:
            image_cost, image_usage = await _analyze_url_images_stage(
                cfg, workdir, llm_output_path, downloaded_images, proc, url
            )
            cost_usd += image_cost
            merge_llm_usage(llm_usage, image_usage)
        except Exception as e:
            logger.warning(
                f"[URL] Image analysis failed for {url}: {format_error_message(e)}"
            )

    # The profile moves assets and rewrites their references, so it runs
    # after every stage that writes image links. Profile failures are not
    # LLM failures: they surface as their own exception rather than through
    # llm_error_policy.
    if llm_output_path is not None and llm_output_path.exists():
        _apply_profile(llm_output_path, workdir, cfg)

    if llm_error is not None and llm_error_policy == "raise":
        raise ConversionError(f"LLM processing failed: {llm_error}")

    return UrlCascadeResult(
        markdown=markdown,
        output_path=output_path,
        llm_output_path=llm_output_path,
        target_file=output_file,
        llm_error=llm_error,
        cost_usd=cost_usd,
        llm_usage=llm_usage,
        title=title,
        fetch_strategy_used=fetch_result.strategy_used,
        screenshot_path=fetch_result.screenshot_path,
        screenshot_tiles=list(fetch_result.screenshot_tiles or []),
        extra_meta=extra_meta,
        cache_hit=fetch_result.cache_hit,
    )


def _apply_profile(md_file: Path, workdir: Path, cfg: MarkitaiConfig) -> None:
    """Post-process a written markdown file for the active output profile."""
    if cfg.output.profile is not None:
        from markitai.output_profiles import apply_profile_to_file

        apply_profile_to_file(md_file, workdir, cfg)


def uses_vision_enhancement(cfg: MarkitaiConfig, fetch_result: FetchResult) -> bool:
    """Multi-source vision enhancement: screenshot plus static/browser text.

    Shared by the cascade and by callers (the CLI) that pre-check which
    branch a fetch result will take.
    """
    return bool(
        cfg.llm.enabled
        and not cfg.llm.pure
        and fetch_result.screenshot_path
        and (
            fetch_result.static_content is not None
            or fetch_result.browser_content is not None
        )
    )


def uses_screenshot_only(cfg: MarkitaiConfig, fetch_result: FetchResult) -> bool:
    """Screenshot-only extraction: the vision model reads the page image(s).

    Shared by the cascade and by callers (the CLI) that pre-check which
    branch a fetch result will take.
    """
    return bool(
        cfg.llm.enabled
        and not cfg.llm.pure
        and cfg.screenshot.screenshot_only
        and fetch_result.screenshot_path
    )


_StageResult = tuple[Path | None, float, dict[str, dict[str, Any]], str | None]


async def _default_document_stage(
    markdown: str,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    fetch_result: FetchResult,
    proc: LLMProcessor,
    *,
    extra_meta: dict[str, Any] | None,
    title: str | None,
) -> _StageResult:
    """Standard text LLM stage: process_document/pure -> .llm.md."""
    from markitai.security import atomic_write_text

    if cfg.llm.pure:
        content = await proc.clean_document_pure(markdown, url)
    else:
        cleaned, llm_frontmatter = await proc.process_document(
            markdown,
            url,
            fetch_strategy=fetch_result.strategy_used,
            extra_meta=extra_meta,
            title=title,
        )
        content = proc.format_llm_output(cleaned, llm_frontmatter)
    target = output_file.with_suffix(".llm.md")
    atomic_write_text(target, content)
    return (
        target,
        proc.get_context_cost(url),
        proc.get_context_usage(url),
        None,
    )


async def _vision_llm_stage(
    markdown: str,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    fetch_result: FetchResult,
    proc: LLMProcessor,
) -> _StageResult:
    """Vision enhancement: screenshot as visual reference for the text.

    Falls back to the standard document stage when the vision call fails.
    """
    from markitai.security import atomic_write_text

    screenshot_path = fetch_result.screenshot_path
    assert screenshot_path is not None  # guaranteed by uses_vision_enhancement
    try:
        cleaned, frontmatter = await proc.enhance_url_with_vision(
            markdown,
            screenshot_path,
            context=url,
            original_title=fetch_result.title,
            fetch_strategy=fetch_result.strategy_used,
            extra_meta=fetch_result.metadata.get("source_frontmatter"),
        )
    except Exception as e:
        from markitai.utils.text import format_error_message

        logger.warning(
            f"[URL] Vision enhancement failed ({format_error_message(e)}); "
            "falling back to standard processing"
        )
        return await _default_document_stage(
            markdown,
            url,
            cfg,
            output_file,
            fetch_result,
            proc,
            extra_meta=fetch_result.metadata.get("source_frontmatter"),
            title=fetch_result.title,
        )

    content = proc.format_llm_output(cleaned, frontmatter)
    content += screenshot_reference_comment([screenshot_path])
    target = output_file.with_suffix(".llm.md")
    atomic_write_text(target, content)
    return (
        target,
        proc.get_context_cost(url),
        proc.get_context_usage(url),
        None,
    )


def screenshot_reference_comment(tiles: list[Path]) -> str:
    """HTML comment(s) pointing at the screenshot(s) an LLM output was read from.

    Downstream stages (page splitting, reference-image handling) look for
    this marker, so every vision stage appends it in the same shape.
    """
    if len(tiles) == 1:
        return (
            "\n\n<!-- Screenshot for reference -->\n"
            f"<!-- ![Screenshot]({SCREENSHOTS_REL_PATH}/{tiles[0].name}) -->"
        )
    return "\n\n<!-- Screenshots for reference (tiles) -->\n" + "\n".join(
        f"<!-- ![Screenshot {i + 1}]({SCREENSHOTS_REL_PATH}/{t.name}) -->"
        for i, t in enumerate(tiles)
    )


async def _screenshot_only_llm_stage(
    markdown: str,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    fetch_result: FetchResult,
    proc: LLMProcessor,
) -> _StageResult:
    """Screenshot-only extraction: the vision model reads the page tiles."""
    from markitai.security import atomic_write_text

    screenshot_path = fetch_result.screenshot_path
    assert screenshot_path is not None  # guaranteed by uses_screenshot_only
    tiles = list(fetch_result.screenshot_tiles or [screenshot_path])
    cleaned_parts: list[str] = []
    frontmatter = ""
    for i, tile in enumerate(tiles):
        cleaned, fm = await proc.extract_from_screenshot(
            tile,
            context=url,
            original_title=fetch_result.title if i == 0 else None,
        )
        if i == 0:
            frontmatter = fm
        if cleaned.strip():
            if len(tiles) > 1:
                cleaned_parts.append(f"<!-- Tile {i + 1} -->\n\n{cleaned}")
            else:
                cleaned_parts.append(cleaned)

    cleaned_content = "\n\n".join(cleaned_parts)
    content = proc.format_llm_output(cleaned_content, frontmatter)
    content += screenshot_reference_comment(tiles)
    target = output_file.with_suffix(".llm.md")
    atomic_write_text(target, content)
    return (
        target,
        proc.get_context_cost(url),
        proc.get_context_usage(url),
        None,
    )


async def _analyze_url_images_stage(
    cfg: MarkitaiConfig,
    workdir: Path,
    llm_md: Path,
    image_paths: list[Path],
    proc: LLMProcessor,
    url: str,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Analyze downloaded images: update alt text in the .llm.md and write
    ``images.json`` when descriptions are enabled.

    Serial counterpart of the CLI's concurrent image-analysis branches —
    correct first, fast enough for the serve/API surfaces that reach it.
    """
    from datetime import datetime

    from markitai.constants import ASSETS_REL_PATH
    from markitai.output_profiles import assets_visible
    from markitai.utils.text import image_ref_pattern, markdown_image_reference
    from markitai.workflow.helpers import (
        extract_document_context,
        write_images_json,
    )

    context = f"{llm_md.resolve()}:images"
    analyses = await proc.analyze_images_batch(
        image_paths,
        context=context,
        document_context=extract_document_context(llm_md.read_text(encoding="utf-8")),
    )
    timestamp = datetime.now().astimezone().isoformat()

    asset_descriptions: list[dict[str, Any]] = []
    llm_content = llm_md.read_text(encoding="utf-8")
    for image_path, analysis in zip(image_paths, analyses):
        if analysis is None:
            continue
        if cfg.image.desc_enabled:
            asset_descriptions.append(
                {
                    "asset": str(image_path.resolve()),
                    "alt": analysis.caption,
                    "desc": analysis.description,
                    "text": analysis.extracted_text or "",
                    "llm_usage": analysis.llm_usage or {},
                    "created": timestamp,
                }
            )
        if cfg.image.alt_enabled:
            old_pattern = image_ref_pattern(image_path.name)
            new_ref = markdown_image_reference(
                analysis.caption, f"{ASSETS_REL_PATH}/{image_path.name}"
            )
            llm_content = re.sub(old_pattern, new_ref, llm_content)

    if cfg.image.alt_enabled and llm_content != llm_md.read_text(encoding="utf-8"):
        from markitai.security import atomic_write_text

        atomic_write_text(llm_md, llm_content)

    if cfg.image.desc_enabled and asset_descriptions:
        from markitai.workflow.single import ImageAnalysisResult

        result = ImageAnalysisResult(
            source_file=llm_md.stem,
            assets=asset_descriptions,
        )
        write_images_json(workdir, [result], visible_assets=assets_visible(cfg))

    cost = proc.get_context_cost(context)
    usage = proc.get_context_usage(context)
    proc.clear_context_usage(context)
    return cost, usage

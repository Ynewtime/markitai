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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.fetch_cache import FetchCache
    from markitai.llm.processor import LLMProcessor


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
    from markitai.workflow.helpers import add_basic_frontmatter, create_llm_processor

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
    if not markdown.strip():
        raise ConversionError(f"No content extracted from {url}")

    filename = output_name or url_to_filename(url)

    # Remote images are inputs to LLM image analysis; without LLM there is
    # nothing to analyze, so skip the downloads entirely.
    if cfg.llm.enabled and (cfg.image.alt_enabled or cfg.image.desc_enabled):
        from markitai.image import download_url_images

        download_result = await download_url_images(
            markdown=markdown,
            output_dir=workdir,
            base_url=url,
            config=cfg.image,
            source_name=filename.removesuffix(".md"),
        )
        markdown = download_result.updated_markdown

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
    if cfg.llm.enabled:
        from markitai.utils.text import format_error_message

        proc = processor if processor is not None else create_llm_processor(cfg)
        try:
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
            llm_target = output_file.with_suffix(".llm.md")
            atomic_write_text(llm_target, content)
            _apply_profile(llm_target, workdir, cfg)
            llm_output_path = llm_target
        except Exception as e:
            # Same policy as the file pipeline: write the base .md as a
            # fallback below, then surface the failure per the policy.
            llm_error = format_error_message(e)
        finally:
            cost_usd = proc.get_context_cost(url)
            llm_usage = proc.get_context_usage(url)
            proc.clear_context_usage(url)

    # Base .md: always without LLM; with LLM for keep_base or as fallback.
    output_path: Path | None = None
    if llm_output_path is None or cfg.llm.keep_base:
        base_content = add_basic_frontmatter(
            markdown,
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

    if llm_error is not None and llm_error_policy == "raise":
        raise ConversionError(f"LLM processing failed: {llm_error}")

    return UrlCascadeResult(
        markdown=markdown,
        output_path=output_path,
        llm_output_path=llm_output_path,
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

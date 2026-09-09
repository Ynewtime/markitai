"""Public programmatic API for markitai (provisional).

This module is the supported way to run markitai conversions from Python
without going through the CLI::

    import markitai

    out = markitai.convert("report.pdf", output_dir="out/")
    print(out.markdown)

It is a thin, UI-free facade over the same orchestration the CLI and the
serve app use: local files go through ``workflow.core.convert_document_core``
and URLs follow the programmatic recipe established by ``serve.jobs``
(``fetch.fetch_url`` + ``workflow.helpers`` + ``LLMProcessor``).

Stability: **provisional** — markitai is 0.x and this API may change in
minor releases. Signatures and ``ConversionOutput`` fields are expected to
grow; existing fields will not be silently repurposed.

Layering: this module sits at the orchestration level, next to ``workflow``
and ``serve``, and must never import ``markitai.cli`` (enforced by the
import-linter contracts in the root ``pyproject.toml``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from markitai.types import LLMUsageByModel
from markitai.utils.errors import ConversionError
from markitai.utils.suppress import suppress_parser_noise

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.workflow.core import ConversionContext

__all__ = [
    "ConversionOutput",
    "ConversionUsage",
    "aconvert",
    "convert",
]

# Output profile names accepted by the ``profile`` keyword
OutputProfileName = Literal["rag", "obsidian", "okf"]


@dataclass
class ConversionUsage:
    """Aggregated LLM usage for one conversion (provisional).

    Attributes:
        cost_usd: Total LLM API cost in USD (0.0 without LLM enhancement).
        requests: Total number of LLM requests.
        input_tokens: Total input tokens across all models.
        output_tokens: Total output tokens across all models.
        by_model: Per-model breakdown, same shape as the workflow layer's
            usage dicts: ``{model: {requests, input_tokens, output_tokens,
            cost_usd}}``.
    """

    cost_usd: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: LLMUsageByModel = field(default_factory=dict)

    @classmethod
    def from_usage_dict(
        cls, cost_usd: float, by_model: dict[str, dict[str, Any]]
    ) -> ConversionUsage:
        """Build totals from a workflow-layer per-model usage dict."""
        usage = cls(cost_usd=cost_usd, by_model=by_model)  # type: ignore[arg-type]
        for stats in by_model.values():
            usage.requests += int(stats.get("requests", 0))
            usage.input_tokens += int(stats.get("input_tokens", 0))
            usage.output_tokens += int(stats.get("output_tokens", 0))
        return usage


@dataclass
class ConversionOutput:
    """Typed result of a single programmatic conversion (provisional).

    Attributes:
        source: The input as given — a local file path string or a URL.
        markdown: Base converted Markdown body (frontmatter block stripped).
        llm_markdown: LLM-enhanced Markdown body (frontmatter stripped), or
            None when LLM enhancement was disabled or produced no output.
        frontmatter: Parsed YAML frontmatter of the richest written output
            (the ``.llm.md`` variant when present, else the base ``.md``).
            Empty dict when no frontmatter was produced.
        output_path: Path to the written base ``.md`` file, or None when it
            was not written (in-memory mode, or LLM mode without
            ``llm.keep_base``).
        llm_output_path: Path to the written ``.llm.md`` file, or None.
        assets: Image files extracted for this conversion (under
            ``<output_dir>/.markitai/assets``, or ``<output_dir>/assets``
            when a rag/obsidian output profile is active). Empty in
            in-memory mode.
        screenshots: Page/slide screenshot files rendered for this
            conversion. Empty in in-memory mode.
        images: Per-image LLM analysis entries (the ``images.json`` payload)
            when alt/description analysis ran, else empty.
        usage: Aggregated LLM cost and token usage.
        skip_reason: Why the conversion was skipped (currently only
            ``"exists"`` under ``output.on_conflict = "skip"``), else None.
        duration: Wall-clock conversion time in seconds.

    Note:
        In in-memory mode (``output_dir=None``) intermediate files live in a
        deleted temp directory: all path fields are None, ``assets`` is
        empty, and image references inside the markdown keep their relative
        ``.markitai/assets/...`` form. Pass ``output_dir`` to keep assets.
    """

    source: str
    markdown: str
    llm_markdown: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    llm_output_path: Path | None = None
    assets: list[Path] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    usage: ConversionUsage = field(default_factory=ConversionUsage)
    skip_reason: str | None = None
    duration: float = 0.0


def _resolve_config(
    config: MarkitaiConfig | None,
    *,
    llm: bool | None,
    ocr: bool | None,
    screenshot: bool | None,
    alt: bool | None,
    desc: bool | None,
    profile: OutputProfileName | None = None,
) -> MarkitaiConfig:
    """Resolve the effective config for one conversion.

    Follows the CLI's precedence: config file (or the given config object)
    first, explicit keyword overrides on top, then the ``MODEL`` environment
    variable to auto-populate an empty model list when LLM is enabled.

    Args:
        config: Base configuration. None loads the same file hierarchy the
            CLI uses (``MARKITAI_CONFIG``, project ``.markitai.json``,
            ``~/.markitai/config.json``, built-in defaults).
        llm: Override for ``llm.enabled`` (None keeps the config value).
        ocr: Override for ``ocr.enabled``.
        screenshot: Override for ``screenshot.enabled``.
        alt: Override for ``image.alt_enabled``.
        desc: Override for ``image.desc_enabled``.
        profile: Override for ``output.profile``.

    Returns:
        A private config copy; the caller's object is never mutated.

    Raises:
        ValueError: If LLM is enabled but no model can be resolved.
    """
    from markitai.config import ConfigManager

    if config is None:
        cfg = ConfigManager().load()
    else:
        cfg = config.model_copy(deep=True)

    if llm is not None:
        cfg.llm.enabled = llm
    if ocr is not None:
        cfg.ocr.enabled = ocr
    if screenshot is not None:
        cfg.screenshot.enabled = screenshot
    if alt is not None:
        cfg.image.alt_enabled = alt
    if desc is not None:
        cfg.image.desc_enabled = desc
    if profile is not None:
        cfg.output.profile = profile

    # Mirror the CLI's MODEL env var fallback for an empty model list
    if cfg.llm.enabled and not cfg.llm.model_list:
        model_env = os.environ.get("MODEL")
        if model_env:
            from markitai.config import LiteLLMParams, ModelConfig

            cfg.llm.model_list = [
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=model_env),
                )
            ]
            logger.debug("[API] Using MODEL env var: {}", model_env)
        else:
            raise ValueError(
                "LLM enhancement is enabled but no models are configured. "
                "Set the MODEL environment variable (e.g. MODEL=openai/gpt-4o-mini), "
                "add models to llm.model_list in your markitai config file, or "
                "pass a config with llm.model_list set."
            )

    return cfg


def _parse_output_file(path: Path) -> tuple[dict[str, Any], str]:
    """Read a written output file into (frontmatter dict, markdown body)."""
    from markitai.utils.frontmatter import split_frontmatter

    raw, body = split_frontmatter(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}, body

    import yaml

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}, body
    return (data if isinstance(data, dict) else {}), body


def _referenced_assets(
    markdown: str, workdir: Path, *, visible: bool = False
) -> list[Path]:
    """Resolve asset image refs in markdown to existing files.

    Args:
        markdown: Markdown content to scan for asset references.
        workdir: Output directory the assets live under.
        visible: Look in the profile-visible ``assets/`` directory instead
            of the default hidden ``.markitai/assets/`` one.
    """
    from markitai.constants import ASSETS_REL_PATH, VISIBLE_ASSETS_REL_PATH

    if visible:
        from markitai.output_profiles import visible_asset_names as extract_names

        assets_dir = workdir / VISIBLE_ASSETS_REL_PATH
    else:
        from markitai.utils.text import extract_asset_image_names as extract_names

        assets_dir = workdir / ASSETS_REL_PATH
    return [
        assets_dir / name
        for name in extract_names(markdown)
        if (assets_dir / name).is_file()
    ]


def _build_file_output(
    ctx: ConversionContext,
    source: str,
    workdir: Path,
    *,
    in_memory: bool,
) -> ConversionOutput:
    """Assemble a ConversionOutput from a completed file conversion context."""
    from markitai.constants import SCREENSHOTS_REL_PATH
    from markitai.workflow.core import get_saved_images

    assert ctx.conversion_result is not None  # guaranteed by successful pipeline

    markdown = ctx.conversion_result.markdown
    frontmatter: dict[str, Any] = {}
    llm_markdown: str | None = None
    output_path: Path | None = None
    llm_output_path: Path | None = None

    if ctx.output_file is not None:
        llm_file = ctx.output_file.with_suffix(".llm.md")
        if llm_file.exists():
            llm_output_path = llm_file
            frontmatter, llm_markdown = _parse_output_file(llm_file)
        if ctx.output_file.exists():
            output_path = ctx.output_file
            base_frontmatter, markdown = _parse_output_file(ctx.output_file)
            if not frontmatter:
                frontmatter = base_frontmatter

    from markitai.output_profiles import assets_visible

    if assets_visible(ctx.config):
        # An asset-visible profile moved images to assets/ and rewrote refs
        combined = "\n".join(part for part in (markdown, llm_markdown) if part)
        assets = _referenced_assets(combined, workdir, visible=True)
    else:
        assets = get_saved_images(ctx)
    screenshots_dir = workdir / SCREENSHOTS_REL_PATH
    page_images = ctx.conversion_result.metadata.get("page_images", [])
    screenshots = [
        screenshots_dir / img["name"]
        for img in page_images
        if isinstance(img, dict)
        and "name" in img
        and (screenshots_dir / img["name"]).is_file()
    ]

    images: list[dict[str, Any]] = []
    if ctx.image_analysis is not None:
        images = list(ctx.image_analysis.assets)

    if in_memory:
        output_path = None
        llm_output_path = None
        assets = []
        screenshots = []

    return ConversionOutput(
        source=source,
        markdown=markdown,
        llm_markdown=llm_markdown,
        frontmatter=frontmatter,
        output_path=output_path,
        llm_output_path=llm_output_path,
        assets=assets,
        screenshots=screenshots,
        images=images,
        usage=ConversionUsage.from_usage_dict(ctx.llm_cost, ctx.llm_usage),
    )


async def _aconvert_file(
    path: Path, cfg: MarkitaiConfig, workdir: Path, *, in_memory: bool
) -> ConversionOutput:
    """Convert one local file via ``workflow.core.convert_document_core``."""
    from markitai.constants import MAX_DOCUMENT_SIZE
    from markitai.utils.paths import derive_output_name
    from markitai.workflow.core import ConversionContext, convert_document_core

    ctx = ConversionContext(input_path=path, output_dir=workdir, config=cfg)
    result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

    if not result.success:
        raise ConversionError(result.error or "Unknown conversion error")

    if result.skip_reason == "image_only":
        raise ConversionError(
            f"{path.name} is an image file with no text to extract. "
            f"Enable LLM (llm=True) or OCR (ocr=True) for content extraction."
        )

    if result.skip_reason == "exists":
        existing = workdir / derive_output_name(path.name)
        return _skipped_output(str(path), existing)

    if cfg.image.desc_enabled and ctx.image_analysis is not None:
        from markitai.output_profiles import assets_visible
        from markitai.workflow.helpers import write_images_json

        write_images_json(
            workdir, [ctx.image_analysis], visible_assets=assets_visible(cfg)
        )

    return _build_file_output(ctx, str(path), workdir, in_memory=in_memory)


def _skipped_output(source: str, existing: Path) -> ConversionOutput:
    """Build the result for an ``on_conflict = "skip"`` early exit."""
    frontmatter: dict[str, Any] = {}
    markdown = ""
    llm_markdown: str | None = None
    llm_file = existing.with_suffix(".llm.md")
    llm_output_path: Path | None = None
    output_path: Path | None = None
    if llm_file.exists():
        llm_output_path = llm_file
        frontmatter, llm_markdown = _parse_output_file(llm_file)
    if existing.exists():
        output_path = existing
        base_frontmatter, markdown = _parse_output_file(existing)
        if not frontmatter:
            frontmatter = base_frontmatter
    return ConversionOutput(
        source=source,
        markdown=markdown,
        llm_markdown=llm_markdown,
        frontmatter=frontmatter,
        output_path=output_path,
        llm_output_path=llm_output_path,
        skip_reason="exists",
    )


async def _aconvert_url(
    url: str, cfg: MarkitaiConfig, workdir: Path, *, in_memory: bool
) -> ConversionOutput:
    """Convert one URL via the shared cascade, shaping a ConversionOutput.

    Thin API wrapper over ``workflow.url.convert_url_cascade`` — the
    cascade owns fetch/images/LLM/frontmatter/profile; this wrapper owns
    API-facing concerns: frontmatter/asset extraction from the written
    files, in-memory mode, and the raise-on-LLM-failure policy. The CLI's
    vision/screenshot-only URL branches are not replicated.
    """
    from markitai.output_profiles import assets_visible
    from markitai.workflow.url import convert_url_cascade

    result = await convert_url_cascade(
        url,
        cfg,
        workdir,
        llm_error_policy="raise",
    )

    if result.skipped:
        assert result.skip_target is not None
        return _skipped_output(url, result.skip_target)

    # Read back frontmatter/body from the written files (post-profile, so
    # the returned markdown matches what is on disk).
    frontmatter: dict[str, Any] = {}
    markdown = result.markdown
    llm_markdown: str | None = None
    if result.llm_output_path is not None:
        frontmatter, llm_markdown = _parse_output_file(result.llm_output_path)
    if result.output_path is not None:
        if not frontmatter:
            frontmatter, markdown = _parse_output_file(result.output_path)
        elif cfg.output.profile is not None:
            _, markdown = _parse_output_file(result.output_path)

    output_path = result.output_path
    llm_output_path = result.llm_output_path
    if assets_visible(cfg):
        combined = "\n".join(part for part in (markdown, llm_markdown) if part)
        assets = _referenced_assets(combined, workdir, visible=True)
    else:
        assets = _referenced_assets(markdown, workdir)
    screenshots = [result.screenshot_path] if result.screenshot_path else []

    if in_memory:
        output_path = None
        llm_output_path = None
        assets = []
        screenshots = []

    return ConversionOutput(
        source=url,
        markdown=markdown,
        llm_markdown=llm_markdown,
        frontmatter=frontmatter,
        output_path=output_path,
        llm_output_path=llm_output_path,
        assets=assets,
        screenshots=screenshots,
        usage=ConversionUsage.from_usage_dict(result.cost_usd, result.llm_usage),
    )


async def aconvert(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: MarkitaiConfig | None = None,
    llm: bool | None = None,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
    profile: OutputProfileName | None = None,
) -> ConversionOutput:
    """Convert one local file or URL to Markdown (async, provisional).

    The event loop stays responsive throughout: CPU-bound converter work
    (PyMuPDF, ONNX, Office extraction) runs in the shared converter thread
    pool, and LLM/fetch work is natively async.

    Args:
        source: Local file path or ``http(s)://`` URL.
        output_dir: Directory to write outputs into (created if missing).
            None converts in a private temp directory and returns the
            markdown in memory only — see ``ConversionOutput`` notes.
        config: Base ``MarkitaiConfig``. None loads the same config file
            hierarchy the CLI uses.
        llm: Enable LLM enhancement (None keeps the config value).
        ocr: Enable OCR for scanned documents/images.
        screenshot: Enable page screenshots (PDF/Office/URLs).
        alt: Enable LLM alt-text generation for images.
        desc: Enable LLM image descriptions (writes ``images.json``).
        profile: Output profile ("rag", "obsidian", or "okf"; None keeps
            the config value). Shapes the written output for a downstream
            consumer — with "rag"/"obsidian", assets land in a visible
            ``assets/`` directory instead of ``.markitai/assets/``.

    Returns:
        A ``ConversionOutput`` with the converted markdown and metadata.

    Raises:
        ConversionError: The conversion pipeline failed, or produced no
            content.
        FetchError: A URL could not be fetched.
        FileNotFoundError: The source path does not exist.
        IsADirectoryError: The source is a directory (batch conversion is
            CLI-only for now).
        ValueError: LLM was enabled with no resolvable model.
    """
    # Native noise suppression must precede converter imports; run it off
    # the loop because it may import pymupdf (a slow C extension import)
    await asyncio.to_thread(suppress_parser_noise)

    cfg = _resolve_config(
        config,
        llm=llm,
        ocr=ocr,
        screenshot=screenshot,
        alt=alt,
        desc=desc,
        profile=profile,
    )

    from markitai.utils.cli_helpers import is_url

    src = str(source)
    started = time.time()

    in_memory = output_dir is None
    if in_memory:
        workdir = Path(tempfile.mkdtemp(prefix="markitai_"))
    else:
        workdir = Path(output_dir).expanduser()

    try:
        if is_url(src):
            from markitai.security import check_symlink_safety
            from markitai.utils.paths import ensure_dir

            check_symlink_safety(workdir, allow_symlinks=cfg.output.allow_symlinks)
            ensure_dir(workdir)
            result = await _aconvert_url(src, cfg, workdir, in_memory=in_memory)
        else:
            path = Path(source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Source path does not exist: {path}")
            if path.is_dir():
                raise IsADirectoryError(
                    f"{path} is a directory; the programmatic API converts one "
                    f"file or URL per call (use the CLI for directory batches)."
                )
            result = await _aconvert_file(path, cfg, workdir, in_memory=in_memory)
    finally:
        if in_memory:
            shutil.rmtree(workdir, ignore_errors=True)

    result.duration = time.time() - started
    return result


def convert(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: MarkitaiConfig | None = None,
    llm: bool | None = None,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
    profile: OutputProfileName | None = None,
) -> ConversionOutput:
    """Convert one local file or URL to Markdown (sync, provisional).

    Blocking wrapper around :func:`aconvert` for scripts and notebooks
    without an event loop. Runs the conversion in a fresh loop and releases
    loop-bound shared resources afterwards, so repeated calls in one
    process are safe. See :func:`aconvert` for parameters, return value,
    and raised exceptions.

    Raises:
        RuntimeError: If called from a running event loop — ``await
            markitai.aconvert(...)`` instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "markitai.convert() cannot be called from a running event loop; "
            "use `await markitai.aconvert(...)` instead"
        )

    async def _run() -> ConversionOutput:
        try:
            return await aconvert(
                source,
                output_dir=output_dir,
                config=config,
                llm=llm,
                ocr=ocr,
                screenshot=screenshot,
                alt=alt,
                desc=desc,
                profile=profile,
            )
        finally:
            await _close_loop_bound_resources()

    return asyncio.run(_run())


async def _close_loop_bound_resources() -> None:
    """Release shared state bound to the closing event loop.

    Mirrors the CLI's end-of-run cleanup (``run_workflow_with_cleanup``)
    minus the converter thread pool, which is loop-independent and stays
    warm for subsequent calls. Only touches subsystems that were actually
    imported, so file-only conversions never pay for fetch/LLM teardown.
    """
    if "markitai.fetch_session" in sys.modules:
        from markitai.fetch_session import get_default_session

        await get_default_session().close()
    else:
        from markitai.utils.executor import reset_heavy_task_semaphore

        reset_heavy_task_semaphore()

    if "litellm" in sys.modules:
        try:
            from litellm.llms.custom_httpx.async_client_cleanup import (
                close_litellm_async_clients,
            )

            await close_litellm_async_clients()
        except Exception as e:
            logger.debug("[API] LiteLLM client cleanup failed: {}", e)

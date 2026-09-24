"""Core document conversion logic.

This module provides the unified core conversion flow shared between
single-file and batch processing modes.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from loguru import logger

from markitai.constants import (
    IMAGE_EXTENSIONS,
    MARKITAI_META_DIR,
    SCREENSHOTS_REL_PATH,
)
from markitai.converter.base import (
    FileFormat,
    detect_format,
    get_converter,
    unsupported_format_message,
)
from markitai.security import (
    atomic_write_text,
    check_symlink_safety,
    validate_file_size,
)
from markitai.utils.frontmatter import split_frontmatter
from markitai.utils.paths import ensure_dir
from markitai.utils.text import format_error_message, markdown_image_reference
from markitai.workflow.helpers import (
    add_basic_frontmatter,
    append_reference_image_comments,
    is_failed_image_entry,
    merge_llm_usage,
)

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.converter.base import ConvertResult
    from markitai.llm import LLMProcessor
    from markitai.workflow.single import ImageAnalysisResult


@dataclass
class ConversionContext:
    """Context for a document conversion operation.

    This dataclass holds all the input parameters and intermediate state
    for a single document conversion.
    """

    # Required inputs
    input_path: Path
    output_dir: Path
    config: MarkitaiConfig

    # Optional inputs
    shared_processor: LLMProcessor | None = None
    output_name: str | None = None  # Pre-planned output filename (batch collision)

    # Processing flags
    use_multiprocess_images: bool = False
    paged_stabilized: bool = False

    # Intermediate state (set during processing)
    detected_format: FileFormat | None = None
    converter: Any = None
    conversion_result: ConvertResult | None = None
    output_file: Path | None = None
    llm_output_file: Path | None = None  # set only by a successful LLM write
    # Whether this run wrote the base .md: a file already on disk may be a
    # previous run's output (on_conflict=overwrite), not this conversion's
    base_written: bool = False
    embedded_images_count: int = 0
    screenshots_count: int = 0

    # LLM tracking
    llm_cost: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_analysis: ImageAnalysisResult | None = None
    # Problems that did not fail the item (an image whose analysis failed
    # and kept its alt text); surfaced as the item's result warnings
    warnings: list[str] = field(default_factory=list)

    # Additional tracking (for caller use)
    duration: float = 0.0
    cache_hit: bool = False
    input_base_dir: Path | None = None  # For batch relative path calculation

    # Optional callback for stage completion (stage_name, duration)
    on_stage_complete: Callable[[str, float], None] | None = None


@dataclass
class ConversionStepResult:
    """Result of a conversion step."""

    success: bool
    error: str | None = None
    skip_reason: str | None = None


async def run_in_converter_thread(func, *args, **kwargs):
    """Run a converter function in the shared thread pool.

    Uses the shared ThreadPoolExecutor from utils.executor to avoid
    creating a new executor for each conversion.
    """
    from markitai.utils.executor import run_in_converter_thread as _run_in_thread

    return await _run_in_thread(func, *args, **kwargs)


def validate_and_detect_format(
    ctx: ConversionContext, max_size: int
) -> ConversionStepResult:
    """Validate file size and detect format.

    Args:
        ctx: Conversion context
        max_size: Maximum file size in bytes

    Returns:
        ConversionStepResult indicating success or failure
    """
    try:
        validate_file_size(ctx.input_path, max_size)
    except ValueError as e:
        return ConversionStepResult(success=False, error=str(e))

    fmt = detect_format(ctx.input_path)
    ctx.detected_format = fmt
    if fmt == FileFormat.UNKNOWN:
        return ConversionStepResult(
            success=False, error=unsupported_format_message(ctx.input_path)
        )

    # Check if Cloudflare toMarkdown is explicitly enabled (-b cloudflare)
    cf_config = (
        ctx.config.fetch.cloudflare if hasattr(ctx.config.fetch, "cloudflare") else None
    )
    cf_forced = cf_config and cf_config.convert_enabled

    if cf_forced and cf_config is not None:
        # -b cloudflare: prefer CF converter over local converters
        from markitai.converter.cloudflare import (
            CF_SUPPORTED_FORMATS,
            CloudflareConverter,
        )

        if fmt in CF_SUPPORTED_FORMATS:
            api_token = cf_config.get_resolved_api_token()
            account_id = cf_config.get_resolved_account_id()
            if api_token and account_id:
                # Warn if a higher-quality local converter exists
                local_converter = get_converter(ctx.input_path, config=ctx.config)
                if local_converter is not None:
                    logger.warning(
                        f"Using Cloudflare toMarkdown for {fmt.value} "
                        f"(local converter available — output quality may be lower)"
                    )
                ctx.converter = CloudflareConverter(
                    api_token=api_token,
                    account_id=account_id,
                    config=ctx.config,
                )
                logger.debug(f"Using CF toMarkdown for {fmt.value} (explicit)")
            else:
                from markitai.utils.guidance import cloudflare_credentials_error

                return ConversionStepResult(
                    success=False,
                    error=cloudflare_credentials_error(),
                )

    # Fall back to local converter if CF not used
    if ctx.converter is None:
        ctx.converter = get_converter(ctx.input_path, config=ctx.config)

    if ctx.converter is None:
        return ConversionStepResult(
            success=False, error=f"No converter available for format: {fmt.value}"
        )

    return ConversionStepResult(success=True)


def prepare_output_directory(ctx: ConversionContext) -> ConversionStepResult:
    """Create output directory with symlink safety check.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    try:
        check_symlink_safety(
            ctx.output_dir, allow_symlinks=ctx.config.output.allow_symlinks
        )
        ensure_dir(ctx.output_dir)
        return ConversionStepResult(success=True)
    except Exception as e:
        return ConversionStepResult(success=False, error=str(e))


async def convert_document(ctx: ConversionContext) -> ConversionStepResult:
    """Execute document conversion.

    Uses heavy task semaphore for memory-intensive formats (PPT, PDF with screenshots)
    to prevent OOM in concurrent environments.
    """
    from markitai.utils.executor import get_heavy_task_semaphore

    try:
        # Determine if this is a heavy conversion task
        # Heavy tasks include:
        # - Legacy formats (.ppt, .doc) that need an Office app or LibreOffice
        #   (.xls converts in-process via xlrd and is not heavy)
        # - PDF/PPTX/DOCX with screenshots enabled (page rendering)
        # - PDF with OCR, local or +LLM (renders every page; local RapidOCR
        #   holds a full-page bitmap per worker, several GB for a long scan)
        # - Images with local OCR (RapidOCR inference on the full bitmap;
        #   with --llm the vision model reads them instead, unless
        #   MARKITAI_NO_VLM_OCR sends them back to RapidOCR)
        # - PPTX with OCR (renders slide images)
        from markitai.vision_consent import vlm_ocr_allowed

        ext = ctx.input_path.suffix.lower()
        legacy_formats = {".ppt", ".doc"}
        heavy_extensions = {".ppt", ".pptx", ".pdf", ".doc", ".docx"}
        ocr_image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".svg",
            ".heic",
            ".heif",
            ".avif",
        }
        use_ocr = ctx.config.ocr.enabled
        use_llm = ctx.config.llm.enabled
        local_image_ocr = use_ocr and (not use_llm or not vlm_ocr_allowed())

        is_heavy = (
            ext in legacy_formats
            or (ext in heavy_extensions and ctx.config.screenshot.enabled)
            or (ext == ".pdf" and use_ocr)
            or (ext in ocr_image_extensions and local_image_ocr)
            or (ext == ".pptx" and use_ocr)
        )

        logger.debug(
            f"Converting {ctx.input_path.name}..." + (" [HEAVY]" if is_heavy else "")
        )

        # Lightweight text formats do not benefit from thread offloading and
        # have shown unstable behavior under the shared executor in tests.
        lightweight_text_extensions = {".txt", ".md", ".markdown"}

        # CloudflareConverter has a native async API — dispatch directly
        from markitai.converter.cloudflare import CloudflareConverter

        if isinstance(ctx.converter, CloudflareConverter):
            ctx.conversion_result = await ctx.converter.convert_async(
                ctx.input_path,
                output_dir=ctx.output_dir,
            )
        elif ext in lightweight_text_extensions:
            ctx.conversion_result = ctx.converter.convert(
                ctx.input_path,
                output_dir=ctx.output_dir,
            )
        elif is_heavy:
            async with get_heavy_task_semaphore(ctx.config.batch.heavy_task_limit):
                ctx.conversion_result = await run_in_converter_thread(
                    ctx.converter.convert,
                    ctx.input_path,
                    output_dir=ctx.output_dir,
                )
        else:
            ctx.conversion_result = await run_in_converter_thread(
                ctx.converter.convert,
                ctx.input_path,
                output_dir=ctx.output_dir,
            )
        return ConversionStepResult(success=True)
    except Exception as e:
        # The step error below flattens the exception into a user-facing
        # string (self-explanatory errors even drop their class name), so
        # keep the full type + traceback retrievable in the DEBUG log.
        logger.opt(exception=True).debug(
            "[Convert] {} failed with {}", ctx.input_path.name, type(e).__name__
        )
        return ConversionStepResult(
            success=False, error=f"Conversion failed: {format_error_message(e)}"
        )


def resolve_output_file(ctx: ConversionContext) -> ConversionStepResult:
    """Resolve output file path with conflict handling.

    Output naming appends ``.md`` to the input filename (``sample.pdf`` ->
    ``sample.pdf.md``); ``ctx.output_name`` overrides it (explicit ``-o``
    file targets, URL entries with custom names). Inside a batch the name
    is claimed from the batch's reservation table (see
    ``utils.output.output_claim_scope``), so no other item of the batch can
    pick it while this one converts.

    Once the name is known, the converter is told the asset prefix derived
    from it (see :func:`asset_base_name`), so a renamed ``report.pdf.v2.md``
    extracts ``report.pdf.v2-…`` images instead of overwriting the ones
    ``report.pdf.md`` still references.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult - may have skip_reason if file exists
    """
    from markitai.utils.output import resolve_item_output_path
    from markitai.utils.paths import derive_output_name

    output_name = ctx.output_name
    if output_name is None:
        output_name = derive_output_name(ctx.input_path.name)

    base_output_file = ctx.output_dir / output_name
    ctx.output_file = resolve_item_output_path(
        base_output_file, ctx.config.output.on_conflict
    )

    if ctx.output_file is None:
        logger.debug(f"[SKIP] Output exists: {base_output_file}")
        return ConversionStepResult(success=True, skip_reason="exists")

    if ctx.converter is not None:
        ctx.converter.asset_prefix = asset_base_name(ctx)

    return ConversionStepResult(success=True)


def asset_base_name(ctx: ConversionContext) -> str:
    """Prefix for the asset files this conversion writes.

    Derived from the resolved output name (``report.pdf.v2.md`` ->
    ``report.pdf.v2``), not the input name: assets live in one shared
    ``.markitai/assets/`` directory, so an input-named prefix let a renamed
    re-run overwrite images an older output still references. Before the
    output is resolved, the input filename is used.
    """
    from markitai.utils.output import split_markdown_name

    if ctx.output_file is None:
        return ctx.input_path.name
    return split_markdown_name(ctx.output_file.name)[0]


async def process_embedded_images(ctx: ConversionContext) -> ConversionStepResult:
    """Extract and process embedded images from markdown.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None:
        return ConversionStepResult(success=False, error="No conversion result")

    conversion_result = ctx.conversion_result
    # Rendered pages/slides are screenshots, never embedded images. A path
    # that must not expose page_images (the local-OCR slide render) reports
    # its count as screenshot_count instead.
    ctx.screenshots_count = len(
        conversion_result.metadata.get("page_images", [])
    ) or int(conversion_result.metadata.get("screenshot_count", 0))
    ctx.embedded_images_count = len(conversion_result.images)
    if "data:image" not in conversion_result.markdown:
        return ConversionStepResult(success=True)

    from markitai.image import ImageProcessor

    image_processor = ImageProcessor(config=ctx.config.image)
    base64_images = image_processor.extract_base64_images(conversion_result.markdown)

    # Add inline images to those already saved by the converter.
    converter_images = ctx.embedded_images_count
    ctx.embedded_images_count = len(base64_images) + converter_images

    if base64_images:
        logger.debug(f"Processing {len(base64_images)} embedded images...")

        # Use multiprocess for large batches if enabled
        from markitai.constants import DEFAULT_IMAGE_MULTIPROCESS_THRESHOLD

        if (
            ctx.use_multiprocess_images
            and len(base64_images) > DEFAULT_IMAGE_MULTIPROCESS_THRESHOLD
        ):
            image_result = await image_processor.process_and_save_multiprocess(
                base64_images,
                output_dir=ctx.output_dir,
                base_name=asset_base_name(ctx),
            )
        else:
            image_result = await asyncio.to_thread(
                lambda: image_processor.process_and_save(
                    base64_images,
                    output_dir=ctx.output_dir,
                    base_name=asset_base_name(ctx),
                )
            )

        # Update markdown with image paths using index mapping for correct replacement
        conversion_result.markdown = await asyncio.to_thread(
            lambda: image_processor.replace_base64_with_paths(
                conversion_result.markdown,
                image_result.saved_images,
                index_mapping=image_result.index_mapping,
            )
        )

        # Also update extracted_text in metadata if present (for PPTX+LLM mode)
        if "extracted_text" in conversion_result.metadata:
            conversion_result.metadata["extracted_text"] = await asyncio.to_thread(
                lambda: image_processor.replace_base64_with_paths(
                    conversion_result.metadata["extracted_text"],
                    image_result.saved_images,
                    index_mapping=image_result.index_mapping,
                )
            )

        # Update count: saved base64 images + converter-extracted images
        ctx.embedded_images_count = len(image_result.saved_images) + converter_images

    return ConversionStepResult(success=True)


def _base_markdown_content(ctx: ConversionContext) -> str:
    """The base ``.md`` content: raw in pure mode, with frontmatter otherwise.

    ``--pure`` means "no frontmatter" for every base file markitai writes —
    with or without LLM, the ``--keep-base`` copy and the LLM-failure
    fallback alike.
    """
    assert ctx.conversion_result is not None
    base_markdown = append_reference_image_comments(
        ctx.conversion_result.markdown,
        ctx.conversion_result.metadata.get("reference_images"),
    )
    if ctx.config.llm.pure:
        return base_markdown
    title = ctx.conversion_result.metadata.get("title")
    return add_basic_frontmatter(
        base_markdown,
        ctx.input_path.name,
        title=title if isinstance(title, str) else None,
    )


def write_base_markdown(ctx: ConversionContext) -> ConversionStepResult:
    """Write base markdown file with basic frontmatter.

    Decision tree:
    1. LLM enabled without --keep-base: skip writing (in-memory only)
    2. Pure mode (with or without LLM): write raw markdown without frontmatter
    3. Default: write with frontmatter

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(
            success=False, error="Missing conversion result or output file"
        )

    # In LLM mode, skip writing .md unless --keep-base is set
    if ctx.config.llm.enabled and not ctx.config.llm.keep_base:
        logger.debug(
            f"[Core] Skipped writing base .md (LLM mode, no --keep-base): "
            f"{ctx.output_file}"
        )
        return ConversionStepResult(success=True)

    atomic_write_text(ctx.output_file, _base_markdown_content(ctx))
    ctx.base_written = True
    logger.debug(f"Written output: {ctx.output_file}")

    return ConversionStepResult(success=True)


def _write_base_md_fallback(ctx: ConversionContext) -> None:
    """Write base .md as fallback when LLM processing fails.

    Called when LLM mode is active but LLM processing fails. Ensures the user
    gets at least the base conversion result instead of nothing.

    The base is written unless this run already wrote it (``--keep-base``):
    a file merely present on disk may be a previous run's output that
    ``on_conflict=overwrite`` is replacing, and keeping it would hand the
    user stale content next to a failed item. Under ``overwrite`` the
    entry's own ``.llm.md`` goes too — the failed run produced none, so a
    leftover one is the previous run's. Under ``rename`` the resolved name
    is fresh, so a same-named ``.llm.md`` is another output's and stays.

    Args:
        ctx: Conversion context
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return
    if not ctx.base_written:
        atomic_write_text(ctx.output_file, _base_markdown_content(ctx))
        ctx.base_written = True
        logger.warning(
            f"[Core] LLM processing failed, wrote base .md as fallback: "
            f"{ctx.output_file}"
        )
    if ctx.config.output.on_conflict == "overwrite" and ctx.llm_output_file is None:
        stale_llm_output = ctx.output_file.with_suffix(".llm.md")
        try:
            stale_llm_output.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"[Core] Could not remove stale {stale_llm_output}: {e}")
        else:
            logger.debug(f"[Core] Removed stale LLM output: {stale_llm_output}")

    # Every failure path returns before the pipeline's profile step, so
    # without this the one document whose LLM call failed keeps the default
    # layout while the rest of the batch gets the profile's — a reader
    # consuming the directory by profile then trips on that one file.
    apply_output_profile(ctx)


def get_saved_images(ctx: ConversionContext) -> list[Path]:
    """Get list of saved images for this file from assets directory.

    Handles transcoded formats (e.g. BMP/TIFF→PNG) by checking the
    ``asset_path`` metadata from the converter first. Otherwise resolves
    the image refs found in the converted markdown (plus demoted
    ``reference_images`` metadata): refs are written by the converter, so
    they always match on-disk names even when the extractor sanitizes the
    source filename (pymupdf4llm rewrites spaces/parentheses, so a prefix
    glob on the input name would match nothing). Falls back to the assets
    named after this output when the markdown references none: exactly the
    names markitai's converters write (see :func:`_own_asset_pattern`), so
    ``a.pdf`` never claims a renamed sibling's ``a.pdf.v2-0001-01.jpg``.

    Args:
        ctx: Conversion context

    Returns:
        List of image file paths
    """
    assets_dir = ctx.output_dir / MARKITAI_META_DIR / "assets"
    if not assets_dir.exists():
        return []

    # Check converter metadata for explicit asset_path (handles transcoded files)
    if ctx.conversion_result and "asset_path" in ctx.conversion_result.metadata:
        asset_path = ctx.output_dir / ctx.conversion_result.metadata["asset_path"]
        if asset_path.exists() and asset_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [asset_path]

    if ctx.conversion_result is not None:
        from markitai.utils.text import extract_asset_image_names

        ref_names = extract_asset_image_names(ctx.conversion_result.markdown)
        reference_images = ctx.conversion_result.metadata.get("reference_images")
        if isinstance(reference_images, list):
            for ref in reference_images:
                name = ref.get("name") if isinstance(ref, dict) else None
                if isinstance(name, str) and name and name not in ref_names:
                    ref_names.append(name)
        if ref_names:
            found = [
                assets_dir / name
                for name in ref_names
                if (assets_dir / name).is_file()
                and Path(name).suffix.lower() in IMAGE_EXTENSIONS
            ]
            if found:
                return found
            logger.warning(
                f"[Image] {ctx.input_path.name}: markdown references "
                f"{len(ref_names)} asset image(s) but none were found in "
                f"{assets_dir}"
            )

    prefix = asset_base_name(ctx)
    pattern = _own_asset_pattern(prefix)
    try:
        candidates = sorted(assets_dir.iterdir())
    except OSError:
        return []
    return [
        p
        for p in candidates
        if (p.name == prefix or pattern.match(p.name))
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.is_file()
    ]


def _own_asset_pattern(prefix: str) -> re.Pattern[str]:
    """Numbered asset names one output with this *prefix* writes.

    ``<prefix>.0001.<ext>`` (embedded Office/EPUB images),
    ``<prefix>-0001-01.<ext>`` (PDF images) and ``<prefix>-0001.<ext>``
    (PDF images named by position); an image's own copy is found through
    ``asset_path`` or as the bare prefix. A plain prefix glob also matched
    every renamed sibling output (``a.pdf`` -> ``a.pdf.v2-0001-01.jpg``).
    """
    return re.compile(rf"^{re.escape(prefix)}(?:\.\d+|-\d+-\d+|-\d+)\.[A-Za-z0-9]+$")


def apply_alt_text_updates(
    llm_file: Path,
    image_analysis: Any,
) -> bool:
    """Apply alt text updates from image analysis to .llm.md file.

    This is called after document processing completes to update alt text
    in the .llm.md file with results from parallel image analysis.

    Entries that record a failed analysis (see ``is_failed_image_entry``)
    are skipped: their placeholder caption must not replace the author's
    alt text.

    Args:
        llm_file: Path to the .llm.md file
        image_analysis: ImageAnalysisResult with analyzed images

    Returns:
        True if updates were applied, False otherwise
    """
    if not llm_file.exists() or image_analysis is None:
        return False

    try:
        llm_content = llm_file.read_text(encoding="utf-8")

        replacements = {
            Path(asset.get("asset", "")).name: asset["alt"]
            for asset in image_analysis.assets
            if asset.get("alt")
            and Path(asset.get("asset", "")).name
            and not is_failed_image_entry(asset)
        }
        pattern = re.compile(
            r"!\[(?:[^\]\\]|\\.)*\]\((?P<markdown>[^)]+)\)"
            r"|!\[\[(?P<wiki>[^|\]]+)(?:\|[^\]]*)?\]\]"
        )

        def replace_match(match: re.Match[str]) -> str:
            target = match.group("markdown") or match.group("wiki")
            decoded = unquote(target)
            alt = replacements.get(decoded.replace("\\", "/").rsplit("/", 1)[-1])
            if alt is None:
                return match.group(0)
            if match.group("wiki") is not None:
                safe_alt = alt.replace("|", " ").replace("]", "\\]")
                return f"![[{target}|{safe_alt}]]"
            return markdown_image_reference(alt, decoded)

        new_content = pattern.sub(replace_match, llm_content)
        if new_content != llm_content:
            atomic_write_text(llm_file, new_content)
            logger.debug("Applied alt text updates to {}", llm_file)
            return True

    except Exception as e:
        logger.warning(f"Failed to apply alt text updates: {e}")

    return False


def apply_output_profile(ctx: ConversionContext) -> None:
    """Apply the configured output profile to this conversion's written files.

    Runs as the final pipeline step so every earlier stage (conversion,
    image extraction, LLM enhancement, alt-text updates) operates on the
    default layout. No-op when no profile is configured.

    Args:
        ctx: Completed conversion context.
    """
    if ctx.config.output.profile is None or ctx.output_file is None:
        return

    from markitai.output_profiles import apply_profile_to_file

    for candidate in (ctx.output_file, ctx.llm_output_file):
        if candidate is not None:
            apply_profile_to_file(candidate, ctx.output_dir, ctx.config)


def stabilize_written_llm_output(
    ctx: ConversionContext,
    processor: Any,
) -> bool:
    """Re-stabilize a written .llm.md file against its base .md sibling."""
    if ctx.output_file is None or ctx.conversion_result is None:
        return False
    if ctx.paged_stabilized:
        return False

    llm_output = ctx.output_file.with_suffix(".llm.md")
    if not llm_output.exists():
        return False

    from markitai.workflow.helpers import maybe_stabilize_markdown
    from markitai.workflow.single import _read_markdown_body

    # Only a base written by this run is a baseline: under
    # on_conflict=overwrite without --keep-base the file on disk is the
    # previous run's output
    baseline_markdown = (
        _read_markdown_body(ctx.output_file, ctx.conversion_result.markdown)
        if ctx.base_written
        else ctx.conversion_result.markdown
    )
    llm_content = llm_output.read_text(encoding="utf-8")
    frontmatter, llm_body = split_frontmatter(llm_content)
    stabilized = maybe_stabilize_markdown(
        processor, baseline_markdown, llm_body, ctx.input_path.name
    )
    if stabilized == llm_body:
        return False

    rewritten_body = stabilized.rstrip()
    if frontmatter is None:
        rewritten = rewritten_body
    else:
        rewritten = f"---\n{frontmatter}\n---\n\n{rewritten_body}"

    atomic_write_text(llm_output, rewritten)
    logger.debug(f"[{ctx.input_path.name}] Rewrote stabilized LLM output file")
    return True


async def process_with_pure_llm(ctx: ConversionContext) -> ConversionStepResult:
    """Pure mode: send raw markdown to LLM, write response as-is.

    No ContentProtection, stabilization, frontmatter, vision, or image analysis.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(ctx.config, processor=processor)

    try:
        (
            ctx.conversion_result.markdown,
            doc_cost,
            doc_usage,
        ) = await workflow.process_document_pure(
            ctx.conversion_result.markdown,
            ctx.input_path.name,
            ctx.output_file,
        )
        ctx.llm_output_file = ctx.output_file.with_suffix(".llm.md")
        ctx.llm_cost += doc_cost
        merge_llm_usage(ctx.llm_usage, doc_usage)
    except Exception as e:
        return ConversionStepResult(
            success=False,
            error=f"Pure LLM processing failed: {format_error_message(e)}",
        )

    return ConversionStepResult(success=True)


async def process_image_with_vision_pure(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Pure Vision mode: analyze standalone image with Vision model, write raw result.

    For --llm --pure with image-only inputs. Uses analyze_image() to get
    structured ImageAnalysis, then formats as markdown and writes to .llm.md.

    No frontmatter or post-processing is applied — the output is constructed
    directly from the ImageAnalysis structured fields.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    # Get saved image from assets (handles transcoded formats via get_saved_images)
    saved_images = get_saved_images(ctx)
    if not saved_images:
        return ConversionStepResult(
            success=False,
            error=f"No saved image found for {ctx.input_path.name}",
        )

    image_path = saved_images[0]

    # Use shared processor or create new one
    from markitai.workflow.helpers import create_llm_processor

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    # Keyed by the full path, not the basename: same-named images in one
    # batch share the processor, and a shared key would pool their request
    # budgets and usage (same scheme as SingleFileWorkflow.analyze_images)
    context = f"{ctx.input_path.resolve()}:images"
    try:
        analysis = await processor.analyze_image(image_path, context=context)
        ctx.llm_cost += processor.get_context_cost(context)
        merge_llm_usage(ctx.llm_usage, processor.get_context_usage(context))
    except Exception as e:
        return ConversionStepResult(
            success=False,
            error=f"Vision analysis failed: {format_error_message(e)}",
        )
    finally:
        processor.clear_context_usage(context)

    # Format output: # {filename}\n\n{description}\n\n{extracted_text}
    sections = [f"# {ctx.input_path.stem}\n"]

    if analysis.description:
        sections.append(f"{analysis.description.strip()}\n")

    if analysis.extracted_text and analysis.extracted_text.strip():
        sections.append(f"{analysis.extracted_text.strip()}\n")

    content = "\n".join(sections)

    # Write to .llm.md
    llm_output = ctx.output_file.with_suffix(".llm.md")
    atomic_write_text(llm_output, content)
    ctx.llm_output_file = llm_output
    logger.info(f"[Core] Written pure Vision output: {llm_output}")

    return ConversionStepResult(success=True)


async def process_with_vision_llm(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Process document with Vision LLM (screenshot mode).

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    page_images = ctx.conversion_result.metadata.get("page_images", [])
    if not page_images:
        return ConversionStepResult(success=True)

    logger.info(f"[LLM] {ctx.input_path.name}: Starting Screenshot+LLM processing")

    # Use shared processor or create new one
    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )
    document_title = ctx.conversion_result.metadata.get("title")
    if not isinstance(document_title, str):
        document_title = None

    # Check for screenshot-only mode
    use_screenshot_only = ctx.config.screenshot.screenshot_only

    if use_screenshot_only:
        # Screenshot-only mode: extract content purely from screenshots
        (
            cleaned_content,
            frontmatter,
            enhance_cost,
            enhance_usage,
        ) = await workflow.extract_from_screenshots(
            page_images,
            source=ctx.input_path.name,
            original_title=document_title,
        )
    else:
        # Standard mode: use extracted text + screenshots for enhancement
        # Get extracted text (use markdown which has base64 replaced)
        extracted_text = ctx.conversion_result.markdown

        # Enhance with vision
        (
            cleaned_content,
            frontmatter,
            enhance_cost,
            enhance_usage,
        ) = await workflow.enhance_with_vision(
            extracted_text,
            page_images,
            source=ctx.input_path.name,
            original_title=document_title,
        )
    ctx.llm_cost += enhance_cost
    merge_llm_usage(ctx.llm_usage, enhance_usage)

    # Build final content with page image comments
    commented_images_str = ""
    if page_images:
        commented_images = [
            f"<!-- ![Page {img['page']}]({SCREENSHOTS_REL_PATH}/{img['name']}) -->"
            for img in sorted(page_images, key=lambda x: x.get("page", 0))
        ]
        commented_images_str = "\n\n<!-- Page images for reference -->\n" + "\n".join(
            commented_images
        )

    ctx.conversion_result.markdown = cleaned_content + commented_images_str
    ctx.conversion_result.markdown = append_reference_image_comments(
        ctx.conversion_result.markdown,
        ctx.conversion_result.metadata.get("reference_images"),
    )

    # Strip any hallucinated base64 images
    from markitai.image import ImageProcessor

    image_processor = ImageProcessor(config=ctx.config.image)
    ctx.conversion_result.markdown = image_processor.strip_base64_images(
        ctx.conversion_result.markdown
    )

    # Validate image references
    assets_dir = ctx.output_dir / MARKITAI_META_DIR / "assets"
    if assets_dir.exists():
        ctx.conversion_result.markdown = ImageProcessor.remove_nonexistent_images(
            ctx.conversion_result.markdown, assets_dir
        )

    # Write LLM version
    llm_output = ctx.output_file.with_suffix(".llm.md")
    llm_content = processor.format_llm_output(
        ctx.conversion_result.markdown, frontmatter
    )
    atomic_write_text(llm_output, llm_content)
    ctx.llm_output_file = llm_output
    logger.info(f"Written LLM version: {llm_output}")

    return ConversionStepResult(success=True)


async def process_with_standard_llm(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Process document with standard LLM (no screenshots).

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    # Check if standalone image
    is_standalone_image = ctx.input_path.suffix.lower() in IMAGE_EXTENSIONS
    saved_images = get_saved_images(ctx)

    # Use shared processor or create new one
    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )

    if is_standalone_image and saved_images:
        # Standalone image: only run image analysis
        logger.info(f"[LLM] {ctx.input_path.name}: Processing standalone image")
        (
            _,
            image_cost,
            image_usage,
            ctx.image_analysis,
        ) = await workflow.analyze_images(
            saved_images,
            ctx.conversion_result.markdown,
            ctx.output_file,
            ctx.input_path,
        )
        ctx.llm_cost += image_cost
        merge_llm_usage(ctx.llm_usage, image_usage)
        _record_image_warnings(ctx)
    else:
        # Standard LLM processing
        logger.info(f"[LLM] {ctx.input_path.name}: Starting standard LLM processing")

        document_title = ctx.conversion_result.metadata.get("title")
        if not isinstance(document_title, str):
            document_title = None

        # Check if image analysis should run
        should_analyze_images = (
            ctx.config.image.alt_enabled or ctx.config.image.desc_enabled
        ) and saved_images
        reference_images = ctx.conversion_result.metadata.get("reference_images")

        # Run document processing and image analysis in parallel
        # These are independent: doc processing writes .llm.md, image analysis generates descriptions
        if should_analyze_images:
            doc_task = workflow.process_document_with_llm(
                ctx.conversion_result.markdown,
                ctx.input_path.name,
                ctx.output_file,
                reference_images=reference_images,
                title=document_title,
            )
            img_task = workflow.analyze_images(
                saved_images,
                ctx.conversion_result.markdown,
                ctx.output_file,
                ctx.input_path,
            )

            # Execute in parallel. return_exceptions=True so a failure in one
            # task doesn't leave the sibling running detached — an orphaned
            # doc task could still write .llm.md after the fallback path runs
            doc_result, img_result = await asyncio.gather(
                doc_task, img_task, return_exceptions=True
            )

            # Document processing failure is fatal: the caller writes the
            # base-markdown fallback and marks the file failed
            if isinstance(doc_result, BaseException):
                raise doc_result

            # Unpack results
            ctx.conversion_result.markdown, doc_cost, doc_usage = doc_result
            ctx.llm_cost += doc_cost
            merge_llm_usage(ctx.llm_usage, doc_usage)

            # Image-analysis failure degrades gracefully (same policy as
            # analyze_embedded_images): keep .llm.md without alt text
            if isinstance(img_result, BaseException):
                logger.warning(
                    f"Image analysis failed (continuing without alt text): {img_result}"
                )
                ctx.image_analysis = None
                ctx.warnings.append(
                    f"image analysis failed ({format_error_message(img_result)}); "
                    "the original alt text was kept"
                )
            else:
                _, image_cost, image_usage, ctx.image_analysis = img_result
                ctx.llm_cost += image_cost
                merge_llm_usage(ctx.llm_usage, image_usage)
                _record_image_warnings(ctx)

            # Apply alt text updates to .llm.md after document processing completes
            # This ensures no race condition - .llm.md is guaranteed to exist
            if ctx.config.image.alt_enabled and ctx.image_analysis:
                llm_output = ctx.output_file.with_suffix(".llm.md")
                apply_alt_text_updates(llm_output, ctx.image_analysis)
        else:
            # Only document processing
            (
                ctx.conversion_result.markdown,
                doc_cost,
                doc_usage,
            ) = await workflow.process_document_with_llm(
                ctx.conversion_result.markdown,
                ctx.input_path.name,
                ctx.output_file,
                reference_images=reference_images,
                title=document_title,
            )
            ctx.llm_cost += doc_cost
            merge_llm_usage(ctx.llm_usage, doc_usage)

        stabilize_written_llm_output(ctx, processor)
        ctx.paged_stabilized = True

        # Re-apply alt text after stabilization — stabilize may rewrite .llm.md
        # from the baseline .md (which has no alt text), overwriting earlier updates
        if (
            should_analyze_images
            and ctx.config.image.alt_enabled
            and ctx.image_analysis
        ):
            llm_output = ctx.output_file.with_suffix(".llm.md")
            apply_alt_text_updates(llm_output, ctx.image_analysis)

    # Success means the enhanced file is on disk: a branch that wrote
    # nothing is a failed enhancement, not a completed one
    llm_output = ctx.output_file.with_suffix(".llm.md")
    if not llm_output.is_file():
        return ConversionStepResult(
            success=False,
            error=f"LLM enhancement produced no output for {ctx.input_path.name}",
        )
    ctx.llm_output_file = llm_output
    return ConversionStepResult(success=True)


async def analyze_embedded_images(ctx: ConversionContext) -> ConversionStepResult:
    """Analyze embedded images with LLM after Vision processing.

    Used in screenshot+LLM mode to also analyze embedded document images.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=True)

    if not (ctx.config.image.alt_enabled or ctx.config.image.desc_enabled):
        return ConversionStepResult(success=True)

    saved_images = get_saved_images(ctx)
    if not saved_images:
        logger.debug(f"[LLM] {ctx.input_path.name}: no embedded images to analyze")
        return ConversionStepResult(success=True)

    # Filter out page/slide screenshots, only analyze embedded images
    import re

    page_pattern = re.compile(r"\.page\d+\.|\.slide\d+\.", re.IGNORECASE)
    embedded_images = [p for p in saved_images if not page_pattern.search(p.name)]

    if not embedded_images:
        return ConversionStepResult(success=True)

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )

    logger.info(
        f"[LLM] {ctx.input_path.name}: Analyzing {len(embedded_images)} embedded images"
    )

    (
        ctx.conversion_result.markdown,
        image_cost,
        image_usage,
        ctx.image_analysis,
    ) = await workflow.analyze_images(
        embedded_images,
        ctx.conversion_result.markdown,
        ctx.output_file,
        ctx.input_path,
    )
    ctx.llm_cost += image_cost
    merge_llm_usage(ctx.llm_usage, image_usage)
    _record_image_warnings(ctx)

    return ConversionStepResult(success=True)


def _record_image_warnings(ctx: ConversionContext) -> None:
    """Copy the images whose analysis failed into the item's warnings."""
    if ctx.image_analysis is not None:
        ctx.warnings.extend(ctx.image_analysis.warnings)


async def run_llm_enhancement(ctx: ConversionContext) -> ConversionStepResult:
    """Run the LLM step of the pipeline (pure, vision or standard branch).

    Every failure — an exception, a failed step, or an enhancement that
    degraded to unenhanced output (``LLMEnhancementDegradedError``) — writes
    the base ``.md`` as the fallback and returns a failed result, so the
    caller reports the file failed instead of passing base output off as
    enhanced.

    Args:
        ctx: Conversion context (LLM enabled, conversion result set)

    Returns:
        ConversionStepResult indicating success or failure
    """
    assert ctx.conversion_result is not None
    # Ensure shared processor exists for all LLM operations
    # This is critical for:
    # 1. Sharing semaphore (concurrency control)
    # 2. Sharing Router instances (avoid duplicate creation)
    # 3. Sharing cache connections
    if ctx.shared_processor is None:
        from markitai.workflow.helpers import create_llm_processor

        ctx.shared_processor = create_llm_processor(ctx.config)

    if ctx.config.llm.pure and not ctx.config.screenshot.screenshot_only:
        # Pure mode: --screenshot-only takes precedence (mutually exclusive)
        from markitai.converter.base import IMAGE_ONLY_FORMATS

        if ctx.detected_format in IMAGE_ONLY_FORMATS:
            # Image input: use Vision model to analyze the actual image
            result = await process_image_with_vision_pure(ctx)
        else:
            # Non-image: raw MD → LLM text cleaning → .llm.md
            result = await process_with_pure_llm(ctx)
        if not result.success:
            _write_base_md_fallback(ctx)
            return result
    else:
        page_images = ctx.conversion_result.metadata.get("page_images", [])
        has_page_images = len(page_images) > 0

        if has_page_images:
            # Vision mode with screenshots — run sequentially to avoid race
            # condition. process_with_vision_llm writes ctx.conversion_result.markdown,
            # and analyze_embedded_images reads it, so embed must run after vision.
            try:
                vision_result: ConversionStepResult = await process_with_vision_llm(ctx)
            except Exception as e:
                _write_base_md_fallback(ctx)
                return ConversionStepResult(
                    success=False,
                    error=f"Vision LLM failed: {format_error_message(e)}",
                )
            if not vision_result.success:
                _write_base_md_fallback(ctx)
                return vision_result

            # Embedded image analysis (non-critical, log warning on failure)
            try:
                embed_result: ConversionStepResult = await analyze_embedded_images(ctx)
                if not embed_result.success:
                    logger.warning(
                        f"Embedded image analysis failed: {embed_result.error}"
                    )
                    ctx.warnings.append(
                        f"image analysis failed ({embed_result.error}); "
                        "the original alt text was kept"
                    )
            except Exception as e:
                logger.warning(
                    f"Embedded image analysis failed: {format_error_message(e)}"
                )
                ctx.warnings.append(
                    f"image analysis failed ({format_error_message(e)}); "
                    "the original alt text was kept"
                )

            stabilize_written_llm_output(ctx, ctx.shared_processor)
            ctx.paged_stabilized = True

            # Apply alt text updates AFTER stabilization — stabilize may rewrite
            # .llm.md from the baseline .md (which has no alt text), so alt text
            # updates must come last to avoid being overwritten
            if ctx.config.image.alt_enabled and ctx.image_analysis and ctx.output_file:
                llm_output = ctx.output_file.with_suffix(".llm.md")
                apply_alt_text_updates(llm_output, ctx.image_analysis)
        else:
            # Standard LLM mode
            try:
                result = await process_with_standard_llm(ctx)
            except Exception as e:
                _write_base_md_fallback(ctx)
                return ConversionStepResult(
                    success=False,
                    error=f"LLM processing failed: {format_error_message(e)}",
                )
            if not result.success:
                _write_base_md_fallback(ctx)
                return result

    return ConversionStepResult(success=True)


async def convert_document_core(
    ctx: ConversionContext,
    max_document_size: int,
) -> ConversionStepResult:
    """Core document conversion pipeline.

    This function implements the unified conversion logic shared between
    single-file and batch processing modes.

    The pipeline:
    1. Validate file size and detect format
    2. Prepare output directory
    3. Resolve output file path (with conflict handling) — early skip avoids
       expensive conversion when on_conflict=skip and output already exists
    4. Execute document conversion
    5. Process embedded images
    6. Write base markdown file
    7. LLM processing (if enabled):
       - Vision mode (with page screenshots)
       - Standard mode (no screenshots)
       - Embedded image analysis
    8. Output profile post-processing (only when ``output.profile`` is set)

    Args:
        ctx: Conversion context with all inputs and state
        max_document_size: Maximum allowed document size in bytes

    Returns:
        ConversionStepResult indicating overall success or failure
    """
    # Step 1: Validate and detect format
    result = validate_and_detect_format(ctx, max_document_size)
    if not result.success:
        return result

    # Step 1.5: Skip image-only formats when neither LLM nor OCR is enabled
    from markitai.converter.base import IMAGE_ONLY_FORMATS

    if (
        ctx.detected_format is not None
        and ctx.detected_format in IMAGE_ONLY_FORMATS
        and not ctx.config.llm.enabled
        and not ctx.config.ocr.enabled
    ):
        logger.info(
            f"[Core] Skipped {ctx.input_path.name} "
            f"(image file, no text to extract without LLM or OCR)"
        )
        return ConversionStepResult(
            success=True,
            skip_reason="image_only",
        )

    # Step 2: Prepare output directory
    result = prepare_output_directory(ctx)
    if not result.success:
        return result

    # Step 3: Resolve output file (early — skip before expensive conversion)
    result = resolve_output_file(ctx)
    if not result.success or result.skip_reason:
        return result

    # Step 4: Execute conversion
    result = await convert_document(ctx)
    if not result.success:
        return result

    # Step 5: Process embedded images
    result = await process_embedded_images(ctx)
    if not result.success:
        return result

    # Step 6: Write base markdown
    result = write_base_markdown(ctx)
    if not result.success:
        return result

    # Step 7: LLM processing (if enabled)
    if ctx.config.llm.enabled and ctx.conversion_result is not None:
        from markitai.llm.engine import track_cache_hits

        # Tally this file's own cache lookups: the processor is shared
        # across a batch, so its global counters cannot say whether this
        # file was served from cache
        with track_cache_hits() as cache_tally:
            result = await run_llm_enhancement(ctx)
        if not result.success:
            return result
        ctx.cache_hit = cache_tally.served_from_cache(ctx.llm_usage)

    # Step 8: output profile post-processing (no-op without a profile)
    apply_output_profile(ctx)

    return ConversionStepResult(success=True)

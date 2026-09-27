"""Map a finished ``convert_document_core`` run onto a batch ProcessResult.

The CLI batch worker and the serve job runner both drive the same core
pipeline and report per-file outcomes as :class:`markitai.batch.ProcessResult`.
The mapping — skip semantics, which file counts as the output, what the
LLM-enhanced signal is — must not drift between them, so it lives here,
below both callers in the import-linter layering. Logging, timing and any
caller-specific side files (``images.json``) stay with the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markitai.batch import ProcessResult
    from markitai.workflow.core import ConversionContext, ConversionStepResult

#: ``ProcessResult.error`` prefix marking a skip: a success, not a failure.
SKIPPED_PREFIX = "skipped ("


def document_process_result(
    ctx: ConversionContext, result: ConversionStepResult
) -> ProcessResult:
    """Build the ProcessResult for one ``convert_document_core`` run.

    Args:
        ctx: The context the core ran with (its outputs and tallies).
        result: What ``convert_document_core`` returned for it.

    Returns:
        * failure: ``success=False`` with the core's error;
        * ``skip_reason == "exists"``: success, the existing output as
          ``output_path`` and ``error="skipped (exists)"``;
        * ``skip_reason == "image_only"``: success, no output,
          ``error="skipped (image_only)"``;
        * otherwise ``ctx.produced_file`` — the ``.llm.md`` when LLM is
          enabled (``ctx.llm_output_file`` is set only after a successful
          LLM write, the real "enhanced" signal, never the file suffix),
          else the base ``.md`` (also when an LLM failure kept it as the
          fallback) — or a failure when that file is not on disk.
    """
    from markitai.batch import ProcessResult
    from markitai.utils.paths import derive_output_name

    if not result.success:
        return ProcessResult(success=False, error=result.error)

    if result.skip_reason == "exists":
        skipped_output = ctx.output_dir / derive_output_name(ctx.input_path.name)
        return ProcessResult(
            success=True,
            output_path=str(skipped_output),
            error=f"{SKIPPED_PREFIX}exists)",
        )
    if result.skip_reason == "image_only":
        return ProcessResult(success=True, error=f"{SKIPPED_PREFIX}image_only)")

    produced = ctx.produced_file
    if produced is None or not produced.is_file():
        return ProcessResult(
            success=False,
            error=f"No output was produced for {ctx.input_path.name}",
        )
    return ProcessResult(
        success=True,
        output_path=str(produced),
        images=ctx.embedded_images_count,
        screenshots=ctx.screenshots_count,
        cost_usd=ctx.llm_cost,
        llm_usage=ctx.llm_usage,
        image_analysis_result=ctx.image_analysis,
        # Served from cache: the core's per-file tally saw only cache hits
        cache_hit=ctx.cache_hit,
        llm_enhanced=ctx.llm_output_file is not None and not ctx.llm_fell_back,
        warnings=list(ctx.warnings),
    )

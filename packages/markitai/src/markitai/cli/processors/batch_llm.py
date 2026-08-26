"""Batch-API LLM enhancement for directory batches (--llm-batch).

Two-phase flow:

1. The caller runs the normal directory batch with LLM disabled (base .md
   files only), then :func:`run_batch_llm_enhancement` prepares one
   structured document call per base file, serves cache hits immediately,
   submits the rest as a Batch API job, and waits (with progress) up to
   the timeout.
2. On timeout the run state is persisted and the process exits with a
   recovery hint; :func:`collect_batch_llm` finishes the job later.

Only OpenAI-compatible pools are supported (litellm's batch helpers); other
pools are refused with an actionable message rather than silently falling
back to real-time pricing. Documents whose batch request fails are re-run
live one by one, so a partial batch never loses output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.llm.batch_api import (
    BatchDocItem,
    BatchRunState,
    build_openai_batch_request,
    download_openai_batch_output,
    parse_batch_result,
    poll_openai_batch,
    read_openai_batch_output,
    submit_openai_batch,
    write_batch_jsonl,
)
from markitai.llm.structured import instructor_mode_for_model
from markitai.utils.errors import ConversionError

BATCH_COST_FACTOR = 0.5  # OpenAI Batch API list-price discount


def _single_openai_model(cfg: Any) -> tuple[str, str]:
    """Resolve the pool to exactly one OpenAI-family model.

    Batches are per-provider; a mixed or multi-model pool is refused with
    guidance instead of guessing.

    Returns:
        (model_id, litellm custom_llm_provider)

    Raises:
        ConversionError: Pool is empty, multi-model, or not OpenAI-family.
    """
    models = [m.litellm_params.model for m in (cfg.llm.model_list or [])]
    if not models:
        raise ConversionError(
            "--llm-batch needs a configured model (llm.model_list or MODEL env)"
        )
    unique = sorted(set(models))
    if len(unique) > 1:
        raise ConversionError(
            f"--llm-batch currently needs a single-model pool; got {', '.join(unique)}"
        )
    model = unique[0]
    if model.startswith("openai/"):
        return model.removeprefix("openai/"), "openai"
    raise ConversionError(
        f"--llm-batch currently supports OpenAI pools only (got {model!r}). "
        "Run without --llm-batch for real-time processing on this pool."
    )


def _prepare_pending(
    processor: Any,
    output_dir: Path,
) -> tuple[list[tuple[BatchDocItem, Any]], int]:
    """Build a plan per base .md, serving cache hits immediately.

    Returns:
        (uncached (item, plan) pairs, number of cache-served documents)
    """
    base_files = sorted(
        p
        for p in output_dir.rglob("*.md")
        if not p.name.endswith(".llm.md") and ".markitai" not in p.parts
    )
    pending: list[tuple[BatchDocItem, Any]] = []
    cached = 0
    for base_md in base_files:
        # Live runs name the LLM context after the input file (note1.md),
        # not the written base (note1.md.md) — keep the naming identical.
        # The base's frontmatter is stripped so the LLM never sees it.
        source = base_md.name.removesuffix(".md")
        markdown = _strip_frontmatter(base_md.read_text(encoding="utf-8"))
        plan = processor.documents._prepare_document_plan(markdown, source)
        hit = processor._engine.try_cached(plan.call)
        if hit is not None:
            cleaned, frontmatter = processor.documents.finalize_document_plan(plan, hit)
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            cached += 1
            continue
        item = BatchDocItem(
            custom_id=f"doc::{len(pending)}::{source}",
            source=source,
            input_md=f"inputs/{len(pending)}.md",
            base_md=str(base_md.relative_to(output_dir)),
        )
        pending.append((item, plan))
    return pending, cached


def _strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block, returning the body."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def _write_llm_md(
    processor: Any, base_md: Path, frontmatter: str, cleaned: str
) -> Path:
    """Write the enhanced .llm.md next to its base file.

    Assembled through ``format_llm_output`` so the frontmatter fences and
    body spacing are byte-identical to the live path.
    """
    from markitai.security import atomic_write_text

    target = base_md.with_suffix(".llm.md")
    atomic_write_text(target, processor.format_llm_output(cleaned, frontmatter))
    return target


async def run_batch_llm_enhancement(
    cfg: Any,
    output_dir: Path,
    *,
    timeout_s: float = 3600.0,
    quiet: bool = False,
) -> int:
    """Submit and (mostly) wait for a Batch API enhancement run.

    Returns:
        0 when everything finished (or there was nothing to do); 2 when the
        batch is still in flight past the timeout — the run state is on disk
        and ``--llm-batch-collect`` finishes it later.

    Raises:
        ConversionError: Pool/config unsupported, or the batch ended in a
            non-completed terminal state.
    """
    from markitai.workflow.helpers import create_llm_processor

    model, provider = _single_openai_model(cfg)
    mode = instructor_mode_for_model(f"openai/{model}")
    processor = create_llm_processor(cfg)

    pending, cached = _prepare_pending(processor, output_dir)
    if cached:
        logger.info(f"[Batch] {cached} document(s) served from cache")
    if not pending:
        if not quiet:
            print("All documents already cached — nothing to submit.")
        return 0

    state_dir = BatchRunState.state_dir_for(output_dir, "pending")
    inputs_dir = state_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for item, plan in pending:
        (state_dir / item.input_md).write_text(plan.original_markdown, encoding="utf-8")
        requests.append(
            build_openai_batch_request(
                item.custom_id,
                messages=plan.call.messages,
                response_model=plan.call.response_model,
                model=model,
                mode=mode,
            )
        )
    jsonl_path = state_dir / "requests.jsonl"
    write_batch_jsonl(requests, jsonl_path)

    if not quiet:
        print(
            f"Submitting {len(pending)} document(s) to the Batch API "
            f"({model}, 50% of list price)..."
        )
    batch_id = await submit_openai_batch(jsonl_path, custom_llm_provider=provider)

    # Persist under the real batch id (move the pending dir into place)
    final_state_dir = BatchRunState.state_dir_for(output_dir, batch_id)
    state_dir.rename(final_state_dir)
    state = BatchRunState(
        batch_id=batch_id,
        model=model,
        mode=mode.value,
        provider=provider,
        created_at=datetime.now().astimezone().isoformat(),
        items=[item for item, _ in pending],
    )
    state.save(final_state_dir)

    def _progress(status: str, done: int, total: int) -> None:
        if not quiet:
            print(f"\rBatch {batch_id}: {status} ({done}/{total})", end="", flush=True)

    try:
        status = await poll_openai_batch(
            batch_id,
            custom_llm_provider=provider,
            timeout_s=timeout_s,
            on_progress=_progress,
        )
    except TimeoutError as e:
        if not quiet:
            print()
            print(f"Batch still in flight: {e}")
            print("It keeps running server-side. Collect it later with:")
            print(f"  markitai --llm-batch-collect {batch_id} -o {output_dir}")
        return 2

    if not quiet:
        print()
    if status != "completed":
        raise ConversionError(f"batch {batch_id} ended with status={status!r}")

    return await _finish_batch(cfg, processor, output_dir, state, quiet=quiet)


async def _finish_batch(
    cfg: Any,
    processor: Any,
    output_dir: Path,
    state: BatchRunState,
    *,
    quiet: bool,
) -> int:
    """Download results and finalize each document (live re-run on failure)."""
    import instructor
    import litellm

    from markitai.llm.models import get_response_cost

    state_dir = BatchRunState.state_dir_for(output_dir, state.batch_id)
    out_path = state_dir / "output.jsonl"
    await download_openai_batch_output(
        state.batch_id, out_path, custom_llm_provider=state.provider
    )

    mode = instructor.Mode(state.mode)
    lines = {line.custom_id: line for line in read_openai_batch_output(out_path)}
    done = 0
    reran = 0
    for item in state.items:
        plan = processor.documents._prepare_document_plan(
            (state_dir / item.input_md).read_text(encoding="utf-8"),
            item.source,
        )
        base_md = output_dir / item.base_md
        line = lines.get(item.custom_id)
        try:
            if line is None:
                raise ConversionError(f"no output line for {item.custom_id}")
            if line.error is not None:
                raise ConversionError(line.error)
            assert line.body is not None
            result = parse_batch_result(
                line.body, response_model=plan.call.response_model, mode=mode
            )
            if plan.call.validate is not None:
                result = plan.call.validate(result)
            cleaned, frontmatter = processor.documents.finalize_document_plan(
                plan, result
            )
            processor._engine.write_cache(plan.call, result)
            # Account batch usage at the discounted rate.
            usage = line.body.get("usage") or {}
            processor._track_usage(
                state.model,
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                get_response_cost(litellm.ModelResponse(**line.body))
                * BATCH_COST_FACTOR,
                item.source,
            )
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            done += 1
        except Exception as e:
            logger.warning(
                f"[Batch] {item.source} failed in batch ({e}); re-running live"
            )
            markdown = (state_dir / item.input_md).read_text(encoding="utf-8")
            cleaned, frontmatter = await processor.documents.process_document(
                markdown, item.source
            )
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            reran += 1

    if not quiet:
        print(
            f"Batch enhancement done: {done} via batch"
            + (f", {reran} re-ran live" if reran else "")
            + (
                f" (billed at {int(BATCH_COST_FACTOR * 100)}% of list price)"
                if done
                else ""
            )
        )
    return 0


async def collect_batch_llm(
    cfg: Any,
    output_dir: Path,
    batch_id: str,
    *,
    quiet: bool = False,
) -> int:
    """Collect a previously submitted batch (two-phase recovery).

    Returns:
        0 when finished; 2 when the batch is still in flight.

    Raises:
        ConversionError: No run state found, or the batch failed.
    """
    state_dir = BatchRunState.state_dir_for(output_dir, batch_id)
    if not (state_dir / "state.json").exists():
        raise ConversionError(
            f"No pending batch state at {state_dir} — was this output_dir "
            "the one used for the original --llm-batch run?"
        )
    state = BatchRunState.load(state_dir)

    try:
        status = await poll_openai_batch(
            batch_id,
            custom_llm_provider=state.provider,
            timeout_s=5,  # collect is a status check, not a wait
            interval_s=2,
        )
    except TimeoutError:
        status = "in_progress"
    if status != "completed":
        if status in ("failed", "expired", "cancelled"):
            raise ConversionError(f"batch {batch_id} ended with status={status!r}")
        if not quiet:
            print(f"Batch {batch_id} is still {status!r} — try again later.")
        return 2

    from markitai.workflow.helpers import create_llm_processor

    processor = create_llm_processor(cfg)
    return await _finish_batch(cfg, processor, output_dir, state, quiet=quiet)

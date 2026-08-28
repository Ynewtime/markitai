"""Batch-API LLM enhancement for directory batches (--llm-batch).

Two-phase flow:

1. The caller runs the normal directory batch with LLM disabled (base .md
   files only), then :func:`run_batch_llm_enhancement` prepares one
   structured document call per base file, serves cache hits immediately,
   submits the rest as a Batch API job, and waits (with progress) up to
   the timeout.
2. On timeout the run state is persisted and the process exits with a
   recovery hint; :func:`collect_batch_llm` finishes the job later.

Only pools on a provider with a batch API (OpenAI-compatible or Anthropic)
are supported; others are refused with an actionable message rather than
silently falling back to real-time pricing. Documents whose batch request
fails are re-run live one by one, so a partial batch never loses output.

Image analysis rides the same job when --alt/--desc are on: alt text is
applied to the written .llm.md rather than fed to the document call, so
the two kinds of request are independent and need no ordering between
them.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.constants import DEFAULT_MAX_PAGES_PER_BATCH
from markitai.llm.batch_api import (
    BatchDocItem,
    BatchRunState,
    build_anthropic_batch_request,
    build_openai_batch_request,
    download_anthropic_batch_output,
    download_openai_batch_output,
    parse_batch_result,
    poll_anthropic_batch,
    poll_openai_batch,
    read_openai_batch_output,
    submit_anthropic_batch,
    submit_openai_batch,
    write_batch_jsonl,
)
from markitai.llm.structured import instructor_mode_for_model
from markitai.llm.types import ImageAnalysis, ImageAnalysisResult
from markitai.utils.errors import ConversionError

BATCH_COST_FACTOR = 0.5  # both providers bill batches at half of list price


_BATCH_PROVIDERS = ("openai", "anthropic")


def _single_batch_model(cfg: Any) -> tuple[str, str]:
    """Resolve the pool to exactly one model on a provider with a batch API.

    Batches are per-provider; a mixed or multi-model pool is refused with
    guidance instead of guessing.

    Returns:
        (model_id without its provider prefix, provider name)

    Raises:
        ConversionError: Pool is empty, multi-model, or on a provider whose
            batch API markitai does not speak.
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
    for provider in _BATCH_PROVIDERS:
        if model.startswith(f"{provider}/"):
            return model.removeprefix(f"{provider}/"), provider
    raise ConversionError(
        f"--llm-batch supports {' and '.join(_BATCH_PROVIDERS)} pools "
        f"(got {model!r}). Run without --llm-batch for real-time processing "
        "on this pool."
    )


def _batch_custom_id(index: int, source: str, kind: str = "doc") -> str:
    """An id both batch APIs accept, still readable in a log.

    Anthropic validates ``custom_id`` against ``^[a-zA-Z0-9_-]{1,64}$``,
    which the obvious ``doc::0::note1.md`` fails on both the colons and the
    dot. The index leads so the id stays unique after the name is squashed
    and truncated.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", source)
    return f"{kind}_{index}_{safe}"[:64]


def _anthropic_max_tokens(model: str) -> int:
    """The output cap Anthropic requires on every batched request.

    The live path lets litellm supply a default; the Messages API has none,
    so read the model's own ceiling and fall back to a value large enough
    for a cleaned document if litellm has never heard of the model.
    """
    from markitai.llm.models import get_model_max_output_tokens

    try:
        return int(get_model_max_output_tokens(f"anthropic/{model}"))
    except Exception:
        logger.debug("[Batch] no max_output_tokens for {}; using 8192", model)
        return 8192


def _document_images(base_md: Path, markdown: str, output_dir: Path) -> list[Path]:
    """The asset files one document's markdown refers to, in document order.

    Read from the refs the converter itself wrote, not from a glob on the
    source name: extractors sanitize filenames, so a prefix glob misses
    them (see ``extract_asset_image_names``).
    """
    from markitai.constants import ASSETS_REL_PATH
    from markitai.utils.text import extract_asset_image_names

    assets_dir = base_md.parent / ASSETS_REL_PATH
    found = [assets_dir / name for name in extract_asset_image_names(markdown)]
    return [path for path in found if path.is_file()]


def _finalize_doc(
    processor: Any, plan: Any, result: Any, vision: bool
) -> tuple[str, str]:
    """Finish a document plan with the half that built it."""
    if vision:
        return processor.documents.finalize_vision_plan(plan, result)
    return processor.documents.finalize_document_plan(plan, result)


def _document_pages(base_md: Path, output_dir: Path) -> list[Path]:
    """The page screenshots rendered for one document, in page order.

    Unlike assets, these are named by markitai itself —
    ``f"{input_path.name}.page{n:04d}.{ext}"`` — from the same string that
    becomes the base ``.md`` name, so the mapping is exact rather than a
    guess at what an extractor did to the filename.
    """
    from markitai.constants import SCREENSHOTS_REL_PATH

    source = base_md.name.removesuffix(".md")
    shots = base_md.parent / SCREENSHOTS_REL_PATH
    if not shots.is_dir():
        return []
    return sorted(shots.glob(f"{source}.page[0-9][0-9][0-9][0-9].*"))


def _prepare_pending(
    processor: Any,
    output_dir: Path,
    *,
    analyze_images: bool = False,
    analyze_pages: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES_PER_BATCH,
) -> tuple[list[tuple[BatchDocItem, Any]], int, list[tuple[Path, str, list[Path]]]]:
    """Build the batch's requests, serving anything already cached.

    One request per base .md for the text enhancement, plus — when image
    analysis is on — one per image that document refers to. Both kinds ride
    the same job: they are independent of each other, because alt text is
    applied to the written ``.llm.md`` rather than fed to the document call.

    Returns:
        (uncached (item, plan) pairs, number of cache-served requests)
    """
    base_files = sorted(
        p
        for p in output_dir.rglob("*.md")
        if not p.name.endswith(".llm.md") and ".markitai" not in p.parts
    )
    pending: list[tuple[BatchDocItem, Any]] = []
    oversized: list[tuple[Path, str, list[Path]]] = []
    cached = 0
    for base_md in base_files:
        # Live runs name the LLM context after the input file (note1.md),
        # not the written base (note1.md.md) — keep the naming identical.
        # The base's frontmatter is stripped so the LLM never sees it.
        source = base_md.name.removesuffix(".md")
        markdown = _strip_frontmatter(base_md.read_text(encoding="utf-8"))
        relative_base = str(base_md.relative_to(output_dir))

        pages = _document_pages(base_md, output_dir) if analyze_pages else []
        if len(pages) > max_pages:
            # The live path splits a long document into ordered rounds, and
            # each batch round is a separate 24-hour wait. Sending every page
            # in one request instead would be one enormous upload. This
            # document is enhanced live below; the rest still get the
            # discount.
            oversized.append((base_md, source, pages))
            continue
        if pages:
            plan = processor.documents.prepare_vision_plan(markdown, pages, source)
        else:
            plan = processor.documents._prepare_document_plan(markdown, source)
        hit = processor._engine.try_cached(plan.call)
        if hit is not None:
            cleaned, frontmatter = _finalize_doc(processor, plan, hit, bool(pages))
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            cached += 1
        else:
            pending.append(
                (
                    BatchDocItem(
                        custom_id=_batch_custom_id(len(pending), source),
                        source=source,
                        input_md=f"inputs/{len(pending)}.md",
                        base_md=relative_base,
                        kind="vision" if pages else "doc",
                    ),
                    plan,
                )
            )

        if not analyze_images:
            continue
        for image in _document_images(base_md, markdown, output_dir):
            image_plan = processor.vision.prepare_image_plan(
                image, context=source, document_context=markdown[:500]
            )
            # An unsupported format or a cache hit is already the answer;
            # it is collected from the plan rather than asked for again.
            if image_plan.answer is not None:
                cached += 1
                continue
            pending.append(
                (
                    BatchDocItem(
                        custom_id=_batch_custom_id(len(pending), image.stem, "img"),
                        source=source,
                        input_md="",
                        base_md=relative_base,
                        kind="image",
                        image=str(image.relative_to(output_dir)),
                    ),
                    image_plan,
                )
            )
    return pending, cached, oversized


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

    model, provider = _single_batch_model(cfg)
    mode = instructor_mode_for_model(f"{provider}/{model}")
    processor = create_llm_processor(cfg)

    analyze_images = bool(cfg.image.alt_enabled or cfg.image.desc_enabled)
    pending, cached, oversized = _prepare_pending(
        processor,
        output_dir,
        analyze_images=analyze_images,
        analyze_pages=bool(cfg.screenshot.enabled),
    )
    for base_md, source, pages in oversized:
        logger.info(
            f"[Batch] {source}: {len(pages)} pages exceed the {DEFAULT_MAX_PAGES_PER_BATCH}-page "
            "single-call limit; enhancing live at full price — "
            "batching it would mean one round trip per round, each up to 24h"
        )
        markdown = _strip_frontmatter(base_md.read_text(encoding="utf-8"))
        cleaned, frontmatter = await processor.documents.enhance_document_complete(
            markdown, pages, source=source
        )
        _write_llm_md(processor, base_md, frontmatter, cleaned)
    if cached:
        logger.info(f"[Batch] {cached} request(s) served from cache")
    if not pending:
        if not quiet:
            print("All documents already cached — nothing to submit.")
        return 0

    state_dir = BatchRunState.state_dir_for(output_dir, "pending")
    inputs_dir = state_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for item, plan in pending:
        if item.kind == "image":
            messages, response_model = plan.messages, ImageAnalysisResult
        else:
            messages, response_model = plan.call.messages, plan.call.response_model
            # The document's input is saved so the collector can rebuild the
            # identical plan; an image's input is the file itself, still on
            # disk under output_dir.
            (state_dir / item.input_md).write_text(
                plan.original_markdown, encoding="utf-8"
            )
        if provider == "anthropic":
            requests.append(
                build_anthropic_batch_request(
                    item.custom_id,
                    messages=messages,
                    response_model=response_model,
                    model=model,
                    max_tokens=_anthropic_max_tokens(model),
                )
            )
        else:
            requests.append(
                build_openai_batch_request(
                    item.custom_id,
                    messages=messages,
                    response_model=response_model,
                    model=model,
                    mode=mode,
                )
            )
    # Written either way: it is the record of what was submitted, and the
    # OpenAI path uploads this exact file.
    jsonl_path = state_dir / "requests.jsonl"
    write_batch_jsonl(requests, jsonl_path)

    if not quiet:
        print(
            f"Submitting {len(pending)} request(s) to the Batch API "
            f"({model}, 50% of list price)..."
        )
    if provider == "anthropic":
        batch_id = await submit_anthropic_batch(requests)
    else:
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
        if provider == "anthropic":
            status = await poll_anthropic_batch(
                batch_id, timeout_s=timeout_s, on_progress=_progress
            )
        else:
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


def _batch_usage(
    body: dict[str, Any], model: str, provider: str
) -> tuple[int, int, float]:
    """Token counts and list-price cost for one batch result.

    The two providers report usage under different names, and only the
    OpenAI-shaped body can be handed to litellm's cost calculator; an
    Anthropic result is priced from the same table the rest of markitai
    estimates with. Caller applies the batch discount.
    """
    usage = body.get("usage") or {}
    if provider == "anthropic":
        from markitai.providers import estimate_model_cost

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cost = estimate_model_cost(model, input_tokens, output_tokens).cost_usd
        return input_tokens, output_tokens, cost

    import litellm

    from markitai.llm.models import get_response_cost

    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        get_response_cost(litellm.ModelResponse(**body)),
    )


def _account_batch_usage(
    processor: Any, state: BatchRunState, line: Any, source: str
) -> None:
    """Record one batched answer's usage at the discounted rate."""
    input_tokens, output_tokens, cost = _batch_usage(
        line.body, state.model, state.provider
    )
    processor._track_usage(
        state.model,
        input_tokens,
        output_tokens,
        cost * BATCH_COST_FACTOR,
        source,
    )


async def _collect_image(
    processor: Any,
    state: BatchRunState,
    output_dir: Path,
    item: BatchDocItem,
    line: Any,
    mode: Any,
) -> dict[str, Any]:
    """Turn one batched image answer into an images.json / alt-text entry.

    A failed image degrades rather than raising: the live path treats image
    analysis as non-critical (the document keeps its alt-less markdown), and
    a batch must not be stricter than the path it replaces.

    The offline path cannot run the live path's language retries — each
    needs to see an answer before deciding whether to ask again — so a
    drifting answer is rewritten here, live, at collect time. That is one
    cheap text call, not another 24-hour round trip.
    """
    from markitai.llm.vision import _should_retry_for_language

    image = output_dir / item.image
    plan = processor.vision.prepare_image_plan(image, context=item.source)
    if plan.answer is not None:
        analysis = plan.answer
    else:
        try:
            if line is None:
                raise ConversionError(f"no output line for {item.custom_id}")
            if line.error is not None:
                raise ConversionError(line.error)
            assert line.body is not None
            result = parse_batch_result(
                line.body,
                response_model=ImageAnalysisResult,
                mode=mode,
                provider=state.provider,
            )
            _account_batch_usage(processor, state, line, item.source)
            if _should_retry_for_language(result, plan.language):
                result = await processor.vision._rewrite_analysis_language(
                    result,
                    language=plan.language,
                    context=item.source,
                    document_context=plan.document_context,
                )
            analysis = processor.vision.finalize_image_plan(plan, result)
        except Exception as e:
            logger.warning(f"[Batch] image {image.name} failed in batch ({e})")
            analysis = ImageAnalysis(
                caption="Image", description="Image analysis failed"
            )

    return {
        "asset": str(image.resolve()),
        "alt": analysis.caption or "Image",
        "desc": analysis.description,
        "text": analysis.extracted_text or "",
        "llm_usage": analysis.llm_usage or {},
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _apply_image_answers(
    cfg: Any, output_dir: Path, images_by_base: dict[str, list[dict[str, Any]]]
) -> None:
    """Write the collected image answers into each document's output.

    Mirrors the live path's split: alt text is substituted into the written
    ``.llm.md``, descriptions go to images.json.
    """
    if not images_by_base:
        return
    from markitai.output_profiles import assets_visible
    from markitai.workflow.core import apply_alt_text_updates
    from markitai.workflow.helpers import write_images_json
    from markitai.workflow.single import ImageAnalysisResult as AnalysisForSource

    results = []
    for relative_base, assets in sorted(images_by_base.items()):
        base_md = output_dir / relative_base
        result = AnalysisForSource(source_file=str(base_md.resolve()), assets=assets)
        results.append(result)
        if cfg.image.alt_enabled:
            apply_alt_text_updates(base_md.with_suffix(".llm.md"), result)
    if cfg.image.desc_enabled:
        write_images_json(output_dir, results, visible_assets=assets_visible(cfg))


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

    state_dir = BatchRunState.state_dir_for(output_dir, state.batch_id)
    out_path = state_dir / "output.jsonl"
    if state.provider == "anthropic":
        await download_anthropic_batch_output(state.batch_id, out_path)
    else:
        await download_openai_batch_output(
            state.batch_id, out_path, custom_llm_provider=state.provider
        )

    mode = instructor.Mode(state.mode)
    lines = {line.custom_id: line for line in read_openai_batch_output(out_path)}
    done = 0
    reran = 0
    # Image answers, gathered per owning document: alt text goes into the
    # .llm.md the document's own result writes, so it can only be applied
    # once every request for that document has been read.
    images_by_base: dict[str, list[dict[str, Any]]] = {}

    for item in state.items:
        base_md = output_dir / item.base_md
        line = lines.get(item.custom_id)

        if item.kind == "image":
            analysis = await _collect_image(
                processor, state, output_dir, item, line, mode
            )
            images_by_base.setdefault(item.base_md, []).append(analysis)
            done += 1
            continue

        markdown = (state_dir / item.input_md).read_text(encoding="utf-8")
        vision = item.kind == "vision"
        if vision:
            # The screenshots are still where the conversion left them, and
            # their names derive from the same source string, so the plan
            # rebuilds identically hours later.
            plan = processor.documents.prepare_vision_plan(
                markdown, _document_pages(base_md, output_dir), item.source
            )
        else:
            plan = processor.documents._prepare_document_plan(markdown, item.source)
        try:
            if line is None:
                raise ConversionError(f"no output line for {item.custom_id}")
            if line.error is not None:
                raise ConversionError(line.error)
            assert line.body is not None
            result = parse_batch_result(
                line.body,
                response_model=plan.call.response_model,
                mode=mode,
                provider=state.provider,
            )
            if plan.call.validate is not None:
                result = plan.call.validate(result)
            cleaned, frontmatter = _finalize_doc(processor, plan, result, vision)
            processor._engine.write_cache(plan.call, result)
            _account_batch_usage(processor, state, line, item.source)
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            done += 1
        except Exception as e:
            logger.warning(
                f"[Batch] {item.source} failed in batch ({e}); re-running live"
            )
            if vision:
                (
                    cleaned,
                    frontmatter,
                ) = await processor.documents.enhance_document_complete(
                    markdown,
                    _document_pages(base_md, output_dir),
                    source=item.source,
                )
            else:
                cleaned, frontmatter = await processor.documents.process_document(
                    markdown, item.source
                )
            _write_llm_md(processor, base_md, frontmatter, cleaned)
            reran += 1

    _apply_image_answers(cfg, output_dir, images_by_base)

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
        # A status check, not a wait — the caller is asking "is it done yet?"
        if state.provider == "anthropic":
            status = await poll_anthropic_batch(batch_id, timeout_s=5, interval_s=2)
        else:
            status = await poll_openai_batch(
                batch_id,
                custom_llm_provider=state.provider,
                timeout_s=5,
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

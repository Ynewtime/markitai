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

Network and API failures end in one actionable line (a ConversionError the
CLI prints), never a traceback: before submission nothing was sent and the
staging directory is removed; after it, the run state stays on disk and
the message names the ``--llm-batch-collect`` command that resumes it.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess  # nosec B404 - list2cmdline only quotes, runs nothing
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.cli.ui import get_stderr_console
from markitai.constants import DEFAULT_MAX_PAGES_PER_BATCH
from markitai.llm.batch_api import (
    BatchCredentials,
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
from markitai.runs.types import Outcome
from markitai.types import LLMUsageByModel, ModelUsageStats
from markitai.utils.errors import ConversionError
from markitai.utils.frontmatter import split_frontmatter
from markitai.utils.text import format_error_message

BATCH_COST_FACTOR = 0.5  # both providers bill batches at half of list price


_BATCH_PROVIDERS = ("openai", "anthropic")

# A failed batch leaves this run's base .md files behind; a plain re-run
# would write .v2 copies next to them.
_RERUN_HINT = (
    'add --config-json \'{"output": {"on_conflict": "overwrite"}}\' '
    "to reuse these outputs instead of writing .v2 copies"
)


def shell_arg(value: str, *, windows: bool | None = None) -> str:
    """Quote *value* for a command the user copies into their shell.

    POSIX shells get ``shlex.quote``; on Windows its single quotes are
    literal to cmd.exe, so arguments are quoted the way Windows parses
    them (double quotes, only when needed).
    """
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


@dataclass(frozen=True)
class BatchHandoff:
    """A submitted batch the run stopped waiting for.

    Returned instead of an exit code so the caller can both tell the user
    how to finish it and put the same facts in ``--json``: the batch keeps
    running server-side, and ``collect_command`` picks it up later.
    """

    batch_id: str
    status: str
    output_dir: Path
    reason: str
    # The run's -c file: collection resolves the pool's credentials from it.
    config_path: Path | None = None

    @property
    def collect_command(self) -> str:
        command = (
            f"markitai --llm-batch-collect {shell_arg(self.batch_id)} "
            f"-o {shell_arg(str(self.output_dir))}"
        )
        if self.config_path is not None:
            command += f" -c {shell_arg(str(self.config_path))}"
        return command

    def to_json(self) -> dict[str, str]:
        return {
            "id": self.batch_id,
            "status": self.status,
            "collect_command": self.collect_command,
        }

    def report(self) -> None:
        """Say why the run stopped and how to finish it.

        Printed even under --quiet/--json: it is the only place the batch id
        appears, and without it the paid-for results cannot be collected.
        """
        console = get_stderr_console()
        console.print(self.reason, markup=False, highlight=False)
        console.print(
            "It keeps running server-side. Collect it later with:",
            markup=False,
            highlight=False,
        )
        console.print(f"  {self.collect_command}", markup=False, highlight=False)


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


def _batch_credentials(cfg: Any, model: str, provider: str) -> BatchCredentials:
    """The pool entry's own key and endpoint, resolved like the live router.

    The live path hands each ``model_list`` entry's ``api_key``/``api_base``
    to litellm (``env:VAR`` resolved by the same ``get_resolved_*``
    helpers); a batch that only read ``OPENAI_*`` from the environment
    would fail on a configured key, or bill whatever account the
    environment happens to hold. No matching entry (a collect run with a
    different config) falls back to the environment.

    Raises:
        ConversionError: The entry references an ``env:`` variable that is
            not set.
    """
    from markitai.config import EnvVarNotFoundError

    target = f"{provider}/{model}"
    for entry in cfg.llm.model_list or []:
        params = entry.litellm_params
        if params.model != target:
            continue
        try:
            return BatchCredentials(
                api_key=params.get_resolved_api_key(),
                api_base=params.get_resolved_api_base(),
            )
        except EnvVarNotFoundError as e:
            raise ConversionError(
                f"--llm-batch: {target} reads its credentials from "
                f"${e.var_name}, which is not set."
            ) from None
    logger.debug(
        "[Batch] no model_list entry for {}; using the provider environment",
        target,
    )
    return BatchCredentials()


def validate_batch_pool(cfg: Any) -> None:
    """Check, before any conversion, that --llm-batch can run on this pool.

    Raises:
        ConversionError: The pool is not a single OpenAI/Anthropic model, or
            its credentials reference an unset environment variable.
    """
    model, provider = _single_batch_model(cfg)
    _batch_credentials(cfg, model, provider)


def _api_failure(action: str, error: Exception, next_step: str) -> ConversionError:
    """One actionable line for a Batch API call that failed."""
    reason = format_error_message(error).rstrip(".")
    return ConversionError(f"Batch API {action} failed: {reason}. {next_step}")


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


def _document_images(base_md: Path, markdown: str) -> list[Path]:
    """The asset files one document's markdown refers to, in document order.

    Read from the refs the converter itself wrote, not from a glob on the
    source name: extractors sanitize filenames, so a prefix glob misses
    them (see ``extract_asset_image_names``).
    """
    from markitai.constants import ASSETS_REL_PATH, VISIBLE_ASSETS_REL_PATH
    from markitai.output_profiles import visible_asset_names
    from markitai.utils.text import extract_asset_image_names

    found = [
        base_md.parent / ASSETS_REL_PATH / name
        for name in extract_asset_image_names(markdown)
    ]
    found.extend(
        base_md.parent / VISIBLE_ASSETS_REL_PATH / name
        for name in visible_asset_names(markdown)
    )
    return [path for path in found if path.is_file()]


def _finalize_doc(
    processor: Any, plan: Any, result: Any, vision: bool
) -> tuple[str, str]:
    """Finish a document plan with the half that built it."""
    if vision:
        return processor.documents.finalize_vision_plan(plan, result)
    return processor.documents.finalize_document_plan(plan, result)


def _document_pages(base_md: Path) -> list[Path]:
    """The page (PDF) or slide (PPTX) screenshots of one document, in order.

    Unlike assets, these are named by markitai itself —
    ``f"{asset_prefix}.page{n:04d}.{ext}"`` (``.slide{n:04d}`` for slides),
    where the prefix is the resolved
    output name without ``.md`` (``report.pdf.v2.md`` -> ``report.pdf.v2``;
    see ``workflow.core.asset_base_name``) — so the mapping is exact rather
    than a guess at what an extractor did to the filename.
    """
    from markitai.constants import SCREENSHOTS_REL_PATH
    from markitai.security import escape_glob_pattern
    from markitai.utils.output import split_markdown_name

    source = escape_glob_pattern(split_markdown_name(base_md.name)[0])
    shots = base_md.parent / SCREENSHOTS_REL_PATH
    if not shots.is_dir():
        return []
    for kind in ("page", "slide"):
        found = sorted(shots.glob(f"{source}.{kind}[0-9][0-9][0-9][0-9].*"))
        if found:
            return found
    return []


def _prepare_pending(
    processor: Any,
    output_dir: Path,
    *,
    analyze_images: bool = False,
    analyze_pages: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES_PER_BATCH,
    base_files: list[Path] | None = None,
    written: set[Path] | None = None,
    cfg: Any = None,
    cached_images: list[BatchDocItem] | None = None,
) -> tuple[list[tuple[BatchDocItem, Any]], int, list[tuple[Path, str, list[Path]]]]:
    """Build the batch's requests, serving anything already cached.

    One request per base .md for the text enhancement, plus — when image
    analysis is on — one per image that document refers to. Both kinds ride
    the same job: they are independent of each other, because alt text is
    applied to the written ``.llm.md`` rather than fed to the document call.
    That also holds for a document too long for one request: it is enhanced
    live, but its images still ride the batch.

    An image the cache (or its unsupported format) already answers is not
    asked about again; its answer goes to ``cached_images``, still to be
    applied to the document's ``.llm.md`` like a batched one.

    Returns:
        (uncached (item, plan) pairs, number of cache-served requests,
        oversized documents that must run live)
    """
    from markitai.workflow.helpers import extract_document_context

    # Production callers always supply the current run's manifest. The
    # directory fallback is only for explicit internal directory planning.
    if base_files is None:
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
        source = str(base_md.relative_to(output_dir)).removesuffix(".md")
        markdown = _body_without_frontmatter(base_md.read_text(encoding="utf-8"))
        relative_base = str(base_md.relative_to(output_dir))

        pages = _document_pages(base_md) if analyze_pages else []
        plan = None
        if len(pages) > max_pages:
            # The live path splits a long document into ordered rounds, and
            # each batch round is a separate 24-hour wait. Sending every page
            # in one request instead would be one enormous upload. This
            # document is enhanced live below; the rest still get the
            # discount.
            oversized.append((base_md, source, pages))
        elif pages:
            plan = processor.documents.prepare_vision_plan(markdown, pages, source)
        else:
            plan = processor.documents._prepare_document_plan(markdown, source)
            if plan.chunk_calls:
                # A document past the per-call size is cleaned chunk by chunk;
                # the batch carries one request per document, so submitting
                # only plan.call would enhance the first chunk alone.
                oversized.append((base_md, source, []))
                plan = None
        if plan is not None:
            hit = processor._engine.try_cached(plan.call)
            if hit is not None:
                cleaned, frontmatter = _finalize_doc(processor, plan, hit, bool(pages))
                _write_llm_md(
                    processor, base_md, frontmatter, cleaned, cfg=cfg, written=written
                )
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
        # The live path's snippet, so both paths share the cache key and the
        # language hint (a longer raw prefix hit neither's cache entries).
        document_context = extract_document_context(markdown)
        for image in _document_images(base_md, markdown):
            image_plan = processor.vision.prepare_image_plan(
                image, context=source, document_context=document_context
            )
            item = BatchDocItem(
                custom_id=_batch_custom_id(len(pending), image.stem, "img"),
                source=source,
                input_md="",
                base_md=relative_base,
                kind="image",
                image=str(image.relative_to(output_dir)),
                document_context=document_context,
            )
            # An unsupported format or a cache hit is already the answer;
            # it is collected from the plan rather than asked for again.
            if image_plan.answer is not None:
                cached += 1
                if cached_images is not None:
                    item.custom_id = _batch_custom_id(
                        len(cached_images), image.stem, "cached"
                    )
                    item.answer = _image_entry(image, image_plan.answer)
                    cached_images.append(item)
                continue
            pending.append((item, image_plan))
    return pending, cached, oversized


def _write_llm_md(
    processor: Any,
    base_md: Path,
    frontmatter: str,
    cleaned: str,
    *,
    cfg: Any = None,
    written: set[Path] | None = None,
) -> Path:
    """Write the enhanced .llm.md next to its base file.

    Assembled through ``format_llm_output`` so the frontmatter fences and
    body spacing are byte-identical to the live path.
    """
    from markitai.security import atomic_write_text

    target = base_md.with_suffix(".llm.md")
    atomic_write_text(target, processor.format_llm_output(cleaned, frontmatter))
    if cfg is not None:
        from markitai.output_profiles import apply_profile_to_file

        apply_profile_to_file(target, base_md.parent, cfg)
    if written is not None:
        written.add(base_md)
    return target


def uncollected_batches(output_dir: Path) -> list[BatchRunState]:
    """Batches submitted from ``output_dir`` whose results nobody collected.

    A batch that ended without results (``ended_status``) is not one: there
    is nothing left to collect for it.
    """
    from markitai.constants import MARKITAI_META_DIR

    found: list[BatchRunState] = []
    meta = output_dir / MARKITAI_META_DIR
    for state_file in sorted(meta.glob("batch-*/state.json")):
        try:
            state = BatchRunState.load(state_file.parent)
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug("[Batch] unreadable run state {}", state_file)
            continue
        if state.collected_at is None and state.ended_status is None:
            found.append(state)
    return found


def report_uncollected_batches(
    output_dir: Path, *, config_path: Path | None = None
) -> int:
    """Name each uncollected batch and the command that finishes it.

    Printed even under --quiet/--json, like a handoff: ``--resume`` leaves
    these documents out, and this is the only place that says why and how
    to get the results already paid for.

    Returns:
        How many such batches there are.
    """
    batches = uncollected_batches(output_dir)
    console = get_stderr_console()
    for state in batches:
        documents = {item.base_md for item in state.items}
        command = BatchHandoff(
            state.batch_id, "unknown", output_dir, "", config_path=config_path
        ).collect_command
        console.print(
            f"Batch {state.batch_id} from an earlier run was submitted but never "
            f"collected; its {len(documents)} document(s) are left out of this "
            "run so they are not paid for twice. Collect it with:",
            markup=False,
            highlight=False,
        )
        console.print(f"  {command}", markup=False, highlight=False)
    return len(batches)


def resumed_unenhanced_outcomes(
    output_dir: Path, known: list[Outcome]
) -> list[Outcome]:
    """Outcomes for documents an earlier run converted but never enhanced.

    ``--llm-batch --resume`` only converts what is left, so this run's
    manifest misses documents the interrupted (or failed-to-submit) run
    already converted: their base ``.md`` is on disk, their ``.llm.md`` is
    not. The freshest state file under ``output_dir`` is the one the
    conversion pass just compacted, and it names them.

    A document still waiting in a submitted, uncollected batch is not one
    of them: submitting it again would pay for it twice.
    ``--llm-batch-collect`` finishes it (see
    :func:`report_uncollected_batches`).

    Args:
        output_dir: The batch output directory.
        known: Outcomes already in the manifest (this run's items).

    Returns:
        One completed file outcome per such document, in state order.
    """
    import json

    from markitai.constants import STATES_REL_PATH

    states = sorted(
        (output_dir / STATES_REL_PATH).glob("*.state.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not states:
        return []
    try:
        data = json.loads(states[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    seen = {item.output_path.resolve() for item in known if item.output_path}
    seen.update(
        (output_dir / item.base_md).resolve()
        for state in uncollected_batches(output_dir)
        for item in state.items
    )
    extra: list[Outcome] = []
    for key, entry in (data.get("documents") or {}).items():
        output = entry.get("output") if isinstance(entry, dict) else None
        if entry.get("status") != "completed" or not isinstance(output, str):
            continue
        stored = Path(output)
        candidates = [stored if stored.is_absolute() else Path.cwd() / stored]
        candidates.append(output_dir / Path(key).parent / stored.name)
        base = next((path for path in candidates if path.is_file()), None)
        if (
            base is None
            or base.name.endswith(".llm.md")
            or base.with_suffix(".llm.md").exists()
            or base.resolve() in seen
        ):
            continue
        seen.add(base.resolve())
        try:
            # Same spelling as this run's outcomes: rooted at output_dir as given
            base = output_dir / base.resolve().relative_to(output_dir.resolve())
        except ValueError:
            continue  # not an output of this directory
        extra.append(
            Outcome(kind="file", source=key, status="completed", output_path=base)
        )
    return extra


async def run_batch_llm_enhancement(
    cfg: Any,
    output_dir: Path,
    *,
    items: list[Outcome],
    timeout_s: float = 3600.0,
    quiet: bool = False,
    config_path: Path | None = None,
    on_submitted: Callable[[BatchHandoff], None] | None = None,
) -> BatchHandoff | None:
    """Enhance only this run's successful artifacts and update their outcomes.

    Only a Markdown output can be enhanced: a URL captured with
    ``--screenshot-only`` (its output is the screenshot) is left as it is,
    with a warning on the item.

    Args:
        config_path: The run's ``-c`` file, named in every collect command.
        on_submitted: Called once the batch is accepted, with the handoff a
            Ctrl-C during the wait should report: from then on the batch
            runs server-side whether or not this process keeps waiting.

    Returns:
        None when the enhancement finished (each outcome then points at its
        ``.llm.md``, or is marked failed with the reason). A
        :class:`BatchHandoff` when the batch is still in flight: the
        outcomes it covers become ``"pending"`` — their base ``.md`` is
        written and the enhancement is paid for, just not collected yet.
        The same happens when the wait is interrupted after submission.

    Raises:
        ConversionError: Pool/config unsupported, or a Batch API call
            failed (the outcomes are marked failed with the same message).
    """
    from markitai.workflow.helpers import create_llm_processor

    _single_batch_model(cfg)
    completed = [
        item
        for item in items
        if item.status == "completed" and item.output_path is not None
    ]
    selected: list[Outcome] = []
    unenhanceable: list[Outcome] = []
    for item in completed:
        assert item.output_path is not None
        is_markdown = item.output_path.suffix == ".md"
        (selected if is_markdown else unenhanceable).append(item)
    for item in unenhanceable:
        assert item.output_path is not None
        item.warnings.append(
            f"--llm-batch cannot enhance {item.output_path.name} (not Markdown, "
            "e.g. a --screenshot-only capture); it was kept without LLM output. "
            "Run without --llm-batch to enhance it."
        )
    if unenhanceable:
        # Warned under --quiet too: the item looks complete, and nothing
        # else says it got no LLM pass.
        get_stderr_console().print(
            f"[yellow]{len(unenhanceable)} item(s) have no Markdown to enhance "
            "in batch mode (e.g. --screenshot-only captures); they were kept "
            "as is. Run without --llm-batch to enhance them.[/yellow]"
        )
    if not selected:
        # Nothing converted (every item failed) or nothing enhanceable: no
        # processor, no API, no misleading "all cached" line.
        return None

    processor = create_llm_processor(cfg)
    base_files = list(
        dict.fromkeys(
            item.output_path for item in selected if item.output_path is not None
        )
    )
    written: set[Path] = set()
    failures: dict[Path, str] = {}
    warnings: dict[Path, list[str]] = {}
    submitted: list[BatchHandoff] = []

    def _record_submission(handoff: BatchHandoff) -> None:
        submitted.append(handoff)
        if on_submitted is not None:
            on_submitted(handoff)

    started = time.perf_counter()
    error: str | None = None
    handoff: BatchHandoff | None = None
    finished = False
    try:
        handoff = await _run_batch_llm_enhancement(
            cfg,
            output_dir,
            processor=processor,
            base_files=base_files,
            written=written,
            failures=failures,
            warnings=warnings,
            timeout_s=timeout_s,
            quiet=quiet,
            config_path=config_path,
            on_submitted=_record_submission,
        )
        finished = True
        return handoff
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        duration = time.perf_counter() - started
        # Ctrl-C (CancelledError/KeyboardInterrupt) after submission: the
        # batch still runs and is paid for, exactly like a timeout.
        in_flight = handoff is not None or (
            not finished and error is None and bool(submitted)
        )
        for item in selected:
            base = item.output_path
            assert base is not None
            source = str(base.relative_to(output_dir)).removesuffix(".md")
            item.duration = (item.duration or 0.0) + duration
            item.cost_usd += processor.get_context_cost(source)
            from markitai.workflow.helpers import merge_llm_usage

            merge_llm_usage(item.llm_usage, processor.get_context_usage(source))
            item.warnings.extend(warnings.get(base, []))
            if base in written:
                item.output_path = base.with_suffix(".llm.md")
            elif base in failures:
                # Already failed live (an oversized document): a handoff
                # does not make it pending, nothing in the batch covers it.
                item.status = "failed"
                item.error = failures[base]
            elif in_flight:
                item.status = "pending"
            elif error is not None:
                item.status = "failed"
                item.error = error


async def _run_batch_llm_enhancement(
    cfg: Any,
    output_dir: Path,
    *,
    processor: Any,
    base_files: list[Path],
    written: set[Path],
    failures: dict[Path, str] | None = None,
    warnings: dict[Path, list[str]] | None = None,
    timeout_s: float = 3600.0,
    quiet: bool = False,
    config_path: Path | None = None,
    on_submitted: Callable[[BatchHandoff], None] | None = None,
) -> BatchHandoff | None:
    """Submit and (mostly) wait for a Batch API enhancement run.

    Returns:
        None when everything finished (or there was nothing to do); a
        :class:`BatchHandoff` when the batch is still in flight past the
        timeout, or the wait lost contact with the API — the run state is on
        disk and ``--llm-batch-collect`` finishes it later.

    Raises:
        ConversionError: Pool/config unsupported, a submit/download call
            failed, or the batch ended in a non-completed terminal state.
    """
    model, provider = _single_batch_model(cfg)
    mode = instructor_mode_for_model(f"{provider}/{model}")

    analyze_images = bool(cfg.image.alt_enabled or cfg.image.desc_enabled)
    cached_images: list[BatchDocItem] = []
    pending, cached, oversized = _prepare_pending(
        processor,
        output_dir,
        base_files=base_files,
        written=written,
        cfg=cfg,
        analyze_images=analyze_images,
        analyze_pages=bool(cfg.screenshot.enabled),
        cached_images=cached_images,
    )
    for base_md, source, pages in oversized:
        if pages:
            logger.info(
                f"[Batch] {source}: {len(pages)} pages exceed the "
                f"{DEFAULT_MAX_PAGES_PER_BATCH}-page single-call limit; enhancing "
                "live at full price — batching it would mean one round trip per "
                "round, each up to 24h"
            )
        else:
            logger.info(
                f"[Batch] {source}: longer than one request; enhancing live, "
                "chunk by chunk, at full price"
            )
        markdown = _body_without_frontmatter(base_md.read_text(encoding="utf-8"))
        try:
            if pages:
                (
                    cleaned,
                    frontmatter,
                ) = await processor.documents.enhance_document_complete(
                    markdown, pages, source=source
                )
            else:
                cleaned, frontmatter = await processor.documents.process_document(
                    markdown, source
                )
        except Exception as e:
            message = f"LLM enhancement failed: {format_error_message(e)}"
            logger.error(f"[Batch] {source}: {message}")
            if failures is not None:
                failures[base_md] = message
            continue
        _write_llm_md(
            processor, base_md, frontmatter, cleaned, cfg=cfg, written=written
        )
    if cached:
        logger.info(f"[Batch] {cached} request(s) served from cache")
    # A cache-served image answer belongs in its document's .llm.md. Those
    # already written (cached or enhanced live) take it now; the rest wait
    # in the run state for the batch that writes them.
    ready: list[BatchDocItem] = []
    deferred: list[BatchDocItem] = []
    for item in cached_images:
        (ready if output_dir / item.base_md in written else deferred).append(item)
    if not pending:
        _apply_image_answers(cfg, output_dir, _answers_by_base(cached_images))
        if cached and not quiet:
            get_stderr_console().print(
                "All documents already cached — nothing to submit."
            )
        return None
    _apply_image_answers(cfg, output_dir, _answers_by_base(ready))

    # Only a submission needs them: a fully cached run talks to no API.
    credentials = _batch_credentials(cfg, model, provider)
    state_dir = BatchRunState.state_dir_for(output_dir, "pending")
    # Leftovers of a run that died before submitting describe nothing that
    # exists server-side; staging over them would mix two runs' inputs.
    shutil.rmtree(state_dir, ignore_errors=True)
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
        get_stderr_console().print(
            f"Submitting {len(pending)} request(s) to the Batch API "
            f"({model}, 50% of list price)..."
        )
    try:
        if provider == "anthropic":
            batch_id = await submit_anthropic_batch(requests, credentials=credentials)
        else:
            batch_id = await submit_openai_batch(
                jsonl_path, custom_llm_provider=provider, credentials=credentials
            )
    except Exception as e:
        # Nothing exists server-side, so the staged requests describe no
        # batch anyone could collect.
        shutil.rmtree(state_dir, ignore_errors=True)
        raise _api_failure(
            "submission",
            e,
            "Nothing was submitted; the base .md files are kept. Fix the "
            f"cause and re-run ({_RERUN_HINT}).",
        ) from e

    # Persist under the real batch id (move the pending dir into place)
    final_state_dir = BatchRunState.state_dir_for(output_dir, batch_id)
    state_dir.rename(final_state_dir)
    state = BatchRunState(
        batch_id=batch_id,
        model=model,
        mode=mode.value,
        provider=provider,
        created_at=datetime.now(UTC).astimezone().isoformat(),
        items=[item for item, _ in pending] + deferred,
        alt_enabled=bool(cfg.image.alt_enabled),
        desc_enabled=bool(cfg.image.desc_enabled),
    )
    state.save(final_state_dir)
    if on_submitted is not None:
        on_submitted(
            BatchHandoff(
                batch_id,
                "in_progress",
                output_dir,
                f"Interrupted while waiting for batch {batch_id}.",
                config_path=config_path,
            )
        )

    def _progress(status: str, done: int, total: int) -> None:
        if not quiet:
            get_stderr_console().print(
                f"\rBatch {batch_id}: {status} ({done}/{total})", end=""
            )

    try:
        if provider == "anthropic":
            status = await poll_anthropic_batch(
                batch_id,
                credentials=credentials,
                timeout_s=timeout_s,
                on_progress=_progress,
            )
        else:
            status = await poll_openai_batch(
                batch_id,
                custom_llm_provider=provider,
                credentials=credentials,
                timeout_s=timeout_s,
                on_progress=_progress,
            )
    except TimeoutError as e:
        if not quiet:
            get_stderr_console().print()
        return BatchHandoff(
            batch_id,
            "in_progress",
            output_dir,
            f"Batch still in flight: {e}",
            config_path=config_path,
        )
    except Exception as e:
        # The batch was accepted; losing contact while waiting does not stop
        # it. That is the same situation as a timeout, not a failed run.
        if not quiet:
            get_stderr_console().print()
        return BatchHandoff(
            batch_id,
            "unknown",
            output_dir,
            f"Lost contact with the Batch API while waiting for {batch_id}: "
            f"{format_error_message(e)}",
            config_path=config_path,
        )

    if not quiet:
        get_stderr_console().print()
    if status != "completed":
        state.ended_status = status
        state.save(final_state_dir)
        raise ConversionError(
            f"Batch {batch_id} ended with status={status!r}; no results to "
            "collect. The base .md files are kept; re-run to submit a new "
            f"batch ({_RERUN_HINT})."
        )

    await _finish_batch(
        cfg,
        processor,
        output_dir,
        state,
        quiet=quiet,
        written=written,
        failures=failures,
        warnings=warnings,
        credentials=credentials,
    )
    return None


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
) -> LLMUsageByModel:
    """Record one batched answer's usage at the discounted rate.

    Returns:
        The same usage in ``llm_usage`` shape, for the answer's own record.
    """
    input_tokens, output_tokens, cost = _batch_usage(
        line.body, state.model, state.provider
    )
    discounted = cost * BATCH_COST_FACTOR
    processor._track_usage(
        state.model,
        input_tokens,
        output_tokens,
        discounted,
        source,
    )
    return {
        state.model: ModelUsageStats(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=discounted,
        )
    }


async def _collect_image(
    processor: Any,
    state: BatchRunState,
    output_dir: Path,
    item: BatchDocItem,
    line: Any,
    mode: Any,
) -> dict[str, Any] | None:
    """Turn one batched image answer into an images.json / alt-text entry.

    A failed image degrades rather than raising: the live path treats image
    analysis as non-critical, and a batch must not be stricter than the path
    it replaces. It returns None, so the document keeps the alt text it was
    converted with and images.json gets no made-up entry; the caller records
    the failure where the run's results can show it.

    The plan is rebuilt with the document snippet the request was submitted
    with, so the cache key and the language hint match the submission.

    The offline path cannot run the live path's language retries — each
    needs to see an answer before deciding whether to ask again — so a
    drifting answer is rewritten here, live, at collect time. That is one
    cheap text call, not another 24-hour round trip.
    """
    from markitai.llm.vision import _should_retry_for_language

    image = output_dir / item.image
    try:
        plan = processor.vision.prepare_image_plan(
            image, context=item.source, document_context=item.document_context
        )
        if plan.answer is not None:
            analysis = plan.answer
        else:
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
            usage = _account_batch_usage(processor, state, line, item.source)
            # The parser hands back the bare response model; the rest of the
            # pipeline (language rewrite, images.json) reads ImageAnalysis.
            analysis = ImageAnalysis(
                caption=result.caption.strip(),
                description=result.description,
                extracted_text=result.extracted_text,
                llm_usage=usage,
            )
            if _should_retry_for_language(analysis, plan.language):
                analysis = await processor.vision._rewrite_analysis_language(
                    analysis,
                    language=plan.language,
                    context=item.source,
                    document_context=plan.document_context,
                )
            analysis = processor.vision.finalize_image_plan(plan, analysis)
    except Exception as e:
        logger.warning(
            f"[Batch] image {image.name} failed in batch "
            f"({format_error_message(e)}); keeping its original alt text"
        )
        return None

    return _image_entry(image, analysis)


def _image_entry(image: Path, analysis: Any) -> dict[str, Any]:
    """One image's images.json / alt-text entry, as the live path records it."""
    return {
        "asset": str(image.resolve()),
        "alt": analysis.caption or "Image",
        "desc": analysis.description,
        "text": analysis.extracted_text or "",
        "llm_usage": analysis.llm_usage or {},
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _answers_by_base(items: list[BatchDocItem]) -> dict[str, list[dict[str, Any]]]:
    """Cache-served image answers, grouped by the document they belong to."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item.answer is not None:
            grouped.setdefault(item.base_md, []).append(item.answer)
    return grouped


def _apply_image_answers(
    cfg: Any,
    output_dir: Path,
    images_by_base: dict[str, list[dict[str, Any]]],
    state: BatchRunState | None = None,
) -> None:
    """Write the collected image answers into each document's output.

    Mirrors the live path's split: alt text is substituted into the written
    ``.llm.md``, descriptions go to images.json. A batch's own record of
    ``--alt``/``--desc`` wins over ``cfg``: a collect run's config need not
    repeat the submitting run's flags.
    """
    if not images_by_base:
        return
    alt_enabled = cfg.image.alt_enabled
    desc_enabled = cfg.image.desc_enabled
    if state is not None and state.alt_enabled is not None:
        alt_enabled = state.alt_enabled
    if state is not None and state.desc_enabled is not None:
        desc_enabled = state.desc_enabled
    from markitai.output_profiles import assets_visible
    from markitai.workflow.core import apply_alt_text_updates
    from markitai.workflow.helpers import write_images_json
    from markitai.workflow.single import ImageAnalysisResult as AnalysisForSource

    results = []
    for relative_base, assets in sorted(images_by_base.items()):
        base_md = output_dir / relative_base
        result = AnalysisForSource(source_file=str(base_md.resolve()), assets=assets)
        results.append(result)
        if alt_enabled:
            apply_alt_text_updates(base_md.with_suffix(".llm.md"), result)
    if desc_enabled:
        write_images_json(output_dir, results, visible_assets=assets_visible(cfg))


async def _finish_batch(
    cfg: Any,
    processor: Any,
    output_dir: Path,
    state: BatchRunState,
    *,
    quiet: bool,
    written: set[Path] | None = None,
    failures: dict[Path, str] | None = None,
    warnings: dict[Path, list[str]] | None = None,
    credentials: BatchCredentials | None = None,
) -> int:
    """Download results and finalize each document (live re-run on failure).

    Every item is attempted even when an earlier one fails: the batch is
    already paid for, so one document's failed live re-run must not cost
    the rest their results. Such a document lands in ``failures`` (keyed by
    its base .md), an image that could not be analyzed in ``warnings``.

    Returns:
        0 when every document was written, 10 (partial failure) otherwise.
        The state is marked collected only in the first case, so a second
        ``--llm-batch-collect`` can still retry the documents that failed.

    Raises:
        ConversionError: The results could not be downloaded.
    """
    import instructor

    state_dir = BatchRunState.state_dir_for(output_dir, state.batch_id)
    out_path = state_dir / "output.jsonl"
    try:
        if state.provider == "anthropic":
            await download_anthropic_batch_output(
                state.batch_id, out_path, credentials=credentials
            )
        else:
            await download_openai_batch_output(
                state.batch_id,
                out_path,
                custom_llm_provider=state.provider,
                credentials=credentials,
            )
    except Exception as e:
        raise _api_failure(
            "download",
            e,
            f"Batch {state.batch_id} is kept; retry with: "
            + BatchHandoff(state.batch_id, "completed", output_dir, "").collect_command,
        ) from e

    mode = instructor.Mode(state.mode)
    lines = {line.custom_id: line for line in read_openai_batch_output(out_path)}
    done = 0
    reran = 0
    doc_failures: dict[Path, str] = {}
    image_failures: dict[Path, list[str]] = {}
    # Image answers, gathered per owning document: alt text goes into the
    # .llm.md the document's own result writes, so it can only be applied
    # once every request for that document has been read.
    images_by_base: dict[str, list[dict[str, Any]]] = {}

    for item in state.items:
        base_md = output_dir / item.base_md
        line = lines.get(item.custom_id)

        if item.kind == "image" and item.answer is not None:
            # Served from the cache at submission; only its document's
            # .llm.md had to wait for the batch.
            images_by_base.setdefault(item.base_md, []).append(item.answer)
            continue
        if item.kind == "image":
            analysis = await _collect_image(
                processor, state, output_dir, item, line, mode
            )
            if analysis is None:
                image_failures.setdefault(base_md, []).append(
                    f"image analysis failed for {Path(item.image).name}; "
                    "its original alt text was kept"
                )
                continue
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
                markdown, _document_pages(base_md), item.source
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
            _write_llm_md(
                processor, base_md, frontmatter, cleaned, cfg=cfg, written=written
            )
            done += 1
        except Exception as e:
            logger.warning(
                f"[Batch] {item.source} failed in batch "
                f"({format_error_message(e)}); re-running live"
            )
            try:
                if vision:
                    (
                        cleaned,
                        frontmatter,
                    ) = await processor.documents.enhance_document_complete(
                        markdown,
                        _document_pages(base_md),
                        source=item.source,
                    )
                else:
                    cleaned, frontmatter = await processor.documents.process_document(
                        markdown, item.source
                    )
            except Exception as live_error:
                message = (
                    "LLM enhancement failed in the batch and again live: "
                    f"{format_error_message(live_error)}"
                )
                logger.error(f"[Batch] {item.source}: {message}")
                doc_failures[base_md] = message
                continue
            _write_llm_md(
                processor, base_md, frontmatter, cleaned, cfg=cfg, written=written
            )
            reran += 1

    _apply_image_answers(cfg, output_dir, images_by_base, state)
    if failures is not None:
        failures.update(doc_failures)
    if warnings is not None:
        for base_md, messages in image_failures.items():
            warnings.setdefault(base_md, []).extend(messages)

    if not doc_failures:
        state.collected_at = datetime.now(UTC).astimezone().isoformat()
        state.save(state_dir)

    console = get_stderr_console()
    if not quiet:
        console.print(
            f"Batch enhancement done: {done} via batch"
            + (f", {reran} re-ran live" if reran else "")
            + (
                f" (billed at {int(BATCH_COST_FACTOR * 100)}% of list price)"
                if done
                else ""
            )
        )
    failed_images = sum(len(messages) for messages in image_failures.values())
    if failed_images and not quiet:
        console.print(
            f"[yellow]{failed_images} image(s) could not be analyzed; their "
            "original alt text was kept.[/yellow]"
        )
    if doc_failures:
        # Printed under --quiet too, like any other error: these documents
        # have no .llm.md.
        console.print(
            f"[red]{len(doc_failures)} document(s) could not be enhanced; "
            "their base .md is kept. Re-run --llm-batch-collect to retry "
            "them.[/red]"
        )
        return 10
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
        0 when finished (or already collected); 2 when the batch is still in
        flight; 10 when some documents could not be enhanced.

    Raises:
        ConversionError: No run state found, the batch failed, or a Batch
            API call failed.
    """
    state_dir = BatchRunState.state_dir_for(output_dir, batch_id)
    if not (state_dir / "state.json").exists():
        raise ConversionError(
            f"No pending batch state at {state_dir} — was this output_dir "
            "the one used for the original --llm-batch run?"
        )
    state = BatchRunState.load(state_dir)
    if state.collected_at is not None:
        # Collecting again would download the same results and rewrite
        # every .llm.md, overwriting any edit made since.
        if not quiet:
            get_stderr_console().print(
                f"Batch {batch_id} was already collected ({state.collected_at}); "
                f"its outputs are in {output_dir}. Nothing to do."
            )
        return 0

    credentials = _batch_credentials(cfg, state.model, state.provider)
    try:
        # A status check, not a wait — the caller is asking "is it done yet?"
        if state.provider == "anthropic":
            status = await poll_anthropic_batch(
                batch_id, credentials=credentials, timeout_s=5, interval_s=2
            )
        else:
            status = await poll_openai_batch(
                batch_id,
                custom_llm_provider=state.provider,
                credentials=credentials,
                timeout_s=5,
                interval_s=2,
            )
    except TimeoutError:
        status = "in_progress"
    except Exception as e:
        raise _api_failure(
            "status check",
            e,
            f"Batch {batch_id} is kept; retry the same command later.",
        ) from e
    if status != "completed":
        if status in ("failed", "expired", "cancelled"):
            # Nothing will ever be collected: let --resume submit these
            # documents again rather than skip them as still in flight.
            state.ended_status = status
            state.save(state_dir)
            raise ConversionError(
                f"Batch {batch_id} ended with status={status!r}; no results to "
                "collect. The base .md files are kept."
            )
        if not quiet:
            get_stderr_console().print(
                f"Batch {batch_id} is still {status!r} — try again later."
            )
        return 2

    from markitai.workflow.helpers import create_llm_processor

    processor = create_llm_processor(cfg)
    return await _finish_batch(
        cfg, processor, output_dir, state, quiet=quiet, credentials=credentials
    )


def _body_without_frontmatter(text: str) -> str:
    """The document body, with any leading YAML frontmatter removed."""
    _frontmatter, body = split_frontmatter(text)
    return body

"""Offline Batch API execution for document LLM calls (OpenAI-compatible).

The interactive structured ladder (engine.run_structured_ladder) descends
rungs on live failure; a batch has no in-flight interaction, so the rung is
preselected statically (``instructor_mode_for_model``) and documents whose
request fails in the batch are the caller's problem — the intended fallback
is a live re-run of just those documents.

Both request building and result parsing reuse instructor's own pure
functions (``handle_response_model`` / ``process_response``), so an offline
request is byte-identical in shape to what the live ladder would send.

Submit/poll/download go through litellm's batch helpers (OpenAI and
OpenAI-compatible providers). Anthropic's Message Batches API is a
different shape and intentionally not wired here yet — callers should
refuse non-OpenAI pools with an actionable message.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import instructor
from loguru import logger

# ---------------------------------------------------------------------------
# Pending-run state (survives process exit for two-phase collect)
# ---------------------------------------------------------------------------


@dataclass
class BatchDocItem:
    """One document's slot in a pending batch run."""

    custom_id: str
    source: str  # document name (LLM context identifier)
    input_md: str  # LLM-input markdown file, relative to the state dir
    base_md: str  # base .md path, relative to output_dir (.llm.md derives)


@dataclass
class BatchRunState:
    """Everything needed to collect a submitted batch hours later.

    Written to ``<output_dir>/.markitai/batch-<batch_id>/state.json`` next to
    the request jsonl and the per-document LLM-input markdown files, so
    collection never depends on the original input directory.
    """

    batch_id: str
    model: str
    mode: str  # instructor mode value (tools / json_schema / md_json)
    provider: str  # litellm custom_llm_provider
    created_at: str
    items: list[BatchDocItem] = field(default_factory=list)

    def save(self, state_dir: Path) -> Path:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, state_dir: Path) -> BatchRunState:
        data = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        items = [BatchDocItem(**item) for item in data.pop("items")]
        return cls(items=items, **data)

    @staticmethod
    def state_dir_for(output_dir: Path, batch_id: str) -> Path:
        return output_dir / ".markitai" / f"batch-{batch_id}"


# ---------------------------------------------------------------------------
# Request building (pure, no network)
# ---------------------------------------------------------------------------


def build_openai_batch_request(
    custom_id: str,
    *,
    messages: list[dict[str, Any]],
    response_model: type,
    model: str,
    mode: instructor.Mode,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build one OpenAI Batches API request from a structured call's parts.

    The body is produced by instructor's own ``handle_response_model``, so
    TOOLS/JSON_SCHEMA/MD_JSON requests are shaped exactly as the live
    ladder would shape them (including MD_JSON's system-message schema
    append).

    Args:
        custom_id: Caller-chosen id echoed back in the output file.
        messages: Chat messages (mutated per mode by instructor; the input
            is deep-copied first).
        response_model: Pydantic model the answer is validated against.
        model: Concrete deployment id (e.g. ``"gpt-5.6-luna"``).
        mode: Preselected instructor mode (see ``instructor_mode_for_model``).
        max_tokens: Optional explicit output cap.

    Returns:
        One ``{"custom_id", "method", "url", "body"}`` request dict.
    """
    from instructor.v2.core.response import handle_response_model

    _, mode_kwargs = handle_response_model(
        response_model=response_model,
        mode=mode,
        messages=copy.deepcopy(messages),
    )
    body: dict[str, Any] = {"model": model, **mode_kwargs}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def write_batch_jsonl(requests: list[dict[str, Any]], path: Path) -> None:
    """Write batch requests as jsonl (one request per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Result parsing (pure, no network)
# ---------------------------------------------------------------------------


@dataclass
class BatchOutputLine:
    """One line of an OpenAI batch output file."""

    custom_id: str
    body: dict[str, Any] | None  # chat-completion body when status 200
    error: str | None  # provider error message, or HTTP status when not 200


def read_openai_batch_output(path: Path) -> Iterator[BatchOutputLine]:
    """Iterate a downloaded OpenAI batch output jsonl."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            response = entry.get("response") or {}
            status = response.get("status_code")
            error = entry.get("error")
            if error:
                yield BatchOutputLine(custom_id, None, str(error))
            elif status == 200:
                yield BatchOutputLine(custom_id, response.get("body"), None)
            else:
                yield BatchOutputLine(custom_id, None, f"HTTP {status}")


def parse_batch_result(
    body: dict[str, Any],
    *,
    response_model: type,
    mode: instructor.Mode,
) -> Any:
    """Validate a batch output body into the response model.

    Reuses instructor's ``process_response`` — the same parser the live
    ladder uses — so tool-call and JSON payloads are read identically.
    Raises (ValidationError, JSONDecodeError, ...) on unparseable output;
    callers route those documents to a live re-run.
    """
    import litellm
    from instructor.v2.core.response import process_response

    response = litellm.ModelResponse(**body)
    return process_response(
        response,
        response_model=response_model,
        stream=False,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Submit / poll / download (NETWORK CALLS — they cost money and time)
# ---------------------------------------------------------------------------


async def submit_openai_batch(
    jsonl_path: Path, *, custom_llm_provider: str = "openai"
) -> str:
    """Upload + submit a batch job via litellm. Returns the batch id."""
    import litellm

    file_obj = await litellm.acreate_file(
        file=jsonl_path,
        purpose="batch",
        custom_llm_provider=cast("Any", custom_llm_provider),
    )
    batch = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider=cast("Any", custom_llm_provider),
    )
    logger.info(f"[Batch] Submitted {jsonl_path.name} as {batch.id}")
    return batch.id


async def poll_openai_batch(
    batch_id: str,
    *,
    custom_llm_provider: str = "openai",
    timeout_s: float = 3600.0,
    interval_s: float = 30.0,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """Poll a batch to a terminal status. NETWORK CALL.

    Args:
        batch_id: Provider batch id.
        custom_llm_provider: litellm provider name.
        timeout_s: Give up after this many seconds (TimeoutError; the batch
            keeps running server-side — collect it later by id).
        interval_s: Seconds between status polls.
        on_progress: Optional ``(status, completed, total)`` hook for UI.

    Returns:
        Terminal status string ("completed" / "failed" / "expired" /
        "cancelled"). Only "completed" has downloadable output.

    Raises:
        TimeoutError: Still in flight after ``timeout_s``.
    """
    import litellm

    elapsed = 0.0
    while True:
        batch = await litellm.aretrieve_batch(
            batch_id, custom_llm_provider=cast("Any", custom_llm_provider)
        )
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        completed = int(getattr(counts, "completed", 0) or 0)
        total = int(getattr(counts, "total", 0) or 0)
        if on_progress is not None:
            on_progress(status, completed, total)
        if status in ("completed", "failed", "expired", "cancelled"):
            return status
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"batch {batch_id} still {status!r} after {timeout_s:.0f}s "
                f"({completed}/{total} done)"
            )
        await asyncio.sleep(interval_s)
        elapsed += interval_s


async def download_openai_batch_output(
    batch_id: str, output_path: Path, *, custom_llm_provider: str = "openai"
) -> Path:
    """Download a completed batch's output file. NETWORK CALL."""
    import litellm

    batch = await litellm.aretrieve_batch(
        batch_id, custom_llm_provider=cast("Any", custom_llm_provider)
    )
    if batch.status != "completed":
        raise RuntimeError(f"batch {batch_id} is {batch.status!r}, not completed")
    if not batch.output_file_id:
        raise RuntimeError(f"batch {batch_id} completed with no output_file_id")
    content = await litellm.afile_content(
        batch.output_file_id, custom_llm_provider=cast("Any", custom_llm_provider)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(cast("Any", content).content)
    return output_path

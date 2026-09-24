"""Offline Batch API execution for document LLM calls (OpenAI-compatible).

The interactive structured ladder (engine.run_structured_ladder) descends
rungs on live failure; a batch has no in-flight interaction, so the rung is
preselected statically (``instructor_mode_for_model``) and documents whose
request fails in the batch are the caller's problem — the intended fallback
is a live re-run of just those documents.

Both request building and result parsing reuse instructor's own pure
functions (``handle_response_model`` / ``process_response``), so an offline
request is byte-identical in shape to what the live ladder would send.

Two providers, two transports. OpenAI-compatible batches go through
litellm's helpers (upload a jsonl of ``/v1/chat/completions`` requests,
then poll a batch id). Anthropic's Message Batches API takes the requests
inline and streams results back, and litellm only implements the retrieve
half of it (``transform_create_batch_request`` raises NotImplementedError),
so that side talks to the official ``anthropic`` SDK directly.

What both paths share is everything above the transport: instructor
shapes the request (``ANTHROPIC_TOOLS`` for Anthropic, the live ladder's
own rung otherwise) and parses the answer, results are keyed by
``custom_id``, and a request the batch could not satisfy falls through to
a live re-run of that one document.
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
    """One request's slot in a pending batch run.

    Two kinds share the list because they share a batch: a document's text
    enhancement, and the analysis of one image belonging to a document.
    ``base_md`` identifies the owning document either way, so an image
    result knows which ``.llm.md`` its alt text belongs in.
    """

    custom_id: str
    source: str  # document name (LLM context identifier)
    input_md: str  # LLM-input markdown file, relative to the state dir
    base_md: str  # base .md path, relative to output_dir (.llm.md derives)
    kind: str = "doc"  # "doc" | "image"; absent in states written before images
    image: str = ""  # kind="image": image path relative to output_dir
    # kind="image": the document snippet the request was built with. It
    # feeds both the cache key and the language hint, so the collector must
    # rebuild the plan from the same text rather than from nothing.
    document_context: str = ""
    # kind="image" answered from the cache when the batch was built: its
    # images.json entry, kept here because no request was sent for it. It
    # still has to wait for its document's .llm.md, which the batch writes.
    answer: dict[str, Any] | None = None


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
    # Set once every result was written: a second collect then reports the
    # batch as done instead of downloading and rewriting the outputs again.
    collected_at: str | None = None
    # Set when the batch ended without results (failed / expired /
    # cancelled): nothing is left to collect, so ``--resume`` may submit its
    # documents again instead of waiting on it forever.
    ended_status: str | None = None
    # The submitting run's --alt / --desc. Collection applies the image
    # answers the way that run asked: the collect command's own config may
    # not repeat the flags. None (older states): the collect config decides.
    alt_enabled: bool | None = None
    desc_enabled: bool | None = None

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


@dataclass(frozen=True)
class BatchCredentials:
    """The key and endpoint a batch job talks to.

    Resolved from the pool's ``model_list`` entry the same way the live
    router resolves it, so a configured ``api_key``/``api_base`` wins over
    the provider's environment variables. A ``None`` field falls back to the
    SDK's own environment lookup.
    """

    api_key: str | None = None
    api_base: str | None = None

    def litellm_kwargs(self) -> dict[str, Any]:
        """litellm files/batches kwargs for this credential (unset ones omitted)."""
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs


# ---------------------------------------------------------------------------
# Request building (pure, no network)
# ---------------------------------------------------------------------------


def _supports_reasoning(model: str) -> bool:
    """Whether litellm's capability table marks the model as reasoning-capable.

    An unknown model must not be assumed capable (same rule as the live
    structured ladder), so any lookup failure reads as ``False``.
    """
    import litellm

    try:
        return bool(litellm.supports_reasoning(model))
    except Exception:
        return False


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
    append), with one batch-only addition: a TOOLS request to a reasoning
    model carries ``reasoning_effort="none"``, which the batch deployments
    require of function-tool calls.

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
    if "tools" in body and _supports_reasoning(model):
        # OpenAI's batch deployments refuse function tools while reasoning is
        # on: "Function tools with reasoning_effort are not supported for
        # <model>-batch in /v1/chat/completions. To use function tools, use
        # /v1/responses or set reasoning_effort to 'none'." Live calls keep
        # the deployment's default effort; a batched TOOLS request cannot.
        body["reasoning_effort"] = "none"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def build_anthropic_batch_request(
    custom_id: str,
    *,
    messages: list[dict[str, Any]],
    response_model: type,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one Message Batches request from a structured call's parts.

    Anthropic's Messages API is not the OpenAI shape: the system prompt is
    a top-level field rather than the first message, tools carry
    ``input_schema``, and ``tool_choice`` names the tool directly. Rather
    than translate by hand, this asks instructor for its ``ANTHROPIC_TOOLS``
    rendering — the same function the OpenAI path uses, in the mode that
    speaks Anthropic — so request and reply stay each other's inverse.

    Args:
        custom_id: Caller-chosen id echoed back with the result.
        messages: Chat messages (deep-copied; instructor mutates its input).
        response_model: Pydantic model the answer is validated against.
        model: Concrete model id (e.g. ``"claude-haiku-4-5"``).
        max_tokens: Output cap. Required by the API — unlike OpenAI, there
            is no server-side default to fall back on.

    Returns:
        One ``{"custom_id", "params"}`` request dict.
    """
    from instructor.v2.core.response import handle_response_model

    _, mode_kwargs = handle_response_model(
        response_model=response_model,
        mode=instructor.Mode.ANTHROPIC_TOOLS,
        messages=copy.deepcopy(messages),
    )
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        **mode_kwargs,
    }
    # instructor lifts the system prompt out but leaves content blocks alone,
    # and the plans build OpenAI-shaped image blocks, which the Messages API
    # rejects ("Input tag 'image_url' ... does not match any of the expected
    # tags"). The live path gets this translation from litellm.
    params["messages"] = [
        _anthropic_message(message) for message in params.get("messages", [])
    ]
    return {"custom_id": custom_id, "params": params}


def _anthropic_message(message: dict[str, Any]) -> dict[str, Any]:
    """One chat message with its OpenAI image blocks in Anthropic's shape."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {**message, "content": [_anthropic_block(block) for block in content]}


def _anthropic_block(block: Any) -> Any:
    """Translate an ``image_url`` block; every other block passes through.

    A ``data:`` URI becomes an inline base64 source, anything else a URL
    source the API fetches itself.
    """
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return block
    image_url = block.get("image_url")
    url = str(
        (image_url.get("url") if isinstance(image_url, dict) else image_url) or ""
    )
    if url.startswith("data:") and ";base64," in url:
        header, data = url.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


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
    provider: str = "openai",
) -> Any:
    """Validate a batch output body into the response model.

    Reuses instructor's ``process_response`` — the same parser the live
    ladder uses — so tool-call and JSON payloads are read identically. The
    provider decides which envelope the body is wrapped in first: an
    Anthropic result is a ``Message``, everything else a litellm
    ``ModelResponse``.

    Raises (ValidationError, JSONDecodeError, ...) on unparseable output;
    callers route those documents to a live re-run.
    """
    from instructor.v2.core.response import process_response

    if provider == "anthropic":
        from anthropic.types import Message

        return process_response(
            Message.model_validate(body),
            response_model=response_model,
            stream=False,
            mode=instructor.Mode.ANTHROPIC_TOOLS,
        )

    import litellm

    return process_response(
        litellm.ModelResponse(**body),
        response_model=response_model,
        stream=False,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Submit / poll / download (NETWORK CALLS — they cost money and time)
# ---------------------------------------------------------------------------


async def submit_openai_batch(
    jsonl_path: Path,
    *,
    custom_llm_provider: str = "openai",
    credentials: BatchCredentials | None = None,
) -> str:
    """Upload + submit a batch job via litellm. Returns the batch id."""
    import litellm

    auth = (credentials or BatchCredentials()).litellm_kwargs()
    file_obj = await litellm.acreate_file(
        file=jsonl_path,
        purpose="batch",
        custom_llm_provider=cast("Any", custom_llm_provider),
        **auth,
    )
    batch = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider=cast("Any", custom_llm_provider),
        **auth,
    )
    logger.info(f"[Batch] Submitted {jsonl_path.name} as {batch.id}")
    return batch.id


async def _retrieve_openai_batch(
    batch_id: str, custom_llm_provider: str, auth: dict[str, Any]
) -> Any:
    """``aretrieve_batch`` without litellm's batch-cost logging hook.

    On a completed batch litellm's success logger downloads the whole output
    file to price it, as a background task of every retrieve. That doubles
    the download for each status check and, when the event loop closes
    before the task runs, leaves ``OpenAIFilesAPI.afile_content`` never
    awaited (the RuntimeWarning a repeated collect printed). markitai prices
    the results itself from the file it downloads, so the hook is switched
    off with litellm's own opt-out for polling callers.
    """
    import litellm

    return await litellm.aretrieve_batch(
        batch_id,
        custom_llm_provider=cast("Any", custom_llm_provider),
        litellm_metadata={"batch_ignore_default_logging": True},
        **auth,
    )


async def poll_openai_batch(
    batch_id: str,
    *,
    custom_llm_provider: str = "openai",
    credentials: BatchCredentials | None = None,
    timeout_s: float = 3600.0,
    interval_s: float = 30.0,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """Poll a batch to a terminal status. NETWORK CALL.

    Args:
        batch_id: Provider batch id.
        custom_llm_provider: litellm provider name.
        credentials: The pool entry's key/endpoint (environment otherwise).
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
    auth = (credentials or BatchCredentials()).litellm_kwargs()
    elapsed = 0.0
    while True:
        batch = await _retrieve_openai_batch(batch_id, custom_llm_provider, auth)
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
    batch_id: str,
    output_path: Path,
    *,
    custom_llm_provider: str = "openai",
    credentials: BatchCredentials | None = None,
) -> Path:
    """Download a completed batch's output file. NETWORK CALL.

    A completed batch whose requests all failed has no ``output_file_id`` —
    its error file has the same line shape (custom_id + non-200 response),
    so it is downloaded as the output instead: per-line errors then flow
    into the caller's live re-run fallback instead of aborting the collect.
    """
    import litellm

    auth = (credentials or BatchCredentials()).litellm_kwargs()
    batch = await _retrieve_openai_batch(batch_id, custom_llm_provider, auth)
    if batch.status != "completed":
        raise RuntimeError(f"batch {batch_id} is {batch.status!r}, not completed")
    file_id = batch.output_file_id or batch.error_file_id
    if not file_id:
        raise RuntimeError(
            f"batch {batch_id} completed with no output_file_id or error_file_id"
        )
    if not batch.output_file_id:
        logger.warning(
            f"[Batch] {batch_id} has only an error file — all requests will "
            "fall back to live re-runs"
        )
    content = await litellm.afile_content(
        file_id, custom_llm_provider=cast("Any", custom_llm_provider), **auth
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(cast("Any", content).content)
    return output_path


# ---------------------------------------------------------------------------
# Anthropic Message Batches (NETWORK CALLS — they cost money and time)
# ---------------------------------------------------------------------------


def _anthropic_client(credentials: BatchCredentials | None = None) -> Any:
    """An async Anthropic client, or a readable error about the missing key.

    A configured ``api_key``/``api_base`` wins; the SDK reads
    ``ANTHROPIC_API_KEY``/``ANTHROPIC_BASE_URL`` for whatever is left unset.
    """
    import anthropic

    kwargs: dict[str, Any] = {}
    if credentials is not None and credentials.api_key:
        kwargs["api_key"] = credentials.api_key
    if credentials is not None and credentials.api_base:
        kwargs["base_url"] = credentials.api_base
    try:
        return anthropic.AsyncAnthropic(**kwargs)
    except Exception as e:  # anthropic raises when it finds no credential
        raise RuntimeError(
            "--llm-batch on an Anthropic pool needs an API key: set api_key "
            "on the model_list entry, or ANTHROPIC_API_KEY in the environment "
            f"or ~/.markitai/.env ({e})"
        ) from e


async def submit_anthropic_batch(
    requests: list[dict[str, Any]], *, credentials: BatchCredentials | None = None
) -> str:
    """Create a Message Batch from inline requests. Returns the batch id.

    Unlike the OpenAI path there is no file to upload: the requests travel
    in the create call itself.
    """
    async with _anthropic_client(credentials) as client:
        batch = await client.messages.batches.create(requests=cast("Any", requests))
    logger.info(f"[Batch] Submitted {len(requests)} request(s) as {batch.id}")
    return batch.id


async def poll_anthropic_batch(
    batch_id: str,
    *,
    credentials: BatchCredentials | None = None,
    timeout_s: float = 3600.0,
    interval_s: float = 30.0,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """Poll a Message Batch to a terminal status. NETWORK CALL.

    Returns:
        ``"completed"`` once ``processing_status`` reaches ``"ended"``, so
        callers can treat it exactly like the OpenAI terminal status. A
        batch that ended with every request failed still returns
        "completed" — the per-request outcome is in the results.

    Raises:
        TimeoutError: Still in flight after ``timeout_s``. The batch keeps
            running server-side; collect it later by id.
    """
    elapsed = 0.0
    async with _anthropic_client(credentials) as client:
        while True:
            batch = await client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts
            done = counts.succeeded + counts.errored + counts.canceled + counts.expired
            total = done + counts.processing
            if on_progress is not None:
                on_progress(status, done, total)
            if status == "ended":
                return "completed"
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"batch {batch_id} still {status!r} after {timeout_s:.0f}s "
                    f"({done}/{total} done)"
                )
            await asyncio.sleep(interval_s)
            elapsed += interval_s


async def download_anthropic_batch_output(
    batch_id: str, output_path: Path, *, credentials: BatchCredentials | None = None
) -> Path:
    """Stream a Message Batch's results into the OpenAI output-file shape.

    Writing the same jsonl envelope both paths already read
    (``{"custom_id", "response": {"status_code", "body"}}`` on success,
    ``{"custom_id", "error"}`` otherwise) keeps collection provider-blind:
    ``read_openai_batch_output`` parses either, and only
    ``parse_batch_result`` needs to know whose body it is holding.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with _anthropic_client(credentials) as client:
        with output_path.open("w", encoding="utf-8") as f:
            async for entry in await client.messages.batches.results(batch_id):
                result = entry.result
                if result.type == "succeeded":
                    line = {
                        "custom_id": entry.custom_id,
                        "response": {
                            "status_code": 200,
                            "body": result.message.model_dump(mode="json"),
                        },
                    }
                elif result.type == "errored":
                    line = {
                        "custom_id": entry.custom_id,
                        "error": str(result.error.model_dump(mode="json")),
                    }
                else:  # canceled / expired
                    line = {"custom_id": entry.custom_id, "error": result.type}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return output_path

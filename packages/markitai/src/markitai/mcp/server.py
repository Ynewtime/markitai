"""markitai MCP server: document and URL conversion as agent tools.

A thin stdio MCP server over the public Python API (``markitai.aconvert``),
built on the official MCP SDK's high-level server (``MCPServer``, the API
formerly named FastMCP). One long-lived event loop serves all requests, which
is exactly the safe path the markitai API documents for embedding.

Design notes:

* Every conversion writes real files (caller-supplied ``output_dir`` or a
  fresh temp directory), so large results never need to travel through the
  model context: the inline markdown is truncated past a threshold and the
  caller reads the written file instead.
* ``batch_convert`` runs in-process as a background asyncio task with an
  in-memory job table — no persistence, jobs vanish with the server.
* LLM enhancement is always opt-in per call (``llm=False`` by default), so a
  misconfigured or absent model never breaks plain conversions.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from markitai import __version__, aconvert
from markitai.api import ConversionOutput

# Inline markdown budget per tool result. Past this, the result carries a
# preview and the path to the full file — a 500 KB document belongs on disk,
# not in the model context.
MAX_INLINE_CHARS = 40_000
PREVIEW_CHARS = 2_000

_LLM_MCP_HINT = (
    "For this MCP server: set the MODEL environment variable (and your "
    "provider API key) in the `env` block of the server's mcpServers entry, "
    "or configure llm.model_list in ~/.markitai/config.json."
)


class ConvertResult(TypedDict):
    """Result of a single conversion tool call."""

    source: str
    markdown: str
    truncated: bool
    markdown_file: str | None
    output_dir: str
    assets: list[str]
    screenshots: list[str]
    cost_usd: float
    skip_reason: str | None
    duration_s: float


class BatchStarted(TypedDict):
    """Acknowledgement returned by ``batch_convert``."""

    job_id: str
    status: str
    total: int
    output_dir: str


class JobStatus(TypedDict):
    """Progress snapshot returned by ``job_status``."""

    job_id: str
    status: str
    total: int
    done: int
    failed: int
    output_dir: str
    results: list[dict[str, Any]]


@dataclass
class _Job:
    """One in-memory batch job (results carry file paths, not content)."""

    id: str
    total: int
    output_dir: str
    status: str = "running"  # "running" | "completed"
    done: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    # Strong reference: asyncio only keeps weak references to running tasks.
    task: asyncio.Task[None] | None = None


_JOBS: dict[str, _Job] = {}
# Finished jobs stay queryable until this many newer ones have finished.
_MAX_FINISHED_JOBS = 100

server = MCPServer(
    name="markitai",
    version=__version__,
    instructions=(
        "Convert documents (PDF, DOCX, PPTX, XLSX, images, HTML, text) and "
        "web pages to clean Markdown. Use convert_document/convert_url for "
        "single sources, batch_convert + job_status for many. Results are "
        "written to disk; large markdown is truncated inline — read the "
        "returned markdown_file for the full text."
    ),
)


def _workdir(output_dir: str | None) -> Path:
    """Resolve the directory conversions write into.

    Without an explicit ``output_dir`` a fresh temp directory is created; it
    outlives the call so the caller can read the files it names.

    Raises:
        ToolError: When ``output_dir`` is a relative path — the server's
            working directory is not the client's, so relative paths would
            land somewhere the caller cannot predict.
    """
    if output_dir:
        resolved = Path(output_dir).expanduser()
        if not resolved.is_absolute():
            raise ToolError(
                f"output_dir must be absolute, got {output_dir!r} — the MCP "
                f"server's working directory is not the client's."
            )
        return resolved
    return Path(tempfile.mkdtemp(prefix="markitai-mcp-"))


def _to_result(out: ConversionOutput, workdir: Path) -> ConvertResult:
    """Map a ConversionOutput onto the wire shape, truncating big bodies."""
    text = out.llm_markdown if out.llm_markdown is not None else out.markdown
    truncated = len(text) > MAX_INLINE_CHARS
    markdown_file = out.llm_output_path or out.output_path
    return {
        "source": out.source,
        "markdown": text[:PREVIEW_CHARS] if truncated else text,
        "truncated": truncated,
        "markdown_file": str(markdown_file) if markdown_file else None,
        "output_dir": str(workdir),
        "assets": [str(p) for p in out.assets],
        "screenshots": [str(p) for p in out.screenshots],
        "cost_usd": out.usage.cost_usd,
        "skip_reason": out.skip_reason,
        "duration_s": round(out.duration, 2),
    }


async def _convert_source(
    source: str,
    workdir: Path,
    *,
    llm: bool,
    ocr: bool | None,
    screenshot: bool | None,
    alt: bool | None,
    desc: bool | None,
) -> ConvertResult:
    """Run one conversion into ``workdir`` and shape the tool result.

    Raises:
        ToolError: When LLM enhancement is requested but no model resolves;
            the markitai guidance is passed through with an MCP-specific hint.
    """
    try:
        out = await aconvert(
            source,
            output_dir=workdir,
            llm=llm,
            ocr=ocr,
            screenshot=screenshot,
            alt=alt,
            desc=desc,
        )
    except ValueError as e:
        # "LLM enabled but no model configured" — keep the guidance readable
        raise ToolError(f"{e} {_LLM_MCP_HINT}") from e
    return _to_result(out, workdir)


@server.tool()
async def convert_document(
    path: str,
    output_dir: str | None = None,
    llm: bool = False,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
) -> ConvertResult:
    """Convert one local document to Markdown.

    Handles PDF, Office (docx/pptx/xlsx and legacy formats), HTML, CSV/JSON,
    plain text/Markdown, and images. Returns the converted markdown plus the
    paths of everything written to disk.

    Args:
        path: Absolute path to one local file. Directories are rejected —
            pass files individually or use batch_convert.
        output_dir: Absolute directory to write outputs into (created if
            missing).
            Omit to use a fresh temporary directory; its path is returned.
        llm: Enable LLM enhancement (cleanup + frontmatter). Off by default;
            requires a configured model — without one the call fails with
            setup instructions (set MODEL in the server's env block).
        ocr: Enable OCR for scanned documents/images (needs markitai[ocr]).
            Omit to follow the server's markitai config.
        screenshot: Render page screenshots (PDF/Office). Omit to follow the
            server's markitai config.
        alt: Generate LLM alt text for embedded images (needs llm=true).
        desc: Generate LLM image descriptions (needs llm=true).

    Returns:
        Object with: markdown (full text, or a preview when truncated=true),
        truncated, markdown_file (path to the complete .md on disk — read it
        when truncated), output_dir, assets (extracted images), screenshots,
        cost_usd (LLM spend), skip_reason, duration_s.

    Failure modes: nonexistent path, a directory, an unsupported format, or
    llm=true without a configured model (the error explains the fix).
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        raise ToolError(
            f"path must be absolute, got {path!r} — the MCP server's working "
            f"directory is not the client's."
        )
    return await _convert_source(
        str(resolved),
        _workdir(output_dir),
        llm=llm,
        ocr=ocr,
        screenshot=screenshot,
        alt=alt,
        desc=desc,
    )


@server.tool()
async def convert_url(
    url: str,
    output_dir: str | None = None,
    llm: bool = False,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
) -> ConvertResult:
    """Fetch a web page and convert it to clean Markdown.

    Uses markitai's fetch cascade (static HTTP with readability extraction,
    optional browser rendering per the server's markitai config) and returns
    the main-content markdown.

    Args:
        url: The http(s) URL to fetch and convert.
        output_dir: Absolute directory to write outputs into (created if
            missing).
            Omit to use a fresh temporary directory; its path is returned.
        llm: Enable LLM enhancement (cleanup + frontmatter). Off by default;
            requires a configured model — without one the call fails with
            setup instructions (set MODEL in the server's env block).
        ocr: Enable OCR. Omit to follow the server's markitai config.
        screenshot: Capture a full-page screenshot (needs markitai[browser]).
            Omit to follow the server's markitai config.
        alt: Generate LLM alt text for page images (needs llm=true).
        desc: Generate LLM image descriptions (needs llm=true).

    Returns:
        Same shape as convert_document: markdown (or a preview when
        truncated=true), markdown_file with the complete text, output_dir,
        assets, screenshots, cost_usd, skip_reason, duration_s.

    Failure modes: unreachable URL, a page with no extractable content, or
    llm=true without a configured model (the error explains the fix).
    """
    if not url.startswith(("http://", "https://")):
        raise ToolError(
            f"url must start with http:// or https://, got {url!r} — for "
            f"local files use convert_document."
        )
    return await _convert_source(
        url,
        _workdir(output_dir),
        llm=llm,
        ocr=ocr,
        screenshot=screenshot,
        alt=alt,
        desc=desc,
    )


async def _run_batch(
    job: _Job,
    sources: list[str],
    *,
    llm: bool,
    ocr: bool | None,
    screenshot: bool | None,
    alt: bool | None,
    desc: bool | None,
) -> None:
    """Convert sources sequentially, recording one result entry per item."""
    workdir = Path(job.output_dir)
    try:
        await _convert_all(
            job,
            sources,
            workdir,
            llm=llm,
            ocr=ocr,
            screenshot=screenshot,
            alt=alt,
            desc=desc,
        )
    finally:
        # Cancellation must not leave the job "running" forever.
        job.status = "completed"
        job.task = None
        _forget_finished_jobs()


async def _convert_all(
    job: _Job,
    sources: list[str],
    workdir: Path,
    *,
    llm: bool,
    ocr: bool | None,
    screenshot: bool | None,
    alt: bool | None,
    desc: bool | None,
) -> None:
    for source in sources:
        try:
            result = await _convert_source(
                source,
                workdir,
                llm=llm,
                ocr=ocr,
                screenshot=screenshot,
                alt=alt,
                desc=desc,
            )
            job.results.append(
                {
                    "source": source,
                    "status": "ok",
                    "markdown_file": result["markdown_file"],
                    "cost_usd": result["cost_usd"],
                }
            )
        except Exception as e:
            job.results.append({"source": source, "status": "error", "error": str(e)})
        job.done += 1


def _forget_finished_jobs(keep: int = _MAX_FINISHED_JOBS) -> None:
    """Drop the oldest finished jobs so a long-lived server stays bounded."""
    finished = [job_id for job_id, job in _JOBS.items() if job.status == "completed"]
    for job_id in finished[:-keep] if keep else finished:
        del _JOBS[job_id]


@server.tool()
async def batch_convert(
    sources: list[str],
    output_dir: str | None = None,
    llm: bool = False,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
) -> BatchStarted:
    """Convert many files and/or URLs in the background; returns a job id.

    Items run sequentially inside the server process and results are written
    to one shared output directory. Poll job_status with the returned job_id
    for progress and the per-item result list. Jobs live in server memory
    only — a server restart forgets them (the written files remain).

    Args:
        sources: Local absolute file paths and/or http(s) URLs, converted in
            order. One failing item does not stop the rest.
        output_dir: Absolute shared directory for all outputs (created if
            missing).
            Omit to use a fresh temporary directory; its path is returned.
        llm: Enable LLM enhancement for every item. Off by default. With no
            configured model each item fails with the same setup guidance —
            configure a model (MODEL env var) before enabling.
        ocr: Enable OCR. Omit to follow the server's markitai config.
        screenshot: Render page/screen captures. Omit to follow the config.
        alt: Generate LLM alt text for images (needs llm=true).
        desc: Generate LLM image descriptions (needs llm=true).

    Returns:
        Object with job_id (pass to job_status), status ("running"), total,
        and output_dir.
    """
    if not sources:
        raise ToolError("sources must contain at least one path or URL.")
    workdir = _workdir(output_dir)
    job = _Job(id=uuid.uuid4().hex[:8], total=len(sources), output_dir=str(workdir))
    _JOBS[job.id] = job
    job.task = asyncio.get_running_loop().create_task(
        _run_batch(
            job,
            list(sources),
            llm=llm,
            ocr=ocr,
            screenshot=screenshot,
            alt=alt,
            desc=desc,
        )
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "total": job.total,
        "output_dir": job.output_dir,
    }


@server.tool()
async def job_status(job_id: str) -> JobStatus:
    """Report progress and results of a batch_convert job.

    Args:
        job_id: The id returned by batch_convert.

    Returns:
        Object with status ("running" until every item finished, then
        "completed"), total, done, failed, output_dir, and results — one
        entry per finished item: {source, status: "ok"|"error",
        markdown_file, cost_usd} or {source, status, error}. Poll until
        status is "completed", then read the markdown_file paths.

    Failure modes: unknown job_id (jobs are in-memory and lost when the
    server restarts).
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise ToolError(
            f"Unknown job id {job_id!r}. Jobs are held in server memory and "
            f"are lost when the server restarts."
        )
    failed = sum(1 for r in job.results if r.get("status") == "error")
    return {
        "job_id": job.id,
        "status": job.status,
        "total": job.total,
        "done": job.done,
        "failed": failed,
        "output_dir": job.output_dir,
        "results": list(job.results),
    }


def main() -> None:
    """Run the markitai MCP server on stdio."""
    server.run("stdio")


if __name__ == "__main__":
    main()

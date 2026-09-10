"""In-process tests for the markitai MCP tools.

The ``@server.tool()`` decorator returns the function unchanged, so every
tool is exercised directly as a plain async function: small fixtures go
through the real conversion pipeline (no LLM, no network), and the URL tool
is tested against a stubbed ``aconvert`` because fetching is network-bound.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from markitai.api import ConversionOutput
from markitai.mcp import server as server_module
from markitai.mcp.server import (
    MAX_INLINE_CHARS,
    PREVIEW_CHARS,
    batch_convert,
    convert_document,
    convert_url,
    job_status,
    server,
)


async def _wait_for_completion(job_id: str, timeout: float = 30.0) -> dict:
    """Poll job_status until the background task finishes."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        status = await job_status(job_id)
        if status["status"] == "completed":
            return dict(status)
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"job {job_id} did not complete within {timeout}s")
        await asyncio.sleep(0.05)


# =============================================================================
# Server construction
# =============================================================================


class TestServerSmoke:
    async def test_exposes_exactly_the_four_tools(self) -> None:
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert set(tools) == {
            "convert_document",
            "convert_url",
            "batch_convert",
            "job_status",
        }
        for tool in tools.values():
            assert tool.description, f"{tool.name} has no description"

    async def test_llm_failure_mode_is_documented_for_agents(self) -> None:
        """Agents must learn from the schema that llm needs a configured model."""
        tools = {tool.name: tool for tool in await server.list_tools()}
        for name in ("convert_document", "convert_url", "batch_convert"):
            assert "MODEL" in (tools[name].description or "")

    def test_console_entrypoint_is_wired(self) -> None:
        """The `markitai-mcp` script must point at a callable that exists."""
        import tomllib

        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "scripts"
        ]
        module_name, _, attribute = scripts["markitai-mcp"].partition(":")
        assert module_name == server_module.__name__
        assert callable(getattr(server_module, attribute))


# =============================================================================
# convert_document
# =============================================================================


class TestConvertDocument:
    async def test_converts_a_real_file(self, sample_md: Path) -> None:
        result = await convert_document(str(sample_md))
        assert "Hello" in result["markdown"]
        assert result["truncated"] is False
        assert result["cost_usd"] == 0.0
        assert result["skip_reason"] is None
        markdown_file = result["markdown_file"]
        assert markdown_file is not None
        assert Path(markdown_file).is_file()
        assert Path(markdown_file).is_relative_to(result["output_dir"])

    async def test_writes_into_the_given_output_dir(
        self, sample_md: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = await convert_document(str(sample_md), output_dir=str(out))
        assert result["output_dir"] == str(out)
        assert result["markdown_file"] is not None
        assert Path(result["markdown_file"]).is_relative_to(out)

    async def test_large_output_is_truncated_to_a_preview(self, tmp_path: Path) -> None:
        big = tmp_path / "big.txt"
        big.write_text("All work and no play. " * 4000, encoding="utf-8")

        result = await convert_document(str(big))

        assert result["truncated"] is True
        assert len(result["markdown"]) == PREVIEW_CHARS
        assert result["markdown_file"] is not None
        full_text = Path(result["markdown_file"]).read_text(encoding="utf-8")
        assert len(full_text) > MAX_INLINE_CHARS

    async def test_relative_path_is_rejected_with_guidance(self) -> None:
        with pytest.raises(ToolError, match="absolute"):
            await convert_document("relative/file.pdf")

    async def test_relative_output_dir_is_rejected(self, sample_md: Path) -> None:
        with pytest.raises(ToolError, match="output_dir must be absolute"):
            await convert_document(str(sample_md), output_dir="relative/out")

    async def test_directory_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError, match="directory"):
            await convert_document(str(tmp_path))

    async def test_llm_without_model_passes_guidance_through(
        self, sample_md: Path
    ) -> None:
        """The api's ValueError must surface as a readable MCP error."""
        with pytest.raises(ToolError) as excinfo:
            await convert_document(str(sample_md), llm=True)
        message = str(excinfo.value)
        assert "MODEL" in message  # the markitai guidance
        assert "mcpServers" in message  # the MCP-specific hint


# =============================================================================
# convert_url
# =============================================================================


class TestConvertUrl:
    async def test_non_http_url_is_rejected(self) -> None:
        with pytest.raises(ToolError, match="convert_document"):
            await convert_url("file:///etc/hosts")

    async def test_maps_llm_output_over_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The enhanced body and .llm.md path win when LLM ran (no network)."""
        base = tmp_path / "page.md"
        base.write_text("# Base\n", encoding="utf-8")
        enhanced = tmp_path / "page.llm.md"
        enhanced.write_text("# Enhanced\n", encoding="utf-8")

        async def fake_aconvert(source: str, **kwargs) -> ConversionOutput:
            return ConversionOutput(
                source=str(source),
                markdown="# Base",
                llm_markdown="# Enhanced",
                output_path=base,
                llm_output_path=enhanced,
            )

        monkeypatch.setattr(server_module, "aconvert", fake_aconvert)
        result = await convert_url("https://example.com/article")

        assert result["source"] == "https://example.com/article"
        assert result["markdown"] == "# Enhanced"
        assert result["markdown_file"] == str(enhanced)
        assert result["truncated"] is False


# =============================================================================
# batch_convert + job_status
# =============================================================================


class TestBatchConvert:
    async def test_empty_sources_is_rejected(self) -> None:
        with pytest.raises(ToolError, match="at least one"):
            await batch_convert([])

    async def test_runs_in_background_and_reports_per_item_results(
        self, sample_md: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.md"
        other.write_text("# Other\n", encoding="utf-8")
        missing = tmp_path / "missing.pdf"

        started = await batch_convert(
            [str(sample_md), str(other), str(missing)],
            output_dir=str(tmp_path / "batch-out"),
        )
        assert started["status"] == "running"
        assert started["total"] == 3

        status = await _wait_for_completion(started["job_id"])
        assert status["done"] == 3
        assert status["failed"] == 1
        by_source = {entry["source"]: entry for entry in status["results"]}
        assert by_source[str(sample_md)]["status"] == "ok"
        assert Path(by_source[str(sample_md)]["markdown_file"]).is_file()
        assert by_source[str(missing)]["status"] == "error"
        assert "not exist" in by_source[str(missing)]["error"]

    async def test_llm_without_model_fails_items_with_guidance(
        self, sample_md: Path
    ) -> None:
        started = await batch_convert([str(sample_md)], llm=True)
        status = await _wait_for_completion(started["job_id"])
        assert status["failed"] == 1
        assert "MODEL" in status["results"][0]["error"]

    async def test_unknown_job_id_is_a_tool_error(self) -> None:
        with pytest.raises(ToolError, match="Unknown job id"):
            await job_status("does-not-exist")

    async def test_forgotten_job_says_it_expired_not_unknown(
        self, sample_md: Path
    ) -> None:
        """A bounded server must not report an expired job as never-seen."""
        server_module._JOBS.clear()
        server_module._FORGOTTEN_JOBS.clear()
        started = await batch_convert([str(sample_md)])
        await _wait_for_completion(started["job_id"])

        server_module._forget_finished_jobs(keep=0)
        with pytest.raises(ToolError, match="has been forgotten"):
            await job_status(started["job_id"])


class TestJobBookkeeping:
    async def test_finished_jobs_are_bounded(self, sample_md: Path) -> None:
        """A long-lived server forgets old finished jobs, never running ones."""
        server_module._JOBS.clear()
        for _ in range(3):
            started = await batch_convert([str(sample_md)])
            await _wait_for_completion(started["job_id"])
        assert len(server_module._JOBS) == 3

        server_module._forget_finished_jobs(keep=1)
        assert len(server_module._JOBS) == 1
        assert await job_status(next(iter(server_module._JOBS)))

    async def test_cancelled_batch_reports_cancelled(self, sample_md: Path) -> None:
        """A cancelled job must not claim "completed" while done < total."""
        server_module._JOBS.clear()
        started = await batch_convert([str(sample_md)] * 5)
        job = server_module._JOBS[started["job_id"]]
        assert job.task is not None
        await asyncio.sleep(0)  # let the task start converting before cancelling it
        job.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job.task
        assert job.status == "cancelled"
        assert job.task is None
        status = await job_status(started["job_id"])
        assert status["status"] == "cancelled"


class TestConfigParity:
    """MCP defaults and options must line up with the CLI and the config file."""

    async def test_llm_is_tri_state_on_every_conversion_tool(self) -> None:
        """Omit llm to follow the config; only an explicit false forces it off."""
        tools = {tool.name: tool for tool in await server.list_tools()}
        for name in ("convert_document", "convert_url", "batch_convert"):
            variants = tools[name].input_schema["properties"]["llm"]["anyOf"]
            assert {"type": "boolean"} in variants, f"{name}.llm must accept a boolean"
            assert {"type": "null"} in variants, f"{name}.llm must accept null"

    async def test_profile_and_concurrency_are_exposed(self) -> None:
        tools = {tool.name: tool for tool in await server.list_tools()}
        for name in ("convert_document", "convert_url", "batch_convert"):
            assert "profile" in tools[name].input_schema["properties"]
        assert "concurrency" in tools["batch_convert"].input_schema["properties"]

    def test_batch_concurrency_prefers_the_explicit_argument(self) -> None:
        assert server_module._batch_concurrency(3) == 3
        assert server_module._batch_concurrency(0) == 1

    def test_batch_concurrency_falls_back_to_a_positive_default(self) -> None:
        assert (
            server_module._batch_concurrency(None)
            == server_module._DEFAULT_BATCH_CONCURRENCY
        )

    def test_default_concurrency_mirrors_the_package_constant(self) -> None:
        """The mcp layer cannot import constants, so pin the copy."""
        from markitai.constants import DEFAULT_BATCH_CONCURRENCY

        assert server_module._DEFAULT_BATCH_CONCURRENCY == DEFAULT_BATCH_CONCURRENCY

    async def test_batch_runs_bounded_parallelism(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A semaphore caps in-flight conversions; every item still finishes."""
        server_module._JOBS.clear()
        in_flight = 0
        peak = 0

        async def fake_convert(source: str, workdir: Path, **kwargs: object) -> dict:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return {
                "source": source,
                "markdown_file": None,
                "cost_usd": 0.0,
            }

        monkeypatch.setattr(server_module, "_convert_source", fake_convert)
        started = await batch_convert([f"item-{i}.md" for i in range(6)], concurrency=2)
        status = await _wait_for_completion(started["job_id"])

        assert status["done"] == 6
        assert status["failed"] == 0
        assert peak == 2, f"expected 2 in flight, saw {peak}"


class TestBatchResultOrder:
    """`results[i]` must answer `sources[i]` even though items finish out of order."""

    async def test_slow_first_item_still_leads_the_result_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server_module._JOBS.clear()

        async def fake_convert(source: str, workdir: Path, **kwargs: object) -> dict:
            # The first source is the slowest, so completion order is reversed.
            await asyncio.sleep(0.05 if source == "slow" else 0.0)
            return {"source": source, "markdown_file": None, "cost_usd": 0.0}

        monkeypatch.setattr(server_module, "_convert_source", fake_convert)
        started = await batch_convert(["slow", "fast"], concurrency=2)
        status = await _wait_for_completion(started["job_id"])

        assert [entry["source"] for entry in status["results"]] == ["slow", "fast"]


async def test_parallel_same_name_sources_preserve_every_result(tmp_path: Path) -> None:
    sources = []
    for label in ("alpha", "beta", "gamma", "delta"):
        source = tmp_path / label / "report.csv"
        source.parent.mkdir()
        source.write_text(f"name,value\n{label},source-{label}\n")
        sources.append(str(source))
    created = await batch_convert(
        sources, output_dir=str(tmp_path / "out"), llm=False, concurrency=4
    )
    status = await _wait_for_completion(created["job_id"])
    results = status["results"]
    paths = [Path(result["markdown_file"]) for result in results]
    assert len(set(paths)) == 4
    assert all(result["status"] == "ok" for result in results)
    for path, label in zip(paths, ("alpha", "beta", "gamma", "delta"), strict=True):
        assert f"source-{label}" in path.read_text()

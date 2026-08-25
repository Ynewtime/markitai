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

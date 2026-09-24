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

from markitai.api import ConversionError, ConversionOutput, FetchError
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
        with pytest.raises(ToolError, match="directory"):
            await convert_document(str(tmp_path))

    async def test_expected_failures_keep_their_reason(self, tmp_path: Path) -> None:
        """Regression: only ValueError became a ToolError; the SDK turned the
        rest into an opaque "Error executing tool convert_document"."""
        with pytest.raises(ToolError, match="does not exist") as excinfo:
            await convert_document(str(tmp_path / "missing.pdf"))
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)
        assert "MODEL" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "error",
        [
            ConversionError("No content extracted from scan.pdf"),
            FetchError("HTTP 404 fetching https://example.com/x"),
            IsADirectoryError("/tmp/dir is a directory"),
            ValueError("Refusing to write through symlink /tmp/out"),
        ],
    )
    async def test_every_expected_error_becomes_a_tool_error(
        self,
        error: Exception,
        sample_md: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def failing_aconvert(*args: object, **kwargs: object) -> None:
            raise error

        monkeypatch.setattr(server_module, "aconvert", failing_aconvert)
        with pytest.raises(ToolError) as excinfo:
            await convert_document(str(sample_md))
        assert str(excinfo.value) == str(error)
        # The LLM setup hint belongs to the no-model error only
        assert "mcpServers" not in str(excinfo.value)

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

    async def test_passes_conversion_warnings_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notice = "[URL] Screenshot not captured for https://example.com/a"

        async def fake_aconvert(source: str, **kwargs) -> ConversionOutput:
            return ConversionOutput(
                source=str(source), markdown="# Page", warnings=[notice]
            )

        monkeypatch.setattr(server_module, "aconvert", fake_aconvert)
        result = await convert_url("https://example.com/a")
        assert result["warnings"] == [notice]

        started = await batch_convert(
            ["https://example.com/a"], output_dir=str(tmp_path / "batch")
        )
        status = await _wait_for_completion(started["job_id"])
        assert status["results"][0]["warnings"] == [notice]


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

    async def test_relative_sources_are_rejected(
        self, sample_md: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: batch_convert resolved relative paths against the
        server's cwd while convert_document rejected them."""
        monkeypatch.chdir(sample_md.parent)
        with pytest.raises(ToolError, match="absolute") as excinfo:
            await batch_convert([str(sample_md), "sample.md", "https://example.com"])
        assert "'sample.md'" in str(excinfo.value)
        assert "example.com" not in str(excinfo.value)
        assert server_module._JOBS == {}

    async def test_urls_and_home_relative_paths_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_convert(source: str, workdir: Path, **kwargs: object) -> dict:
            return {
                "source": source,
                "markdown_file": None,
                "cost_usd": 0.0,
                "warnings": [],
            }

        monkeypatch.setattr(server_module, "_convert_source", fake_convert)
        started = await batch_convert(["https://example.com/a", "~/doc.md"])
        status = await _wait_for_completion(started["job_id"])
        assert status["failed"] == 0

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
                "warnings": [],
            }

        monkeypatch.setattr(server_module, "_convert_source", fake_convert)
        started = await batch_convert(
            [f"/abs/item-{i}.md" for i in range(6)], concurrency=2
        )
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
            await asyncio.sleep(0.05 if source == "/abs/slow" else 0.0)
            return {
                "source": source,
                "markdown_file": None,
                "cost_usd": 0.0,
                "warnings": [],
            }

        monkeypatch.setattr(server_module, "_convert_source", fake_convert)
        started = await batch_convert(["/abs/slow", "/abs/fast"], concurrency=2)
        status = await _wait_for_completion(started["job_id"])

        assert [entry["source"] for entry in status["results"]] == [
            "/abs/slow",
            "/abs/fast",
        ]


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


# =============================================================================
# Model resolution and environment loading
# =============================================================================


class TestModelResolutionParity:
    async def test_provider_key_is_auto_detected_like_the_cli(
        self,
        sample_md: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: MCP only fell back to MODEL; a provider key alone
        (the CLI's documented quick path) failed with "no models"."""
        from markitai.providers.detect import ProviderDetectionResult

        captured: dict[str, object] = {}

        async def fake_core(ctx, _max_size):  # type: ignore[no-untyped-def]
            from markitai.workflow.core import ConversionStepResult

            captured["models"] = [
                m.litellm_params.model for m in ctx.config.llm.model_list
            ]
            return ConversionStepResult(success=False, error="stop here")

        monkeypatch.setattr(
            "markitai.providers.detect.detect_all_providers",
            lambda: [
                ProviderDetectionResult(
                    provider="openai",
                    model="openai/gpt-5.6-luna",
                    authenticated=True,
                    source="env",
                )
            ],
        )
        monkeypatch.setattr("markitai.workflow.core.convert_document_core", fake_core)
        with pytest.raises(ToolError, match="stop here"):
            await convert_document(str(sample_md), output_dir=str(tmp_path), llm=True)
        assert captured["models"] == ["openai/gpt-5.6-luna"]


class TestDotenvLoading:
    def test_main_loads_cwd_then_home_env_without_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: markitai-mcp skipped the .env files markitai mcp (via
        the CLI's import-time load_dotenv) and markitai init rely on."""
        import os

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "MARKITAI_T_BOTH=cwd\nMARKITAI_T_CWD=cwd\nMARKITAI_T_PRESET=cwd\n"
        )
        home_env = Path.home() / ".markitai" / ".env"
        home_env.parent.mkdir(parents=True, exist_ok=True)
        home_env.write_text("MARKITAI_T_BOTH=home\nMARKITAI_T_HOME=home\n")
        monkeypatch.chdir(project)
        for key in ("MARKITAI_T_BOTH", "MARKITAI_T_CWD", "MARKITAI_T_HOME"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("MARKITAI_T_PRESET", "server-env")

        runs: list[str] = []
        monkeypatch.setattr(server, "run", lambda transport: runs.append(transport))
        try:
            server_module.main()
            assert runs == ["stdio"]
            assert os.environ["MARKITAI_T_BOTH"] == "cwd"
            assert os.environ["MARKITAI_T_CWD"] == "cwd"
            assert os.environ["MARKITAI_T_HOME"] == "home"
            # An mcpServers env block (already in the environment) wins
            assert os.environ["MARKITAI_T_PRESET"] == "server-env"
        finally:
            for key in ("MARKITAI_T_BOTH", "MARKITAI_T_CWD", "MARKITAI_T_HOME"):
                os.environ.pop(key, None)

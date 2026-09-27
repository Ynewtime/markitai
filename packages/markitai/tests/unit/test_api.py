"""Tests for the public programmatic API (markitai.api).

Covers the three hard guarantees of the API:

* library calls keep stdout clean (native parser noise is suppressed
  without any CLI initialization) — locked by a subprocess regression test;
* ``aconvert`` never blocks the event loop on CPU-bound converter work;
* the facade stays a thin layer over the existing config/workflow system
  (config precedence, MODEL env fallback, skip/error semantics).

LLM behavior is mocked at the ``create_llm_processor`` seam, following the
serve test convention — no external API is ever called.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pytest

import markitai
from markitai.api import (
    ConversionOutput,
    ConversionUsage,
    NoModelConfiguredError,
    _resolve_config,
    aconvert,
    convert,
)
from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.utils.errors import ConversionError

# =============================================================================
# Fakes
# =============================================================================


class FakeLLMProcessor:
    """Minimal stand-in for LLMProcessor.

    Deliberately a plain class, not a MagicMock: ``maybe_stabilize_markdown``
    probes ``getattr(processor, "_stabilize_paged_markdown", None)`` and a
    MagicMock would fabricate that attribute and corrupt the output.
    """

    def __init__(self) -> None:
        self.document_calls: list[str] = []

    async def process_document(
        self,
        markdown: str,
        source: str,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> tuple[str, str]:
        self.document_calls.append(source)
        return f"# Enhanced\n\n{markdown.strip()}", "title: Enhanced\nsource: test"

    async def clean_document_pure(self, markdown: str, source: str) -> str:
        self.document_calls.append(source)
        return f"# Pure\n\n{markdown.strip()}"

    def format_llm_output(self, markdown: str, frontmatter: str) -> str:
        return f"---\n{frontmatter}\n---\n\n{markdown}"

    def get_context_cost(self, context: str) -> float:
        return 0.0123

    def get_context_usage(self, context: str) -> dict[str, dict[str, Any]]:
        return {
            "openai/test": {
                "requests": 2,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0123,
            }
        }

    def clear_context_usage(self, context: str) -> None:
        pass


def _llm_config() -> MarkitaiConfig:
    """Config with LLM enabled and a dummy model entry (never called)."""
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
        )
    ]
    return cfg


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text("Hello from the API test.\n", encoding="utf-8")
    return path


# =============================================================================
# ConversionUsage
# =============================================================================


class TestConversionUsage:
    def test_from_usage_dict_aggregates_totals(self) -> None:
        usage = ConversionUsage.from_usage_dict(
            0.03,
            {
                "model-a": {"requests": 2, "input_tokens": 10, "output_tokens": 5},
                "model-b": {"requests": 1, "input_tokens": 7, "output_tokens": 3},
            },
        )
        assert usage.cost_usd == 0.03
        assert usage.requests == 3
        assert usage.input_tokens == 17
        assert usage.output_tokens == 8
        assert set(usage.by_model) == {"model-a", "model-b"}

    def test_defaults_are_empty(self) -> None:
        usage = ConversionUsage()
        assert usage.cost_usd == 0.0
        assert usage.by_model == {}


# =============================================================================
# Config resolution
# =============================================================================


class TestResolveConfig:
    def test_overrides_apply_without_mutating_caller(self) -> None:
        base = MarkitaiConfig()
        cfg = _resolve_config(
            base, llm=None, ocr=True, screenshot=None, alt=True, desc=None
        )
        assert cfg.ocr.enabled is True
        assert cfg.image.alt_enabled is True
        assert base.ocr.enabled is False
        assert base.image.alt_enabled is False

    def test_model_env_populates_empty_model_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "openai/gpt-test")
        cfg = _resolve_config(
            MarkitaiConfig(), llm=True, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert cfg.llm.model_list[0].litellm_params.model == "openai/gpt-test"

    def test_llm_without_model_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.setattr("markitai.providers.detect.detect_all_providers", list)
        with pytest.raises(NoModelConfiguredError, match="MODEL"):
            _resolve_config(
                MarkitaiConfig(),
                llm=True,
                ocr=None,
                screenshot=None,
                alt=None,
                desc=None,
            )

    def test_no_model_error_is_still_a_value_error(self) -> None:
        """The documented ``ValueError`` contract keeps holding."""
        assert issubclass(NoModelConfiguredError, ValueError)

    @staticmethod
    def _detected(*models: str) -> list[Any]:
        from markitai.providers.detect import ProviderDetectionResult

        return [
            ProviderDetectionResult(
                provider=model.split("/")[0],
                model=model,
                authenticated=True,
                source="env",
            )
            for model in models
        ]

    def test_provider_key_auto_detection_populates_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the API only fell back to MODEL, so a provider key in
        the environment — enough for the CLI — raised "no models" here."""
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.setattr(
            "markitai.providers.detect.detect_all_providers",
            lambda: self._detected("openai/gpt-5.6-luna"),
        )
        cfg = _resolve_config(
            MarkitaiConfig(), llm=True, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert [m.litellm_params.model for m in cfg.llm.model_list] == [
            "openai/gpt-5.6-luna"
        ]

    def test_real_env_key_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End to end through the real detector: an API key is enough."""
        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr(
            "markitai.providers.detect.shutil.which", lambda _name: None
        )
        monkeypatch.setattr(
            "markitai.providers.detect._check_chatgpt_auth", lambda: False
        )
        cfg = _resolve_config(
            MarkitaiConfig(), llm=True, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert [m.litellm_params.model for m in cfg.llm.model_list] == [
            "deepseek/deepseek-v4-flash"
        ]

    def test_model_env_beats_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL", "openai/gpt-test")

        def fail() -> list[Any]:
            raise AssertionError("detection must not run when MODEL is set")

        monkeypatch.setattr("markitai.providers.detect.detect_all_providers", fail)
        cfg = _resolve_config(
            MarkitaiConfig(), llm=True, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert cfg.llm.model_list[0].litellm_params.model == "openai/gpt-test"

    def test_configured_model_list_skips_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail() -> list[Any]:
            raise AssertionError("detection must not run with a model_list")

        monkeypatch.setattr("markitai.providers.detect.detect_all_providers", fail)
        cfg = _resolve_config(
            _llm_config(), llm=None, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert cfg.llm.model_list[0].litellm_params.model == "openai/test"

    def test_pooled_providers_warn_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Several detected providers share a pool: say so, like the CLI."""
        from loguru import logger

        monkeypatch.delenv("MODEL", raising=False)
        monkeypatch.setattr("markitai.api._POOLED_NOTICES_SHOWN", set())
        monkeypatch.setattr(
            "markitai.providers.detect.detect_all_providers",
            lambda: self._detected("anthropic/claude-haiku-4-5", "openai/gpt-5.6-luna"),
        )
        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            for _ in range(2):
                cfg = _resolve_config(
                    MarkitaiConfig(),
                    llm=True,
                    ocr=None,
                    screenshot=None,
                    alt=None,
                    desc=None,
                )
        finally:
            logger.remove(handler_id)

        assert len(cfg.llm.model_list) == 2
        [notice] = messages
        assert "requests are spread" in notice
        assert "anthropic/claude-haiku-4-5" in notice
        assert "openai/gpt-5.6-luna" in notice
        assert "MODEL=" in notice

    async def test_aconvert_detects_off_the_event_loop(
        self, sample_txt: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Detection probes CLI auth with asyncio.run, which raises inside a
        running loop — aconvert must resolve config in a worker thread."""
        monkeypatch.delenv("MODEL", raising=False)

        def detect() -> list[Any]:
            async def probe() -> bool:
                return True

            assert asyncio.run(probe())
            return self._detected("openai/gpt-5.6-luna")

        monkeypatch.setattr("markitai.providers.detect.detect_all_providers", detect)
        processor = FakeLLMProcessor()
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: processor,
        )
        out = await aconvert(sample_txt, config=MarkitaiConfig(), llm=True)
        assert out.llm_markdown is not None
        assert processor.document_calls == ["sample.txt"]

    def test_none_config_loads_the_cli_hierarchy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        class StubManager:
            def load(self) -> MarkitaiConfig:
                calls.append(True)
                return MarkitaiConfig()

        monkeypatch.setattr("markitai.config.ConfigManager", StubManager)
        _resolve_config(None, llm=None, ocr=None, screenshot=None, alt=None, desc=None)
        assert calls == [True]


# =============================================================================
# File conversion
# =============================================================================


class TestConvertFile:
    def test_in_memory_mode_returns_markdown_only(self, sample_txt: Path) -> None:
        out = convert(sample_txt, config=MarkitaiConfig())
        assert isinstance(out, ConversionOutput)
        assert "Hello from the API test." in out.markdown
        assert out.frontmatter.get("source") == "sample.txt"
        assert out.output_path is None
        assert out.llm_output_path is None
        assert out.assets == []
        assert out.llm_markdown is None
        assert out.usage.cost_usd == 0.0
        assert out.duration > 0

    def test_output_dir_mode_writes_base_markdown(
        self, sample_txt: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        out = convert(sample_txt, output_dir=out_dir, config=MarkitaiConfig())
        assert out.output_path == out_dir / "sample.txt.md"
        assert (out_dir / "sample.txt.md").exists()
        assert "Hello from the API test." in out.markdown
        # The written file carries frontmatter; the field strips it
        assert not out.markdown.startswith("---")
        assert out.frontmatter.get("source") == "sample.txt"

    async def test_llm_enhancement_with_mocked_processor(
        self, sample_txt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processor = FakeLLMProcessor()
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: processor,
        )
        out_dir = tmp_path / "out"
        out = await aconvert(sample_txt, output_dir=out_dir, config=_llm_config())

        assert out.llm_output_path == out_dir / "sample.txt.llm.md"
        assert (out_dir / "sample.txt.llm.md").exists()
        assert out.llm_markdown is not None
        assert out.llm_markdown.startswith("# Enhanced")
        assert out.frontmatter.get("title") == "Enhanced"
        # LLM mode without keep_base writes no base .md
        assert out.output_path is None
        assert out.markdown  # base conversion still returned in memory
        assert out.usage.cost_usd == pytest.approx(0.0123)
        assert out.usage.requests == 2
        assert out.usage.input_tokens == 100
        assert processor.document_calls == ["sample.txt"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert(tmp_path / "nope.txt", config=MarkitaiConfig())

    def test_directory_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError, match="CLI"):
            convert(tmp_path, config=MarkitaiConfig())

    def test_image_without_llm_or_ocr_raises_guidance(
        self, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ConversionError, match="image file"):
            convert(
                fixtures_dir / "sample.jpg",
                output_dir=tmp_path,
                config=MarkitaiConfig(),
            )

    def test_on_conflict_skip_returns_existing_output(
        self, sample_txt: Path, tmp_path: Path
    ) -> None:
        cfg = MarkitaiConfig()
        cfg.output.on_conflict = "skip"
        out_dir = tmp_path / "out"
        first = convert(sample_txt, output_dir=out_dir, config=cfg)
        assert first.skip_reason is None
        second = convert(sample_txt, output_dir=out_dir, config=cfg)
        assert second.skip_reason == "exists"
        assert second.output_path == first.output_path
        assert "Hello from the API test." in second.markdown


# =============================================================================
# URL conversion (fetch mocked — no network)
# =============================================================================


def _fake_fetch(content: str = "# Fetched Page\n\nBody text.\n"):
    from markitai.fetch_types import FetchResult

    async def fake_fetch_url(url: str, strategy: Any, config: Any, **kwargs: Any):
        return FetchResult(
            content=content,
            strategy_used="static",
            title="Fetched Page",
            url=url,
        )

    return fake_fetch_url


class TestConvertUrl:
    async def test_url_writes_base_markdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.fetch.fetch_url", _fake_fetch())
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        out = await aconvert(
            "https://example.com/page.html", output_dir=tmp_path, config=cfg
        )
        assert out.source == "https://example.com/page.html"
        assert out.output_path == tmp_path / "page.html.md"
        assert (tmp_path / "page.html.md").exists()
        assert "Body text." in out.markdown
        assert out.frontmatter.get("source") == "https://example.com/page.html"
        assert out.llm_markdown is None

    async def test_url_with_llm_writes_enhanced_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.fetch.fetch_url", _fake_fetch())
        processor = FakeLLMProcessor()
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: processor,
        )
        cfg = _llm_config()
        cfg.cache.enabled = False
        out = await aconvert(
            "https://example.com/page.html", output_dir=tmp_path, config=cfg
        )
        assert out.llm_output_path == tmp_path / "page.html.llm.md"
        assert (tmp_path / "page.html.llm.md").exists()
        assert out.llm_markdown is not None
        assert out.llm_markdown.startswith("# Enhanced")
        assert out.frontmatter.get("title") == "Enhanced"
        assert out.usage.cost_usd == pytest.approx(0.0123)
        assert processor.document_calls == ["https://example.com/page.html"]

    async def test_empty_fetch_content_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.fetch.fetch_url", _fake_fetch(content="  \n"))
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        with pytest.raises(ConversionError, match="No content extracted"):
            await aconvert("https://example.com/x", output_dir=tmp_path, config=cfg)

    async def test_url_llm_failure_writes_base_fallback_then_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same policy as the file pipeline: base .md fallback + error."""
        monkeypatch.setattr("markitai.fetch.fetch_url", _fake_fetch())

        class ExplodingProcessor(FakeLLMProcessor):
            async def process_document(self, *args: Any, **kwargs: Any):
                raise RuntimeError("model unavailable")

        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor",
            lambda *_args, **_kwargs: ExplodingProcessor(),
        )
        cfg = _llm_config()
        cfg.llm.on_failure = "fail"  # the raise under test
        cfg.cache.enabled = False
        with pytest.raises(ConversionError, match="LLM processing failed"):
            await aconvert(
                "https://example.com/page.html", output_dir=tmp_path, config=cfg
            )
        base = tmp_path / "page.html.md"
        assert base.exists(), "base .md fallback was not written"
        assert not (tmp_path / "page.html.llm.md").exists()


# =============================================================================
# Hard constraint: aconvert must not block the event loop
# =============================================================================


class TestConversionWarnings:
    """User notices come back on ConversionOutput.warnings, per call."""

    async def test_concurrent_calls_get_only_their_own_notices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.notices import user_notice
        from markitai.utils.executor import run_in_converter_thread

        both_running = asyncio.Barrier(2)

        async def fake_convert_file(
            path: Path, cfg: Any, workdir: Path, *, in_memory: bool
        ) -> ConversionOutput:
            await both_running.wait()
            # Raised from the converter pool, as the real converters do
            await run_in_converter_thread(
                user_notice, "[PDF] {}: 1 page(s) look scanned/garbled", path.name
            )
            await both_running.wait()
            return ConversionOutput(source=str(path), markdown="# x")

        monkeypatch.setattr("markitai.api._aconvert_file", fake_convert_file)
        first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        a, b = await asyncio.gather(
            aconvert(first, config=MarkitaiConfig()),
            aconvert(second, config=MarkitaiConfig()),
        )
        assert a.warnings == ["[PDF] a.pdf: 1 page(s) look scanned/garbled"]
        assert b.warnings == ["[PDF] b.pdf: 1 page(s) look scanned/garbled"]

    def test_clean_conversion_has_no_warnings(self, sample_txt: Path) -> None:
        assert convert(sample_txt, config=MarkitaiConfig()).warnings == []

    async def test_missing_url_screenshot_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.fetch.fetch_url", _fake_fetch())
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        out = await aconvert(
            "https://example.com/page.html",
            output_dir=tmp_path,
            config=cfg,
            screenshot=True,
        )
        assert out.warnings == [
            "[URL] Screenshot not captured for https://example.com/page.html; "
            "the page was converted without it"
        ]


class TestAconvertNonBlocking:
    async def test_cpu_bound_converter_leaves_loop_responsive(
        self, fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A converter stalling for 1s must not produce a 1s heartbeat gap.

        The stub converter simulates a CPU-bound C extension (pymupdf/ONNX)
        with a blocking sleep. ``convert_document_core`` must run it in the
        shared converter thread pool; if a refactor ever moves it inline,
        the heartbeat gap jumps to ~1s and this test fails.
        """
        from markitai.converter.base import ConvertResult

        class SleepyConverter:
            def convert(self, path: Path, output_dir: Path | None = None):
                time.sleep(1.0)  # blocking, like a C-extension parse
                return ConvertResult(markdown="# Slow\n\nbody\n")

        monkeypatch.setattr(
            "markitai.workflow.core.get_converter",
            lambda *_args, **_kwargs: SleepyConverter(),
        )

        beats: list[float] = []
        stop = asyncio.Event()

        async def heartbeat() -> None:
            while not stop.is_set():
                beats.append(time.monotonic())
                await asyncio.sleep(0.02)

        hb = asyncio.create_task(heartbeat())
        try:
            out = await aconvert(
                fixtures_dir / "sample.pdf",
                output_dir=tmp_path,
                config=MarkitaiConfig(),
            )
        finally:
            stop.set()
            await hb

        assert out.markdown.startswith("# Slow")
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        assert gaps, "heartbeat never ran concurrently with aconvert"
        assert max(gaps) < 0.6, (
            f"event loop stalled for {max(gaps):.2f}s during aconvert — "
            f"CPU-bound converter work is no longer running off-loop"
        )

    async def test_convert_refuses_to_run_inside_a_loop(self) -> None:
        with pytest.raises(RuntimeError, match="aconvert"):
            convert("anything.txt", config=MarkitaiConfig())


# =============================================================================
# Hard constraint: library calls keep stdout clean
# =============================================================================

_STDOUT_PROBE = """
import sys

import markitai
from markitai.utils.shutdown import finalize_process

out = markitai.convert(
    sys.argv[1], output_dir=sys.argv[2], config=markitai.MarkitaiConfig()
)
assert out.markdown, "conversion produced no markdown"
assert out.output_path is not None
# End the way a short-lived host should (see markitai.utils.shutdown): an
# assertion above still fails loudly, but interpreter shutdown — where a
# native dependency of markitdown can abort — is skipped.
finalize_process(0)
"""


class TestStdoutStaysClean:
    """Regression lock for the parser-noise suppression sink.

    Runs a real conversion through ``python -c "import markitai; ..."`` in a
    fresh interpreter — no CLI, no logging setup — and asserts nothing lands
    on stdout. The PDF fixture exercises the PyMuPDF path whose legacy
    ``fitz`` alias used to print its deprecation notice to stdout.
    """

    @pytest.mark.parametrize("fixture_name", ["sample.pdf", "sample.html"])
    def test_library_conversion_writes_nothing_to_stdout(
        self, fixture_name: str, fixtures_dir: Path, tmp_path: Path
    ) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _STDOUT_PROBE,
                str(fixtures_dir / fixture_name),
                str(tmp_path),
            ],
            capture_output=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
        assert proc.stdout == b"", (
            f"library conversion of {fixture_name} leaked to stdout: "
            f"{proc.stdout[:500]!r}"
        )


# =============================================================================
# Parser-noise suppression module
# =============================================================================


class TestSuppressParserNoise:
    def test_sets_ort_env_defaults_and_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.utils import suppress

        monkeypatch.setattr(suppress, "_APPLIED", False)
        monkeypatch.delenv("ORT_LOGGING_LEVEL", raising=False)
        monkeypatch.delenv("ORT_CPP_LOG_SEVERITY_LEVEL", raising=False)

        suppress.suppress_parser_noise()
        assert os.environ["ORT_LOGGING_LEVEL"] == "3"
        assert os.environ["ORT_CPP_LOG_SEVERITY_LEVEL"] == "3"

        # Second call is a no-op and never overrides user values
        os.environ["ORT_LOGGING_LEVEL"] = "0"
        suppress.suppress_parser_noise()
        assert os.environ["ORT_LOGGING_LEVEL"] == "0"

    def test_respects_preexisting_user_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.utils import suppress

        monkeypatch.setattr(suppress, "_APPLIED", False)
        monkeypatch.setenv("ORT_LOGGING_LEVEL", "1")
        suppress.suppress_parser_noise()
        assert os.environ["ORT_LOGGING_LEVEL"] == "1"


# =============================================================================
# Package surface
# =============================================================================


class TestPackageExports:
    def test_lazy_exports_resolve(self) -> None:
        assert markitai.convert is convert
        assert markitai.aconvert is aconvert
        assert markitai.ConversionOutput is ConversionOutput
        assert markitai.MarkitaiConfig is MarkitaiConfig

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = markitai.does_not_exist


@pytest.mark.parametrize("on_conflict", ["overwrite", "rename"])
async def test_disabled_llm_never_returns_previous_enhancement(
    tmp_path: Path, on_conflict: Literal["overwrite", "rename"]
) -> None:
    source = tmp_path / "document.txt"
    source.write_text("Current source content")
    output = tmp_path / "out"
    output.mkdir()
    stale = output / "document.txt.llm.md"
    stale.write_text("OLD LLM CONTENT")
    cfg = MarkitaiConfig()
    cfg.output.on_conflict = on_conflict
    result = await aconvert(
        source, output_dir=output, config=cfg, llm=False, profile="rag"
    )
    assert result.llm_markdown is None
    assert result.llm_output_path is None
    assert "Current source content" in result.markdown
    assert stale.read_text() == "OLD LLM CONTENT"


@pytest.mark.parametrize("profile,wikilinks", [("rag", False), ("obsidian", True)])
async def test_profile_assets_survive_api_batch_and_history(
    tmp_path: Path, profile, wikilinks
) -> None:
    from unittest.mock import MagicMock

    from markitai.cli.processors.batch_llm import _document_images, _prepare_pending
    from markitai.runs.history import record_cli_job
    from markitai.runs.types import Outcome

    source = Path(__file__).parents[1] / "fixtures" / "sample.pdf"
    cfg = MarkitaiConfig()
    cfg.output.wikilinks = wikilinks
    cfg.cache.enabled = False
    result = await aconvert(
        source, config=cfg, output_dir=tmp_path / "out", profile=profile, llm=False
    )
    assert result.output_path is not None
    assert result.assets
    assert all(path.is_file() for path in result.assets)
    assert set(_document_images(result.output_path, result.markdown)) == set(
        result.assets
    )
    processor = MagicMock()
    processor._engine.try_cached.return_value = None
    processor.documents._prepare_document_plan.return_value.chunk_calls = []
    processor.vision.prepare_image_plan.return_value.answer = None
    pending, cached, oversized = _prepare_pending(
        processor,
        result.output_path.parent,
        base_files=[result.output_path],
        analyze_images=True,
        cfg=cfg,
    )
    image_requests = [item for item, _plan in pending if item.kind == "image"]
    assert cached == 0 and not oversized
    assert len(image_requests) == len(result.assets)
    assert {
        result.output_path.parent / item.image
        for item in image_requests
        if item.image is not None
    } == set(result.assets)
    assert {
        call.args[0] for call in processor.vision.prepare_image_plan.call_args_list
    } == set(result.assets)
    history = record_cli_job(
        [
            Outcome(
                kind="file",
                source=source.name,
                status="completed",
                output_path=result.output_path,
            )
        ],
        options={"profile": profile},
        jobs_root=tmp_path / "jobs",
    )
    assert history is not None
    for asset in result.assets:
        assert (
            history / "out" / "assets" / asset.name
        ).read_bytes() == asset.read_bytes()

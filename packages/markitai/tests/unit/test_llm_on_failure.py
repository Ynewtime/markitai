"""``llm.on_failure``: what a failed LLM enhancement does to its item.

``"fallback"`` (default): the item succeeds on its unenhanced ``.md`` and a
warning names the failure. ``"fail"``: the item fails; the unenhanced
``.md`` is still written. Each entry point is checked under both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.fetch_types import FetchResult


def _cfg(on_failure: str | None = None) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.cache.enabled = False
    if on_failure is not None:
        cfg.llm.on_failure = on_failure  # type: ignore[assignment]
    return cfg


def _failing_processor() -> MagicMock:
    processor = MagicMock()
    processor.process_document = AsyncMock(side_effect=RuntimeError("invalid api key"))
    return processor


def test_the_default_keeps_the_unenhanced_output() -> None:
    assert MarkitaiConfig().llm.on_failure == "fallback"


def test_other_values_are_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MarkitaiConfig.model_validate({"llm": {"on_failure": "ignore"}})


class TestFilePipeline:
    async def _convert(self, tmp_path: Path, cfg: MarkitaiConfig) -> Any:
        from markitai.workflow.core import ConversionContext, convert_document_core
        from markitai.workflow.results import document_process_result

        source = tmp_path / "note.txt"
        source.write_text("# Note\n\nSome text.\n", encoding="utf-8")
        ctx = ConversionContext(
            input_path=source,
            output_dir=tmp_path / "out",
            config=cfg,
            shared_processor=_failing_processor(),
        )
        result = await convert_document_core(ctx, 100 * 1024 * 1024)
        return ctx, result, document_process_result(ctx, result)

    @pytest.mark.asyncio
    async def test_fallback_succeeds_on_the_base_md_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        ctx, result, processed = await self._convert(tmp_path, _cfg())

        base = tmp_path / "out" / "note.txt.md"
        assert result.success
        assert ctx.produced_file == base
        assert processed.success
        assert processed.output_path == str(base)
        assert processed.llm_enhanced is False
        assert base.is_file()
        assert not base.with_suffix(".llm.md").exists()
        (warning,) = processed.warnings
        assert "invalid api key" in warning
        assert "kept the unenhanced output" in warning

    @pytest.mark.asyncio
    async def test_fail_fails_the_item_and_still_writes_the_base(
        self, tmp_path: Path
    ) -> None:
        _ctx, result, processed = await self._convert(tmp_path, _cfg("fail"))

        assert not result.success
        assert not processed.success
        assert "invalid api key" in (processed.error or "")
        assert (tmp_path / "out" / "note.txt.md").is_file()


class TestSingleFileCli:
    async def _run(self, tmp_path: Path, cfg: MarkitaiConfig) -> list[Any]:
        from markitai.cli.processors.file import process_single_file
        from markitai.runs import Outcome

        source = tmp_path / "note.txt"
        source.write_text("# Note\n\nSome text.\n", encoding="utf-8")
        history: list[Outcome] = []
        with patch(
            "markitai.workflow.helpers.create_llm_processor",
            return_value=_failing_processor(),
        ):
            await process_single_file(
                source,
                tmp_path / "out",
                cfg,
                dry_run=False,
                quiet=True,
                history=history,
            )
        return history

    @pytest.mark.asyncio
    async def test_fallback_exits_0_with_the_base_md(self, tmp_path: Path) -> None:
        (outcome,) = await self._run(tmp_path, _cfg())

        assert outcome.status == "completed"
        assert outcome.output_path == tmp_path / "out" / "note.txt.md"
        assert any("invalid api key" in w for w in outcome.warnings)

    @pytest.mark.asyncio
    async def test_fail_exits_1(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            await self._run(tmp_path, _cfg("fail"))
        assert exc.value.code == 1
        assert (tmp_path / "out" / "note.txt.md").is_file()


class TestUrlCascade:
    URL = "https://example.com/page"

    async def _cascade(self, tmp_path: Path, cfg: MarkitaiConfig) -> Any:
        from markitai.workflow.url import convert_url_cascade

        async def _stage(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("provider timed out")

        return await convert_url_cascade(
            self.URL,
            cfg,
            tmp_path,
            processor=MagicMock(),
            fetch_result=FetchResult(
                content="# Page\n\nBody.\n", strategy_used="static", url=self.URL
            ),
            llm_stage=_stage,
        )

    @pytest.mark.asyncio
    async def test_fallback_returns_the_base_md_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        result = await self._cascade(tmp_path, _cfg())

        assert result.llm_error is not None
        assert result.llm_output_path is None
        assert result.output_path is not None and result.output_path.is_file()
        assert any("provider timed out" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_fail_raises_after_writing_the_base_md(self, tmp_path: Path) -> None:
        from markitai.utils.errors import ConversionError

        with pytest.raises(ConversionError, match="provider timed out"):
            await self._cascade(tmp_path, _cfg("fail"))
        assert list(tmp_path.glob("*.md"))


class TestBatchUrlCliBranch:
    """The CLI's own URL branches (here: --screenshot-only) in a batch."""

    URL = "https://example.com/docs"

    async def _run(self, tmp_path: Path, cfg: MarkitaiConfig) -> Any:
        from markitai.cli.processors.batch import create_url_processor
        from markitai.fetch_types import FetchStrategy

        shot = tmp_path / "out" / ".markitai" / "screenshots" / "docs.full.jpg"
        shot.parent.mkdir(parents=True)
        shot.write_bytes(b"jpeg")

        async def _fetch(url: str, *_a: Any, **_kw: Any) -> FetchResult:
            return FetchResult(
                content="# Docs\n\nText layer.\n",
                strategy_used="playwright",
                url=url,
                screenshot_path=shot,
            )

        async def _screenshot_llm(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("vision provider exploded")

        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True
        with (
            patch("markitai.fetch.fetch_url", _fetch),
            patch(
                "markitai.cli.processors.url.run_url_screenshot_only_llm",
                _screenshot_llm,
            ),
            patch(
                "markitai.workflow.helpers.create_llm_processor",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            process_url = create_url_processor(
                cfg=cfg,
                output_dir=tmp_path / "out",
                fetch_strategy=FetchStrategy.PLAYWRIGHT,
                explicit_fetch_strategy=True,
                renderer=MagicMock(),
            )
            return await process_url(self.URL)

    @pytest.mark.asyncio
    async def test_fallback_completes_on_the_base_md(self, tmp_path: Path) -> None:
        result, extra = await self._run(tmp_path, _cfg())

        assert result.success, result.error
        assert result.output_path.endswith("docs.md")
        assert Path(result.output_path).is_file()
        assert any("vision provider exploded" in w for w in extra["warnings"])

    @pytest.mark.asyncio
    async def test_fail_fails_the_url(self, tmp_path: Path) -> None:
        result, _extra = await self._run(tmp_path, _cfg("fail"))

        assert not result.success
        assert "vision provider exploded" in (result.error or "")


class TestLlmBatch:
    def _setup(self, tmp_path: Path, on_failure: str | None) -> tuple[Any, Path, list]:
        from markitai.runs.types import Outcome

        cfg = _cfg(on_failure)
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-4o-mini", api_key="sk-t"
                ),
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        base = out / "a.txt.md"
        base.write_text("# A\n\nbody", encoding="utf-8")
        items = [
            Outcome(kind="file", source="a.txt", status="completed", output_path=base)
        ]
        return cfg, out, items

    @pytest.mark.asyncio
    async def test_fallback_keeps_items_when_nothing_was_submitted(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement

        cfg, out, items = self._setup(tmp_path, None)
        with patch(
            "markitai.cli.processors.batch_llm.submit_openai_batch",
            new_callable=AsyncMock,
            side_effect=ConnectionError("connection refused"),
        ):
            handoff = await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert handoff is None
        (item,) = items
        assert item.status == "completed"
        assert item.output_path == out / "a.txt.md"
        assert any("submission failed" in w for w in item.warnings)

    @pytest.mark.asyncio
    async def test_fail_raises_and_fails_the_items(self, tmp_path: Path) -> None:
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.utils.errors import ConversionError

        cfg, out, items = self._setup(tmp_path, "fail")
        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                side_effect=ConnectionError("connection refused"),
            ),
            pytest.raises(ConversionError, match="connection refused"),
        ):
            await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert items[0].status == "failed"


def test_the_fallback_hint_is_given_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from markitai.workflow import llm_failure

    notices: list[str] = []
    monkeypatch.setattr(llm_failure, "_hinted", False)
    monkeypatch.setattr(
        llm_failure,
        "user_notice",
        lambda message, *args: notices.append(message.format(*args)),
    )

    first = llm_failure.llm_fallback_warning("a.pdf", "LLM processing failed: boom")
    llm_failure.llm_fallback_warning("b.pdf", "boom")

    assert first == "LLM enhancement failed (boom); kept the unenhanced output"
    assert sum("llm.on_failure" in n for n in notices) == 1
    assert [n for n in notices if n.startswith(("a.pdf", "b.pdf"))] == [
        "a.pdf: " + first,
        "b.pdf: " + first,
    ]

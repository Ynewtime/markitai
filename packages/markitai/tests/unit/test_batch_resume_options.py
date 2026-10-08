"""Regression tests: --resume state is tied to the options that shape output,
and --llm-batch refuses URL input instead of silently running at full price.

The resume state hash covered only llm/ocr/screenshot/alt/desc (plus the
directory scan settings), so ``--resume`` after switching ``--pure``,
``--keep-base``, ``--screenshot-only``, the output profile or ``-s`` skipped
every item done under the old settings and mixed both outputs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from markitai.batch import BatchProcessor
from markitai.cli import app
from markitai.cli.processors.batch import output_affecting_options
from markitai.config import MarkitaiConfig
from markitai.fetch import FetchStrategy
from tests.unit.test_batch_resume_state import (
    ONE,
    TWO,
    _Entry,
    _llm_cfg,
)
from tests.unit.test_batch_resume_state import (
    url_fakes as url_fakes,  # pytest fixture
)

BASE_OPTIONS = {"llm": True, "ocr": False, "screenshot": False}


def _hash(tmp_path: Path, options: dict) -> str:
    return BatchProcessor(
        MarkitaiConfig().batch,
        tmp_path / "out",
        input_path=tmp_path,
        task_options=options,
    ).task_hash


class TestOutputAffectingOptions:
    def test_default_config_adds_nothing(self) -> None:
        """A default run keeps the state file earlier versions wrote."""
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        assert output_affecting_options(cfg, FetchStrategy.AUTO, False) == {}

    @pytest.mark.parametrize(
        "change",
        ["pure", "keep_base", "screenshot_only", "profile", "fetch_strategy"],
    )
    def test_each_setting_changes_the_state_hash(
        self, tmp_path: Path, change: str
    ) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        strategy, explicit = FetchStrategy.AUTO, False
        if change == "pure":
            cfg.llm.pure = True
        elif change == "keep_base":
            cfg.llm.keep_base = True
        elif change == "screenshot_only":
            cfg.screenshot.screenshot_only = True
        elif change == "profile":
            cfg.output.profile = "rag"
        else:
            strategy, explicit = FetchStrategy.JINA, True

        changed = {**BASE_OPTIONS, **output_affecting_options(cfg, strategy, explicit)}

        assert _hash(tmp_path, changed) != _hash(tmp_path, BASE_OPTIONS)


@pytest.mark.asyncio
async def test_url_list_resume_with_other_output_settings_starts_over(
    tmp_path: Path,
    url_fakes: dict,  # noqa: F811
) -> None:
    from markitai.cli.processors.url import process_url_batch

    source = tmp_path / "list.urls"
    source.write_text(f"{ONE}\n{TWO}\n")

    async def run(cfg: MarkitaiConfig) -> None:
        await process_url_batch(
            [_Entry(ONE), _Entry(TWO)],
            tmp_path / "out",
            cfg,
            dry_run=False,
            verbose=False,
            quiet=True,
            resume=True,
            source_file=source,
        )

    await run(_llm_cfg())
    url_fakes["llm"].clear()

    keep_base = _llm_cfg()
    keep_base.llm.keep_base = True
    await run(keep_base)

    # Nothing was done under these settings yet: both URLs run again
    assert sorted(url_fakes["llm"]) == [ONE, TWO]


class TestLlmBatchRejectsUrlInput:
    def test_single_url(self, tmp_path: Path) -> None:
        with patch(
            "markitai.cli.processors.url.process_url", new=AsyncMock()
        ) as process_url:
            result = CliRunner().invoke(
                app,
                [
                    "https://example.com/page",
                    "-o",
                    str(tmp_path / "out"),
                    "--llm",
                    "--llm-batch",
                ],
            )

        assert result.exit_code == 1
        assert "directory batches only" in result.output
        process_url.assert_not_called()

    def test_url_list_file(self, tmp_path: Path) -> None:
        urls = tmp_path / "list.urls"
        urls.write_text("https://example.com/page\n")
        with patch(
            "markitai.cli.processors.url.process_url_batch", new=AsyncMock()
        ) as process_url_batch:
            result = CliRunner().invoke(
                app, [str(urls), "-o", str(tmp_path / "out"), "--llm", "--llm-batch"]
            )

        assert result.exit_code == 1
        assert "directory batches only" in result.output
        process_url_batch.assert_not_called()

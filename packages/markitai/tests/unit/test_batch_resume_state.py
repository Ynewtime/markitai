"""An interrupted batch has to leave something `--resume` can read.

State is written as a base ``.state.json`` plus a ``.state.jsonl`` sidecar
that later saves append their deltas to, and ``load_state`` gives up the
moment the base file is missing. The CLI batch path called ``init_state``
without writing that base — only ``BatchProcessor.process_batch``, the
library entry point, did — so a run interrupted before it finished left a
sidecar nothing could replay. ``--resume`` then started from zero and paid
for every LLM call a second time, while reporting nothing unusual.

The check is the timing, not the file: by the time the first document is
converted the base state must already be on disk, because that is the
earliest moment an interrupt can arrive.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from markitai.llm import LLMProcessor

import pytest

from markitai.config import MarkitaiConfig


def _state_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / ".markitai" / "states").glob("*.state.json"))


@pytest.mark.asyncio
async def test_base_state_exists_before_the_first_document_is_converted(
    tmp_path: Path,
) -> None:
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    for index in range(3):
        (input_dir / f"doc{index}.txt").write_text(f"content {index}")
    output_dir = tmp_path / "out"

    seen_at_first_file: list[list[Path]] = []
    from markitai.cli.processors import batch as batch_module

    original = batch_module.create_process_file

    def _recording_factory(  # noqa: ANN202
        cfg: MarkitaiConfig,
        input_dir: Path,
        output_dir: Path,
        shared_processor: LLMProcessor | None,
    ):
        process_file = original(cfg, input_dir, output_dir, shared_processor)

        async def _wrapped(path: Path):  # noqa: ANN202
            if not seen_at_first_file:
                seen_at_first_file.append(_state_files(output_dir))
            return await process_file(path)

        return _wrapped

    with patch.object(batch_module, "create_process_file", _recording_factory):
        await process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            cfg=MarkitaiConfig(),
            resume=False,
            dry_run=False,
            quiet=True,
        )

    assert seen_at_first_file, "no document was processed; the test proves nothing"
    assert seen_at_first_file[0], (
        "the batch started converting with no base state file on disk — an "
        "interrupt here leaves only a .jsonl sidecar, which load_state() "
        "cannot replay, so --resume restarts from zero"
    )

"""A document whose LLM call failed still gets the profile's layout.

`--profile rag` moves assets from `.markitai/assets/` to `assets/` and
rewrites the markdown to match. That happens in the pipeline's last step,
which every LLM-failure path returned before reaching — so one failed
document in a directory kept the default layout while its siblings got the
profile's, and anything consuming the directory by profile tripped on that
one file. Nothing reported it: both layouts are individually valid.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.config import MarkitaiConfig, OutputProfile
from markitai.constants import ASSETS_REL_PATH, VISIBLE_ASSETS_REL_PATH
from markitai.converter.base import ConvertResult
from markitai.workflow.core import ConversionContext, _write_base_md_fallback


def _context(tmp_path: Path, profile: OutputProfile | None) -> ConversionContext:
    config = MarkitaiConfig()
    config.output.profile = profile
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    source = tmp_path / "note.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    return ConversionContext(
        input_path=source,
        output_dir=output_dir,
        output_file=output_dir / "note.pdf.md",
        config=config,
        conversion_result=ConvertResult(
            markdown=f"# Note\n\n![figure]({ASSETS_REL_PATH}/note.pdf-0001.png)\n",
            images=[],
            metadata={"title": "Note"},
        ),
    )


def test_the_fallback_file_follows_the_active_profile(tmp_path: Path) -> None:
    ctx = _context(tmp_path, "rag")

    _write_base_md_fallback(ctx)

    assert ctx.output_file is not None
    written = ctx.output_file.read_text(encoding="utf-8")
    assert VISIBLE_ASSETS_REL_PATH in written, (
        "the fallback wrote the default asset layout while every document "
        f"that succeeded got the rag profile's:\n{written}"
    )
    assert ASSETS_REL_PATH not in written


def test_no_profile_leaves_the_fallback_untouched(tmp_path: Path) -> None:
    """The common case must not pay for the fix."""
    ctx = _context(tmp_path, None)

    _write_base_md_fallback(ctx)

    assert ctx.output_file is not None
    assert ASSETS_REL_PATH in ctx.output_file.read_text(encoding="utf-8")


def test_a_base_file_already_on_disk_is_profiled_too(tmp_path: Path) -> None:
    """`--keep-base` writes it earlier; the profile step is skipped all the same."""
    ctx = _context(tmp_path, "rag")
    assert ctx.output_file is not None
    ctx.output_file.write_text(
        f"# Note\n\n![figure]({ASSETS_REL_PATH}/note.pdf-0001.png)\n",
        encoding="utf-8",
    )

    _write_base_md_fallback(ctx)

    assert VISIBLE_ASSETS_REL_PATH in ctx.output_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("profile", ["rag", "obsidian", "okf"])
def test_every_profile_reaches_the_fallback(
    tmp_path: Path, profile: OutputProfile
) -> None:
    """The fix must not be specific to the profile it was found with."""
    ctx = _context(tmp_path / profile, profile)

    with patch("markitai.workflow.core.apply_output_profile") as applied:
        _write_base_md_fallback(ctx)

    applied.assert_called_once_with(ctx)

"""`--llm-batch` covers image analysis and page screenshots too.

The Batch API bills at half price, and `--alt/--desc` were refused with it —
a user who wanted image descriptions had to give up the discount on the
whole run, or give up the descriptions.

They can share one job because they do not depend on each other: the live
path runs document enhancement and image analysis as parallel tasks and
substitutes alt text into the written `.llm.md` afterwards, so the document
call never sees an image's answer. What the offline path genuinely cannot
reproduce is the language retry — it has to read an answer before deciding
whether to ask again — so that one call is made live at collect time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from markitai.cli.processors.batch_llm import (
    _apply_image_answers,
    _collect_image,
    _document_images,
    _prepare_pending,
)
from markitai.constants import ASSETS_REL_PATH
from markitai.llm.batch_api import BatchDocItem, BatchRunState
from markitai.llm.types import ImageAnalysis


def _write_doc_with_image(out: Path, name: str, image: str) -> Path:
    assets = out / ASSETS_REL_PATH
    assets.mkdir(parents=True, exist_ok=True)
    (assets / image).write_bytes(b"\x89PNG\r\n\x1a\n")
    base = out / f"{name}.md"
    base.write_text(f"# {name}\n\n![]({ASSETS_REL_PATH}/{image})\n", encoding="utf-8")
    return base


def _processor(plan_answer: ImageAnalysis | None = None) -> MagicMock:
    processor = MagicMock()
    processor._engine.try_cached.return_value = None
    processor.documents._prepare_document_plan.side_effect = lambda *_args, **_kwargs: (
        MagicMock()
    )
    plan = MagicMock()
    plan.answer = plan_answer
    plan.language = ""
    plan.document_context = ""
    processor.vision.prepare_image_plan.return_value = plan
    return processor


class TestDocumentImages:
    def test_finds_the_images_the_markdown_refers_to(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        base = _write_doc_with_image(out, "a", "a-0001.png")

        found = _document_images(base, base.read_text(encoding="utf-8"))

        assert [p.name for p in found] == ["a-0001.png"]

    def test_a_reference_with_no_file_is_dropped(self, tmp_path: Path) -> None:
        """A stale ref must not become a request for a file that is not there."""
        out = tmp_path / "out"
        out.mkdir()
        base = out / "a.md"
        base.write_text(f"![]({ASSETS_REL_PATH}/gone.png)", encoding="utf-8")

        assert _document_images(base, base.read_text(encoding="utf-8")) == []


class TestPreparePendingWithImages:
    def test_images_are_requested_alongside_their_document(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        _write_doc_with_image(out, "a", "a-0001.png")

        pending, cached, _oversized = _prepare_pending(
            _processor(), out, analyze_images=True
        )

        kinds = [item.kind for item, _ in pending]
        assert kinds == ["doc", "image"], kinds
        assert cached == 0
        image_item = pending[1][0]
        assert image_item.base_md == "a.md", "an image must name its document"
        assert image_item.custom_id.startswith("img_")
        assert Path(image_item.image).name == "a-0001.png"

    def test_no_image_requests_when_analysis_is_off(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        _write_doc_with_image(out, "a", "a-0001.png")

        pending, _cached, _oversized = _prepare_pending(
            _processor(), out, analyze_images=False
        )

        assert [item.kind for item, _ in pending] == ["doc"]

    def test_an_already_answered_image_is_not_sent(self, tmp_path: Path) -> None:
        """A cache hit or an unsupported format is the answer; asking again
        would pay for something already known."""
        out = tmp_path / "out"
        out.mkdir()
        _write_doc_with_image(out, "a", "a-0001.png")
        cached_answer = ImageAnalysis(caption="known", description="from cache")

        pending, cached, _oversized = _prepare_pending(
            _processor(plan_answer=cached_answer), out, analyze_images=True
        )

        assert [item.kind for item, _ in pending] == ["doc"]
        assert cached == 1

    def test_custom_ids_stay_unique_across_kinds(self, tmp_path: Path) -> None:
        """Both APIs key their answers by custom_id; a collision loses one."""
        out = tmp_path / "out"
        out.mkdir()
        _write_doc_with_image(out, "a", "a-0001.png")
        _write_doc_with_image(out, "b", "b-0001.png")

        pending, _cached, _oversized = _prepare_pending(
            _processor(), out, analyze_images=True
        )

        ids = [item.custom_id for item, _ in pending]
        assert len(set(ids)) == len(ids), ids


class TestCollectImage:
    def _item(self, image: str = f"{ASSETS_REL_PATH}/a-0001.png") -> BatchDocItem:
        return BatchDocItem(
            custom_id="img_1_a-0001",
            source="a.pdf",
            input_md="",
            base_md="a.md",
            kind="image",
            image=image,
        )

    def _state(self) -> BatchRunState:
        return BatchRunState(
            batch_id="b1",
            model="gpt-5.6-luna",
            mode="tools",
            provider="openai",
            created_at="2026-08-28T00:00:00",
        )

    @pytest.mark.asyncio
    async def test_a_failed_image_degrades_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        """The live path treats image analysis as non-critical — the document
        keeps its alt-less markdown. A batch must not be stricter."""
        out = tmp_path / "out"
        (out / ASSETS_REL_PATH).mkdir(parents=True)
        (out / ASSETS_REL_PATH / "a-0001.png").write_bytes(b"x")

        line = MagicMock()
        line.error = "the model refused this image"
        line.body = None

        entry = await _collect_image(
            _processor(), self._state(), out, self._item(), line, MagicMock()
        )

        assert entry["alt"] == "Image"
        assert entry["desc"] == "Image analysis failed"

    @pytest.mark.asyncio
    async def test_a_missing_output_line_degrades_too(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / ASSETS_REL_PATH).mkdir(parents=True)
        (out / ASSETS_REL_PATH / "a-0001.png").write_bytes(b"x")

        entry = await _collect_image(
            _processor(), self._state(), out, self._item(), None, MagicMock()
        )

        assert entry["desc"] == "Image analysis failed"

    @pytest.mark.asyncio
    async def test_an_already_answered_image_costs_nothing(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        (out / ASSETS_REL_PATH).mkdir(parents=True)
        (out / ASSETS_REL_PATH / "a-0001.png").write_bytes(b"x")
        processor = _processor(
            plan_answer=ImageAnalysis(caption="cached", description="known")
        )

        entry = await _collect_image(
            processor, self._state(), out, self._item(), None, MagicMock()
        )

        assert entry["alt"] == "cached"
        processor._track_usage.assert_not_called()


class TestApplyImageAnswers:
    def _config(self, *, alt: bool, desc: bool) -> Any:
        cfg = MagicMock()
        cfg.image.alt_enabled = alt
        cfg.image.desc_enabled = desc
        cfg.output.profile = None
        return cfg

    def _entry(self, out: Path, alt: str) -> dict[str, Any]:
        return {
            "asset": str((out / ASSETS_REL_PATH / "a-0001.png").resolve()),
            "alt": alt,
            "desc": "a chart",
            "text": "",
            "llm_usage": {},
            "created": "2026-08-28T00:00:00",
        }

    def test_alt_text_lands_in_the_enhanced_markdown(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _write_doc_with_image(out, "a", "a-0001.png")
        (out / "a.llm.md").write_text(
            f"# a\n\n![]({ASSETS_REL_PATH}/a-0001.png)\n", encoding="utf-8"
        )

        _apply_image_answers(
            self._config(alt=True, desc=False),
            out,
            {"a.md": [self._entry(out, "a revenue chart")]},
        )

        assert "![a revenue chart]" in (out / "a.llm.md").read_text(encoding="utf-8")

    def test_descriptions_land_in_images_json(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        _write_doc_with_image(out, "a", "a-0001.png")

        _apply_image_answers(
            self._config(alt=False, desc=True),
            out,
            {"a.md": [self._entry(out, "a chart")]},
        )

        written = out / ASSETS_REL_PATH / "images.json"
        assert written.is_file(), "no images.json was written"
        assert "a chart" in json.dumps(json.loads(written.read_text(encoding="utf-8")))

    def test_nothing_is_written_when_no_images_were_analyzed(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()

        _apply_image_answers(self._config(alt=True, desc=True), out, {})

        assert list(out.iterdir()) == []


class TestPreparePendingWithPages:
    """`--screenshot` documents batch as one vision request each."""

    def _write_pages(self, out: Path, source: str, count: int) -> None:
        from markitai.constants import SCREENSHOTS_REL_PATH

        shots = out / SCREENSHOTS_REL_PATH
        shots.mkdir(parents=True, exist_ok=True)
        for n in range(1, count + 1):
            (shots / f"{source}.page{n:04d}.jpg").write_bytes(b"\xff\xd8\xff")

    def test_pages_are_found_by_their_exact_name(self, tmp_path: Path) -> None:
        """Screenshots are named by markitai from the same string that names
        the base .md, so the mapping is exact — not a guess at what an
        extractor did to the filename."""
        from markitai.cli.processors.batch_llm import _document_pages

        out = tmp_path / "out"
        out.mkdir()
        (out / "a.pdf.md").write_text("# a", encoding="utf-8")
        self._write_pages(out, "a.pdf", 3)
        # A neighbour's pages must not be picked up
        self._write_pages(out, "b.pdf", 2)

        found = _document_pages(out / "a.pdf.md")

        assert [p.name for p in found] == [
            "a.pdf.page0001.jpg",
            "a.pdf.page0002.jpg",
            "a.pdf.page0003.jpg",
        ]

    def test_a_screenshot_document_becomes_one_vision_request(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.pdf.md").write_text("# a\n\nbody", encoding="utf-8")
        self._write_pages(out, "a.pdf", 3)

        processor = _processor()
        pending, _cached, oversized = _prepare_pending(
            processor, out, analyze_pages=True
        )

        assert [item.kind for item, _ in pending] == ["vision"]
        assert oversized == []
        processor.documents.prepare_vision_plan.assert_called_once()

    def test_without_screenshots_it_stays_a_text_request(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.pdf.md").write_text("# a\n\nbody", encoding="utf-8")
        self._write_pages(out, "a.pdf", 3)

        pending, _cached, _oversized = _prepare_pending(
            _processor(), out, analyze_pages=False
        )

        assert [item.kind for item, _ in pending] == ["doc"]

    def test_a_long_document_is_kept_out_of_the_batch(self, tmp_path: Path) -> None:
        """One request carrying every page of a 40-page PDF is an enormous
        upload, and the live path's answer — ordered rounds — costs a
        separate 24-hour wait per round in a batch."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.pdf.md").write_text("# a\n\nbody", encoding="utf-8")
        self._write_pages(out, "a.pdf", 40)

        pending, _cached, oversized = _prepare_pending(
            _processor(), out, analyze_pages=True, max_pages=10
        )

        assert pending == []
        assert len(oversized) == 1
        base_md, source, pages = oversized[0]
        assert base_md.name == "a.pdf.md"
        assert source == "a.pdf"
        assert len(pages) == 40

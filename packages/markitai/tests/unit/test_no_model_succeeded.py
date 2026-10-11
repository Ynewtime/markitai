"""No model succeeded: only the base Markdown is kept (contract v1.6, 6.6).

An answer counts only if the request returned 2xx, was not cut short or
filtered (``finish_reason`` ``length`` / ``content_filter``), is not a
refusal, and no page came back empty; a fallback that puts the original
text back is no answer either. When no request produced an accepted answer
the item keeps only ``<name>.md``, reports ``llm_enhanced: false`` and no
``models``, and its warnings end with the fixed sentence the workbench
shows as one notice. A failed rerun keeps the previous result and records
``rerun_failure``, which survives a restart.

The provider is a local OpenAI-compatible server: nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.llm.vision import declined_caption
from markitai.workflow.llm_failure import NO_MODEL_SUCCEEDED
from tests.local_http import LocalHTTPServer

REFUSAL = "I'm sorry, but I can't help with describing this image."


class _Provider(BaseHTTPRequestHandler):
    """``/refuse/v1``: every answer declines; ``/filter/v1``: every answer is
    stopped by the content filter."""

    requests = 0

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        type(self).requests += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = self.path.strip("/").split("/")[0]
        content, finish = (
            (REFUSAL, "stop") if mode == "refuse" else ("Partial an", "content_filter")
        )
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 12,
                    "total_tokens": 132,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        pass


@pytest.fixture
def provider() -> Iterator[str]:
    """Base URL of the provider; append ``/<mode>/v1``."""
    _Provider.requests = 0
    server = LocalHTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _config(api_base: str, tmp_path: Path) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(
                model="openai/gpt-4o-mini", api_key="sk-test-0000", api_base=api_base
            ),
        )
    ]
    cfg.llm.router_settings.num_retries = 0
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return cfg


def _png(path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 200), "white")
    ImageDraw.Draw(image).rectangle((40, 60, 120, 180), fill=(37, 99, 235))
    image.save(path)
    return path


@pytest.mark.parametrize(
    "caption",
    [
        "I am sorry, I cannot help with that.",
        "I’m sorry, but I can't describe this image.",
        "I cannot see images.",
        "As an AI, I cannot see pictures",
        "抱歉，我无法查看这张图片",
        "申し訳ありませんが、この画像は説明できません",
    ],
)
def test_a_caption_that_declines_the_request_is_no_alt_text(caption: str) -> None:
    assert declined_caption(caption)


@pytest.mark.parametrize(
    "caption",
    ["Sorry sign on a shop window", "A bar chart of revenue", "Icannot logo"],
)
def test_a_caption_about_an_apology_is_still_alt_text(caption: str) -> None:
    assert not declined_caption(caption)


async def _convert(source: Path, cfg: MarkitaiConfig, out_dir: Path) -> tuple[Any, Any]:
    from markitai.workflow.core import ConversionContext, convert_document_core
    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.results import document_process_result

    ctx = ConversionContext(
        input_path=source,
        output_dir=out_dir,
        config=cfg,
        shared_processor=create_llm_processor(cfg),
    )
    result = await convert_document_core(ctx, 100 * 1024 * 1024)
    return result, document_process_result(ctx, result)


def _assert_base_only(out_dir: Path, name: str, processed: Any) -> None:
    assert processed.success, processed.error
    assert processed.llm_enhanced is False
    assert processed.output_path == str(out_dir / f"{name}.md")
    assert (out_dir / f"{name}.md").is_file()
    assert not list(out_dir.rglob("*.llm.md"))
    assert processed.warnings[-1] == NO_MODEL_SUCCEEDED
    assert any("LLM enhancement failed" in w for w in processed.warnings)


@pytest.mark.parametrize("mode", ["refuse", "filter"])
async def test_an_image_whose_every_answer_failed_keeps_only_its_base(
    tmp_path: Path, provider: str, mode: str
) -> None:
    cfg = _config(f"{provider}/{mode}/v1", tmp_path)
    cfg.image.alt_enabled = True
    cfg.image.desc_enabled = True
    out = tmp_path / "out"
    _, processed = await _convert(_png(tmp_path / "chart.png"), cfg, out)
    _assert_base_only(out, "chart.png", processed)
    assert REFUSAL not in (out / "chart.png.md").read_text(encoding="utf-8")
    assert not (out / "images.json").exists()


async def test_a_refusal_is_never_cached(tmp_path: Path, provider: str) -> None:
    cfg = _config(f"{provider}/refuse/v1", tmp_path)
    cfg.cache.enabled = True
    cfg.image.alt_enabled = True
    image = _png(tmp_path / "chart.png")
    await _convert(image, cfg, tmp_path / "first")
    asked = _Provider.requests
    _, processed = await _convert(image, cfg, tmp_path / "second")
    # Asked again, not served the refusal from the cache
    assert _Provider.requests > asked
    _assert_base_only(tmp_path / "second", "chart.png", processed)


@pytest.mark.parametrize("pure", [False, True])
async def test_a_filtered_document_answer_keeps_only_the_base(
    tmp_path: Path, provider: str, pure: bool
) -> None:
    cfg = _config(f"{provider}/filter/v1", tmp_path)
    cfg.llm.pure = pure
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\n" + "Plain text. " * 60, encoding="utf-8")
    out = tmp_path / "out"
    _, processed = await _convert(source, cfg, out)
    _assert_base_only(out, "notes.md", processed)


def _paged_pdf(path: Path, pages: int = 2) -> Path:
    import pymupdf

    pdf = pymupdf.open()
    for number in range(1, pages + 1):
        page = pdf.new_page(width=400, height=300)
        page.insert_text((40, 60), f"Quarterly report, part {number}", fontsize=16)
    pdf.save(path)
    return path


async def test_screenshot_pages_read_as_empty_are_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from markitai.llm.processor import LLMProcessor

    async def blank(self: Any, image_path: Path, **_kwargs: Any) -> tuple[str, str]:
        return "", "title: blank"

    monkeypatch.setattr(LLMProcessor, "extract_from_screenshot", blank)
    cfg = _config(f"{provider}/refuse/v1", tmp_path)
    cfg.screenshot.enabled = True
    cfg.screenshot.screenshot_only = True
    out = tmp_path / "out"
    _, processed = await _convert(_paged_pdf(tmp_path / "report.pdf"), cfg, out)
    _assert_base_only(out, "report.pdf", processed)


async def test_some_empty_screenshot_pages_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from markitai.llm.processor import LLMProcessor

    async def second_blank(
        self: Any, image_path: Path, **_kwargs: Any
    ) -> tuple[str, str]:
        if "0002" in image_path.name:
            return "", ""
        return "# Read from page one", "title: report"

    monkeypatch.setattr(LLMProcessor, "extract_from_screenshot", second_blank)
    cfg = _config(f"{provider}/refuse/v1", tmp_path)
    cfg.screenshot.enabled = True
    cfg.screenshot.screenshot_only = True
    cfg.llm.keep_base = True  # as serve runs: the base is no baseline here
    out = tmp_path / "out"
    _, processed = await _convert(_paged_pdf(tmp_path / "report.pdf"), cfg, out)
    assert processed.llm_enhanced is True
    enhanced = (out / "report.pdf.llm.md").read_text(encoding="utf-8")
    assert "Read from page one" in enhanced
    assert any("no text for page 2" in w for w in processed.warnings), (
        processed.warnings
    )
    assert NO_MODEL_SUCCEEDED not in processed.warnings


def test_an_answer_the_original_text_replaced_whole_is_removed(
    tmp_path: Path,
) -> None:
    from markitai.converter.base import ConvertResult
    from markitai.workflow.core import (
        ConversionContext,
        LLMAnswerNotKeptError,
        stabilize_written_llm_output,
    )

    class Restoring:
        def stabilize_paged_answer(
            self, original: str, cleaned: str, source: str, *, vision: bool = False
        ) -> tuple[str, bool]:
            return original, False

    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    ctx = ConversionContext(
        input_path=tmp_path / "deck.pptx", output_dir=tmp_path, config=cfg
    )
    ctx.output_file = tmp_path / "deck.pptx.md"
    ctx.conversion_result = ConvertResult(markdown="<!-- Slide number: 1 -->\nBase")
    llm_output = tmp_path / "deck.pptx.llm.md"
    llm_output.write_text("<!-- Page 1 -->\nAnswer", encoding="utf-8")
    with pytest.raises(LLMAnswerNotKeptError):
        stabilize_written_llm_output(ctx, Restoring())
    assert not llm_output.exists()
    assert ctx.llm_output_file is None

    # A screenshot-only reading is never measured against the text layer
    llm_output.write_text("<!-- Page 1 -->\nAnswer", encoding="utf-8")
    cfg.screenshot.screenshot_only = True
    assert stabilize_written_llm_output(ctx, Restoring()) is False
    assert llm_output.read_text(encoding="utf-8") == "<!-- Page 1 -->\nAnswer"


# ---------------------------------------------------------------------------
# serve: models, the fixed warning, and rerun_failure
# ---------------------------------------------------------------------------


async def test_a_serve_item_says_no_model_succeeded_and_names_no_models(
    tmp_path: Path, provider: str
) -> None:
    pytest.importorskip("fastapi")
    from tests.unit.serve.test_serve_api import (
        _make_app,
        _multipart,
        _serve_client,
        _wait_job_done,
    )

    cfg = _config(f"{provider}/refuse/v1", tmp_path)
    png = _png(tmp_path / "chart.png")
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        created = await client.post(
            "/api/jobs",
            files=_multipart(
                files=[("chart.png", png.read_bytes())],
                options={"llm": True, "alt": True},
            ),
        )
        job = await asyncio.wait_for(
            _wait_job_done(client, created.json()["job_id"]), timeout=90
        )
    (item,) = job["items"]
    assert item["status"] == "done"
    assert item["llm_enhanced"] is False
    assert item["output"] == "chart.png.md"
    assert item["models"] == []
    # Paid for: the refusals still cost
    assert item["pricing"] is not None
    assert item["warnings"][-1] == NO_MODEL_SUCCEEDED


async def test_a_failed_enhance_keeps_the_result_and_records_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from markitai.batch import ProcessResult
    from markitai.workflow.llm_failure import llm_fallback_warnings
    from tests.unit.serve.test_serve_api import (
        _make_app,
        _multipart,
        _serve_client,
        _wait_job_done,
    )

    runs: list[int] = []

    async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
        runs.append(1)
        base = out_dir / f"{file_path.name}.md"
        if len(runs) == 1:
            base.write_text("BASE-V1", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(base))
        if len(runs) == 2:
            # An enhance whose every request failed: base only, warned
            base.write_text("BASE-V2", encoding="utf-8")
            return ProcessResult(
                success=True,
                output_path=str(base),
                warnings=llm_fallback_warnings(file_path.name, "RateLimitError: 429"),
            )
        enhanced = base.with_suffix(".llm.md")
        enhanced.write_text("ENHANCED", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(enhanced), llm_enhanced=True)

    cfg = MarkitaiConfig()
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model="openai/test", api_key="test"),
        )
    ]
    monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        lambda *_args, **_kwargs: object(),
    )
    enhance = {"operation": "enhance", "options": {"llm": True}}
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        created = await client.post(
            "/api/jobs", files=_multipart(files=[("doc.txt", b"hello")])
        )
        job_id = created.json()["job_id"]
        await _wait_job_done(client, job_id)
        queued = await client.post(f"/api/jobs/{job_id}/items/i1/retry", json=enhance)
        assert queued.status_code == 202
        failed = (await _wait_job_done(client, job_id))["items"][0]
        result = await client.get(f"/api/jobs/{job_id}/items/i1/result")

    assert failed["status"] == "done"
    assert failed["operation"] == "convert"
    assert failed["llm_enhanced"] is False
    assert result.json()["markdown"] == "BASE-V1"
    failure = failed["rerun_failure"]
    assert failure["operation"] == "enhance"
    assert failure["error_code"] == "enhancement_failed"
    assert failure["error"] == "LLM enhancement failed: RateLimitError: 429"
    assert failure["failed_at"]

    # It survives a restart, and an enhance that stands clears it
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        restored = (await client.get(f"/api/jobs/{job_id}")).json()["items"][0]
        assert restored["rerun_failure"] == failure
        queued = await client.post(f"/api/jobs/{job_id}/items/i1/retry", json=enhance)
        assert queued.status_code == 202
        enhanced = (await _wait_job_done(client, job_id))["items"][0]
    assert enhanced["llm_enhanced"] is True
    assert enhanced["rerun_failure"] is None

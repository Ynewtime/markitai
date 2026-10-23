"""onnxruntime threads of the models markitai runs in its own process."""

from __future__ import annotations

import sys
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from markitai.converter import onnx_threads, pdf_parallel

ort = pytest.importorskip("onnxruntime")


class _RecordingSession:
    """Stands in for ``onnxruntime.InferenceSession``: keeps its options."""

    made: list[_RecordingSession] = []

    def __init__(
        self, path: str, sess_options: Any = None, providers: Any = None
    ) -> None:
        self.path = path
        self.options = sess_options
        _RecordingSession.made.append(self)


def _fake_pymupdf4llm(tmp_path: Path, body: str) -> None:
    """A ``pymupdf4llm`` on sys.path whose import runs *body*."""
    (tmp_path / "pymupdf4llm.py").write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def fake_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delitem(sys.modules, "pymupdf4llm", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(ort, "InferenceSession", _RecordingSession)
    _RecordingSession.made = []
    return tmp_path


def test_the_layout_models_get_at_most_four_threads(
    fake_library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pymupdf4llm loads its layout sessions as it is imported: those get the
    cap, on one inter-op thread; any other session built meanwhile keeps
    onnxruntime's default, and the class is put back afterwards."""
    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 16)
    _fake_pymupdf4llm(
        fake_library,
        """
        import onnxruntime as ort
        layout = ort.InferenceSession(
            "/site/pymupdf/layout/resources/onnx/layout.onnx",
            sess_options=ort.SessionOptions(),
        )
        bare = ort.InferenceSession("/site/pymupdf/layout/resources/onnx/feat.onnx")
        other = ort.InferenceSession("/host/model.onnx", ort.SessionOptions())
        """,
    )
    module = onnx_threads.import_pymupdf4llm()
    assert module.layout.options.intra_op_num_threads == 4
    assert module.layout.options.inter_op_num_threads == 1
    assert module.bare.options.intra_op_num_threads == 4
    assert module.other.options.intra_op_num_threads == 0
    assert ort.InferenceSession is _RecordingSession


def test_a_small_machine_gives_each_session_its_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 2)
    assert onnx_threads.local_threads() == 2
    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 10)
    assert onnx_threads.local_threads(share=3) == 3
    assert onnx_threads.local_threads(share=20) == 1
    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 64)
    assert onnx_threads.local_threads() == onnx_threads.LOCAL_SESSION_THREADS


def test_an_import_under_way_on_another_thread_is_waited_for(
    fake_library: Path,
) -> None:
    """serve's warm-up imports pymupdf4llm while a job's conversion may ask
    for it: the conversion gets the loaded module, never the half-loaded one
    (``partially initialized module 'pymupdf4llm' has no attribute
    'to_markdown'`` failed the owner-check's first PDF)."""
    started, release = threading.Event(), threading.Event()
    sys.modules["_markitai_test_gate"] = type(sys)("_markitai_test_gate")
    sys.modules["_markitai_test_gate"].started = started  # type: ignore[attr-defined]
    sys.modules["_markitai_test_gate"].release = release  # type: ignore[attr-defined]
    _fake_pymupdf4llm(
        fake_library,
        """
        import _markitai_test_gate as gate
        gate.started.set()
        gate.release.wait(10)
        def to_markdown():
            return "ok"
        """,
    )
    warm = threading.Thread(target=onnx_threads.import_pymupdf4llm, daemon=True)
    warm.start()
    try:
        assert started.wait(10)
        got: list[Any] = []
        job = threading.Thread(
            target=lambda: got.append(onnx_threads.import_pymupdf4llm()), daemon=True
        )
        job.start()
        job.join(0.3)
        assert job.is_alive()  # waiting for the warm-up's import
        release.set()
        job.join(10)
        warm.join(10)
        assert got and got[0].to_markdown() == "ok"
        assert ort.InferenceSession is _RecordingSession
    finally:
        release.set()
        del sys.modules["_markitai_test_gate"]


def test_ocr_engines_default_to_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine built without a thread count no longer runs every core."""
    from markitai.ocr import OCRProcessor

    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 16)
    monkeypatch.setattr(OCRProcessor, "_global_engine", None)
    monkeypatch.setattr(OCRProcessor, "_global_config", None)
    monkeypatch.setattr(OCRProcessor, "_global_threads", None)
    with patch.object(OCRProcessor, "_create_engine_impl") as create:
        OCRProcessor().engine  # noqa: B018
    assert create.call_args.kwargs["threads"] == 4


def test_pdf_pages_read_here_count_the_cores_this_process_may_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container's CPU set, not the machine's core count, sizes the page
    threads (and with them each engine run's share)."""
    import os

    from markitai.converter.pdf import PdfConverter

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(pdf_parallel, "usable_cores", lambda: 4)
    assert PdfConverter._ocr_thread_count(3) == 2
    assert onnx_threads.local_threads(PdfConverter._ocr_thread_count(3)) == 2


def test_serve_warms_the_converters_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    done = threading.Event()

    def fake_import() -> None:
        calls.append("pymupdf4llm")
        done.set()

    monkeypatch.setattr(onnx_threads, "_warm_started", False)
    monkeypatch.setattr(onnx_threads, "import_pymupdf4llm", fake_import)
    monkeypatch.setattr(onnx_threads, "_CONVERTER_LIBRARIES", ())
    onnx_threads.warm_up_in_background()
    onnx_threads.warm_up_in_background()
    assert done.wait(10)
    assert calls == ["pymupdf4llm"]

"""The OCR stack's first load after an install (owner check, macOS).

On an Apple silicon test machine the first process to load the OCR stack
after an install spent 33 s importing OpenCV (macOS checks each of its ~100
newly installed native libraries once) and 7 s building the engine; later
processes took 0.3 s. That cost belongs where setup is expected (doctor,
the setup scripts, serve's start), and a conversion that has to pay it
says so. Conversions without OCR never load the OCR stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from markitai import ocr

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Nothing prepared in this process; engine builds recorded, not run."""
    built: list[Any] = []
    monkeypatch.setattr(ocr, "_prepared_here", False)
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: True)

    monkeypatch.setattr(ocr, "_load_in_child", lambda _config: built.append(1))
    return built


def test_the_stack_is_prepared_once_and_remembered(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not ocr.ocr_stack_prepared()
    took = ocr.prepare_ocr_stack()
    assert took is not None and fresh == [1]
    assert ocr.prepare_ocr_stack() is None
    # Another process of the same installation: the record says so
    monkeypatch.setattr(ocr, "_prepared_here", False)
    assert ocr.ocr_stack_prepared()
    assert ocr.prepare_ocr_stack() is None and fresh == [1]
    record = json.loads((Path.home() / ".markitai" / "ocr-prepared.json").read_text())
    assert record["python"] == sys.executable


def test_a_reinstall_is_prepared_again(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """New files (an upgrade, a reinstall of the same version) are checked
    by macOS again: the record names where and when they were written."""
    ocr.prepare_ocr_stack()
    monkeypatch.setattr(ocr, "_prepared_here", False)
    real = ocr._ocr_stack_fingerprint
    monkeypatch.setattr(
        ocr, "_ocr_stack_fingerprint", lambda: {**real(), "cv2": ["elsewhere", 1]}
    )
    assert not ocr.ocr_stack_prepared()


def test_threads_asking_at_once_load_once(fresh: list[Any]) -> None:
    results: list[float | None] = []
    threads = [
        threading.Thread(target=lambda: results.append(ocr.prepare_ocr_stack()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert fresh == [1]
    assert sum(r is not None for r in results) == 1


def test_the_child_loads_it_and_reports_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child builds a real engine (RapidOCR installed) and keeps nothing
    in this process; a configuration it cannot build is an error."""
    pytest.importorskip("rapidocr")
    from markitai.config import OCRConfig

    before = "cv2" in sys.modules
    ocr._load_in_child(OCRConfig(enabled=True))
    assert ("cv2" in sys.modules) == before
    with pytest.raises(RuntimeError, match="ocr.lang"):
        ocr._load_in_child(OCRConfig.model_construct(enabled=True, lang="klingon"))


def test_without_the_ocr_extra_nothing_happens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_prepared_here", False)
    monkeypatch.setattr(ocr, "is_ocr_available", lambda: False)
    assert ocr.prepare_ocr_stack() is None


def test_the_cli_says_it_before_a_first_local_ocr(
    fresh: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    from rich.console import Console

    from markitai.cli.processors.file import prepare_ocr_first_use
    from markitai.config import MarkitaiConfig

    console = Console(file=sys.stderr, force_terminal=False, width=200)
    cfg = MarkitaiConfig()
    cfg.ocr.enabled = True
    cfg.llm.enabled = True  # pages go to the vision model: nothing to load
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    assert fresh == []
    cfg.llm.enabled = False
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    assert fresh == [1]
    assert capsys.readouterr().err.count("Preparing OCR (first time only)") == 1


def test_doctor_names_the_first_preparation(fresh: list[Any]) -> None:
    from markitai.cli.commands.doctor import _rapidocr_label

    assert "first time only" in _rapidocr_label()
    ocr.prepare_ocr_stack()
    assert _rapidocr_label() == "RapidOCR"


def test_the_setup_warm_up_prepares_it(fresh: list[Any]) -> None:
    from markitai.warmup import _prepare_ocr

    assert _prepare_ocr()["ocr"] == "prepared" and fresh == [1]


def test_serve_warm_up_prepares_it_after_the_pdf_stack(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.converter import onnx_threads

    order: list[str] = []
    done = threading.Event()

    def prepared(config: Any = None) -> None:
        order.append("ocr")
        done.set()

    monkeypatch.setattr(onnx_threads, "_warm_started", False)
    monkeypatch.setattr(onnx_threads, "import_pymupdf4llm", lambda: order.append("pdf"))
    monkeypatch.setattr(onnx_threads, "_CONVERTER_LIBRARIES", ())
    monkeypatch.setattr(ocr, "prepare_ocr_stack", prepared)
    onnx_threads.warm_up_in_background()
    assert done.wait(10)
    assert order == ["pdf", "ocr"]


@pytest.mark.parametrize(
    ("name", "absent"),
    [
        ("sample.pptx", ("cv2", "rapidocr", "onnxruntime", "pymupdf")),
        ("sample.docx", ("cv2", "rapidocr", "onnxruntime", "pymupdf", "numpy")),
        ("sample.pdf", ("cv2", "rapidocr")),
    ],
)
def test_a_conversion_without_ocr_loads_no_ocr_stack(
    tmp_path: Path, name: str, absent: tuple[str, ...]
) -> None:
    """Each of these is seconds of first load on macOS after an install
    (OpenCV 33 s); a PDF needs onnxruntime for its layout models only."""
    script = (
        "import asyncio, json, sys\n"
        "from pathlib import Path\n"
        "from markitai.config import MarkitaiConfig\n"
        "from markitai.serve.jobs import process_file_item\n"
        "cfg = MarkitaiConfig(); cfg.cache.enabled = False\n"
        f"r = asyncio.run(process_file_item(Path({str(FIXTURES / name)!r}), cfg, "
        f"Path({str(tmp_path)!r}), None))\n"
        "print(json.dumps({'ok': r.success, 'loaded': sorted("
        "{m.split('.')[0] for m in sys.modules})}))\n"
    )
    pytest.importorskip("fastapi")
    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
        check=False,
    )
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["ok"], run.stderr[-2000:]
    assert not set(absent) & set(report["loaded"])

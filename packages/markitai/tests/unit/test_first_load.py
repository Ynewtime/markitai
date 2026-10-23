"""The first load of an installation's native libraries (owner check, macOS).

On an Apple silicon test machine the first process to load them after an
install paid macOS's check of each newly installed library: OpenCV 33 s,
PyMuPDF 2.9 s, NumPy 1.1 s, the first OCR engine 7 s; later processes 0.3 s.
A first conversion took 8 s for a PPTX and a PDF and 38 s for a 3-page
scan. ``warmup.prepare`` pays it once, in a child process, from doctor,
the setup scripts and serve's start; an OCR conversion that still has to
wait says so. Conversions without OCR never load the OCR stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from markitai import warmup

FIXTURES = Path(__file__).parents[1] / "fixtures"
#: The real child, taken before conftest stubs it for every test
_LOAD_IN_CHILD = warmup._load_in_child


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Nothing prepared in this process; the child recorded, not run."""
    children: list[Any] = []
    monkeypatch.setattr(warmup, "_prepared_here", False)
    monkeypatch.setattr(warmup, "_load_in_child", children.append)
    return children


def test_an_installation_is_prepared_once_and_remembered(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not warmup.prepared()
    assert warmup.prepare() is not None and fresh == [None]
    assert warmup.prepare() is None
    # Another process of the same installation: the record says so
    monkeypatch.setattr(warmup, "_prepared_here", False)
    assert warmup.prepared()
    assert warmup.prepare() is None and len(fresh) == 1
    record = json.loads(warmup.prepared_record().read_text())
    assert record["python"] == sys.executable
    assert "pymupdf" in record["packages"] and "cv2" in record["packages"]


def test_a_changed_installation_is_prepared_again(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """New files (an upgrade, a reinstall of the same version) are checked
    by macOS again: the record names where and when they were written."""
    warmup.prepare()
    monkeypatch.setattr(warmup, "_prepared_here", False)
    real = warmup._fingerprint
    monkeypatch.setattr(
        warmup, "_fingerprint", lambda: {**real(), "site_packages": ["x", 1]}
    )
    assert not warmup.prepared()


def test_threads_asking_at_once_load_once(fresh: list[Any]) -> None:
    results: list[float | None] = []
    threads = [
        threading.Thread(target=lambda: results.append(warmup.prepare()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(fresh) == 1
    assert sum(r is not None for r in results) == 1


def test_the_child_loads_every_library_a_conversion_uses(tmp_path: Path) -> None:
    """The child (``--native``) leaves the libraries loaded behind in its
    own process only; a configuration it cannot build is an error."""
    pytest.importorskip("rapidocr")
    from markitai.config import OCRConfig

    script = (
        "import sys\n"
        "from markitai.warmup import _load_everything\n"
        "_load_everything(None)\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    run = subprocess.run(
        [sys.executable, "-P", "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        check=False,
    )
    loaded = set(run.stdout.split())
    assert {"pymupdf", "pymupdf4llm", "onnxruntime", "numpy", "PIL", "lxml"} <= loaded
    assert {"pptx", "openpyxl", "cv2", "rapidocr"} <= loaded
    before = "cv2" in sys.modules
    _LOAD_IN_CHILD(OCRConfig(enabled=True))
    assert ("cv2" in sys.modules) == before
    with pytest.raises(RuntimeError, match="ocr.lang"):
        _LOAD_IN_CHILD(OCRConfig.model_construct(enabled=True, lang="klingon"))


def test_the_cli_says_it_before_a_first_local_ocr(
    fresh: list[Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    from markitai import ocr
    from markitai.cli.processors.file import prepare_ocr_first_use
    from markitai.config import MarkitaiConfig

    monkeypatch.setattr(ocr, "is_ocr_available", lambda: True)
    console = Console(file=sys.stderr, force_terminal=False, width=200)
    cfg = MarkitaiConfig()
    cfg.ocr.enabled = True
    cfg.llm.enabled = True  # pages go to the vision model: nothing to load
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    assert fresh == []
    cfg.llm.enabled = False
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    prepare_ocr_first_use(cfg, console=console, quiet=False)
    assert len(fresh) == 1
    assert capsys.readouterr().err.count("Preparing OCR (first time only)") == 1


def test_doctor_loads_them_the_first_time_only(fresh: list[Any]) -> None:
    from markitai.cli.commands.doctor import _libraries_label, _prepare_libraries
    from markitai.config import MarkitaiConfig

    assert "first time only" in _libraries_label()
    first = _prepare_libraries(MarkitaiConfig())
    assert first["status"] == "ok" and "loaded once" in first["message"]
    assert _prepare_libraries(MarkitaiConfig()).get("hidden")
    assert len(fresh) == 1


def test_the_setup_warm_up_prepares_them(fresh: list[Any]) -> None:
    assert warmup._prepare_libraries()["libraries"] == "prepared" and len(fresh) == 1


def test_serve_prepares_in_a_child_before_loading_here(
    fresh: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.converter import onnx_threads

    order: list[str] = []
    done = threading.Event()

    def pdf_stack() -> None:
        order.append("pdf")
        done.set()

    monkeypatch.setattr(onnx_threads, "_warm_started", False)
    monkeypatch.setattr(warmup, "prepare", lambda _config=None: order.append("child"))
    monkeypatch.setattr(onnx_threads, "import_pymupdf4llm", pdf_stack)
    monkeypatch.setattr(onnx_threads, "_CONVERTER_LIBRARIES", ())
    onnx_threads.warm_up_in_background()
    assert done.wait(10)
    assert order == ["child", "pdf"]


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

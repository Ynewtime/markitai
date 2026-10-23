"""Prepare a fresh install for its first run: ``python -m markitai.warmup``.

Python compiles a module the first time it is imported and keeps the
bytecode beside it. Left to the first run, that compiling is what made the
first ``markitai doctor`` after an install take minutes on a test machine
(Windows 11 on Arm, x64 Python under emulation, every new file scanned by
Defender). Compiling everything at install time (``UV_COMPILE_BYTECODE``)
moved the cost into setup instead: about 13,400 files, most of which
markitai never imports (LiteLLM's 2,700 modules, the OpenAI and Anthropic
SDKs' 3,800), and setup did it twice.

This imports what markitai's own code imports, and nothing else: every
markitai module, and every module its sources name in an import statement
(read from the sources, so the list cannot drift), in a few processes at
once and below normal priority. The setup scripts start it in the
background right after installing markitai and wait for it at the end.
It then loads every native library a conversion may use once, in a child
process, and records it (:func:`prepare`; ``markitai doctor`` and
``markitai serve``'s start do the same): the first load after an install is
the slow one on macOS, and the first conversion no longer pays it.
``--browser`` launches Chromium once as ``markitai doctor`` does (the first
launch after a download is the slow one on Windows) and remembers it.

Usage::

    python -m markitai.warmup [--jobs N] [--report FILE] [--browser]
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_PACKAGE = Path(__file__).resolve().parent
# Modules that run something when imported
_SKIP = {"markitai.__main__", "markitai.warmup"}
# Windows' BELOW_NORMAL_PRIORITY_CLASS
_BELOW_NORMAL = 0x00004000


def module_names() -> list[str]:
    """Every markitai module, then every installed module markitai imports."""
    own: list[str] = []
    named: set[str] = set()
    for path in sorted(_PACKAGE.rglob("*.py")):
        relative = path.relative_to(_PACKAGE.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        own.append(".".join(parts))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                named.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                named.add(node.module)
    stdlib = set(sys.stdlib_module_names)
    third_party = sorted(
        name
        for name in named
        if name.split(".")[0] not in stdlib
        and name.split(".")[0] != "markitai"
        and _installed(name.split(".")[0])
    )
    return [name for name in own if name not in _SKIP] + third_party


def _installed(top: str) -> bool:
    try:
        return importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        return False


def _weight(top: str) -> int:
    """How much compiling a top-level package takes, roughly (its .py files)."""
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return 1
    folders = getattr(spec, "submodule_search_locations", None) or []
    return max(1, sum(1 for folder in folders for _ in Path(folder).rglob("*.py")))


def _shares(names: list[str], jobs: int) -> list[list[str]]:
    """Split *names* into *jobs* lists of similar weight, a package per list."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.split(".")[0], []).append(name)
    shares: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0] * jobs
    for top in sorted(groups, key=_weight, reverse=True):
        lightest = loads.index(min(loads))
        shares[lightest].extend(groups[top])
        loads[lightest] += _weight(top)
    return [share for share in shares if share]


def _import_all(names: list[str]) -> list[str]:
    """Import *names* (compiling them), returning the ones that could not be.

    A module of an extra that is not installed is no loss and not listed; one
    that fails otherwise (an optional library whose native part is missing,
    such as cairosvg without cairo) compiles on first use, if ever used.
    """
    from markitai.utils.litellm_import import prepare_litellm_import

    # Before LiteLLM loads: no cost-map fetch over the network
    prepare_litellm_import()
    failed: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # A markitai module of an extra that is not installed (serve
            # without fastapi): nothing to prepare
            missing = (exc.name or "").split(".")[0]
            if not missing or _installed(missing):
                failed.append(name)
        except BaseException as exc:  # noqa: BLE001 - an optional module may fail any way
            if isinstance(exc, KeyboardInterrupt):
                raise
            failed.append(name)
    return failed


def _worker(names: list[str]) -> int:
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass
    import contextlib
    import io

    # Some packages print when imported; only the report goes to the parent
    with contextlib.redirect_stdout(io.StringIO()):
        unavailable = _import_all(names)
    print(json.dumps({"unavailable": unavailable, "modules": len(sys.modules)}))
    return 0


def _launch_browser() -> dict[str, object]:
    """Launch Chromium once, as doctor does, and remember the launch."""
    from markitai.fetch_playwright import is_playwright_browser_installed

    if not is_playwright_browser_installed(use_cache=False):
        return {"browser": "not installed"}
    doctor = importlib.import_module("markitai.cli.commands.doctor")
    started = time.perf_counter()
    ok, detail = doctor._smoke_test_playwright_browser(remembered=False)
    return {
        "browser": "launched" if ok else f"failed: {detail}",
        "browser_seconds": round(time.perf_counter() - started, 3),
    }


def run(jobs: int, browser: bool) -> dict[str, object]:
    """Compile in *jobs* processes (and launch Chromium); the report."""
    started = time.perf_counter()
    sys.dont_write_bytecode = False
    names = module_names()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    flags = _BELOW_NORMAL if sys.platform == "win32" else 0
    workers = [
        subprocess.Popen(  # noqa: S603 - this interpreter, this module
            [sys.executable, "-m", "markitai.warmup", "--worker", ",".join(share)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=flags,
        )
        for share in _shares(names, jobs)
    ]
    unavailable: list[str] = []
    crashed = 0
    for worker in workers:
        out, _ = worker.communicate()
        lines = (out or b"").decode("utf-8", "replace").strip().splitlines()
        try:
            unavailable += json.loads(lines[-1]).get("unavailable", [])
        except (IndexError, ValueError):
            crashed += 1
    report: dict[str, object] = {
        "modules": len(names),
        "processes": len(workers),
        "unavailable": sorted(unavailable),
        # A process that died before its report: its share compiles on use
        "crashed": crashed,
    }
    report.update(_prepare_libraries())
    if browser:
        report.update(_launch_browser())
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


def _prepare_libraries() -> dict[str, object]:
    """Load every native library a conversion may use once, as ``markitai
    doctor`` does (:func:`prepare`)."""
    try:
        took = prepare()
    except Exception as exc:  # noqa: BLE001 - the first conversion reports it
        return {"libraries": f"failed: {exc}"}
    return {"libraries": "prepared", "libraries_seconds": round(took or 0.0, 3)}


# --- the first load after an install ---------------------------------------------
#
# macOS checks every newly installed native library the first time a process
# loads it, and remembers it for every later process: on an Apple silicon test
# machine OpenCV took 33 s (about 100 libraries), PyMuPDF 2.9 s, NumPy 1.1 s,
# the first OCR engine 7 s; later processes 0.3 s in all. A first conversion
# that paid it took 8 s for a PPTX and a PDF, 38 s for a 3-page scan. The
# setup scripts (``run``), ``markitai doctor`` and ``markitai serve``'s start
# pay it instead, in a child process: the caller keeps none of it in memory
# and is not held up meanwhile (a process loads one native library at a time).

#: Packages with native libraries a conversion loads, whose files the record
#: names (an upgrade or a reinstall writes new files, which macOS checks again).
_NATIVE_PACKAGES = (
    "numpy",
    "PIL",
    "lxml",
    "pymupdf",
    "onnxruntime",
    "psutil",
    "pydantic_core",
    "cv2",
    "rapidocr",
    "shapely",
    "pyclipper",
    "charset_normalizer",
    "yaml",
)
#: A child loading everything for the first time may take this long (40 s on
#: the slowest machine measured) before it is given up.
_PREPARE_TIMEOUT = 600

#: What a document conversion loads (PDF, office files, images): the child
#: loads these first and then says so, so a conversion waiting for it starts
#: before the LLM stack, the browser and the OCR stack have loaded.
_CONVERSION_MODULES = (
    "numpy",
    "PIL.Image",
    "lxml.etree",
    "pymupdf",
    "psutil",
    "charset_normalizer",
    "pptx",
    "openpyxl",
    "docx",
    "mammoth",
)
#: Top-level packages only OCR uses: the last phase of the child.
_OCR_PACKAGES = frozenset({"rapidocr", "cv2", "shapely", "pyclipper"})
#: The line the child prints once everything but OCR is loaded.
_CORE_READY = "markitai-core-ready"

#: Said on a conversion that waits for the first load (the workbench shows
#: it on the job's items while they wait, and after; the CLI prints its own).
FIRST_CONVERSION_NOTICE = (
    "First conversion since installing: the libraries conversions use load "
    "once (several seconds on macOS); later conversions start right away."
)

_prepare_lock = threading.Lock()
_state = threading.Condition()
#: What this process knows: everything but OCR loaded (or given up on),
#: OCR loaded too, a child running now.
_core_here = False
_ocr_here = False
_running = False


def prepared_record() -> Path:
    """Where :func:`prepare` records that it ran for this installation."""
    return Path.home() / ".markitai" / "prepared.json"


def _fingerprint() -> dict[str, object]:
    """This installation: the interpreter, its site-packages folder and the
    native packages' files, each with when it was written. Stats only."""
    import sysconfig

    def stamp(path: str | None) -> list[object] | None:
        if not path:
            return None
        try:
            return [path, os.stat(path).st_mtime_ns]
        except OSError:
            return [path, None]

    packages: dict[str, object] = {}
    for name in _NATIVE_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        packages[name] = stamp(getattr(spec, "origin", None))
    return {
        "python": sys.executable,
        "site_packages": stamp(sysconfig.get_paths().get("purelib")),
        "packages": packages,
    }


def _record() -> dict[str, Any] | None:
    try:
        record = json.loads(prepared_record().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("installation") != _fingerprint():
        return None
    return record


def prepared(ocr: bool = False) -> bool:
    """Whether this installation's native libraries have loaded once
    (:func:`prepare` ran since it was installed or changed), the OCR stack's
    too when *ocr*."""
    if _ocr_here or (_core_here and not ocr):
        return True
    record = _record()
    return record is not None and (bool(record.get("ocr")) or not ocr)


def _write_record(ocr: bool) -> None:
    record = prepared_record()
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps({"installation": _fingerprint(), "ocr": ocr}), encoding="utf-8"
        )
    except OSError:  # asked again next time; nothing breaks
        pass


def _core_loaded() -> None:
    global _core_here
    with _state:
        _core_here = True
        _state.notify_all()


def _load_in_child(
    ocr_config: Any = None, *, ocr: bool = False, rest: bool = True, on_core: Any = None
) -> None:
    """Run :func:`_load_everything` in a child process of this interpreter:
    what conversions use, then (*rest*) the other libraries, then (*ocr*) the
    OCR stack. *on_core* is called once the first part is loaded, while the
    child goes on.

    Raises:
        RuntimeError: The child failed (its last line of error output).
        subprocess.TimeoutExpired: It took over ten minutes.
    """
    command = [sys.executable, "-P", "-m", "markitai.warmup", "--native"]
    if not rest:
        command.append("--core-only")
    if ocr:
        command.append("--with-ocr")
        if ocr_config is not None:
            command += ["--ocr-config", ocr_config.model_dump_json()]
    child = subprocess.Popen(  # noqa: S603 - this interpreter, this module
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    killed: list[bool] = []

    def kill() -> None:
        killed.append(True)
        child.kill()

    timer = threading.Timer(_PREPARE_TIMEOUT, kill)
    timer.start()
    errors: list[str] = []
    drain = threading.Thread(
        target=lambda: errors.extend(child.stderr or []), daemon=True
    )
    drain.start()
    try:
        for line in child.stdout or []:
            if line.strip() == _CORE_READY and on_core is not None:
                on_core()
        code = child.wait()
        drain.join()
    finally:
        timer.cancel()
    if code != 0:
        lines = [line.strip() for line in errors if line.strip()]
        if killed:
            raise subprocess.TimeoutExpired(command, _PREPARE_TIMEOUT)
        raise RuntimeError(lines[-1] if lines else f"exit status {code}")


def prepare(
    ocr_config: Any = None, *, ocr: bool | None = None, rest: bool = True
) -> float | None:
    """Load every native library a conversion may use once, in a child
    process, and record it (see :func:`prepared`).

    The child imports every installed module markitai's sources name,
    builds the PDF layout models' sessions and loads Pillow's codecs, the
    document conversion libraries first; without *rest*, only those (a
    conversion waiting for it). Then, with *ocr* (default: when the ``ocr``
    extra is installed), it imports the OCR stack and builds an engine with
    *ocr_config*'s language. Once the conversion libraries are loaded,
    :func:`wait_for_core` lets conversions without OCR go on. Calls at once wait for the one that loads; a child that failed
    is not run again in this process (each conversion would wait for it).

    Returns:
        The seconds it took, or None when there was nothing to do.

    Raises:
        RuntimeError, subprocess.TimeoutExpired: The child failed.
    """
    global _ocr_here, _running
    if ocr is None:
        from markitai.ocr import is_ocr_available

        ocr = is_ocr_available()
    with _prepare_lock:
        if prepared(ocr):
            return None
        record = _record()
        core_done = record is not None
        if core_done:
            _core_loaded()
        with _state:
            _running = True
        started = time.perf_counter()

        def on_core() -> None:
            if not core_done:
                _write_record(False)
            _core_loaded()

        try:
            _load_in_child(ocr_config, ocr=ocr, rest=rest, on_core=on_core)
        except BaseException:
            # Given up on in this process: nothing waits for it again
            _core_loaded()
            if ocr:
                _ocr_here = True
            raise
        finally:
            with _state:
                _running = False
                _state.notify_all()
        _write_record(ocr)
        _core_loaded()
        if ocr:
            _ocr_here = True
        return time.perf_counter() - started


def start_preparing(ocr_config: Any = None) -> threading.Thread | None:
    """:func:`prepare` in a daemon thread (``markitai serve``'s start),
    counted as under way from now on, so :func:`wait_for_core` waits for it
    instead of starting another; None when there is nothing to do."""
    global _running
    from markitai.ocr import is_ocr_available

    ocr = is_ocr_available()
    if prepared(ocr):
        return None
    with _state:
        _running = True

    def run() -> None:
        global _running
        try:
            prepare(ocr_config, ocr=ocr)
        except Exception:  # a conversion reports it where it matters
            pass
        finally:
            with _state:
                _running = False
                _state.notify_all()

    thread = threading.Thread(target=run, name="markitai-first-load", daemon=True)
    thread.start()
    return thread


def wait_for_core(ocr_config: Any = None) -> None:
    """Return once everything but OCR has loaded once: at once when it has,
    else when the preparation under way gets there, else after preparing
    it here (without OCR)."""
    if prepared():
        return
    with _state:
        if _running:
            _state.wait_for(lambda: _core_here or not _running, _PREPARE_TIMEOUT)
            if _core_here:
                return
    try:
        prepare(ocr_config, ocr=False, rest=False)
    except Exception:  # the conversion loads what it needs itself
        _core_loaded()


def _load_everything(
    ocr_config_json: str | None, ocr: bool = True, rest: bool = True
) -> int:
    """The child of :func:`prepare`: load what document conversions use
    and say so, then the other libraries markitai uses, then (*ocr*) the OCR
    stack, building an engine; then exit."""
    import contextlib
    import io

    names = [
        name
        for name in module_names()
        if not name.startswith("markitai") and name.split(".")[0] not in _OCR_PACKAGES
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        _import_all(
            [name for name in _CONVERSION_MODULES if _installed(name.split(".")[0])]
        )
        from markitai.converter.onnx_threads import import_pymupdf4llm

        try:
            import_pymupdf4llm()  # its layout models' sessions
        except Exception:  # noqa: BLE001 - the conversion reports it
            pass
        try:
            from PIL import features

            for feature in ("webp", "avif", "freetype2", "littlecms2"):
                features.check(feature)
        except Exception:  # noqa: BLE001 - optional codecs
            pass
    print(_CORE_READY, flush=True)
    if rest:
        with contextlib.redirect_stdout(io.StringIO()):
            # The LLM stack, the browser, the optional extras
            _import_all(names)
    from markitai.ocr import OCRProcessor, is_ocr_available

    if ocr and is_ocr_available():
        config = None
        if ocr_config_json:
            from markitai.config import OCRConfig

            config = OCRConfig.model_validate_json(ocr_config_json)
        with contextlib.redirect_stdout(io.StringIO()):
            OCRProcessor._create_engine_impl(config, threads=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m markitai.warmup``: see the module docstring."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m markitai.warmup")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--only-browser", action="store_true")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--with-ocr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--core-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ocr-config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.native:
        return _load_everything(
            args.ocr_config, ocr=args.with_ocr, rest=not args.core_only
        )
    if args.worker is not None:
        return _worker([name for name in args.worker.split(",") if name])
    if args.only_browser:
        started = time.perf_counter()
        report = _launch_browser()
        report["seconds"] = round(time.perf_counter() - started, 3)
    else:
        report = run(max(1, args.jobs), args.browser)
    text = json.dumps(report)
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

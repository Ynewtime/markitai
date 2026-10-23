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
With the ``ocr`` extra installed it also builds an OCR engine once
(:func:`markitai.ocr.prepare_ocr_stack`), so the first ``--ocr`` does not
load the OCR stack for the first time.
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
import time
from pathlib import Path

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
    report.update(_prepare_ocr())
    if browser:
        report.update(_launch_browser())
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


def _prepare_ocr() -> dict[str, object]:
    """Build an OCR engine once and remember it, as ``markitai doctor`` does
    (``ocr.prepare_ocr_stack``): the engine's first build after an install
    took 7 s on macOS, on top of OpenCV's first load the imports paid."""
    from markitai.ocr import is_ocr_available, prepare_ocr_stack

    if not is_ocr_available():
        return {"ocr": "not installed"}
    try:
        took = prepare_ocr_stack()
    except Exception as exc:  # noqa: BLE001 - the first --ocr reports it
        return {"ocr": f"failed: {exc}"}
    return {"ocr": "prepared", "ocr_seconds": round(took or 0.0, 3)}


def main(argv: list[str] | None = None) -> int:
    """``python -m markitai.warmup``: see the module docstring."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m markitai.warmup")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--only-browser", action="store_true")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
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

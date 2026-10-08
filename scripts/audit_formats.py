#!/usr/bin/env python3
"""Compare isolated native and reference API outputs; never modify the reference."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import Counter

from isolation import isolated_env


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assets(paths: list[str]) -> list[dict]:
    return [{"name": Path(path).name, "bytes": Path(path).stat().st_size,
             "sha256": digest(Path(path))} for path in paths]


def worker(request: dict, engine: str) -> dict:
    case = Path(request["case_dir"])
    output = case / engine
    output.mkdir(parents=True, exist_ok=True)
    isolation = Path(os.environ["MARKITAI_HOME"])
    os.environ["MARKITAI_LOG_DIR"] = str(isolation / "logs")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    if engine == "reference":
        sys.path.insert(0, str(Path(request["reference"]) / "packages/markitai/src"))
        from markitai import convert
        from markitai.config import MarkitaiConfig

        config = MarkitaiConfig()
        config.cache.global_dir = str(isolation / "cache")
        config.cache.enabled = False
        result = convert(request["source"], output_dir=output, config=config,
                         llm=False, ocr=False, screenshot=False, alt=False, desc=False)
        return {"ok": True, "markdown": result.markdown,
                "frontmatter": result.frontmatter, "warnings": result.warnings,
                "skip_reason": result.skip_reason,
                "assets": assets([str(path) for path in result.assets])}

    class Buffer(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)), ("len", ctypes.c_size_t)]

    library = ctypes.CDLL(request["library"])
    library.markitai_convert_json.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
    library.markitai_convert_json.restype = Buffer
    library.markitai_buffer_free.argtypes = [ctypes.POINTER(Buffer)]
    library.markitai_buffer_free.restype = None
    encoded = json.dumps({"source": request["source"], "options": {
        "output_dir": str(output), "config": {"cache": {"enabled": False}},
        "llm": False, "ocr": False, "screenshot": False, "alt": False, "desc": False,
    }}).encode()
    input_buffer = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    response = library.markitai_convert_json(input_buffer, len(encoded))
    try:
        response_json = json.loads(ctypes.string_at(response.data, response.len))
    finally:
        library.markitai_buffer_free(ctypes.byref(response))
    if not response_json.get("ok"):
        return response_json
    result = response_json["result"]
    return {"ok": True, "markdown": result["markdown"],
            "frontmatter": result["frontmatter"], "warnings": result["warnings"],
            "skip_reason": result.get("skip_reason"), "assets": assets(result["assets"])}


def run_worker(executable: str, engine: str, request_path: Path, timeout: int) -> dict:
    response_path = request_path.with_name(f"{engine}-response.json")
    log_path = request_path.with_name(f"{engine}.log")
    # Credentials are not forwarded. Explicit configuration disables all model,
    # browser and remote conversion paths; `~` names a private home per engine.
    isolation = request_path.parent / "state" / engine
    environment = isolated_env(isolation / "home", isolation, {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR"}})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [executable, str(Path(__file__).resolve()), "--worker", engine,
               "--request", str(request_path), "--response", str(response_path)]
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(command, env=environment, cwd=request_path.parent,
                                       stdout=log, stderr=log, timeout=timeout, check=False)
        if completed.returncode != 0:
            return {"ok": False, "error": {"code": "worker_exit", "message":
                    f"Worker exit {completed.returncode}; inspect {log_path.name}"}}
        return json.loads(response_path.read_text())
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": {"code": "worker_timeout", "message": f"Exceeded {timeout}s"}}


def comparable_metadata(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "markitai_processed"}


def compare(reference: dict, native: dict, case_dir: Path) -> dict:
    if not native.get("ok"):
        return {"status": "unsupported" if native.get("error", {}).get("code") == "unsupported" else "native_error",
                "native_error": native.get("error"), "reference_ok": reference.get("ok", False)}
    if not reference.get("ok"):
        return {"status": "reference_error", "reference_error": reference.get("error")}
    before, after = reference["markdown"], native["markdown"]
    matches = {
        "markdown": before == after,
        "frontmatter": comparable_metadata(reference["frontmatter"]) == comparable_metadata(native["frontmatter"]),
        "asset_hashes": sorted(a["sha256"] for a in reference["assets"]) == sorted(a["sha256"] for a in native["assets"]),
        "skip_reason": reference.get("skip_reason") == native.get("skip_reason"),
    }
    if before != after:
        diff = difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="reference.md", tofile="native.md")
        (case_dir / "markdown.diff").write_text("".join(diff))
    before_words = Counter(re.findall(r"\w+", before.casefold()))
    after_words = Counter(re.findall(r"\w+", after.casefold()))
    recall = sum((before_words & after_words).values()) / max(1, sum(before_words.values()))
    return {"status": "parity_pass" if all(matches.values()) else "output_drift", "matches": matches,
            "reference_characters": len(before), "native_characters": len(after),
            "reference_assets": len(reference["assets"]), "native_assets": len(native["assets"]),
            "reference_token_recall": round(recall, 4),
            "native_warnings": native.get("warnings", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument("--worker", choices=["reference", "native"])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            result = worker(json.loads(args.request.read_text()), args.worker)
        except Exception as error:
            result = {"ok": False, "error": {"code": type(error).__name__, "message": str(error)}}
        args.response.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if not all([args.reference, args.library]):
        parser.error("--reference and --library are required")
    reference, library = args.reference.resolve(), args.library.resolve()
    # Keep the venv entry point: resolving its symlink loses pyvenv.cfg discovery.
    python = (args.reference_python or reference / ".venv/bin/python").absolute()
    for path in [library, python]:
        if not path.is_file(): parser.error(f"Required file does not exist: {path}")
    generated = dt.datetime.now(dt.timezone.utc)
    output = (args.output or Path(".local/audits") / generated.strftime("formats-%Y%m%dT%H%M%SZ")).resolve()
    if output == reference or reference in output.parents:
        parser.error("Audit output must be outside the read-only reference repository")
    if output.exists() and any(output.iterdir()): parser.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen_library = output / library.name
    shutil.copy2(library, frozen_library)
    fixtures = reference / "packages/markitai/tests/fixtures"
    sources = sorted(path for path in fixtures.glob("sample.*") if path.suffix != ".urls")
    sources += sorted(path for path in (fixtures / "legacy").glob("*") if path.suffix in {".doc", ".ppt", ".xls"})
    cases = []
    for index, source in enumerate(sources):
        case_name = str(source.relative_to(fixtures))
        case_dir = output / f"{index:02}-{source.name}"
        case_dir.mkdir()
        request = {"source": str(source), "case_dir": str(case_dir),
                   "reference": str(reference), "library": str(frozen_library)}
        request_path = case_dir / "request.json"
        request_path.write_text(json.dumps(request))
        before_hash = digest(source)
        before = run_worker(str(python), "reference", request_path, args.timeout)
        after = run_worker(sys.executable, "native", request_path, args.timeout)
        if digest(source) != before_hash: raise RuntimeError(f"Source fixture changed during audit: {source}")
        result = {"name": case_name, "input_sha256": before_hash, **compare(before, after, case_dir)}
        cases.append(result)
        print(f"{case_name}: {result['status']}", flush=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=reference, capture_output=True, text=True, check=False).stdout.strip()
    source_status = subprocess.run(["git", "status", "--porcelain"], cwd=reference, capture_output=True, text=True, check=False).stdout
    report = {"schema": 1, "generated_at": generated.isoformat(), "reference_revision": revision,
              "reference_dirty": bool(source_status), "library_sha256": digest(frozen_library), "library": str(library),
              "scope": "Public API output and asset content; no timing or performance claims",
              "normalization": ["frontmatter.markitai_processed"],
              "counts": dict(Counter(case["status"] for case in cases)), "cases": cases}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"report": str(report_path), "counts": report["counts"]}, indent=2))
    return int(args.require_parity and any(case["status"] != "parity_pass" for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())

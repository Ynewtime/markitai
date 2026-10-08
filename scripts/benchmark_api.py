#!/usr/bin/env python3
"""Compare repeated public API calls in isolated Python-hosted worker processes."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

from isolation import install_state_guard


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def peak_rss_bytes() -> int:
    # getrusage documents different units on these two supported platforms.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def configuration(state: Path) -> dict:
    return {
        "llm": {"enabled": False, "pure": True},
        "ocr": {"enabled": False}, "screenshot": {"enabled": False},
        "image": {"alt_enabled": False, "desc_enabled": False},
        "cache": {"enabled": False, "global_dir": str(state / "cache")},
        "history": {"record": False},
    }


def worker(request: dict) -> dict:
    isolation = install_state_guard("Python path audit events only; not metadata/descriptor syscalls, "
                                    "native FFI or an OS sandbox")
    cfg = request["config"]
    options = dict(llm=False, ocr=False, screenshot=False, alt=False, desc=False)
    if request["engine"] == "reference":
        sys.path.insert(0, str(Path(request["reference"]) / "packages/markitai/src"))
        from markitai import convert
        from markitai.config import MarkitaiConfig
        from loguru import logger

        logger.remove()
        config = MarkitaiConfig.model_validate(cfg)

        def call():
            return convert(request["source"], config=config, **options)

        def fields(result):
            return result.markdown, result.assets, result.skip_reason, result.warnings
    else:
        class Buffer(ctypes.Structure):
            _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)), ("len", ctypes.c_size_t)]

        library = ctypes.CDLL(request["library"])
        library.markitai_convert_json.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
        library.markitai_convert_json.restype = Buffer
        library.markitai_buffer_free.argtypes = [ctypes.POINTER(Buffer)]
        library.markitai_buffer_free.restype = None
        encoded = json.dumps({"source": request["source"], "options": {
            **options, "config": cfg,
        }}).encode("utf-8")
        input_buffer = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)

        def call():
            response = library.markitai_convert_json(input_buffer, len(encoded))
            try:
                decoded = json.loads(ctypes.string_at(response.data, response.len))
            finally:
                library.markitai_buffer_free(ctypes.byref(response))
            if not decoded.get("ok"):
                raise RuntimeError(f"Native conversion failed: {decoded.get('error')}")
            return decoded["result"]

        def fields(result):
            return result["markdown"], result["assets"], result.get("skip_reason"), result["warnings"]

    # Imports, caller configuration and request construction are outside the timer.
    imported_peak = peak_rss_bytes()
    warmup = call()
    if isolation["blocked_state_events"] or isolation["blocked_network_events"]:
        raise RuntimeError(f"Warmup attempted forbidden access: {isolation}")
    expected, assets, skip_reason, warnings = fields(warmup)
    if assets or skip_reason:
        raise RuntimeError("Synthetic fixture unexpectedly produced assets or was skipped")
    expected_bytes = expected.encode("utf-8")
    expected_hash = sha256(expected_bytes)
    Path(request["markdown_output"]).write_bytes(expected_bytes)
    del warmup
    samples = []
    for _ in range(request["iterations"]):
        started = time.perf_counter_ns()
        result = call()
        elapsed = time.perf_counter_ns() - started
        # No output, hashing, comparison or statistics runs inside the timer.
        markdown, assets, skip_reason, iteration_warnings = fields(result)
        if markdown != expected or assets or skip_reason or iteration_warnings != warnings:
            raise RuntimeError("Repeated calls changed their output, assets, skip reason or warnings")
        samples.append(elapsed / 1_000_000)
        del result, markdown
    if isolation["blocked_state_events"] or isolation["blocked_network_events"]:
        raise RuntimeError(f"Timed calls attempted forbidden access: {isolation}")
    return {
        "ok": True, "engine": request["engine"], "pid": os.getpid(),
        "python": sys.version, "python_executable": sys.executable,
        "samples_ms": samples, "median_ms": statistics.median(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "peak_rss_bytes": peak_rss_bytes(), "post_import_peak_rss_bytes": imported_peak,
        "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "markdown_sha256": expected_hash, "markdown_bytes": len(expected_bytes),
        "warnings": warnings, "python_isolation": isolation,
    }


def run_worker(python: Path, request: dict, directory: Path, timeout: int) -> dict:
    directory.mkdir()
    state = directory / "state"
    state.mkdir()
    temporary = state / "tmp"
    temporary.mkdir()
    request["config"] = configuration(state)
    request["markdown_output"] = str(directory / "output.md")
    config_path, request_path, response_path = (directory / name for name in
                                               ("config.json", "request.json", "response.json"))
    write_json(config_path, request["config"])
    write_json(request_path, request)
    # No caller credentials, Python paths, provider variables or user config are inherited.
    environment = {
        "PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "MARKITAI_HOME": str(state), "MARKITAI_CONFIG": str(config_path),
        "MARKITAI_LOG_DIR": str(state / "logs"), "TMPDIR": str(temporary),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    command = [str(python), str(Path(__file__).resolve()), "--worker",
               "--request", str(request_path), "--response", str(response_path)]
    with (directory / "worker.log").open("w") as log:
        completed = subprocess.run(command, env=environment, cwd=directory,
                                   stdout=log, stderr=log, timeout=timeout, check=False)
    if not response_path.is_file():
        raise RuntimeError(f"Worker failed; inspect {directory / 'worker.log'}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("ok"):
        raise RuntimeError(f"Worker failed in {directory}: {response.get('error')}")
    if completed.returncode:
        raise RuntimeError(f"Worker exited {completed.returncode}; inspect {directory / 'worker.log'}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-profile", choices=["release", "dist", "debug"], default="release")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--response", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform not in {"darwin", "linux"}:
        parser.error("Peak-RSS units are implemented for macOS and Linux only")
    if args.worker:
        try:
            result = worker(json.loads(args.request.read_text(encoding="utf-8")))
        except Exception as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        write_json(args.response, result)
        return int(not result["ok"])
    if not all([args.reference, args.library, args.output]):
        parser.error("--reference, --library and --output are required")
    if args.processes < 2 or args.iterations < 3 or args.timeout < 1:
        parser.error("Use at least 2 fresh processes, 3 timed calls and a positive timeout")
    reference, library, output = args.reference.resolve(), args.library.resolve(), args.output.resolve()
    # Resolving this symlink would lose venv discovery; use the same entry point for both engines.
    python = (args.reference_python or reference / ".venv/bin/python").absolute()
    if not library.is_file() or not python.is_file():
        parser.error("Provide a built native library and the installed reference Python environment")
    if output == reference or reference in output.parents:
        parser.error("Output must be outside the read-only reference repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / library.name
    shutil.copy2(library, frozen)
    cases = {
        "text_105k": ("input.txt", "# Benchmark\n\n" + "A paragraph with facts and 中文.\n" * 3000),
        "csv_1000_rows": ("input.csv", "name,value\n" + "".join(f"row{i},{i}\n" for i in range(1000))),
    }
    report = {
        "schema": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python_entrypoint": str(python), "processes_per_engine_per_case": args.processes,
        "timed_calls_per_process": args.iterations, "warmup_calls_per_process": 1,
        "measurement": "Native C ABI with prebuilt JSON request + ctypes + JSON response decode/free versus reference Python public API; excludes new Python binding request encoding and dataclass construction",
        "rss_scope": "Fresh Python worker peak, including imports, configuration, warmup and timed calls; not bare Rust or CLI RSS",
        "library": {"source": str(library), "frozen": str(frozen), "bytes": frozen.stat().st_size,
                    "sha256": sha256(frozen.read_bytes()), "declared_build_profile": args.build_profile},
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "reference_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip(),
        "reference_status_before": subprocess.check_output(["git", "status", "--porcelain"], cwd=reference, text=True),
        "cases": [],
    }
    for name, (filename, content) in cases.items():
        case_dir = output / name
        case_dir.mkdir()
        source = case_dir / filename
        source.write_bytes(content.encode("utf-8"))
        input_hash = sha256(source.read_bytes())
        workers = {"reference": [], "native": []}
        for repetition in range(args.processes):
            order = ("reference", "native") if repetition % 2 == 0 else ("native", "reference")
            for engine in order:
                request = {"engine": engine, "reference": str(reference), "library": str(frozen),
                           "source": str(source), "iterations": args.iterations}
                workers[engine].append(run_worker(python, request, case_dir / f"{repetition:02}-{engine}", args.timeout))
        if sha256(source.read_bytes()) != input_hash:
            raise RuntimeError(f"Benchmark source changed: {source}")
        payloads = [(case_dir / f"{i:02}-{engine}" / "output.md").read_bytes()
                    for engine in workers for i in range(args.processes)]
        parity = all(payload == payloads[0] for payload in payloads)
        measurements = {}
        for engine, results in workers.items():
            medians = [item["median_ms"] for item in results]
            peaks = [item["peak_rss_bytes"] for item in results]
            measurements[engine] = {"median_of_process_medians_ms": statistics.median(medians),
                                    "process_medians_ms": medians, "peak_rss_bytes_each": peaks,
                                    "median_peak_rss_bytes": statistics.median(peaks), "workers": results}
        ratio = measurements["reference"]["median_of_process_medians_ms"] / measurements["native"]["median_of_process_medians_ms"]
        report["cases"].append({"name": name, "input_bytes": source.stat().st_size,
                                "input_sha256": input_hash, "markdown_exact_match": parity,
                                "measurements": measurements,
                                "reference_over_native_time_ratio": ratio if parity else None})
        print(f"{name}: exact Markdown {'PASS' if parity else 'FAIL'}; measurements saved", flush=True)
    report["reference_status_after"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=reference, text=True)
    report["all_markdown_exact"] = all(case["markdown_exact_match"] for case in report["cases"])
    report["reference_status_unchanged"] = report["reference_status_before"] == report["reference_status_after"]
    write_json(output / "report.json", report)
    print(output / "report.json")
    return int(not report["all_markdown_exact"] or not report["reference_status_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())

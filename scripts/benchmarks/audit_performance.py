"""Fixed, offline performance audit: cold CLI, converters, API and batch.

Run before changing production code and retain that source snapshot. Each child
is isolated, uses the same inputs, records CPU/RSS and hashes actual Markdown.
No wall-clock threshold belongs in unit tests; compare repeated measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORMATS = (
    "txt",
    "md",
    "html",
    "csv",
    "tsv",
    "docx",
    "xlsx",
    "pptx",
    "pdf",
    "epub",
    "ipynb",
    "eml",
    "msg",
    "xml",
    "rtf",
    "odt",
    "ods",
    "org",
    "rst",
    "tex",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def worker(mode: str, path: Path, output: Path) -> None:
    from loguru import logger

    logger.remove()
    from markitai.config import MarkitaiConfig

    config = MarkitaiConfig()
    config.llm.enabled = config.ocr.enabled = config.screenshot.enabled = False
    config.image.alt_enabled = config.image.desc_enabled = False
    config.cache.enabled = False
    config.fetch.remote_consent = "never"
    start = time.perf_counter()
    rows = []
    if mode == "converter":
        from markitai.converter.base import get_converter

        for _ in range(4):
            before = time.perf_counter()
            converter = get_converter(path, config=config)
            initialized = time.perf_counter()
            if converter is None:
                raise ValueError(f"No converter for {path}")
            result = converter.convert(path, output)
            rows.append(
                {
                    "init_ms": (initialized - before) * 1000,
                    "convert_ms": (time.perf_counter() - initialized) * 1000,
                    "sha256": digest(result.markdown),
                    "chars": len(result.markdown),
                    "images": len(result.images),
                }
            )
    else:
        from markitai.api import _close_loop_bound_resources, aconvert

        async def run() -> None:
            try:
                for _ in range(4):
                    before = time.perf_counter()
                    result = await aconvert(path, config=config)
                    rows.append(
                        {
                            "total_ms": (time.perf_counter() - before) * 1000,
                            "sha256": digest(result.markdown),
                            "chars": len(result.markdown),
                        }
                    )
            finally:
                await _close_loop_bound_resources()

        asyncio.run(run())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        "AUDIT_RESULT="
        + json.dumps(
            {
                "iterations": rows,
                "wall_ms": (time.perf_counter() - start) * 1000,
                "cpu_seconds": usage.ru_utime + usage.ru_stime,
                "peak_rss_mb": usage.ru_maxrss
                / (1024**2 if sys.platform == "darwin" else 1024),
                "loaded_heavy_modules": [
                    m
                    for m in (
                        "litellm",
                        "markitdown",
                        "pandas",
                        "onnxruntime",
                        "pymupdf",
                        "openpyxl",
                        "pptx",
                    )
                    if m in sys.modules
                ],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=REPO / "packages/markitai/src")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=["converter", "api"])
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--only", nargs="+", help="Run only named cases (for paired measurements)"
    )
    parser.add_argument(
        "--append", action="store_true", help="Retain results for other cases"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.input, args.output)
        return
    inputs = args.output / "inputs"
    inputs.mkdir(exist_ok=True)
    for ext in FORMATS:
        fixture = REPO / f"packages/markitai/tests/fixtures/sample.{ext}"
        if fixture.exists():
            shutil.copyfile(fixture, inputs / fixture.name)
    for ext in ("txt", "md"):
        (inputs / f"sample.{ext}").write_text(
            "# Performance audit\n\nA complete deterministic document.\n" * 20
        )
    config = args.output / "config.json"
    config.write_text(
        json.dumps(
            {
                "llm": {"enabled": False},
                "fetch": {"remote_consent": "never"},
                "history": {"record": False},
            }
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.source.resolve())
    env["MARKITAI_NO_REMOTE_FETCH"] = "1"
    env["MARKITAI_RECORD_HISTORY"] = "0"
    cases: list[tuple[str, list[str]]] = []
    for flag in ("--version", "--help"):
        cases.append((f"cli-{flag[2:]}", ["-m", "markitai", flag]))
    for ext in FORMATS:
        path = inputs / f"sample.{ext}"
        for mode in ("converter", "api"):
            cases.append(
                (
                    f"{mode}-{ext}",
                    [
                        str(Path(__file__).resolve()),
                        "--worker",
                        mode,
                        "--input",
                        str(path),
                        "--output",
                        str(args.output / f"work-{mode}-{ext}"),
                    ],
                )
            )
        if ext in {"txt", "html", "docx", "xlsx", "pptx", "pdf"}:
            cases.append(
                (
                    f"cli-{ext}",
                    [
                        "-m",
                        "markitai",
                        str(path),
                        "--preset",
                        "minimal",
                        "--no-cache",
                        "--no-remote-fetch",
                        "--config",
                        str(config),
                    ],
                )
            )
    batch = inputs / "batch"
    batch.mkdir(exist_ok=True)
    for n in range(10):
        for ext in ("txt", "html", "docx"):
            shutil.copyfile(inputs / f"sample.{ext}", batch / f"{n:02d}.{ext}")
    cases.append(
        (
            "cli-batch-30",
            [
                "-m",
                "markitai",
                str(batch),
                "-o",
                str(args.output / "batch-output"),
                "--preset",
                "minimal",
                "--no-cache",
                "--no-remote-fetch",
                "--config",
                str(config),
            ],
        )
    )
    if args.only:
        unknown = set(args.only) - {name for name, _ in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [(name, command) for name, command in cases if name in args.only]
    previous = args.output / "audit.json"
    results = (
        json.loads(previous.read_text()) if args.append and previous.exists() else []
    )
    replacing = {name for name, _ in cases}
    results = [row for row in results if row["case"] not in replacing]
    for name, command in cases:
        runs = []
        for i in range(args.repeats):
            start = time.perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, *command],
                    check=False,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                wall = time.perf_counter() - start
                (args.output / f"{name}-{i}.stderr").write_text(result.stderr)
                (args.output / f"{name}-{i}.stdout").write_text(result.stdout)
                payload = next(
                    (
                        json.loads(line.removeprefix("AUDIT_RESULT="))
                        for line in result.stdout.splitlines()
                        if line.startswith("AUDIT_RESULT=")
                    ),
                    None,
                )
                runs.append(
                    {
                        "wall_seconds": wall,
                        "exit": result.returncode,
                        "stdout_sha256": digest(result.stdout),
                        "detail": payload,
                    }
                )
            except subprocess.TimeoutExpired:
                runs.append({"timeout": True})
                break
        passed = [r["wall_seconds"] for r in runs if r.get("exit") == 0]
        row = {
            "case": name,
            "runs": runs,
            "median_seconds": statistics.median(passed) if passed else None,
        }
        results.append(row)
        (args.output / "audit.json").write_text(json.dumps(results, indent=2))
        print(name, row["median_seconds"], flush=True)
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs.glob("sample.*")
    }
    (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

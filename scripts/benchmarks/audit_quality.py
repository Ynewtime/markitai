"""Compare complete file outputs separately from the fixed timing workload.

Uses the same inputs and frozen source baseline as the performance audit. API
conversions keep their assets on disk for inspection. No quality instrumentation
is inserted into measured conversions. Raw snapshots retain engine provenance;
only the explicitly reviewed engine migrations below are accepted differences.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE_TRANSITIONS = {
    "converter-csv": (("markitdown", "delimited"),),
    "converter-ipynb": (("markitdown", "notebook"),),
    "converter-docx": (
        ("markitdown", "mammoth"),
        ("markitdown", "native-docx"),
        ("mammoth", "native-docx"),
    ),
    "converter-xlsx": (
        ("markitdown", "pandas"),
        ("markitdown", "openpyxl"),
        ("pandas", "openpyxl"),
    ),
    "converter-pptx": (("markitdown", "python-pptx"),),
}


def compare_snapshot(case: str, before: dict, after: dict) -> list[str]:
    """Compare every captured field, with one narrow provenance exception."""
    a, b = dict(before), dict(after)
    ma, mb = dict(a.get("metadata", {})), dict(b.get("metadata", {}))
    transition = ENGINE_TRANSITIONS.get(case)
    if transition and (ma.get("converter"), mb.get("converter")) in transition:
        ma["converter"] = mb["converter"] = "reviewed engine transition"
        a["metadata"], b["metadata"] = ma, mb
    return [
        f"{case}: {key} changed"
        for key in sorted(a.keys() | b.keys())
        if key not in a or key not in b or a[key] != b[key]
    ]


def file_snapshot(path: Path) -> dict:
    # Missing output is a failed audit, never an empty successful attachment.
    data = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def worker(path: Path, output: Path) -> dict:
    from loguru import logger

    logger.remove()
    from markitai.config import MarkitaiConfig
    from markitai.utils import frontmatter

    # Control the clock input rather than discard generated metadata fields.
    frontmatter.frontmatter_timestamp = lambda: "2026-09-14T00:00:00+00:00"
    from markitai.api import _close_loop_bound_resources, aconvert
    from markitai.converter.base import get_converter

    config = MarkitaiConfig()
    config.llm.enabled = config.ocr.enabled = config.screenshot.enabled = False
    config.image.alt_enabled = config.image.desc_enabled = False
    config.cache.enabled = False
    config.history.record = False
    config.fetch.remote_consent = "never"
    converter = get_converter(path, config=config)
    if converter is None:
        raise ValueError(f"No converter for {path}")
    converted = converter.convert(path, output / "converter")
    images = []
    for item in converted.images:
        data = item.data if item.data is not None else item.path.read_bytes()
        images.append(
            {
                "index": item.index,
                "original_name": item.original_name,
                "mime_type": item.mime_type,
                "width": item.width,
                "height": item.height,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    async def run() -> dict:
        try:
            result = await aconvert(path, output_dir=output / "api", config=config)
            return {
                "markdown": result.markdown,
                "llm_markdown": result.llm_markdown,
                "frontmatter": result.frontmatter,
                "assets": [file_snapshot(p) for p in result.assets],
                "screenshots": [file_snapshot(p) for p in result.screenshots],
                "images": result.images,
                "skip_reason": result.skip_reason,
                "written": result.output_path.read_text()
                if result.output_path is not None
                else None,
            }
        finally:
            await _close_loop_bound_resources()

    return {
        "converter": {
            "markdown": converted.markdown,
            "metadata": converted.metadata,
            "images": images,
        },
        "api": asyncio.run(run()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path)
    parser.add_argument(
        "--after-source", type=Path, default=REPO / "packages/markitai/src"
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args.worker.resolve(), args.output)
        (args.output / "snapshot.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return
    if not args.before_source or not args.inputs:
        parser.error("--before-source and --inputs are required")
    from audit_performance import FORMATS

    issues, manifest, cases = [], {}, []
    for ext in FORMATS:
        path = (args.inputs / f"sample.{ext}").resolve()
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshots = []
        for label, source in [
            ("before", args.before_source),
            ("after", args.after_source),
        ]:
            target = args.output / label / ext
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(path),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                env={
                    **os.environ,
                    "PYTHONPATH": str(source.resolve()),
                    "MARKITAI_NO_REMOTE_FETCH": "1",
                    "MARKITAI_RECORD_HISTORY": "0",
                },
            )
            (args.output / f"{label}-{ext}.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                issues.append(f"{label}-{ext}: failed conversion")
                snapshots.append(None)
            else:
                # Only normalize the output directory, whose absolute path is
                # intentionally different for independent runs.
                text = (
                    (target / "snapshot.json")
                    .read_text()
                    .replace(str(target), "<OUTPUT>")
                )
                snapshots.append(json.loads(text))
        for mode in ("converter", "api"):
            case = f"{mode}-{ext}"
            cases.append(case)
            if all(s is not None for s in snapshots):
                issues.extend(
                    compare_snapshot(case, snapshots[0][mode], snapshots[1][mode])
                )
        print(ext, flush=True)
    report = {
        "schema": 1,
        "inputs": manifest,
        "cases": cases,
        "issues": issues,
        "passed": not issues,
        "engine_transitions": ENGINE_TRANSITIONS,
    }
    (args.output / "quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()

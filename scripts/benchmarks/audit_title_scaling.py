"""Paired metadata/cleanup and complete API timings for long Markdown text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--operation", choices=["title", "normalization"], default="title"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        from markitai.utils.frontmatter import extract_title_from_content
        from markitai.utils.markdown_quality import normalize_markdown

        body = args.worker.read_text()
        operation = (
            extract_title_from_content
            if args.operation == "title"
            else normalize_markdown
        )
        times = []
        for _ in range(7):
            start = time.perf_counter()
            result = operation(body)
            times.append((time.perf_counter() - start) * 1000)
        details = (
            {"title": result}
            if args.operation == "title"
            else {
                "sha256": hashlib.sha256(result.encode()).hexdigest(),
                "chars": len(result),
            }
        )
        print(json.dumps({**details, "median_ms": statistics.median(times[1:])}))
        return
    if not args.before_source:
        parser.error("--before-source is required")
    records = []
    for size in (100, 1000, 10000, 100000):
        path = (args.output / f"rows-{size}.txt").resolve()
        path.write_text(
            "## Sheet1\n\n| name | value |\n| --- | --- |\n"
            + "".join(
                f"| Row {i} | Some long text with 中文 and a link [source](https://example.com) |\n"
                for i in range(size)
            )
        )
        for repeat in range(3):
            sources = [
                ("before", args.before_source),
                ("after", REPO / "packages/markitai/src"),
            ]
            for label, source in sources[:: 1 if repeat % 2 == 0 else -1]:
                env = {
                    **os.environ,
                    "PYTHONPATH": str(source.resolve()),
                    "MARKITAI_NO_REMOTE_FETCH": "1",
                    "MARKITAI_RECORD_HISTORY": "0",
                }
                r = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        str(path),
                        "--output",
                        str(args.output),
                        "--operation",
                        args.operation,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    timeout=60,
                )
                title = json.loads(r.stdout)
                r = subprocess.run(
                    [
                        sys.executable,
                        str(REPO / "scripts/benchmarks/audit_performance.py"),
                        "--worker",
                        "api",
                        "--input",
                        str(path),
                        "--output",
                        str(args.output / "work"),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    timeout=120,
                )
                detail = next(
                    json.loads(line.removeprefix("AUDIT_RESULT="))
                    for line in r.stdout.splitlines()
                    if line.startswith("AUDIT_RESULT=")
                )
                records.append(
                    {
                        "rows": size,
                        "engine": label,
                        "repeat": repeat,
                        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        args.operation: title,
                        **detail,
                    }
                )
                (args.output / "scaling.json").write_text(json.dumps(records, indent=2))
                print(size, label, repeat, flush=True)
        group = [r for r in records if r["rows"] == size]
        assert len({i["sha256"] for r in group for i in r["iterations"]}) == 1
        if args.operation == "title":
            assert {r["title"]["title"] for r in group} == {"Sheet1"}
        else:
            assert len({r["normalization"]["sha256"] for r in group}) == 1


if __name__ == "__main__":
    main()

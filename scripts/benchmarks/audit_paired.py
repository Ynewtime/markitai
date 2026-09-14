"""Run the fixed audit case by case on both source trees to reduce host drift."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from audit_performance import FORMATS, REPO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ["cli-version", "cli-help"]
    for ext in FORMATS:
        cases.extend([f"converter-{ext}", f"api-{ext}"])
        if ext in {"txt", "html", "docx", "xlsx", "pptx", "pdf"}:
            cases.append(f"cli-{ext}")
    cases.append("cli-batch-30")
    for index, case in enumerate(cases):
        sources = [
            ("before", args.before_source),
            ("after", REPO / "packages/markitai/src"),
        ]
        for label, source in sources[:: 1 if index % 2 == 0 else -1]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/benchmarks/audit_performance.py"),
                    "--source",
                    str(source.resolve()),
                    "--output",
                    str((args.output / label).resolve()),
                    "--repeats",
                    "3",
                    "--only",
                    case,
                    "--append",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=240,
            )
            (args.output / f"{case}-{label}.log").write_text(
                result.stdout + result.stderr
            )
            assert result.returncode == 0, f"Audit failed for {case}/{label}"
            print(label, result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()

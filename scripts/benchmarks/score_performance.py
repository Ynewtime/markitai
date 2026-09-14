"""Score three equally weighted workloads; fail closed on missing/changed output."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def body_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for path in directory.glob("*.md"):
        text = path.read_text()
        if text.startswith("---\n"):
            text = text.partition("\n---\n")[2]
        result[path.name] = hashlib.sha256(text.encode()).hexdigest()
    return result


def conversion_hashes(row: dict) -> set[str]:
    return {i["sha256"] for run in row["runs"] for i in run["detail"]["iterations"]}


def quality_issues(report: dict | None, inputs: dict, cases: set[str]) -> list[str]:
    if report is None:
        return ["Quality evidence missing; run audit_quality.py"]
    issues = list(report.get("issues", []))
    if report.get("schema") != 1 or report.get("passed") is not True:
        issues.append("Quality audit did not pass")
    if report.get("inputs") != inputs:
        issues.append("Quality audit inputs do not match timed inputs")
    if set(report.get("cases", [])) != cases:
        issues.append("Quality audit cases incomplete or changed")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--urls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality", type=Path, help="audit_quality.py quality.json report"
    )
    args = parser.parse_args()
    before = {
        r["case"]: r for r in json.loads((args.before / "audit.json").read_text())
    }
    after = {r["case"]: r for r in json.loads((args.after / "audit.json").read_text())}
    inputs = json.loads((args.before / "inputs.json").read_text())
    issues = quality_issues(
        json.loads(args.quality.read_text()) if args.quality else None,
        inputs,
        {name for name in before if name.startswith(("converter-", "api-"))},
    )
    if inputs != json.loads((args.after / "inputs.json").read_text()):
        issues.append("File inputs changed")
    if before.keys() != after.keys():
        issues.append("Audit cases changed")
    if len(before) != 49:
        issues.append("Expected 49 audit cases")
    before_batch = body_hashes(args.before / "batch-output")
    if not before_batch or before_batch != body_hashes(args.after / "batch-output"):
        issues.append("Batch outputs missing or Markdown changed")
    ratios = {}
    for name in sorted(before.keys() & after.keys()):
        a, b = before[name], after[name]
        if min(len(a["runs"]), len(b["runs"])) < 3 or any(
            r.get("exit") != 0 for r in [*a["runs"], *b["runs"]]
        ):
            issues.append(f"{name}: failed run")
            continue
        ratios[name] = a["median_seconds"] / b["median_seconds"]
        if name.startswith(("converter-", "api-")) and conversion_hashes(
            a
        ) != conversion_hashes(b):
            issues.append(f"{name}: Markdown changed")
        if name.startswith("converter-"):
            counts = [
                {
                    i.get("images")
                    for run in row["runs"]
                    for i in run["detail"]["iterations"]
                }
                for row in (a, b)
            ]
            if None in counts[0] or None in counts[1] or counts[0] != counts[1]:
                issues.append(f"{name}: image count changed or missing")
    urls = json.loads((args.urls / "urls.json").read_text())
    if {r["case"] for r in urls} != {
        "author-contact-block",
        "general--wikipedia",
        "table-layout--paulgraham.com-makersschedule",
        "math--katex",
        "short-note",
    }:
        issues.append("URL workload changed")
    url_ratios = {}
    for name in sorted({r["case"] for r in urls}):
        groups = [
            [r for r in urls if r["case"] == name and r["engine"] == engine]
            for engine in ["before", "after"]
        ]
        if any(
            len(g) < 3 or any(r["exit"] or not r["chars"] for r in g) for g in groups
        ):
            issues.append(f"URL {name}: incomplete or failed")
            continue
        if len({r["markdown_sha256"] for g in groups for r in g}) != 1:
            issues.append(f"URL {name}: Markdown changed")
        url_ratios[name] = statistics.median(
            r["seconds"] for r in groups[0]
        ) / statistics.median(r["seconds"] for r in groups[1])
    groups = {
        "url": statistics.geometric_mean(url_ratios.values()) if url_ratios else None,
        "office_pdf": statistics.geometric_mean(
            ratios[f"cli-{ext}"] for ext in ["docx", "xlsx", "pptx", "pdf"]
        )
        if all(f"cli-{ext}" in ratios for ext in ["docx", "xlsx", "pptx", "pdf"])
        else None,
        "batch": ratios.get("cli-batch-30"),
    }
    score = statistics.geometric_mean(groups.values()) if all(groups.values()) else None
    report = {
        "weights": {g: 1 / 3 for g in groups},
        "groups": groups,
        "composite_speedup": score,
        "target": 10,
        "target_met": not issues and score is not None and score >= 10,
        "issues": issues,
        "url_speedups": url_ratios,
        "diagnostic_speedups": ratios,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()

"""Compare built local defuddle with native Markitai on identical offline HTML.

No LLM scoring or network calls. Token overlap is a diagnostic, not a quality
verdict: richer discussion extraction can legitimately differ from the reference.
Raw outputs and per-case times are retained so differences can be reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fixture_url(html: str, name: str) -> str:
    match = re.match(r"\s*<!--\s*(\{.*?\})\s*-->", html, re.DOTALL)
    if match:
        try:
            url = json.loads(match[1]).get("url")
            if isinstance(url, str):
                return url
        except ValueError:
            pass
    for pattern in (
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)',
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)',
    ):
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match[1]
    return f"https://{name.split('--', 1)[-1]}"


def tokens(markdown: str) -> Counter[str]:
    # Retain link labels, omit destinations, and compare CJK character units.
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
    text = re.sub(r"https?://[^\s<>]+", "", text)
    text = re.sub(r"([\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])", r" \1 ", text)
    return Counter(re.findall(r"\w+", text.lower()))


def overlap(reference: str, actual: str) -> dict[str, float]:
    expected, observed = tokens(reference), tokens(actual)
    matched = sum((expected & observed).values())
    return {
        "token_recall": matched / max(1, sum(expected.values())),
        "token_precision": matched / max(1, sum(observed.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defuddle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=REPO / "packages/markitai/tests/defuddle_fixtures/fixtures",
    )
    parser.add_argument("--extra-fixture-dir", type=Path)
    parser.add_argument(
        "--markitai-source", type=Path, default=REPO / "packages/markitai/src"
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.markitai_source.resolve()))
    from loguru import logger
    from markitai.fetch import detect_js_required
    from markitai.webextract import extract_web_content, is_native_extraction_acceptable

    logger.remove()
    supports_native_gate = (
        "allow_short_content" in inspect.signature(detect_js_required).parameters
    )
    roots = [args.fixture_dir]
    if args.extra_fixture_dir:
        roots.append(args.extra_fixture_dir)
    cases = []
    expected = {}
    for root in roots:
        for path in sorted(root.glob("*.html")):
            html = path.read_text()
            cases.append(
                {
                    "name": path.stem,
                    "path": str(path.resolve()),
                    "url": fixture_url(html, path.stem),
                    "bytes": len(html.encode()),
                    "sha256": hashlib.sha256(html.encode()).hexdigest(),
                }
            )
            golden = root.parent / "expected" / (path.stem + ".md")
            if golden.exists():
                expected[path.stem] = re.sub(
                    r"\A```json\n.*?\n```\s*",
                    "",
                    golden.read_text(),
                    count=1,
                    flags=re.DOTALL,
                )
    if not cases:
        parser.error("No HTML fixtures found")
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(cases, indent=2))
    reference = subprocess.run(
        [
            "bun",
            str(Path(__file__).with_name("defuddle_runner.mjs")),
            str(args.defuddle_root.resolve()),
            str(manifest.resolve()),
            str(args.iterations),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    defuddle = json.loads(reference.stdout)
    (args.output / "defuddle.json").write_text(
        json.dumps(defuddle, ensure_ascii=False, indent=2)
    )
    extract_web_content(
        "<article><p>Warm up the extraction pipeline.</p></article>",
        "https://example.com",
    )
    native = []
    for item in cases:
        html = Path(item["path"]).read_text()
        times = []
        try:
            for _ in range(args.iterations):
                start = time.perf_counter()
                result = extract_web_content(html, item["url"])
                times.append((time.perf_counter() - start) * 1000)
            native.append(
                {
                    "name": item["name"],
                    "milliseconds": times,
                    "markdown": result.markdown,
                    "title": result.metadata.title,
                    "author": result.metadata.author,
                    "published": result.metadata.published,
                    "diagnostics": result.diagnostics,
                    "accepted": is_native_extraction_acceptable(result),
                    "would_request_browser": detect_js_required(
                        result.markdown,
                        **(
                            {
                                "allow_short_content": is_native_extraction_acceptable(
                                    result
                                )
                            }
                            if supports_native_gate
                            else {}
                        ),
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain per-fixture failures in the report
            native.append(
                {"name": item["name"], "milliseconds": times, "error": str(exc)}
            )
    (args.output / "markitai.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2, default=str)
    )
    rows = []
    for item, a, b in zip(cases, defuddle, native, strict=True):
        row = {"name": item["name"], "bytes": item["bytes"]}
        for name, result in (("defuddle", a), ("markitai", b)):
            row[name + "_ms"] = (
                statistics.median(result["milliseconds"])
                if result["milliseconds"]
                else None
            )
            if "error" in result:
                row[name + "_error"] = result["error"]
        if "error" not in a and "error" not in b:
            row.update(overlap(a["markdown"], b["markdown"]))
            row["exact_markdown"] = a["markdown"].strip() == b["markdown"].strip()
            row["markitai_accepted"] = b["accepted"]
            row["markitai_would_request_browser"] = b["would_request_browser"]
            if item["name"] in expected:
                row["defuddle_vs_golden"] = overlap(
                    expected[item["name"]], a["markdown"]
                )
                row["markitai_vs_golden"] = overlap(
                    expected[item["name"]], b["markdown"]
                )
        rows.append(row)
    summary = {"cases": len(rows), "iterations": args.iterations, "rows": rows}
    for name in ("defuddle", "markitai"):
        times = sorted(
            row[name + "_ms"] for row in rows if row[name + "_ms"] is not None
        )
        summary[name] = {
            "median_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))],
            "sum_median_ms": sum(times),
            "errors": sum(name + "_error" in row for row in rows),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

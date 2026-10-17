#!/usr/bin/env python3
"""Compare local OCR engines on the parity corpus's scans (standard library only).

    python3 scripts/bench_ocr_backends.py --bin path/to/markitai --corpus path/to/corpus/ocr \
        [--runs 3] [--configs vision,vision-detect,paddle] [--out bench-out]

Configurations (each is the same binary with other environment variables):
  vision         MARKITAI_OCR_BACKEND=vision            (the default reading policy: en, then zh, ko, ja, detected)
  vision-detect  MARKITAI_OCR_BACKEND=vision MARKITAI_OCR_VISION_POLICY=detect
                 (one request for all languages with detection)
  paddle         MARKITAI_OCR_BACKEND=paddle            (tract PP-OCR; models installed first with `doctor --fix`)

For each configuration and file it reports the wall time of the first run in a fresh
process with a fresh cache (cold) and the median of --runs later runs (warm), seconds
per page, and how much of the expected text came out: lines found alone on a line,
and lines found anywhere. "Cold" for Vision copies the binary to a new path, because
Vision compiles its models into ~/Library/Caches/<binary>/com.apple.e5rt.e5bundlecache
for each new binary; for paddle it is the first process after the models are installed.
Every run uses its own MARKITAI_HOME and HOME-independent state; nothing is sent anywhere.
The outputs are kept in --out so the text can be read next to the scores.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

FILES = {
    # name: (pages, expected lines)
    "cjk-scanned.pdf": (1, [
        "第一章概述",
        "本文档用于测试中文文本提取表格识别和光学字符识别",
        "Markitai将各种文档格式转换为Markdown并可选地使用大语言模型进行增强",
        "日本語のテキストも含まれています東京は日本の首都です",
        "한국어문장도포함합니다서울은대한민국의수도입니다",
        "第二章数据表",
        "项目数量单价",
        "苹果123.50",
        "香蕉301.20",
        "结论所有字符都应该被正确保留没有乱码",
    ]),
    "scanned-sample-p1-3.pdf": (3, [
        "Loremipsum",
        "Maecenasmauris lectus,lobortisetpurusmattis,blanditdictumtellus.",
        "Maecenasnonloremquistellusplaceratvarius.",
        "Nullafacilisi.",
        "Aeneancongue fringilla justo ut aliquam.",
        "Morbiviverrasemperlorem necmolestie.",
    ]),
    "gen-photographed-scan.pdf": (1, [
        "Sitereport14",
        "Thenorthwallshowstwohairlinecracks.",
        "Moisturereadingsstayedbelow18percent.",
        "Recommendedaction:monitorforsixmonths.",
        "Inspector:J.Rivera,9October2026.",
    ]),
}

CONFIGS = {
    "vision": {"MARKITAI_OCR_BACKEND": "vision"},
    "vision-detect": {"MARKITAI_OCR_BACKEND": "vision", "MARKITAI_OCR_VISION_POLICY": "detect"},
    "paddle": {"MARKITAI_OCR_BACKEND": "paddle"},
}


def squeeze(text):
    """Letters and digits only (NFKC), so spacing, case-free punctuation and table bars do not matter."""
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum() or c == ".").lower()


def score(markdown, expected):
    body = [line for line in markdown.splitlines() if line.strip() and not line.startswith(("---", "title:", "source:", "markitai_processed", "![", "<!--"))]
    lines = [squeeze(line) for line in body]
    wanted = [squeeze(e) for e in expected]
    alone = sum(1 for w in wanted if any(w in l and sum(1 for o in wanted if o in l) == 1 for l in lines))
    anywhere = sum(1 for w in wanted if any(w in l for l in lines) or w in "".join(lines))
    return alone, anywhere, len(wanted), len(body)


def run(binary, env_extra, home, pdf, out, trace):
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "TMPDIR", "LANG", "LC_ALL")}
    env.update(env_extra)
    env["MARKITAI_HOME"] = home
    env["HOME"] = home
    env["MARKITAI_LANG"] = "en"
    if trace:
        env["MARKITAI_OCR_TRACE"] = "1"
    shutil.rmtree(out, ignore_errors=True)
    started = time.perf_counter()
    done = subprocess.run([binary, pdf, "--ocr", "--no-llm", "--no-cache", "-o", out], env=env,
                          capture_output=True, text=True, timeout=600)
    took = time.perf_counter() - started
    md = os.path.join(out, os.path.basename(pdf) + ".md")
    text = open(md, encoding="utf-8").read() if os.path.exists(md) else ""
    return took, text, done.returncode, done.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--corpus", required=True, help="the corpus's ocr directory")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--configs", default="vision,vision-detect,paddle")
    ap.add_argument("--out", default="bench-out")
    args = ap.parse_args()
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="ocr-bench-")
    rows = []
    try:
        for name in args.configs.split(","):
            env_extra = CONFIGS[name]
            home = os.path.join(work, "home-" + name)
            os.makedirs(home)
            binary = os.path.abspath(args.bin)
            if name.startswith("vision"):
                # A new path is a new Vision cache: the first run of it is cold.
                binary = os.path.join(work, "bin-" + name + "-" + str(os.getpid()), "markitai")
                os.makedirs(os.path.dirname(binary))
                shutil.copy2(args.bin, binary)
            if name == "paddle":
                subprocess.run([binary, "doctor", "--fix"], env={**os.environ, "MARKITAI_HOME": home, "HOME": home},
                               capture_output=True, text=True, timeout=900)
            first = True
            for file, (pages, expected) in FILES.items():
                pdf = os.path.join(args.corpus, file)
                out = os.path.join(out_root, name, file)
                times = []
                text = ""
                for index in range(1 + args.runs):
                    took, text, code, err = run(binary, env_extra, home, pdf, os.path.join(work, "run"), trace=index == 1)
                    if code != 0:
                        print(f"{name} {file}: exit {code}\n{err[-400:]}", file=sys.stderr)
                    if index == 1:
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        open(out + ".trace.txt", "w", encoding="utf-8").write(err)
                    times.append(took)
                    if index == 0:
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        open(out + ".md", "w", encoding="utf-8").write(text)
                cold, warm = times[0], statistics.median(times[1:])
                alone, anywhere, total, lines = score(text, expected)
                rows.append({"config": name, "file": file, "pages": pages,
                             "cold_s": round(cold, 2) if first else None, "first_in_config": first,
                             "file_first_run_s": round(cold, 2), "warm_median_s": round(warm, 2),
                             "warm_s_per_page": round(warm / pages, 2),
                             "lines_alone": f"{alone}/{total}", "lines_anywhere": f"{anywhere}/{total}",
                             "output_lines": lines})
                first = False
    finally:
        shutil.rmtree(work, ignore_errors=True)
    json.dump(rows, open(os.path.join(out_root, "results.json"), "w"), indent=1, ensure_ascii=False)
    width = max(len(r["file"]) for r in rows)
    print(f"{'config':14} {'file':{width}} pages  cold(first file) first-run  warm  s/page  alone  anywhere")
    for r in rows:
        print(f"{r['config']:14} {r['file']:{width}} {r['pages']:5}  "
              f"{(str(r['cold_s']) + 's') if r['cold_s'] is not None else '-':>16} {r['file_first_run_s']:>8}s {r['warm_median_s']:>5}s {r['warm_s_per_page']:>6}  "
              f"{r['lines_alone']:>5}  {r['lines_anywhere']:>8}")
    print(f"\nresults.json, the texts (*.md) and the traces (*.trace.txt) are in {out_root}")


if __name__ == "__main__":
    main()

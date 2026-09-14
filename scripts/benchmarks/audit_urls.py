"""Paired cold URL CLI timings against a fixed loopback HTTP origin.

Network transport is real but internet latency and remote-service caches are
excluded. Keep live-site measurements separate from this reproducible metric.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASES = (
    "author-contact-block",
    "general--wikipedia",
    "table-layout--paulgraham.com-makersschedule",
    "math--katex",
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument(
        "--after-source", type=Path, default=REPO / "packages/markitai/src"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pages = output / "pages"
    pages.mkdir(exist_ok=True)
    for name in CASES:
        fixture = (
            REPO / "packages/markitai/tests/defuddle_fixtures/fixtures" / f"{name}.html"
        )
        (pages / fixture.name).write_bytes(fixture.read_bytes())
    (pages / "short-note.html").write_text(
        "<html><head><title>短笔记</title></head><body><article><p>"
        "这是一条完整的中文笔记。保留上下文很重要。"
        "</p></article></body></html>"
    )
    config = output / "config.json"
    config.write_text(
        json.dumps(
            {
                "llm": {"enabled": False},
                "fetch": {"remote_consent": "never"},
                "history": {"record": False},
            }
        )
    )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pages.glob("*.html")
    }
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    handler = functools.partial(QuietHandler, directory=str(pages))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for name in (*CASES, "short-note"):
                url = f"http://127.0.0.1:{server.server_port}/{name}.html"
                for repeat in range(args.repeats):
                    # Alternate order to reduce time-of-run bias.
                    sources = [
                        ("before", args.before_source),
                        ("after", args.after_source),
                    ]
                    for engine, source in sources[:: 1 if repeat % 2 == 0 else -1]:
                        env = os.environ.copy()
                        env.update(
                            PYTHONPATH=str(source.resolve()),
                            MARKITAI_NO_REMOTE_FETCH="1",
                            MARKITAI_RECORD_HISTORY="0",
                            NO_PROXY="localhost,127.0.0.1",
                        )
                        command = [
                            sys.executable,
                            "-m",
                            "markitai",
                            url,
                            "--preset",
                            "minimal",
                            "--no-cache",
                            "--no-remote-fetch",
                            "--config",
                            str(config),
                        ]
                        start = time.perf_counter()
                        result = subprocess.run(
                            command,
                            check=False,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=60,
                            cwd=output,
                        )
                        elapsed = time.perf_counter() - start
                        markdown = result.stdout
                        if markdown.startswith("---\n"):
                            markdown = markdown.partition("\n---\n")[2]
                        stem = f"{name}-{engine}-{repeat}"
                        (output / f"{stem}.stdout").write_text(result.stdout)
                        (output / f"{stem}.stderr").write_text(result.stderr)
                        rows.append(
                            {
                                "case": name,
                                "engine": engine,
                                "repeat": repeat,
                                "seconds": elapsed,
                                "exit": result.returncode,
                                "stdout_sha256": hashlib.sha256(
                                    result.stdout.encode()
                                ).hexdigest(),
                                "markdown_sha256": hashlib.sha256(
                                    markdown.encode()
                                ).hexdigest(),
                                "chars": len(result.stdout),
                            }
                        )
                        (output / "urls.json").write_text(json.dumps(rows, indent=2))
                        print(stem, result.returncode, round(elapsed, 4), flush=True)
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()

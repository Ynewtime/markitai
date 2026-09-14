"""Measure local HTML/file/stdin CLI behavior without model or remote services.

Each engine gets a separate loopback origin to prevent SPA-cache contamination.
Use --before-source to compare a saved Markitai source directory as well.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defuddle-root", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-source", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = args.output / "pages"
    pages.mkdir(exist_ok=True)
    html = args.html.read_text()
    (pages / "note.html").write_text(html)
    (pages / "short.html").write_text(
        "<html><head><title>Short note</title></head><body><article><p>A short but complete note.</p></article></body></html>"
    )
    cli = str(args.defuddle_root.resolve() / "dist/cli.js")
    commands = {
        "defuddle-node": ["node", cli, "parse"],
        "defuddle-bun": ["bun", cli, "parse"],
        "markitai-after": [sys.executable, "-m", "markitai"],
    }
    if args.before_source:
        commands["markitai-before"] = [sys.executable, "-m", "markitai"]
    rows = []
    for engine, command in commands.items():
        handler = functools.partial(QuietHandler, directory=str(pages.resolve()))
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for case in ["file-note", "url-note", "url-short", "stdin-note"]:
                    source = (
                        str((pages / "note.html").resolve())
                        if case == "file-note"
                        else "-"
                        if case == "stdin-note"
                        else f"http://127.0.0.1:{server.server_port}/"
                        + ("short.html" if case == "url-short" else "note.html")
                    )
                    options = (
                        ["--markdown"]
                        if engine.startswith("defuddle")
                        else ["--no-cache", "--no-remote-fetch", "--preset", "minimal"]
                    )
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(
                        args.before_source.resolve()
                        if engine == "markitai-before"
                        else REPO / "packages/markitai/src"
                    )
                    env["NO_PROXY"] = env.get("NO_PROXY", "") + ",localhost,127.0.0.1"
                    start = time.perf_counter()
                    try:
                        result = subprocess.run(
                            [*command, source, *options],
                            check=False,
                            env=env,
                            text=True,
                            input=html if case == "stdin-note" else None,
                            capture_output=True,
                            timeout=25,
                            cwd=args.output,
                        )
                    except subprocess.TimeoutExpired:
                        rows.append({"case": case, "engine": engine, "timeout": True})
                        continue
                    for suffix, data in [
                        ("stdout", result.stdout),
                        ("stderr", result.stderr),
                    ]:
                        (args.output / f"{case}-{engine}.{suffix}").write_text(data)
                    row = {
                        "case": case,
                        "engine": engine,
                        "seconds": round(time.perf_counter() - start, 3),
                        "exit": result.returncode,
                        "bytes": len(result.stdout.encode()),
                        "short_body_present": "A short but complete note."
                        in result.stdout,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                server.shutdown()
                thread.join()
    (args.output / "cli-summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

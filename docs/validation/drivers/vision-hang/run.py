#!/usr/bin/env python3
"""How often macOS Vision stops for good: rounds of Vision tests, each run a
process of its own under a throwaway HOME, beside a build of this repository
as load. See "Known issue" in docs/ocr.md.

    cargo test -p markitai-core --lib --no-run
    python3 docs/validation/drivers/vision-hang/run.py \
        --binary target/debug/deps/markitai_core-HASH --rounds 100 --out out.jsonl

Each round runs, in turn so that what else the machine does falls on all alike:
`corrected`, a page of Chinese read as corrected English, which is what can stop;
`scan`, the test that reads the same page by the default policy; `all`, every
Vision test on 8 threads. A run is stuck when Vision reports it stopped responding,
or at --limit. --no-load leaves the build out. Prints runs and stuck per kind.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CORRECTED = "ocr::vision::tests::reading_chinese_as_corrected_english"
SCAN = "ocr::tests::vision_reads_a_degraded_scan_of_chinese_with_a_table_by_default"
KINDS = {
    "corrected": [CORRECTED, "--exact", "--ignored", "--test-threads=1"],
    "scan": [SCAN, "--exact", "--test-threads=1"],
    "all": ["ocr::tests::vision", "ocr::tests::the_default_language", "--test-threads=8"],
}


def run(binary, argv, scratch, limit):
    home = tempfile.mkdtemp(dir=scratch)
    env = dict(os.environ, HOME=home, MARKITAI_HOME=os.path.join(home, "state"))
    began = time.time()
    try:
        done = subprocess.run(
            [binary, *argv], env=env, capture_output=True, text=True, timeout=limit)
        code, output = done.returncode, done.stdout + done.stderr
        stuck = "Vision stopped responding" in output
    except subprocess.TimeoutExpired:
        code, stuck = "timeout", True
    shutil.rmtree(home, ignore_errors=True)
    return dict(code=code, stuck=stuck, secs=round(time.time() - began, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--limit", type=float, default=360)
    parser.add_argument("--no-load", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    binary = os.path.abspath(args.binary)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    load = None
    with tempfile.TemporaryDirectory(prefix="vision-hang-") as scratch:
        # The first use of a binary compiles the models, which is no hang.
        run(binary, KINDS["scan"], scratch, args.limit)
        if not args.no_load:
            target = os.path.join(scratch, "target")
            build = f'while :; do rm -rf "{target}"; cargo test -p markitai-core --lib --no-run; done'
            load = subprocess.Popen(
                ["sh", "-c", build], env=dict(os.environ, CARGO_TARGET_DIR=target),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        totals = {kind: [0, 0, 0] for kind in KINDS}
        try:
            for number in range(args.rounds):
                for kind, argv in KINDS.items():
                    row = run(binary, argv, scratch, args.limit)
                    row.update(kind=kind, round=number, load1=round(os.getloadavg()[0], 1))
                    with open(args.out, "a") as out:
                        out.write(json.dumps(row) + "\n")
                    totals[kind][0] += 1
                    totals[kind][1] += int(row["stuck"])
                    totals[kind][2] += int(row["code"] != 0 and not row["stuck"])
        finally:
            if load:
                os.killpg(load.pid, signal.SIGKILL)
    print("kind runs stuck failed")
    for kind, (runs, stuck, failed) in totals.items():
        print(kind, runs, stuck, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

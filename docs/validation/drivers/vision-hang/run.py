#!/usr/bin/env python3
"""Whether a Vision reader comes back from a recognition that stopped for good (see
docs/ocr.md): fresh readers, each sent N images at once, since only a process's first
requests can stop, beside a build of this repository as load.

    cargo build --release -p markitai-cli --bin markitai
    python3 docs/validation/drivers/vision-hang/run.py --binary target/release/markitai \
        --image PAGE.png --language fr --minutes 10 --out out

PAGE.png is a page of several lines of Chinese. A reader with requests unanswered and
no CPU for 5 seconds, while no model is compiling, has its threads sampled into
out/stopped-PID.sample.txt and is killed, as the program would, and a fresh one gets
what it left. The stopped ones should show VNCRImageReaderDetector and
semaphore_wait_trap, and their answers should be those of the readers that did not stop.
Prints the trials, how many stopped, and the distinct answers of each.
"""

import argparse
import collections
import ctypes
import hashlib
import json
import os
import pathlib
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

READER = "__markitai-vision-reader"
IDLE = 5.0
TRIES = 6


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16), ("user", ctypes.c_uint64), ("system", ctypes.c_uint64),
                ("rest", ctypes.c_uint64 * 32)]


libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def cpu(pid):
    usage = Usage()
    return None if libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0 else usage.user + usage.system


def compiling():
    listing = subprocess.run(["/bin/ps", "-A", "-o", "pcpu=,comm="], capture_output=True, text=True).stdout
    return any("ANECompilerService" in line and float(line.split()[0]) > 3 for line in listing.splitlines())


def exact(stream, count):
    data = stream.read(count)
    if data is None or len(data) != count:
        raise EOFError
    return data


def answer(stream):
    """One answer: its id, and its bytes after the id."""
    number, kind = struct.unpack("<QB", exact(stream, 9))
    body = bytearray([kind])

    def take(count):
        data = exact(stream, count)
        body.extend(data)
        return data

    def string():
        take(struct.unpack("<I", take(4))[0])

    if kind == 0:
        take(4)
        string()
        take(1)
        for _ in range(struct.unpack("<I", take(4))[0]):
            string()
            take(4 + 16 + 8)
    else:
        string()
    return number, bytes(body)


class Driver:
    def __init__(self, args, scratch):
        self.args = args
        self.scratch = scratch
        self.png = pathlib.Path(args.image).read_bytes()
        self.width, self.height = struct.unpack(">II", self.png[16:24])
        self.lock = threading.Lock()
        self.rows = open(pathlib.Path(args.out) / "trials.jsonl", "a")
        self.next = iter(range(10**9))
        self.stopped = threading.Event()

    def request(self, number):
        spelling = self.args.language.encode()
        return (struct.pack("<QI", number, len(spelling)) + spelling
                + struct.pack("<IIQ", self.width, self.height, len(self.png)) + self.png)

    def reader(self, numbers, home):
        """One reader over `numbers`: the answers it gave, and whether it stopped."""
        env = dict(os.environ, HOME=home, MARKITAI_HOME=os.path.join(home, "state"))
        process = subprocess.Popen([self.args.binary, READER], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        answers = {}

        def listen():
            try:
                while True:
                    number, body = answer(process.stdout)
                    answers[number] = body
            except EOFError:
                pass

        def send():
            try:
                for number in numbers:
                    process.stdin.write(self.request(number))
                process.stdin.flush()
            except OSError:
                pass

        listening = threading.Thread(target=listen, daemon=True)
        listening.start()
        threading.Thread(target=send, daemon=True).start()
        last, since, stopped = cpu(process.pid) or 0, time.time(), False
        while len(answers) < len(numbers) and process.poll() is None:
            time.sleep(0.25)
            used = cpu(process.pid)
            if used is None:
                break
            if used - last > 100_000:
                since = time.time()
            last = used
            if time.time() - since >= IDLE:
                if compiling():
                    since = time.time()
                    continue
                stopped = True
                target = pathlib.Path(self.args.out) / f"stopped-{process.pid}.sample.txt"
                subprocess.run(["sample", str(process.pid), "1", "-file", str(target)], capture_output=True)
                break
        process.kill()
        process.wait()
        listening.join(2)
        return answers, stopped

    def trial(self, number):
        home = tempfile.mkdtemp(dir=self.scratch)
        began = time.time()
        wanted = list(range(1, self.args.n + 1))
        answers, stopped = self.reader(wanted, home)
        tries = 1
        while len(answers) < self.args.n and tries < TRIES:
            more, _ = self.reader([n for n in wanted if n not in answers], home)
            answers.update(more)
            tries += 1
        shutil.rmtree(home, ignore_errors=True)
        row = dict(trial=number, stopped=stopped, tries=tries, complete=len(answers) == self.args.n,
                   secs=round(time.time() - began, 2), load=round(os.getloadavg()[0], 1),
                   answers=sorted({hashlib.sha1(body).hexdigest()[:12] for body in answers.values()}))
        with self.lock:
            self.rows.write(json.dumps(row) + "\n")
            self.rows.flush()
        return row

    def work(self, deadline, totals):
        while not self.stopped.is_set() and time.time() < deadline:
            row = self.trial(next(self.next))
            kind = "stopped" if row["stopped"] else "plain"
            with self.lock:
                totals["trials"] += 1
                totals[kind] += 1
                totals["incomplete"] += not row["complete"]
                for digest in row["answers"]:
                    totals[f"{kind} answer {digest}"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-load", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.binary = os.path.abspath(args.binary)
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    totals = collections.Counter()
    load = None
    with tempfile.TemporaryDirectory(prefix="vision-hang-") as scratch:
        driver = Driver(args, scratch)
        signal.signal(signal.SIGTERM, lambda *_: driver.stopped.set())
        if not args.no_load:
            target = os.path.join(scratch, "target")
            build = f'while :; do rm -rf "{target}"; cargo test -p markitai-core --lib --no-run; done'
            load = subprocess.Popen(["sh", "-c", build], env=dict(os.environ, CARGO_TARGET_DIR=target),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + args.minutes * 60
        try:
            threads = [threading.Thread(target=driver.work, args=(deadline, totals)) for _ in range(args.workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            if load:
                os.killpg(load.pid, signal.SIGKILL)
    for name, count in sorted(totals.items()):
        print(name, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())

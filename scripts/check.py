"""Run the native development gate without requiring a Unix shell.

`check.py` runs everything: formatting, the Rust tests, clippy and the helper
tests. `check.py --quick` runs only what the files changed since the branch
left main (committed or not) can affect: the Rust tests of the changed crates
and of the crates that depend on them (with nextest; `cargo test` runs them
all), clippy when any crate changed, and the helper tests (which take seconds). Use the full gate before a merge or a release.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from ci_packages import npm_command
from isolation import isolated_env
from web_deps import WEB, stale_web_dependencies

# Each interpreter names its own executable.
PROBES = {"python3": ["-c", "import sys; print(sys.executable)"], "node": ["-p", "process.execPath"]}
# The workspace crates with tests.
TESTED = {"markitai-core", "markitai-cli", "markitai-ffi"}
# Changes here can affect every crate.
EVERY_CRATE = ("Cargo.toml", "Cargo.lock", "vendor/", ".cargo/", "rust-toolchain", "clippy.toml", "rustfmt.toml",
               ".config/nextest.toml")
# Workbench sources: the helper tests build and test them; only the built dist/ is embedded in the CLI.
WEB_SOURCES = "crates/markitai-cli/src/server/web/"


def unshimmed_path(path):
    """Put real interpreters first: HOME-relative shims (mise, asdf) stop resolving once HOME moves."""
    directories = []
    for tool, arguments in PROBES.items():
        found = shutil.which(tool, path=path)
        if not found:
            continue
        try:
            real = subprocess.run([found, *arguments], capture_output=True, text=True,
                                  timeout=120, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if real and Path(real).resolve() != Path(found).resolve():
            directories.append(str(Path(real).parent))
    return os.pathsep.join(directories + ([path] if path else []))


def install_web_dependencies(root, environment):
    """Run `npm ci` for the workbench when its packages differ from package-lock.json.

    The workbench tests import them (Preact, Pele), so a checkout that has never
    run `npm ci`, or ran it before the lockfile changed, would otherwise fail
    them with module-not-found errors. Without Node the web tests skip.
    """
    node = shutil.which("node", path=environment["PATH"])
    stale = stale_web_dependencies(WEB) if node else []
    if not stale:
        return 0
    where = WEB.relative_to(root).as_posix()
    more = f" and {len(stale) - 1} more" if len(stale) > 1 else ""
    print(f"check.py: {where}/node_modules differs from package-lock.json ({stale[0]}{more}); running npm ci",
          flush=True)
    npm = shutil.which("npm", path=environment["PATH"])
    if not npm:
        print(f"check.py: npm is not on PATH; install Node.js with npm, or run `npm ci` in {where}", file=sys.stderr)
        return 1
    result = subprocess.run([*npm_command(npm, node), "ci"], cwd=WEB, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        print(f"check.py: npm ci failed in {where} (exit {result.returncode})", file=sys.stderr)
    return result.returncode


def test_skips(environment):
    """`--skip` filters for the cargo tests from MARKITAI_CHECK_SKIP (space-separated names)."""
    names = environment.get("MARKITAI_CHECK_SKIP", "").split()
    return ["--", *(argument for name in names for argument in ("--skip", name))] if names else []


def run_lane(steps, failed, skipped):
    """Run `steps` in order, even after one fails; record names in `failed` and `skipped`.

    Each step is `(name, action, needs)`: `action()` returns an exit code, and a
    step whose `needs` (an earlier step's name) failed is not run. Returns the
    first failure's exit code, or 0.
    """
    first = 0
    for name, action, needs in steps:
        if needs in failed:
            print(f"check.py: not running {name}: {needs} failed", file=sys.stderr, flush=True)
            skipped.append(name)
            continue
        started = time.monotonic()
        code = action()
        print(f"check.py: {name} {'failed (exit %s)' % code if code else 'passed'} in "
              f"{time.monotonic() - started:.1f} s", flush=True)
        if code:
            failed.append(name)
            first = first or code
    return first


def summarize(failed, skipped):
    if failed:
        summary = f"check.py: failed: {', '.join(failed)}"
        if skipped:
            summary += f"; not run: {', '.join(skipped)}"
        print(summary, file=sys.stderr, flush=True)


def run_steps(steps):
    """Run every step, even after one fails, and name the failed ones last.

    Returns the first failure's exit code, or 0.
    """
    failed, skipped = [], []
    first = run_lane(steps, failed, skipped)
    summarize(failed, skipped)
    return first


class _ThreadOutput(io.TextIOBase):
    """sys.stdout/sys.stderr stand-in: a thread with a buffer writes there, others to the stream."""

    def __init__(self, stream, buffers):
        self.stream, self.buffers = stream, buffers

    def write(self, text):
        buffer = self.buffers.get(threading.get_ident())
        return buffer.write(text) if buffer is not None else self.stream.write(text)

    def flush(self):
        if threading.get_ident() not in self.buffers:
            self.stream.flush()


def run_lanes(foreground, background):
    """Run two lanes at once and name every failure last, as `run_steps` does.

    The foreground lane writes as it goes. The background lane's actions must
    not let a child process write to the console (`command(..., quiet=True)`);
    what they print is held and shown, as one block, after the foreground lane.
    """
    failed, skipped, codes = [], [], {}
    buffer = io.StringIO()
    buffers = {}
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _ThreadOutput(saved[0], buffers), _ThreadOutput(saved[1], buffers)

    def lane():
        buffers[threading.get_ident()] = buffer
        try:
            codes["background"] = run_lane(background, background_failed, background_skipped)
        finally:
            buffers.pop(threading.get_ident(), None)

    background_failed, background_skipped = [], []
    thread = threading.Thread(target=lane, daemon=True)
    try:
        thread.start()
        codes["foreground"] = run_lane(foreground, failed, skipped)
        if thread.is_alive():
            print("check.py: waiting for " + ", ".join(name for name, _, _ in background), flush=True)
        thread.join()
    finally:
        sys.stdout, sys.stderr = saved
    sys.stdout.write(buffer.getvalue())
    sys.stdout.flush()
    failed += background_failed
    skipped += background_skipped
    summarize(failed, skipped)
    return codes.get("foreground") or codes.get("background", 1)


# A helper test file with more tests than this runs in parts of at most PART
# tests, each in an interpreter of its own, so no single file is the tail.
SPLIT_AT, PART = 16, 8


def _parts(names):
    """Test IDs in parts of at most PART: a class that fits stays whole (its tests may
    rely on running one after another), a larger one is dealt in turn across parts
    so slow neighbours spread, and small classes share parts."""
    classes = {}
    for name in names:
        classes.setdefault(name.rsplit(".", 1)[0], []).append(name)
    units = []
    for members in classes.values():
        count = -(-len(members) // PART)
        units += [members[index::count] for index in range(count)]
    parts = []
    for unit in sorted(units, key=len, reverse=True):
        part = next((part for part in parts if len(part) + len(unit) <= PART), None)
        if part is None:
            parts.append(list(unit))
        else:
            part.extend(unit)
    return parts


def helper_parts(root, environment, files):
    """`(file, -k patterns or None)` work items, the parts of large files first.

    A file is split by the test IDs unittest's loader lists (inherited tests
    included); each part selects its tests by exact `-k` patterns.
    """
    large = [path for path in files
             if len(re.findall(r"^\s+def test", path.read_text(encoding="utf-8"), re.MULTILINE)) > SPLIT_AT]

    def listed(path):
        code = ("import sys, unittest\n"
                "def ids(suite):\n"
                "    for item in suite:\n"
                "        yield from ids(item) if isinstance(item, unittest.TestSuite) else [item.id()]\n"
                f"print('\\n'.join(ids(unittest.defaultTestLoader.discover('scripts', pattern={path.name!r}))))")
        result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        names = result.stdout.split()
        # A file that does not load runs whole, and reports its own error.
        return path, (names if result.returncode == 0 and names else None)

    items = []
    with ThreadPoolExecutor(max_workers=max(1, len(large))) as pool:
        for path, names in pool.map(listed, large):
            if names is None:
                continue
            for part in _parts(names):
                items.append((path, ["*." + name.split(".", 1)[1] for name in part]))
    split = {path for path, _ in items}
    return items + [(path, None) for path in files if path not in split]


def helper_tests(root, environment, jobs=None):
    """The `scripts/test_*.py` suites, one interpreter per file (or part of a large file), several at once.

    Each runs as `unittest discover` would run it from the repository root; a
    failing one's whole output is shown, a passing file gets a line.
    """
    files = sorted((root / "scripts").glob("test_*.py"))
    items = helper_parts(root, environment, files)

    def run(item):
        path, patterns = item
        started = time.monotonic()
        selection = [argument for pattern in patterns or [] for argument in ("-k", pattern)]
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", path.name,
                                 *selection], cwd=root, env=environment, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        return item, result, time.monotonic() - started

    total, code = 0, 0
    summary = {}  # file name: [tests run, skipped, parts, slowest part's seconds, tests expected or None]
    with ThreadPoolExecutor(max_workers=jobs or min(len(items), max(2, os.cpu_count() or 2))) as pool:
        for (path, patterns), result, seconds in pool.map(run, items):
            output = result.stdout + result.stderr
            ran = re.search(r"^Ran (\d+) tests?", output, re.MULTILINE)
            skipped = re.search(r"\(skipped=(\d+)\)", output)
            count = int(ran.group(1)) if ran else 0
            total += count
            row = summary.setdefault(path.name, [0, 0, 0, 0.0, 0 if patterns else None])
            row[0] += count
            row[1] += int(skipped.group(1)) if skipped else 0
            row[2] += 1
            row[3] = max(row[3], seconds)
            if patterns:
                row[4] += len(patterns)
            if result.returncode:
                code = code or result.returncode
                part = f" (part {row[2]})" if patterns else ""
                print(f"--- {path.name}{part} failed (exit {result.returncode}) in {seconds:.1f} s\n{output}",
                      flush=True)
    for name, (count, skipped, parts, seconds, expected) in sorted(summary.items()):
        if expected is not None and count != expected:
            code = code or 1
            print(f"--- {name}: its {parts} parts ran {count} of its {expected} tests", flush=True)
        detail = f", {skipped} skipped" if skipped else ""
        timing = f"in {parts} parts, the slowest {seconds:.1f} s" if parts > 1 else f"in {seconds:.1f} s"
        print(f"{name}: {count} tests{detail} {timing}", flush=True)
    print(f"helper tests: {total} tests in {len(files)} files, {'FAILED' if code else 'OK'}", flush=True)
    return code


def nextest_available(environment, root):
    """Whether `cargo nextest` runs here; MARKITAI_CHECK_RUNNER=cargo|nextest decides instead when set."""
    chosen = environment.get("MARKITAI_CHECK_RUNNER", "").strip().lower()
    if chosen in {"cargo", "nextest"}:
        return chosen == "nextest"
    try:
        return subprocess.run(["cargo", "nextest", "--version"], cwd=root, env=environment,
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def test_command(packages, nextest, environment):
    """The Rust test command: every test binary runs even after a failure, so one run reports them all.

    nextest runs each test in a process of its own, all binaries at once; it
    has no doc tests, and the workspace has none (the libraries set doctest =
    false). It builds the whole workspace, so dependencies keep the features
    the full gate builds them with, and runs only the tests of `packages`
    (all when None); `cargo test` runs them all.
    """
    if not nextest:
        return ["cargo", "test", "--workspace", "--locked", "--no-fail-fast", *test_skips(environment)]
    selection = [] if packages is None else ["-E", " | ".join(f"package({package})" for package in packages)]
    return ["cargo", "nextest", "run", "--workspace", "--locked", "--no-fail-fast", *selection,
            *test_skips(environment)]


def merge_base(root, base):
    for candidate in ([base] if base else ["origin/main", "main"]):
        found = subprocess.run(["git", "merge-base", "HEAD", candidate], cwd=root, capture_output=True, text=True)
        if found.returncode == 0:
            return found.stdout.strip()
    raise SystemExit(f"check.py --quick: no merge base with {base or 'origin/main or main'}; pass --base REF")


def changed_files(root, base):
    """Files that differ from the merge base with `base` (committed, staged, unstaged or untracked)."""
    def lines(*arguments):
        return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True,
                              check=True).stdout.splitlines()
    return sorted(set(lines("diff", "--name-only", merge_base(root, base)) +
                      lines("ls-files", "--others", "--exclude-standard")))


def workspace_crates(root):
    """`{name: (directory, [workspace crates it depends on])}` from `cargo metadata`."""
    metadata = json.loads(subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
                                         cwd=root, capture_output=True, text=True, check=True).stdout)
    names = {package["name"] for package in metadata["packages"]}
    return {package["name"]: (Path(package["manifest_path"]).parent.relative_to(root.resolve()).as_posix() + "/",
                              [dependency["name"] for dependency in package["dependencies"]
                               if dependency["name"] in names])
            for package in metadata["packages"]}


def affected_crates(files, crates):
    """The workspace crates `files` can affect (their own and every crate depending on them), or None for all."""
    changed = set()
    for file in files:
        if file.startswith(EVERY_CRATE) or file in EVERY_CRATE:
            return None
        if file.startswith(WEB_SOURCES) and not file.startswith(WEB_SOURCES + "dist/"):
            continue
        for name, (directory, _) in crates.items():
            if file.startswith(directory):
                changed.add(name)
    while True:
        dependents = {name for name, (_, needs) in crates.items() if changed.intersection(needs)} - changed
        if not dependents:
            return changed
        changed |= dependents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="only what the files changed since the branch left main can affect")
    parser.add_argument("--base", help="with --quick: compare with the merge base of HEAD and BASE (default origin/main)")
    options = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    local = root / ".local"
    environment = isolated_env(local / "test-user-home", local / "test-home")
    environment["PATH"] = unshimmed_path(os.environ.get("PATH", ""))

    def command(*arguments, quiet=False):
        def action():
            if not quiet:
                return subprocess.run(arguments, cwd=root, env=environment).returncode
            result = subprocess.run(arguments, cwd=root, env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            print(result.stdout, end="", flush=True)
            return result.returncode
        return action

    rust, tested = True, None
    if options.quick:
        files = changed_files(root, options.base)
        crates = workspace_crates(root)
        packages = affected_crates(files, crates)
        rust = packages is None or bool(packages)
        print(f"check.py --quick: {len(files)} changed files; Rust crates: "
              f"{'all' if packages is None else ', '.join(sorted(packages)) or 'none'}", flush=True)
        # Only crates with tests narrow the run (the bindings' crates have none).
        if packages is not None and not packages.issuperset(TESTED):
            tested = sorted(packages & TESTED)
    nextest = rust and nextest_available(environment, root)
    foreground = []
    if rust:
        if tested is None or tested:
            foreground.append(("cargo test", command(*test_command(tested, nextest, environment)), None))
        # Clippy checks the workspace: crates that did not change are already checked and cost nothing.
        foreground.append(("cargo clippy", command("cargo", "clippy", "--workspace", "--all-targets", "--locked", "--",
                                                   "-D", "warnings"), None))
    background = [("cargo fmt", command("cargo", "fmt", "--all", "--check", quiet=True), None)] if rust else []
    background += [
        # The helper tests include the workbench's, which need its locked npm packages.
        ("npm ci", lambda: install_web_dependencies(root, environment), None),
        ("helper tests", lambda: helper_tests(root, environment), "npm ci"),
    ]
    return run_lanes(foreground, background)


if __name__ == "__main__":
    sys.exit(main())

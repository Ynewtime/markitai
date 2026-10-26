"""Helper-run tests and audits never resolve `~` to the developer's real home."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from audit_formats import run_worker
import bounded_run
from check import PART, SPLIT_AT, affected_crates, helper_parts, helper_tests, run_lanes, run_steps, test_command, test_skips, unshimmed_path
from isolation import isolated_env


class IsolatedEnvironment(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_home_and_markitai_state_are_private_and_created(self):
        source = {"PATH": "tools", "HOME": "real-home", "USERPROFILE": "real-profile"}
        environment = isolated_env(self.root / "home", self.root / "state", source)
        self.assertEqual(environment["HOME"], str(self.root / "home"))
        self.assertEqual(environment["USERPROFILE"], str(self.root / "home"))
        self.assertEqual(environment["MARKITAI_HOME"], str(self.root / "state"))
        self.assertTrue((self.root / "home").is_dir() and (self.root / "state").is_dir())
        self.assertEqual(environment["PATH"], "tools")
        self.assertEqual(source["HOME"], "real-home", "the caller's mapping is not modified")

    def test_toolchain_homes_default_to_the_original_home(self):
        environment = isolated_env(self.root / "home", self.root / "state", {})
        self.assertEqual(environment["CARGO_HOME"], str(Path.home() / ".cargo"))
        self.assertEqual(environment["RUSTUP_HOME"], str(Path.home() / ".rustup"))
        explicit = isolated_env(self.root / "home", self.root / "state",
                                {"CARGO_HOME": "cargo", "RUSTUP_HOME": "rustup"})
        self.assertEqual((explicit["CARGO_HOME"], explicit["RUSTUP_HOME"]), ("cargo", "rustup"))

    def test_audit_workers_get_a_private_home_and_no_credentials(self):
        request = self.root / "case" / "request.json"
        request.parent.mkdir()
        seen = {}

        def capture(command, env, **_):
            seen.update(env)
            return subprocess.CompletedProcess(command, 1)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic"}), \
                patch("audit_formats.subprocess.run", side_effect=capture):
            run_worker("python", "native", request, 5)
        state = request.parent / "state" / "native"
        self.assertEqual(seen["HOME"], str(state / "home"))
        self.assertEqual(seen["USERPROFILE"], str(state / "home"))
        self.assertEqual(seen["MARKITAI_HOME"], str(state))
        self.assertNotIn("OPENAI_API_KEY", seen)


class StateGuard(unittest.TestCase):
    """Audit hooks cannot be removed, so each case runs in its own interpreter."""

    def setUp(self):
        bounded_run.watch(self)

    def guarded(self, code):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            (home / ".markitai").mkdir(parents=True)
            (home / ".markitai" / "config.json").write_text("{}", encoding="utf-8")
            (home / "elsewhere.txt").write_text("ok", encoding="utf-8")
            script = "\n".join([
                "import json, os, socket, sys", f"sys.path.insert(0, {str(Path(__file__).parent)!r})",
                "from isolation import ProtectedStateAccess, install_state_guard", code])
            return bounded_run.run([sys.executable, "-c", script], text=True,
                                   env=isolated_env(home, Path(directory) / "state"), cwd=directory)

    def test_real_state_and_network_are_blocked_and_recorded(self):
        result = self.guarded("""
state = install_state_guard("test")
assert set(state["self_tests"].values()) == {"blocked"}, state
assert open(os.path.expanduser("~/elsewhere.txt")).read() == "ok"
try:
    open(os.path.expanduser("~/.markitai/config.json"))
except ProtectedStateAccess:
    pass
try:
    socket.getaddrinfo("127.0.0.1", 9)
except PermissionError:
    pass
print(json.dumps(state))
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(result.stdout)
        self.assertEqual([event["event"] for event in state["blocked_state_events"]], ["open"])
        self.assertEqual(state["blocked_network_events"], ["socket.getaddrinfo"])

    def test_an_ordinary_permission_error_does_not_pass_the_self_test(self):
        # An earlier hook stands in for a real EACCES on the probe path.
        result = self.guarded("""
def denied(event, args):
    if event == "open" and "__markitai_guard_probe_" in os.fsdecode(args[0]):
        raise PermissionError("operating-system denial")
sys.addaudithook(denied)
try:
    install_state_guard("test")
except ProtectedStateAccess:
    sys.exit(2)
except PermissionError:
    sys.exit(0)
sys.exit(1)
""")
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(os.name == "nt", "the fixture tools are POSIX shell scripts")
class UnshimmedPath(unittest.TestCase):
    def tool(self, directory, name, prints):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(f"#!/bin/sh\necho '{prints}'\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_a_shim_is_replaced_by_the_directory_it_resolves_to(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = self.tool(root / "installs", "node", "")
            self.tool(root / "shims", "node", real)
            self.assertEqual(unshimmed_path(str(root / "shims")),
                             os.pathsep.join([str(root / "installs"), str(root / "shims")]))

    def test_a_real_interpreter_leaves_path_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.tool(root / "bin", "node", root / "bin" / "node")
            self.assertEqual(unshimmed_path(str(root / "bin")), str(root / "bin"))


class CheckSkips(unittest.TestCase):
    def test_each_name_in_markitai_check_skip_becomes_a_skip_filter(self):
        self.assertEqual(test_skips({}), [])
        self.assertEqual(test_skips({"MARKITAI_CHECK_SKIP": " "}), [])
        self.assertEqual(test_skips({"MARKITAI_CHECK_SKIP": " vision  ocr::tests::macos "}),
                         ["--", "--skip", "vision", "--skip", "ocr::tests::macos"])


class CheckSteps(unittest.TestCase):
    def test_every_step_runs_after_a_failure_and_the_failed_ones_are_named(self):
        ran = []

        def step(name, code):
            return name, lambda: ran.append(name) or code, None

        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = run_steps([step("fmt", 0), step("test", 101), step("clippy", 1), step("npm ci", 0),
                              ("helpers", lambda: ran.append("helpers") or 0, "npm ci")])
        self.assertEqual(code, 101)
        self.assertEqual(ran, ["fmt", "test", "clippy", "npm ci", "helpers"])
        self.assertEqual(stderr.getvalue().splitlines()[-1], "check.py: failed: test, clippy")

    def test_a_step_whose_prerequisite_failed_is_not_run(self):
        ran = []
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            code = run_steps([("npm ci", lambda: 1, None),
                              ("helpers", lambda: ran.append("helpers") or 0, "npm ci")])
        self.assertEqual((code, ran), (1, []))
        self.assertEqual(stderr.getvalue().splitlines()[-1], "check.py: failed: npm ci; not run: helpers")

    def test_all_passing_prints_nothing_and_returns_zero(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(run_steps([("fmt", lambda: 0, None)]), 0)
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()


class CheckLanes(unittest.TestCase):
    def test_the_background_lane_is_shown_after_the_foreground_and_both_failures_are_named(self):
        def step(name, code):
            return name, lambda: print(f"{name} says hello") or code, None

        with (patch("sys.stdout", new_callable=io.StringIO) as stdout,
              patch("sys.stderr", new_callable=io.StringIO) as stderr):
            code = run_lanes([step("test", 0), step("clippy", 1)], [step("fmt", 3), step("helpers", 0)])
        self.assertEqual(code, 1)
        said = [line for line in stdout.getvalue().splitlines() if line.endswith("says hello")]
        self.assertEqual(said[:2], ["test says hello", "clippy says hello"])
        self.assertEqual(sorted(said[2:]), ["fmt says hello", "helpers says hello"])
        self.assertEqual(stderr.getvalue().splitlines()[-1], "check.py: failed: clippy, fmt")


class CheckQuick(unittest.TestCase):
    CRATES = {
        "markitai-core": ("crates/markitai-core/", []),
        "markitai-cli": ("crates/markitai-cli/", ["markitai-core"]),
        "markitai-ffi": ("crates/markitai-ffi/", ["markitai-core"]),
        "markitai-python": ("crates/markitai-python/", ["markitai-core", "markitai-cli"]),
    }

    def test_a_crate_brings_every_crate_that_depends_on_it(self):
        self.assertEqual(affected_crates(["crates/markitai-core/src/lib.rs"], self.CRATES), set(self.CRATES))
        self.assertEqual(affected_crates(["crates/markitai-cli/tests/cache.rs"], self.CRATES),
                         {"markitai-cli", "markitai-python"})
        self.assertEqual(affected_crates(["crates/markitai-ffi/src/lib.rs", "docs/ci.md"], self.CRATES),
                         {"markitai-ffi"})

    def test_workspace_files_bring_every_crate(self):
        for path in ["Cargo.lock", "Cargo.toml", "vendor/lopdf/src/lib.rs", ".config/nextest.toml"]:
            self.assertIsNone(affected_crates([path], self.CRATES), path)

    def test_workbench_sources_bring_no_crate_but_its_embedded_build_does(self):
        web = "crates/markitai-cli/src/server/web/"
        self.assertEqual(affected_crates([web + "src/app.tsx", "scripts/check.py"], self.CRATES), set())
        self.assertEqual(affected_crates([web + "dist/app.js"], self.CRATES), {"markitai-cli", "markitai-python"})

    def test_nextest_builds_the_workspace_and_runs_the_chosen_crates(self):
        skip = {"MARKITAI_CHECK_SKIP": "vision"}
        self.assertEqual(test_command(None, True, skip),
                         ["cargo", "nextest", "run", "--workspace", "--locked", "--no-fail-fast", "--", "--skip", "vision"])
        self.assertEqual(test_command(["markitai-cli", "markitai-ffi"], True, {}),
                         ["cargo", "nextest", "run", "--workspace", "--locked", "--no-fail-fast",
                          "-E", "package(markitai-cli) | package(markitai-ffi)"])
        self.assertEqual(test_command(["markitai-cli"], False, {}),
                         ["cargo", "test", "--workspace", "--locked", "--no-fail-fast"])


class CheckHelperParts(unittest.TestCase):
    def test_a_large_file_runs_in_parts_that_together_run_all_its_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            # One test is inherited: only the loader's list, not the file's text, names it twice.
            tests = "".join(f"    def test_{index:02}(self):\n        pass\n" for index in range(SPLIT_AT + 1))
            (root / "scripts/test_large.py").write_text(
                "import unittest\n\n\nclass Base(unittest.TestCase):\n" + tests
                + "\n\nclass Child(Base):\n    pass\n", encoding="utf-8")
            (root / "scripts/test_small.py").write_text(
                "import unittest\n\n\nclass Small(unittest.TestCase):\n    def test_one(self):\n        pass\n",
                encoding="utf-8")
            files = sorted((root / "scripts").glob("test_*.py"))
            items = helper_parts(root, dict(os.environ), files)
            parts = [patterns for path, patterns in items if path.name == "test_large.py"]
            expected = 2 * (SPLIT_AT + 1)
            self.assertEqual(len(parts), -(-expected // PART))
            self.assertEqual(sorted(p for part in parts for p in part),
                             sorted(f"*.{cls}.test_{index:02}" for cls in ["Base", "Child"]
                                    for index in range(SPLIT_AT + 1)))
            self.assertEqual(items[-1], (root / "scripts/test_small.py", None))
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(helper_tests(root, dict(os.environ)), 0)
            self.assertIn(f"helper tests: {expected + 1} tests in 2 files, OK", stdout.getvalue())

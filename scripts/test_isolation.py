"""Helper-run tests and audits never resolve `~` to the developer's real home."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from audit_formats import run_worker
from check import unshimmed_path
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


if __name__ == "__main__":
    unittest.main()

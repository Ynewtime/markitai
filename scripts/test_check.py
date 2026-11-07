"""check.py: the nextest archive split that lets a slow platform only run what a fast one built."""
from pathlib import Path
import contextlib
import io
import os
import subprocess
import tempfile
import unittest

import check


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "repo"
        self.root.mkdir()
        for arguments in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@example.test", "commit", "-q",
                                           "--allow-empty", "-m", "one"]):
            subprocess.run(["git", *arguments], cwd=self.root, check=True)
        self.archive = Path(directory.name) / "tests.tar.zst"
        self.archive.write_bytes(b"archive bytes")

    def commit(self):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.test", "commit", "-q",
                        "--allow-empty", "-m", "two"], cwd=self.root, check=True)

    def record(self):
        with contextlib.redirect_stdout(io.StringIO()):
            check.write_archive_manifest(self.archive, self.root, "x86_64-apple-darwin")

    def test_an_archive_of_this_commit_verifies(self):
        self.record()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(check.verify_archive(self.archive, self.root))
        self.assertIn("x86_64-apple-darwin archive of commit", out.getvalue())

    def test_an_archive_of_another_commit_is_refused(self):
        self.record()
        self.commit()
        self.assertIn("was built from", check.verify_archive(self.archive, self.root))

    def test_a_changed_archive_is_refused(self):
        self.record()
        self.archive.write_bytes(b"archive bytez")
        self.assertIn("differs", check.verify_archive(self.archive, self.root))
        self.archive.write_bytes(b"shorter")
        self.assertIn("differs", check.verify_archive(self.archive, self.root))

    def test_a_missing_archive_or_manifest_is_refused(self):
        self.assertIn("is missing", check.verify_archive(self.archive, self.root))
        self.record()
        check.manifest_path(self.archive).unlink()
        self.assertIn("is missing", check.verify_archive(self.archive, self.root))


class CommandTests(unittest.TestCase):
    def test_the_archive_is_built_for_the_target_from_the_whole_workspace(self):
        command = check.archive_command(Path("a.tar.zst"), "x86_64-apple-darwin")
        self.assertEqual(command[:3], ["cargo", "nextest", "archive"])
        for expected in ("--workspace", "--locked"):
            self.assertIn(expected, command)
        self.assertEqual(command[command.index("--target") + 1], "x86_64-apple-darwin")

    def test_the_archive_runs_against_the_checkout_and_one_partition(self):
        environment = {"MARKITAI_CHECK_SKIP": "slow_one other"}
        command = check.archive_run_command(Path("a.tar.zst"), Path("repo"), "count:2/3", environment)
        self.assertEqual(command[command.index("--workspace-remap") + 1], "repo")
        self.assertEqual(command[command.index("--partition") + 1], "count:2/3")
        self.assertIn("--no-fail-fast", command)
        self.assertEqual(command[-5:], ["--", "--skip", "slow_one", "--skip", "other"])

    def test_without_a_partition_every_test_runs(self):
        self.assertNotIn("--partition", check.archive_run_command(Path("a"), Path("r"), None, {}))

    def test_the_modes_need_their_arguments(self):
        for arguments in (["--build-tests", "a"], ["--lint"], ["--partition", "count:1/2"],
                          ["--quick", "--run-tests", "a"], ["--lint", "--run-tests", "a", "--target", "t"]):
            with self.subTest(arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                check.main(arguments)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

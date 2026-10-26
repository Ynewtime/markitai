"""ci_cache: what the check workflow's Cargo cache keeps and when vendored crates are reused."""
from pathlib import Path
from unittest import mock
import contextlib
import io
import json
import os
import subprocess
import tempfile
import time
import unittest

import ci_cache


class CrateOfTests(unittest.TestCase):
    def test_artifact_names_map_to_their_crate(self):
        for entry, crate in [("libmarkitai_core-0123abcd.rlib", "markitai_core"),
                             ("markitai_core-0123abcd.d", "markitai_core"),
                             ("markitai-core-0123abcd", "markitai_core"),
                             ("integration-0123abcd.exe", "integration"),
                             ("markitai_core-0123abcd.markitai_core.1.rcgu.o", "markitai_core"),
                             ("markitai-webp-sys-0123abcd", "markitai_webp_sys")]:
            self.assertEqual(ci_cache.crate_of(entry), crate, entry)


LOCK = """version = 4

[[package]]
name = "markitai-core"
version = "2.3.0"

[[package]]
name = "serde"
version = "1.0.1"
source = "registry+https://github.com/rust-lang/crates.io-index"
"""


class DependencyKeyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "crates" / "core").mkdir(parents=True)
        self.write(LOCK, '[workspace.package]\nversion = "2.3.0"\n')

    def write(self, lock, manifest):
        (self.root / "Cargo.lock").write_text(lock)
        (self.root / "Cargo.toml").write_text(manifest)
        (self.root / "crates" / "core" / "Cargo.toml").write_text('[package]\nversion.workspace = true\n')
        return ci_cache.dependency_key(self.root)

    def test_a_release_commit_keeps_the_key(self):
        key = ci_cache.dependency_key(self.root)
        self.assertEqual(self.write(LOCK.replace("2.3.0", "2.4.0"), '[workspace.package]\nversion = "2.4.0"\n'), key)

    def test_a_dependency_or_profile_change_changes_it(self):
        key = ci_cache.dependency_key(self.root)
        self.assertNotEqual(self.write(LOCK.replace("1.0.1", "1.0.2"), '[workspace.package]\nversion = "2.3.0"\n'),
                            key)
        self.assertNotEqual(self.write(LOCK, '[workspace.package]\nversion = "2.3.0"\n[profile.dev]\nopt-level = 1\n'),
                            key)


class CacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "target"
        self.git("init", "-q")
        (self.root / "vendor" / "lib").mkdir(parents=True)
        (self.root / "vendor" / "lib" / "lib.rs").write_text("pub fn a() {}\n")
        self.commit()
        names = mock.patch.object(ci_cache, "crate_names", return_value=({"markitai_core", "integration"},
                                                                          {"pdf_inspector"}))
        names.start()
        self.addCleanup(names.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def git(self, *arguments):
        subprocess.run(["git", *arguments], cwd=self.root, check=True, capture_output=True)

    def commit(self):
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "c")

    def artifact(self, relative, mtime=None):
        path = self.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def build(self):
        kept = [self.artifact("debug/deps/libserde-1111.rlib"), self.artifact("debug/.fingerprint/serde-1111/a"),
                self.artifact("debug/build/markitai-webp-sys-2222/out/a.o"),
                self.artifact("debug/deps/libpdf_inspector-3333.rlib")]
        dropped = [self.artifact("debug/deps/libmarkitai_core-4444.rlib"),
                   self.artifact("debug/deps/integration-5555"), self.artifact("debug/.fingerprint/markitai-core-4444/a"),
                   self.artifact("debug/markitai"), self.artifact("debug/incremental/x"), self.artifact("nextest/x"),
                   self.artifact(".rustc_info.json")]
        return kept, dropped

    def test_prune_keeps_dependencies_and_records_the_vendor_tree(self):
        kept, dropped = self.build()
        ci_cache.prune(self.root, self.target)
        self.assertEqual([path for path in kept if not path.exists()], [])
        self.assertEqual([path for path in dropped if path.exists()], [])
        state = json.loads((self.target / ci_cache.STATE).read_text())
        self.assertEqual(state, {"vendor": ci_cache.vendor_tree(self.root)})

    def test_restore_ages_vendor_sources_only_for_the_same_tree(self):
        self.build()
        ci_cache.prune(self.root, self.target)
        source = self.root / "vendor" / "lib" / "lib.rs"
        ci_cache.restore(self.root, self.target)
        self.assertEqual(source.stat().st_mtime, ci_cache.OLD)

        source.write_text("pub fn b() {}\n")
        self.commit()
        ci_cache.restore(self.root, self.target)
        self.assertGreater(source.stat().st_mtime, ci_cache.OLD)
        self.assertFalse(json.loads((self.target / ci_cache.STATE).read_text())["vendor_reused"])

    def test_prune_after_a_vendor_change_drops_the_old_vendored_artifacts(self):
        self.build()
        ci_cache.prune(self.root, self.target)
        old = self.artifact("debug/deps/libpdf_inspector-3333.rlib", mtime=time.time() - 3600)
        (self.root / "vendor" / "lib" / "lib.rs").write_text("pub fn b() {}\n")
        self.commit()
        ci_cache.restore(self.root, self.target)
        new = self.artifact("debug/deps/libpdf_inspector-6666.rlib", mtime=time.time() + 60)
        other = self.artifact("debug/deps/libserde-1111.rlib", mtime=time.time() - 3600)
        ci_cache.prune(self.root, self.target)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(other.exists())

    def test_restore_without_a_cache_changes_nothing(self):
        source = self.root / "vendor" / "lib" / "lib.rs"
        before = source.stat().st_mtime
        ci_cache.restore(self.root, self.target)
        self.assertEqual(source.stat().st_mtime, before)
        self.assertFalse((self.target / ci_cache.STATE).exists())


if __name__ == "__main__":
    unittest.main()

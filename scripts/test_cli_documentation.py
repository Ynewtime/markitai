"""Public-guide whitelist and source-contract counterexamples; no product runs."""
from pathlib import Path
import os
import re
import shutil
from types import SimpleNamespace
import stat
import tempfile
import unittest
from unittest.mock import patch
import cli_documentation as collector

from cli_documentation import (DOCUMENTATION_PATHS, MAX_DOCUMENT_BYTES,
                               MAX_DOCUMENTATION_BYTES, cli_documentation,
                               validate_documentation)


EXPECTED_GUIDES = {"README.md", "llms.txt", "llms-full.txt", "docs/index.md",
                   "docs/quickstart.md", "docs/cli.md", "docs/mcp.md"}


class DocumentationCollectionTests(unittest.TestCase):
    def setUp(self):
        # Always a fixture root; never read personal state or shell profiles.
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.expected = {name: ("# " + name + "\n\nOffline 世界.\n").encode() for name in EXPECTED_GUIDES}
        for name, content in self.expected.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def test_exact_whitelist_ignores_internal_and_unrelated_files(self):
        self.assertEqual(set(DOCUMENTATION_PATHS), EXPECTED_GUIDES)
        for name in ["docs/CONTROL.md", "docs/STATUS.md", "docs/validation/evidence.json",
                     ".env", ".markitai/config.json"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture must not be included")
        self.assertEqual(cli_documentation(self.root), self.expected)

    def test_missing_or_nonregular_guide_and_parent_fail(self):
        selected = self.root / "llms.txt"
        selected.unlink()
        with self.assertRaises(FileNotFoundError):
            cli_documentation(self.root)
        selected.mkdir()
        with self.assertRaisesRegex(RuntimeError, "regular"):
            cli_documentation(self.root)
        selected.rmdir()
        selected.write_bytes(self.expected["llms.txt"])
        shutil.rmtree(self.root / "docs")
        (self.root / "docs").write_bytes(b"not a directory")
        with self.assertRaisesRegex(RuntimeError, "directory"):
            cli_documentation(self.root)

    def test_non_utf8_nul_and_oversized_files_fail(self):
        selected = self.root / "llms-full.txt"
        for content in [b"", b"\xff", b"# text\0hidden", b"x" * (MAX_DOCUMENT_BYTES + 1)]:
            selected.write_bytes(content)
            with self.subTest(bytes=len(content)), self.assertRaises(RuntimeError):
                cli_documentation(self.root)

    def test_incomplete_extra_wrong_type_and_total_bound_are_rejected(self):
        for supplied in [{}, {**self.expected, "docs/CONTROL.md": b"private"},
                         {**self.expected, "README.md": "not bytes"}]:
            with self.assertRaises(RuntimeError):
                validate_documentation(supplied)
        supplied = {name: b"x" * MAX_DOCUMENT_BYTES for name in EXPECTED_GUIDES}
        self.assertGreater(sum(map(len, supplied.values())), MAX_DOCUMENTATION_BYTES)
        with self.assertRaisesRegex(RuntimeError, "total"):
            validate_documentation(supplied)

    def test_reparse_and_symlink_metadata_are_rejected_before_open(self):
        original_lstat = Path.lstat
        selected = self.root / "README.md"
        for mode, attributes in [(stat.S_IFREG, 0x400), (stat.S_IFLNK, 0)]:
            def status(path):
                if path == selected:
                    return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
                return original_lstat(path)
            with self.subTest(mode=mode, attributes=attributes), \
                    patch("cli_documentation.Path.lstat", autospec=True, side_effect=status), \
                    patch("cli_documentation.os.open") as opened:
                with self.assertRaisesRegex(RuntimeError, "regular"):
                    cli_documentation(self.root)
                opened.assert_not_called()

    def test_file_content_change_during_read_is_rejected(self):
        original_fdopen = os.fdopen
        selected = self.root / "README.md"

        class ChangedReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def read(self, count):
                content = self.stream.read(count)
                selected.write_bytes(content + b"changed")
                return content

        with patch("cli_documentation.os.fdopen", side_effect=lambda fd, mode: ChangedReader(original_fdopen(fd, mode))):
            with self.assertRaisesRegex(RuntimeError, "changed while reading"):
                cli_documentation(self.root)

    def test_windows_named_and_held_ctime_semantics_keep_same_file_binding(self):
        named = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o600,
                                st_size=12, st_mtime_ns=34, st_ctime_ns=50, st_birthtime_ns=50)
        held = SimpleNamespace(**{**vars(named), "st_ctime_ns": 60})
        with patch.object(collector, "os", SimpleNamespace(name="nt")):
            self.assertEqual(collector._file_binding(named), collector._file_binding(held))
            self.assertNotEqual(collector._identity(named), collector._identity(held))
            held.st_ino += 1
            self.assertNotEqual(collector._file_binding(named), collector._file_binding(held))

    def test_windows_legacy_stat_without_birthtime_retains_creation_time_binding(self):
        named = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o600,
                                st_size=12, st_mtime_ns=34, st_ctime_ns=50)
        held = SimpleNamespace(**vars(named))
        with patch.object(collector, "os", SimpleNamespace(name="nt")):
            self.assertEqual(collector._file_binding(named), collector._file_binding(held))
            held.st_ino += 1
            self.assertNotEqual(collector._file_binding(named), collector._file_binding(held))
            held.st_ino = named.st_ino
            held.st_ctime_ns += 1
            self.assertNotEqual(collector._file_binding(named), collector._file_binding(held))

    def test_posix_named_and_held_binding_keeps_complete_ctime(self):
        named = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o600,
                                st_size=12, st_mtime_ns=34, st_ctime_ns=50, st_birthtime_ns=50)
        held = SimpleNamespace(**vars(named))
        with patch.object(collector, "os", SimpleNamespace(name="posix")):
            self.assertEqual(collector._file_binding(named), collector._file_binding(held))
            held.st_ctime_ns += 1
            self.assertNotEqual(collector._file_binding(named), collector._file_binding(held))
            self.assertEqual(collector._file_binding(held), collector._identity(held))

    def test_held_metadata_only_ctime_change_during_read_is_rejected(self):
        original = os.fstat
        selected = self.root / "README.md"
        file_id = selected.stat().st_ino
        calls = 0
        def changed(fd):
            nonlocal calls
            status = original(fd)
            if status.st_ino == file_id:
                calls += 1
                if calls == 2:
                    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
                    return SimpleNamespace(**{**{k: getattr(status, k) for k in fields},
                        "st_file_attributes": getattr(status, "st_file_attributes", 0),
                        "st_ctime_ns": status.st_ctime_ns + 1})
            return status
        with patch("cli_documentation.os.fstat", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed while reading"):
                cli_documentation(self.root)

    def test_named_metadata_only_ctime_change_during_read_is_rejected(self):
        original = Path.lstat
        selected = self.root / "README.md"
        calls = 0
        def changed(path):
            nonlocal calls
            status = original(path)
            if path == selected:
                calls += 1
                if calls == 2:
                    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
                    return SimpleNamespace(**{**{k: getattr(status, k) for k in fields},
                        "st_file_attributes": getattr(status, "st_file_attributes", 0),
                        "st_ctime_ns": status.st_ctime_ns + 1})
            return status
        with patch("cli_documentation.Path.lstat", autospec=True, side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed while reading"):
                cli_documentation(self.root)

    def test_real_parent_mtime_change_during_read_is_rejected(self):
        original = os.fdopen
        selected = (self.root / "docs/index.md").stat().st_ino
        parent = self.root / "docs"
        class ChangedReader:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, count):
                content = self.stream.read(count)
                if os.fstat(self.fileno()).st_ino == selected:
                    before = parent.stat()
                    os.utime(parent, ns=(before.st_atime_ns, before.st_mtime_ns + 10_000_000))
                return content
        with patch("cli_documentation.os.fdopen", side_effect=lambda fd, mode: ChangedReader(original(fd, mode))):
            with self.assertRaisesRegex(RuntimeError, "parent changed while reading"):
                cli_documentation(self.root)


class RepositoryGuideContractTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_real_guides_fit_bounds_and_llms_index_links_resolve_in_archive(self):
        files = cli_documentation(self.root)
        text = files["llms.txt"].decode()
        targets = re.findall(r"\]\(([^)]+)\)", text)
        self.assertGreaterEqual(len(targets), 6)
        self.assertTrue(all(target in EXPECTED_GUIDES for target in targets))
        self.assertEqual(set(files), EXPECTED_GUIDES)

    def test_quickstart_extension_list_matches_actual_reader_constants(self):
        # Derive the source set independently of documentation generation.
        extensions = set()
        for path, constant in [("crates/markitai-core/src/formats/mod.rs", "DOCUMENT_EXTENSIONS"),
                               ("crates/markitai-core/src/images.rs", "IMAGE_EXTENSIONS")]:
            source = (self.root / path).read_text(encoding="utf-8")
            expression = re.search(r"\b" + constant + r"\s*:[^=]+=\s*&?\s*\[(.*?)\];", source, re.S)
            self.assertIsNotNone(expression, constant)
            extensions.update(re.findall(r'"([a-z0-9]+)"', expression.group(1)))
        self.assertEqual(len(extensions), 61)
        guide = (self.root / "docs/quickstart.md").read_text(encoding="utf-8")
        section = guide.split("Recognized extensions (61", 1)[1].split("```text\n", 1)[1].split("```", 1)[0]
        stated = re.findall(r"\.([a-z0-9]+)\b", section)
        self.assertEqual(len(stated), len(set(stated)), "duplicate extension in guide")
        self.assertEqual(set(stated), extensions)


if __name__ == "__main__":
    unittest.main()

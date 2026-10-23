from pathlib import Path
import contextlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
import zipfile

import release_build
from release_build import (TARGETS, archive_name, check_versions, expected_assets, extract_release_archive,
                           glibc_requirement, python_version, release_notes, server_errors, write_sums)

ROOT = Path(__file__).resolve().parents[1]


def tar_with(path, members):
    """members: (name, kind, mode, payload-or-linkname)."""
    with tarfile.open(path, "w:gz") as bundle:
        for name, kind, mode, value in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = value
                bundle.addfile(info)
            else:
                info.size = len(value)
                bundle.addfile(info, io.BytesIO(value))


GOOD = [("markitai", "file", 0o755, b"\x7fELF"), ("mkai", "link", 0o777, "markitai"),
        ("markitai-mcp", "link", 0o777, "markitai"), ("LICENSE", "file", 0o644, b"MIT"),
        ("NOTICE", "file", 0o644, b"n"), ("licenses/x/LICENSE", "file", 0o644, b"x")]


class VersionTests(unittest.TestCase):
    def test_python_versions(self):
        self.assertEqual(python_version("2.0.0"), "2.0.0")
        self.assertEqual(python_version("2.0.0-dev"), "2.0.0.dev0")
        self.assertEqual(python_version("2.0.0-rc.1"), "2.0.0rc1")
        self.assertEqual(python_version("2.1.0-beta.2"), "2.1.0b2")
        with self.assertRaises(ValueError):
            python_version("2.0")

    def test_repository_versions_agree(self):
        result = check_versions(ROOT)
        self.assertEqual(result["python_version"], python_version(result["version"]))
        with self.assertRaises(SystemExit):
            check_versions(ROOT, tag="v0.0.1")

    def test_server_json_names_the_pypi_package(self):
        server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
        self.assertEqual(server_errors(server), [])
        self.assertEqual(server_errors(server, server["version"]), [])
        self.assertTrue(server_errors(server, "9.9.9"))
        arguments = server["packages"][0]["runtimeArguments"]
        self.assertIn({"--from": f"markitai=={server['version']}"},
                      [{item["name"]: item["value"]} for item in arguments])


class NotesTests(unittest.TestCase):
    def test_section_until_next_version(self):
        changelog = "# Changelog\n\n## [2.0.0] - 2026-11-01\n\n### Added\n- a\n\n## [1.2.0]\n- old\n"
        self.assertEqual(release_notes(changelog, "2.0.0"), "### Added\n- a\n")

    def test_missing_or_empty_section_fails(self):
        with self.assertRaises(SystemExit):
            release_notes("# Changelog\n\n## [2.0.1]\n- x\n", "2.0.0")
        with self.assertRaises(SystemExit):
            release_notes("# Changelog\n\n## [2.0.0]\n\n## [1.0.0]\n- x\n", "2.0.0")
        with self.assertRaises(SystemExit):
            release_notes("## [2.0.0-rc.1]\n- x\n", "2.0.0")


class AssetTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(archive_name("2.0.0", "x86_64-unknown-linux-gnu"),
                         "markitai-v2.0.0-x86_64-unknown-linux-gnu.tar.gz")
        self.assertEqual(archive_name("2.0.0", "aarch64-pc-windows-msvc"),
                         "markitai-v2.0.0-aarch64-pc-windows-msvc.zip")
        assets = expected_assets("2.0.0")
        self.assertEqual(len(assets), 13)
        self.assertIn("markitai-2.0.0-cp310-abi3-manylinux_2_28_aarch64.whl", assets)
        self.assertIn("markitai-2.0.0.tar.gz", assets)

    def test_sums_require_the_release_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in expected_assets("2.0.0"):
                (folder / name).write_bytes(name.encode())
            with contextlib.redirect_stdout(io.StringIO()):
                write_sums(folder, "2.0.0")
            lines = (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 13)
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}  \S+", line) for line in lines))
            (folder / "stray.txt").write_text("x")
            with self.assertRaises(SystemExit):
                write_sums(folder, "2.0.0")

    def test_workflow_and_setup_scripts_use_the_release_targets(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        for job in ["cli", "wheel"]:
            block = workflow.split(f"\n  {job}:\n", 1)[1].split("\n  # ", 1)[0]
            self.assertEqual(set(re.findall(r"target: ([\w-]+)", block)), set(TARGETS), job)
        shell = (ROOT / "scripts/install/setup.sh").read_text(encoding="utf-8")
        for target in TARGETS:
            if "windows" in target:
                continue
            arch, os_name = target.split("-", 1)
            self.assertTrue(f"_arch={arch}" in shell and f"_os={os_name}" in shell, target)
        powershell = (ROOT / "scripts/install/setup.ps1").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r'"([\w]+-pc-windows-msvc)"', powershell)),
                         {target for target in TARGETS if "windows" in target})
        for text in [shell, powershell]:
            self.assertTrue("markitai-v" in text and "SHA256SUMS" in text)

    def test_container_images_match(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        images = re.findall(r"image: '(quay\.io/pypa/manylinux_2_28_\w+@sha256:[0-9a-f]{64})'", workflow)
        self.assertEqual(len(images), 4)
        self.assertEqual(len(set(images)), 2)


class SdistTests(unittest.TestCase):
    def test_files_embedded_from_outside_a_crate_are_in_the_sdist(self):
        """maturin copies each crate's own files; anything else a build embeds is in SDIST_EXTRA."""
        outside = set()
        for crate in ["markitai-core", "markitai-cli", "markitai-python"]:
            base = ROOT / "crates" / crate
            for source in (base / "src").rglob("*.rs"):
                text = source.read_text(encoding="utf-8")
                if "test" in source.relative_to(base).as_posix():
                    continue
                for relative in re.findall(r'include_(?:bytes|str)!\(\s*"([^"]+)"', text):
                    outside.add((source.parent / relative).resolve())
                for relative in re.findall(r'env!\("CARGO_MANIFEST_DIR"\),\s*"(/[^"]+)"', text):
                    outside.add((base / relative.lstrip("/")).resolve())
        # tests/support is only included under #[cfg(test)], which a wheel build never compiles.
        outside = {path for path in outside if not path.is_relative_to(ROOT / "tests") and not any(
            path.is_relative_to(ROOT / "crates" / crate) for crate in ["markitai-core", "markitai-cli", "markitai-python"])}
        self.assertTrue(outside, "the scan finds the workbench's vendored assets")
        for path in outside:
            relative = path.relative_to(ROOT).as_posix()
            self.assertTrue(any(relative == extra or relative.startswith(extra.rstrip("/") + "/")
                                for extra in release_build.SDIST_EXTRA), relative)


class ArchiveTests(unittest.TestCase):
    def extract(self, members, name="markitai-v2.0.0-x86_64-unknown-linux-gnu.tar.gz"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        archive = Path(directory.name) / name
        tar_with(archive, members)
        return extract_release_archive(archive, Path(directory.name) / "out")

    def test_release_tar_shape(self):
        self.assertEqual(self.extract(GOOD)["members"], len(GOOD))

    def test_rejects_other_links_executables_and_paths(self):
        for members in [
            GOOD + [("evil", "link", 0o777, "/etc/passwd")],
            [*GOOD[:1], ("mkai", "link", 0o777, "../markitai"), *GOOD[2:]],
            GOOD + [("tool", "file", 0o755, b"x")],
            GOOD + [("../escape", "file", 0o644, b"x")],
            GOOD + [("/abs", "file", 0o644, b"x")],
            GOOD[1:],
            [("markitai", "file", 0o700, b"x"), *GOOD[1:]],
        ]:
            with self.subTest(members=[member[0] for member in members]), self.assertRaises(SystemExit):
                self.extract(members)

    def test_windows_zip_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "markitai-v2.0.0-x86_64-pc-windows-msvc.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe", "LICENSE", "NOTICE"]:
                    bundle.writestr(name, b"x")
            self.assertEqual(extract_release_archive(archive, Path(directory) / "out")["members"], 5)
            bad = Path(directory) / "bad.zip"
            with zipfile.ZipFile(bad, "w") as bundle:
                for name in ["markitai.exe", "mkai.exe", "LICENSE", "NOTICE", "../x"]:
                    bundle.writestr(name, b"x")
            with self.assertRaises(SystemExit):
                extract_release_archive(bad, Path(directory) / "bad")


class CaptureTests(unittest.TestCase):
    def test_a_failed_probe_shows_what_it_printed(self):
        command = [sys.executable, "-c", "import sys; print('out'); sys.exit('the cause')"]
        with self.assertRaises(SystemExit) as raised:
            release_build.capture(command)
        self.assertIn("exited 1", str(raised.exception))
        self.assertIn("out", str(raised.exception))
        self.assertIn("the cause", str(raised.exception))
        self.assertEqual(release_build.capture([sys.executable, "-c", "print('ok')"]).strip(), "ok")

    def test_a_python_probe_writes_utf8_whatever_the_locale(self):
        # Windows runners give a child Python's pipes cp1252, which has no 世界.
        probe = [sys.executable, "-c", "print('世界')"]
        with self.assertRaises(SystemExit):
            release_build.capture(probe, env={**os.environ, "PYTHONIOENCODING": "cp1252"})
        self.assertEqual(release_build.capture(probe, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).strip(),
                         "世界")
        source = Path(release_build.__file__).read_text(encoding="utf-8")
        self.assertIn('"PYTHONIOENCODING": "utf-8"', source.split("def smoke_wheel", 1)[1].split("\ndef ", 1)[0])


class GlibcTests(unittest.TestCase):
    def test_newest_symbol_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elf"
            path.write_bytes(b"\x7fELF...GLIBC_2.17\x00GLIBC_2.28\x00GLIBC_2.3.4\x00GLIBC_PRIVATE\x00")
            self.assertEqual(glibc_requirement(path), (2, 28))
            path.write_bytes(b"\x7fELF")
            self.assertIsNone(glibc_requirement(path))
        self.assertEqual(release_build.GLIBC_FLOOR, (2, 28))


if __name__ == "__main__":
    unittest.main()

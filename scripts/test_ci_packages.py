"""Counterexamples for package validation; one offline npm pack, no installation."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

from ci_packages import (npm_command, source_snapshot, stage_node_licenses, supplement_wheel_licenses,
                         verify_node_licenses, write_cli_zip)


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_windows_cmd_uses_node_and_a_real_cli_without_a_shell(self):
        directory = self.root / "Node & tools"
        cli = directory / "node_modules/npm/bin/npm-cli.js"
        cli.parent.mkdir(parents=True)
        cli.write_text("fixture", encoding="utf-8")
        node = directory / "node.exe"
        npm = directory / "npm.cmd"
        self.assertEqual(npm_command(npm, node, "win32"), [str(node), str(cli)])
        cli.unlink()
        with self.assertRaisesRegex(RuntimeError, "npm-cli.js"):
            npm_command(npm, node, "win32")
        self.assertEqual(npm_command("/bin/npm", None, "linux"), ["/bin/npm"])

    def test_windows_zip_contains_a_real_mcp_executable_with_matching_bytes(self):
        binary = self.root / "markitai.exe"
        alternate = self.root / "mkai.exe"
        binary.write_bytes(b"MZ" + bytes(range(256)) * 7)
        alternate.write_bytes(b"MZ alternate CLI")
        archive = self.root / "windows.zip"
        licenses = {"LICENSE": b"license", "licenses/example/LICENSE": b"upstream"}
        write_cli_zip(binary, alternate, archive, licenses, True)
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(set(bundle.namelist()), {"markitai.exe", "mkai.exe", "markitai-mcp.exe", *licenses})
            self.assertEqual(bundle.read("markitai-mcp.exe"), binary.read_bytes())
            self.assertEqual(bundle.read("mkai.exe"), alternate.read_bytes())
            self.assertFalse(any(name.endswith(".cmd") for name in bundle.namelist()))
            for name, content in licenses.items():
                self.assertEqual(bundle.read(name), content)

    def test_unix_zip_keeps_the_relative_mcp_symlink(self):
        binary = self.root / "markitai"
        binary.write_bytes(b"native CLI")
        archive = self.root / "unix.zip"
        write_cli_zip(binary, binary, archive, {}, False)
        with zipfile.ZipFile(archive) as bundle:
            alias = bundle.getinfo("markitai-mcp")
            self.assertEqual(alias.external_attr >> 16, 0o120777)
            self.assertEqual(bundle.read(alias), b"markitai")

    def test_same_size_and_mtime_source_change_is_detected(self):
        source = self.root / "source.rs"
        source.write_bytes(b"old bytes")
        first = source_snapshot(self.root, ["source.rs"])
        timestamp = source.stat().st_mtime_ns
        source.write_bytes(b"new bytes")
        os.utime(source, ns=(timestamp, timestamp))
        self.assertNotEqual(first, source_snapshot(self.root, ["source.rs"]))
        source.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            source_snapshot(self.root, ["source.rs"])

    @unittest.skipIf(os.name == "nt", "symlink privileges are host dependent")
    def test_symlink_source_tracks_target_bytes_and_rejects_external_files(self):
        source = self.root / "actual.rs"
        source.write_bytes(b"one")
        link = self.root / "alias.rs"
        link.symlink_to(source.name)
        first = source_snapshot(self.root, ["alias.rs"])
        source.write_bytes(b"two")
        self.assertNotEqual(first, source_snapshot(self.root, ["alias.rs"]))
        link.unlink()
        link.symlink_to(self.root.parent)
        with self.assertRaisesRegex(RuntimeError, "in-repository"):
            source_snapshot(self.root, ["alias.rs"])

    def test_node_notice_must_be_a_real_file_with_exact_bytes(self):
        for mode in ["missing", "symlink", "different", "valid"]:
            package = self.root / f"{mode}.tgz"
            with tarfile.open(package, "w:gz") as archive:
                if mode != "missing":
                    member = tarfile.TarInfo("package/NOTICE")
                    if mode == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "LICENSE"
                        archive.addfile(member)
                    else:
                        value = b"notice" if mode == "valid" else b"changed"
                        member.size = len(value)
                        archive.addfile(member, io.BytesIO(value))
            if mode == "valid":
                verify_node_licenses(package, {"NOTICE": b"notice"})
            else:
                with self.assertRaises(RuntimeError):
                    verify_node_licenses(package, {"NOTICE": b"notice"})

    @unittest.skipUnless(shutil.which("npm"), "npm required for actual archive regression")
    def test_npm_preserves_nested_original_manifest_and_hidden_provenance(self):
        licenses = {
            "NOTICE": b"fixture notice\n",
            "licenses/upstream/local-evidence/example/Cargo.toml.orig": b"original bytes\n",
            "licenses/upstream/local-evidence/example/.cargo_vcs_info.json": b'{}\n',
        }
        directories = stage_node_licenses(self.root, licenses)
        (self.root / "package.json").write_text(json.dumps({
            "name": "markitai-attribution-fixture", "version": "1.0.0", "files": [*licenses, *directories]
        }), encoding="utf-8")
        result = subprocess.run([*npm_command(shutil.which("npm"), shutil.which("node")),
                                 "pack", "--json", "--ignore-scripts", "--offline"],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        package = self.root / json.loads(result.stdout)[0]["filename"]
        verify_node_licenses(package, licenses)
        with tarfile.open(package) as archive:
            self.assertFalse(any(name.endswith(".npmignore") for name in archive.getnames()))

    def test_supplemented_wheel_retains_native_bytes_and_has_verifiable_record(self):
        source = self.root / "raw.whl"
        target = self.root / "final.whl"
        original = {"markitai/_native.so": bytes(range(256)) * 4,
                    "markitai/__init__.py": b"# fixture\n",
                    "markitai-1.dist-info/METADATA": b"Metadata-Version: 2.4\nName: markitai\n",
                    "markitai-1.dist-info/RECORD": b"old record\n"}
        with zipfile.ZipFile(source, "x") as archive:
            for name, data in original.items():
                archive.writestr(name, data)
        raw = source.read_bytes()
        result = supplement_wheel_licenses(source, target, {"LICENSE": b"license", "NOTICE": b"notice"})
        self.assertEqual(source.read_bytes(), raw)
        self.assertNotEqual(result["original"]["sha256"], result["supplemented"]["sha256"])
        with zipfile.ZipFile(target) as archive:
            for name, data in original.items():
                if not name.endswith("/RECORD"):
                    self.assertEqual(archive.read(name), data)
            rows = list(csv.reader(io.StringIO(archive.read("markitai-1.dist-info/RECORD").decode())))
            self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
            for name, digest, size in rows:
                if name.endswith("/RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                else:
                    value = archive.read(name)
                    expected = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
                    self.assertEqual((digest, size), ("sha256=" + expected, str(len(value))))
            self.assertEqual(archive.read("markitai-1.dist-info/licenses/NOTICE"), b"notice")
        with self.assertRaises(FileExistsError):
            supplement_wheel_licenses(source, target, {})


if __name__ == "__main__":
    unittest.main()

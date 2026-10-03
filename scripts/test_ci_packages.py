"""Counterexamples for package validation; one offline npm pack, no installation."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import warnings
from unittest.mock import patch
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

from ci_packages import (npm_command, source_snapshot, stage_node_licenses, supplement_wheel_licenses,
                         verify_node_licenses, write_cli_zip, extract_cli_zip, doctor_probe, mcp_probe, identity, package_attribution,
                         write_single_binary_tar, extract_single_binary_tar)
from cli_documentation import DOCUMENTATION_PATHS


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)


    def zip_inputs(self):
        binary = self.root / "markitai.exe"
        alternate = self.root / "mkai.exe"
        binary.write_bytes(b"synthetic primary executable fixture")
        alternate.write_bytes(b"synthetic alternate fixture")
        return binary, alternate, {"LICENSE": b"license", "licenses/hayro/CMAP_LICENSE.txt": b"original cmap notice"}

    def test_zip_installs_exact_payloads_in_a_unicode_directory(self):
        binary, alternate, licenses = self.zip_inputs()
        archive = self.root / "archive.zip"
        write_cli_zip(binary, alternate, archive, licenses, True)
        installed = self.root / "安装 with spaces"
        record = extract_cli_zip(archive, installed, binary, alternate, licenses, True)
        self.assertEqual(record["mcp_alias"]["kind"], "executable_copy")
        self.assertEqual(identity(installed / "markitai-mcp.exe"), identity(binary))
        self.assertEqual(identity(installed / "mkai.exe"), identity(alternate))
        self.assertEqual(set(record["attribution"]), set(licenses))
        with self.assertRaises(FileExistsError):
            extract_cli_zip(archive, installed, binary, alternate, licenses, True)

    def test_zip_missing_duplicate_changed_link_and_extra_payloads_are_rejected_before_extraction(self):
        binary, alternate, licenses = self.zip_inputs()
        valid = self.root / "valid.zip"
        write_cli_zip(binary, alternate, valid, licenses, True)
        for defect in ["missing", "duplicate", "mcp-bytes", "notice-bytes", "link", "extra"]:
            broken = self.root / (defect + ".zip")
            with zipfile.ZipFile(valid) as source, zipfile.ZipFile(broken, "x") as target:
                for info in source.infolist():
                    if defect == "missing" and info.filename == "markitai-mcp.exe":
                        continue
                    content = source.read(info)
                    if defect == "mcp-bytes" and info.filename == "markitai-mcp.exe":
                        content = bytes([content[0] ^ 1]) + content[1:]
                    if defect == "notice-bytes" and info.filename == "licenses/hayro/CMAP_LICENSE.txt":
                        content = content[::-1]
                    if defect == "link" and info.filename == "mkai.exe":
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    target.writestr(info, content)
                if defect == "extra":
                    target.writestr("../outside", b"extra")
                elif defect == "duplicate":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        target.writestr("LICENSE", licenses["LICENSE"])
            destination = self.root / (defect + "-installed")
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                extract_cli_zip(broken, destination, binary, alternate, licenses, True)
            self.assertFalse(destination.exists())
        self.assertFalse((self.root / "outside").exists())

    def test_zip_attribution_cannot_collide_with_executable_or_escape(self):
        binary, alternate, _ = self.zip_inputs()
        for name in ["markitai.exe", "MARKITAI-MCP.EXE", "mkai.exe/a", "../outside", "/absolute", ".", "",
                     "C:outside", "licenses\\outside", "licenses/trailing.", "licenses/trailing ", "licenses/nul\0tail"]:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Unsafe"):
                write_cli_zip(binary, alternate, self.root / "unsafe.zip", {name: b"x"}, True)
            self.assertFalse((self.root / "unsafe.zip").exists())

    def documentation_inputs(self):
        return {name: ("# " + name + "\n\nOffline guide 世界.\n").encode("utf-8")
                for name in DOCUMENTATION_PATHS}

    def test_zip_offline_guides_are_separate_from_all_attribution(self):
        binary, alternate, licenses = self.zip_inputs()
        docs = self.documentation_inputs()
        archive = self.root / "documented.zip"
        write_cli_zip(binary, alternate, archive, licenses, True, docs)
        installed = self.root / "离线 安装"
        record = extract_cli_zip(archive, installed, binary, alternate, licenses, True, docs)
        self.assertEqual(set(record["attribution"]), set(licenses))
        self.assertEqual(set(record["documentation"]), set(DOCUMENTATION_PATHS))
        for name, content in {**licenses, **docs}.items():
            self.assertEqual((installed / name).read_bytes(), content)
            field = "documentation" if name in docs else "attribution"
            self.assertEqual(record[field][name], {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})

    def test_zip_document_defects_fail_before_any_extraction(self):
        binary, alternate, licenses = self.zip_inputs()
        docs = self.documentation_inputs()
        valid = self.root / "documented.zip"
        write_cli_zip(binary, alternate, valid, licenses, True, docs)
        selected = "docs/quickstart.md"
        for defect in ["missing", "duplicate", "changed", "symlink", "extra"]:
            broken = self.root / ("document-" + defect + ".zip")
            with zipfile.ZipFile(valid) as source, zipfile.ZipFile(broken, "x") as target:
                for info in source.infolist():
                    if defect == "missing" and info.filename == selected:
                        continue
                    content = source.read(info)
                    if defect == "changed" and info.filename == selected:
                        content = bytes([content[0] ^ 1]) + content[1:]
                    if defect == "symlink" and info.filename == selected:
                        info.create_system = 3
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    target.writestr(info, content)
                if defect == "extra":
                    target.writestr("docs/CONTROL.md", b"private work record")
                if defect == "duplicate":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        target.writestr(selected, docs[selected])
            destination = self.root / ("document-" + defect + "-installed")
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                extract_cli_zip(broken, destination, binary, alternate, licenses, True, docs)
            self.assertFalse(destination.exists())

    def test_document_maps_and_collisions_are_rejected_before_archive_creation(self):
        binary, alternate, licenses = self.zip_inputs()
        docs = self.documentation_inputs()
        invalid_maps = [{}, {**docs, "docs/CONTROL.md": b"private"},
                        {**docs, "../outside": b"escape"}, {**docs, "README.md": b"\xff"}]
        for index, supplied in enumerate(invalid_maps):
            archive = self.root / f"invalid-map-{index}.zip"
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                write_cli_zip(binary, alternate, archive, licenses, True, supplied)
            self.assertFalse(archive.exists())
        for index, collision in enumerate(["README.md", "readme.MD", "docs", "DOCS/cli.md", "docs/cli.md/child"]):
            supplied = {**licenses, collision: b"notice"}
            for suffix in ["zip", "tar.gz"]:
                archive = self.root / f"collision-{index}.{suffix}"
                with self.subTest(name=collision, kind=suffix), self.assertRaises(RuntimeError):
                    if suffix == "zip":
                        write_cli_zip(binary, alternate, archive, supplied, True, docs)
                    else:
                        write_single_binary_tar(binary, archive, supplied, docs)
                self.assertFalse(archive.exists())

    def test_tar_document_defects_fail_before_any_extraction(self):
        binary, _, licenses = self.zip_inputs()
        docs = self.documentation_inputs()
        valid = self.root / "documented.tar.gz"
        write_single_binary_tar(binary, valid, licenses, docs)
        selected = "docs/cli.md"
        for defect in ["missing", "duplicate", "changed", "symlink", "extra"]:
            broken = self.root / ("document-" + defect + ".tar.gz")
            with tarfile.open(valid, "r:gz") as source, tarfile.open(broken, "x:gz") as target:
                for member in source.getmembers():
                    if defect == "missing" and member.name == selected:
                        continue
                    content = source.extractfile(member).read() if member.isfile() else None
                    if defect == "changed" and member.name == selected:
                        content = bytes([content[0] ^ 1]) + content[1:]
                    if defect == "symlink" and member.name == selected:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../outside"
                        member.size = 0
                        content = None
                    target.addfile(member, io.BytesIO(content) if content is not None else None)
                if defect in {"extra", "duplicate"}:
                    member = tarfile.TarInfo("docs/CONTROL.md" if defect == "extra" else selected)
                    member.mode = 0o644
                    content = b"private" if defect == "extra" else docs[selected]
                    member.size = len(content)
                    target.addfile(member, io.BytesIO(content))
            destination = self.root / ("tar-" + defect + "-installed")
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                extract_single_binary_tar(broken, destination, binary, licenses, docs)
            self.assertFalse(destination.exists())
        self.assertFalse((self.root / "outside").exists())

    @unittest.skipIf(os.name == "nt", "Unix tar installation creates relative symlinks")
    def test_tar_offline_guides_preserve_relative_aliases_and_all_notices(self):
        binary, _, licenses = self.zip_inputs()
        docs = self.documentation_inputs()
        archive = self.root / "documented.tar.gz"
        write_single_binary_tar(binary, archive, licenses, docs)
        installed = self.root / "离线 tar 安装"
        record = extract_single_binary_tar(archive, installed, binary, licenses, docs)
        self.assertEqual(set(record["attribution"]), set(licenses))
        self.assertEqual(set(record["documentation"]), set(DOCUMENTATION_PATHS))
        for name, content in {**licenses, **docs}.items():
            self.assertEqual((installed / name).read_bytes(), content)
        for name in ["mkai", "markitai-mcp"]:
            self.assertEqual(os.readlink(installed / name), "markitai")
            self.assertEqual(identity(installed / name), identity(binary))

    def test_common_packages_include_all_portable_notices(self):
        source = Path(__file__).resolve().parents[1]
        files = package_attribution(source)
        expected = {"licenses/hayro/" + name for name in ["LICENSE_FOXIT", "CGATS_LICENSE.txt", "CMAP_LICENSE.txt",
                    "hayro-interpret-assets-README.md", "README.md"]} | {
                    "licenses/paddleocr/" + name for name in ["PaddleOCR-LICENSE", "RapidOCR-LICENSE", "README.md", "provenance.json"]}
        self.assertTrue(expected <= set(files))
        for name in expected:
            self.assertEqual(files[name], (source / name).read_bytes())

    def test_doctor_readiness_is_reported_without_automatic_repair(self):
        for returncode in [0, 1]:
            report = {"ocr": {"status": "warning", "message": "Local models are not installed", "optional": True}}
            result = subprocess.CompletedProcess([], returncode, json.dumps(report).encode(), b"")
            with patch("ci_packages.subprocess.run", return_value=result) as run:
                observed = doctor_probe(self.root / "markitai", self.root, {}, self.root / "doctor.log")
            self.assertEqual(observed["configuration_ready"], returncode == 0)
            self.assertEqual(observed["checks"], report)
            self.assertEqual(run.call_args.args[0], [str(self.root / "markitai"), "doctor", "--json"])
            self.assertEqual(bool(observed["limitations"]), returncode != 0)
        for returncode, payload in [(2, b'{"ocr":{"status":"error"}}'), (0, b'{}'), (0, b'{"ocr":{"status":"unknown"}}'), (1, b'[]')]:
            with patch("ci_packages.subprocess.run", return_value=subprocess.CompletedProcess([], returncode, payload, b"")):
                with self.assertRaises(RuntimeError):
                    doctor_probe(self.root / "markitai", self.root, {}, self.root / "invalid-doctor.log")

    def test_mcp_probe_uses_real_pipes_and_rejects_partial_protocol(self):
        # An independent Python peer tests transport/error handling, not Markitai support.
        program = self.root / "peer.py"
        program.write_text("""import json,sys,time
mode=sys.argv[1]
for line in sys.stdin:
    request=json.loads(line)
    method=request['method']
    if method=='initialize':
        if mode=='timeout':
            time.sleep(5)
        result={'protocolVersion':'2025-11-25'}
    elif method=='tools/list':
        names=['convert_document','convert_url','batch_convert','job_status']
        result={'tools':[{'name':n} for n in names[:3 if mode=='partial' else 4]]}
    else:
        continue
    response={'jsonrpc':'2.0','id':request['id']+(1 if mode=='wrong-id' else 0),'result':result}
    print(json.dumps(response),flush=True)
""", encoding="utf-8")
        real_popen = subprocess.Popen
        for mode in ["valid", "partial", "wrong-id", "timeout"]:
            def start(command, **kwargs):
                return real_popen([sys.executable, str(program), mode], **kwargs)
            with patch("ci_packages.subprocess.Popen", side_effect=start):
                if mode == "valid":
                    observed = mcp_probe(program, self.root, os.environ.copy(), self.root / "peer-valid.log", timeout=2)
                    self.assertEqual(observed["exit_code"], 0)
                    self.assertEqual(len(observed["tools"]), 4)
                else:
                    with self.assertRaises(Exception):
                        mcp_probe(program, self.root, os.environ.copy(), self.root / (mode + ".log"), timeout=0.2)

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

    def test_windows_zip_contains_regular_mcp_bytes_with_matching_identity(self):
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

    @unittest.skipIf(os.name == "nt", "Unix ZIP aliases are extracted on a native Unix host")
    def test_unix_zip_keeps_the_relative_mcp_symlink(self):
        binary = self.root / "markitai"
        binary.write_bytes(b"native CLI")
        archive = self.root / "unix.zip"
        write_cli_zip(binary, binary, archive, {}, False)
        with zipfile.ZipFile(archive) as bundle:
            alias = bundle.getinfo("markitai-mcp")
            self.assertEqual(alias.external_attr >> 16, 0o120777)
            self.assertEqual(bundle.read(alias), b"markitai")
        installed = self.root / "Unix 安装 with spaces"
        record = extract_cli_zip(archive, installed, binary, binary, {}, False)
        self.assertTrue((installed / "markitai-mcp").is_symlink())
        self.assertEqual(os.readlink(installed / "markitai-mcp"), "markitai")
        self.assertEqual(record["mcp_alias"]["kind"], "symlink")
        self.assertEqual(identity(installed / "markitai-mcp"), identity(binary))

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

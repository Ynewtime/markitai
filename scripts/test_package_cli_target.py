"""Counterexamples for cross-target CLI packaging; nothing is built."""
from pathlib import Path
import json
import os
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from package_cli_target import executable_arch, macho_minimum, main, workspace_version, validate_target


class TargetPackagingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def file(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_thin_64_bit_executables_name_their_instruction_set(self):
        macho = lambda cpu: b"\xcf\xfa\xed\xfe" + struct.pack("<I", cpu) + bytes(24)
        elf = lambda machine, width=2: b"\x7fELF" + bytes([width, 1]) + bytes(12) + struct.pack("<H", machine) + bytes(8)
        self.assertEqual(executable_arch(self.file("a", macho(0x01000007))), "x86_64")
        self.assertEqual(executable_arch(self.file("b", macho(0x0100000C))), "aarch64")
        self.assertEqual(executable_arch(self.file("c", elf(62))), "x86_64")
        self.assertEqual(executable_arch(self.file("d", elf(183))), "aarch64")
        # A universal binary, a 32-bit ELF, an unknown CPU and a short file name none.
        self.assertIsNone(executable_arch(self.file("e", b"\xca\xfe\xba\xbe" + bytes(28))))
        self.assertIsNone(executable_arch(self.file("f", elf(62, width=1))))
        self.assertIsNone(executable_arch(self.file("g", macho(0x01000012))))
        self.assertIsNone(executable_arch(self.file("h", b"\xcf\xfa\xed\xfe")))

    def test_the_minimum_macos_comes_from_the_first_version_command(self):
        def macho(*commands):
            body = b"".join(commands)
            return (b"\xcf\xfa\xed\xfe" + struct.pack("<IIIII", 0x01000007, 3, 2, len(commands), len(body))
                    + bytes(8) + body)
        uuid = struct.pack("<II", 0x1B, 24) + bytes(16)
        build = lambda platform, minos: struct.pack("<IIIIII", 0x32, 24, platform, minos, 0x1B0000, 0)
        legacy = struct.pack("<IIII", 0x24, 16, 0x0A0C00, 0x1B0000)
        self.assertEqual(macho_minimum(self.file("a", macho(uuid, build(1, 0x0B0000)))), "11.0")
        self.assertEqual(macho_minimum(self.file("b", macho(legacy))), "10.12")
        self.assertEqual(macho_minimum(self.file("c", macho(build(1, 0x0D0301)))), "13.3.1")
        # An iOS build version is not a macOS minimum; a malformed command stops the walk.
        self.assertIsNone(macho_minimum(self.file("d", macho(build(2, 0x0B0000)))))
        self.assertIsNone(macho_minimum(self.file("e", macho(struct.pack("<II", 0x1B, 0)))))
        self.assertIsNone(macho_minimum(self.file("f", b"\x7fELF" + bytes(60))))


    def test_windows_support_requires_the_exact_native_msvc64_host(self):
        for target in ["x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc"]:
            self.assertTrue(validate_target(target, target, "win32"))
            for host, platform in [(target, "darwin"), ("x86_64-unknown-linux-gnu", "linux"),
                                   ("aarch64-pc-windows-msvc" if target.startswith("x86_64") else "x86_64-pc-windows-msvc", "win32")]:
                with self.assertRaisesRegex(SystemExit, "matching native"):
                    validate_target(target, host, platform)
        for target in ["i686-pc-windows-msvc", "x86_64-pc-windows-gnu"]:
            with self.assertRaisesRegex(SystemExit, "only"):
                validate_target(target, target, "win32")
        with self.assertRaisesRegex(SystemExit, "Unix host"):
            validate_target("aarch64-apple-darwin", "x86_64-pc-windows-msvc", "win32")
        self.assertFalse(validate_target("x86_64-apple-darwin", "aarch64-apple-darwin", "darwin"))

    def test_the_version_comes_from_the_workspace_package_section(self):
        manifest = '[package]\nversion = "0.1.0"\n\n[workspace.package]\nedition = "2024"\nversion = "1.3.0-dev"\n'
        self.assertEqual(workspace_version(manifest), "1.3.0-dev")
        with self.assertRaisesRegex(RuntimeError, "workspace.package"):
            workspace_version('[package]\nversion = "0.1.0"\n')

    def test_the_host_target_and_malformed_triples_are_refused_before_any_build(self):
        host = next(line.split(": ", 1)[1] for line in
                    subprocess.check_output(["rustc", "-vV"], text=True).splitlines() if line.startswith("host: "))
        cases = [("x86_64", "triple"), ("x86_64-apple-darwin;rm", "triple"),
                                ("riscv64gc-unknown-linux-gnu", "Unsupported"),
                                ("x86_64-pc-windows-gnu", "Windows")]
        if "windows" not in host:
            cases.extend([(host, "ci_packages.py"), ("x86_64-pc-windows-msvc", "Windows")])
        for target, message in cases:
            if target == "x86_64-pc-windows-msvc" and host == target:
                continue
            with self.assertRaises(SystemExit) as raised:
                main(["--target", target, "--output", str(self.root / "out")])
            self.assertIn(message, str(raised.exception))
        self.assertFalse((self.root / "out").exists())

    def test_windows_standalone_build_uses_static_cli_target_and_isolated_environment(self):
        host = "aarch64-pc-windows-msvc"
        output = self.root / "standalone-output"
        calls = []
        root = Path(__file__).resolve().parents[1]

        def check_output(command, **kwargs):
            if command[0] == "rustc":
                return f"host: {host}\n"
            if command[1] == "ls-files":
                return b""
            return "" if command[1] == "status" else "authored-source-revision"

        def run(command, **kwargs):
            calls.append((command, kwargs["env"].copy()))
            if command[0] == "rustc":
                kwargs["stdout"].write(f"host: {host}\n".encode())
            return subprocess.CompletedProcess(command, 0)

        with (patch("package_cli_target.sys.platform", "win32"),
              patch.dict(os.environ, {"HOME": str(self.root), "CARGO_TARGET_DIR": "target with spaces",
                                      "CARGO_ENCODED_RUSTFLAGS": "-Cdebuginfo=1"}, clear=True),
              patch("package_cli_target.subprocess.check_output", side_effect=check_output),
              patch("package_cli_target.subprocess.run", side_effect=run),
              patch("package_cli_target.identity", return_value={}),
              patch("package_cli_target.verify_target_executable", side_effect=RuntimeError("stop before execution")) as verify):
            with self.assertRaisesRegex(RuntimeError, "stop before execution"):
                main(["--target", host, "--output", str(output)])
        builds = [(command, env) for command, env in calls if command[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        command, child = builds[0]
        self.assertEqual(command[command.index("--target") + 1], host)
        self.assertEqual(child["CARGO_ENCODED_RUSTFLAGS"], "-Cdebuginfo=1\x1f-Ctarget-feature=+crt-static")
        self.assertEqual(calls[0][1]["CARGO_ENCODED_RUSTFLAGS"], "-Cdebuginfo=1")
        release = root / "target with spaces/cli-static-crt" / host / "release"
        verify.assert_called_once_with(release / "markitai.exe", host)
        record = json.loads((output / "record.json").read_text())
        self.assertEqual(record["cli_build"]["release_directory"], str(release))
        self.assertEqual(record["status"], "failed")


if __name__ == "__main__":
    unittest.main()

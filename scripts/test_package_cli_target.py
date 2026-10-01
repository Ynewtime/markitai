"""Counterexamples for cross-target CLI packaging; nothing is built."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from package_cli_target import executable_arch, main, workspace_version


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

    def test_the_version_comes_from_the_workspace_package_section(self):
        manifest = '[package]\nversion = "0.1.0"\n\n[workspace.package]\nedition = "2024"\nversion = "1.3.0-dev"\n'
        self.assertEqual(workspace_version(manifest), "1.3.0-dev")
        with self.assertRaisesRegex(RuntimeError, "workspace.package"):
            workspace_version('[package]\nversion = "0.1.0"\n')

    def test_the_host_target_and_malformed_triples_are_refused_before_any_build(self):
        host = next(line.split(": ", 1)[1] for line in
                    subprocess.check_output(["rustc", "-vV"], text=True).splitlines() if line.startswith("host: "))
        for target, message in [(host, "ci_packages.py"), ("x86_64", "triple"), ("x86_64-apple-darwin;rm", "triple"),
                                ("riscv64gc-unknown-linux-gnu", "Unsupported"),
                                ("x86_64-pc-windows-msvc", "Windows")]:
            if target == "x86_64-pc-windows-msvc" and host == target:
                continue
            with self.assertRaises(SystemExit) as raised:
                main(["--target", target, "--output", str(self.root / "out")])
            self.assertIn(message, str(raised.exception))
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()

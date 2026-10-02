"""Synthetic headers test static identity only, never native execution."""
from pathlib import Path
import struct
import tempfile
import unittest

from executable_identity import executable_info, executable_arch, verify_target_executable


def pe(machine=0x8664, offset=128, sections=1, optional_size=240, flags=0x22, magic=0x20B, subsystem=3):
    data = bytearray(max(64, offset + 24 + optional_size + 40 * sections))
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, offset)
    if offset >= 64:
        data[offset:offset + 4] = b"PE\0\0"
        struct.pack_into("<HH", data, offset + 4, machine, sections)
        struct.pack_into("<HH", data, offset + 20, optional_size, flags)
        if optional_size >= 70:
            struct.pack_into("<H", data, offset + 24, magic)
            struct.pack_into("<H", data, offset + 24 + 68, subsystem)
    return data


class ExecutableIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "candidate"

    def write(self, data):
        self.path.write_bytes(data)
        return self.path

    def test_regular_amd64_and_arm64_console_headers_are_distinct(self):
        for machine, arch in [(0x8664, "x86_64"), (0xAA64, "aarch64")]:
            info = executable_info(self.write(pe(machine)))
            self.assertEqual(info, {"format": "PE32+", "arch": arch, "machine": f"0x{machine:04x}", "subsystem": "console"})
            self.assertEqual(verify_target_executable(self.path, arch + "-pc-windows-msvc"), info)
            other = "x86_64" if arch == "aarch64" else "aarch64"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                verify_target_executable(self.path, other + "-pc-windows-msvc")

    def test_wrong_machine_dll_pe32_gui_and_incomplete_tables_are_rejected(self):
        for parameters in [{"machine": 0x14C}, {"machine": 0xA641}, {"machine": 0xA64E},
                           {"flags": 0x2022}, {"flags": 0}, {"magic": 0x10B}, {"subsystem": 2},
                           {"sections": 0}, {"sections": 97}, {"optional_size": 70}, {"optional_size": 4097}]:
            with self.subTest(parameters=parameters):
                self.assertIsNone(executable_info(self.write(pe(**parameters))))
        self.assertIsNone(executable_info(self.write(pe()[:-1])))
        self.assertIsNone(executable_info(self.write(pe()[:150])))

    def test_dos_offset_and_signature_are_bounded(self):
        for offset in [0, 63, 1024 * 1024 + 1, 0xFFFFFFFF]:
            data = pe()
            struct.pack_into("<I", data, 0x3C, offset)
            self.assertIsNone(executable_info(self.write(data)))
        data = pe()
        data[128:132] = b"NOPE"
        self.assertIsNone(executable_info(self.write(data)))
        self.assertIsNone(executable_info(self.write(b"MZ")))

    def test_same_architecture_in_a_foreign_format_does_not_pass(self):
        elf = b"\x7fELF\x02\x01" + bytes(12) + struct.pack("<H", 62) + bytes(44)
        self.assertEqual(executable_arch(self.write(elf)), "x86_64")
        with self.assertRaises(RuntimeError):
            verify_target_executable(self.path, "x86_64-pc-windows-msvc")
        with self.assertRaises(RuntimeError):
            verify_target_executable(self.write(pe()), "x86_64-unknown-linux-gnu")
        with self.assertRaises(RuntimeError):
            verify_target_executable(self.path, "x86_64-pc-windows-gnu")

    def test_symlink_is_not_an_executable_identity(self):
        real = self.root / "real"
        real.write_bytes(pe())
        try:
            self.path.symlink_to(real)
        except OSError as error:
            self.skipTest(f"This host cannot create a test symlink: {error}")
        with self.assertRaisesRegex(RuntimeError, "non-symlink"):
            executable_info(self.path)


if __name__ == "__main__":
    unittest.main()

"""Synthetic headers test static identity only, never native execution."""
from pathlib import Path
import struct
import tempfile
import unittest

from executable_identity import (executable_info, executable_arch, verify_target_executable,
                                 windows_imports, verify_windows_cli_runtime)


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


def importing_pe(normal=b"KERNEL32.dll", delay=b"USER32.dll", machine=0x8664):
    data = pe(machine=machine)
    data.extend(bytes(2048 - len(data)))
    optional = 128 + 24
    struct.pack_into("<Q", data, optional + 24, 0x140000000)
    struct.pack_into("<I", data, optional + 60, 512)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<IIII", data, optional + 240 + 8, 1536, 0x1000, 1536, 512)
    struct.pack_into("<II", data, optional + 112 + 8, 0x1000, 40)
    struct.pack_into("<II", data, optional + 112 + 13 * 8, 0x1080, 64)
    struct.pack_into("<I", data, 512 + 12, 0x1200)
    struct.pack_into("<II", data, 512 + 128, 1, 0x1300)
    data[1024:1024 + len(normal) + 1] = normal + b"\0"
    data[1280:1280 + len(delay) + 1] = delay + b"\0"
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

    def test_normal_and_delay_imports_checked_on_both_windows_architectures(self):
        for machine, target in [(0x8664, "x86_64-pc-windows-msvc"), (0xAA64, "aarch64-pc-windows-msvc")]:
            info = verify_windows_cli_runtime(self.write(importing_pe(machine=machine)), target)
            self.assertEqual(info["imports"], ["kernel32.dll"])
            self.assertEqual(info["delay_imports"], ["user32.dll"])
            for key in ["normal", "delay"]:
                for dll in [b"VCRUNTIME140.dll", b"MSVCP140_ATOMIC_WAIT.dll", b"ucrtbased.dll", b"CONCRT140.dll"]:
                    with self.subTest(key=key, dll=dll):
                        with self.assertRaisesRegex(RuntimeError, "non-system CRT"):
                            verify_windows_cli_runtime(self.write(importing_pe(machine=machine, **{key: dll})), target)

    def test_system_ucrt_is_allowed_but_non_windows_target_is_rejected(self):
        self.write(importing_pe(normal=b"api-ms-win-crt-runtime-l1-1-0.dll", delay=b"ucrtbase.dll"))
        verify_windows_cli_runtime(self.path, "x86_64-pc-windows-msvc")
        self.write(importing_pe(normal=b"WINSPOOL.DRV", delay=b"msvcp_win.dll"))
        info = verify_windows_cli_runtime(self.path, "x86_64-pc-windows-msvc")
        self.assertEqual(info["imports"], ["winspool.drv"])
        self.assertEqual(info["delay_imports"], ["msvcp_win.dll"])
        with self.assertRaisesRegex(RuntimeError, "MSVC target"):
            verify_windows_cli_runtime(self.path, "x86_64-unknown-linux-gnu")

    def test_legacy_delay_virtual_address_is_resolved(self):
        data = importing_pe(delay=b"VCRUNTIME140.dll")
        struct.pack_into("<Q", data, 152 + 24, 0x400000)
        struct.pack_into("<II", data, 640, 0, 0x401300)
        self.assertEqual(windows_imports(self.write(data))["delay_imports"], ["vcruntime140.dll"])

    def test_malformed_tables_cannot_certify_missing_crt(self):
        for kind in ["missing-terminator", "unmapped-name", "virtual-only", "huge-directory",
                     "reserved-delay-flags", "bad-directory-count", "path-name", "unterminated-name"]:
            data = importing_pe()
            if kind == "missing-terminator":
                struct.pack_into("<I", data, 152 + 112 + 8 + 4, 20)
            elif kind == "unmapped-name":
                struct.pack_into("<I", data, 512 + 12, 0xFFFFFFFF)
            elif kind == "virtual-only":
                struct.pack_into("<I", data, 392 + 8, 4096)
                struct.pack_into("<I", data, 512 + 12, 0x1800)
            elif kind == "huge-directory":
                struct.pack_into("<I", data, 152 + 112 + 8 + 4, 0xFFFFFFFF)
            elif kind == "reserved-delay-flags":
                struct.pack_into("<I", data, 640, 3)
            elif kind == "bad-directory-count":
                struct.pack_into("<I", data, 152 + 108, 17)
            elif kind == "path-name":
                data[1024:1034] = b"../bad.dll"
            else:
                data[1024:1285] = b"x" * 261
            with self.subTest(kind=kind):
                with self.assertRaises(RuntimeError):
                    windows_imports(self.write(data))


if __name__ == "__main__":
    unittest.main()

"""Bounded instruction-set probes for supported thin executable formats.

PE fields follow Microsoft's PE/COFF specification:
https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
A header identity does not demonstrate that an executable runs on its target.
"""
from pathlib import Path
import stat
import struct

MACHO = {0x01000007: "x86_64", 0x0100000C: "aarch64"}
ELF = {62: "x86_64", 183: "aarch64"}
PE = {0x8664: "x86_64", 0xAA64: "aarch64"}
WINDOWS_TARGETS = {f"{arch}-pc-windows-msvc" for arch in PE.values()}
MAX_PE_OFFSET = 1024 * 1024


def executable_info(path):
    """Identify Mach-O/ELF64 or a bounded PE32+ console executable, else None."""
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise RuntimeError("Executable probe requires a regular non-symlink file")
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 20:
            return None
        if header[:4] == b"\xcf\xfa\xed\xfe":
            arch = MACHO.get(struct.unpack_from("<I", header, 4)[0])
            return {"format": "Mach-O", "arch": arch} if arch else None
        if header[:4] == b"\x7fELF" and header[4:6] == b"\x02\x01":
            arch = ELF.get(struct.unpack_from("<H", header, 18)[0])
            return {"format": "ELF", "arch": arch} if arch else None
        if len(header) != 64 or header[:2] != b"MZ":
            return None
        offset = struct.unpack_from("<I", header, 0x3C)[0]
        if not 64 <= offset <= MAX_PE_OFFSET or offset + 24 > info.st_size:
            return None
        stream.seek(offset)
        coff = stream.read(24)
        if coff[:4] != b"PE\0\0":
            return None
        machine, sections = struct.unpack_from("<HH", coff, 4)
        optional_size, flags = struct.unpack_from("<HH", coff, 20)
        arch = PE.get(machine)
        if (arch is None or not 1 <= sections <= 96 or not 112 <= optional_size <= 4096
                or not flags & 0x2 or flags & 0x2000
                or offset + 24 + optional_size + sections * 40 > info.st_size):
            return None
        optional = stream.read(optional_size)
        if struct.unpack_from("<H", optional)[0] != 0x20B or struct.unpack_from("<H", optional, 68)[0] != 3:
            return None
        return {"format": "PE32+", "arch": arch, "machine": f"0x{machine:04x}", "subsystem": "console"}


def executable_arch(path):
    info = executable_info(path)
    return info["arch"] if info else None


def verify_target_executable(path, target):
    expected_format = ("PE32+" if target in WINDOWS_TARGETS else "Mach-O" if target.endswith("-apple-darwin")
                       else "ELF" if "-linux-" in target else None)
    info = executable_info(path)
    if (expected_format is None or info is None or info["format"] != expected_format
            or info["arch"] != target.split("-", 1)[0]):
        raise RuntimeError(f"Executable format/architecture does not match {target}")
    return info

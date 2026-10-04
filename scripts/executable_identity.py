"""Bounded instruction-set probes for supported thin executable formats.

PE fields follow Microsoft's PE/COFF specification:
https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
A header identity does not demonstrate that an executable runs on its target.
"""
from pathlib import Path
import re
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


def windows_imports(path):
    """Read bounded normal/delay DLL names from a PE32+ executable.

    Uses file-backed RVA ranges, never maps/executes the candidate. Invalid or
    unterminated tables fail closed instead of certifying absent dependencies.
    """
    info = executable_info(path)
    if not info or info["format"] != "PE32+":
        raise RuntimeError("DLL inspection requires a PE32+ executable")
    with Path(path).open("rb") as stream:
        size = stream.seek(0, 2)

        def read(offset, length):
            if offset < 0 or length < 0 or offset + length > size:
                raise RuntimeError("PE import data lies outside the executable")
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise RuntimeError("PE import data was truncated")
            return data

        pe_offset = struct.unpack("<I", read(0x3C, 4))[0]
        coff = read(pe_offset, 24)
        count = struct.unpack_from("<H", coff, 6)[0]
        optional_size = struct.unpack_from("<H", coff, 20)[0]
        optional = read(pe_offset + 24, optional_size)
        image_base = struct.unpack_from("<Q", optional, 24)[0]
        headers = struct.unpack_from("<I", optional, 60)[0]
        directories = struct.unpack_from("<I", optional, 108)[0]
        if directories > (optional_size - 112) // 8 or headers > size:
            raise RuntimeError("Invalid PE import directory/header extent")
        ranges = [(0, headers, 0)] if headers else []
        for index in range(count):
            section = read(pe_offset + 24 + optional_size + index * 40, 40)
            virtual_size, rva, raw_size, raw = struct.unpack_from("<IIII", section, 8)
            extent = min(raw_size, virtual_size) if virtual_size else raw_size
            if raw + raw_size > size or rva + extent > 0x100000000:
                raise RuntimeError("Invalid PE section extent")
            if extent:
                ranges.append((rva, extent, raw))

        def locate(rva, length):
            matches = [(raw + rva - start, extent - (rva - start))
                       for start, extent, raw in ranges
                       if start <= rva and rva + length <= start + extent]
            if len(matches) != 1:
                raise RuntimeError("Unmapped or ambiguous PE import RVA")
            return matches[0]

        def dll_name(rva):
            offset, available = locate(rva, 1)
            data = read(offset, min(available, 261))
            end = data.find(b"\0")
            if end < 1 or end > 260:
                raise RuntimeError("Invalid or unterminated PE DLL name")
            try:
                name = data[:end].decode("ascii")
            except UnicodeDecodeError as error:
                raise RuntimeError("Non-ASCII PE DLL name") from error
            # The PE loader also accepts system modules such as WINSPOOL.DRV.
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", name):
                raise RuntimeError("Invalid PE DLL basename")
            return name.lower()

        result = {}
        for label, index, width, name_offset in [("imports", 1, 20, 12),
                                                  ("delay_imports", 13, 32, 4)]:
            names = []
            result[label] = names
            if directories <= index:
                continue
            rva, length = struct.unpack_from("<II", optional, 112 + index * 8)
            if not rva and not length:
                continue
            if not rva or length < width or length > width * 4096:
                raise RuntimeError("Invalid PE import table size")
            offset, _ = locate(rva, length)
            for position in range(0, length - width + 1, width):
                entry = read(offset + position, width)
                if not any(entry):
                    break
                name_rva = struct.unpack_from("<I", entry, name_offset)[0]
                if index == 13:
                    attributes = struct.unpack_from("<I", entry)[0]
                    if attributes & ~1:
                        raise RuntimeError("Unsupported PE delay import attributes")
                    if not attributes & 1:
                        name_rva -= image_base
                names.append(dll_name(name_rva))
            else:
                raise RuntimeError("Unterminated PE import table")
            result[label] = sorted(set(names))
        return result


def verify_windows_cli_runtime(path, target):
    """Reject redistributable/debug CRT dependencies in standalone Windows CLI.

    This checks direct normal and delay imports, not clean-machine execution or
    arbitrary runtime LoadLibrary calls. Windows system UCRT remains permitted.
    """
    if target not in WINDOWS_TARGETS:
        raise RuntimeError("Windows CLI runtime inspection requires an MSVC target")
    info = verify_target_executable(path, target)
    imports = windows_imports(path)
    forbidden = [name for names in imports.values() for name in names
                 if re.match(r"(?:vcruntime|msvcp|msvcr|concrt|vccorlib)\d", name)
                 or name == "ucrtbased.dll"]
    if forbidden:
        raise RuntimeError("Standalone Windows CLI requires non-system CRT: " + ", ".join(sorted(set(forbidden))))
    return {**info, **imports, "redistributable_crt_imports": []}

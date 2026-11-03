#!/usr/bin/env python3.13
"""Build the Windows app: the markitai_app.dll core, the app icon, and Markitai.exe.

Run on Windows from any folder:

    python3.13 native/windows/build.py            # build; prints the app folder
    python3.13 native/windows/build.py --check    # build, then run Markitai.Core.Tests

The core is built with a static C runtime (as the Windows CLI is), so the app
folder needs no Visual C++ redistributable; the build fails if the DLL still
imports one. Tests run with HOME, USERPROFILE, LOCALAPPDATA, APPDATA and
MARKITAI_HOME inside --state, never the user's own folders.
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
import zlib
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = "x86_64-pc-windows-msvc"
APP = HERE / "Markitai.App"
TESTS = HERE / "Markitai.Core.Tests"
ICON_SOURCE = REPO / "desktop" / "icon.png"
ICON = APP / "Assets" / "Markitai.ico"
ICON_SIZES = (256, 128, 64, 48, 32, 24, 16)
TFM = "net10.0-windows10.0.26100.0"


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def version():
    with open(REPO / "Cargo.toml", "rb") as file:
        return tomllib.load(file)["workspace"]["package"]["version"]


def build_core():
    """cargo rustc for the cdylib with +crt-static, in its own target folder so other builds keep their flags."""
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR") or REPO / "target") / "native-windows"
    env = dict(os.environ)
    flags = env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    flags = flags.split("\x1f") if flags else env.pop("RUSTFLAGS", "").split()
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join([*flags, "-Ctarget-feature=+crt-static"])
    run(["cargo", "rustc", "-p", "markitai-app", "--release", "--locked", "--lib", "--crate-type", "cdylib",
         "--target", TARGET, "--target-dir", str(target_dir)], cwd=REPO, env=env)
    return target_dir / TARGET / "release" / "markitai_app.dll"


def dll_imports(path):
    """The DLLs a PE32+ image imports, normal and delay-loaded, in lower case.

    scripts/executable_identity.py reads console executables only, so the DLL gets
    this reader of the two import directories (PE/COFF specification).
    """
    data = Path(path).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
        raise SystemExit(f"{path} is not a PE image")
    sections, optional_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        raise SystemExit(f"{path} is not PE32+")
    table = [struct.unpack_from("<IIII", data, optional + optional_size + i * 40 + 8) for i in range(sections)]

    def offset(rva):
        for virtual_size, start, raw_size, raw in table:
            if start <= rva < start + max(virtual_size, raw_size):
                return raw + rva - start
        raise SystemExit(f"{path}: RVA {rva:#x} lies outside every section")

    names = []
    # Data directory 1 holds 20-byte import descriptors, 13 holds 32-byte delay-load ones.
    for directory, size, name_field in ((1, 20, 12), (13, 32, 4)):
        rva = struct.unpack_from("<I", data, optional + 112 + directory * 8)[0]
        at = offset(rva) if rva else len(data)
        while any(entry := data[at:at + size]):
            name = offset(struct.unpack_from("<I", entry, name_field)[0])
            names.append(data[name:data.index(b"\0", name)].decode("ascii").lower())
            at += size
    return names


def check_runtime(dll):
    names = sorted(set(dll_imports(dll)))
    redistributable = [name for name in names if name.startswith(("vcruntime", "msvcp", "msvcr", "concrt", "vccorlib")) or name == "ucrtbased.dll"]
    print(f"{dll.name} imports: {', '.join(names)}")
    if redistributable:
        raise SystemExit(f"{dll.name} needs the Visual C++ redistributable: {', '.join(redistributable)}")


# --- The app icon: desktop/icon.png scaled into a PNG-compressed .ico ---------

def read_png(path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} is not a PNG")
    at, idat = 8, bytearray()
    while at < len(data):
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, color, interlace) != (8, 6, 0):
                raise SystemExit(f"{path} must be 8-bit RGBA, not interlaced")
        elif kind == b"IDAT":
            idat += body
        at += 12 + length
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    pixels = bytearray(height * stride)
    previous = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        if kind == 1:
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                b = previous[i]
                c = previous[i - 4] if i >= 4 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        pixels[y * stride:(y + 1) * stride] = line
        previous = line
    return width, height, pixels


def scale(width, height, pixels, size):
    """Area average with premultiplied alpha, so transparent edges do not darken."""
    out = bytearray(size * size * 4)
    for oy in range(size):
        y0, y1 = oy * height // size, max((oy + 1) * height // size, oy * height // size + 1)
        for ox in range(size):
            x0, x1 = ox * width // size, max((ox + 1) * width // size, ox * width // size + 1)
            r = g = b = a = 0
            for y in range(y0, y1):
                row = y * width * 4
                for x in range(x0, x1):
                    i = row + x * 4
                    alpha = pixels[i + 3]
                    r += pixels[i] * alpha
                    g += pixels[i + 1] * alpha
                    b += pixels[i + 2] * alpha
                    a += alpha
            o = (oy * size + ox) * 4
            count = (y1 - y0) * (x1 - x0)
            if a:
                out[o:o + 4] = bytes((round(r / a), round(g / a), round(b / a), round(a / count)))
    return out


def png(size, pixels):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    stride = size * 4
    raw = b"".join(b"\x00" + bytes(pixels[y * stride:(y + 1) * stride]) for y in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def build_icon():
    """Markitai.ico from desktop/icon.png; rebuilt only when the PNG is newer."""
    if ICON.exists() and ICON.stat().st_mtime >= ICON_SOURCE.stat().st_mtime:
        return
    print(f"+ icon {ICON_SOURCE} -> {ICON}", flush=True)
    width, height, pixels = read_png(ICON_SOURCE)
    # Halve to 256 first so the smaller sizes average fewer pixels.
    base_size = min(256, width)
    base = scale(width, height, pixels, base_size)
    images = [png(size, base if size == base_size else scale(base_size, base_size, base, size)) for size in ICON_SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for size, image in zip(ICON_SIZES, images):
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(image), offset)
        offset += len(image)
    ICON.parent.mkdir(parents=True, exist_ok=True)
    ICON.write_bytes(header + entries + b"".join(images))


# --- The app and the tests --------------------------------------------------

def build_app(configuration):
    run(["dotnet", "build", str(APP / "Markitai.App.csproj"), "-c", configuration, "-p:Platform=x64",
         f"-p:Version={version()}"], cwd=HERE)
    folder = APP / "bin" / "x64" / configuration / TFM / "win-x64"
    exe = folder / "Markitai.exe"
    if not exe.exists():
        raise SystemExit(f"{exe} was not built")
    return folder


def isolated_env(state):
    """Every user-state variable the core or .NET reads, inside state."""
    env = dict(os.environ)
    home = state / "home"
    for name, path in {
        "HOME": home,
        "USERPROFILE": home,
        "LOCALAPPDATA": home / "AppData" / "Local",
        "APPDATA": home / "AppData" / "Roaming",
        "MARKITAI_HOME": state / "markitai-home",
        "TEMP": state / "temp",
        "TMP": state / "temp",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["DOTNET_NOLOGO"] = "1"
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    return env


def run_tests(dll, state):
    """Builds with the developer's environment (NuGet cache and all), then runs with the isolated one."""
    project = str(TESTS / "Markitai.Core.Tests.csproj")
    run(["dotnet", "build", project, "-c", "Release"], cwd=HERE)
    env = isolated_env(state)
    env["MARKITAI_APP_LIBRARY"] = str(dll)
    results = state / "results"
    shutil.rmtree(results, ignore_errors=True)
    run(["dotnet", "test", project, "-c", "Release", "--no-build", "--logger", "console;verbosity=normal",
         "--logger", "trx;LogFileName=Markitai.Core.Tests.trx", "--results-directory", str(results)], cwd=HERE, env=env)
    check_results(results / "Markitai.Core.Tests.trx")


def check_results(trx):
    """A passing run proves something only if its tests ran: every one of them passes, none is skipped."""
    if not trx.exists():
        raise SystemExit(f"dotnet test wrote no results at {trx}")
    counters = ElementTree.parse(trx).getroot().find(".//{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}Counters")
    total, passed = int(counters.get("total")), int(counters.get("passed"))
    print(f"Markitai.Core.Tests: {passed} of {total} passed ({trx})")
    if total == 0 or passed != total:
        raise SystemExit(f"Markitai.Core.Tests: {total - passed} of {total} did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="run Markitai.Core.Tests after building")
    parser.add_argument("--configuration", default="Release", choices=("Release", "Debug"))
    parser.add_argument("--state", type=Path, help="folder for the tests' user state (default: a new temporary folder)")
    parser.add_argument("--core", type=Path, help="use this markitai_app.dll instead of building it")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("build.py builds the Windows app and runs on Windows")
    # The app first: it builds in minutes, the core in many more.
    build_icon()
    folder = build_app(args.configuration)
    dll = args.core.resolve() if args.core else build_core()
    check_runtime(dll)
    shutil.copy2(dll, folder / dll.name)
    print(f"app: {folder / 'Markitai.exe'}")
    if args.check:
        state = args.state or Path(tempfile.mkdtemp(prefix="markitai-tests-"))
        run_tests(dll, state.resolve())


if __name__ == "__main__":
    main()

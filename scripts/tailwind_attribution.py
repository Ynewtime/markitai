"""Complete terms and exact embedded-value provenance for the Tailwind CSS palette."""
from pathlib import Path
import hashlib
import json
import re
import stat

NAMES = ("LICENSE", "README.md", "provenance.json")
EMBEDDED = "crates/markitai-core/src/formats/html/svg.rs"
MAX_FILE = 1024 * 1024
TABLE = re.compile(r"const PALETTE: &\[\(&str, \[&str; 11\]\)\] = &\[(.*?)\n\];", re.S)
ROW = re.compile(r'\(\s*"([a-z]+)",\s*\[([^\]]*)\]', re.S)
VALUE = re.compile(r'"(#[0-9a-f]{6})"')


def bounded_regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_FILE:
        raise RuntimeError("Tailwind attribution requires bounded regular source files")
    value = path.read_bytes()
    if len(value) != info.st_size:
        raise RuntimeError("Tailwind attribution changed during reading")
    return value


def embedded_palette(source, shades):
    """The digest lines of the palette table compiled into the reader."""
    table = TABLE.search(source)
    if not table:
        raise RuntimeError("The embedded Tailwind palette table was not found")
    rows = ROW.findall(table.group(1))
    lines = []
    for name, body in rows:
        values = VALUE.findall(body)
        if len(values) != len(shades):
            raise RuntimeError("An embedded Tailwind palette row has the wrong number of shades")
        lines.extend(f"{name} {shade} {value}\n" for shade, value in zip(shades, values))
    return [name for name, _ in rows], "".join(lines)


def tailwind_files(root):
    root = Path(root)
    directory = root / "licenses/tailwind"
    for path in [directory.parent, directory]:
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise RuntimeError("Tailwind attribution directory is not a regular directory")
    values = {name: bounded_regular(directory / name) for name in NAMES}
    provenance = json.loads(values["provenance.json"])
    if (provenance.get("schema_version") != 1 or provenance.get("license_file") != "LICENSE"
            or provenance.get("embedded_in") != EMBEDDED):
        raise RuntimeError("Unsupported Tailwind attribution manifest")
    if hashlib.sha256(values["LICENSE"]).hexdigest() != provenance.get("license_sha256"):
        raise RuntimeError("Tailwind attribution license digest differs from provenance")
    path = root
    for component in Path(EMBEDDED).parts[:-1]:
        path /= component
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise RuntimeError("The embedded Tailwind palette must remain inside regular source directories")
    names, lines = embedded_palette(bounded_regular(root / EMBEDDED).decode("utf-8"), provenance.get("shades") or [])
    if names != provenance.get("palettes") or hashlib.sha256(lines.encode()).hexdigest() != provenance.get("palette_sha256"):
        raise RuntimeError("The embedded Tailwind palette differs from provenance")
    return {"licenses/tailwind/" + name: content for name, content in values.items()}


def stage_tailwind_files(root, destination):
    record = {}
    for relative, content in tailwind_files(root).items():
        path = Path(destination) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        if path.read_bytes() != content:
            raise RuntimeError("Packaged Tailwind attribution differs from source")
        record[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    return record

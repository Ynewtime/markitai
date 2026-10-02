"""Exact, offline attribution for the portable PDF and OCR engines."""
from pathlib import Path
import hashlib
import json
import stat

HAYRO_NAMES = ("LICENSE_FOXIT", "CGATS_LICENSE.txt", "CMAP_LICENSE.txt",
               "hayro-interpret-assets-README.md", "README.md")
PADDLE_NAMES = ("PaddleOCR-LICENSE", "RapidOCR-LICENSE", "README.md", "provenance.json")
MAX_FILE = 1024 * 1024


def portable_files(root):
    """Read only fixed, bounded regular notice files; models are not redistributed."""
    root = Path(root)
    files = {}
    for category, names in [("hayro", HAYRO_NAMES), ("paddleocr", PADDLE_NAMES)]:
        directory = root / "licenses" / category
        for parent in [directory.parent, directory]:
            if (not stat.S_ISDIR(parent.lstat().st_mode)
                    or getattr(parent.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise RuntimeError("Portable attribution requires ordinary source directories")
        for name in names:
            path = directory / name
            before = path.lstat()
            if (not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_FILE
                    or getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise RuntimeError("Portable attribution requires bounded regular source files")
            with path.open("rb") as stream:
                content = stream.read(MAX_FILE + 1)
            after = path.lstat()
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or len(content) != before.st_size):
                raise RuntimeError("Portable attribution changed during reading")
            files[f"licenses/{category}/{name}"] = content
    provenance = json.loads(files["licenses/paddleocr/provenance.json"])
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != 1
            or provenance.get("manifest") != "crates/markitai-core/src/ocr/paddle/models.json"):
        raise RuntimeError("Unsupported portable OCR attribution manifest")
    entries = provenance.get("licenses")
    expected = {"PaddleOCR-LICENSE", "RapidOCR-LICENSE"}
    if (not isinstance(entries, list) or len(entries) != len(expected)
            or any(not isinstance(item, dict) or not isinstance(item.get("file"), str) for item in entries)
            or {item.get("file") for item in entries} != expected):
        raise RuntimeError("Portable OCR attribution inventory differs from provenance")
    for item in entries:
        content = files["licenses/paddleocr/" + item["file"]]
        if len(content) != item.get("bytes") or hashlib.sha256(content).hexdigest() != item.get("sha256"):
            raise RuntimeError("Portable OCR license bytes differ from provenance")
    return files


def stage_portable_files(root, destination):
    record = {}
    for relative, content in portable_files(root).items():
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        if target.read_bytes() != content:
            raise RuntimeError("Packaged portable attribution differs from source")
        record[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    return record

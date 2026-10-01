"""The extension leaves macOS media frameworks to their first use."""

import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from markitai import _native

MEDIA = ("CoreFoundation", "Foundation", "CoreGraphics", "ImageIO", "Vision")
MARKER = "markitai test: extension loaded"
# From macOS 15 (Darwin 24) dyld postpones the initialization of an image
# linked delay-initialized, and DYLD_PRINT_LIBRARIES reports each image it
# maps, postpones, and initializes later.
POSTPONES = sys.platform == "darwin" and int(platform.release().split(".")[0]) >= 24


def dyld_images(trace):
    mapped, postponed, later = set(), set(), set()
    for line in trace.splitlines():
        image = re.match(r"dyld\[\d+\]: <[0-9A-F-]+> (.+)$", line)
        if image:
            mapped.add(os.path.basename(image[1]))
        moved = re.match(r"dyld\[\d+\]: move (loaded to delayed|delayed to loaded): (.+)$", line)
        if moved:
            (postponed if moved[1] == "loaded to delayed" else later).add(moved[2])
    return mapped, postponed, later


def delayed_dependencies(path):
    """A thin 64-bit Mach-O image's dependencies whose dylib_use_command
    carries DYLIB_USE_DELAYED_INIT."""
    image = Path(path).read_bytes()
    magic, _, _, _, count = struct.unpack_from("<5I", image)
    if magic != 0xFEEDFACF:
        raise AssertionError(f"{path} is not a thin 64-bit Mach-O image")
    delayed, at = set(), 32
    for _ in range(count):
        command, size, name, marker = struct.unpack_from("<4I", image, at)
        if command in (0xC, 0x80000018) and marker == 0x1A741800:
            if struct.unpack_from("<I", image, at + 24)[0] & 0x8:
                text = image[at + name:at + size].split(b"\0", 1)[0]
                delayed.add(os.path.basename(text.decode()))
        at += size
    return delayed


def page_pdf():
    """One page holding a filled rectangle and no text."""
    content = "0 0.4 0.8 rg 20 20 160 60 re f"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
    ]
    pdf, offsets = "%PDF-1.4\n", []
    for index, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n{body}\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    pdf += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    return pdf + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"


@unittest.skipUnless(POSTPONES, "dyld postpones delay-initialized images from macOS 15")
class SystemFrameworkTests(unittest.TestCase):
    def traced(self, code, home):
        environment = dict(os.environ, DYLD_PRINT_LIBRARIES="1", MARKITAI_HOME=str(home))
        run = subprocess.run([sys.executable, "-c", code], env=environment,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        return run

    def test_import_postpones_media_frameworks_until_a_conversion_needs_them(self):
        missing = set(MEDIA) - delayed_dependencies(_native.__file__)
        self.assertEqual(missing, set(), "linked delay-initialized")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host_mapped, host_postponed, _ = dyld_images(self.traced("pass", root / "home").stderr)
            if not host_mapped:
                self.skipTest("this interpreter drops dyld's diagnostic variables")
            # Frameworks that the interpreter itself initializes at launch stay
            # initialized.
            expected = {name for name in MEDIA if name not in host_mapped or name in host_postponed}
            self.assertIn("Vision", expected)
            pdf = root / "page.pdf"
            pdf.write_text(page_pdf(), encoding="ascii")
            options = {"config": {"llm": {"enabled": False}, "cache": {"enabled": False},
                                  "history": {"record": False}},
                       "output_dir": str(root / "out"), "llm": False, "ocr": False,
                       "screenshot": True, "alt": False, "desc": False}
            request = json.dumps({"source": str(pdf), "options": options})
            code = "\n".join([
                "import os, sys",
                "import markitai",
                "from markitai import _native",
                f"os.write(2, {(MARKER + chr(10)).encode()!r})",
                f"sys.stdout.write(_native.convert_json({request!r}))",
            ])
            run = self.traced(code, root / "home")
            loading, separator, converting = run.stderr.partition(MARKER + "\n")
            self.assertTrue(separator, "the marker separates importing from converting")
            _, postponed, later = dyld_images(loading)
            self.assertEqual(expected - postponed, set())
            self.assertEqual(later, set(), "importing initializes no postponed image")
            # Page rendering opens CoreGraphics on first use.
            envelope = json.loads(run.stdout)
            self.assertTrue(envelope["ok"], envelope)
            screenshots = envelope["result"]["screenshots"]
            self.assertEqual(len(screenshots), 1)
            self.assertEqual(Path(screenshots[0]).read_bytes()[:3], b"\xff\xd8\xff")
            if "CoreGraphics" in expected:
                self.assertIn("CoreGraphics", dyld_images(converting)[2])


if __name__ == "__main__":
    unittest.main()

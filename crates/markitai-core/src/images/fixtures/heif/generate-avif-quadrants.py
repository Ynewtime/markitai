"""Authored test fixture generator; no runtime use or external inputs.

Writes quadrants-121x81-orientation6.avif: red, green / magenta, blue
quadrants of an odd 121 by 81 size, 4:2:0, with EXIF orientation 6, which
Pillow's AVIF encoder stores as an `irot` box (a quarter turn clockwise).
usage: python generate-avif-quadrants.py NEW_OUTPUT_DIRECTORY  (Pillow 12.3, libavif 1.4.2)
"""
import sys
from pathlib import Path

from PIL import Image

image = Image.new("RGB", (121, 81))
for box, color in [((0, 0, 61, 41), (255, 0, 0)), ((61, 0, 121, 41), (0, 255, 0)),
                   ((0, 41, 61, 81), (255, 0, 255)), ((61, 41, 121, 81), (0, 0, 255))]:
    image.paste(color, box)
exif = Image.Exif()
exif[0x112] = 6
image.save(Path(sys.argv[1]) / "quadrants-121x81-orientation6.avif", quality=90,
           subsampling="4:2:0", exif=exif.tobytes())

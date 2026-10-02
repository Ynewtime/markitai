"""Bind the R45 OCR corpus images into scanned PDFs with known text (stays under .local).

usage: <reference venv python> make-scanned.py <ocr corpus> <out>

<ocr corpus> is the directory written by ocr-quality-round45/make-corpus.py
(manifest.json plus images). Each set (prose, numbers, cjk, columns) becomes
PDFs of at most eight pages, twice:
- `scan`: the scan-like JPEG (150 DPI page, rotated, blurred, quality 55)
  embedded byte for byte as DCTDecode, page size = pixels at 150 DPI;
- `300`: the 300 DPI PNG decoded with Pillow and stored as FlateDecode, page
  size = pixels at 300 DPI, so a 150 DPI render halves it.
Nothing is re-encoded lossily. manifest.json lists every PDF with its pages'
source images, language and ground-truth lines in reading order.
"""
import hashlib, json, sys, zlib
from pathlib import Path

from PIL import Image

corpus = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
rows = json.loads((corpus / 'manifest.json').read_text())['rows']


def jpeg_page(path):
    data = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    space = {'L': '/DeviceGray', 'RGB': '/DeviceRGB', 'CMYK': '/DeviceCMYK'}[mode]
    return width, height, space, '/DCTDecode', data


def png_page(path):
    with Image.open(path) as image:
        image = image.convert('L' if image.mode in ('1', 'L', 'LA') else 'RGB')
        width, height = image.size
        space = '/DeviceGray' if image.mode == 'L' else '/DeviceRGB'
        return width, height, space, '/FlateDecode', zlib.compress(image.tobytes(), 9)


def write_pdf(target, pages, dpi):
    """pages: (width, height, colour space, filter, data); one image per page."""
    objects = []

    def add(body):
        objects.append(body)
        return len(objects)

    catalog = add(None)
    tree = add(None)
    kids = []
    for width, height, space, filter_name, data in pages:
        image = add(f'<< /Type /XObject /Subtype /Image /Width {width} /Height {height} '
                    f'/ColorSpace {space} /BitsPerComponent 8 /Filter {filter_name} '
                    f'/Length {len(data)} >>\nstream\n'.encode() + data + b'\nendstream')
        w, h = width * 72 / dpi, height * 72 / dpi
        content = f'q {w:.4f} 0 0 {h:.4f} 0 0 cm /Im Do Q'.encode()
        stream = add(f'<< /Length {len(content)} >>\nstream\n'.encode() + content + b'\nendstream')
        kids.append(add(f'<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {w:.4f} {h:.4f}] '
                        f'/Resources << /XObject << /Im {image} 0 R >> >> /Contents {stream} 0 R >>'.encode()))
    objects[catalog - 1] = f'<< /Type /Catalog /Pages {tree} 0 R >>'.encode()
    objects[tree - 1] = (f'<< /Type /Pages /Count {len(kids)} /Kids ['
                         + ' '.join(f'{kid} 0 R' for kid in kids) + '] >>').encode()
    body = bytearray(b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n')
    offsets = []
    for number, content in enumerate(objects, 1):
        offsets.append(len(body))
        body += f'{number} 0 obj\n'.encode() + content + b'\nendobj\n'
    xref = len(body)
    body += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode()
    body += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets)
    body += f'trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    target.write_bytes(body)
    return hashlib.sha256(body).hexdigest(), len(body)


manifest = []
for variant, suffix, dpi, reader in [('scan', 'scan', 150, jpeg_page), ('300', '300', 300, png_page)]:
    for name in ['prose', 'numbers', 'cjk', 'columns']:
        members = sorted((row for row in rows if row['set'] == name and row['variant'] == suffix),
                         key=lambda row: row['file'])
        for start in range(0, len(members), 8):
            chunk = members[start:start + 8]
            target = out / f'{name}-{variant}-{start // 8 + 1}.pdf'
            digest, size = write_pdf(target, [reader(corpus / row['file']) for row in chunk], dpi)
            manifest.append({'pdf': target.name, 'sha256': digest, 'bytes': size, 'set': name,
                             'variant': variant, 'lang': chunk[0]['lang'],
                             'pages': [{'image': row['file'], 'image_sha256': row['sha256'],
                                        'truth': row['truth']} for row in chunk]})
(out / 'manifest.json').write_text(json.dumps({'corpus': str(corpus), 'rows': manifest}, indent=1) + '\n')
print(len(manifest), 'PDFs,', sum(len(row['pages']) for row in manifest), 'pages')

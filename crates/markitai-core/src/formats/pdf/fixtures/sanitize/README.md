# Authored hidden-text policy fixtures

These tiny PDFs contain only authored literal text and drawing operators. No
external document, account information or personal file was used. Run
`python3 create-fixtures.py` here to regenerate their deterministic bytes;
`author-fixtures.json` records the SHA-256 and size of each result.

- `policy-text.pdf`: visible and white copies of identical words, transparent
  text, one-point-or-smaller text, and a visible final paragraph.
- `policy-form.pdf`: the same Form called with transparent and opaque state.
- `policy-white-on-black.pdf`: actually visible white text on a painted panel.
- `policy-assets.pdf`: the hidden-text cases with a separately owned RGB image.
- `policy-searchable-scan.pdf`: an invisible OCR layer over authored gray
  pseudo-print, matching the existing scan-layer geometry/ink tests. The image
  is a grid of strokes, not an OCR transcription-quality benchmark.

All pages inherit a valid direct Resources dictionary from their Pages node.
This exercises a pinned lopdf resource-helper limitation without using an
invalid PDF or promoting hidden text as a new OCR layer.

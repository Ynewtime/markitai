# hayro embedded assets

Windows and Linux builds (and macOS builds with the `portable-media` feature)
render PDF pages with hayro 0.7.1 (MIT/Apache-2.0). hayro embeds data files
with their own licences, copied here unchanged from the crates.io packages:

| File | Source package | Covers |
| --- | --- | --- |
| `LICENSE_FOXIT` | hayro-interpret 0.7.0 `assets/` | Foxit standard-14 substitute fonts (`*.pfb`) extracted from PDFium (BSD-3-Clause) |
| `CGATS_LICENSE.txt` | hayro-interpret 0.7.0 `assets/` | `CGATS001Compat-v2-micro.icc` CMYK profile (CC0-1.0) |
| `hayro-interpret-assets-README.md` | hayro-interpret 0.7.0 `assets/` | provenance of the fonts and ICC profiles (`LAB.icc` generated with LCMS2) |
| `CMAP_LICENSE.txt` | hayro-cmap 0.1.0 `assets/` | Adobe predefined CMaps (BSD-3-Clause) |

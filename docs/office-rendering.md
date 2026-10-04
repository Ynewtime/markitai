# Complete Office page rendering

Office text extraction remains native Rust. Page screenshots optionally use an
installed LibreOffice to export PDF, followed by the in-process
[PDF page renderer](pdf-rendering.md) (CoreGraphics on macOS, hayro on Windows
and Linux). LibreOffice is not bundled; its installation and fonts are
additional runtime requirements. The CLI itself remains one executable and
invokes no Python conversion code. Installed CLI builds have exercised Office
export and PDF page rendering on macOS ARM64, Windows ARM64 and Linux x86-64,
using selected authored Word, presentation and workbook fixtures. These checks
do not establish every accepted format on every platform or Microsoft Office
layout parity; current findings and source boundaries are described below.

When page OCR or page screenshots are requested (`--ocr`, `--screenshot`, or a
preset that implies them such as `rich`) and LibreOffice or the page renderer is
missing, an Office document is converted from its own text with one warning
instead of failing, so a folder of mixed documents still converts. The warning
names what was skipped and how to get it: install LibreOffice (on macOS
`brew install --cask libreoffice`, or put `soffice` on `PATH`), or pass
`--no-screenshot` / `--no-ocr` to stop asking. With an LLM the text is still
enhanced; only the page images are missing. Only `--screenshot-only`, where the
screenshots are the whole requested output, keeps failing without them, with an
error that names the same remedy. Numbers capture is unsupported with or without
LibreOffice and keeps its own error (see below), so installing the program would
not change it. `markitai doctor` reports whether LibreOffice and the renderer
are available.

The adapter accepts existing presentation aliases `ppt`, `pps`, `pot`, `pptx`,
`pptm`, `ppsx`, `ppsm`, `odp`, and word-processing aliases `doc`, `docx`, `docm`,
`odt`, `rtf`. Workbook capture accepts `xls`, `xlsx`, `xlsm`, `xlsb` and `ods`
through Calc's complete-sheet export. XLSX, XLS and ODS have dedicated authored
fixtures; XLSM/XLSB import fidelity has not been independently established.
Numbers screenshots remain explicitly unsupported: the installed LibreOffice
could not import the retained Numbers sample. Native Numbers table reading is
still available without screenshot/OCR options.

For presentations, the ordered source slide list includes hidden and blank
slides. OOXML relationships and ODP presentation elements provide the count.
Legacy binary PowerPoint is first imported into a private PPTX copy so the
imported slide model can be counted; this is a LibreOffice import consistency
check, not proof of perfect Microsoft PowerPoint import fidelity. The PDF must
contain exactly that many pages. Hidden-slide export is enabled and notes pages
are disabled. Word exports all pages with blank-page omission disabled; it has no
reliable source paragraph-to-page mapping. These export options are described by
[LibreOffice's PDF parameter documentation](https://help.libreoffice.org/latest/en-US/text/shared/guide/pdf_params.html).

## Use and recovery

```sh
markitai report.docx --screenshot --no-llm -o out/
markitai scan.docx --ocr --no-llm -o out/
markitai workbook.xlsx --screenshot-only --no-llm -o out/
```

These examples keep conversion local. Install LibreOffice separately and run
`markitai doctor` if page capture is skipped. `doctor --fix` does not install
LibreOffice. Missing fonts or different LibreOffice versions can change the
layout even when capture succeeds; install the document's fonts and inspect
its page images. For portable OCR model problems, follow [OCR model repair](ocr.md#models).

If export times out or exceeds a page/pixel/output budget, it fails rather than
returning an incomplete set of screenshots. Retry without `--screenshot-only`
and with `--no-screenshot --no-ocr` when native text alone is sufficient, or split
the source document before requesting full capture again. Preserve the original
file; the exporter's temporary normalization never edits it.

## Complete workbook sheets

Workbook capture deliberately uses `calc_pdf_Export` with `SinglePageSheets=true`.
The [official export contract](https://help.libreoffice.org/latest/en-US/text/shared/01/ref_pdf_export_general.html)
includes hidden sheets and ignores paper sizes, print areas and manual print
pagination, fitting each entire sheet to its own PDF page. This is a complete
sheet canvas, not a reproduction of the workbook's printed pages. No `PageRange`
or current-sheet selection is supplied. A worksheet configured to print on
several paper pages therefore produces one full-sized canvas by this explicit
mode; its bottom/right content is not silently discarded or forced into A4.

Hidden sheets can consequently appear in screenshots and, when visual LLM
processing is enabled, in model requests. Hidden rows/columns retain the imported
document's display behavior; complete-sheet export does not promise to reveal
every hidden cell. Native table text remains the original Rust extraction, so
its saved values and inclusion rules can differ from LibreOffice's displayed
or recalculated values. A runtime warning explains complete-sheet capture and
its printing differences.

For simple XLSX/XLSM workbooks without a theme, the private export copy uses
black for fonts that declare no color. This is a rendering compatibility policy;
explicit RGB, indexed, theme and automatic color declarations are preserved,
and the original file is unchanged. The adjustment is skipped when conditional
formatting, rich text, unknown XML parts or ambiguous inheritance prevent a safe
decision, or when an inspected part uses an unsupported encoding such as UTF-16.
Those workbooks retain LibreOffice's original color handling. Native table text
is unaffected. Any later right-edge geometry repair preserves the styles in its
input copy; it does not override explicit colors to make text visible.

XLSX/XLSM sheet lists and ODS tables are counted from bounded source XML, including
hidden and empty sheets. Binary XLS/XLSB is first imported into a private ODS copy,
whose ordered sheet model supplies the expected count. Matching that import is
not proof that LibreOffice imported every source feature correctly. The exported
PDF must contain exactly one page per counted sheet; fewer or additional pages
fail explicitly. The original file is never modified. The reference Python
XLSX/XLS readers extract tables rather than sheet screenshots, so this is a new
native capability, not an assertion of reference screenshot parity.

The tested LibreOffice preserves a genuinely empty sheet as an extremely thin
white PDF page. Its actual positive page dimensions are retained and the native
rasterizer rounds up to at least one pixel; no arbitrary paper-sized page is
inserted. An exporter that omits an empty sheet fails the count check instead of
reporting complete capture. Blank content cannot serve as proof of OCR accuracy.

The native rasterizer validates the full page dimensions before allocating or
applying output-image compression: a sheet over the per-page or document pixel
budget fails, and the caller receives no truncated success. Existing user image
size/format options still control the encoded screenshots after admission. This
does not establish legibility after a user-requested reduction of a large sheet.

The existing document body is preserved. Ordered Slide/Page screenshot comments
are appended in an explicit appendix, rather than inventing a correspondence
between native paragraphs and rendered pages. Root conversion integration applies
the existing screenshot, local OCR, pure, VLM opt-out and model-page budgets.
Local OCR supplements native text; it does not replace it with PDF extraction.
See [PDF rendering](pdf-rendering.md) for page pixel and encoding limits.

## Execution and limits

Discovery searches absolute PATH entries and standard LibreOffice locations;
relative/empty PATH entries do not select a document-local executable. On
Windows the search tries the PATHEXT extensions and prefers `soffice.com`, the
[documented command-line entry](https://help.libreoffice.org/latest/en-GB/text/shared/guide/start_parameters.html?DbPAR=WRITER&System=WIN),
to `soffice.exe` in the same installation; the GUI launcher remains a fallback
when the console wrapper is absent. The
executable receives argument-array parameters, never a shell command. Each export
owns a private temporary directory containing an input copy, fresh LibreOffice
profile, and output directory. The profile is passed as
`-env:UserInstallation=<file URL>` on every platform, so no export reads or
writes the user's LibreOffice profile; macros and automatic link updates are
disabled in it. Model credentials and other process environment values are not
inherited; PATH (and SystemRoot on Windows) is retained, TMPDIR/TMP/TEMP point
into the private directory everywhere, XDG_CACHE_HOME too on Unix (for caches
of libraries such as fontconfig), and HOME is never reassigned. The exported PDF
is drawn by the platform's [page renderer](pdf-rendering.md): CoreGraphics on
macOS, hayro on Windows and Linux.

At most two exports run concurrently per process. A shared 120-second deadline
covers admission, private input preparation and export subprocesses. A timeout
kills and waits for the child. Launcher descendants are included: Unix uses a
dedicated process group; Windows starts LibreOffice suspended in a Job Object
that ends its whole tree, also when Markitai itself ends, and is emptied before
the private directory is removed. The adapter has run in an actual Windows
ARM64 guest, beyond cross-target compilation. Output growth is polled every 25 ms.
A temporary child file that disappears between directory enumeration and its
metadata read is tolerated only while the export is running; it still counts
toward the entry budget. Other I/O errors, file types, symlinks and byte limits
remain strict, as does the final scan after the process exits. Inputs are
limited to 100 MiB; normalized or repaired Office copies and exported PDFs share
a 100 MiB output budget.
Presentations, workbooks and resulting PDFs have a 1,000-slide/sheet/page limit.
Workbook ZIP packages allow at most 16,384 entries and the sheet-index XML at
most 32 MiB, with nesting capped at 128. Missing/ambiguous sheet-index parts,
document types, incomplete XML and empty sheet lists are rejected. These bounds
apply before requesting PDF export; the optional LibreOffice importer still has
its own internal allocations.

A successful exit alone is insufficient: exactly one expected regular output
file must exist, the PDF signature and document structure must parse, encryption
is rejected, and nonempty page count is checked. Empty, partial, malformed,
symlinked, excessive or unexpected outputs fail explicitly. Temporary output is
removed when its owner is dropped. Subprocess timing, output checks and private
profiles are not an OS memory/network sandbox; LibreOffice layout and native PDF
parsing can consume CPU/memory internally between checks. External document
resources and fonts remain subject to installed LibreOffice behavior.

## Accuracy and platform scope

Selected Word, presentation and workbook workflows have been exercised on
macOS ARM64, Windows ARM64 and Linux x86-64. That does not establish every
legacy import, matching fonts or pixel-identical output across platforms.
LibreOffice versions and installed fonts affect layout; use a digital source's
native text for searchable content and inspect page images when layout matters.

# Complete Office page rendering

Office text extraction remains native Rust. Page screenshots optionally use an
installed LibreOffice to export PDF, followed by the native macOS PDF renderer.
LibreOffice is not bundled; its installation and fonts are additional runtime
requirements. The CLI itself remains one executable and invokes no Python
conversion code. macOS is the first rendering platform covered by this adapter;
having LibreOffice installed does not make unsupported PDF platforms available.

When OCR is requested without screenshots and LibreOffice or the page renderer is
missing, an Office document is converted from its own text with a warning instead
of failing, so OCR over a folder still converts its Office files; explicitly
requested screenshots are the output itself and still fail without them.

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
relative/empty PATH entries do not select a document-local executable. The
executable receives argument-array parameters, never a shell command. Each export
owns a private temporary directory containing an input copy, fresh LibreOffice
profile, and output directory. Macros and automatic link updates are disabled in
the private profile. Model credentials and other process environment values are
not inherited; PATH/platform loader state is retained, temporary/cache locations
are private, and HOME is never reassigned.

At most two exports run concurrently per process. A shared 120-second deadline
covers admission and both legacy normalization/PDF export subprocesses. A timeout
kills and waits for the child; Unix uses a dedicated process group to include
launcher descendants. Windows process-tree termination is implemented but has not
been validated on a Windows host. Output growth is polled every 25 ms. Inputs are
limited to 100 MiB; normalized PPTX/ODS plus exported PDF share a 100 MiB budget.
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

## Fixtures and validation

Original fixture generators and hashes are under `office_render/fixtures`:
three presentation slides (visible, hidden, blank) with independent corner
colors; and Word pages with first/third-page text and a deliberately blank middle
page. SDK pixel checks establish full-frame rendering and slide order, separately
from text checks. Fault tests cover missing, malformed, short, unexpected and
symlinked PDFs, input preservation, timeout descendant cleanup and byte limits.
Optional installed-LibreOffice tests are explicitly ignored in the default gate;
they must be run separately and cannot count as success when the backend is absent.

Workbook fixtures are independently authored ZIP/XML packages:
`whole-workbook.xlsx` and `whole-workbook.ods`. Their first sheet has a narrow
print area, manual pagination, a wide canvas and a far bottom-right marker; the
following sheets are hidden, truly empty and visible, in that order. The XLS
fixture is derived from the authored XLSX using LibreOffice's `MS Excel 97`
export filter. The generator and `workbooks-provenance.json` retain source hashes
and provenance; they contain no user document data.

The initial isolated export probe confirms why ordinary printing is insufficient:
both XLSX and ODS export only two ordinary print pages, omitting the hidden sheet
and the far marker. Complete-sheet export produces four correctly ordered pages;
the XLS→ODS import path retains the same four sheets. These are direct exporter
observations under `.local/workbook-round24`. The subsequent source-frozen
R24 gate also passes the real Rust/core and public API workbook cases.
New optional Rust tests cover full canvas colors and dimensions, empty-page
retention, unchanged native tables, complete page references, model-budget
rejection before any request, and pure/visual-only routing. Their final execution
status is recorded by the coordinator in [CONTROL](CONTROL.md).

Round 22's r4 validation passed the full gate and explicitly ran all six installed
backend tests: two core tests and four public API tests on macOS. These verify
hidden/blank slide order and full-frame pixels, the blank middle Word page,
all-page publication with native text retained, rejection before model requests
when the page budget is exceeded, pure/screenshot-only routing, and local OCR
supplements without replacing native text.

Initial direct backend probe used LibreOfficeDev `26.8.0.0.alpha0`, build
`2c87e51eeaa2b413ff4ae097b2705eea1995d8e5`. Both original fixtures produced PDFs;
that initial probe alone did not establish renderer/API acceptance. Its raw
commands, output hashes and logs remain in
`.local/media-llm-round22/office-probe/record.json`, separate from the later passing
tests above. Round 22 release CLI and installed-binding validation subsequently
passed, as recorded below. The six tests do not establish legacy binary input import fidelity,
cross-platform rendering, performance or Microsoft Office layout parity.

The initial Word fixture put two adjacent page breaks in one paragraph. This
LibreOffice version imported it as two pages even with blank-page omission
disabled. Separate paragraphs produce the intended three-page document; an
independent export checked first/third-page text and an entirely white middle
page. The authored generator and fixture now use those explicit paragraph
boundaries. The production PDF filter and three-page assertions were retained;
the original failed fixture and six comparative exports remain in the round's
investigation evidence.

The retained round-22 release CLI also passes full-page PPTX, legacy PPT and Word
acceptance, including hidden/blank pages and full-frame pixel checks. Refreshed
installed Node/Python/Go packages pass their binding tests against this core. See
[round-22 evidence](validation/media-llm-round22.md); these authored samples do not
establish Microsoft Office layout equivalence or other-host support.

`office_diagnostic()` uses the same private process setup and cleanup, acquiring
an export slot within a ten-second deadline and running only `--version`. It
reports discoverable-and-startable status separately from PDF platform support
and document fidelity. Child stderr is discarded and failure messages are fixed.

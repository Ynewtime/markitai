# Document formats and conversion contracts

The Rust engine parses documents locally and returns a `Document`: Markdown,
metadata, embedded asset bytes and explicit warnings. It never writes assets or
fetches external content inside a format adapter. Output naming, profiles,
network policy and optional model enhancement belong to the orchestration layer.

## Implemented readers

| Inputs | Reader | Current behavior |
| --- | --- | --- |
| TXT, MD, MARKDOWN | Rust text decoder | Preserves text and existing frontmatter; accepts UTF-8, BOM-marked UTF-16, Windows-1252 and detected GBK/GB18030, Big5, Shift_JIS, EUC-JP and EUC-KR (see [text encodings](#text-encodings)) |
| HTML, HTM, XHTML | scraper + htmd | Honors a `<meta>` charset declaration; selects an article/main candidate, extracts metadata, removes navigation/scripts/hidden content, controls, entry-box forms and the page footer (keeping a form's prose and a part's footer), resolves relative HTTP links and images; `<br>` is a hard break, task lists keep `[x]`, ruby reads `漢字(kanji)`, table cells use `<br>` and `\|` ([HTML](html.md#rendering-and-url-handling)) |
| CSV, TSV | csv | Decoded like TXT; the delimiter (`,`, `;`, tab or `|`) is sniffed from the first records (see [delimited text](#delimited-text)); both keep every field (the table is as wide as the widest row) and escape `|`, backslashes and line breaks in cells; the reference cut CSV rows to the header's width and left those characters raw |
| IPYNB | serde_json | Markdown cells, fenced code and raw cells; metadata title and code language; code fences sized to protect embedded backticks; each code cell's printed text, results, errors and images follow it, and markdown-cell image attachments are assets (see [notebook outputs](#notebook-outputs)) |
| JSON | serde_json | Validated, pretty-printed fenced JSON; an additive Rust format |
| XML | quick-xml | Structured headings, attributes and mixed text, plus a source fence for small inputs; document types are rejected |
| EML | mail-parser | Decoded subject, body and MIME attachments; attachment bytes returned separately |
| MSG | cfb + native properties | Outlook headers, Unicode/ANSI body, HTML fallback and bounded by-value attachments |
| RST, Org, TeX | native markup readers | Structured sections, lists, code, math, links and tables; unsupported constructs retained with warnings |
| JPEG, PNG, GIF, BMP, TIFF, WebP | image + native LLM transport + local OCR | Standalone vision inputs, shared raster assets and complete TIFF page OCR/vision with bounded decoding |
| DOC, DOCX, DOCM; templates DOT, DOTX, DOTM | anydoc document model | Headings, styled text, lists, tables, links, formulas, notes and embedded assets; a manual line break is a hard break (two trailing spaces), two end the paragraph, and text is escaped only where Markdown would read it as syntax ([below](#document-line-breaks-and-escaping)) |
| PPT, PPS, POT | anydoc document model | Legacy presentation content through the shared Markdown renderer, with optional numbered slide markers; embedded charts and worksheets read as their data tables |
| PPTX, PPTM, PPSX, PPSM; templates POTX, POTM | bounded ZIP + PresentationML reader | Optional ordered slide markers, hidden-slide markers, title placeholders, text frames with bullets as nested lists and web/mail hyperlinks as links, grouped shapes, tables, referenced images, cached chart data, SmartArt text as lists, speaker notes and review comments |
| XLS, XLSX, XLSM, XLSB; templates XLT, XLTX, XLTM | anydoc document model | Native sheet content with number formats, cell links, cell notes and the text of uncalculated formulas (see [spreadsheets](#spreadsheets)); XLS/XLSX/XLSM/XLSB single-sheet names are recovered from package metadata; exact cell-format compatibility has not been established |
| ODT, ODS, ODP, RTF; templates OTT, OTS, OTP | anydoc document model | Native structured documents through the same Markdown renderer (line breaks and escaping as for Word); ODP slides carry optional numbered slide markers |
| NUMBERS | bounded ZIP/directory IWA preflight + iwork | Ordered sheets/tables, rectangular saved values and explicit formatting/unsupported-content warnings; see [Numbers](numbers.md) |
| EPUB | anydoc + OPF metadata | Spine content and the original title/authors/language/publisher/date/description/identifier preamble; ruby as base text then reading (`漢字(kanji)`), definition terms as bold paragraphs, and footnote marks the author wrote as Markdown (`[^5]`, `[^5]: …`) kept unescaped; `<br>` and text escaping as for Word (`[!tip]` stays as written) |
| PDF | pdf-inspector + lopdf; CoreGraphics/hayro page rendering and local OCR | Per-page text/layout, link targets, partial recovery, embedded images and conservative bar-chart crops; explicit local-file page OCR and screenshots through the shared media pipeline |

### Text encodings

TXT, MD, CSV and TSV files carry no encoding label. A byte-order mark (UTF-8,
UTF-16LE/BE) decides first, then valid UTF-8, which stays a single validation
pass. Other bytes are offered to GB18030 (which includes GBK and GB2312), Big5,
Shift_JIS, EUC-JP and EUC-KR, the encodings Chinese, Japanese and Korean Excel
and Notepad still write. A reading with any invalid or unmapped byte sequence
is discarded; Western text rarely survives, because an accented letter
followed by a space, digit or punctuation is invalid in all five. Each
remaining reading is scored by the share of its non-ASCII characters in the
frequent part of its character set (punctuation, kana, Hangul, first-level
ideographs), with three structural rules: EUC-JP kanji count only beside kana,
a Big5 reading of eight or more characters without a low trail byte is not
Big5, and a reading of twelve or more KS X 1001 Hangul syllables with nothing
else is Korean. Windows-1252 competes with the share of non-ASCII bytes that
stand alone between ASCII bytes as letters or common punctuation (accented
letters, curly quotes, dashes). A multibyte reading needs a score of 0.7 and a
lead of 0.2 over every other reading; otherwise the text is read as
Windows-1252 as before, with a warning when an East Asian reading was plausible
but not decisive. Very short inputs (a few characters) are often undecidable.
There is no option to name the encoding.

HTML follows the HTML standard's order: a BOM, then a `<meta charset>` or
`<meta http-equiv="Content-Type" content="…; charset=…">` declaration found
by its prescan of the first 1,024 bytes, with its label substitutions (UTF-16
labels mean UTF-8, `x-user-defined` means Windows-1252). Bytes that are valid
UTF-8 and not plain ASCII are read as UTF-8 whatever they declare, because a
page re-saved as UTF-8 keeps its old declaration; the `replacement` labels
(ISO-2022-KR, HZ-GB-2312) are ignored. Byte sequences invalid in the declared
encoding become U+FFFD with a warning. HTML without a declaration is decoded
like TXT. Ambiguous short inputs may need conversion to UTF-8 before processing.

The native Office renderer reads the document once and preserves referenced
embedded bytes. Shared image preparation then applies configured filtering and
compression. Unreferenced archive images are omitted. References use `.markitai/assets/{name}` until the output layer assigns
final paths. A merged table's origin contains its content; covered cells are
empty, and a warning records this Markdown representation.

Word documents (DOCX/DOCM) are read as the document looks with its tracked
changes accepted: inserted and moved-in text is kept; deleted text and rows are
not, and neither is hidden text (`w:vanish`). Comments, headers and footers are
not part of the Markdown; a document with comments gets a warning naming their
number. A table's first row is a Markdown header only when Word marks it as a
repeated header row; otherwise the header line is blank and every row is data.
The base text of a phonetic guide (ruby) is kept and the guide is not, and the
words of VML WordArt are read from the shape.
Parts a Word document embeds with `w:altChunk` for Word to convert when it
opens the file (report generators, mail merges and the Open XML SDK write
HTML, RTF, plain text or whole Word documents that way; the html-docx-js
library writes every document as one MHT web archive) are read where they
stand, by the content type `[Content_Types].xml` gives them, else their
extension or first bytes; before, such content was missing, all of it for an
html-docx-js file. HTML and XHTML go through the HTML reader once a bounded
pass has made well-formed XML of them (void and implicitly closed elements,
unquoted attributes, a lone `&` or `<`, Word's `o:p` markup, comments,
conditional comments and scripts), `data:` pictures becoming assets; a web
archive's HTML page is read with its quoted-printable or base64 parts and its
pictures by `Content-Location` or `cid:`; RTF, plain text (a paragraph per
line) and Word documents go through their readers, a Word document at most
three documents deep and reading against the outer file's 512 MiB
decompression budget, with its notes, pictures, anchors and note ids joining
the document's under a scope of their own. A part in another format (Word's
XML formats), nested deeper, missing or unreadable adds nothing, and one
warning per reason names how many parts were left out.
Superscript and subscript runs are written as Unicode super/subscript
characters when every character has one (`10⁻³`, `H₂O`) and stay at the
baseline otherwise. `w:sym` characters from the Symbol and Wingdings fonts map
to their Unicode marks, and so does text set in those fonts, as Word shows it
(a Symbol `m` is `μ`, a stored private-use `F0FC` in Wingdings is `✓`; a real
`α` stays); a non-breaking hyphen is `-`. Chinese, Japanese and
enclosed-digit numbering (`一、`, `（二）`, `①`) keeps the document's own
characters in list and heading labels. A list label Markdown does not read as
a marker (`a)`, `(1)`, `一、`) is kept as text, one item to a line. Emphasis
beside a letter has its edge punctuation moved outside the markers
(`**注意**：请`), since CommonMark would not open or close it otherwise.
Indentation typed with spaces is kept, because code pasted into Word depends
on it; four or more leading spaces therefore render as a code block. Fields keep
their last computed result, and floating shapes appear where their anchor
paragraph is. A legacy form field shows what Word shows: a text field its
result, a check box `☒` or `☐` (its `w:checked` state, else its default) and a
drop-down list its chosen entry (`w:result`, else the default, else the
first), which Word keeps in the field data rather than in the result; before,
a filled-in form lost every answer but its text fields. A VML picture (Word
2003, compatibility mode) takes its shape's `alt` as alt text when the
drawing has no description.
Text set in a monospaced font (Courier, Consolas, Menlo, or a family named
`… Mono`, also by its PostScript name such as `Menlo-Regular`; by the run's own
font, else its character style's, else its paragraph style's or the document
default) is code: a paragraph that is all such text is a line of a fenced code
block, blank lines of the listing included, and such text inside prose is
inline code. A heading set in it stays a plain heading, a list item stays a list
item, and a table cell keeps it inline. When monospace carries more than three
quarters of the document's visible characters it is the document's typeface (a
typewriter manuscript) and marks nothing. A code block whose lines carry their
own numbers, as a column before them or alternating with them (a web page's
numbered listing saved as a document), loses the numbers, and a table that only
lays out a listing (one row holding the code after an optional cell of line
numbers, as syntax highlighters build it; one numbered line per row; or a single
cell of code) is that code block, without the numbers.

A paragraph that continues a list item is part of it. In Word, RTF and Word 97
files each list level sets its text in from the margin (a numbering level's
indent, the item paragraph's own `w:ind`, `\li` or `sprmPDxaLeft`); a
paragraph after an item with no number of its own whose lines start at least as
far right as the list's outermost text (a List Paragraph after Enter and
Backspace, a direct indent as Google Docs exports, TextEdit's or Word's
`\pard\li720`) continues the deepest open item whose text starts no further
right, so it follows that item's nested list when it is set back to the outer
text. It is written under the item after a blank line, indented to the
marker's width, and the numbering goes on in one list. A paragraph numbered at
a level that shows no marker (Word's `none` format, or a bullet of spaces, as
pandoc writes an item's later paragraphs, which read as extra bullets before)
continues the item at that level. Empty paragraphs between an item and its
continuation are passed over; between two items they split the list as before.
Text back at the body's indent ends the list, and nothing continues a list
whose text starts less than ten points right of the body text before it (a
memo set in throughout). In OpenDocument several paragraphs of one
`text:list-item` were already one item; an unnumbered entry LibreOffice writes
as a `text:list-header` opening a list that continues the one before it
(`text:continue-numbering`, `text:continue-list`) now goes into the last item
of that list, as deep as it is nested, and the items after it carry on that
list. In lists typed by hand an item with a hanging indent is continued the
same way by a paragraph set in to its text. Not covered: OpenDocument
paragraphs outside a list matched against its indents (label positions there
depend on the list style's positioning mode), code or quotation paragraphs
inside an item, and list items in RTF table cells.

Columns set with tab stops become a table. A run of at least three rows at the
same tab stops, each split by its tabs into the same number of cells and no
longer than one printed line (100 characters), is a table: consecutive plain
body paragraphs (not headings, list items, table cells, text boxes, notes or
code), or the lines of such a paragraph split by line breaks (Shift+Enter
between rows), every line holding a tab. At the default stops a run of tabs
separates one pair of columns,
since authors press Tab until the text lines up; at stops the author set each
tab moves to the next column, so a cell may be empty. A column no row fills,
such as a tab that indents every row, is dropped. Three columns are enough; two
need stops the author set and a first column that is not a list label (`1.`,
`a)`) or a field label (`Date:`). A first column of bullets is a list typed by
hand (below), not a table; a tab stop with a leader or a
table-of-contents or index style, or a last column of page numbers counting up
with no header above them (a contents page) keeps the paragraphs as text. The
first row is the header only when it alone is bold; otherwise every row is data
and the header line is blank, as for Word tables. Every other tab, including
those in headings, lists, cells, notes and code, reads as a single space as
before. On 35 cases written as DOCX, ODT and RTF (stops set on the paragraph or
its style, default stops pressed several times, decimal stops, empty cells,
links in cells, an indented table, a typewriter document, rows split by line
breaks at set and at default stops and after a header paragraph; contents
pages with leaders, by hand and in a contents style, verse, a lone tab, two
rows, lists typed by hand, memo labels, two default-stop columns, prose, rows
inside lists, headings, code or cells, and line-broken look-alikes: two lines,
memo labels, verse, a caption line, prose and a signature block), each format
finds all 12 tables and makes no table of the 23 text cases (before the
line-break rule, 9 tables). The 108 `textutil` pages,
whose only tabs belong to lists typed by hand, give no table.

Lists typed by hand are lists. Many documents type their lists: each item is
a paragraph that starts with a bullet or a number and a tab or a space, as
macOS `textutil` saves every HTML list in Word documents (`Tab • Tab text`
under a hanging indent, `Tab ◦ Tab text` one level down, `Tab 1 Tab text` in an
ordered list). In DOCX, ODT, RTF and Word 97 files, a run of consecutive plain
body paragraphs (not headings, real list items, table cells, text boxes,
notes, code, a table-of-contents style or a table set with tab stops) that
open that way is a list:

- a bullet that is only ever one (`•`, `◦`, `▪`, `●`, `➢`, `・`, Word's Symbol
  and Wingdings bullets, a ballot box, a tick) makes an item by itself; a
  ballot box becomes a task-list box (`[ ]`, `[x]`) and a tick stays in the
  item's text;
- a character that also starts sentences (`-`, `–`, `*`, `+`, `o`, an arrow)
  needs a tab after it, an item above it that it is indented under, or a
  sibling at its level with the same mark and no other such mark beside
  them (`+ fast`, `- loud` stay text). An em dash needs a tab or a parent,
  and hyphens or en dashes whose lines end as speech does (`?`, `!`, `…`, a
  quotation mark), set narration apart (`– Oui, – dit-elle.`) or carry a
  speech incise after a comma (`– Je pars, dit-il.`, `- Ya voy, dijo Ana.`,
  `– Vi ses, sa han.`: a clause of at most three words led by one of about
  a hundred speech verbs in fourteen languages or by a French inversion such
  as `murmura-t-elle`, or `he`/`she`/`I` and such a verb) are dialogue and stay
  text, as do a lone `- and then…`, an attribution (`— Name`) and `* * *`;
- numbers make items when at least two at one level count up by one in one
  form (`1.` `2.`, `a)` `b)`, `(i)` `(ii)`, `一、` `二、`), or when a decimal
  number stands alone before a tab (`Tab 2 Tab`, an `<ol start="2">`),
  unless numbers that do not count up stand beside it (an inventory) or its
  text is all bold (a numbered heading typed by hand). A number before a
  space (`3 apples`), an initial (`A. Smith`), a year, an outline number
  (`1.2`) or a quantity (`1.5 kg`) is text;
- the level comes from where the marker sits (the paragraph's left and
  hanging indent, then the tabs and spaces before the marker): an item set
  further right than the one before it is nested under it;
- decimal numbers keep their count (`2.` for `<ol start="2">`), and other
  labels are kept as written, one item to a line, as for real lists.

A bullet line set in a monospaced font is a list item with inline code, not a
code block, and a paragraph opening with a bullet is never taken for a heading
set by hand. RTF text in a `Symbol` or `Wingdings` font reads as that font's
characters where they are known (Word's Wingdings square bullet `\'a7` was
`§`). On the 108 `textutil` pages, DOCX now gives 261 list items (11 before),
exactly as many as the ODT and RTF versions of each page, whose lists are real
ones; 256 of them have the same level, number and text (two keep the
`<ol start>` numbers the ODT loses, two sit under section headings numbered in
their text that the page's CSS sets as body text, which therefore read as a
numbered list, and one differs by a thin space). The Word 97 files give 260 (12
before; the other two sit in a one-cell layout table). The strict word
comparison with the reference is unchanged in every format, and DOCX now has
the same 64 code blocks and 35 inline code spans as ODT and RTF. On 80 cases
written as DOCX, ODT and RTF (42 lists with 105 items: bullets by hand, by tab,
by Symbol and Wingdings codes, nested by tabs, spaces or indents, numbers in
twelve forms, checklists, a monospaced bullet; 38 look-alikes: dialogue in four
languages, attributions, quotations, scene breaks, numbered headings, numbers
that skip, inventories, years, initials, outline numbers, quantities, pros and
cons, section signs, bullets in a table cell, a heading or code), each format
finds every item and nothing else. Of these, 25 were written after the rules
were first set; they found three look-alikes read as lists (French dialogue
ending in full stops, Spanish dialogue with a spaced dash, pros and cons),
which the em-dash and mixed-mark rules now keep as text. Thirteen more cases
(dialogue with a speech incise in six languages, lists whose items hold
commas, continuation paragraphs, body text after a list) make 93: each format
finds all 120 items and nothing else (before: 13 dialogue lines read as items
and four continuations left the list). On the 21,825 distinct dash and `<li>`
list items of the local Markdown and HTML (crate READMEs and changelogs,
these documents, the reference's fixture pages) the incise rule fires on none;
of 25 dialogue scenes written for the check in fourteen languages with full
stops, it keeps the 19 that carry an incise as text. A bare exchange of
statements (`– Bonsoir.` `– Vous désirez.`) has nothing that tells it from a
list and still reads as one.

OpenDocument text and RTF follow the same conventions. An ODT run's font is
the face `style:font-name` names (its `svg:font-family`), else `fo:font-family`
(a list ending in the generic `monospace` counts), through `parent-style-name`
over the paragraph default style; RTF reads the font table's names (TextEdit's
fonts one after another or Word's group per font) and `\deff`. A face the
document only declares fixed-pitch (ODF `style:font-pitch="fixed"` or the
`modern` family, RTF `\fmodern` or `\fprq1`) is a code face too, unless its
name or RTF charset is CJK (MS Gothic, SimSun), whose Latin letters are
fixed-width under body text. Tab stops are a paragraph style's
`style:tab-stops` (a `style:leader-style` other than `none` is a leader) and RTF
`\tx` after `\tqr`, `\tqc`, `\tqdec` or a leader such as `\tldot`, until
`\pard`. On the 108 `textutil` pages, where ODT and RTF had no code at all, each
now gives 64 code blocks and 35 inline code spans, as DOCX does on the same
pages since a hand-typed list item set in Menlo became a list item with inline
code there too (it was DOCX's 65th block); the only words that leave ODT and
RTF output are 55 listing line numbers.

Raised and lowered text (`style:text-position`; RTF `\super` and `\sub`) is
written in Unicode super/subscript forms where every character has one. ODF
text a style hides
(`text:display="none"`) is left out, the base text of a phonetic guide is kept
without the guide, and comments (`office:annotation`) are left out with a
warning naming their number, as for Word. A chart embedded in an ODT or ODP
reads as its title and the data table the chart document stores, first row as
the header, instead of its replacement picture. In RTF a charset-0 font's bytes
are Windows-1252 whatever `\ansicpg` declares (TextEdit on a Chinese system
writes `\ansicpg936` over 1252 text), a hyperlink spanning paragraphs links
each paragraph's part, lists inside table cells stay lists, and a nested
table TextEdit closes with `\nestcell` directly after `\nestrow` (no `\itap`)
is read at the right depth; text after a nested table in a cell stays after
it. How the shared document renderer writes line breaks and escapes text is
described [below](#document-line-breaks-and-escaping).

RTF is read as Word shows it. Hidden text (`\v`) and tracked deletions
(`\deleted`) are left out, as Word's are, and a hidden or deleted paragraph
mark joins its paragraph to the next; inserted text stays. Index and contents
entries (`{\xe …}`, `{\tc …}`) are not text. `{\upr{…}{\*\ud{…}}}` is read
once, from its Unicode part (the code-page part has `?` for what the code page
lacks), and a `\uN` character in a Symbol or Wingdings font maps as the font's
bytes do (Word's `\u-3913` is the Symbol bullet). A header, footer, comment,
footnote separator or page background keeps everything it holds out of the
body: the text boxes, pictures, object results, footnotes and bookmarks inside
one ran into the document's first paragraph before (a letterhead's logo, a
footer's note). Legacy form fields show their state as in Word documents: a
check box `☒` or `☐` (`\ffres`, else `\ffdefres`), a drop-down list its chosen
entry (`\*\ffl`), a text field its result. A shape's own picture (the `pib`
property) is read in place of the Windows metafile copy in `\shprslt`, a
picture's or shape's `wzDescription` becomes its alt text, and WordArt shows
its words (`gtextUNICODE`) where it stands, its copy left out; a shape with no
picture or WordArt of its own still gives its copy, as before. TextEdit's
picture attachments (`\NeXTGraphic`, an RTFD's pictures, which an RTF read
alone does not have) no longer put their file name and attachment mark into
the sentence, and a blank line of a code listing laid out in a table cell is
kept (two listings of the corpus had lost one). Not read: Word 6 drawing
objects outside a shape (`\*\do` text boxes), OLE objects' data beyond their
result, and a text box's place (its paragraphs end the paragraph it is
anchored in, as before).

OpenDocument text reads frames and shapes wherever they stand. A frame,
picture or shape anchored to the page is a child of the body itself, which
upstream skipped with its text; in a text document it is now read where it
stands (LibreOffice writes it before the first paragraph of its page), while a
spreadsheet's shape anchored to a cell still adds nothing to the cell. A shape
holding text
(custom shape, rectangle, ellipse, line, connector, caption and the like),
in a paragraph or in a group, is read as a text box is, its paragraphs after
the paragraph it is anchored in; upstream ran its words into the sentence
(`Shape anchor.Custom shape words`). A `text:numbered-paragraph` (ODF 1.2's
numbered paragraph outside a `text:list`) is a list item labelled with the
number it shows, and consecutive ones that number on form one list; user,
table and object indexes are read like the table of contents; a heading's
`text:number` is its label instead of text run into the heading. A field set
to show nothing (`text:display="none"`, an invisible variable), a script's
source (`text:script`) and a section hidden outright (`text:section
text:display="none"`) are left out. Text in a Symbol or Wingdings face (by
`style:font-name`, through the face declarations, or `fo:font-family`) maps as
in Word documents; LibreOffice's Unicode OpenSymbol does not. Not read: a
section hidden by a condition (conditions are not evaluated) and paragraphs
hidden by a `text:hidden-paragraph` field (shown, as LibreOffice can be set to
show them); form controls show nothing.

A DOCX, ODT, RTF or Word 97 document with no heading style or outline level
anywhere often still shows its headings as short bold paragraphs set above the
body size (TextEdit and `textutil` save a web page's headings that way). In
such a document a plain top-level paragraph (not a list item, table cell, text
box or quote/code container) becomes a heading when all its visible text has
one size, at least one point above the body size in bold or a third above it
without bold, with at most 15 words (two CJK characters count as one), a letter,
no image, and no closing `.`, `,`, `;`, `:` or their full-width forms. The body
size is the size of most of the body's visible characters (notes excluded).
The largest such size is level 1, the next level 2, and so on to 6; the heading
drops its bold and the line breaks around it. Bold text at the body size stays
a paragraph, as does anything smaller. Headings styled at body size are not
inferred. A large bold sidebar label can be mistaken for a heading when the
source exporter has flattened its layout container.

Legacy DOC character formatting preserves underlining from direct runs and
inherited character styles as `<u>…</u>`. Hyperlink labels stay ordinary
links, since a link already reads as underlined. Explicit cancellation restores
plain text; unrelated repetitions of the same words stay
plain. Markdown has no underline delimiter, so viewers must support inline HTML.
Different underline stroke patterns are represented as ordinary underlining.

Legacy DOC floating pictures in the main document are emitted at their text
anchors, including supported image data stored separately in `WordDocument`.
Repeated references reuse one asset. Header pictures and complex grouped
picture placement are not reconstructed.

OOXML presentations have a separate reader because the generic document model
flattens slide boundaries. The package's presentation relationships and
`sldIdLst` determine slide order, including empty slides; filename sorting and
heading counts do not determine boundaries. Each slide internally starts with
`<!-- Slide number: N -->`; final Markdown omits these comments by default.
Keep them with `--page-markers` or `"output": {"page_markers": true}` in the
configuration file; `--no-page-markers` overrides that setting for one run. The
setting applies to PPTX, PPT and ODP slides and to PDF and multi-page TIFF page
comments, including base and enhanced output, files and stdout. It leaves
literal code examples intact; internal boundaries remain available for LLM page
alignment. For example: `markitai slides.ppt --page-markers -o out/`.

Shapes are stably ordered by their effective top/left
coordinates, with layout/master placeholder coordinates used when missing.
Group children are ordered within their group. As in the reference reader, the
first top-level placeholder with index zero supplies the first-level title;
its type need not literally be `title`. Grouped or later placeholders do not
turn the remainder of the slide into headings. Body text keeps its paragraph boundaries. A paragraph is a list item when its
bullet is set: the paragraph's own properties, then the shape's list style, the
layout's matching placeholder, the master's placeholder and the master's text
styles (`titleStyle`, `bodyStyle`, `otherStyle`) decide, the closest layer that
says anything winning, and nothing set is no bullet (a text box, a subtitle the
layout un-bullets, a `buNone` paragraph). Items are `*` for a bullet and `N.` for
automatic numbering (`startAt` honoured, counted per level; the numbering scheme
itself, such as letters, is not kept), nested by `lvl` the way an outline nests,
and a list is set apart from the text and the shapes around it by blank lines,
which would otherwise continue an item. This differs from the reference, whose
plain text-frame contract writes bullets as bare lines. A run's
`a:hlinkClick` naming an external `http`, `https` or `mailto` relationship is a
`[text](url)` link, runs of one target joined, in text frames, titles and table
cells; other targets (scripts, files, slide jumps) stay text. Speaker notes are
plain text without bullets or links, and the presentation's default text style is
not consulted. Shared normal-mode cleanup remains responsible for repeated
footers; pure output keeps the extracted text.

Presentation tables use their first row as the Markdown header. Images remain
at their shape position, keep normalized description text, and return their
original embedded bytes; shared image processing owns encoding. Repeated
references to the same package image share one asset. Cached chart category and
series values become a table; linked workbooks are never opened or recalculated.
Cached numbers read through their cache's (or point's) format code with the
spreadsheet readers' engine, so date categories are ISO dates rather than
serial numbers (`c:date1904` honoured) and `0%` values are percentages; text
points and General numbers are unchanged.
Speaker notes follow their slide under `### Notes:`, and its review comments
close it under `### Comments:`, one item per comment or reply, `Author: text`
on one line: legacy comments (`p:cmLst`, authors from `commentAuthors.xml`) and
Microsoft 365 threads (`modernComment_*.xml`, authors from `authors.xml`). A
slide hidden from the slide show (`show="0"`) keeps its content, with
`<!-- Hidden slide -->` where its slide begins. A SmartArt diagram reads as the
text points of its data part (`dgm:relIds/@r:dm`) as a bullet list in the order
the part lists them, as the Word reader reads one; its hierarchy and layout are
not kept, and a diagram whose data part is missing or holds no text keeps the
frame's own text with a warning. Missing or malformed slides retain their
numbered marker with a warning, while readable slides survive; a package with
no readable slide fails. Unknown shapes retain available DrawingML text with a
warning, and unsupported charts are explicitly identified. These fallbacks do
not imply complete drawing or chart-type support.

The vendored anydoc records where each slide of an ODP or a legacy PPT begins
(`slide_starts`, see its `MARKITAI-PATCH.md`), and the renderer writes the same
optional `<!-- Slide number: N -->` line (`--page-markers`) before each slide, blank slides included, with a
blank line between slides. This differs from the reference, whose legacy PPT
output has no slide markers and which does not read ODP. In these formats a
slide's speaker notes keep their place after it as a quote. An ODP table styled
with a header row (`table:use-first-row-styles`, Impress's default) takes its
first row as the header, as PPTX and PPT tables do. A legacy PPT whose persist directory is
unusable is read in raw stream order, where slides cannot be told apart, and
carries no markers.

A legacy PPT table is a group shape whose `tableProperties` mark it as one; each
cell is a shape of the group with its own text box. The vendored anydoc places
the cells on the grid their anchors draw (edges within 8 master units, 1/72
inch, are one grid line), keeps a cell spanning several lines as a merged cell,
adds the text of a shape stored over a cell to that cell, and takes the first
row as the header, as PowerPoint styles it. Border lines are ignored, trailing
empty rows are dropped, and a group with a cell that has no anchor keeps its
cells' text as paragraphs, as before.

A legacy PPT shape that shows an embedded OLE object (an `ExObjRefAtom` in its
client data) reads as the data the object holds, at the shape's place in the
slide's drawing order; before, such a slide kept only its text. The deck's
`ExObjList` names the object's storage (`ExOleObjStg`, zlib-compressed or not),
a compound file read by what it contains:

- a BIFF workbook (`Workbook` or `Book` stream). When the workbook window shows
  a chart sheet (an Excel chart object), or the stream has no sheet directory
  and holds a chart substream (MS Graph), the chart's cached series become a
  table under the chart's title: categories down the first column, one column
  per series named by its cached name (`Series N` when it has none), points
  without a category numbered from 1, and trendlines or error bars that cache
  no values of their own left out. Otherwise (an Excel worksheet object) the
  workbook reads as the XLS reader reads it; the name of a lone sheet with data
  is dropped and several sheets keep their names as paragraphs, because a
  heading would read as another slide title. Values show their number formats
  in both cases.
- LibreOffice's `package_stream`, an OpenDocument package: a chart reads as its
  title and the data table it keeps, first row as the header, as in an ODP,
  and a spreadsheet as its tables.
- an Excel 2007 object's `Package` stream, an OOXML workbook, read as one.

Equations, documents, pictures, linked objects and controls still add no
text; a debug log names each object left out and why. The object's preview
picture stays a document asset as before, while an image inside an embedded
spreadsheet keeps only its alt text. Decompression stops at the 128 MiB
entry cap and at 512 MiB for all of a deck's objects, chart caches nested
more than 64 `Begin` levels deep are rejected, and an object's tables count
against the 4,000,000-position grid budget each time a shape shows it.
An embedded worksheet is read whole, not only the cell range the object
displays on the slide.

The chart of the reference `sample.ppt` is a LibreOffice object; its table
matches both the chart's own data and the bar heights of the preview picture
PowerPoint displays. Excel and MS Graph objects are read by the [MS-XLS] chart
cache layout and tested with objects generated from the specifications
(`crates/markitai-core/tests/fixtures/legacy-ppt`); no object saved by Excel,
MS Graph or PowerPoint was available, so their real-world layout is not yet
verified.

A PPTX graphic frame holding a `p:oleObj` reads the same way: the object's own
part (`ppt/embeddings/…`) is a compound file read as above, or a zipped package
read as an OpenDocument chart or spreadsheet or else as an OOXML workbook (an
`Excel.Sheet.12` object), and its tables replace the frame's text. A linked
object, an unreadable one and one of any other kind (an equation, a document)
keep the frame's DrawingML text, with a warning naming the object's ProgID.
The part is read under the presentation's 64 MiB asset limit.

The presentation reader limits packages to 16,384 entries and 10,000 slides,
each XML part to 16 MiB, each asset to 64 MiB and total decompressed parts to
256 MiB. XML has per-part limits of 200,000 nodes and 127 nested elements.
The layout/master cache retains at most 8 MiB of source XML and 50,000 parsed
nodes; larger individual parts can be read within the part limits without being
cached. This avoids accumulating expanded layout trees across large decks.
Document types, repeated ZIP entry names, parts whose data differs from their
declared size, escaping package paths, duplicate relationship IDs and malformed
XML are rejected. Internal targets are resolved inside the archive as relative
references (`%XX` escapes need two hex digits; an escaped `/`, an empty segment
or a `//` network path is rejected); a `TargetMode` other than `Internal` is
never read from the package. External image HTTP(S) URLs may remain references
but are not fetched. Failed optional
notes, layouts, charts or images are named in warnings without discarding the
remaining slide content. These limits cover this reader's explicit passes, not
every allocation inside ZIP/XML libraries.

PDF pages with unreliable or missing native text retain their numbered marker
and emit a warning naming the page and the parser's OCR reason. Other pages are
retained. A document with neither readable text nor recoverable images fails
explicitly. An unnamed layout/GID rejection, or a scan classification caused by
declared but unused image resources, can fall back to bounded font-decoded plain
text only when execution inspection finds no used image, inline/pattern-image
warning, visibility signal or stream error, no Type3 font is present, and decoded
text passes conservative Unicode/content checks. Such
pages explicitly warn that reading order, paragraph boundaries and styling may
differ. A scan with executed raster content, garbled text, vector-text and
invisible-text reasons are never overridden by this fallback. Used image XObjects, including those inside nested Form XObjects,
are collected in page order and deduplicated by PDF object identity. Original
JPEG streams are preserved; supported 8-bit DeviceRGB/DeviceGray samples become
PNG. Images are appended to their page rather than positioned within its text.
Unsupported filters, color spaces, remapping, transparency masks and inline
images produce explicit warnings. Metadata/stream/image/asset limits bound the
additional package and image passes; they are not a claim that every upstream
parser allocation is bounded.

Eligible upright text pages also have a conservative positioned-text refinement
for document-wide heading levels (including bold headings at the body size),
paragraph gaps and short last lines, continuous styling, web and mail link
targets and complete ruled tables. Empty cells/columns are preserved; ambiguous
geometry retains the existing page reader. Running page headers repeated on most
pages are removed, and a document information title that is only a file name is
not the document title. See [PDF layout](pdf.md) for acceptance checks, resource
limits and the cost of the additional parsing pass.

PDF inspection reports invisible rendering modes, transparent text, white text
and very small text when they accompany text operators, including inside forms.
This is a diagnostic contract: complete hidden-text removal and
image-region text grouping remain
unimplemented. The pinned reader patch suppresses nonpainting Tr3/Tr7 text while
preserving graphics state and cursor movement; other visibility and layout
heuristics retain their documented limits in [PDF layout](pdf.md).
The browser runtime is outside this module. No Python interpreter,
Node runtime, Office installation, LibreOffice or hosted extraction service is
used by the readers above.

### Document line breaks and escaping

Text that goes through the shared document renderer (Word, OpenDocument, RTF,
EPUB, legacy PowerPoint and spreadsheets) follows the HTML reader's rules for
line breaks. A manual line break (Word `w:br`, ODT `text:line-break`, RTF
`\line`, a Word 97 vertical tab, EPUB `<br>`) is a hard break, written as two
spaces at the line's end; normal output's cleanup of line ends keeps them
where the next line continues the paragraph. Two or more breaks in a row end the
paragraph, breaks at the edge of a paragraph show nothing, and the
indentation after a break is dropped (it would otherwise start a code block
after a paragraph break). In a heading or a link's text a break is a space; a
break at the edge of a link's text moves outside the link (as a `<br>` at the
edge of an HTML `<a>` does), and a link the renderer does not write as one
(no destination, or a scheme other than web, mail and telephone, such as
`file:`) leaves its text and breaks in the line. In
a table cell a break is `<br>`. A list item whose label Markdown does not read
(`a)`, `(1)`, `一、`) keeps its line with a hard break after a paragraph, or a
blank line after a block a hard break cannot follow.

Text is escaped only where CommonMark (with GFM tables, strikethrough and
footnotes) would read it as syntax in that position, so `snake_case`,
`[!tip]`, `C:\Users`, `5 * 3` and a form's `________` stay as written:

- `*`, `_` and `~` runs that can open or close emphasis by CommonMark's
  flanking rules (also micromark's, which lets a run open or close beside
  another delimiter character) and have a partner in the line that could pair
  with them, or stand inside the renderer's own emphasis or touch its
  markers;
- a backtick run that a later run of the same length could close; one
  touching the renderer's code span is written `&#96;`, since a backslash
  would not part it from the span's backticks;
- `\` before punctuation or at a line's end; `<` before a letter, `/`, `!` or
  `?` when a `>` follows or it starts a line (an HTML block); an `&` that
  starts a character reference; `]` before `(` after a `[`; `[` that would
  open a footnote mark (`[^1]`) or follows a `!`; `!` before a link the
  renderer writes, as `&#33;` (normal output's image repairs would read
  `\![` as an image); and either bracket inside a link's text or an image's
  description;
- at the start of a line (a paragraph's, or after a break): ATX headings,
  `-`, `+` and `*` bullets, ordered markers, `>`, thematic breaks, fences,
  link reference definitions and HTML blocks; after a line, also setext
  underlines (`==`, `--`) and table delimiter rows. A delimiter mark there
  (`***`, `___`, ```` ``` ````, `~~~`) is escaped whole, since the rest of its
  run would still pair with one elsewhere;
- a heading's closing `#` sequence (`## Issue \#`).

A table cell holds inline content only, so block marks at the start of its
lines are not escaped there. On 18,000 generated one-paragraph Word documents
(text over an alphabet heavy in Markdown punctuation, with bold, italic,
struck, monospaced and linked runs and line breaks) every output parses with
micromark to exactly the document's text and styles, where the previous
escaping misparsed 1,008; the outputs carry 37% fewer backslashes. PDF
and PPTX text keep their own escaping (every `*`, `_`, `[`, `]`, backtick and
backslash); PDF paragraphs share the start-of-line rules above.

### Delimited text

A CSV or TSV file names no delimiter, and the extension is often wrong: Excel
in a European locale exports `;`-separated `.csv` files with decimal commas,
its "Unicode Text" export is tab-separated UTF-16 (with a byte-order mark)
under any name, and dumps use `|`. After decoding (see
[text encodings](#text-encodings)), each of `,`, tab, `;` and `|` splits the
first 50 records (at most 64 KiB), quote-aware, blank lines left out. A
delimiter qualifies when its most common field count is more than one and at
least four in five records have it, so a title line or a total row does not
outvote the table. The qualifying delimiter whose count the most records share
wins, then the one that gives more fields, then the extension's (`,` for
`.csv`, a tab for `.tsv`); when none qualifies the extension's stays, so
single-column and ragged files read as before. Commas never compete with a
qualifying `;` or tab when every field holding one is a number written with a
decimal comma (`1.234,50`, `99,00`, `€ 12,5`), so a European amount stays one
value even in a file without a header whose every row has one. A file whose
header and rows split the same way on two delimiters (`a,b;c`) keeps the
extension's. The reference sniffs only `.tsv` files, from their first line.

### Spreadsheets

Workbooks (XLSX, XLSM, XLSB, XLS and their templates) are read through one
grid assembly, so a workbook saved in any container converts the same:

- A formula cell saved without its value (openpyxl, pandas and other writers
  that do not calculate leave `<v>` out or empty) shows its formula as code,
  `` `=B2*C2` ``, and the workbook gets one warning counting such formulas
  with the remedy: open and save it in Excel or LibreOffice to compute them.
  A cached value, an empty string result included, is always what shows; no
  formula is evaluated. A shared formula's later cells, which carry no text of
  their own, stay empty but are counted.
- Hidden rows and columns are omitted, as hidden sheets are, and a warning
  names how many held content (`Worksheet "Data" has 1 hidden row and 1
  hidden column holding content; …`); hidden empty rows and columns are not
  counted.
- A cell's hyperlink to an `http`, `https` or `mailto` address is a
  `[text](url)` link (XLSX, XLSM). Links into the workbook (`#Sheet2!A1`) and
  other schemes keep the text.
- Cell notes follow the sheet's table as a list under `### Notes`, in reading
  order, each led by its cell (`* B3: text`), on one line. A legacy note's text
  is what the note box shows, which Excel begins with the author's name; a
  threaded comment (Microsoft 365) is one item per message, `Author: text`,
  and replaces the placeholder Excel writes for it into the legacy part. Notes
  of hidden rows and columns are omitted with them (XLSX, XLSM). Word and
  OpenDocument text comments stay out of the Markdown with a count warning:
  they annotate a span of text rather than a cell.
- A line break inside a cell is `<br>`.
- Number formats render as Excel shows them in the en-US locale. The currency (5–8) and
  accounting (41–44) format ids, which Excel always defines in the file and
  openpyxl does not, show their currency in the system locale, which the file
  does not name: they render with grouping, decimals and negative parentheses
  but no currency symbol (`(1,234.50)`), while a code that writes its own
  symbol keeps it (`"$"#,##0.00` gives `($1,234.50)`, `[$€-407]` gives `€`). Sections, conditions,
  `[Red]` and other colours, grouping, scaling, fractions and scientific
  notation follow their codes. Dates stay ISO (`2026-03-04`), since `m/d` and
  `d/m` are ambiguous, except a date format that names its month or weekday in
  English, or labels a numeric month the CJK way, and shows a four-digit year:
  `dddd, mmmm d, yyyy` gives `Wednesday, March 4, 2026` and
  `yyyy"年"m"月"d"日"` gives `2026年3月4日`. A format whose `[$-lcid]`
  locale is not English keeps the ISO date, since its month names are not
  English.
- A sheet's first row is its header. A first row that holds one cell merged
  across every column with content (a title) is written as a line above the
  table when the row it leads to holds at least two cells and no merge, which
  then is the header; a title that repeats the sheet's heading is not written
  again. ODS sheets do the same.

Large XLSX-family workbooks have fixed reading limits: at most 2,000,000 XML
nodes (elements and text runs) per package part, 128 MiB decompressed per part,
and 4,000,000 grid positions per sheet table. These measure the workbook's
structure, not its compressed file size or a fixed row count. Cell density,
inline strings, notes and other XML content affect when the limit is reached.
A node-limit error preserves the original reason and suggests exporting the
needed sheets as CSV or splitting the workbook into smaller XLSX files. Neither
`--max-depth` nor `batch.scan_max_files` changes this limit. Export each needed
sheet separately when using CSV: CSV keeps displayed text but loses sheet
structure, formatting, comments, links and formulas; check the exported result
before conversion. The converter does not return a successful partial workbook.

### Notebook outputs

A code cell is followed by what it produced, one block per output, in order:

- `stream` text (stdout and stderr; consecutive writes to one stream are one
  block) and `text/plain` results as a fenced block tagged `text`;
- `text/markdown` results as Markdown;
- `image/png`, `image/jpeg`, `image/gif`, `image/webp` and `image/bmp` outputs as
  image assets (`![output](.markitai/assets/notebook-image-N.png)`), which then
  pass through the shared image filtering and compression; an image replaces the
  result's `text/plain` form, which is only the object's name (`<Figure size ...>`);
- `error` outputs as the traceback in a `text` block, or `ename: evalue` when
  there is none.

Terminal escape sequences are removed from printed text and a carriage return
keeps the last state of its line (a progress bar). Output is bounded: a block
keeps its first 40 and last 40 lines (a traceback its first 6 and last 24) with a
`… [N lines omitted]` note, a line keeps 500 characters, the notebook keeps 4 MiB
of output text and 64 MiB of decoded images, and what is beyond is replaced by a
note with a warning. Outputs with only other types (`text/html`,
`application/json`, widgets) are not converted and a warning names their types;
a malformed output is skipped. In markdown cells, `![](attachment:name.png)` and
`<img src="attachment:name.png">` point at the attachment's image asset when the
cell uses it. Nothing in a notebook is executed or fetched.

## Explicit remaining compatibility work

Numbers table decoding accepts modern single-file ZIP and directory packages with scoped limits in [Numbers](numbers.md). A `.numbers` directory is one document, including when its contents are invalid; ordinary directories keep their existing batch/API behavior.
Office presentations and word-processing files can opt into complete page capture
and local OCR supplements through [isolated LibreOffice export](office-rendering.md);
this optional installed program is separate from the CLI binary. Native text
extraction does not launch it. Templates are captured as the documents they
make. XLS, XLSX, XLSM, XLSB and ODS screenshots use
complete-sheet export, including hidden and empty sheets; this is not printed-page
pagination. Numbers screenshots and OCR remain unsupported. See the Office guide
for import fidelity, font handling and resource limits.
macOS HEIF/AVIF primary images use native
ImageIO decoding, with the scoped limits in [images](images.md); other platforms
still return an explicit unsupported error. Local and static/automatic URL PDFs support explicit
page rendering, screenshots and OCR through the [PDF media pipeline](pdf-ocr.md),
including its documented accuracy gap. URL media preserves original request
identity while processing downloaded bytes without a second download.
Local image OCR uses [Vision on macOS or the portable engine on Windows/Linux](ocr.md);
the portable engine requires separately installed model weights. Standalone SVG
rasterization is implemented with bounded native rendering; referenced embedded
images can use [caption/description analysis](image-enrichment.md).
Multi-page TIFF preserves its original download and every page preview, applying
orientation before local OCR or complete-page model requests. Pixel, page and
encoded-byte budgets reject oversized documents without silently omitting pages. `supports_extension` reports local text readers; standalone image
classification and vision extraction are separate orchestration paths. See
[images](images.md), [MSG](msg.md), [markup](markup.md) and [HTML](html.md) for
their scoped contracts and limits.

The HTML reader implements article selection, structural chrome removal,
technical code blocks, scoped footnote/math recovery and structured announcement
content. It does not reproduce the original engine's complete set of site
resolvers, conversation-thread models, adaptive recovery, schema.org body
fallback or browser CSS visibility. The precise implemented boundaries are in
[HTML](html.md), [code blocks](html-code.md) and
[article selection](html-article.md). Basic success on an HTML fixture is not
evidence that its full extraction contract matches.

The [EML reader](eml.md) resolves Content-ID images within the selected MIME body
scope and retains missing or ambiguous references with warnings. The email readers
preserve body and attachments, but complete header, attachment
and layout parity is pending. XML now has structured prose and the sample fixture
is exact; arbitrary dialect parity remains open.
Office conversion can differ in whitespace, table header selection, numbering,
anchors, font-driven headings and metadata. Such differences must remain visible
in differential reports rather than being normalized away.

As in the reference, legacy DOC/PPT, RTF, ODT and ODS output ends with a newline
while DOCX, XLS/XLSX and EPUB output does not; headings carry no trailing spaces.
Presentation output has no trailing whitespace on a line and collapses the blank
runs that text-free shapes leave, as the reference's final pass does.

Word 97 files saved by the macOS exporter (TextEdit, `textutil`) declare a mini
stream no stream uses and write it inconsistently: its MiniFAT chains unused
mini sectors to sector 0, or the FAT is one sector short so the exporter's own
directory, MiniFAT and mini-stream sectors lie beyond it. Strict OLE readers,
the reference's included, reject every such file. When the original fails, a
bounded repair reads a copy that appends the missing FAT entries (extending
the directory while its entries name siblings not yet read) and detaches the
unused mini stream; a file whose mini stream holds a stream is never changed,
the original error stands if the copy fails too, and a warning reports the
repair.

Word 97 text raised or lowered (`sprmCIss`, set directly, by a character style
or by a paragraph style) is written in Unicode superscript or subscript forms
when every character of the run has one (`claim.¹`, `H₂O`, `x₁`), as the DOCX,
RTF and ODT readers write it; a run with a character that has none (`1st`)
stays at the baseline rather than half converted.

That exporter also writes a picture as the object replacement character U+FFFC
and stores no picture data (the file has no Data stream). The reader drops the
character, so no stray `\ufffc` paragraph remains where the picture was, and
extracts the picture of a file whose character does carry picture data
(`sprmCPicLocation`, as for the Word special character `\u{1}`).

Adjacent text runs of one style render as one run, so a word the source split
into runs (RTF writes each `\u` character as its own; Word splits at revision
marks) keeps one emphasis span. A source list label that is only digits becomes
`1.`, a bullet glyph becomes the list's bullet, and other labels stay as the
source wrote them. In document formats (not spreadsheets) a table whose rows
never hold two non-empty cells, and whose cells hold a table or several blocks
with content, lays out a saved web page: its cells are written as the document's
blocks. A table nested in a data table's cell becomes one line per row. Tables
of single paragraphs keep their structure even with empty columns.

The renderer retains referenced anchors and omits unused ones. EPUB links within
the assembled book keep working through these anchors; they intentionally differ
from the reference's links to source XHTML files that are not exported. A kept
anchor is written as `<a id="..."></a>` on a line of its own before the heading
or paragraph it marks, never inside a heading's text (a Word bookmark around a
heading, a table-of-contents target, an EPUB chapter's start), so headings
and their slugs stay clean; a heading in a table cell, which has no lines of its
own, keeps the anchor inline. DOCX
tables keep their first row as data with a blank Markdown header. ODS uses its
first row as the header and removes trailing columns that are empty in every row,
whatever the sheet declares or a merged title spans (a title merged over five
columns above three columns of data gives three, as the reference renders the
ODS fixture); empty columns between filled ones stay. A merged title row above
the real header is text above the table, as in XLSX (see
[spreadsheets](#spreadsheets)); the reference makes it the header. RTF heading bold markers are omitted while other emphasis is
retained. Hidden XLS/XLSX/XLSB worksheets, rows and columns are omitted by the upstream parser
and reported explicitly. Older XLS code pages other than
Windows-1252, exact presentation image encoding, PDF table
layout and PDF image placement require further compatibility work.

## Error and output principles

- A malformed or unsupported input returns an error. An error description is
  never persisted as a successful Markdown document.
- Native document parsers do not call remote OCR automatically.
- HTML links are restricted to ordinary HTTP(S), mail and telephone references.
  Relative references become absolute when the caller supplies a source URL; a
  saved web page's links resolve against its `<base href>` or canonical link,
  while its image paths and other local files' references stay relative. Script links lose the destination while
  keeping their visible label.
- Format adapters return embedded bytes rather than downloading remote images.
- Text readers retain original Markdown frontmatter for the output layer to
  merge according to its public contract.

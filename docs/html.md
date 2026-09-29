# Native HTML extraction

The Rust HTML reader parses a DOM with `scraper`, selects an article candidate,
serializes an allowlisted representation, and renders Markdown with `htmd`.
It does not run JavaScript, fetch images, follow links, start a browser, or call
the Python implementation. The same reader serves local HTML and fetched HTML;
fetching itself belongs to the separate HTTP layer.

Full-page article selection and structural widget removal are described in
[article boundaries](html-article.md). That policy is disabled for document
fragments so EPUB/Office/email content retains its own tables of contents.
[Technical code blocks](html-code.md) describes highlighter normalization,
literal content, language labels and resource limits.

## Metadata

Title selection follows the reference metadata layer: JSON-LD `headline`, then
JSON-LD `name`, document `<title>`, and Open Graph title. Twitter title is an
additional fallback. JSON-LD arrays and nested `@graph` objects are inspected;
malformed JSON is ignored. Scripts remain excluded from rendered content.

Repeated site prefixes and suffixes are removed when the site name is known.
A truncated title can be replaced by a longer matching document or Open Graph
title. Without site metadata, a ` | Site` suffix is removed only when a headline
or first heading matches the preceding text. Ordinary hyphenated subtitles stay
intact. Pathological titles are capped at 300 Unicode characters, preferably at
a word boundary, followed by an ellipsis.

Author meta tags precede JSON-LD author strings or author objects. Publication
metadata precedes JSON-LD `datePublished`, then a date-shaped `<time datetime>`.
Description and site values come from meta tags. Page canonical URLs are retained;
a homepage canonical on a deeper source URL is discarded. The local-format
dispatcher must preserve the reader's title so normal output can use it.

## Rendering and URL handling

Full HTML documents first run bounded [streamed-content recovery](html-stream.md).
Recognized React transport instructions are applied to existing DOM nodes without
executing JavaScript. Sparse script-stripped snapshots use a separate, explicitly
documented article heuristic; ordinary hidden descendants remain hidden.

- `<q>` retains quotation marks and nested inline markup. Source newlines remain
  soft Markdown line breaks outside code; structural block elements establish
  paragraph boundaries. This follows the source renderer's line behavior, which
  can differ from a browser's visual whitespace collapse.
- Preformatted code keeps indentation and empty lines. A `<code>` element with
  inline `white-space: pre` styling is treated as a code block. Chemical and
  mathematical `<sub>`/`<sup>` text retains those tags.
- Valid absolute HTTP, HTTPS, mail and telephone URLs retain their spelling.
  URL validation does not add a slash to a hostname-only link or change percent
  escapes. Literal angle brackets and double quotes are percent-encoded before
  Markdown emission. Relative URLs resolve against a supplied page URL. Local extraction
  without a base retains relative destinations.
- An image whose source is inline `data:` content keeps its alt text and
  position as the reference's `![alt](data:<type>...)` placeholder; the payload
  is never retained. `data:text/html`, `data:image/svg+xml` and
  `data:text/javascript` images are removed, and links never accept `data:`.
- URL attributes with control characters or unsafe schemes are removed. Event
  handlers and arbitrary styles are not emitted. Script, form, frame, template,
  hidden and navigation content is excluded. A hidden ancestor also disqualifies
  an article candidate. DOM serialization stops with an error beyond 256 levels.

Inline visibility uses parsed CSS declarations rather than substring matching.
Only the actual `display` and `visibility` property names affect this check;
custom properties such as `--footer-display: none` do not hide content. Quoted
strings, comments and nested function/block values cannot introduce declarations.
Recognized keyword declarations honor order and `!important`; `display: none`
and `visibility: hidden` or `collapse` hide the element. The HTML `hidden`
attribute still applies independently. This does not compute CSS variables,
escaped property names, external stylesheets or the browser's complete cascade.
The correction follows the r1 corpus audit; that historical artifact and its
two recorded conversion errors remain unchanged.

## Mathematical content

Mathematics is recovered as TeX data before ordinary script removal. Only an
exact `math/tex` script MIME type is recognized, with optional `mode=display`.
JavaScript and other executable script types are still discarded. No script,
TeX engine, browser or external renderer is executed.

The reader recognizes KaTeX, MathJax, MediaWiki and Temml wrappers. It prefers
explicit `data-latex`, `data-math`, `data-entry` or `alttext` values, then TeX
annotations, then presentation MathML structure. A recognized wrapper can read
its hidden assistive MathML without unhiding unrelated content. The wrapper is
replaced once, preventing duplicate visual glyphs, fallback images and accessible
math from appearing together. MathJax v2 preview siblings are discarded only
when their parent also contains a nonempty TeX-source script.
Structural traversal and text collection discard active script/style elements
and XML annotations, including those nested inside MathML text elements.

Structural MathML handles nested rows, fractions, roots, sub/superscripts,
under/overscripts, common accents and Greek/operators, fenced expressions and
tables. Display math uses `$$…$$`, inline math `$…$`; TeX source is preserved
without Markdown backslash escaping. Literal angle brackets in a math source
become TeX comparison commands so annotations cannot emit raw HTML. MathML depth
is bounded at 64 elements, with an explicit conversion error beyond that limit.

This is not a complete MathML or TeX renderer. Raw dollar/backslash-delimited
math in ordinary text, visual-only MathJax CHTML/SVG reconstruction, TeX image
services, arbitrary MathML layout/variants, multiscripts and exact tagged-equation
layout remain outside this round. Unknown presentation elements retain their
child text/structure where possible; specialized layout can differ. Existing
source code blocks are kept separate from math interpretation.

## Footnote recovery

The reader collects references and definitions from the immutable DOM before
sanitization, then emits numeric Markdown references and a definition section.
Repeated references share one definition. Numbering follows definition order in
the document, including inline notes. Replacing a reference wrapper preserves its
visible leading/trailing whitespace; hidden popover text cannot supply whitespace
to the surrounding article. Ordinary link labels remain links; numeric
markers and explicit footnote semantics establish a reference.

Supported structures include footnote/reference lists, `doc-footnote` and
`doc-noteref` roles, `fn:`/`fn-`/`ftnt` identifiers, named anchors, Word export
backlinks, Google Docs notes, WordPress/Easy Footnotes lists, standalone
`footnote-definition` blocks, inline popovers, Org-style labels and sidenotes,
`aside > ol[start]`, and `data-definition` references to hidden asides. A labeled
footnote section outside the selected article can supply definitions, while
unrelated external content stays outside the article.

Generic numeric anchor targets need a note/reference signal in the section's
role, identifier, class or heading, at least two matched definitions, and coverage
of at least three quarters of the external numeric targets. Concentration alone
is insufficient: ordinary numbered instructions and navigation remain links.
A wrapper containing the article references itself does not qualify. This also
keeps ordinary equation/theorem links from becoming footnotes. Fragment identities
are parsed once for this scan, and sections with fewer than two matching targets
are rejected before coverage checks. Loose
numbered paragraphs require at least two corroborating body superscripts; an
explicit footnote class can identify a single definition. Section delimiters
allow continuation paragraphs/lists to move with the definition; otherwise only
the numbered paragraph moves and following updates remain in the article.
Generic named/ID definitions adopt only adjacent paragraphs, lists, quotes and
code blocks under the same parent. Another target, an explicit identifier/named
anchor, a section boundary, or a related/update label stops adoption. Hidden
continuation blocks are never promoted to visible definition roots. A plain
leading `N.` label is removed only when it exactly matches the resolved source
reference number; years, decimal values and unmatched prose prefixes are kept.

Only resolved definitions with readable content after backlink removal are
removed from their original positions. Missing targets, unreferenced definitions
and backlink-only entries remain visible; they do not create dangling references.
Ordered-list start offsets use checked arithmetic, and output numbers are assigned
after empty definitions have been excluded. Return links and numeric
labels are removed within resolved definitions, while ordinary body links remain.
A duplicate inline sidenote is removed only when its text matches the adjacent
resolved definition. Lists, block quotes, code and later paragraphs receive
Markdown continuation indentation instead of being flattened or detached from
the footnote. This can differ from the source renderer's unindented continuations.

Footnote bodies use the same URL validation, script removal, depth bound and
math/code handling as the article. An explicitly linked hidden definition may
expose its own content root; unrelated hidden descendants remain hidden. The
reference itself and its ancestors must remain visible and outside discarded
navigation/forms. Hidden inline popovers cannot reintroduce their definitions.
Detection excludes code, scripts, MathML and every recognized math wrapper,
including KaTeX and MathJax. Duplicate MathJax previews with a sibling TeX source
are excluded as well; visual superscripts cannot introduce references or move
otherwise unreferenced definitions. Remote URLs with matching fragment names do not
become local references unless they identify the current page or carry explicit
HTMLBook `noteref` semantics.

This is structural recovery, not the entire upstream footnote system. Multiple
named definitions packed into one paragraph with only `<br>` separators,
arbitrary publisher citation groups, separate sidenote columns and recursive
references inside definitions are not fully standardized. Malformed or
ambiguous structures retain their ordinary content where possible. Definition
order and punctuation/line-layout choices can differ from the reference. The
reference local API itself leaves some labeled-list and line-break patterns
unstandardized; corpus strict parity and recovered-content quality remain
separate measurements. Historical audit results are not rewritten.

## Structured BBCode announcements

`data-partnereventstore` JSON can contain the primary article in
`announcement_body.body`. The reader reconstructs the first nonempty readable
announcement from an array or object and uses its headline and timestamp;
`data-groupvanityinfo` can supply the group/author. Invalid JSON falls back to
ordinary HTML extraction. No JavaScript state is evaluated.

The BBCode reader builds a bounded nesting tree. It supports paragraphs, headings,
bold/italic/strike markup, quoted blocks, nested ordered/unordered lists
with implicit `[*]` item endings, links, images, literal code blocks and YouTube
preview identifiers. Raw HTML text retains entity escaping in the final Markdown,
after intermediate HTML decoding; code blocks keep their fenced literal contents.
Body text, headings, link labels and quote attribution share this protection.
Generated images have an empty alt label; their destinations are validated.
Link/image targets use the same URL validation as HTML. Legacy slash/quote escapes are normalized in tag arguments
without rewriting literal code contents. Unknown tags remain visible as text,
and nesting beyond 64 tags fails explicitly.

The reader does not select among announcements by page/event ID, implement every
BBCode dialect, resolve arbitrary embeds or fetch media. Underlined text is
retained without underline styling. Ordered list marker
styles and complex malformed-tag recovery can differ from upstream renderers.
In the r2 corpus, the source **local file API** returns empty Markdown for the
BBCode fixture, while its upstream **URL extraction** expectation contains the
announcement. Rust intentionally recovers that body and metadata; this remains
a strict local-API difference rather than an exact parity pass. URL-expectation
quality must be checked separately. Historical audit reports are unchanged.

The generic reader is not yet the source's complete web extraction pipeline.
It does not replicate its full site resolvers, browser/CSS visibility model,
adaptive content scoring, schema-body fallback, full footnotes, callouts, shadow
roots or complete math standardization. Static input can therefore retain site chrome, omit
content outside the selected article, or differ in headings, tables and images.
Those differences must be measured rather than inferred from successful parsing.

## Corpus diagnostic

The [latest full corpus audit](validation/html-corpus.md) records r5: 43/209
strict local-file API matches, 166 output differences and no conversion errors.
Fourteen of 28 footnote fixtures match exactly. All r3 strict passes remain;
the single lost r4 match preserves source whitespace that the reference local
API drops. The audit also verifies repaired continuation ownership in the
Dhammatalks fixture. Successful conversion does not establish full compatibility.

The reference checkout contains 209 HTML files paired with 209 upstream expected
Markdown files under `packages/markitai/tests/defuddle_fixtures`. Its quality test
suite checks nonempty text, title presence, site-chrome phrases and word-count
tolerance. Passing those heuristics does not establish exact output parity.

Run the new diagnostic from the Rust repository after a coordinator release build:

```sh
python3 scripts/audit_html.py \
  --reference /Users/example-user/work/markitai \
  --library target/release/libmarkitai_ffi.dylib \
  --output .local/audits/html-corpus-20260928-r5 \
  --jobs 4
```

Use `.so` or `.dll` on other platforms. The reference virtual environment is used
by default; `--reference-python` can select another installed environment.
`--pattern 'elements--*'` and `--limit 5` make an explicitly labelled subset.
The output directory must be empty and outside the reference checkout.

Each fixture runs through both **local file public APIs** in separate subprocesses.
The harness reuses `audit_formats.py` for worker isolation: explicit configuration,
disabled LLM/OCR/screenshot/alt/description features, disabled cache, an isolated
`MARKITAI_HOME`, and a credential-free child environment. Source files are hashed
before and after each conversion. A frozen copy of the native library, worker
responses/logs, exact diffs and a machine-readable report remain in the audit
directory. The source checkout is never edited.

A strict parity pass requires exact Markdown, metadata except the processing
timestamp, asset names/content hashes, warnings and skip reason. Matching failures
are not passes. `--require-parity` returns a failure exit code if any selected
fixture differs or errors.

Upstream expected-body equality, title equality, word ratios and chrome phrases
are reported separately for both implementations. These are **diagnostics**:
upstream expectations describe URL extraction, while this harness exercises local
file conversion, including the reference's fallback behavior. They do not prove
URL/site-resolver parity, do not count as migrated source tests, and do not turn a
strict output mismatch into a pass. The report records selected/available fixture
counts, source revisions and the exact library hash. It makes no performance claim.

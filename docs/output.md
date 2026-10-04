# Output assembly

The Rust core separates extraction from output assembly. Readers return Markdown,
source metadata and byte assets. The output layer resolves the public title,
normalizes ordinary Markdown, reserves paired filenames and writes files atomically.

## Ordinary and pure output

Ordinary local output has generated `title`, `source` and `markitai_processed`
fields. Frontmatter strings that a YAML 1.1 loader such as the reference's PyYAML
would read as a timestamp, boolean, number or null, including the processing
time, are single-quoted exactly as the reference writer does, so they stay strings. Reader metadata does not automatically become local frontmatter. URL
output can retain trusted fetch metadata; canonical fields and unreliable language
metadata are excluded from that merge.

Explicit reader titles take precedence. Otherwise the basic workflow uses its
first heading or the source stem. CSV, TSV and XML fall back to the complete source
filename because a data row or element name is not a reliable document title.

Ordinary Markdown receives heading spacing, trailing whitespace and blank-line
cleanup, residual placeholder removal, broken-link repair and repeated page-footer
cleanup. It ends with one newline. Existing input YAML remains part of the body
beneath the generated frontmatter, matching the reference workflow.

Pure output bypasses this normalization. Existing YAML is parsed for API metadata,
but its original bytes and the original body are retained for file output. Pure
input without YAML has an empty frontmatter map. The OKF profile still adds its
required document type and generator identity; generated timestamps use UTC `Z`.

Pure model requests receive the complete original reader Markdown, including any
input YAML. API metadata is parsed from the output actually retained. Normal LLM
output is normalized separately; a retained base file keeps its own metadata.
Normal text enhancement validates a structured description and tags, which are
merged into the enhanced frontmatter. The source title, source identity and
processing timestamp remain application-owned. Protected code, links and page
markers are restored before final output; pure and visual requests retain their
separate request contracts. See [document processing](llm.md).
Profiles run after enhancement on both outputs so restored page markers and
image references receive the same transformations as reader-produced content.

## Names from URLs

A URL names its output after the last path segment, percent-decoded when that
yields UTF-8 (`https://example.com/path/to/doc` gives `doc.md`, and
`/posts/人是什么单位` gives `人是什么单位.md`), or after the host when there is
none (`example_com.md`). When the URL has a query, the host comes first
(`example_com_8080_search.md` for `https://example.com:8080/search?q=x`). Names
are bounded to 200 characters and replace characters that file systems reject.

Two refinements keep different pages apart, and differ from the reference, which
named `news.ycombinator.com/item?id=1` and `?id=2` both `news_ycombinator_com_item`:

- A query parameter that names the page itself adds its value: `id`, `v`, `p`,
  `pid`, `tid`, `nid`, `aid`, `vid`, `story`, `article`, `post`, `topic`,
  `thread`, `item`, `video`, `doc`, and the `…id` and `…_id` spellings of those
  and of `page` (a bare `page` is a page number and is not used).
  `youtube.com/watch?v=abc`
  becomes `youtube_com_watch_abc`, `news.ycombinator.com/item?id=8863` becomes
  `news_ycombinator_com_item_8863`, and WordPress's `example.com/?p=123` becomes
  `example_com_123`. The first such parameter wins; its value keeps letters,
  digits, `-`, `_` and `.` (other runs become one `-`), is at most 64
  characters, and is left out when the last segment already contains it.
  Searches (`q`), views, tracking parameters and anything that could be a secret
  never enter a name.
- A post on X or Twitter (`x.com`, `twitter.com` and their `www.`, `mobile.`,
  `fxtwitter.com`, `vxtwitter.com`, `fixupx.com`, `fixvx.com` and `twittpr.com`
  forms) is named `<user>-status-<id>`, so `https://x.com/NASA/status/20/photo/1`
  gives `NASA-status-20.md` rather than `1.md`; `/i/web/status/<id>` uses `i`.

## Files and assets

On Unix, documents, assets, image sidecars and reports are created with the
process umask (commonly 0644), like ordinary files and the reference writer.
Ownership records, receipts, recovery state and locks keep their private modes.
Windows uses inherited ACLs rather than Unix mode bits; private coordination
files additionally undergo the [owner and identity checks](output-ownership.md#platform-primitives).

Output names preserve the source extension: `report.pdf.md` and
`report.pdf.llm.md`. Both share a conflict namespace; renamed results begin with
`report.pdf.v2`. The CLI never renames silently: a single conversion writes
`Wrote out/report.pdf.v2.md (report.pdf.md already exists)`, a batch summary lists
its renamed results, and `--dry-run` prints each item's final file name with the
conflict policy already applied (`.v2.md`, `skip (…)`, or `replaces the existing
file`), computed with the same naming rules and without creating anything. A
batch that discovers files skips dot-files and dot-directories, `node_modules` and
Office lock files (`~$name.docx`) unless a glob names them. An image that can only
end as a skip or an error (no `--ocr`, no model) is decided before the output
claim, so it creates neither an output directory nor ownership files.
Explicit CLI filenames and batch-reserved stems reach this layer
as private configuration fields. Assets are addressed by SHA-256 content prefixes.
The public result contains only durable asset paths when an output directory was
provided. Asset remapping recognizes inline images/links, wiki references,
reference definitions, HTML media attributes and candidate lists, and
[CSS resource destinations](css-resources.md), including multiline HTML.
It preserves titles and unrelated attributes while protecting fenced/indented
code, inline code spans, comments and pre/code/script bodies. Asset preparation
and publication each use one original-to-final path map, so an inserted filename
cannot be mistaken for another original reference. Query/fragment suffixes are
kept separate from replacement filename data.
Reference definition destinations and optional titles must share one physical
line. Filtered reference images are removed together with their definitions;
download-link labels remain readable.

The core serializes file reservations and writing within one process. The CLI
also has [per-member publication ownership](output-ownership.md); this does not
turn the in-process binding API into a batch recovery coordinator. CLI document
publication orders its intermediate steps and completes one
[durability fence](output-ownership.md#ordering-and-durability-fences) per volume
before reporting success. The core never synchronizes the directory entry of an
asset, sidecar or immediate (binding) document it writes, so none was guaranteed
durable when the write returned; each staged file is synchronized and then ordered before its
rename (`F_BARRIERFSYNC` on a verified local macOS APFS/HFS volume, `File::sync_all`
elsewhere), so a name that survives a crash refers to complete bytes. Existing
content-addressed asset bytes are verified before reuse. The policy checks of a
document's base and enhanced members share one walk of their common parent, and
every write observes the path afresh. Malformed input
frontmatter remains content. [Round-eleven measurements](validation/html-media-round11.md)
record a limited asset-heavy CLI comparison; binding overhead and peak memory
remain unmeasured by that experiment.

# Markitai's HTML-to-Markdown converter

This directory contains the published `htmd` 0.5.5 package's library source,
integration tests, the example page one of those tests reads
(`examples/page-to-markdown/html/Hacker News.html`), `README.md`, `LICENSE`
(Apache-2.0, unchanged) and its package manifest. `UPSTREAM.json` records the
original archive checksum, upstream commit and every copied file's original
checksum. No Cargo registry source was modified. The workspace coordinator
owns the path patch and lockfile.

## Why

Markitai parses a page with scraper (html5ever 0.39), cleans it, and gives
htmd the cleaned markup as a string. Upstream htmd parsed that string with
html5ever 0.38 into a markup5ever_rcdom tree, so the binary carried a second
copy of html5ever's tokenizer and tree builder (generic code instantiated for
the rcdom sink, about 137 KB) beside markup5ever 0.38 and markup5ever_rcdom,
which also pulled in xml5ever. This copy parses with scraper's own entry point
and walks scraper's tree, so one parser instance serves both parses and those
crates leave the dependency graph.

Parsing the cleaned markup again still cost a run of html5ever over every page
(about 9% of Markitai's HTML conversion time once its attribute lookups were
fixed). `TreeWriter` builds the tree that markup parses into from its parts as
Markitai writes them, so a page is parsed once. Building it with html5ever's
own tree builder (fed tokens instead of text) would have added a second
instance of that generic code, 132,320 bytes in a release CLI (scraper's
instance is private to its parse functions); the writer instead applies the
tree construction rules markup written from a tree needs and leaves the rest
to the parser.

## Changes

Each change is marked `markitai` in a comment.

- `Cargo.toml`: depends on scraper 0.27 (without default features),
  html5ever 0.39 and ego-tree 0.11 instead of html5ever 0.38 and
  markup5ever_rcdom 0.38. The benches and examples are not vendored, so their
  targets and the criterion dev-dependency are gone; the tests use the regular
  scraper dependency instead of scraper 0.26. A `scripting-option` feature (on
  by default) keeps `HtmlToMarkdownBuilder::scripting_enabled`: parsing with
  scripting disabled needs this crate's own instance of html5ever's generic
  parser (scraper's entry point takes no options), which measured 123,616
  bytes of code in a release CLI. Markitai never disables scripting and
  depends on this crate without default features. A `markitai_tests` target
  is added.
- `src/lib.rs`: `Node` is scraper's node and `NodeRef<'a>` names a node of its
  tree. `Element::node` is a `NodeRef` and `Element::attrs` scraper's
  attributes; `Element::attr` returns the first attribute with a local name,
  as the built-in handlers look one up. `html_to_tree` returns a
  `scraper::Html` (through `Html::parse_document`, or with scripting disabled
  through html5ever's driver with scraper's sink) and `tree_to_markdown` takes
  any `NodeRef`.
- `src/element_handler/mod.rs`: `Handlers` is implemented by a per-conversion
  `Walker` instead of `ElementHandlers`, and gains `node_children` and
  `node_text`. While walking a node's children upstream merged each inline
  element into an identical previous sibling holding only text, by editing its
  `Rc` tree: the second element left the parent's children and its text was
  appended to the first's. The scraper tree is shared and not edited; the
  walker records the same removals and appended text, and every read of
  children or text in this crate goes through it. A merge therefore applies
  from the moment the parent's children are first walked, as the edit did,
  and children that are never walked (an ordered list's, read one by one) are
  never merged. `can_combine` moved here from `dom_walker.rs`; a template's
  contents, which scraper keeps in a fragment child and markup5ever_rcdom kept
  outside the children, are not among the children.
- `src/dom_walker.rs`, `src/node_util.rs` and the handlers in
  `src/element_handler/` (`code.rs`, `pre.rs`, `list.rs`, `li.rs`, `span.rs`,
  `table.rs`, `td_th.rs`, `html.rs`, `img.rs`, `anchor.rs`): read scraper nodes
  and attributes, and children and text through `Handlers`. `pre.rs` computes
  its faithful-mode test only in faithful mode.
- `src/element_handler/element_util.rs`: faithful-mode HTML serializes a
  subtree as markup5ever_rcdom's `SerializableHandle` did, with the children
  and text the conversion sees.
- `src/element_handler/img.rs`: the lines of an alt text or title join with a
  space instead of a line feed, and a title with nothing in it is left out
  (see the differences below). `tests/html/turndown_test_index.html`: the
  three cases that expected the line feed ("img with a new line in alt", "img
  with more than one new line in alt", "img with new lines in title") expect
  the joined text.
- `src/element_handler/emphasis.rs`: an emphasis element inside one that
  writes the same emphasis (`b` or `strong` in either, `i` or `em` in either,
  in the same block) adds no markers (see the differences below).
  `tests/markitai_tests.rs`: two tests for these changes.
- `tests/code_tests.rs`, `tests/basic_tests.rs`: the three tests that used the
  rcdom tree or `Attribute` directly use the scraper equivalents;
  `faithful_mode_inline` expects attributes in name order (below).
- `src/tree_writer.rs` (added) and its export from `src/lib.rs`: `TreeWriter`
  takes start tags, attributes, end tags and text, standing for the markup
  `<name a="v">`, `</name>` and escaped text, and returns the `scraper::Html`
  that `Html::parse_document` builds from that markup, or `None` where it
  leaves the markup to the parser. It reads text and names as html5ever's
  tokenizer does (line breaks, NUL, a leading byte order mark, lowercased
  names, the first of two attributes with one name, the raw text of `title`,
  `textarea`, `style`, `xmp`, `iframe`, `noembed`, `noframes`, `noscript` and
  `plaintext`) and applies html5ever 0.39's tree construction rules for the
  implied `html`, `head` and `body`, head elements before and after the head,
  a second `<html>` or `<body>` lending attributes, tables, sections, rows and
  column groups, the line feed after `<pre>`, `<listing>` and `<textarea>`,
  and ignored end tags of elements closed at once. Markup the parser would
  rearrange (a block, heading, list item, `<hr>`, `<xmp>` or `<plaintext>`
  closing an open paragraph, nested headings, list items, links, buttons and
  `nobr`, ruby annotations closing elements, text or elements moved out of a
  table, implied table parts, an end tag that is not the current element's)
  or read otherwise (foreign content, forms, selects, templates, frames,
  scripts, a `meta` naming an encoding, names it would split or read as text,
  a NUL outside the body) returns `None`.
- `tests/markitai_tests.rs` (added): merging as upstream, including what is
  never merged, text read after a merge (math spans, faithful HTML), template
  contents, an unchanged tree after converting it twice, the scripting option
  and `Element::attr`. Each expected string is htmd 0.5.5's output for the
  same input. The `TreeWriter` tests compare its trees with
  `Html::parse_document`'s for the same markup: the rules above case by case,
  what it leaves to the parser, and 10,000 generated trees written as markup
  (more than 85% built) and 10,000 random part sequences (more than 20% built),
  every built tree equal to the parsed one.

## Differences from upstream

- Attributes come in name order, as scraper stores them. Faithful-mode HTML
  writes them in that order; where two attributes set the same value (an
  image's `href` and `src`, both its link) the later in name order wins
  rather than the later in source order; and similar adjacent elements merge
  when their attribute sets are equal in any order (upstream: the same order).
  Markitai's cleaned markup writes every element's attributes in one fixed
  order, `href` before `src`, so its output is unaffected.
- html5ever 0.39 reconstructs active formatting elements before an `<svg>` or
  `<math>` start tag, as the HTML standard requires and 0.38 did not, so a
  formatting element closed implicitly just before foreign content wraps it.
- `TreeWriter` is an addition; nothing upstream calls it. A tree it builds has
  no parse errors recorded in `Html::errors`, which this crate does not read.
- An image's alt text and title are one line: `<img alt="a\nb">` is `![a b](…)`
  (upstream: `![a\nb](…)`, whose text a line-break repair of the Markdown
  later cut at the first line) and `title=""` writes no title (upstream:
  `![](a.png "")`).
- `<b><strong>x</strong></b>` is `**x**` and `<i><em>x</em></i>` is `*x*`
  (upstream: `****x****` and `**x**`, which read as an empty span between
  literal asterisks and as bold). A block element between the two ends the
  search, so the inner element keeps its markers there; `<b><i>x</i></b>` is
  `***x***` as before.

## Verification

A program linking this crate and htmd 0.5.5 from crates.io compared both on
the same inputs with Markitai's handlers, default options, faithful mode, two
other option sets and scripting disabled. On 3,788 real pages (the reference
repository's HTML, the HTML corpora under `.local`, every twentieth page of
the Rust documentation) the outputs were identical except faithful mode,
where all 7,162 differences were attribute order. Of 200,000 generated
documents with attributes in name order, every difference was faithful-mode
attribute order on the same tree or a document with `<svg>`/`<math>` whose
trees differ, and renaming those two tags removed all of them. With shuffled
attributes, every difference outside faithful mode disappeared once the
attributes were sorted.

The upstream suite, run in an isolated copy with default features, passes
(96 tests, one expectation updated for attribute order) with the 7 added
tests; each of 13 mutations of the port fails an added test. `cargo clippy
--all-targets` reports nothing, as for upstream.

For `TreeWriter`, the same isolated copy passes the suite with 13 Markitai
tests; 36 of 37 mutations of its rules fail a test, and the one that does not
(ignoring an open `select` before `<input>`/`<hr>`) cannot change a result,
since the writer leaves every `<select>` to the parser. On Markitai's HTML
corpora (2,049 pages, EML and MSG files of the html-single-parse round) every
tree it built was compared with the parsed markup in a test build and was
equal; it left 54 conversions to the parser (52 with inline SVG, 2 with a
block inside a paragraph). `cargo clippy --all-targets` and `cargo clippy --lib
--no-default-features` report nothing.

The comparison with htmd 0.5.5 above predates the image-text and nested
emphasis changes: since them the two differ on an alt text or title that spans
lines, an empty title, and an emphasis element nested in one of its own kind.
In the isolated copy (it sits under another workspace, so a `[workspace]` table
is appended to its manifest) the suite passes with 114 tests, 112 before the two
added ones, and the three turndown expectations above updated.

`rustfmt` 1.9 with default settings leaves the modified files unchanged and
would join one call onto a line in three unmodified upstream files
(`caption.rs`, `tbody.rs`, `thead.rs`); no setting reproduces upstream's
formatting exactly, so they are kept as published and there is no
`rustfmt.toml`. The workspace's `cargo fmt --all` does not reach this
directory.

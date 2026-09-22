# defuddle port audits

Dated records of comparing `markitai.webextract` against upstream
[defuddle](https://github.com/kepano/defuddle). What the port tracks today is
in
[`PORT_MANIFEST.md`](../packages/markitai/src/markitai/webextract/PORT_MANIFEST.md);
these are the occasions that produced it.

## 2026-08-25 — resync to upstream 0.19.3

The corpus moved to release 0.19.3
(`a332b4d5d539066ddfe19fc4ef6f1b6ffaf914b8`), and all 208 fixtures passed the
parity quality tests afterwards.

The first resync attempt surfaced six porting gaps, all since ported:

- aria-hidden overlay articles
- CodeMirror code blocks
- mid-article image rows
- Substack note permalinks
- SVG external-CSS fallbacks
- inline related-stories blocks

The full-corpus benchmark surfaced a further set, also ported: SVG
CSS-variable, `light-dark()` and Tailwind colour resolution; noscript
lazy-image resolution; lightbox image dedup; line-number gutter handling; and
LaTeX-image-service conversion.

## 2026-09-13 — build-and-run comparison against a local defuddle

Compared against defuddle `a0984a817518565cedd0f89423c85cfff9e8ba45`. Passing
quality floors turned out not to imply matching behavior: this run found
collapsed Obsidian callouts, hidden-content retry selection, arbitrary Tailwind
variants and Hacker News comment permalinks all diverging. Each is now covered
by a focused regression test. The corpus pin was left unchanged.

The same audit adopted a dedicated known-HTML conversion path, and moved cheap
selector checks ahead of subtree protection scans.

Differences left standing, by choice: CLI startup cost, no stdin HTML input,
and the Markdown that markitai's richer site-specific extractors produce.

Comparison scripts are in
[`scripts/benchmarks/`](../scripts/benchmarks/README.md).

## 2026-09-22 — resync to upstream 0.19.4

The corpus moved to release 0.19.4
(`d4c4bad0dc96ca31b2383328e38061cc1490db47`, 27 commits past the previous
pin), prompted by the watch workflow's issue #41. 209 fixtures; the
benchmark mean went from 96.01 to 96.20, with a local build of upstream used
as the oracle for each change.

Ported from the upstream diff:

- code fences grow past the longest backtick run instead of escaping
  backticks (#359)
- `<sub>`/`<sup>` keep their tags and hug their neighbours (#379); the port
  had been flattening them to bare text, merging `2021<sub>5ya</sub>` into
  `20215ya`
- a date inside a labeled row (`Date:`, `Published:` …) is no longer stripped
  from the row, which left an orphaned label
- declarative shadow roots: `closed` mode, the legacy `shadowroot`
  attribute, and nested roots (hoisted inside-out, depth-bounded)
- `<template>` fragments and SVG SMIL elements are removed, and a content
  root that is itself unsafe is emptied rather than serialized

Found missing while comparing, and ported in the same pass: arXiv
`span.ltx_note_outer` removal (upstream since March), which took
`issues--144-arxiv-footnote-marks` from 74 to 100.

Not ported, recorded under "Known divergences" in the manifest: presentation
MathML preserved through arXiv equation tables, and YouTube default-caption
selection. Not applicable: linkedom heading-case and `compareDocumentPosition`
workarounds, the C2 wiki extractor.

Two footnote fixtures upstream recognises and markitai does not
(`footnotes--br-separated-named-anchors`, `footnotes--labeled-section-ol`)
dropped about 1.5 points each: their `<sup>` markers now render as tags
instead of bare digits. The footnote patterns themselves were already a gap.


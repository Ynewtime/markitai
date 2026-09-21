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

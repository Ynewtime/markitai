# Tailwind CSS colour palette

The native HTML reader keeps an inline SVG chart as a drawing. Without the
page's style sheet, it resolves colours the page gave through Tailwind CSS
classes or variables (`fill-orange-500`, `var(--color-amber-600)`) with
Tailwind CSS's default colour palette. `crates/markitai-core/src/formats/html/svg.rs`
embeds that palette's 22 colour scales (`slate` … `rose`, shades 50–950: 242
hex values) and the `black`, `white`, `transparent` and `current` keywords,
taken from Tailwind CSS 3.4.19 (the npm package `tailwindcss`,
`src/public/colors.js`; version 4 writes its palette in `oklch()`).

LICENSE is the package's unmodified MIT license (Copyright (c) Tailwind Labs,
Inc.). provenance.json records the npm tarball and its integrity, the source
file's digest and URL at the `v3.4.19` tag, the license digest, and a digest of
the embedded values (one `<palette> <shade> <hex>` line per value, in
Tailwind's order). `scripts/tailwind_attribution.py` checks the license digest
and recomputes that digest from `svg.rs` before any package carries this
notice, so a changed palette cannot ship under it.

Changes: only these values are used, written as a Rust table and match arms;
`inherit` and the renamed-colour warnings (`lightBlue`, `warmGray`, `trueGray`,
`coolGray`, `blueGray`) are not. No Tailwind CSS code is included. defuddle and
Markitai 1.3.0, whose behaviour this reader follows, carry the same palette;
the values here were checked against the Tailwind CSS source itself.

This notice does not complete a legal review of the dependency bundle.

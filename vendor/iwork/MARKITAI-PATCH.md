# Markitai's pinned iWork reader

This directory contains the published `iwork` 0.2.1 package's library and
command source, integration tests, examples, `README.md`, `LICENSE` (MIT,
unchanged) and its package manifest. `UPSTREAM.json` records the original
archive checksum, upstream commit and every copied file's original checksum.
No Cargo registry source was modified. The workspace coordinator owns the path
patch and lockfile.

## Why

The Numbers reader uses iwork 0.2.1. Every published iwork release (0.2.1 to
0.4.0) requires zip 2, whose `extract()` carries RUSTSEC-2026-0367, 0368 and
0369, fixed only in zip 9. Markitai never extracts with iwork, but the
advisories stay in the lockfile while any zip 2 is resolved. iwork 0.2.1
reads and writes its packages through APIs zip 9 keeps, so this copy depends
on zip 9 and the workspace resolves a single zip.

## Changes

Each change is marked `markitai` in a comment.

- `Cargo.toml`: zip 9.0.5 with the `deflate-flate2-zlib-rs` backend the
  workspace already uses, instead of zip 2 with `deflate` (which also built
  zopfli and miniz_oxide).
- `tests/fixtures.rs`: zip 9's `ZipFile::name` returns a `Result`; the
  assertion message unwraps it.

Run its own suite in a copy outside the workspace (`cargo test` in the copied
directory); all of it passes against zip 9.

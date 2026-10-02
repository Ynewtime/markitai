# Markitai's pinned PDF object library

This directory contains the published `lopdf` 0.45.0 package (`src`, `tests`,
`examples`, `README.md`, `CHANGELOG.md`, `LICENSE`, `rustfmt.toml`) and its
package manifest. `UPSTREAM.json` records the original archive checksum, the
upstream commit the package names (packaged from a work tree with uncommitted
changes) and every copied file's original checksum. The original MIT license
remains in place. No Cargo registry source was modified. The workspace
coordinator owns the path patch and lockfile.

The first local change, marked `markitai` in its comments, touches one upstream
file and adds three (later changes are described at the end):

- `src/encodings/glyphnames.rs`: `Glyph::from_name` finds a name by binary
  search instead of one `match` arm per glyph name. The `match` had 4,495 arms
  compared in turn and was the largest function in markitai's CLI: 531,880
  bytes of code (symbol-address difference in an unstripped release-profile
  build). The `glyphs!` list, the associated constants, `Debug` and the
  duplicate-code check are unchanged; in test builds the macro also lists
  every name with its code in listing order.
- `src/encodings/glyphnames/lookup.rs` (new): the search. At compile time it
  checks that the sorted names are strictly ascending (so no name appears
  twice) and that their bytes fit 16-bit offsets, and packs them into one byte
  string with a bounds array and a code array, which hold no pointers for a
  position-independent executable to relocate at load.
- `src/encodings/glyphnames/sorted_names.rs` (new, generated): the 4,495 names
  with their codes in ascending byte order.
- `scripts/sorted_glyph_names.py` (new): writes `sorted_names.rs` from the
  `glyphs!` list. After changing the list, run
  `python3 vendor/lopdf/scripts/sorted_glyph_names.py`; `--check` exits 1 when
  the table is stale. A stale or hand-edited table out of order fails to
  compile, and one with a name missing, added or recoded fails the tests.

`from_name` returns what the `match` returned for every input. Tests beside
the change (three, in `lookup.rs`): each of the 4,495 listed names gives its
code and the table holds exactly those names; 251,701 changed names (every
prefix, a byte before or after, one byte changed by one, letters' case
flipped; 8,207 of them are other names) agree with the listing; a few named
cases. Seventeen mutants of the search, packing, order check, `from_name`,
the test listing and the table data are all caught (three at compile time).

`rustfmt.toml` is upstream's. Stable rustfmt warns that three of its options
(`wrap_comments`, `comment_width`, `format_strings`) need nightly and ignores
them; with the rest, `cargo fmt` in this directory changes no file upstream
wrote, and the added Rust files follow it. The workspace's `cargo fmt --all`
does not reach this directory.

The upstream suite was run in isolated copies of the unmodified and the
patched package, prepared the same way for an offline run: the workspace's
`Cargo.lock`; without the dev-dependencies `shellexpand` (used only by two
examples that need the `serde` feature), `criterion` and `wasm-bindgen-test`
(used by nothing packaged), and without the optional dependencies `image`,
`tokio`, `skrifa`, `jiff` and `time` and their features, whose sources are
not in the local registry; with an empty placeholder `assets/example.pdf` so
the unit-test target, which embeds that file, compiles. Unmodified: 304
passed, 33 failed, 3 ignored; patched: 307 passed (the three new tests), the
same 33 failed, 3 ignored. Every failure reads the `assets` directory, which
the published package leaves out.

Measured at `0c75bc0` with `cargo build --release -p markitai-cli` (rustc
1.98.1, macOS 27.0.1 on an 18-core Apple M5 Max with other builds running):
the CLI is 23,103,712 bytes with the registry package and with this copy
unchanged, and 22,641,360 bytes with the change (−462,352, −2.00%; code
−532,036, read-only data +61,440). The output of the 216 Chrome- and
Quartz-printed PDFs of the R41 PDF quality corpus is byte-identical. Two
paired, alternating timings of those files changed the total by −0.17% and
+0.03%, while the unchanged copy against the registry build (identical code)
moved −0.17% and −0.86%: no measurable difference. The corpus's
`/Differences` arrays name only glyphs outside the table (Chrome's `g0`,
`g3`, …), as do those of every other local test PDF; a lopdf built with and
without the change gives identical `extract_text` and per-code
`decode_text` results on those 250 distinct files and on a generated PDF
that names all 4,495 glyphs through `/Differences`.

## Later changes

Also marked `markitai`, these touch three more upstream files and add one
test:

- `src/document.rs` (`get_pages`), `src/parser_aux.rs` (the font encodings of
  `extract_text_chunks_from_page`) and `src/reader.rs` (`load_objects_raw`):
  maps that were collected from iterators are built by inserting their
  entries in turn, and the map from compressed objects to their containers
  becomes a list searched by halves. Collecting a `BTreeMap` first sorts the
  entries with the standard library's stable sort, compiled again for every
  source iterator type: 21,276 bytes of sort code for these four in an
  unstripped release build of markitai's CLI. Their keys arrive ascending and
  distinct (page numbers, another map's font names, the cross-reference
  table's object numbers), so inserting builds the same maps; the filtered
  object-stream map still keeps a repeated id's last object, as collecting
  did. The container list is collected in the table's order, so
  `binary_search_by_key` finds exactly the entries `BTreeMap::get` found.
- `tests/linearized_objstm_test.rs`: one test, with its own small builder,
  loads a file whose two object streams each hold an object no other stream
  holds, one where the cross-reference stream puts it and one where it does
  not, with and without a filter function (the two branches of
  `load_objects_raw`). It passes on this copy before and after the change,
  and catches each of four mutants of the new code (the container rule
  ignored, a neighbouring entry's container, the rule dropped from the filter
  branch, pages numbered from 0).

Measured at `452bf47` with `cargo build --release --locked -p markitai-cli
--bin markitai` (rustc 1.98.1, macOS 27 on an Apple M5 Max): the sort code
for these maps falls to none, and the stripped CLI loses 28,440 bytes of
`__TEXT` sections (33,088 bytes of file, which grows in 16 KiB pages). The
upstream suite, prepared as above: 308 passed (the added test), the same 33
failed, 3 ignored; `cargo clippy --all-targets` reports the same findings as
before (none in the changed code). The CLI's output for 219 PDFs (the 216
Chrome- and Quartz-printed PDFs of the R41 corpus, two trial prints and the
reference's `sample.pdf`) is byte-identical apart from each run's
`markitai_processed` time. Timing the CLI with these and the same round's other size changes on
50 of those PDFs (the largest and a sample; five alternating rounds after a
warm-up) moved the total by −0.22%, against −0.47% for an identical copy of
the old build: no measurable difference.

A further change, also marked `markitai`, touches `src/document.rs` and adds
one test to `tests/decryption.rs`:

- `get_encrypted` (and so `is_encrypted`, authentication and
  `EncryptionState::decode`) reads a trailer `/Encrypt` that holds the
  dictionary itself, as MuPDF and PyMuPDF write it, as well as a reference to
  one. Only a reference was read, so such a file looked unencrypted, was never
  authenticated, and loaded with every string and stream still encrypted;
  markitai then reported an owner-password-only PDF (RC4 40/128, AES-128,
  AES-256) as needing a password. `decrypt_raw` skips and removes the
  dictionary's object only when there is one; a document so decrypted records
  no `/Encrypt` object id, so its incremental save is refused as before.
- `an_encryption_dictionary_in_the_trailer_is_authenticated_and_decrypted`
  moves the dictionary lopdf writes into the trailer: with the empty user
  password the file opens on load; with a user password it stays encrypted
  without one and opens with it; `decrypt` reads the trailer's dictionary.

The upstream suite, prepared as above: 309 passed (the added test), the same
33 failed, 3 ignored.

# Numbers tables

The local reader accepts modern `.numbers` ZIP files and directory packages, decoding their IWA streams in Rust. Both containers use the same allocation checks, semantic decoder and renderer; directory packages are not recompressed into ZIP files. It uses the pinned MIT-licensed `iwork = 0.2.1` library. Conversion does not launch Numbers, Python, a browser or a remote service.

## What is retained

Sheets follow the document's sheet-reference order. Tables follow each sheet's drawable-reference order. Object identifiers distinguish tables with the same name; names are never used to deduplicate content. A sheet becomes a level-one heading and each table becomes a level-two heading. Decoded tables lacking a sheet reference are appended with a warning.

The reader retains Unicode text, blank positions, booleans, saved numeric values, dates and durations. Formula cells use the value saved in the document; they are not recalculated. A warning identifies this limitation. Error cells become `#ERROR!`.

Tables with one header row and no merged cells use Markdown pipe tables. Other tables use HTML, preserving header cells and merged anchors through `rowspan` and `colspan`. Covered positions are not repeated. Overlapping or out-of-bounds merges are errors. Hidden and filtered rows and columns remain in stored order, with a warning; this reader does not reproduce Numbers' current filtered view.

Extractor metadata includes `sheet_count`, `table_count`, `numbers_value_mode: saved-values` and an ordered `numbers_tables` list. Each table records its sheet/name, dimensions, header/footer counts, cached-formula count and merged ranges. Merge row/column coordinates are one-based. This extractor metadata does not add fields to the existing conversion frontmatter contract.

## Values are not a screenshot of Numbers

The adapter does not treat `CellValue::to_text()` as a complete display formatter. It explicitly handles basic decimal places, percentages and currency codes, using decimal128's integer mantissa rather than passing numbers through binary floating point. Fixed precision is supported through twelve places and rounds halfway values away from zero. Percentages multiply the saved value by 100. Currency output uses an available three-letter code, such as `USD 12.50`, rather than guessing a locale-specific symbol. Very large exponents retain an exact scientific representation.

Dates use an ISO-style calendar/time representation without an invented timezone; the stored value does not identify a timezone. Fractional date seconds are not retained. Durations use seconds. Rich-text cell text is retained, but fonts, colors, bullets and inline styling are flattened. Scientific/custom/fraction formats, locale separators, accounting signs, conditional formatting and interactive controls are not reproduced exactly. A formatting warning accompanies such normalized output.

Only tables are extracted. Linked non-table canvas objects, including charts, images, shapes and text boxes, produce an omission warning. Their geometry, captions and image assets are not exported. This is not a canvas renderer. Encrypted documents, older XML/pre-BNC cell storage and packages containing only a nested `Index.zip` are not supported by this reader; damaged or unsupported tables fail rather than silently becoming empty successful output.

## Resource bounds

Checks precede high-level table decoding:

| Resource | Limit |
| --- | --- |
| Input ZIP / expanded ZIP entries / directory file bytes | 128 MiB each aggregate |
| ZIP entries / directory descendants, including directories | 4,096 |
| Directory relative-path components | 8 |
| Individual expanded ZIP entry / directory file / IWA stream | 32 MiB |
| Expanded IWA streams combined | 64 MiB |
| Snappy block | 64 KiB expanded |
| IWA objects | 100,000 |
| Tables / sheets | 1,024 each |
| Rows / columns per table | 100,000 / 1,024 |
| Declared table positions combined | 2,000,000 |
| Individual cell text | 1 MiB |
| Rendered table body | 32 MiB |

An additional conservative decoded-cell estimate charges every declared position for a cell structure plus the document's longest interned/rich-text value, with a 256 MiB ceiling. It can reject a large sparse table containing a long string even when many cells are blank. Referenced table side-list payloads are charged again for each model and limited to 64 MiB. These are allocation guards, not a promise that total process RSS stays below one of those numbers.

The ZIP reader limits actual bytes read and verifies entry data. It rejects duplicate names, path traversal, non-regular entries and encrypted packages. Nothing is extracted to the filesystem. IWA headers, Snappy expanded lengths, object identities and table dimensions are checked before `iwork` allocates table rows and columns. Multiple table-info objects may not point to the same table model: otherwise the decoder would multiply allocations outside the dimension budget.

Directory packages are inventoried before their content is read. Entries are sorted by their UTF-8 relative paths. Every descendant, including directories and `.DS_Store`, counts toward the node limit; the package root itself does not. Every regular file counts toward the byte limits, even ignored Finder metadata. Internal symbolic links, sockets, FIFOs, devices and invalid path names are rejected. A `.DS_Store` regular file is ignored only after these checks. The caller applies its existing symlink policy to the supplied package path; allowing an outer alias never permits links inside the package.

Files are opened with bounded reads and checked against their inventoried type, length and modification metadata. Unix also checks device/inode/change-time identity and opens with `O_NOFOLLOW | O_NONBLOCK`. Directory and file metadata are checked again before decoding. Ordinary replacement or mutation during reading causes an error; these checks do not provide an atomic snapshot or a filesystem sandbox against adversarial concurrent ancestor replacement. Packages should remain unchanged while being converted.

Output size is checked after each heading and cell, so rejection may temporarily exceed the body limit by one escaped cell, at most about 6 MiB. The adapter avoids scanning every merge for every cell by using a bounded coverage bitmap.

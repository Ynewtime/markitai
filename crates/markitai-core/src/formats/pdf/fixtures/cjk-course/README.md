# Authored CJK layout fixtures

Small one-page PDFs, each reproducing one way a Chinese form printed by iText
was misread (the page reader dropped its first page, set lines in inline code
and fenced blocks, made fields and sentences headings, split its table and
printed U+FFFF). They are set as an invented community course enrolment form:
only authored operators and invented text, no document, account, person or
organisation. The fonts are not embedded; Type0 fonts named after Microsoft
YaHei and SimSun with ToUnicode CMaps, described as their producers describe
them. Run `python3 create-fixtures.py` here to regenerate their deterministic
bytes; `author-fixtures.json` records the SHA-256 and size of each.

- `nul-separated.pdf`: content operators separated by NUL bytes (white space
  in PDF), which the content parser stopped at.
- `segmented-header-table.pdf`: a header row between two rules drawn as one
  segment per column, rows without rules below, a course name wrapped over
  two or three lines beside its row's centred values.
- `form-grid.pdf`: `label：value` fields set apart on 11 pt rows over 10 pt
  body text, under section titles on coloured bands.
- `long-sentences.pdf`: bold 11 pt sentences ending in `。`, running on past
  `，`, one of them with an address followed by bold text.
- `fixed-pitch-cjk.pdf`: text in a face flagged FixedPitch whose widths
  differ (Microsoft YaHei as iText writes it) and in a face whose `/W`
  lists only its 500-unit Latin glyphs, its ideographs left to `/DW 1000`
  (SimSun as Word writes it).
- `notdef-ffff.pdf`: a template Form drawing 32 `.notdef` glyphs, which the
  ToUnicode CMap maps to `<ffff>`.
- `cjk-enumerators.pdf`: items numbered `1．` and `1）`, a wrapped item, and
  paragraphs after items at the same indent.
- `double-spaced.pdf`: clauses at 10.4 pt on 20 pt leading, justified, the
  first line indented, one clause's text hanging past its number `1.1`,
  under a centred `总则`.
- `short-last-lines.pdf`: paragraphs set without spacing, each first line
  running to the column's edge and the last stopping short after `。`, then
  a step numbered `2、` standing alone.
- `symbol-fonts.pdf`: Word's Symbol and Wingdings glyphs as non-embedded
  TrueType fonts mapping each byte to U+F000 plus the byte: a 26 pt Wingdings
  2 pointing hand before a section's title, contents entries after Wingdings
  diamonds, a SymbolMT minus, and a note's title in a face of its own a
  little larger than the body text, right above its paragraph.
- `heading-over-rule.pdf`: a heading right above a table whose top rule is
  drawn a segment per column under the heading's text.
- `wrapped-title.pdf`: a 20 pt title running past `，`, wrapped between two
  ideographs, over a borderless two-row table whose cells stand far apart
  (the page goes to the page reader).

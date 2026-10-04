# Native markup readers

The RST, Org, and TeX readers operate on decoded source text in Rust. They have
no external runtime or parser dependency. They do not execute source code,
expand user macros, follow include directives, read referenced images, or
start an interpreter. Image references remain links; these readers do not
claim to extract image assets from adjacent files.

The implementation is in `crates/markitai-core/src/formats/markup.rs`.
Block boundaries, indentation, balanced groups, and literal contexts determine
how text is read. Code and mathematics bypass prose formatting. Unsupported
directives and environments retain their source and produce warnings. Parsing
recursion is bounded at 64 levels; content beyond the limit is preserved.

## reStructuredText

Implemented constructs include section underlines and overlines, inline
literals, named and embedded hyperlinks, text substitutions, numbered
footnotes, `code`/`code-block`/`sourcecode` directives, indented literal blocks,
`math` directives and inline math roles, images/figures, admonitions, and
rectangular grid/simple tables. Code directive options are consumed before
the literal body; generated fences are longer than any backtick run inside
the body. Lists, emphasis, and field lists retain their readable source form.

The blocks the readers build (quotes, admonitions, tables, footnote definitions,
fences, displayed formulas, image directives, captions and, in TeX, lists) are
separated from the text around them by blank lines in all three readers, since a
table row after a quote line is a lazy quote continuation in CommonMark and a
line after a table is one more row. Source lines of one paragraph, and the lines
of an RST or Org list, keep their line breaks, and runs of blank lines collapse
to one. A definition list (a term at the margin with its definition indented on the
next line) becomes the term in bold followed by its definition read as RST;
lines opening list items, fields, options, line blocks or quotations are never
taken for a term.

Heading styles receive levels in order of appearance, capped at six. An
overline style and an underline-only style are distinct, following RST's
section structure. Link destinations escape characters that would otherwise
break a Markdown destination. Figure captions retain their prose. Grid cell
continuations become line breaks within Markdown table cells.

`include`, raw content, and unknown directives remain fenced RST with a
warning. Table spans and inconsistent boundaries also retain their original
table rather than silently dropping cells. Anonymous/indirect references,
image substitutions, full role resolution, complex definition/option lists,
and arbitrary nested directive semantics are not yet implemented. Simple
tables use character positions; terminal display-width alignment involving
wide characters or tabs is not a complete RST layout implementation.

## Org

Implemented constructs include starred headings, title/author/date metadata,
described and bare links, image links, inline verbatim/code, bold/italic/
underline/strikethrough, named footnotes, tables, source/example blocks,
quotes, verse, Markdown/HTML export blocks, and TeX math delimiters. Code
markers require appropriate boundaries, so ordinary equations and filenames
are not interpreted as formatting. Source blocks retain their literal
contents, including percent signs and heading markers. Unclosed blocks are
closed in the Markdown output and reported through a warning.

`#+AUTHOR` and `#+DATE` are kept in the metadata (`author`, `published`) and
also written as `**Author:**` and `**Date:**` lines where the source has them,
because the frontmatter of a local file holds only the title. Consecutive `: `
example lines are one fenced block.

Tables with a header separator become Markdown tables with headers. Tables
without headers receive an empty Markdown header so their first data row
remains data. Unknown block types and keywords remain visible with a warning.
Comment blocks and comment lines are omitted as comments.

No Babel evaluation, table formulas, agenda state computation, transclusion,
property inheritance, or TODO workflow interpretation is performed. Drawer
contents remain readable source. Complex table separator layouts and
multiline inline constructs require additional compatibility work. Org file
links are not resolved relative to a project or copied into the output.

## TeX

The reader isolates a document body when `document` delimiters exist and
extracts title/author metadata from the preamble. It supports nested balanced
formatting arguments, section commands, common text styles, escaped symbols,
comments, inline verbatim, verbatim/listing/minted blocks, nested lists,
description labels, links, image references, captions, quotes, and ordinary
rectangular tabular environments. Dollar and bracket math keep their TeX
contents; equation/alignment environments become display mathematics. A list,
formula, fence, table or other environment is a block with a blank line on each
side, except that a list inside a list item follows its item closely.

The scanner distinguishes prose from math and literal contexts: a percent
sign inside verbatim text remains content, braces inside a comment do not
close a group, and escaped ampersands do not split table cells. Code fences
and spans adapt to embedded backticks. Unbalanced input keeps available
content and reports a warning.

This is a text extraction reader, not a TeX engine. It does not execute
macros, evaluate conditionals, compute counters, resolve citations, access
included files, or reproduce page layout. Reference labels and citation keys
stay visible with warnings. Unknown commands preserve their argument text;
unknown environments and tables with row/column spans preserve their source.
Advanced package syntax, custom environments, macro-defined document
structure, complete accent normalization, and TeX's whitespace/typesetting
rules are not fully implemented. Alignment conversion does not promise the
full semantics of every AMS environment.

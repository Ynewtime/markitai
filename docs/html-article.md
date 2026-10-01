# HTML article selection

Full HTML extraction selects a coherent content region before rendering it to
Markdown. This policy is separate from the fragment renderer used by EPUB,
Office documents and email: a book's table of contents is useful content.

The selector accumulates visible text and paragraph scores once per DOM node.
Hidden elements and structurally identified page controls do not contribute.
Asides and recognized footnote sections are neutral for selection scoring, so
long external definitions cannot displace a short explicit article. This does
not delete asides inside the selected article or on pages with no article.
It considers `main`, `role=main`, `article`, `itemprop=articleBody` and a bounded
set of established content container names such as `entry-content`,
`post-content` and `blog-post-content`. One content container is selected when
it accounts for at least three fifths of the surrounding region's score.
Substantial introductory or concluding text outside that container keeps the
wider region selected. Multiple sibling articles retain their common main or
body region, which protects portfolios and indexes.

Two explicit structures have more precise boundaries: a GitHub-style README
article with `markdown-body`, `entry-content` and `itemprop=text`, and the
complete `data-testid=issue-viewer-container`. Individual `markdown-body`
comments are not treated as an entire page. Issue discussion content remains;
known metadata sidebars, sticky header duplicates and comment action controls
are excluded.

Page chrome exclusion uses container roles or complete class/ID tokens for
table-of-contents widgets, related/recommended post widgets, subscription
forms, CTA sections, sharing controls and cookie banners. Bare words inside
paragraphs, class substrings and link density do not trigger removal. A generic
Related Posts/Stories heading requires at least two structurally identifiable
linked cards and no independent prose, table, quotation or note content in the
surrounding block. A heading by itself is insufficient.

## Furniture beside the article body

Names only weigh the choice of region; a second pass reads what stands beside
the body from how much text each block holds and how it is made. The body is
the innermost element that holds three fifths of the region's running text
(words outside links and headings; CJK characters count as words, as in the
reference's word count). It must read as an article: at least 70 words in at
least two runs of 14 words or more. A region without such a body, a GitHub
README and a complete issue keep every block. Then, on the way from the region's
root to the body:

- before the body, a block that is only links or at most two words (a banner, a
  back link, a breadcrumb trail) goes, unless it holds a heading, a picture, a
  video or a date: the title, its lead image, a lede and a byline stay;
- after the body, the sibling blocks go when together they hold under fifty
  words and under a third of the body's: a subscribe box, a call to action,
  related-post cards, an author bio, share counters, a category list,
  previous and next links;
- a block with code, math, a recognized note, a section headed as notes,
  references, sources or an appendix, or a data table (a grid that says more
  than a row of navigation links) always stays; a layout table's cell or row is
  left out like any block, while fragments keep their furniture;
- a short label of words above the first `h1` in the same container (at most
  four words, no digits, no sentence punctuation: a category, a kicker) goes,
  while a date stays;
- inside the body, a block after the last text, code or picture that is only a
  row of links with at most a label (two or more links; tags, related posts) goes. A
  paragraph or a list is text however short, a heading seals what follows it
  (a labelled "See also" or "External links" list stays), and text after the row
  makes it part of the article.

After rendering, a heading of a full page with no content before the next
heading of its level or a higher one (usually every paragraph under it was
hidden or left out) is dropped, except one that labels notes or sources, whose
entries may have moved to the end as footnotes. A heading whose text is a link
to a fragment of the same page, or an anchor glyph beside the text, is written
as plain text; a list item with nothing left to show (an icon, a share button)
is not written as an empty bullet; an element with `role=tooltip` is a control's
label, not text. A note's definition beside the body is still written at the end,
and a reference inside a block that is left out creates no footnote.

Known limits: a forum thread whose first post holds most of the text, followed
by under fifty words of replies, loses those replies; an index page whose
introduction is wrapped apart from a list of plain links can lose the list; a
separate conclusion of under fifty words after a long body is treated as furniture.
Text-pattern clutter (bylines, read times, social counters, quoted email
replies) is not removed, and neither are inline comments.

Explicit footnote/endnote/bibliography containers are protected. The parent
footnote pass resolves referenced definitions against the full document even
when the selected article is narrower. Chrome references do not create notes.
These rules preserve an article about navigation, ordinary comment bodies,
reading lists and unlabelled image galleries. They intentionally leave some
ambiguous material that the reference implementation removes using text and
link-density heuristics, including trailing link lists under a heading, inline
featured comments and gallery-like related tiles without a semantic label
inside the article body (see the furniture rules above for what stands beside it).

The design addresses concrete extra content observed in the retained round-five
HTML corpus: `related--inline-related-stories-block`,
`scoring--related-posts-byline`, `general--trailing-cta-newsletter`, and
`general--github.com-issue-56`. The repository fixture
`github-repo--panniantong-agent-reach` provides the README structure. New authored
unit cases reduce these structures to their relevant evidence and include
retention negatives. This document describes policy and fixture provenance;
executed corpus results and any remaining differences belong in validation
records, not an inferred claim of complete extraction parity.

Before selection, [streamed-content recovery](html-stream.md) handles explicit
React completions and eligible script-stripped article chains. The LessWrong
snapshot now retains its substantive body and notes; the
[round-fifteen review](validation/native-performance-round15.md) identifies the
remaining formatting differences. Selection itself does not unhide arbitrary
content based on size. The [historical gap record](validation/html-streaming-gap-round14.md)
preserves the earlier four-word output and the distinction between timeout and
content loss.

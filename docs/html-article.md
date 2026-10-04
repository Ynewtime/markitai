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

Forms and footers weigh the choice as the cleaner writes them
([rendering](html.md#rendering-and-url-handling)): a form's prose and the footer
of an article, section, figure or quotation are page text, so a page an ASP.NET
`<form id="aspnetForm">` wraps whole has its article chosen instead of no
content at all, while entry-box forms (search, sign-in, newsletter, comment
entry), controls and the page's own footer count for nothing.

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
  previous and next links. Three kinds of block stay and are not counted:
  replies of a thread (the same element with the same first class as the post
  they follow, or plain `article`s, both dated, the reply with words of its own
  and not a teaser of another page); the block right after the body when the
  body ends with a sentence of at least five words ending with a colon (an
  index's introduction and its list; "Related:" is a label, not a sentence);
  and that block when it is a conclusion of plain prose (a paragraph-long run,
  with or without a heading, no links, pictures or dates);
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
  (a labelled "See also" or "External links" list stays), text after the row
  makes it part of the article, and so does a sentence ending with a colon
  right before it.

Marks of the page go wherever they stand off the way to the body, on a page
whose body reads as an article (an index keeps the reading times of its cards):

- a block whose whole text is a reading time ("8 min read", "5-minute read",
  "Reading time: 3 minutes") or a count of likes, views, shares or comments
  ("9 Likes", "1.2K views"), unless it is a list item, a table cell, a heading or
  part of a sentence;
- before the body's first paragraph, a breadcrumb trail (a list of two to five
  links, one per item, ending on the page's own name as plain text of at most
  twelve words) and a block that holds nothing but a `time` whose date an earlier
  `time` already gave (the first stays; a date inside a byline stays);
- outside the body, an author's card: the smallest block with a picture, a
  `rel=author` or `itemprop=author` link and a paragraph-long sentence, in at
  most eighty words, without a heading or a date. A byline, a dek beside a
  byline and a card inside the body stay;
- inside the body, a featured comment: a block whose class or id names a top,
  featured, hot, best or pinned comment (`top-comment`, `hotComment`,
  `featured_comments`), with the article's own prose both before and after it.
  A page of such comments, a featured comment ending the article and ordinary
  comment blocks stay.

In a web mail thread saved from the browser, each message body (Gmail's
`a3s`) leaves out what the client folds away: the quoted history
(`gmail_quote`, its "On … wrote:" line `gmail_attr`, a quote right after that
line, Apple Mail's `blockquote type=cite`, Outlook's `divRplyFwdMsg` header with
the rule above it and the message after it), Gmail's trimmed content (`adL`)
and its controls (`role=button`). Quotes outside such a body, an article about
email and every mail fragment keep their quotes.

After rendering, a heading of a full page with no content before the next
heading of its level or a higher one (usually every paragraph under it was
hidden or left out) is dropped, except one that labels notes or sources, whose
entries may have moved to the end as footnotes. A heading whose text is a link
to a fragment of the same page, or an anchor glyph beside the text, is written
as plain text; a list item with nothing left to show (an icon, a share button)
is not written as an empty bullet; an element with `role=tooltip` is a control's
label, not text, and so is one with a `popover` attribute (the browser's style
sheet hides it until a script opens it; GitHub's tooltips are such spans). A
note's definition beside the body is still written at the end, and a reference
inside a block that is left out creates no footnote.

In a GitHub issue (`data-testid=issue-viewer-container`) the interface text the
viewer writes among the discussion is left out: Primer tooltips (`Copy link`,
`Issue body actions`, by their `popover` attribute or `prc-TooltipV2-Tooltip`
class), the signed-out banner (`SignedOutBanner-module__signedOutBanner`: "Sign
up for free to join this conversation on GitHub. Already have an account? Sign
in to comment"), links that send a signed-out reader to `/login?return_to=` or
`/signup?return_to=` (`New issue`), and an element whose whole text is one of
the labels `Copy link`, `Issue body actions` or `Reactions are currently
unavailable`. The state, labels, opener, date and comments around them stay; the
same words inside a sentence of a comment are text.

Known limits: replies after a dominant first post stay only when they repeat its
markup and carry dates, so undated or differently classed short replies are
still left out; an index list stays only after an introduction of five or more
words ending with a colon, and a conclusion only when it follows the body
directly without links or pictures. A reading time or counter inside a byline
line ("Oct 18, 2019 · 8 min read"), a list of post facts ("March 13, 2026 / 4 min
/ Share"), marks in other languages, and dates repeated as text without `time`
stay. Of web mail, only Gmail's message body is recognized; a plain-text "On …
wrote:" line that Gmail shows above its trimmed-content control stays, and so do
the recipients line and quoted replies on mailing-list archives.

Explicit footnote/endnote/bibliography containers are protected. The parent
footnote pass resolves referenced definitions against the full document even
when the selected article is narrower. Chrome references do not create notes.
These rules preserve an article about navigation, ordinary comment bodies,
reading lists and unlabelled image galleries. They intentionally leave some
ambiguous material that the reference implementation removes using text and
link-density heuristics, including trailing link lists under a heading and
gallery-like related tiles without a semantic label inside the article body
(see the furniture rules above for what stands beside it).

Before selection, [streamed-content recovery](html-stream.md) handles explicit
React completions and eligible script-stripped article chains. Selection does
not unhide arbitrary content based on size.

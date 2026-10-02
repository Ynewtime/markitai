# Native HTML extraction

The Rust HTML reader parses a DOM with `scraper`, selects an article candidate,
writes an allowlisted representation of it as the tree `htmd` renders to Markdown,
and parses the page only once (see [single parse](#single-parse)); what its
passes ask of an element is computed once (see [facts read once](#facts-read-once)).
It does not run JavaScript, fetch images, follow links, start a browser, or call
the Python implementation. The same reader serves local HTML and fetched HTML;
fetching itself belongs to the separate HTTP layer. Bytes become text by a BOM,
the `<meta>` charset prescan or UTF-8 (see
[text encodings](formats.md#text-encodings)); a fetched page's HTTP charset
comes first.

Full-page article selection and structural widget removal are described in
[article boundaries](html-article.md). That policy is disabled for document
fragments so EPUB/Office/email content retains its own tables of contents.
[Technical code blocks](html-code.md) describes highlighter normalization,
literal content, language labels and resource limits.

## Metadata

Title selection follows the reference metadata layer: JSON-LD `headline`, then
JSON-LD `name`, document `<title>`, and Open Graph title. Twitter title is an
additional fallback. JSON-LD arrays and nested `@graph` objects are inspected;
malformed JSON is ignored. Scripts remain excluded from rendered content.

Repeated site prefixes and suffixes are removed when the site name is known.
A truncated title can be replaced by a longer matching document or Open Graph
title. Without site metadata, a ` | Site` suffix is removed only when a headline
or first heading matches the preceding text. Ordinary hyphenated subtitles stay
intact. Pathological titles are capped at 300 Unicode characters, preferably at
a word boundary, followed by an ellipsis.

A JSON-LD `headline` is not a title when it names nothing the first heading,
the document title and the Open Graph title say while the heading and one of the
titles agree with each other (Wikipedia writes the article's short description,
`memory-safe programming language without garbage collection`, there); the next
layer is used. A page with no heading to confirm the titles keeps its headline.

The author is the `author` meta tag, else the JSON-LD author, else the
`article:author` meta tag, the first of them that is a name: a web address (an
`article:author` profile link, a JSON-LD name that is a URL) is never written as
the author. The JSON-LD author is a string, an object with a `name`, an
object that points (`@id`) at another object of the page's graph, or an array of
them; an array that has `Person` objects keeps only those, and several names are
joined with `, `. Publication metadata precedes JSON-LD `datePublished`, then a
date-shaped `<time datetime>`. Description and site values come from meta tags,
read from either the `property` or the `name` attribute (a page that writes both,
or an Open Graph value under `name`, is read either way). A page whose meta tags
name no site has the JSON-LD `publisher` name (a string or an object's `name`) as
its `site`, as defuddle reads it, which is not used to clean the title; a Substack
page with neither is `Substack`, as the reference's Substack reader has it.
Page canonical URLs are retained; a homepage canonical on a deeper source URL is
discarded. The local-format dispatcher must preserve the reader's title so
normal output can use it.

## Rendering and URL handling

Full HTML documents first run bounded [streamed-content recovery](html-stream.md).
Recognized React transport instructions are applied to existing DOM nodes without
executing JavaScript. Sparse script-stripped snapshots use a separate, explicitly
documented article heuristic; ordinary hidden descendants remain hidden.

- `<q>` retains quotation marks and nested inline markup. Source newlines remain
  soft Markdown line breaks outside code; structural block elements establish
  paragraph boundaries. This follows the source renderer's line behavior, which
  can differ from a browser's visual whitespace collapse.
- Tables with a header use the reference's compact spelling: `| a | b |` rows and
  one `---` per column, without width padding. Tables kept as HTML are unchanged.
- A table with header cells is written to htmd as a regular grid, since htmd
  keeps only the first row's `th` cells as the header and only `td` cells after
  it: one header row (the first `thead` row, a first row of only `th` cells, or
  empty cells when only rows have header cells), `td` body rows, each span's
  other positions as empty cells, empty rows dropped and the caption as a
  paragraph above. A table without header cells is data when it has at least
  two rows and two columns of short cells (at most 40 words and one paragraph,
  no nested table, heading, list, quote or rule), fewer than two thirds of them
  empty, and no `presentation` role; its first non-empty row is the header.
  Other tables lay out the page and are written as their cells' content, so a
  data table inside one is still a table. Columns empty in every row are left
  out. Markdown cannot nest tables: a table inside a table's cell is written as
  text in that cell.
- A link that shows nothing (no text and no kept image) is left out instead of
  being written as `[](url)`. A link wrapped around blocks (a card's cover,
  heading and summary) cannot be one Markdown link: it is written as its
  content, with the link on its first heading, else on its first block with text
  and no block inside.
- Preformatted code keeps indentation and empty lines. A `<code>` element with
  inline `white-space: pre` styling is treated as a code block. Chemical and
  mathematical `<sub>`/`<sup>` text retains those tags.
- Valid absolute HTTP, HTTPS, mail and telephone URLs retain their spelling.
  URL validation does not add a slash to a hostname-only link or change percent
  escapes. Literal angle brackets and double quotes are percent-encoded before
  Markdown emission. Relative URLs resolve against a supplied page URL. A full
  page read from a local file resolves relative links against its absolute
  `<base href>`, else its canonical link (only root-relative links when that is
  the home page, whose directory is unknown); images keep their relative paths,
  which a page saved with its files points at those files. Without such an
  address, and in fragments, relative destinations stay as written. A
  scheme-relative address (`//host/path`) takes `https:` when no page URL is
  given, since Markdown would read it as a local path.
- An image is its largest `srcset` candidate (`data-srcset` first): the widest
  `w` descriptor, else the densest `x` one (no descriptor counts as `1x`), the
  first of equals, where `src` is often a thumbnail. Candidates are read as the
  HTML standard reads them, so a comma inside an address (a CDN's
  `w_728,c_limit`) does not split it; only a comma that ends an address or
  follows a descriptor does, and a candidate with an unknown descriptor is
  dropped. An image is never downgraded: the largest candidate replaces the
  plain address (the lazy `data-src` or `data-original`, else `src`) only when
  that address is missing or an inline `data:` placeholder, or is itself a
  candidate of the set (a thumbnail kept as `src`, compared as written), or, when
  the set lists widths, the candidate is at least the `width` attribute (in
  pixels; a percentage or no attribute keeps the address), or, when the set lists
  densities only, the candidate is denser than the address's implicit `1x`.
  Otherwise the address stays (BBC's `src` is 2560 pixels wide and its set ends at
  1920). An inline `data:` candidate is never chosen. `<picture><source>`
  elements are not read, since their types and media queries choose between
  formats and crops, not sizes. The lines of an alt text or title are joined with
  a space, so a figure's description is not cut at its first line, and an empty
  title is left out (`![](a.webp "")` becomes `![](a.webp)`).
- Emphasis the page nests with itself (`<b><strong>x</strong></b>`,
  `<em><i>x</i></em>`) is written once (`**x**`, not `****x****`); differently
  nested emphasis keeps both (`***x***`).
- Text drawn for screen readers only (`visually-hidden`, `sr-only`,
  `screen-reader-text`, a CSS module's `…VisuallyHidden`) is kept as the label of
  what follows it, set apart by a space when the markup puts none between
  (`Published 1 hour ago`, `By Katya Adler`, not `Published1 hour ago`). Other
  adjacent inline elements stay joined, since a style sheet's gap between them
  cannot be told from a word's inner markup.
- An image whose source is inline `data:` content keeps its alt text and
  position as the reference's `![alt](data:<type>...)` placeholder; the payload
  is never retained. `data:text/html`, `data:image/svg+xml` and
  `data:text/javascript` images are removed, and links never accept `data:`.
- URL attributes with control characters or unsafe schemes are removed. Event
  handlers and arbitrary styles are not emitted. Script, form, frame, template,
  hidden and navigation content is excluded. A hidden ancestor also disqualifies
  an article candidate. Content nested deeper than 256 levels (unclosed legacy
  tags such as `<font>` reach that depth) is kept as plain text without its
  formatting, with a warning.

Inline visibility uses parsed CSS declarations rather than substring matching.
Only the actual `display` and `visibility` property names affect this check;
custom properties such as `--footer-display: none` do not hide content. Quoted
strings, comments and nested function/block values cannot introduce declarations.
Recognized keyword declarations honor order and `!important`; `display: none`
and `visibility: hidden` or `collapse` hide the element. The HTML `hidden`
attribute still applies independently. This does not compute CSS variables,
escaped property names, external stylesheets or the browser's complete cascade.
The correction follows the r1 corpus audit; that historical artifact and its
two recorded conversion errors remain unchanged.

### Single parse

The cleaned page is the markup of its allowlisted elements, attributes and text
(`<name a="v">`, `</name>`, escaped text). Instead of writing that markup and
parsing it again, the cleaner hands each part to `htmd`'s tree writer, which
builds the tree html5ever would build from it and applies the parser's rules
where markup written from a tree needs them: the `html`, `head` and `body`
elements it implies, head elements written before the body (an email's
`<title>`, `<meta>`, `<style>`), the line feed dropped after `<pre>`,
`<listing>` or `<textarea>`, whitespace in table rows, ignored end tags of
elements closed at once (`</param>`), and the content of raw-text elements read
as text. Where the parser would rearrange the markup (a block closing an open
paragraph, nested headings, list items or links, text moved out of a table) or
read it as foreign content (inline SVG or MathML left as markup), the markup is
written and parsed as before. Either way the tree, and so the Markdown, is the
one the cleaned markup parses into; tests compare the two trees for every
conversion they make.

On the HTML corpora of the R44 round (the 209 defuddle fixtures and 32
captured pages, 681 + 840 + 10 further pages, 264 EML and one MSG file) every
output is byte-identical to parsing the markup, and 54 of the conversions parse
it (52 with inline SVG, 2 with a block inside a paragraph). The same change
reads attributes by comparing names instead of interning them (scraper's
`Element::attr` takes a global lock and allocates for every name outside
html5ever's static set) and reads an element's class and id once when
matching it against a list of names. Converting the 1,740 pages in process
(release profile, macOS arm64, medians of 30 passes; two copies of one binary
differed by 0.03%): 1,389 ms CPU before, 890 ms with the attribute lookup,
802 ms with class and id read once, 739 ms with the single parse; the 265
email files 167, 134, 134 and 132 ms. Whole-directory CLI runs (`-j 1`, `-o`,
publication included): CPU −16% and wall −12% for the pages, CPU −4% and wall
−4% for the email files (A/A spread under 0.5%). The tree writer and its
fallback add about 37 KB of code (the CLI grows by 49,632 bytes).

### Facts read once

The choice of the content region, the reading of its furniture, footnote
recovery and the cleaner all read the elements of a full page and asked them
the same questions again (is it hidden, page chrome, a note, a note's
context), some of them for every ancestor of an element. Each answer is now
kept the first time it is computed, in a table with a cell for every node of
the page's tree (ego_tree numbers a tree's nodes by their place in its
vector), so an element is classified once however often it is asked about.
Footnote recovery reads what it asks of every element (inside a literal
container, in scope, in a note's context) in one walk in document order, each
element after its ancestors; one walk over the region finds its links around
blocks, reference candidates and inline notes, and one walk the saved page's
base and canonical addresses. The page's metadata and the site readers' tests
(Steam event data, Substack, X, Hacker News) take the elements they look for
from one walk over the page's nodes, in the order and with the selection
`Html::select` uses, instead of about a dozen selector walks. Class tests read
an element's classes once and without interning each name (as scraper's
`classes` does); a copy header longer than any language name is not read
further; word counts read ASCII text a byte at a time; the page is searched for
shadow-root templates with the regex engine's literal search; a table of
contents' heading targets are read only for a list of in-page links; and a
link whose text is neither a number nor a note sign is not collected as a
footnote reference.

On the same 2,049 inputs every output is byte-identical. Converting the 1,740
pages in process (release profile, macOS arm64, medians of 30 passes; two
copies of one binary differed by under 0.4%): 717 ms CPU before, 538 ms after
(−25%); the 265 email files 122 and 103 ms (−16%). Whole-directory CLI runs
(`-j 1`, `-o`, publication included): CPU −6.4% and wall −4.6% for the pages,
CPU −3.0% and wall −3.2% for the email files (A/A spread 0.1–0.8%). The CLI
shrinks by 33,008 bytes (`__text` −12,660 bytes).

## Content region and site readers

A full page is reduced to one content region. A named article region (the
content names defuddle uses as entry points, among them `article-content`,
`js-article-content`, `entry-content`, `post-body`, `role="article"`, and the
plain `article` and `article-text`) is chosen over its page when it holds at
least three fifths of the page's text; subtrees whose class or id words name
page furniture (recommended, related, share, newsletter, subscribe,
disclaimer, toc, breadcrumb, sidebar, footer, nocontent ...) do not count as
page text for that comparison. One article beside teaser cards (articles
titled by a link to another page) is chosen the same way, the cards counting
as furniture; several full articles keep their page. Names only weigh the
choice: nothing is removed for its name. GitHub's Primer page sidebar and
MediaWiki's section edit links, tagline, redirect note and skip links are page
chrome.

An in-page table of contents is left out of a full page: an outermost list of
at least three links that all point to headings of the page, with no other
text than numbering, together with wrappers that hold only a short title for
it and a pair of rules framing it. The headings carry that structure, and the
fragment links would not resolve in Markdown. Fragments (books, email) keep
their tables of contents.

On a full page, framework classes that hide an element are honored as defuddle
reads them: `hidden` or `invisible`, also behind a variant (`md:hidden`, a site's
own `not-machine:hidden`), or a CSS-module `isHidden-…` name, unless a responsive
class shows it again (`hidden md:block`). Arbitrary variants (`[&_.x]:hidden`)
target other elements, and math keeps its hidden accessible copies. Fragments
come without the site's style sheet and keep such elements.
A GitHub issue keeps its complete discussion, where defuddle keeps only the
opening post: the thread is usually what an issue link is converted for.

On a full page an image smaller than 33px on either axis by what it declares
(attributes, inline style, an SVG viewBox, a `1x` srcset URL's width) is a
spacer, pixel or icon and is not written; an emoji image keeps its character,
percentages are not pixels, equation images stay, and fragments keep all
images.

Three sites have their own reader. A Substack note page (by host, note
permalink container or CDN assets, without a rendered article body) is its
main note and attached image. An X post page (an x.com or twitter.com status
or Article URL, or, for a page saved without an address, X's own column markup
or media host) is its main post: text with links, photos (`name=orig` when the
page names the format the photo was uploaded in, `jpg` or `png`; otherwise only
raised to `large`, because the server-rendered page always asks for WebP and
`orig` in a guessed format was a 404 for a 2014 photo), video posters (and a link to the file when the page has one, not
a browser-local `blob:` handle), and the quoted post as a block quote headed
`**Name @handle** · date` (`YYYY-MM-DD` from the quote's timestamp or id, which
does not shift with the reader's time zone, else the text the page shows),
without avatars, player controls, counters, the
`sr-only` heading that repeats the text, or timelines. Three markups are read:
`data-testid` tags, the `data-tweet-id` generation, and the server-rendered page
of late 2026 that has neither (an `article` holding `[data-engagement-action]`
buttons, text in `div[dir=auto]`, the date as the text of its `/status/<id>`
link, a quoted post as a nested `article` in `div[role=link][data-href]`).
Following the reference's thread policy, the author's own consecutive posts
right after the main one continue it, each after a `---` rule; the first reply
from anyone else ends the thread, and replies from other accounts are left out.
Posts of the thread that come before the requested one are not included. An
Article page (`/article/<id>`) is read as an ordinary document, because the
`article` elements around it are cards of other Articles.
The frontmatter uses the reference's fields: `title: Post by @handle on X`,
`author: @handle`, `site: X (Twitter)`, `published: YYYY-MM-DD` (from the page's
timestamp, else the post id, which carries its creation time in UTC, so a page
rendered in another language or time zone keeps the date), `description` (the
first 200 characters) and `content_profile: social_post`, which makes `--llm`
keep the body verbatim and use the model for metadata only. The display name is
added as `author_name`, which the reference does not write. A
Hacker News story or comment page is the story link and text (or the comment
with its author and date) followed by its comments as blockquotes nested by
reply level, read from each comment's indent; a story list is numbered with
each story's source, points, submitter and comment link. A Gmail thread saved
from the browser is read as a page, without the quoted history and trimmed
content that Gmail folds away in each message
([mail threads](html-article.md#furniture-beside-the-article-body)).
Against defuddle's 209 expected outputs, words extra to the expected body fell
from 3,555 to 2,658 with none lost ([R43](validation/html-quality-round43.md)),
then to 2,192 with lost words 228 → 210 and table-row differences 71 → 24
([R44](validation/html-quality-round44.md)); all seven semantic contracts of
the reference's captured pages hold. Reading the blocks beside the article body
([furniture rules](html-article.md#furniture-beside-the-article-body)) lowers the
extra words further, from 2,143 to 1,655 across 25 pages, with lost words
unchanged at 210 in every file and the seven contracts still holding (release
build, macOS arm64, the `html-quality-r44` driver on the 209 + 32 inputs; the
remainder is mostly the GitHub discussion, the Wikipedia infobox, and titles,
subtitles and dates kept on purpose). Marks of the page (reading times,
counters, a breadcrumb trail, a repeated date, an author's card), featured
comments set into the body and a Gmail thread's quoted history take it from
1,655 to 1,484 across 9 pages, with lost words unchanged at 210 in every file,
table-row differences at 17 and the seven contracts holding; replies of a
thread, a list the body introduces and a short conclusion now stay after the
body (same driver and build profile).

## Reading sites

Some reading sites keep the article among site chrome, hide it until a script
runs, or hold it as JSON that the page renders itself. A small reader per site
(`formats/html/sites.rs` and its modules) takes the article from what the page
serves and builds a clean article page, which the generic reader then converts,
so tables, code, images, links and formulas come out as on every other page. A
reader never replaces text the generic reader would keep: it copies the
article's own markup and leaves out only the chrome around it.

A reader runs only for a page that is the site's. A fetched page is known by its
address. A page saved from a browser has none, so it is known by the address it
names for itself, in this order of appearance: the `<!-- saved from url=… -->`
comment browsers write, the canonical link, `og:url` and `<base href>`; a saved
page that names no address at all is known by the site's own markers
(`#js-initialData`, `#js_content`), and another site's canonical link overrides
those markers. When a reader finds none of its markers, or what it builds
converts to nothing, the generic reader reads the original page. A built page
carries a marker and is never read a second time.

| Site | Pages | Read from | Output |
|---|---|---|---|
| Zhihu (`zhihu.com`, `zhuanlan.zhihu.com`) | answer, column article, question | the JSON in `<script id="js-initialData">` (`initialState.entities.answers`, `.articles`, `.questions`: the HTML `content`, author, `voteupCount`, `createdTime`), else the rendered markup (`.RichContent-inner`, `.Post-RichText`) | the question or article title, a line `author · headline · 赞同 N · 发布于 date`, the body with its lazy images (`data-original`, never the `<noscript>` twin), TeX formulas as `$…$`, code with its language; the answer the address names, not the other answers of the page (a question address gives the question and the answers the page carries, the most upvoted first) |
| WeChat (`mp.weixin.qq.com`) | article | `#js_content`, which the page serves with inline `visibility:hidden`, with `#activity-name`, `#js_name`, `<meta name="author">` and the time (`#publish_time`, else the script's `ct`) | title, a line `author · account · time`, the body with its images (`data-src`), code blocks joined line by line; `account` is added to the frontmatter |
| cnblogs (`cnblogs.com`) | post | `#cnblogs_post_body`, `#cb_post_title_url`, `.postDesc`, `#post-date` | title, author, time and the body, without the blog's header, navigation and footer; the time is read from the page, not from the JSON-LD that writes `+` as `&#x2B;` |
| Jianshu (`jianshu.com`) | note | the page's `article`; author and first publication time from `__NEXT_DATA__` | the body with its images (`data-original-src`), without the editor's default `image` caption |
| OSCHINA (`oschina.net`) | blog post | `.blog-content .editor` with `h1.blog-content-title` and `.blog-content-info` (the page is rendered by scripts, so this is the local browser's result or a saved page) | title, a line `author · date` and the body, without the AI summary box, the advertisement, tags, comments and recommended posts |
| Bilibili (`bilibili.com`) | opus and column pages | `.opus-module-content` with `.opus-module-title__text` and `.opus-module-author__*` of the rendered page | title, `author · time` and the body; images as uploaded (the size directive after `@` is removed); without the table of contents, the sidebar, tags, comments and the site menu |
| 36Kr (`36kr.com`) | article | the `publishTime` the page's own script keeps next to the article | the page's `article:published_time` is the moment the server wrote the page; the frontmatter `published` is the article's own time |

Two changes apply to every page. The redirect page a site wraps its outward
links in (`link.zhihu.com/?target=…`, `link.juejin.cn?target=…`,
`links.jianshu.com/go?to=…`, `www.douban.com/link2/?url=…`, `link.csdn.net`,
`sspai.com/link`, `gitee.com/link`, `www.oschina.net/action/GoToLink` and
InfoQ's `/link`) is replaced by the address it leads to, when that is an http(s)
page; any other query, a target that is not an address and a redirect page that
points at another one stay as they are. An image's address is also read from
`data-original-src` and `data-actualsrc`, the lazy-loading attributes Jianshu and
Zhihu use, after `data-src` and `data-original`.

A built page also turns the non-breaking spaces these sites pad lines with into
spaces (code keeps its own).

Verification and limits. The cnblogs, Jianshu and 36Kr readers were run on
pages fetched while the readers were written, and the WeChat, Bilibili and
OSCHINA readers on the pages the local browser rendered (the survey in
[fetch.md](fetch.md#sites-that-refuse-automated-clients), one or two public
pages per site); the readers' fixtures are small pages written for the tests,
mirroring each site's markup, not copies of pages. The WeChat reader was also
checked against the markup a regular browser receives from the site (the hidden
`#js_content`, the empty `#publish_time`, `ct`, the NBSP in the author), but
that original markup was not fed to the reader; an article made of
image-and-text cards, video or audio has no `#js_content` text and is read by
the generic reader. WeChat marks its own article images with the class
`js_img_placeholder`, so the reader keeps them (a fixture and a live page cover
it).
Zhihu refused every automated client during development, including a browser
that was not logged in (it redirects to a security check that asks for a login),
so the Zhihu reader is tested only on fixtures that follow the JSON structure
documented by projects that read saved Zhihu pages; no live Zhihu page has been
read. MHTML (`.mht`, `.mhtml`) files are not a supported input; save the page as
"Webpage, HTML Only" or "Webpage, Complete".

## Mathematical content

Mathematics is recovered as TeX data before ordinary script removal. Only an
exact `math/tex` script MIME type is recognized, with optional `mode=display`.
JavaScript and other executable script types are still discarded. No script,
TeX engine, browser or external renderer is executed.

The reader recognizes KaTeX, MathJax, MediaWiki and Temml wrappers. It prefers
explicit `data-latex`, `data-math`, `data-entry` or `alttext` values, then TeX
annotations, then presentation MathML structure. A recognized wrapper can read
its hidden assistive MathML without unhiding unrelated content. The wrapper is
replaced once, preventing duplicate visual glyphs, fallback images and accessible
math from appearing together. MathJax v2 preview siblings are discarded only
when their parent also contains a nonempty TeX-source script.
Structural traversal and text collection discard active script/style elements
and XML annotations, including those nested inside MathML text elements.

Structural MathML handles nested rows, fractions, roots, sub/superscripts,
under/overscripts, common accents and Greek/operators, fenced expressions and
tables. Display math uses `$$…$$`, inline math `$…$`; TeX source is preserved
without Markdown backslash escaping. Literal angle brackets in a math source
become TeX comparison commands so annotations cannot emit raw HTML. MathML depth
is bounded at 64 elements, with an explicit conversion error beyond that limit.

Images rendered by TeX image services become math as in the reference: LaTeX is
read from a `latex`, `chl`, `tex`, `eq` or `math` query parameter, else the whole
query, else a percent-escaped path segment, else the alt text, and is used only
when it contains a TeX command (a backslash and two letters). It is display math
when it contains `\begin{` or the image is its paragraph's only child. Images
without such LaTeX (for example an alt of `A, B`) stay images.

TeX source a page leaves in its text for MathJax or KaTeX auto-render to read
(`$x$`, `$$x$$`, `\(x\)`, `\[x\]`) is math too, and keeps its backslashes,
brackets and underscores as written (the Markdown escaping would turn
`$\mathbf{x}_1$` into `$\\mathbf{x}\_1$`). The rule is conservative, since a
dollar sign is usually a price, and reads one text node at a time (a delimiter
and its partner in different elements are not a pair; code is never read):

- `$ … $` and `\( … \)` are inline math on one line (at most 400 bytes). `$`
  follows Pandoc's rule: no space right after the opening dollar or right before
  the closing one, and no digit right after the closing one, so `$5 and $10`,
  `US$5-$10` and `$HOME/$USER` are text. A dollar sign that cannot close ends the
  search, and `\$` is a dollar sign.
- `$$ … $$` and `\[ … \]` are display math and may span lines. A display
  expression that is all its element holds (`<p>$$…$$</p>`) is a block of its
  own, written `$$…$$`; inside a sentence, or in a table cell, it is inline
  `$…$` (in a cell on one line), as the reference spells it. `\(…\)` is written
  `$…$`.
- Between single delimiters and `\[ … \]` the text has to look like math: at
  most three characters (`$x$`, `\(n\)`) or one with a TeX or operator character
  (`\ ^ _ { } = + < > | ( ) [ ] , '`), so `\[options\]` and `\(some words\)` stay
  text.

This is not a complete MathML or TeX renderer. Visual-only MathJax CHTML/SVG
reconstruction, arbitrary MathML layout/variants, multiscripts and exact
tagged-equation layout remain outside this round. Unknown presentation elements
retain their child text/structure where possible; specialized layout can differ.
Existing source code blocks are kept separate from math interpretation.

## Footnote recovery

The reader collects references and definitions from the immutable DOM before
sanitization, then emits numeric Markdown references and a definition section.
Repeated references share one definition. Numbering follows definition order in
the document, including inline notes. Replacing a reference wrapper preserves its
visible leading/trailing whitespace; hidden popover text cannot supply whitespace
to the surrounding article. Ordinary link labels remain links; numeric
markers and explicit footnote semantics establish a reference.

Supported structures include footnote/reference lists, `doc-footnote` and
`doc-noteref` roles, `fn:`/`fn-`/`ftnt` identifiers, named anchors, Word export
backlinks, Google Docs notes, WordPress/Easy Footnotes lists, standalone
`footnote-definition` blocks, inline popovers, Org-style labels and sidenotes,
`aside > ol[start]`, and `data-definition` references to hidden asides. A labeled
footnote section outside the selected article can supply definitions, while
unrelated external content stays outside the article.

Generic numeric anchor targets need a note/reference signal in the section's
role, identifier, class or heading, at least two matched definitions, and coverage
of at least three quarters of the external numeric targets. Concentration alone
is insufficient: ordinary numbered instructions and navigation remain links.
A wrapper containing the article references itself does not qualify. This also
keeps ordinary equation/theorem links from becoming footnotes. Fragment identities
are parsed once for this scan, and sections with fewer than two matching targets
are rejected before coverage checks. Loose
numbered paragraphs require at least two corroborating body superscripts; an
explicit footnote class can identify a single definition. Section delimiters
allow continuation paragraphs/lists to move with the definition; otherwise only
the numbered paragraph moves and following updates remain in the article.
Generic named/ID definitions adopt only adjacent paragraphs, lists, quotes and
code blocks under the same parent. Another target, an explicit identifier/named
anchor, a section boundary, or a related/update label stops adoption. Hidden
continuation blocks are never promoted to visible definition roots. A plain
leading `N.` label is removed only when it exactly matches the resolved source
reference number; years, decimal values and unmatched prose prefixes are kept.

Only resolved definitions with readable content after backlink removal are
removed from their original positions. Missing targets, unreferenced definitions
and backlink-only entries remain visible; they do not create dangling references.
Ordered-list start offsets use checked arithmetic, and output numbers are assigned
after empty definitions have been excluded. Return links and numeric
labels are removed within resolved definitions, while ordinary body links remain.
A duplicate inline sidenote is removed only when its text matches the adjacent
resolved definition. Lists, block quotes, code and later paragraphs receive
Markdown continuation indentation instead of being flattened or detached from
the footnote. This can differ from the source renderer's unindented continuations.

Footnote bodies use the same URL validation, script removal, depth bound and
math/code handling as the article. An explicitly linked hidden definition may
expose its own content root; unrelated hidden descendants remain hidden. The
reference itself and its ancestors must remain visible and outside discarded
navigation/forms. Hidden inline popovers cannot reintroduce their definitions.
Detection excludes code, scripts, MathML and every recognized math wrapper,
including KaTeX and MathJax. Duplicate MathJax previews with a sibling TeX source
are excluded as well; visual superscripts cannot introduce references or move
otherwise unreferenced definitions. Remote URLs with matching fragment names do not
become local references unless they identify the current page or carry explicit
HTMLBook `noteref` semantics.

This is structural recovery, not the entire upstream footnote system. Multiple
named definitions packed into one paragraph with only `<br>` separators,
arbitrary publisher citation groups, separate sidenote columns and recursive
references inside definitions are not fully standardized. Malformed or
ambiguous structures retain their ordinary content where possible. Definition
order and punctuation/line-layout choices can differ from the reference. The
reference local API itself leaves some labeled-list and line-break patterns
unstandardized; corpus strict parity and recovered-content quality remain
separate measurements. Historical audit results are not rewritten.

## Structured BBCode announcements

`data-partnereventstore` JSON can contain the primary article in
`announcement_body.body`. The reader reconstructs the first nonempty readable
announcement from an array or object and uses its headline and timestamp;
`data-groupvanityinfo` can supply the group/author. Invalid JSON falls back to
ordinary HTML extraction. No JavaScript state is evaluated.

The BBCode reader builds a bounded nesting tree. It supports paragraphs, headings,
bold/italic/strike markup, quoted blocks, nested ordered/unordered lists
with implicit `[*]` item endings, links, images, literal code blocks and YouTube
preview identifiers. Raw HTML text retains entity escaping in the final Markdown,
after intermediate HTML decoding; code blocks keep their fenced literal contents.
Body text, headings, link labels and quote attribution share this protection.
Generated images have an empty alt label; their destinations are validated.
Link/image targets use the same URL validation as HTML. Legacy slash/quote escapes are normalized in tag arguments
without rewriting literal code contents. Unknown tags remain visible as text,
and nesting beyond 64 tags fails explicitly.

The reader does not select among announcements by page/event ID, implement every
BBCode dialect, resolve arbitrary embeds or fetch media. Underlined text is
retained without underline styling. Ordered list marker
styles and complex malformed-tag recovery can differ from upstream renderers.
In the r2 corpus, the source **local file API** returns empty Markdown for the
BBCode fixture, while its upstream **URL extraction** expectation contains the
announcement. Rust intentionally recovers that body and metadata; this remains
a strict local-API difference rather than an exact parity pass. URL-expectation
quality must be checked separately. Historical audit reports are unchanged.

The generic reader is not yet the source's complete web extraction pipeline.
It does not replicate its full site resolvers, browser/CSS visibility model,
adaptive content scoring, schema-body fallback, full footnotes or complete math
standardization. Declarative shadow roots (`<template shadowrootmode>` or legacy
`shadowroot`, open or closed, innermost first and at most ten levels) are replaced
by their markup before parsing, as in the reference; other templates stay inert. Static input can therefore retain site chrome, omit
content outside the selected article, or differ in headings, tables and images.
Those differences must be measured rather than inferred from successful parsing.

## Page metadata

For a URL, frontmatter carries the page's user-facing metadata (title, source,
description, author, published date, canonical URL, domain, fetch strategy) and,
as in the reference, `word_count`: each CJK character is one word and other text
counts whitespace-separated runs. The internal reader identity is not written.
The reference's `content_profile` classification comes from its site resolvers,
which this reader does not have, so it is emitted only where this reader has a
resolver of its own: `social_post` for an X post, which `--llm` treats as a body
to keep verbatim. Other pages carry none rather than a guess.

## Callouts and Markdown spelling

As in the reference, Obsidian callouts, GitHub alerts, Bootstrap alerts, callout
asides and Hugo/Docsy admonitions become blockquotes opening with an Obsidian
marker, `> [!type]fold Title`, before hidden and chrome removal: a collapsed
callout body is content, while hidden elements inside it are still removed.
Types, fold markers, titles (defaulting to the capitalized type) and the title
elements the marker replaces follow the reference rules; an admonition title
keeps its source whitespace where the reference would join inline pieces without
a space.

Lists use one space after the marker (`* item`, `1. item`), rules are `---` and
empty quoted lines are `>`, as the reference writes them. Nested list markers stay
`*`, where the reference cycles `*`, `+`, `-` by depth.

## Corpus diagnostic

The [latest full corpus audit](validation/html-corpus.md) records r5: 43/209
strict local-file API matches, 166 output differences and no conversion errors.
Fourteen of 28 footnote fixtures match exactly. All r3 strict passes remain;
the single lost r4 match preserves source whitespace that the reference local
API drops. The audit also verifies repaired continuation ownership in the
Dhammatalks fixture. Successful conversion does not establish full compatibility.

The reference checkout contains 209 HTML files paired with 209 upstream expected
Markdown files under `packages/markitai/tests/defuddle_fixtures`. Its quality test
suite checks nonempty text, title presence, site-chrome phrases and word-count
tolerance. Passing those heuristics does not establish exact output parity.

Run the new diagnostic from the Rust repository after a coordinator release build:

```sh
python3 scripts/audit_html.py \
  --reference /Users/example-user/work/markitai \
  --library target/release/libmarkitai_ffi.dylib \
  --output .local/audits/html-corpus-20260928-r5 \
  --jobs 4
```

Use `.so` or `.dll` on other platforms. The reference virtual environment is used
by default; `--reference-python` can select another installed environment.
`--pattern 'elements--*'` and `--limit 5` make an explicitly labelled subset.
The output directory must be empty and outside the reference checkout.

Each fixture runs through both **local file public APIs** in separate subprocesses.
The harness reuses `audit_formats.py` for worker isolation: explicit configuration,
disabled LLM/OCR/screenshot/alt/description features, disabled cache, an isolated
`MARKITAI_HOME`, and a credential-free child environment. Source files are hashed
before and after each conversion. A frozen copy of the native library, worker
responses/logs, exact diffs and a machine-readable report remain in the audit
directory. The source checkout is never edited.

A strict parity pass requires exact Markdown, metadata except the processing
timestamp, asset names/content hashes, warnings and skip reason. Matching failures
are not passes. `--require-parity` returns a failure exit code if any selected
fixture differs or errors.

Upstream expected-body equality, title equality, word ratios and chrome phrases
are reported separately for both implementations. These are **diagnostics**:
upstream expectations describe URL extraction, while this harness exercises local
file conversion, including the reference's fallback behavior. They do not prove
URL/site-resolver parity, do not count as migrated source tests, and do not turn a
strict output mismatch into a pass. The report records selected/available fixture
counts, source revisions and the exact library hash. It makes no performance claim.

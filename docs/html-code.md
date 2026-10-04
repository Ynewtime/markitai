# HTML code blocks

The native HTML serializer has a separate DOM reader for confirmed code blocks.
It emits only a sanitized `pre`/`code` pair with an optional safe language class.
Syntax token wrappers and links inside the block contribute their literal text;
link destinations, copy buttons, executable elements, editor chrome, and hidden
content do not become code. It never evaluates scripts or renders a browser.

Supported structural families include Chroma/Pygments/Rouge number columns and
inline line numbers, CodeMirror content, React syntax highlighter numbers,
Rehype/Shiki line spans, Mintlify containers, Expressive Code rows, Hexo `br`
rows, and a `code` wrapper around a `pre`. Language information comes from code
attributes, nearby dedicated highlighter containers, or a recognized language
label in an editor header. The reader does not infer a programming language
from code tokens.

The attributes include `data-language`, `data-lang`, `language`, `lang`,
`language-…` and `lang-…` classes, and SyntaxHighlighter's `brush:` class
(`brush: js notranslate`, also `brush:js` and `brush: js; gutter: false`), which
MDN writes its blocks with.

A label above a block that only repeats its language is not written: MDN's
`div.example-header > span.language-name` ("js") before `pre.brush: js`
would otherwise be a stray line before the fence, which already names the
language. Only a `div` or `span` is read, whose own or inner class names a
label (`language`, `lang`, `header`, `label`), whose text is at most two short
words, that is followed (blank text aside) by the code block, and whose text
names the same language as that block (`js`, `JS` and `JavaScript` are one
language; `ts`, `py`, `rs` and `yml` likewise). A heading, a paragraph, or a
label naming another language is the page's text.

Line containers insert a separator only when their structure requires one and
neither their payload nor the intervening source already supplied it. Explicit
newlines, indentation, tabs and blank rows are retained. CodeMirror's empty-line
cursor placeholder is treated as an empty row. Numeric literals are not line
numbers by themselves: a gutter requires an identified class, or a paired flex
row with a distinct nonselectable/aligned number cell.

Normal Markdown cleanup protects complete fenced blocks before applying link,
image, placeholder, footer, heading, and whitespace repairs to prose. Literal
code retains blank lines, indentation, trailing spaces, and example syntax.
Backtick and tilde fences can have different lengths and occur inside list,
footnote, or quote containers; closing delimiters require a matching marker,
sufficient length, and no trailing text. An unclosed fence protects the
remaining input. The document still receives a final newline when absent.

The same confirmed-block classifier protects code from the earlier math and
footnote inference passes. Ordinary inline code and prose wrappers are left to
the normal serializer. A highlight container containing explanatory prose is not
collapsed into a code block. The reader uses the parent's total DOM-depth budget
and writes output only after extraction succeeds.

Wrapper validation indexes the collected code targets by DOM identity once,
so each sibling is checked without scanning the full target list. A rendered
container resolves its shared language attribute and copy-header fallback at
most once, on demand; target-local and ancestor attributes retain their existing
precedence. Missing headers are cached too. Ordinary-node checks inspect the
necessary child positions directly and avoid constructing child vectors or
collecting header text when no copy button exists.

This is structural normalization rather than a browser layout engine. CSS-only
line wrapping, virtualized editor lines absent from the DOM, unrecognized
application-specific code widgets, and language labels without a reliable
container remain outside its guarantees. Separate Lean command/output fragments
are not merged merely because they are adjacent. Existing source and curated
corpus output are inspected as independent evidence: an earlier strict match can
still share a lost-code defect, and preservation of valid blank lines takes
precedence over matching a defective reference rendering.

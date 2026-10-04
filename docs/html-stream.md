# Static streamed HTML recovery

The full HTML reader restores selected React transport containers before article
selection and footnote collection. It moves existing DOM nodes in place; it does
not execute JavaScript, open a browser, fetch resources, or change the public
conversion interface. Fragment readers do not need this page-level recovery.

Two policies are deliberately separate. An explicit completion instruction is
stronger evidence than a stripped snapshot. Neither policy turns arbitrary
hidden content into visible article text.

## Recognized completion instructions

The reader recognizes complete, top-level `$RC("B:0", "S:0")` and
`$RS("S:1", "P:1")` statements in inline executable script elements. Single
quotes, whitespace, multiple statements, and preceding function assignments are
accepted. An identifier must have the corresponding `B:`, `S:`, or `P:` prefix
and one to sixteen hexadecimal digits. Calls inside strings, comments, functions,
conditionals or expressions do not count. Escaped identifier strings, error
arguments, template literals, regular expressions, external scripts, and other
React instruction forms are not interpreted.

Every referenced ID must be unique. The target must be an empty template with
only an ID; the source must be a hidden `div` transport container without an
additional hiding class, inline style or `aria-hidden` attribute. Instructions
and endpoints inside inert `template`, `noscript`, `iframe`, `object` or `embed`
ancestors are excluded. Conflicting source/target mappings are rejected. A
boundary completion additionally requires
a preceding pending `<!--$?-->` and a balanced matching `<!--/$-->`, accounting
for nested fallback boundaries. Only then are the fallback and template replaced
by the source's children. Segment completion replaces the matching `P:*`
template. A missing endpoint, detached target, cyclic placement, malformed range,
or error-marked boundary stays unchanged.

This follows the structural operations documented in React's
[Fizz instruction implementation](https://github.com/facebook/react/blob/v19.1.0/packages/react-dom-bindings/src/server/fizz-instruction-set/ReactDOMFizzInstructionSetShared.js).
Recognizing a static instruction does not evaluate application logic or establish
that a browser would execute the surrounding page identically. Hidden descendants
remain hidden; moving a transport container does not remove their attributes.

## Compatibility policy for script-stripped snapshots

Some saved pages retain transport containers while stripping both the completion
scripts and React boundary comments. The LessWrong corpus snapshot is such a
case: its substantive `postBody` is in `S:5`, connected to the visible page through
`P:5` inside `S:3`, then `B:3` inside `S:1`, then `B:1`. Matching only `B:5` and
`S:5` would miss this article.

The compatibility policy requires all of the following:

- No executable script or React boundary marker remains, and no stream ID is
  duplicated.
- The visible page contains at most 80 text units. One continuous sequence of
  non-CJK letters/digits counts as one unit; each Chinese ideograph, Japanese
  kana, or Hangul syllable counts as one. Punctuation separates sequences.
- Exactly one hidden transport segment contains at least 200 units under article
  semantics, at least four times the visible page's count, and at least three
  paragraphs containing ten units each. Hidden descendants do not contribute.
- Article semantics are `article`, `main`, `role=main`, `itemprop=articleBody`,
  or exact IDs/classes `postBody`, `post-body`, `post-content`, `article-body`,
  `article-content`, `entry-content`, or `story-body` (names are case-insensitive).
  Generic long hidden prose alone is insufficient.
- Each source has exactly one same-suffix `B:*` or `P:*` target. Following the
  targets' hidden transport ancestors must reach the visible document within
  sixteen segments, without a cycle, missing endpoint, ordinary hidden ancestor,
  or ambiguous target. Snapshot endpoints inside `head`, `pre` or `code` are
  also excluded; explicit completion instructions can still restore streamed
  code inside a real `pre` element.

The reader restores only this candidate's ancestor chain, starting with its
innermost segment. It leaves other hidden transport segments and ordinary hidden
descendants alone. Without boundary comments it cannot identify a fallback range,
so it replaces only the template and retains adjacent nodes. This is a bounded
content heuristic, not verified stream completion or the reference reader's
broader largest-hidden-subtree selection. A deliberately hidden article that
imitates this complete transport structure remains an inherent ambiguity of
script-stripped input.

## Bounds

Pages without transport IDs do not copy or parse script text. Recovery considers
at most 200,000 DOM nodes, 512 distinct transport IDs, 2 MiB of
inline executable script text, and 256 instructions. Content traversal is limited
to 128 element levels. Over-limit discovery or an ineligible snapshot preserves
the original DOM. Existing node identities are retained, and no article body is
copied or reparsed for placement. Recovery does not add network or runtime
dependencies.

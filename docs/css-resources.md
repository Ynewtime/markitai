# CSS resource destinations

Asset preparation, file publication and history archiving share the same
destination rewriter. HTML `style` attributes and CSS `style` elements can refer
to the same assets as Markdown links or HTML media attributes. A reference must
follow its own asset when filtering or name collisions change that asset's path.

This is destination rewriting. It does not evaluate a stylesheet, select browser
resources, download imported stylesheets or inline external CSS files.

## Parsing boundaries

The implementation uses the already locked `cssparser` 0.37 tokenizer. CSS escape
sequences, URL tokens, functions, strings and comments follow the
[CSS Syntax parsing model](https://drafts.csswg.org/css-syntax-3/).
Only resource-bearing syntax is rewritten:

- Quoted or unquoted `url()` values, including escaped function names and paths.
- Static quoted `src()` values and the initial resource in `image()`, including
  its optional direction and fallback color.
- The initial URL or string in an
  [`@import` rule](https://drafts.csswg.org/css-cascade-5/#at-import).
- Resource strings and URLs in
  [`image-set()` candidates](https://drafts.csswg.org/css-images-4/#image-set-notation),
  including the `-webkit-image-set()` spelling.

Image-set descriptors support positive numeric `x`, `dppx`, `dpi` and `dpcm`
resolutions and quoted `type()` values. Candidates using computed resolutions
such as `calc()`/`min()`/`max()` remain unchanged. Nested syntax is bounded at
64 levels; deeper content is preserved without inspecting its resource leaves.

Ordinary quoted content, comments and unrelated attributes retain their original
bytes. Markdown fences, indented examples, inline code, frontmatter examples and
HTML pre/code/script bodies remain protected. Supporting styles changes the prior
behavior that treated an entire style element as a literal example: actual CSS
resource positions now participate in relocation. The opening style attribute
on a pre/code/script element can change while that element's body remains literal.
Only an absent, empty or ASCII case-insensitive `text/css` type activates a style
element; whitespace-padded and other type values remain data.

An HTML attribute is entity-decoded before CSS parsing. A style element contains
raw text, so a spelling such as `&amp;` there is CSS text, not an HTML entity.
Ordinary relocation changes only the resource spans; surrounding source spelling
and whitespace are retained. Filtering may remove an image-set candidate and
its list delimiter. HTML escaping is applied to inserted attribute data, while a raw style
element uses the CSS/URL boundary directly.

## Path identity

The map keys are original filesystem paths. CSS escapes and URI percent escapes
are decoded for lookup, while query and fragment suffixes remain URI syntax.
A replacement filename is encoded as file data before insertion, so literal
percent signs, delimiters, quotes, controls or backslashes cannot become URL,
CSS or HTML syntax. An identity mapping leaves the original spelling untouched.

Unknown at-rule preludes and namespace identifiers remain literal. Static URL
tokens in custom properties can be relocated; ordinary custom-property strings
are not assumed to be paths. Variable substitution and runtime-computed resources
are not evaluated. External stylesheet files are copied as assets without
rewriting their own contents.

Each destination is looked up once. If asset A becomes the original name of B,
the newly inserted path is not looked up again. Documents in a history archive
use the mapping for their own source asset tree. RAG and Obsidian profiles expose
the asset directory without repeatedly decoding URI data.

## Filtering and evidence

An empty CSS URL is an invalid resource under
[CSS Values §4.5.2](https://www.w3.org/TR/css-values-4/#urls); it does not request
the containing document. Filtering can therefore retain the URL position using
`url("")`, preserving font/cursor alternatives and the conditions/layer ordering
of imports. Replacing arbitrary URLs with `none` would invalidate some of those
grammars. Namespace URLs are identifiers rather than fetched resources and must
not be relocated. [Round-twelve validation](validation/css-round12.md) records the
exact accepted cases, frozen release and remaining grammar limits.

Filtering image-set removes the affected candidate while retaining other
candidates. If none remain, the image-set becomes an invalid empty URL resource.
Ordinary URL filters discard their query/fragment together with the old path.

CSS references do not independently trigger image-only detection. A URL can
identify a font, stylesheet, cursor or other non-image resource; a CSS token by
itself does not establish an image suitable for model processing.

Publication tests exercise actual base/enhanced files and all three asset path
profiles. History tests exercise independent copied assets and collision
renaming through the CLI. The [round-twelve report](validation/css-round12.md) records the macOS arm64
release-profile and installed-package checks for that snapshot. Two authored browser cases also load copied images and
imported CSS after deleting the live fixtures; this is not general browser or
cross-platform acceptance.

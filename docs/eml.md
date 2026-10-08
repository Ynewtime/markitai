# EML messages and Content-ID images

The native EML reader uses the MIME parser to decode transfer encodings, character
sets and structured headers, then selects a body from the MIME tree. It performs
no network requests and opens no attachment paths. Its output is an owned Markdown
document and byte buffers for attachments; normal asset processing, model analysis
and publication happen afterward.

## Body selection

HTML takes precedence over plain text among eligible message-body parts.
`Content-Disposition: attachment` parts and attached `message/rfc822` messages
cannot become the outer message's body. Unknown declared disposition types are
treated as attachments, as required by RFC 2183 section 2.8, and cannot become
the body either. Within `multipart/related`, its `start`
parameter selects the root by Content-ID; without a usable unique root, the first
member is selected and an invalid explicit `start` produces a warning. Other
related members are resources, not competing body candidates. Multipart ordering
is preserved while searching alternative/mixed body branches.

Actual HTML runs through the existing fragment sanitizer and Markdown renderer.
Plain text stays plain body content. HTML failures do not cause raw, unsanitized
HTML to be emitted as a fallback. Unknown charsets or malformed transfer encoding
are reported when the MIME parser supplies recovered content. MIME parsing remains
tolerant of recoverable mail transport damage; this is not a strict validator for
every RFC grammar rule.

## Layout

Output follows the reference's layout: `# Email Message`, the From/To/Cc/Date/
Subject header block, `## Content` (a bare heading when the body is empty), and an
`## Attachments` section when the message has attachments. Header angle brackets are
escaped and controls are collapsed.

The attachment section lists, in MIME order and joined by blank lines:

- original attachments, including images, as `- [label](.markitai/assets/...) (size)`,
  with the reference's size spelling (`N B`, `N.N KB`, `N.N MB`);
  a valid `image/*` part not shown through a body CID also has an indented image
  preview under that download, so normal image captions, descriptions and model
  analysis remain available. Preparation gives it a separate preview asset;
  compression, size filtering and image deduplication do not alter the download.
  Tiny previews may disappear while the original remains downloadable;
- then each attached `message/rfc822` as `### Attached message: label` followed
  by that message rendered one level deep as a quotation (`> ` lines, `>` for blank
  lines). Inside it, attachments are listed by name only and nothing nests further.

The label is the decoded filename, or `attachment_N` counted from zero when there
is none. Brackets and parentheses in labels become `_`, as the reference does for
image alt text, and backslashes are escaped so a trailing one cannot end the link
text early.

An explicit `Content-Disposition: attachment`, or an unknown declared disposition,
keeps its original download even
when the body shows that part through a Content-ID. The body uses a separate
image preview. This conservative choice preserves the independent attachment
semantics in [RFC 2183](https://www.rfc-editor.org/rfc/rfc2183.html), including
its treatment of unknown disposition types in section 2.8.
An `inline` or unspecified disposition with an actual bound CID image keeps
the existing single body-image asset instead. Unused CIDs and ordinary CID
download links do not qualify; those parts still receive an original download.
Ordinal numbering still counts body-only resources. A message containing only
genuinely inline body images has no `## Attachments` section.

Original attachments, including attached messages, have their own download links;
the reference only shows a name and size for non-image payloads. Content-ID
images bind inside the body (below), where the reference leaves `cid:` references
in place. Only a declared `image/*` part decoded without a transfer-encoding error
can bind as a CID image. A filename alone never grants image binding, and damaged
bytes remain downloads without gaining an attachment preview. An octet-stream
part with a `.png` filename or sniffable PNG bytes is still only a download.
Two valid unbound images with identical bytes keep both attachment labels and
preview references; normal image deduplication analyzes the prepared image once.
The nested quotation does not bind Content-IDs and owns no assets.

## Content-ID binding

Only real HTML image targets are inspected before sanitization. Images in script,
style, comments, code/preformatted examples, ordinary text, or unrelated attributes
are not rewritten. Real `img` sources and supported image candidate attributes use
the common image-reference parser. HTML text containing literal Markdown image
syntax is not treated as an HTML image element.

A reference binds to a MIME part only when:

- its `cid:` scheme is recognized, its percent encoding decodes once, and its ID
  has no whitespace, controls or malformed angle brackets;
- the part belongs to the same nearest `multipart/related` scope as the selected
  body, or both belong to the outer unscoped message;
- the ID identifies one part, that part declares an `image/*` MIME type, and its
  transfer-decoded binary payload is nonempty without a parser encoding error.

Header names and the URI scheme are case-insensitive. Exact Content-ID spelling
wins. For mail-client compatibility, an ASCII case-insensitive match is accepted
only when it identifies one part; case-folded ambiguity never chooses an arbitrary
image. Duplicate exact IDs, missing IDs, non-image parts, and resources in another
related scope remain unresolved and receive warnings. An ordinary `.png` attachment
with an `application/*` MIME type does not satisfy these rules.

Resolved targets point to owned assets before HTML sanitization. Unresolved targets
survive the sanitizer through per-message collision-checked placeholders and are
restored as CID URIs afterward; their labels and surrounding body remain. The
restoration preserves URI percent escapes rather than interpreting them as raw
filesystem filenames. Unresolved CID schemes must never fall through to local
file lookup during image enrichment.

Attachment filenames are decoded by the MIME parser, then reduced to bounded safe
labels prefixed with their ordinal. Slashes, controls and non-ASCII characters
cannot become path traversal. They are not filesystem inputs. Transfer-decoded
bytes remain associated with the part identity, so filename collisions or misleading
extensions cannot redirect CID lookup.

Binary attachments retain the MIME parser's transfer-decoded bytes. For a text-type
attachment, the parser also decodes its declared charset, and the downloaded buffer
is that UTF-8 text; it is not a byte-exact copy of every charset's original payload.
These are decoded attachment downloads, not preservation of the full raw EML file.
Recovered malformed transfer data retains the existing warning behavior.

## Publication and model analysis

Body-image previews use the existing embedded-image pipeline. Filtering and
compression apply to previews and genuinely inline images. Downloadable original
bytes bypass those image settings, even for a tiny or duplicate image. An explicit
attachment used in the body has separate preview and download references; filtering
its preview cannot remove its download. Final content-addressed paths replace
references, so display filename labels remain readable while physical asset names
may differ. Preview rewrites retain URI query/fragment data and leave ordinary
downloads intact, including a shared Markdown reference definition.
Repeated references to one image share one analysis and asset. With LLM image
caption/description options enabled, successful analysis changes enhanced captions
and contributes `ConversionOutput.images` and the published `images.json` entry.
Download-only attachments are not submitted for image analysis merely because
they have an image filename. Actual body-image references may be analyzed. Base
author captions and attachment download links remain available. Profile
changes, such as the visible `assets/` directory in RAG, apply to both references
and metadata paths.

Unresolved images do not create successful analysis records. The reader itself does
not submit images or trust instructions embedded in mail; downstream model calls
remain opt-in and use the shared document budget. See
[image enrichment](image-enrichment.md) for the full failure and accounting policy.

## Boundaries and reference behavior

Input is rejected above 100 MiB before MIME parsing. Parsed messages are limited to
4,096 total parts, 64 levels of multipart/nested-message structure and 128 MiB of
decoded MIME content. These checks bound admitted documents; they do not claim a
streaming parser or a measured peak-memory ceiling. Image decoding has its own
pixel and payload limits downstream. Nested messages are quoted one level and
remain downloadable `.eml` attachments.

The reference Python reader also prefers HTML bodies and uses the email library's
related-root selection. It renders image attachments as data-URI images, which its
image pipeline then saves as `<document>.<NNNN>` assets; native assets use
content-addressed names (an accepted difference, see
[compatibility](compatibility.md)).

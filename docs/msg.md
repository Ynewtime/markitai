# Outlook MSG reader

The MSG adapter reads Outlook compound files directly through `cfb`. It returns
Markdown, metadata, attachment bytes and warnings without Outlook, Python, local
mail configuration, network requests or writes to the source file.

## Output contract

The Markdown header preserves the existing MSG layout: `# Email Message`,
available From/To/Subject fields, then `## Content`. The subject is also the
document title. Date, From, To, Cc and Bcc are retained in the adapter's metadata;
they do not introduce new Markdown header rows. Header values follow the
[EML](eml.md#layout) rules: controls become spaces, whitespace collapses and angle
brackets are escaped. The output layer applies its ordinary local-file
frontmatter policy.

Display recipient properties take precedence. If a display property is absent,
the adapter reads recipient objects and groups their addresses by recipient type.
Sender address falls back to the SMTP property and then the display name.
Submission time, or delivery time when submission time is absent, is converted
from FILETIME to UTC RFC 3339. Microsoft documents the
[recipient type](https://learn.microsoft.com/en-us/office/client-developer/outlook/mapi/pidtagrecipienttype-canonical-property)
and [submission time](https://learn.microsoft.com/en-us/office/client-developer/outlook/mapi/pidtagclientsubmittime-canonical-property)
properties.

Body precedence is Unicode plain text, ANSI plain text, then HTML. ANSI decoding
uses the declared message code page; HTML uses its Internet code page. Supported
code pages cover UTF-8, ASCII, Latin-1, Windows-1250 through Windows-1258, common
Japanese/Chinese/Korean encodings and selected ISO/KOI8 encodings. Unknown or
malformed declared encodings produce a warning before trying UTF-8 and
Windows-1252. Invalid UTF-16 is an error; it is not silently replaced with damaged
characters. HTML uses the shared email fragment renderer, retaining the message
body instead of selecting an article candidate.

The reference's committed `sample.msg` stores its readable body as HTML and has
a submission timestamp that its Markdown does not display. This reader follows
that distinction. Exact fixture parity must be checked against a frozen native
artifact; passing authored tests alone does not establish corpus parity.

## Attachments

By-value attachment data is extracted byte-for-byte and returned as assets.
Names are reduced to safe basenames with an ordinal prefix; embedded directory
names never become output paths. HTML `cid:` image references resolve to
attachment content IDs with the [EML matching rules](eml.md#content-id-binding):
only real image sources are inspected (not links or attributes such as
`data-src`), the URI is percent-decoded once, the exact ID wins, and an ASCII
case-insensitive match must name exactly one attachment. MSG has no MIME related
scope, and an attachment needs no declared image type to bind. Original
attachments receive a Markdown download link, even when the body also shows their
image. Their downloaded bytes bypass preview compression, filtering and image
deduplication. The body can use a separate prepared preview; removing that preview
does not remove the original. Filename labels remain visible while normal
content-addressed publication may use different physical names. Link labels keep
the name as written: controls collapse, and backslashes, brackets and angle
brackets are escaped. A missing or ambiguous content
ID produces a warning and stays a `cid:` image reference.

Only an unambiguous, correctly typed `PidTagAttachmentHidden=true` (0x7FFE,
Boolean), a correctly typed `PidTagAttachFlags` (0x3714, Integer32) containing
`0x00000004` without HTML-invisible `0x00000001`, and an actual HTML CID image use
together qualify for a single inline image asset. That inline asset keeps the
existing image processing behavior and need not preserve its original encoded
bytes. A missing/false Hidden property, missing flags, wrong type/value, conflicting
classification or a CID used only in a link cannot authorize omitting the original
download. Plain-text image-looking Markdown is not HTML CID provenance. The reader
does not infer this classification from filenames or RenderingPosition.

Microsoft defines [Hidden](https://learn.microsoft.com/en-us/office/client-developer/outlook/mapi/pidtagattachmenthidden-canonical-property),
the [AttachFlags body-format bits](https://learn.microsoft.com/en-us/openspecs/exchange_server_protocols/ms-oxcmsg/af8700bc-9d2a-47e4-b107-5ebf4467a418),
and [inline HTML attachment checks](https://learn.microsoft.com/en-us/openspecs/exchange_server_protocols/ms-oxcmail/96ee0694-0902-43d2-97d3-40f1fad626e7).
Requiring Hidden as well is the reader's conservative preservation policy.

Attachment extraction is an additive fidelity improvement over the inspected
reference MSG converter, which did not emit these assets. External-file/web
references, embedded messages and OLE attachment methods are not dereferenced or
executed; each unsupported attachment produces a warning. The supported method
and remaining methods are distinguished by Microsoft's
[attachment-method specification](https://learn.microsoft.com/en-us/office/client-developer/outlook/mapi/pidtagattachmethod-canonical-property).

The reader does not yet decompress RTF-only bodies, recursively export embedded
messages, interpret named calendar/contact properties, or preserve full Outlook
HTML styling. A message without readable plain text or HTML returns its headers
and supported attachments with a warning explaining the missing body. Corrupt
or oversized attachments do not discard readable message text.

## Bounds and malformed inputs

| Resource | Limit |
| --- | --- |
| Input compound file | 256 MiB |
| Total bytes copied from property streams | 128 MiB |
| Individual text/HTML property | 16 MiB |
| Individual binary attachment | 64 MiB |
| Property table | 1 MiB |
| Compound directory entries | 16,384 |
| Recipient or attachment objects | 1,024 of each |

Stream lengths are checked before payload copying, with a second bound while
reading. These adapter bounds do not claim that all upstream CFB bookkeeping
allocations are independently bounded. A foreign compound document lacking the
message property table is rejected. Truncated property records and conflicting
duplicate values are errors; zero padding after complete records is accepted.
The narrow exception is conflicting attachment Hidden/AttachFlags classification:
those fields are marked ambiguous, produce a warning, and retain readable
by-value data as an original download. Other conflicting attachment properties
and all message/recipient property conflicts keep the strict behavior.
Malformed recipient or attachment subobjects produce explicit warnings while
the main message is retained.

The parser follows the top-level message, recipient and attachment storage
structure described in [MS-OXMSG](https://learn.microsoft.com/en-us/openspecs/exchange_server_protocols/ms-oxmsg/621801cb-b617-474c-bce6-69037d73461a).
It reads fixed values from property tables and variable values from their named
streams. It does not trust attachment paths or named property identifiers as
instructions.

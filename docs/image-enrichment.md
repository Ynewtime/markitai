# Image captions and descriptions

Image enrichment runs in Rust through the same model routing and concurrency
runtime as document enhancement. It requires `llm.enabled` and at least one of
`image.alt_enabled` / `image.desc_enabled`. These switches do not enable a model
by themselves. The feature analyzes actual image references and owned binary
assets; it does not search for image-like text inside code or comments.

```sh
markitai report.docx --llm --alt --desc -o out/   # captions and descriptions
markitai photo.jpg --llm -o out/                  # standalone image: description and visible text
```

Captions replace image alt text in the enhanced Markdown. With descriptions
enabled and `-o`, an `images.json` sidecar in the asset directory also records
each image's caption, description and visible text.

## Conversion order

Referenced local, HTTP(S), and image data-URI resources are first localized into
owned assets. Data-URI images are also localized without enrichment whenever an
output directory exists; see [embedded assets](images.md#embedded-assets). Existing embedded assets, including EPUB images, use their
already-owned bytes. The ordinary embedded-image filter, deduplication and
compression policy then runs before document enhancement. URL query parameters
are part of the original target identity: two URLs with different queries are
not confused while downloading. Successful localization replaces only image
references, including reference-style images sharing a definition with an
ordinary link. The latter link keeps its original destination.

After the document model returns, image analysis changes captions in the enhanced
Markdown. The base Markdown retains author captions. Missing or failed images
retain their original references and receive warnings; an unsuccessful analysis
does not create a success-shaped image metadata record. Duplicate image bytes
share analysis within one conversion. The total model usage counts each paid
response once, regardless of how many references use its caption.

A standalone image uses structured analysis directly, avoiding a redundant
ordinary document-model call. Its enhanced document has one title, the retained
image preview, a description, and visible transcribed text. Extracted text uses
a fence longer than any backtick run in the text. Multipage TIFF previews are
sent in page order in one request; the original TIFF download link and every
preview remain, with page markers when `--page-markers` keeps them. The shared
description/transcription covers that complete request; it is not a collection
of independently verified page transcripts.

## Pure mode and failures

The reference's established precedence is retained:

- A local non-image document in `llm.pure` skips image enrichment.
- A pure standalone image returns the title, description and extracted text,
  without a preview or generated frontmatter. Its public `images` list is empty.
- A pure URL may still localize images and replace image captions after the raw
  document-model response.

Standalone no-model and unsupported-backend errors retain their error categories.
Other standalone failures use the document's `llm.on_failure` policy; an in-memory
conversion cannot claim a usable enriched image document when analysis failed.
Embedded-image failures keep the successfully enhanced document and its author
captions, with explicit warnings.

## Prompts, budgets and model usage

The first request asks for a JSON object containing string `caption` and
`description` fields and optional string/null `extracted_text`. A malformed
structured answer falls back to separate caption and description requests, as
long as the document request budget permits. It does not invent extracted text
from that fallback.

The existing prompt file settings are supported: `image_analysis_system/user`,
`image_caption_system/user`, and `image_description_system/user`. Values are file
paths, not inline prompts. Files in `prompts.dir` are also recognized. Image
prompts can use `{source}`, `{language}`, `{document_context}`, and `{content}`;
the two context names use the first 200 characters of enhanced document content.
Template substitution is one pass, so tokens in inserted source content are not
interpreted again. Built-in instructions treat the image and document as data;
custom system files remain an explicit user-controlled instruction surface.

`llm.max_requests_per_document` includes document processing, image analysis,
fallback calls and retries. A zero limit remains unlimited. A conversion-scoped
RAII counter restores the previous scope after nested calls or unwinding; it
never uses environment variables to identify a document. `LlmRuntime` separately
bounds active model HTTP attempts across conversions when supplied by the caller.
Returned token usage from unsuccessful content validation remains in the
aggregate. Requests without provider usage cannot establish token cost or
billing. Analysis responses are priced like every other model response, through
the bounded [price catalog](pricing.md).

Each analysis request has a 100 MiB encoded-image-bytes ceiling and observes
`llm.max_vision_pages_per_document`. Existing image decoding/pixel limits still
apply. The new counter is shared with main document requests; no additional
provider credentials are read from document content.

## Resource boundaries

A conversion admits at most 1,024 distinct image references for localization,
64 MiB per resource and 100 MiB of newly localized bytes in total. Local paths
must resolve inside the source directory and obey `output.allow_symlinks`.
Absolute local image targets are rejected. A failed localization preserves the
original reference and emits a warning.

HTTP image downloads use direct transport, five redirects, a 30-second request
budget and a 10-second connection budget. Each redirect is validated separately;
resolved addresses are pinned for the request. Public-origin and local-file
conversions cannot fetch private image addresses. Explicit localhost/private-IP
source pages may use private image targets. Browser cookies, authorization
headers, ambient proxies and browser sessions are not forwarded to images.
Consequently, authenticated/proxy-only image resources may remain external with
a warning. System hostname resolution occurs before the HTTP request timeout;
this implementation does not claim a separate bounded DNS deadline. The
[EML reader](eml.md) resolves MIME-scoped CID images before enrichment; unresolved
CID URIs are preserved with warnings and never guessed from local filenames. Native
persistent image-analysis caching and reference provider-specific image options
are not implemented by this module.

## Public image records and images.json

A successful non-pure standalone or embedded analysis contributes a public entry:

```json
{
  "asset": "/absolute/published/image.png",
  "alt": "Accessible caption",
  "desc": "Markdown description",
  "text": "Visible text",
  "llm_usage": {"model": {"requests": 1, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0}},
  "created": "2026-09-29T12:00:00+08:00"
}
```

Publication resolves `asset` to the actual content-addressed asset path. An
in-memory result uses the profile's relative asset prefix instead of inventing
a temporary absolute file. URL conversions also return successful image entries;
this extends the reference URL API, which left its public `images` list empty.

With descriptions enabled and output on disk, the publisher merges `images.json`
in the actual asset directory. Its envelope is `version`, `created`, `updated`,
`images`; each image uses `path` instead of `asset`, omits `llm_usage`, and adds
`source`. Upserts match the final path, preserve the existing envelope creation
time, and refresh the update time. Local source identity is the supplied source;
URL records use the enhanced output stem. Alt-only does not create this sidecar.

The publisher serializes cross-process updates with a stable lock and atomically
replaces the bounded sidecar. Legacy `assets.json` / `assets` / `asset` spellings
are accepted. Malformed existing metadata is preserved and reported as an error,
rather than reset as in the reference implementation. Metadata merging does not
claim a transaction spanning every Markdown and asset file.

# Output Profiles

A profile shapes the written output for a specific consumer. It is separate from a [preset](./configuration.md#presets): a preset picks which features run, a profile picks what the files look like.

Without a profile, output is exactly the default. Every transform below runs only when a profile is set.

## Enabling a profile

```bash
# CLI
markitai document.pdf --profile rag -o out/
```

```json
// markitai.json
{ "output": { "profile": "rag" } }
```

```python
# Python API
import markitai

out = markitai.convert("document.pdf", output_dir="out/", profile="rag")
```

## `rag` — retrieval pipelines

Default output keeps images in a hidden `.markitai/assets/` directory, and most ingestors skip hidden paths (LlamaIndex `SimpleDirectoryReader` does by default). The `rag` profile makes the output ingestor-friendly:

- **Visible assets**: images move to `assets/` and Markdown references are rewritten to match.
- **Page markers**: PDF output carries `<!-- page: N -->` comments at real page boundaries, for text extraction, `--ocr` and screenshot-only runs alike.
- **Table checks**: pipe tables whose rows disagree with the header column count are reported as warnings. Nothing is rewritten.

```text
out/
├── document.pdf.md          # refs like ![](assets/document.pdf-0001-10.jpg)
└── assets/
    ├── document.pdf-0001-10.jpg
    └── images.json          # with --llm --desc
```

Page screenshots (`--screenshot`) stay under `.markitai/screenshots/`. They are referenced from HTML comments only and are not part of the corpus.

## `obsidian` — vault imports

- **Visible assets**: the same relocation as `rag`, so an output folder pasted into a vault shows no hidden directory.
- **Wikilinks (optional)**: with `output.wikilinks: true`, local image references become `![[assets/x.png]]`, keeping alt text as the display text.
- **Frontmatter**: markitai already writes standard YAML frontmatter (`title`, `source`, `tags`), which Obsidian reads as Properties.

```bash
markitai note.docx --profile obsidian -o vault/inbox/
markitai note.docx --profile obsidian --config-json '{"output":{"wikilinks":true}}' -o vault/inbox/
```

## `okf` — Open Knowledge Format

Aligns frontmatter with the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) spec (v0.2):

| markitai field | OKF field |
|---|---|
| — | `type: Document` (injected; OKF's only required field) |
| `title` | `title` |
| `description` | `description` |
| `source` | `resource` |
| `tags` | `tags` |
| `markitai_processed` | `generated: {by: markitai/<version>, at: <UTC timestamp>}` |
| everything else | kept under its current name |

Fields without an OKF equivalent (`author`, `site`, `published`, ...) keep their names; the spec requires consumers to accept unknown fields. Asset layout is untouched.

```yaml
---
type: Document
title: Lorem ipsum
resource: sample.pdf
generated:
  by: markitai/1.0.0
  at: '2026-08-25T01:42:13Z'
---
```

## images.json schema (frozen)

With `--llm --desc`, each assets directory gets an `images.json` describing the analyzed images. The schema is frozen at version 1.0.

Top level:

| Field | Type | Description |
|---|---|---|
| `version` | string | Always `"1.0"` |
| `created` | string | ISO 8601 timestamp of the first write (kept on merge) |
| `updated` | string | ISO 8601 timestamp of the last write |
| `images` | array | One entry per analyzed image |

Each entry in `images`:

| Field | Type | Description |
|---|---|---|
| `path` | string | Absolute path of the image file |
| `alt` | string | Short caption, used as alt text |
| `desc` | string | Detailed description |
| `text` | string | Text extracted from the image (may be empty) |
| `created` | string | ISO 8601 timestamp of the analysis |
| `source` | string | Absolute path of the source document |

## Recipe: LlamaIndex ingestion

With the `rag` profile nothing is hidden, so the Markdown and its images both reach the corpus, and the frontmatter arrives pre-parsed:

```python
import markitai
from llama_index.core import SimpleDirectoryReader

out = markitai.convert("report.pdf", output_dir="corpus/", profile="rag")
print(out.frontmatter["title"])       # ready as metadata
print([p.name for p in out.assets])   # images now under corpus/assets/

# The reader sees every file: markdown, images, images.json
documents = SimpleDirectoryReader("corpus/").load_data()
print(len(documents))
```

## Notes

- A profile applies to the files it writes and never rewrites earlier output. Do not mix profiled and unprofiled runs in one directory; re-run the inputs instead.
- Profiles apply to written files only. Stdout mode (no `-o`) is unaffected.
- Batch reports and `--resume` state stay under `.markitai/`, outside the corpus. In nested batches each subdirectory gets its own `assets/`.

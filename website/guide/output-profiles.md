# Output Profiles

Profiles shape the written output for a specific downstream consumer. They are orthogonal to [presets](./configuration.md#presets): a preset picks which features run (LLM, OCR, screenshots, ...), a profile picks what the result looks like.

**Without a profile, output is byte-identical to the default behavior** — every transform runs only when a profile is set.

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

Default output keeps images in a hidden `.markitai/assets/` directory. Most ingestors skip hidden paths (LlamaIndex `SimpleDirectoryReader` does by default), so images silently disappear from the corpus. The `rag` profile makes the output ingestor-friendly:

- **Visible assets**: images move to `assets/` and markdown references are rewritten to match. Emptied hidden directories are removed.
- **Page markers**: PDF outputs carry `<!-- page: N -->` comments. They are rewritten from the converter's own `<!-- Page number: N -->` markers, which every path that splits a document into pages writes at a real page boundary — text extraction, `--ocr`, and screenshot-only alike. Positions are never guessed.
- **Table checks**: the LLM cleaning prompts gain a hard column-consistency constraint, and after writing, pipe tables whose rows disagree with the header column count are reported as warnings (detection only, content is never rewritten).

```text
out/
├── document.pdf.md          # refs like ![](assets/document.pdf-0001-10.jpg)
└── assets/
    ├── document.pdf-0001-10.jpg
    └── images.json          # with --llm --desc
```

::: warning Limitations
Page screenshots (`--screenshot`) stay under `.markitai/screenshots/` — they are referenced from HTML comments only and are not part of the ingestible corpus.

A profile applies to what the run converts, and never rewrites files it did not produce. Adding `--profile` to a directory that was converted without one leaves the earlier output in its original shape; re-run those inputs to convert them under the profile.
:::

## `obsidian` — vault imports

- **Visible assets**: same relocation as `rag`, so pasted output folders work as vault folders without showing a hidden directory.
- **Wikilinks (optional)**: with `output.wikilinks: true`, local image references become `![[assets/x.png]]` (alt text is kept as the display text: `![[assets/x.png|alt]]`).
- **Frontmatter**: markitai already writes standard YAML frontmatter (`title`, `source`, `tags`, ...), which Obsidian reads as Properties — nothing to change.

```bash
markitai note.docx --profile obsidian -o vault/inbox/
markitai note.docx --profile obsidian --config-json '{"output":{"wikilinks":true}}' -o vault/inbox/
```

## `okf` — Open Knowledge Format

Aligns frontmatter with the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) spec (verified against v0.2):

| markitai field | OKF field |
|---|---|
| — | `type: Document` (injected; OKF's only required field) |
| `title` | `title` (same name) |
| `description` | `description` (same name) |
| `source` | `resource` |
| `tags` | `tags` (same name) |
| `markitai_processed` | `generated: {by: markitai/<version>, at: <UTC timestamp>}` |
| everything else | kept under its current name |

Fields without an OKF equivalent (`author`, `site`, `published`, `canonical_url`, `fetch_strategy`, ...) keep their names: the spec states consumers "MUST NOT reject documents with unrecognized fields". Asset layout is untouched.

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

With `--llm --desc`, each assets directory gets an `images.json` describing the analyzed images. The schema is **frozen at version 1.0**; changes require updating the lock test (`tests/unit/test_images_json_schema.py`) and this page together.

Top level:

| Field | Type | Description |
|---|---|---|
| `version` | string | Always `"1.0"` |
| `created` | string | ISO 8601 timestamp of first write (preserved on merge) |
| `updated` | string | ISO 8601 timestamp of last write |
| `images` | array | One entry per analyzed image |

Each entry in `images`:

| Field | Type | Description |
|---|---|---|
| `path` | string | Absolute path of the image file on disk |
| `alt` | string | Short caption (used as alt text) |
| `desc` | string | Detailed description |
| `text` | string | Text extracted from the image (may be empty) |
| `created` | string | ISO 8601 timestamp of the analysis |
| `source` | string | Absolute path of the source document |

## Recipe: LlamaIndex ingestion

`SimpleDirectoryReader` skips hidden paths by default. With the `rag` profile nothing is hidden, so the markdown **and** its images make it into the corpus, and the frontmatter arrives pre-parsed:

```python
import markitai
from llama_index.core import SimpleDirectoryReader

out = markitai.convert("report.pdf", output_dir="corpus/", profile="rag")
print(out.frontmatter["title"])  # parsed YAML frontmatter, ready as metadata
print([p.name for p in out.assets])  # images now under corpus/assets/

# The reader sees every file — markdown, images, images.json
documents = SimpleDirectoryReader("corpus/").load_data()
print(len(documents))
```

## Notes

- Do not mix profiled and unprofiled runs in the same output directory: earlier outputs referencing `.markitai/assets/` are never rewritten retroactively.
- Profiles apply to written files; stdout mode (no `-o`) is unaffected.
- Batch runs keep their JSON reports and `--resume` state under `.markitai/` — conversion bookkeeping, deliberately outside the ingestible corpus. In nested batches each subdirectory gets its own `assets/`.

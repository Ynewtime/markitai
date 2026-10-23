# JSON output

`-f json` (`--format json`, configuration `output.format: "json"`) writes each
converted document as one JSON object with stable, versioned fields, for
scripts and pipelines that should not parse Markdown or front matter.

```sh
markitai report.pdf -f json                    # the document on stdout
markitai report.pdf -f json -o out/            # out/report.pdf.json beside out/report.pdf.md
markitai report.pdf -f json -o out/result.json # exact name: out/result.json and out/result.md
markitai ./docs -f json -o out/                # one JSON per file + out/markitai.index.json
```

- **stdout**: the document replaces the Markdown. Images are saved in the
  stdout image store and linked by `file://` URI, as for Markdown on stdout.
  A skipped input (`image_only`) is still a document, with `status: "skipped"`.
- **`-o DIR`**: the JSON is written beside the Markdown under the same name
  (`report.pdf.v2.json` beside `report.pdf.v2.md` when `rename` picked a new
  name). The Markdown files stay: resume, reports, history and the workbench
  work with them, and the JSON's image paths point at the same
  `.markitai/assets/` files. Under `rename`, an existing JSON also makes the
  name taken; under `skip`, an existing result is kept and no JSON is written.
  A JSON document that cannot be written fails the item.
- **Batches** write one JSON per converted file plus `markitai.index.json`
  in the output folder (below) rather than JSON Lines: each document is the
  same self-contained object as for a single file, and the index lists every
  item, so nothing needs to scan the folder.
- `--json` is a different thing: the run's status envelope on stdout (see
  [CLI](cli.md)). The two combine: `-f json --json -o out/`.

`markitai serve` keeps the same document for each finished job item at
`GET /api/jobs/{job_id}/items/{item_id}/document` (capability
`features.document_json`).

## Stability rules

- `schema_version` is `"1.0"`. Version 1.x documents only **add** keys: no key
  is removed or renamed and no type changes within major version 1. Ignore keys
  you do not know. A breaking change would be version 2.
- **Every key listed below is always present.** A value that does not apply or
  is unknown is `null`; a list with nothing in it is `[]`. A missing key never
  means anything.
- Keys are written in the order below; the front matter in `properties` keeps
  the order the Markdown file writes it.
- Paths are written as the Markdown links them: relative to the JSON file's
  folder with `/` separators, or `file://` URIs for stdout output. `outputs`
  paths are relative to the JSON's folder when they are inside it.

The machine-readable definitions are
[`schema/document-v1.schema.json`](schema/document-v1.schema.json) and
[`schema/index-v1.schema.json`](schema/index-v1.schema.json) (JSON Schema
2020-12; they list every key as required and allow unknown ones). Tests pin the
field set, its order and types against a frozen version 1.0 list.

## Document fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `tool.name`, `tool.version`, `tool.commit` | string | `markitai`, the version, the short build commit (`unknown` outside a checkout) |
| `source.kind` | `"file"` \| `"url"` | |
| `source.path` | string \| null | The file as given (a workbench item: its upload name); null for a URL |
| `source.url` | string \| null | The URL without credentials or secret query values; null for a file |
| `source.name` | string | File name, or the name a URL's output takes |
| `source.type` | string \| null | Lower-case extension the content was read as (a PDF named `.docx` is `pdf`); null for a URL |
| `source.size`, `source.sha256` | integer \| null, string \| null | Of a local file; null for a URL or a `.numbers` package folder |
| `status` | `"completed"` \| `"skipped"` | A failed conversion has no document |
| `skip_reason` | string \| null | `image_only`, `exists` |
| `title` | string \| null | The document title |
| `properties` | object | The front matter of the final document (`title`, `source`, `description`, `tags`, `markitai_processed`, a document's own facts such as `authors`) |
| `markdown.base` | string | The converted Markdown body, without front matter |
| `markdown.llm` | string \| null | The model-enhanced body; null without one |
| `text` | string | Plain text of the final body (enhanced if present): no markup or pictures, links as their text, table cells separated by tabs, blocks by a blank line |
| `pages[]` | list | One entry per page or slide marker the conversion kept internally (PDF, multi-page TIFF, presentations); `[]` for formats without pages, whether or not `--page-markers` shows them |
| `pages[].number` | integer | Page or slide number |
| `pages[].text`, `pages[].markdown` | string | That page's plain text and Markdown (screenshot comments removed) |
| `pages[].images` | list of string | Pictures on that page |
| `pages[].screenshot` | string \| null | That page's screenshot |
| `images[]` | list | Each picture the final body links (page screenshots excluded), once |
| `images[].path`, `images[].alt` | string | Link target and alt text (a model caption when one replaced it) |
| `images[].description`, `images[].text` | string \| null | The model's description and the text it read in the picture; null when not analysed or empty |
| `images[].page` | integer \| null | Page the picture first appears on |
| `tables[]` | list | Each Markdown (pipe) table of the final body: `page` (integer \| null), `header` (list of string), `rows` (list of lists of string) |
| `notices[]` | list | The conversion's warnings, in order |
| `notices[].code` | string | Stable code of the warning's family (table below); `other` for one not classified yet |
| `notices[].level` | `"warning"` \| `"info"` | `info` explains how the output was produced; nothing is missing |
| `notices[].message` | string | The sentence the terminal and reports show |
| `notices[].items` | list of string | The pages, pictures or sheets it names, when known |
| `usage.requests`, `usage.input_tokens`, `usage.output_tokens` | integer | Model usage of the document |
| `usage.cost_usd` | number | Known priced cost |
| `usage.cost_complete` | boolean | Whether every request has a reviewed price ([pricing](pricing.md)) |
| `usage.models[]` | list | Per model: `model`, `requests`, `input_tokens`, `output_tokens`, `cost_usd`, ordered by model |
| `timings.duration_s` | number | Conversion time in seconds (milliseconds precision) |
| `outputs.markdown`, `outputs.llm_markdown` | string \| null | The Markdown files written; null for stdout or when not written |
| `outputs.assets`, `outputs.screenshots` | list of string | Files written for the document |

### Notice codes

Codes are only ever added. A family can gain new sentences; matching on the
code keeps working when wording changes.

| Code | Level | Family |
|---|---|---|
| `pdf.page_without_text` | warning | PDF pages without recoverable text, or a scanned PDF (OCR needed); `items` every page named |
| `pdf.page_plain_text` | info | A PDF page read as plain text |
| `pdf.page` | warning | Other notes about PDF pages |
| `pdf.images_after_text` | info | PDF pictures placed after their page's text |
| `pdf.embedded_image`, `pdf.layout`, `pdf.hidden_text` | warning | PDF picture OCR, layout refinement, hidden-text removal |
| `ocr.no_text`, `ocr.unread`, `ocr.capped`, `ocr.unavailable` | warning | Local OCR found nothing, could not read a script, stopped at its region limit, or could not run |
| `text_only.no_text` | warning | `--text-only` found no text |
| `llm.enhancement_failed`, `llm.no_model_succeeded` | warning | Model processing failed; only the base Markdown was kept |
| `llm.answer_retried` | info | A model answer was not used, and a later answer of the request sent again was (the extra request is paid and counted) |
| `llm.answer_rejected` | warning | A model answer was rejected and the request sent again, without a usable answer |
| `llm.cache_unavailable`, `llm.cache_entry_ignored`, `llm.text_fallback` | warning | Cache and fallback notes |
| `llm.cost_incomplete` | warning | Some requests have no reviewed price |
| `image.analysis_failed` | warning | A picture's analysis failed; `items` the picture |
| `image.not_localized` | warning | A linked picture could not be saved |
| `office.renderer_missing` | warning | Office page screenshots or vision OCR need LibreOffice |
| `office.page` | warning | A note about rendered Office pages |
| `office.screenshot_format`, `office.count_verified` | info | Office rendering notes |
| `spreadsheet.hidden_sheet` | info | A hidden worksheet left out (older builds; hidden sheets are now converted under a marked heading); `items` the sheet |
| `spreadsheet.screenshot_scope`, `spreadsheet.font_color`, `spreadsheet.numbers_formatting` | info | Workbook rendering notes |
| `spreadsheet.rendering`, `spreadsheet.numbers_objects` | warning | Workbook content that could not be rendered or read |
| `word.picture_records` | warning | Word picture records |
| `table.merged_cells` | warning | Merged cells shown by their first cell |
| `table.malformed` | warning | A pipe table with mismatched columns |
| `mail.eml`, `mail.msg` | warning | E-mail decoding notes |
| `browser.screenshot_failed`, `cloudflare.not_used` | warning | Web capture and backend notes |
| `other` | warning | Not classified yet |

## Batch index

`markitai.index.json` lists the batch's items ordered by source. A later run
into the same folder (a resumed batch) updates the items it converted and
keeps the others.

```json
{
  "schema_version": "1.0",
  "tool": {"name": "markitai", "version": "2.0.0-dev", "commit": "3efaa92"},
  "items": [
    {"source": "a.pdf", "status": "completed", "document": "a.pdf.json",
     "markdown": "a.pdf.md", "error": null, "skip_reason": null}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `items[].source` | string | The item as the batch report names it |
| `items[].status` | `completed` \| `skipped` \| `failed` \| `pending` | |
| `items[].document` | string \| null | Its document JSON, relative to the index |
| `items[].markdown` | string \| null | The Markdown the JSON sits beside |
| `items[].error`, `items[].skip_reason` | string \| null | Why it failed or was skipped |

## Example

`markitai report.pdf -f json` on a one-page PDF with a table (abridged):

```json
{
  "schema_version": "1.0",
  "tool": {"name": "markitai", "version": "2.0.0-dev", "commit": "3efaa92"},
  "source": {"kind": "file", "path": "report.pdf", "url": null, "name": "report.pdf",
             "type": "pdf", "size": 50024, "sha256": "d2275bf8…"},
  "status": "completed",
  "skip_reason": null,
  "title": "Language comparison",
  "properties": {"title": "Language comparison", "source": "report.pdf",
                 "markitai_processed": "2026-10-17T22:46:53.263+08:00"},
  "markdown": {"base": "# Language comparison\n\n|Language|Year|\n|---|---|\n|Rust|2015|\n", "llm": null},
  "text": "Language comparison\n\nLanguage\tYear\nRust\t2015",
  "pages": [{"number": 1, "text": "…", "markdown": "…", "images": [], "screenshot": null}],
  "images": [],
  "tables": [{"page": 1, "header": ["Language", "Year"], "rows": [["Rust", "2015"]]}],
  "notices": [],
  "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "cost_complete": true, "models": []},
  "timings": {"duration_s": 0.045},
  "outputs": {"markdown": null, "llm_markdown": null, "assets": [], "screenshots": []}
}
```

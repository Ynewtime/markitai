---
name: markitai-convert
description: "Convert documents (PDF, DOCX, PPTX, XLSX, EPUB, images) and web pages to clean Markdown with the markitai CLI (alias: mkai). Use when a file, folder, or URL needs to become Markdown — including batch runs with resume, scanned-document OCR, webpage screenshots, and LLM-added frontmatter, alt text, or image descriptions."
---

# Convert files and URLs to Markdown with markitai

`markitai` (short alias: `mkai`) converts documents, images, and web pages to Markdown, local-first. Confirm it is available with `markitai -V`; if missing, install with `uv tool install markitai` (full setup and LLM wiring: the `markitai-setup` skill, or <https://markitai.dev>).

## Workflow

1. **Classify the input** — it decides the output contract:
   - Single file or URL, no `-o` → Markdown goes to stdout. `-o out/` writes into a directory; `-o result.md` writes that exact file.
   - Directory or `.urls` list → batch mode; `-o` is required.
2. **Compose the command** from the flag table below. Start minimal; add LLM flags only when enhancement was asked for. LLM features need a configured provider (`markitai-setup`).
3. **Run it.** Non-interactive overrides for agent/CI runs: `--config-json '{"llm":{"concurrency":4}}'` deep-merges over the config file; explicit CLI flags still win. `--dry-run` previews without writing.
4. **Verify the outcome** — the conversion is done only when this holds:
   - Exit 0, and the expected `.md` / `.llm.md` files exist (or stdout carries the Markdown).
   - Exit 10 after a `.urls` batch = partial success: successful URLs were written, the rest failed. Report which URLs failed.
   - Exit 1 on a single-image input means neither `--ocr` nor `--llm` was enabled, so there was nothing to output — add one of them.
   - Interrupted or partly failed batch → re-run the same command with `--resume`: completed files are skipped, failed/interrupted ones retried, new files picked up.

## Flag selection

| Goal | Flags |
|---|---|
| Plain Markdown, no AI | (none) |
| Raw body without frontmatter, for piping | `--pure` |
| LLM format cleanup + frontmatter | `--llm` — writes only `.llm.md`; add `--keep-base` to also keep the base `.md` |
| Image alt text / detailed descriptions | `--llm --alt` / `--llm --desc` (both require `--llm`) |
| Full enhancement bundle | `--preset rich` (LLM+alt+desc+screenshot), `standard` (no screenshot), `minimal` (plain); disable any piece with `--no-alt`, `--no-desc`, `--no-screenshot`, `--no-llm`, `--no-ocr` |
| Text from scanned PDFs / images | `--ocr` |
| Page/slide/webpage screenshots | `--screenshot`; URL-only screenshots without extraction: `--screenshot-only` |
| JS-heavy page extracts poorly | `-s playwright`, or `--llm --screenshot-only` to let the LLM read the page from screenshots |
| Batch a folder | `markitai ./docs -o out/`, filter with `-g "*.pdf"` (repeatable, `!` excludes), recurse depth `--max-depth N`, parallelism `-j N` |

Behavior worth knowing before you run:

- `--pure` silently overrides `--alt`, `--desc`, and `--screenshot`.
- Social posts (e.g. X/Twitter statuses) keep their body verbatim under `--llm`; the LLM only adds frontmatter.
- URL fetching is local-first (static HTTP → Playwright before any remote API). The first remote fallback (Defuddle/Jina/Cloudflare) in a process is disclosed on stderr; private/intranet/credential-bearing URLs never leave the machine. Hard local-only guarantee: `MARKITAI_NO_REMOTE_FETCH=1`.
- A URL that fails, hits a login wall/CAPTCHA, or needs per-site tuning → [references/url-fetching.md](references/url-fetching.md).
- Full flag, subcommand, cache, and `.urls`-format reference → [references/cli.md](references/cli.md).

## Output layout (with `-o out/`)

Output names append `.md` to the full input filename, so `report.pdf` → `report.pdf.md` and never collides with `report.docx`.

```
out/
├── document.pdf.md          # base conversion (skipped under --llm unless --keep-base)
├── document.pdf.llm.md      # LLM-enhanced (--llm)
└── .markitai/
    ├── assets/              # embedded images + images.json descriptions
    ├── screenshots/         # --screenshot output (pages/slides/full-page)
    ├── reports/             # JSON conversion reports (batch/URL runs)
    └── states/              # batch state for --resume
```

## Supported inputs

| Kind | Formats |
|---|---|
| Office | `.docx` `.doc` `.pptx` `.ppt` `.xlsx` `.xls` `.odt` `.ods` `.numbers` |
| Documents | `.pdf` `.epub` `.eml` `.msg` `.ipynb` |
| Text/markup | `.txt` `.md` `.html` `.xml` `.csv` `.tsv` `.rtf` `.rst` `.org` `.tex` |
| Images | `.jpg` `.png` `.webp` `.svg` `.gif` `.bmp` `.tiff` (+ `.heic` `.heif` `.avif` with the `heif` extra) |
| Web | `http://` / `https://` URLs, and `.urls` list files |

Legacy `.doc`/`.ppt` and slide rendering need LibreOffice on Linux; Windows uses COM; macOS falls back to installed MS Office via AppleScript (one-time consent dialog). Legacy `.xls` converts in pure Python — no Office application needed on any platform. Capability gaps show up in `markitai doctor` — hand those to `markitai-setup`.

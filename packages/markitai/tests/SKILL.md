# Markitai Test Manual

> This file contains manual test steps for developers and LLMs.
> For automated tests, run `uv run pytest -n auto`.
> For integration checks (LLM/OCR), ensure config + API keys are ready.

---

## Prerequisites

### Environment Setup

1. Install dependencies:
   ```bash
   cd packages/markitai
   uv sync
   ```

2. Configure API keys (for LLM tests):
   ```bash
   # Create .env file in project root
   DEEPSEEK_API_KEY=sk-xxx
   GEMINI_API_KEY=xxx
   # Or configure in markitai.json
   ```

3. Test fixtures are located in `tests/fixtures/`:
   - `file-example_PDF_500_kB.pdf` - Sample PDF
   - `Free_Test_Data_500KB_PPTX.pptx` - Sample PPTX
   - `file-sample_100kB.doc` - Legacy Word format
   - `file_example_XLS_100.xls` - Legacy Excel format

---

## Basic Conversion Tests

### 1. DOCX Conversion

```bash
# Run
markitai tests/fixtures/file-sample_100kB.doc -o /tmp/markitai-test

# Verify
- [ ] Output file exists: /tmp/markitai-test/file-sample_100kB.doc.md
- [ ] Markdown format is correct
- [ ] No conversion errors in console
```

### 2. PDF Conversion

```bash
# Run
markitai tests/fixtures/file-example_PDF_500_kB.pdf -o /tmp/markitai-test

# Verify
- [ ] Output file exists: /tmp/markitai-test/file-example_PDF_500_kB.pdf.md
- [ ] Images extracted to assets/ directory
- [ ] Image references in markdown are correct
```

### 3. PPTX Conversion

```bash
# Run
markitai tests/fixtures/Free_Test_Data_500KB_PPTX.pptx -o /tmp/markitai-test

# Verify
- [ ] Output file exists: /tmp/markitai-test/Free_Test_Data_500KB_PPTX.pptx.md
- [ ] Slides are separated with clear markers
- [ ] Embedded images are extracted
```

---

## Preset Tests

> Requires valid LLM configuration and API keys.

### 4. Rich Preset

```bash
# Requires API key configured
markitai tests/fixtures/file-example_PDF_500_kB.pdf --preset rich -o /tmp/markitai-test

# Verify
- [ ] Base .md file created
- [ ] LLM .llm.md file created with frontmatter
- [ ] Images have alt text updated
- [ ] images.json contains image descriptions
- [ ] Page screenshots in assets/ (if PDF has multiple pages)
```

### 5. Standard Preset

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --preset standard -o /tmp/markitai-test

# Verify
- [ ] .md and .llm.md files created
- [ ] Image alt text updated
- [ ] images.json created
- [ ] No page screenshots (--screenshot not enabled)
```

### 6. Minimal Preset

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --preset minimal -o /tmp/markitai-test

# Verify
- [ ] Only base .md file created
- [ ] No .llm.md file
- [ ] No images.json
```

### 7. Preset Override

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --preset rich --no-desc -o /tmp/markitai-test

# Verify
- [ ] .llm.md created (from preset)
- [ ] Alt text generated (from preset)
- [ ] No images.json (overridden by --no-desc)
```

---

## LLM Feature Tests

> Requires valid LLM configuration and API keys.

### 8. LLM Clean + Frontmatter

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --llm -o /tmp/markitai-test

# Verify
- [ ] .llm.md file has YAML frontmatter
- [ ] Frontmatter contains: title, source, description, tags, markitai_processed
- [ ] Markdown formatting is improved
```

### 9. Image Alt Text Generation

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --alt -o /tmp/markitai-test

# Verify
- [ ] Images in markdown have meaningful alt text (not empty)
- [ ] Alt text is in appropriate language (zh/en based on content)
```

### 10. Image Description Generation

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --alt --desc -o /tmp/markitai-test

# Verify
- [ ] images.json file created in output directory
- [ ] JSON contains: path, alt, desc, text, model, created
```

---

## OCR Tests

### 11. OCR Mode (Local)

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --ocr -o /tmp/markitai-test

# Verify
- [ ] Text extracted using RapidOCR
- [ ] Chinese/English text recognized correctly
```

### 12. OCR + LLM Vision Mode

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --ocr --llm --screenshot -o /tmp/markitai-test

# Verify
- [ ] Page screenshots generated
- [ ] LLM Vision used for content enhancement
- [ ] Higher quality markdown output compared to OCR alone
```

---

## Batch Processing Tests

### 13. Directory Batch

```bash
# Prepare test directory
mkdir -p /tmp/markitai-batch-input
cp tests/fixtures/*.pdf /tmp/markitai-batch-input/
cp tests/fixtures/*.pptx /tmp/markitai-batch-input/

# Run
markitai /tmp/markitai-batch-input -o /tmp/markitai-batch-output

# Verify
- [ ] All files converted
- [ ] Progress bar displayed
- [ ] Report generated in reports/ directory
- [ ] Report token_status present (estimated/unknown)
```

### 14. Resume Interrupted Batch

```bash
# Start a batch and interrupt with Ctrl+C
markitai /tmp/markitai-batch-input -o /tmp/markitai-batch-output

# Resume
markitai /tmp/markitai-batch-input -o /tmp/markitai-batch-output --resume

# Verify
- [ ] State file .markitai-state.json exists
- [ ] 文件数超过 scan_max_files 时停止扫描并提示
- [ ] Only pending files are processed
- [ ] Completed files are skipped
```

---

## Config Command Tests

### 15. Config List

```bash
markitai config list

# Verify
- [ ] JSON output displayed
- [ ] All config sections present: output, llm, image, ocr, screenshot, prompts, batch, log
- [ ] batch.scan_max_depth and batch.scan_max_files present
```

### 16. Config Path

```bash
markitai config path

# Verify
- [ ] Shows search order
- [ ] Shows currently used config file (if any)
```

### 17. Config Init

```bash
markitai config init --output /tmp/markitai-test/config.json

# Verify
- [ ] Config file created
- [ ] File contains valid JSON
- [ ] All default values present
```

### 18. Config Validate

```bash
markitai config validate /tmp/markitai-test/config.json

# Verify
- [ ] "Configuration is valid!" message shown
```

### 19. Config Get/Set

```bash
markitai config set llm.enabled true
markitai config get llm.enabled

# Verify
- [ ] Set command succeeds
- [ ] Get returns "true"
```

---

## Dry Run Test

### 20. Dry Run Mode

```bash
markitai tests/fixtures/file-example_PDF_500_kB.pdf --dry-run

# Verify
- [ ] Shows "Would convert" message
- [ ] Shows input file, format, and output path
- [ ] No files actually created
```

---

## Error Handling Tests

### 21. Invalid File

```bash
markitai nonexistent.pdf

# Verify
- [ ] Error message: "Path 'nonexistent.pdf' does not exist"
- [ ] Exit code is non-zero
```

### 22. Unsupported Format

```bash
echo "test" > /tmp/test.xyz
markitai /tmp/test.xyz

# Verify
- [ ] Error message about unsupported format
- [ ] Exit code is non-zero
```

---

## Report Verification

After any LLM-enabled conversion, check the report file:

```bash
cat /tmp/markitai-test/reports/*.report.json | jq .

# Verify
- [ ] summary.total_documents matches input count
- [ ] llm_usage.total_cost_usd is present
- [ ] llm_usage.token_status is present
- [ ] files[] contains per-file details
```

---

## Notes

- All tests assume working directory is the markitai package root
- LLM tests require valid API keys configured
- Some tests may incur API costs
- Use `--verbose` flag for detailed logging during debugging

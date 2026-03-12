# PDF OCR 告警与 URL 图片错位评审计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 2 个已确认的真实问题：未启用 OCR 时 PDF 转换仍触发底层 Tesseract 探测告警；URL 标准 LLM 清洗链路会把原位图片漂移到文末。

**Architecture:** P0 先修行为错误和误导性日志，避免用户看到与实际配置不一致的 OCR 提示；同时把标准 `process_document()` / `clean_markdown()` 路径补齐图片位置保护，禁止“缺图补到文末”的兜底策略继续制造结构性错位。P1 再补 URL 路由分支的回归测试和审计点，防止单 URL / 目录 URL 路径长期漂移。

**Tech Stack:** Python 3.11-3.13, pytest, unittest.mock, asyncio, pymupdf4llm, Markitai workflow / CLI batch pipeline

---

## Scope

本计划覆盖以下两个问题：

1. **PDF 非 OCR 模式的误导性告警**
   - 现象：未显式启用 `--ocr` 时，批处理中出现 `OCR disabled because Tesseract language data not found.`
   - 真实影响：命令成功，但日志与实际配置不一致，误导用户认为 OCR 逻辑参与了本次转换。

2. **URL `.llm.md` 图片位置漂移**
   - 现象：`[人是什么单位.md](/home/y/dev/markitai/output/人是什么单位.md#L13)` 的首图在 `[人是什么单位.llm.md](/home/y/dev/markitai/output/人是什么单位.llm.md#L46)` 被移动到正文末尾。
   - 真实影响：`.llm.md` 破坏原始阅读顺序，且 alt text 更新掩盖了图片漂移根因。

---

## Review Findings

### Finding 1: PDF 非 OCR 路径仍走了底层 OCR 探测

**Evidence:**
- [`packages/markitai/src/markitai/converter/pdf.py:121`](/home/y/dev/markitai/packages/markitai/src/markitai/converter/pdf.py#L121) 的 `pymupdf4llm.to_markdown(...)` 未显式传 `use_ocr=False`
- [`packages/markitai/src/markitai/converter/pdf.py:669`](/home/y/dev/markitai/packages/markitai/src/markitai/converter/pdf.py#L669) 的另一个 `pymupdf4llm.to_markdown(...)` 调用同样未传 `use_ocr=False`
- [`pymupdf4llm/__init__.py:73`](/home/y/dev/markitai/.venv/lib/python3.13/site-packages/pymupdf4llm/__init__.py#L73) 默认 `use_ocr=True`
- [`document_layout.py:946-956`](/home/y/dev/markitai/.venv/lib/python3.13/site-packages/pymupdf4llm/helpers/document_layout.py#L946) 会调用 `pymupdf.get_tessdata()`，失败后直接 `print("OCR disabled because Tesseract language data not found.")`

**Assessment:**
- 这是依赖默认行为泄漏，不是 Markitai 主动开启 OCR。
- 正确修法应是明确关闭底层 OCR 探测，而不是吞掉 stderr。

### Finding 2: 标准文档清洗链路没有图片位置保护

**Evidence:**
- URL 目录批处理会在 LLM 前下载图片并生成 `markdown_for_llm`，见 [`packages/markitai/src/markitai/cli/processors/batch.py:303-315`](/home/y/dev/markitai/packages/markitai/src/markitai/cli/processors/batch.py#L303)
- 但普通 `defuddle` URL 不会走 `enhance_url_with_vision()`，而是走标准 `process_with_llm()`，见 [`packages/markitai/src/markitai/cli/processors/batch.py:438-499`](/home/y/dev/markitai/packages/markitai/src/markitai/cli/processors/batch.py#L438)
- `process_with_llm()` 调用 [`LLMProcessor.process_document()`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/document.py#L1567)
- `process_document()` 只用了 `protect_content()`，没有用 `_protect_image_positions()`，见 [`packages/markitai/src/markitai/llm/document.py:1599-1622`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/document.py#L1599)
- `protect_content()` 明确写明“Images are NOT protected anymore”，见 [`packages/markitai/src/markitai/llm/content.py:147-162`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/content.py#L147)
- 更严重的是，`unprotect_content()` 对缺失图片的兜底是“追加到文末”，见 [`packages/markitai/src/markitai/llm/content.py:289-297`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/content.py#L289)

**Assessment:**
- 这是代码层结构保护缺失，不是 prompt 单独失效。
- 当前 fallback 策略会把“模型漏图”稳定放大成“文末错位”。

### Finding 3: 现有测试覆盖 helper，但未覆盖真实链路

**Evidence:**
- 已有图片占位 helper 测试：[`packages/markitai/tests/unit/test_document_utils.py:27-119`](/home/y/dev/markitai/packages/markitai/tests/unit/test_document_utils.py#L27)
- 但 `process_document()` 的现有测试没有断言“图片仍在原位”
- `process_with_llm()` 的现有测试只断言文件生成与 page comments，见 [`packages/markitai/tests/unit/test_llm_processor_cli.py:101-174`](/home/y/dev/markitai/packages/markitai/tests/unit/test_llm_processor_cli.py#L101)

**Assessment:**
- 这是典型测试缺口：有低层工具测试，无真实行为回归测试。

---

## Acceptance Criteria

满足以下条件才算完成：

1. 运行以下命令时，在 **未启用 `--ocr`** 的前提下不再出现 Tesseract 相关告警：

```bash
markitai packages/markitai/tests/fixtures/ --preset rich --no-cache -o output --verbose
```

2. 对 `https://ynewtime.com/jekyll-ynewtime/人是什么单位` 的标准 URL LLM 处理结果中，首图保持在正文前部，不允许漂移到文末。

3. 不破坏以下现有行为：
   - PDF 页码 marker 稳定化
   - screenshot/page comment 保护
   - alt text 更新
   - vision URL 路径已有的 `__MARKITAI_IMG_*__` 保护

4. 新增测试必须能先失败再通过，符合 TDD。

---

## Implementation Order

- **P0-1:** 关闭 PDF 非 OCR 路径的底层 OCR 探测
- **P0-2:** 修复标准文档清洗链路的图片位置保护与错误 fallback
- **P0-3:** 补真实链路回归测试
- **P1:** 审计 URL 单文件/目录批处理分支差异并补路由回归测试

---

### Task 1: P0 — PDF 非 OCR 路径显式禁用底层 OCR（测试）

**Files:**
- Test: `packages/markitai/tests/unit/test_converter_pdf.py`

**Step 1: Write the failing test**

在 `TestConvertBasic` 中新增测试，断言非 OCR 模式调用 `pymupdf4llm.to_markdown()` 时带 `use_ocr=False`：

```python
@patch("markitai.converter.pdf.pymupdf4llm")
def test_convert_disables_pymupdf4llm_ocr_when_markitai_ocr_disabled(
    self, mock_pymupdf4llm: Mock, tmp_path: Path
) -> None:
    mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()

    config = MarkitaiConfig(ocr=OCRConfig(enabled=False))
    converter = PdfConverter(config)
    converter.convert(pdf_file, tmp_path)

    call_args = mock_pymupdf4llm.to_markdown.call_args
    assert call_args[1]["use_ocr"] is False
```

再给 `_render_pages_for_llm()` 所在路径补一个同类测试，确认 OCR+LLM 预处理文本抽取也不会偷偷启用底层 Tesseract：

```python
@patch("markitai.converter.pdf.pymupdf4llm")
def test_render_pages_for_llm_disables_pymupdf4llm_builtin_ocr(
    self, mock_pymupdf4llm: Mock, tmp_path: Path
) -> None:
    mock_pymupdf4llm.to_markdown.return_value = "Extracted text"

    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()

    config = MarkitaiConfig(
        ocr=OCRConfig(enabled=True),
        llm=LLMConfig(enabled=True),
    )
    converter = PdfConverter(config)

    with patch.object(converter, "_render_pages_parallel", return_value=[]):
        converter._render_pages_for_llm(pdf_file, tmp_path)

    call_args = mock_pymupdf4llm.to_markdown.call_args
    assert call_args[1]["use_ocr"] is False
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_converter_pdf.py -k "disables_pymupdf4llm_ocr or builtin_ocr" -v
```

Expected:
- FAIL with missing `use_ocr` kwarg or `KeyError`

---

### Task 2: P0 — PDF 非 OCR 路径显式禁用底层 OCR（实现）

**Files:**
- Modify: `packages/markitai/src/markitai/converter/pdf.py`
- Test: `packages/markitai/tests/unit/test_converter_pdf.py`

**Step 1: Implement minimal fix**

在两个 `pymupdf4llm.to_markdown(...)` 调用中显式加上：

```python
use_ocr=False,
```

具体位置：
- [`packages/markitai/src/markitai/converter/pdf.py:121-129`](/home/y/dev/markitai/packages/markitai/src/markitai/converter/pdf.py#L121)
- [`packages/markitai/src/markitai/converter/pdf.py:669-676`](/home/y/dev/markitai/packages/markitai/src/markitai/converter/pdf.py#L669)

**Step 2: Run targeted tests**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_converter_pdf.py -k "disables_pymupdf4llm_ocr or builtin_ocr" -v
```

Expected:
- PASS

**Step 3: Run broader PDF test subset**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_converter_pdf.py -v
```

Expected:
- All PASS

---

### Task 3: P0 — 标准 `process_document()` 路径补图片位置回归测试

**Files:**
- Test: `packages/markitai/tests/unit/test_document_utils.py`

**Step 1: Write the failing test — LLM 把图片移到文末时，最终结果仍应保留原位**

新增一个 `process_document()` 级别测试，模拟 structured result 把图片移到文末：

```python
@pytest.mark.asyncio
async def test_process_document_preserves_image_position_when_llm_moves_image(
    self,
    llm_config: LLMConfig,
    prompts_config: PromptsConfig,
    mock_instructor_result_factory,
) -> None:
    from markitai.llm import LLMProcessor

    processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
    original = (
        "![cover](.markitai/assets/article.0001.jpeg)\n\n"
        "第一段。\n\n"
        "第二段。"
    )
    result, raw = mock_instructor_result_factory(
        cleaned_markdown="第一段。\n\n第二段。\n\n![cover](.markitai/assets/article.0001.jpeg)"
    )

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock()
    processor._router = mock_router

    with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(result, raw)
        )
        mock_instructor.return_value = mock_client

        cleaned, _ = await processor.process_document(original, "article.md")

    assert cleaned.splitlines()[0].startswith("![cover]")
```

**Step 2: Write the failing test — 缺失图片占位符时不能补到文末**

```python
@pytest.mark.asyncio
async def test_process_document_does_not_append_missing_image_to_end(
    self,
    llm_config: LLMConfig,
    prompts_config: PromptsConfig,
    mock_instructor_result_factory,
) -> None:
    from markitai.llm import LLMProcessor

    processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
    original = (
        "![cover](.markitai/assets/article.0001.jpeg)\n\n"
        "第一段。\n\n"
        "第二段。"
    )
    result, raw = mock_instructor_result_factory(
        cleaned_markdown="第一段。\n\n第二段。"
    )

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock()
    processor._router = mock_router

    with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(result, raw)
        )
        mock_instructor.return_value = mock_client

        cleaned, _ = await processor.process_document(original, "article.md")

    assert cleaned.splitlines()[0].startswith("![cover]")
    assert cleaned.rstrip().endswith("第二段。")
```

**Step 3: Run test to verify it fails**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_document_utils.py -k "preserves_image_position or does_not_append_missing_image_to_end" -v
```

Expected:
- FAIL，因为当前实现会允许图片漂移或直接补到文末

---

### Task 4: P0 — 在标准文档清洗链路中启用图片位置保护（实现）

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Modify: `packages/markitai/src/markitai/llm/content.py`
- Test: `packages/markitai/tests/unit/test_document_utils.py`

**Step 1: 在 `process_document()` 中引入图片占位保护**

在 [`packages/markitai/src/markitai/llm/document.py:1599-1601`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/document.py#L1599) 附近，将：

```python
protected = self.extract_protected_content(markdown)
protected_content, mapping = self.protect_content(markdown)
```

调整为“先保护图片，再保护 page/slide/comment”：

```python
image_protected_markdown, image_mapping = self._protect_image_positions(markdown)
protected = self.extract_protected_content(image_protected_markdown)
protected_content, mapping = self.protect_content(image_protected_markdown)
```

在 `unprotect_content(...)` 后补：

```python
cleaned = self._restore_image_positions(cleaned, image_mapping)
```

**Step 2: 对 `clean_markdown()` 做同样处理**

原因：
- `process_document()` structured 分支失败时会 fallback 到 `clean_markdown()`
- 如果只修 `process_document()` 主路径，fallback 仍会复现图片错位

所以在 [`packages/markitai/src/markitai/llm/document.py:300-349`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/document.py#L300) 的 `clean_markdown()` 里同步引入同样的图片占位保护与恢复逻辑。

**Step 3: 禁止“缺图补到文末”的兜底策略继续污染结构**

当前问题不只是“没保护图片”，还包括“丢图后错误补尾”。  
在 [`packages/markitai/src/markitai/llm/content.py:289-297`](/home/y/dev/markitai/packages/markitai/src/markitai/llm/content.py#L289) 调整策略：

推荐实现方向：

```python
def unprotect_content(
    content: str,
    mapping: dict[str, str],
    protected: dict[str, list[str]] | None = None,
    restore_missing_images_at_end: bool = True,
) -> str:
```

然后仅在仍依赖旧行为的路径保留 `True`，而 `process_document()` / `clean_markdown()` 在启用 `image_mapping` 时传 `False`。

**Step 4: 对图片 placeholder 丢失增加安全 fallback**

因为如果 LLM 删除了 `__MARKITAI_IMG_*__`，单纯 `restore_image_positions()` 无法恢复原位。  
建议在 `document.py` 新增一个 helper，做类似 page marker 的缺失检查：

```python
def _fallback_if_image_placeholders_missing(
    self,
    llm_output: str,
    original_markdown: str,
    image_mapping: dict[str, str],
    source: str,
    stage: str,
) -> str | None:
```

策略建议：
- 如果任一图片 placeholder 丢失，优先回退到 `original_markdown`
- 理由：结构正确性优先于局部格式优化

**Step 5: Run targeted tests**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_document_utils.py -k "preserves_image_position or does_not_append_missing_image_to_end" -v
```

Expected:
- PASS

---

### Task 5: P0 — 补 CLI 级回归测试，确保 `.llm.md` 最终写入不再漂移

**Files:**
- Test: `packages/markitai/tests/unit/test_llm_processor_cli.py`

**Step 1: Write the failing test**

新增 `process_with_llm()` 级别测试，模拟 processor 返回“图片被移到文末”的 cleaned markdown，断言最终 `.llm.md` 仍保留原位：

```python
async def test_process_with_llm_preserves_original_image_order(
    self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
):
    from markitai.cli.processors.llm import process_with_llm

    output_file = tmp_path / "output.md"
    markdown = "![cover](.markitai/assets/test.jpeg)\n\n第一段。\n\n第二段。"

    mock_llm_processor.process_document = AsyncMock(
        return_value=(
            "第一段。\n\n第二段。\n\n![cover](.markitai/assets/test.jpeg)",
            "title: Test\nsource: test.md",
        )
    )
    mock_llm_processor.format_llm_output = MagicMock(
        side_effect=lambda md, fm, source=None: f"---\n{fm}\n---\n\n{md}"
    )

    assets_dir = tmp_path / ".markitai" / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "test.jpeg").write_bytes(b"fake")

    await process_with_llm(
        markdown=markdown,
        source="test.md",
        cfg=markitai_config,
        output_file=output_file,
        processor=mock_llm_processor,
    )

    content = output_file.with_suffix(".llm.md").read_text(encoding="utf-8")
    body = content.split("\n---\n", 1)[1]
    assert body.lstrip().startswith("\n![cover](.markitai/assets/test.jpeg)")
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_llm_processor_cli.py -k "preserves_original_image_order" -v
```

Expected:
- FAIL

**Step 3: After implementation, re-run**

Run:

```bash
uv run pytest packages/markitai/tests/unit/test_llm_processor_cli.py -k "preserves_original_image_order" -v
```

Expected:
- PASS

---

### Task 6: P1 — 审计 URL 单文件 / 目录批处理路由差异并补保护性测试

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py`
- Test: `packages/markitai/tests/unit/test_url_processor.py`

**Step 1: Add review-only assertions**

需要为以下分支补测试，不一定要求立即重构：
- `has_multi_source and has_screenshot` 时走 `process_url_with_vision()`
- 否则走 `process_with_llm()`
- 即使走标准路径，也必须依赖 `process_document()` 的图片位置保护而不漂移

**Step 2: Decide whether to refactor routing**

如果测试暴露单 URL 与目录 URL 行为不一致，再把“URL LLM 路由决策”抽成共享 helper。  
这一项不是本次 P0 的必要修复，但应作为后续审计项保留。

---

## Verification Matrix

### Unit

```bash
uv run pytest packages/markitai/tests/unit/test_converter_pdf.py -v
uv run pytest packages/markitai/tests/unit/test_document_utils.py -k "image" -v
uv run pytest packages/markitai/tests/unit/test_llm_processor_cli.py -k "image or llm" -v
```

### Focused behavior repro

```bash
python - <<'PY'
from pathlib import Path
from markitai.config import MarkitaiConfig, OCRConfig
from markitai.converter.pdf import PdfConverter

cfg = MarkitaiConfig(ocr=OCRConfig(enabled=False))
PdfConverter(cfg).convert(
    Path("packages/markitai/tests/fixtures/sample.pdf"),
    Path("/tmp/markitai-pdf-repro"),
)
print("done")
PY
```

Expected:
- 输出不再出现 `OCR disabled because Tesseract language data not found.`

### End-to-end smoke

```bash
rm -rf /tmp/markitai-review-output
markitai packages/markitai/tests/fixtures/ --preset rich --no-cache -o /tmp/markitai-review-output --verbose
```

Check:
- 控制台没有误导性 OCR 告警
- `/tmp/markitai-review-output/人是什么单位.llm.md` 的图片仍位于正文前部

---

## Risks

1. **图片 placeholder 恢复策略过于激进**
   - 如果直接“任意 placeholder 丢失就全量 fallback 原文”，可能减少一部分文档清洗收益。
   - 但在当前问题中，结构正确性优先，接受这个保守策略。

2. **`unprotect_content()` 行为变更影响其它路径**
   - 因此建议新增参数控制，仅在标准文档清洗路径关闭“补到文末”。

3. **PDF 两处 `use_ocr=False` 需要同时修**
   - 只修一个调用点，后续仍会在另一路径看到相同告警。

---

## Out of Scope

- 不在本次修复中引入新的 OCR 后端
- 不在本次修复中统一所有 URL 处理分支的架构
- 不在本次修复中优化 prompt 文案本身，除非测试证明结构性修复后仍有稳定问题

---

## Recommended Commit Sequence

1. `test: add pdf builtin-ocr disable regression tests`
2. `fix: disable pymupdf4llm builtin ocr in pdf converter`
3. `test: add image position regressions for process_document`
4. `fix: preserve image positions in standard llm document flow`
5. `test: add cli llm output image order regression`

---

Plan complete and saved to `docs/plans/2026-03-12-pdf-ocr-warning-and-url-image-drift-review-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**

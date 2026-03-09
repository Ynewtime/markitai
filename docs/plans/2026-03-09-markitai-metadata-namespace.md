# `.markitai/` 元数据命名空间 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将输出目录中的元数据目录（assets/, screenshots/, reports/）移入 `.markitai/` 命名空间，消除与输入目录同名子目录的冲突。

**Architecture:** 在每个输出目录层级创建 `.markitai/` 目录，将 `assets/`、`screenshots/`、`reports/` 作为其子目录。所有 markdown 中的相对路径从 `assets/x.png` 变为 `.markitai/assets/x.png`。通过集中常量定义和修改路径工具函数，让上层模块的改动尽量机械化。

**Tech Stack:** Python, pathlib, Ruff, pytest

---

## 变更矩阵

### 常量与路径前缀

| 用途 | 旧值 | 新值 |
|------|------|------|
| 元数据根目录名 | _(无)_ | `MARKITAI_META_DIR = ".markitai"` |
| assets 相对路径前缀 | `"assets"` | `ASSETS_REL_PATH = ".markitai/assets"` |
| screenshots 相对路径前缀 | `"screenshots"` | `SCREENSHOTS_REL_PATH = ".markitai/screenshots"` |
| reports 相对路径前缀 | `"reports"` | `REPORTS_REL_PATH = ".markitai/reports"` |
| 物理目录路径 | `output_dir / "assets"` | `output_dir / ".markitai" / "assets"` |

### 受影响文件清单

#### 源码（按修改顺序）

| # | 文件 | 变更类型 | 说明 |
|---|------|----------|------|
| 1 | `constants.py` | 新增常量 | 4 个新常量 |
| 2 | `utils/paths.py` | 修改函数 | `ensure_assets_dir`, `ensure_screenshots_dir` 路由到 `.markitai/` 下；新增 `ensure_reports_dir` |
| 3 | `converter/image.py` | 字符串替换 | `"assets/{name}"` → 使用常量 |
| 4 | `converter/pdf.py` | 路径+字符串 | 截图目录、markdown 引用、regex 替换 |
| 5 | `converter/office.py` | 路径+字符串 | 截图目录、幻灯片注释 |
| 6 | `converter/legacy.py` | 字符串替换 | assets/screenshots 路径字符串 |
| 7 | `image.py` | regex+字符串 | `remove_nonexistent_images` regex、`remove_hallucinated_images` 判断、`replace_base64_with_paths` 默认参数、`download_images` 路径 |
| 8 | `workflow/core.py` | 路径+字符串 | `assets_dir` 构造、截图注释模板、`remove_nonexistent_images` 调用 |
| 9 | `workflow/helpers.py` | 路径+字符串 | `write_images_json` assets_dir 判断和回退路径 |
| 10 | `workflow/single.py` | 字符串替换 | `"assets/{name}"` 图片引用 |
| 11 | `cli/processors/llm.py` | 路径+字符串 | assets_dir 构造、截图注释、图片引用 |
| 12 | `cli/processors/url.py` | 字符串替换 | 截图引用 |
| 13 | `cli/processors/batch.py` | 路径 | 截图目录创建 |
| 14 | `batch.py` | 路径 | reports_dir 构造 |
| 15 | `utils/cli_helpers.py` | 路径 | `get_report_file_path` reports_dir |
| 16 | `llm/content.py` | regex+字符串 | 截图模式匹配 |
| 17 | `llm/document.py` | regex+字符串 | 截图移除 regex |
| 18 | `utils/text.py` | regex | 空 assets 路径清理 |
| 19 | `prompts/document_vision_system.md` | 示例文本 | `assets/` → `.markitai/assets/` |
| 20 | `prompts/document_process_system.md` | 示例文本 | `assets/` → `.markitai/assets/` |

#### 测试（按对应源码分组）

| 测试文件 | 变更类型 |
|----------|----------|
| `tests/unit/test_image_converter.py` | 路径断言、目录构造 |
| `tests/unit/test_converter_pdf.py` | 路径断言、目录构造、regex 测试 |
| `tests/unit/test_converter_legacy.py` | 路径断言、目录构造 |
| `tests/unit/test_image.py` | 路径断言、目录构造、markdown 字符串 |
| `tests/unit/test_workflow_core.py` | assets_dir 构造、断言 |
| `tests/unit/test_workflow_helpers.py` | assets_dir 构造、断言 |
| `tests/unit/test_workflow_single.py` | markdown 字符串断言 |
| `tests/unit/test_llm_processor_cli.py` | assets_dir 构造、markdown 断言 |
| `tests/unit/test_url_processor.py` | screenshots 路径 |
| `tests/unit/test_batch_processor.py` | assets/screenshots 路径 |
| `tests/unit/test_cli_helpers.py` | reports 路径、assets_dir |
| `tests/unit/test_batch.py` | reports 路径 |
| `tests/unit/test_llm_content.py` | screenshots 字符串 |
| `tests/unit/test_document_utils.py` | assets/screenshots 字符串 |
| `tests/unit/test_content_dedup.py` | assets/screenshots 字符串 |
| `tests/unit/test_content_edge_cases.py` | assets 字符串 |
| `tests/unit/test_llm.py` | assets 字符串 |
| `tests/unit/test_security.py` | assets_dir 构造（可能不需改） |
| `tests/unit/test_cli_main.py` | reports glob pattern |
| `tests/integration/test_output_format.py` | assets_dir 断言 |
| `tests/integration/test_url.py` | assets 字符串 |
| `tests/integration/test_cli.py` | reports 路径 |
| `tests/integration/test_cli_full.py` | reports 路径 |
| `tests/integration/test_real_scenarios.py` | reports 路径 |

---

## Tasks

### Task 1: Foundation — 常量与路径工具

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py`
- Modify: `packages/markitai/src/markitai/utils/paths.py`
- Modify: `packages/markitai/src/markitai/utils/__init__.py`

**Step 1: Add constants**

在 `constants.py` 末尾添加：

```python
# Metadata directory namespace — isolates markitai metadata from user content
MARKITAI_META_DIR = ".markitai"
ASSETS_REL_PATH = f"{MARKITAI_META_DIR}/assets"
SCREENSHOTS_REL_PATH = f"{MARKITAI_META_DIR}/screenshots"
REPORTS_REL_PATH = f"{MARKITAI_META_DIR}/reports"
```

**Step 2: Update `utils/paths.py`**

```python
from markitai.constants import MARKITAI_META_DIR

def ensure_assets_dir(output_dir: Path) -> Path:
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "assets")

def ensure_screenshots_dir(output_dir: Path) -> Path:
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "screenshots")

def ensure_reports_dir(output_dir: Path) -> Path:
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "reports")
```

Export `ensure_reports_dir` from `utils/__init__.py`.

**Step 3: Run existing tests to see cascade failures**

```bash
uv run pytest packages/markitai/tests/unit/ -x --timeout=60 2>&1 | tail -30
```

Expected: Many failures due to path changes. This confirms the foundation change propagated.

**Step 4: Commit**

```
feat: add .markitai/ metadata namespace constants and path utilities
```

---

### Task 2: Image Converter

**Files:**
- Modify: `packages/markitai/src/markitai/converter/image.py`
- Modify: `packages/markitai/tests/unit/test_image_converter.py`

**Changes:**
- `_copy_to_assets` return value: `f"assets/{name}"` → `f"{ASSETS_REL_PATH}/{name}"`
- Import `ASSETS_REL_PATH` from constants
- Tests: update all `"assets/"` assertions to `".markitai/assets/"`、`output_dir / "assets"` → `output_dir / ".markitai" / "assets"`

**Verify:** `uv run pytest tests/unit/test_image_converter.py -v`

**Commit:** `refactor: image converter uses .markitai/assets/ namespace`

---

### Task 3: PDF Converter

**Files:**
- Modify: `packages/markitai/src/markitai/converter/pdf.py`
- Modify: `packages/markitai/tests/unit/test_converter_pdf.py`

**Changes:**
- `_fix_image_paths` regex replacement: `r"![\1](assets/\2)"` → `rf"![\1]({ASSETS_REL_PATH}/\2)"`
- Screenshot markdown comments: `f"screenshots/{image_name}"` → `f"{SCREENSHOTS_REL_PATH}/{image_name}"`
- Tests: update path assertions and directory constructions

**Verify:** `uv run pytest tests/unit/test_converter_pdf.py -v`

**Commit:** `refactor: PDF converter uses .markitai/ namespace`

---

### Task 4: Office & Legacy Converters

**Files:**
- Modify: `packages/markitai/src/markitai/converter/office.py`
- Modify: `packages/markitai/src/markitai/converter/legacy.py`
- Modify: `packages/markitai/tests/unit/test_converter_legacy.py`

**Changes:**
- Office: slide comment template `f"screenshots/{slide_info['name']}"` → use constant
- Legacy: `f"(assets/{name}"` and `f"(screenshots/{name}"` → use constants
- Tests: update path assertions

**Verify:** `uv run pytest tests/unit/test_converter_legacy.py -v`

**Commit:** `refactor: office & legacy converters use .markitai/ namespace`

---

### Task 5: Image Processor (image.py)

**Files:**
- Modify: `packages/markitai/src/markitai/image.py`
- Modify: `packages/markitai/tests/unit/test_image.py`

**Changes:**
- `replace_base64_with_paths`: default `assets_path="assets"` → `assets_path=ASSETS_REL_PATH`
- `remove_nonexistent_images` regex: `r"!\[[^\]]*\]\(assets[/\\]([^)]+)\)"` → 使用 `ASSETS_REL_PATH` 构造（注意转义 `.`）
- `remove_hallucinated_images`: `url.startswith("assets/")` → `url.startswith(ASSETS_REL_PATH + "/")` 或 `url.startswith(".markitai/")`
- `download_images`: `local_path = f"assets/{filename}"` → 使用常量
- Tests: 大量路径和断言更新

**Verify:** `uv run pytest tests/unit/test_image.py -v`

**Commit:** `refactor: image processor uses .markitai/assets/ namespace`

---

### Task 6: Workflow Layer

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py`
- Modify: `packages/markitai/src/markitai/workflow/helpers.py`
- Modify: `packages/markitai/src/markitai/workflow/single.py`
- Modify: `packages/markitai/tests/unit/test_workflow_core.py`
- Modify: `packages/markitai/tests/unit/test_workflow_helpers.py`
- Modify: `packages/markitai/tests/unit/test_workflow_single.py`

**Changes in core.py:**
- `ctx.output_dir / "assets"` → `ctx.output_dir / MARKITAI_META_DIR / "assets"`（2 处：line 405, 646）
- Screenshot comment template: `f"screenshots/{img['name']}"` → use constant
- `apply_alt_text_updates`: `f"assets/{asset_path.name}"` → use constant

**Changes in helpers.py:**
- `write_images_json`: `image_path.parent.name == "assets"` → 检查 parent path 是否在 `.markitai/assets` 下
- Fallback: `output_dir / "assets"` → `output_dir / MARKITAI_META_DIR / "assets"`

**Changes in single.py:**
- `f"assets/{image_path.name}"` → use constant
- `f"assets/{asset_name}"` → use constant

**Verify:** `uv run pytest tests/unit/test_workflow_core.py tests/unit/test_workflow_helpers.py tests/unit/test_workflow_single.py -v`

**Commit:** `refactor: workflow layer uses .markitai/ namespace`

---

### Task 7: CLI Processors

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/llm.py`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py`
- Modify: `packages/markitai/tests/unit/test_llm_processor_cli.py`
- Modify: `packages/markitai/tests/unit/test_url_processor.py`
- Modify: `packages/markitai/tests/unit/test_batch_processor.py`

**Changes in llm.py:**
- `output_file.parent / "assets"` → `output_file.parent / MARKITAI_META_DIR / "assets"`
- Screenshot/page image comment templates: use constants
- `f"assets/{image_path.name}"` → use constant

**Changes in url.py:**
- `f"screenshots/{screenshot_path.name}"` → use constant (6 处)

**Changes in batch.py:**
- `ensure_screenshots_dir(output_dir)` 调用不需改（已在 Task 1 修正）

**Verify:** `uv run pytest tests/unit/test_llm_processor_cli.py tests/unit/test_url_processor.py tests/unit/test_batch_processor.py -v`

**Commit:** `refactor: CLI processors use .markitai/ namespace`

---

### Task 8: Reports

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`
- Modify: `packages/markitai/src/markitai/utils/cli_helpers.py`
- Modify: `packages/markitai/tests/unit/test_batch.py`
- Modify: `packages/markitai/tests/unit/test_cli_helpers.py`

**Changes:**
- `batch.py` line 558: `self.output_dir / "reports"` → `self.output_dir / MARKITAI_META_DIR / "reports"`
- `cli_helpers.py` line 161: `output_dir / "reports"` → 同上
- Tests: update `output_dir / "reports"` 构造和 glob 断言

**Verify:** `uv run pytest tests/unit/test_batch.py tests/unit/test_cli_helpers.py -v`

**Commit:** `refactor: reports use .markitai/ namespace`

---

### Task 9: LLM Content Protection & Document Processing

**Files:**
- Modify: `packages/markitai/src/markitai/llm/content.py`
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Modify: `packages/markitai/tests/unit/test_llm_content.py`
- Modify: `packages/markitai/tests/unit/test_document_utils.py`
- Modify: `packages/markitai/tests/unit/test_content_dedup.py`
- Modify: `packages/markitai/tests/unit/test_content_edge_cases.py`
- Modify: `packages/markitai/tests/unit/test_llm.py`

**Changes in content.py:**
- `"screenshots/" in img_ref` → `".markitai/screenshots/" in img_ref`
- Screenshot comment regex pattern

**Changes in document.py:**
- `_remove_uncommented_screenshots` regex: `screenshots/` → `.markitai/screenshots/`
- `_protect_image_positions`: screenshot path check
- Page screenshot regex pattern (line 45)

**Verify:** `uv run pytest tests/unit/test_llm_content.py tests/unit/test_document_utils.py tests/unit/test_content_dedup.py tests/unit/test_content_edge_cases.py tests/unit/test_llm.py -v`

**Commit:** `refactor: LLM content protection uses .markitai/ namespace`

---

### Task 10: Prompts & Utils

**Files:**
- Modify: `packages/markitai/src/markitai/utils/text.py`
- Modify: `packages/markitai/src/markitai/prompts/document_vision_system.md`
- Modify: `packages/markitai/src/markitai/prompts/document_process_system.md`

**Changes in text.py:**
- Line 344 regex: `r"!\[[^\]]*\]\((?:assets/)?\)\s*\n?"` → `r"!\[[^\]]*\]\((?:\.markitai/assets/)?\)\s*\n?"`

**Changes in prompts:**
- All `assets/` references in examples → `.markitai/assets/`

**Verify:** `uv run pytest tests/unit/test_utils_text.py -v`

**Commit:** `refactor: prompts and text utils use .markitai/ namespace`

---

### Task 11: Remaining Tests & Integration

**Files:**
- Modify: `packages/markitai/tests/unit/test_security.py` (if assets dir construction needs update)
- Modify: `packages/markitai/tests/unit/test_cli_main.py`
- Modify: `packages/markitai/tests/integration/test_output_format.py`
- Modify: `packages/markitai/tests/integration/test_url.py`
- Modify: `packages/markitai/tests/integration/test_cli.py`
- Modify: `packages/markitai/tests/integration/test_cli_full.py`
- Modify: `packages/markitai/tests/integration/test_real_scenarios.py`

**Step 1:** Run full test suite, fix any remaining failures.

```bash
uv run pytest packages/markitai/tests/unit/ -v --timeout=60
```

**Step 2:** Fix each failing test.

**Step 3:** Run pre-commit hooks.

```bash
uv run pre-commit run --all-files
```

**Commit:** `test: update all remaining tests for .markitai/ namespace`

---

### Task 12: Final Verification

**Step 1:** Full unit test suite

```bash
uv run pytest packages/markitai/tests/unit/ --timeout=60
```

**Step 2:** Ruff + Pyright

```bash
uv run ruff check --fix && uv run ruff format && uv run pyright
```

**Step 3:** Manual smoke test

```bash
markitai packages/markitai/tests/fixtures/ --preset rich --no-cache -o /tmp/markitai-namespace-test --verbose
ls -la /tmp/markitai-namespace-test/.markitai/
ls -la /tmp/markitai-namespace-test/.markitai/assets/
ls -la /tmp/markitai-namespace-test/.markitai/reports/
```

Verify:
- `.md` files at output root（not inside `.markitai/`）
- `.markitai/assets/` contains extracted images
- `.markitai/reports/` contains report JSON
- No bare `assets/` or `screenshots/` dirs at output root
- `.md` files reference `.markitai/assets/...` in image links

**Commit:** (no commit, just verification)

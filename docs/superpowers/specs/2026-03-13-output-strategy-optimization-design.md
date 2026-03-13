# 优化单文件输出策略

## 问题

当前 markitai 在所有场景下都生成 `.md` 文件，导致两个问题：

1. **图片格式 + 非 LLM 模式**：生成的 `.md` 仅包含 frontmatter 和一行占位内容（如 `# sample\n\n![sample](path)`），对用户无价值
2. **LLM 模式**：同时生成 `.md` 和 `.llm.md`，`.md` 作为中间产物留在输出目录造成困惑，与 stdout 模式（只输出最终结果）行为不一致

## 设计

引入两条规则：

### 规则 A：图片格式 + 非 LLM + 非 OCR = 跳过

对 `IMAGE_ONLY_FORMATS`（JPEG、JPG、PNG、WEBP、GIF、BMP、TIFF，共 7 种），当**既未启用 LLM 也未启用 OCR** 时：

- 不执行转换，不写任何文件（跳过发生在 converter 调用之前，不会产生 `.markitai/assets/` 下的图片副本）
- 终端提示用户跳过原因及建议，例如：`⚠ Skipped sample.bmp (image file, no text to extract). Use --llm or --ocr for content extraction.`
- stdout 模式和 `-o` 模式行为一致（都跳过）
- 批量模式下同样适用，按文件独立判断

> **注意**：SVG 不在 `IMAGE_ONLY_FORMATS` 中。虽然当前 SVG 由 `ImageConverter` 注册处理，但 SVG 是 XML 格式，可包含 `<text>` 等文本元素，`KreuzbergConverter` 支持提取其文本内容。SVG 应作为文档格式处理，后续可将其转换器从 `ImageConverter` 迁移到 `KreuzbergConverter`。

### 规则 B：LLM 模式 = 只输出 `.llm.md`

启用 LLM 时（`--llm` 或 `--pure`），规则完全相同：

- 默认只写 `.llm.md`，不写 `.md` 到磁盘
  - 基线 markdown 内容保留在内存中（`ctx.conversion_result.markdown`），不影响 `stabilize_written_llm_output` 等依赖基线内容的流程（`_read_markdown_body` 已有 fallback 到内存值的逻辑）
- 新增 `--keep-base` CLI 参数：同时保留 `.md` 基线文件（用于调试/对比）
  - `--keep-base` 在 stdout 模式下无效果（stdout 模式不写文件）
- stdout 模式输出 `.llm.md` 内容（现状不变）
- 批量模式同样适用

### LLM 失败时的回退行为

当 LLM 处理失败（网络错误、配额不足等）时：

- 回退写入 `.md` 文件作为基线输出（即使未指定 `--keep-base`）
- 终端提示 LLM 处理失败，已输出基础转换结果

## 行为矩阵

| 格式类型 | 模式 | stdout | `-o` 文件输出 |
|----------|------|--------|---------------|
| 图片 | 非 LLM, 非 OCR | 提示跳过 | 提示跳过，不写文件 |
| 图片 | OCR (非 LLM) | markdown 内容 | `.md` |
| 图片 | LLM | `.llm.md` 内容 | `.llm.md` |
| 图片 | LLM + `--keep-base` | `.llm.md` 内容 | `.md` + `.llm.md` |
| 图片 | LLM 失败 | `.md` 内容 + 错误提示 | `.md` + 错误提示 |
| 文档 | 非 LLM | markdown 内容 | `.md` |
| 文档 | LLM | `.llm.md` 内容 | `.llm.md` |
| 文档 | LLM + `--keep-base` | `.llm.md` 内容 | `.md` + `.llm.md` |
| 文档 | LLM 失败 | `.md` 内容 + 错误提示 | `.md` + 错误提示 |

## 判定标准

`IMAGE_ONLY_FORMATS` 定义为纯光栅图片格式，不含 SVG：

```
JPEG, JPG, PNG, WEBP, GIF, BMP, TIFF
```

## 变更范围

### 需要修改的文件

- **`converter/base.py`**：定义 `IMAGE_ONLY_FORMATS` 集合（7 种纯光栅图片格式）
- **`workflow/core.py`**：
  - 图片 + 非 LLM + 非 OCR 时 early return（跳过转换，发生在 converter 调用之前）
  - LLM 模式下不写 `.md` 文件到磁盘（除非 `--keep-base`），基线内容保留在 `ctx.conversion_result.markdown`
  - LLM 失败时回退写入 `.md`
- **`workflow/single.py`**：适配不写 `.md` 文件的逻辑，确认 `_read_markdown_body` fallback 正常工作
- **`cli/main.py`**：新增 `--keep-base` CLI 参数
- **`config.py`**：在 LLM 配置中加入 `keep_base: bool = False`
- **`cli/processors/file.py`**：处理跳过逻辑和终端提示信息
- **`cli/processors/batch.py`**：批量模式下的跳过逻辑和统计

### 不变的部分

- 非 LLM + 文档格式：行为完全不变
- `.markitai/assets/` 中的图片资源管理：不受影响（跳过发生在 converter 之前，不会产生 assets）
- LLM 处理流程本身（调用 LLM、缓存等）：不受影响
- stdout 模式下的内容输出逻辑：基本不变

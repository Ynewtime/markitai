# 优化单文件输出策略

## 问题

当前 markitai 在所有场景下都生成 `.md` 文件，导致两个问题：

1. **图片/不支持格式 + 非 LLM 模式**：生成的 `.md` 仅包含 frontmatter 和一行占位内容（如 `# sample\n\n![sample](path)`），对用户无价值
2. **LLM 模式**：同时生成 `.md` 和 `.llm.md`，`.md` 作为中间产物留在输出目录造成困惑，与 stdout 模式（只输出最终结果）行为不一致

## 设计

引入两条规则：

### 规则 A：图片格式 + 非 LLM = 跳过

对 `ImageConverter` 覆盖的 8 种格式（JPEG、JPG、PNG、WEBP、GIF、BMP、TIFF、SVG），非 LLM 模式下：

- 不执行转换，不写任何文件
- 终端提示用户跳过原因及建议，例如：`⚠ Skipped sample.bmp (image file, no text to extract). Use --llm for AI-powered content extraction.`
- stdout 模式和 `-o` 模式行为一致（都跳过）
- 批量模式下同样适用，按文件独立判断

### 规则 B：LLM 模式 = 只输出 `.llm.md`

启用 LLM 时（`--llm` 或 `--pure`）：

- 默认只写 `.llm.md`，不写 `.md`
- 新增 `--keep-base` CLI 参数：同时保留 `.md` 基线文件（用于调试/对比）
- stdout 模式输出 `.llm.md` 内容（现状不变）
- 批量模式同样适用

## 行为矩阵

| 格式类型 | 模式 | stdout | `-o` 文件输出 |
|----------|------|--------|---------------|
| 图片 | 非 LLM | 提示跳过 | 提示跳过，不写文件 |
| 图片 | LLM | `.llm.md` 内容 | `.llm.md` |
| 图片 | LLM + `--keep-base` | `.llm.md` 内容 | `.md` + `.llm.md` |
| 文档 | 非 LLM | markdown 内容 | `.md` |
| 文档 | LLM | `.llm.md` 内容 | `.llm.md` |
| 文档 | LLM + `--keep-base` | `.llm.md` 内容 | `.md` + `.llm.md` |

## 判定标准

"图片格式"由 `ImageConverter` 注册的格式列表决定，即 `FileFormat` 枚举中的：

```
JPEG, JPG, PNG, WEBP, GIF, BMP, TIFF, SVG
```

在代码中定义为 `IMAGE_ONLY_FORMATS` 集合，复用 `ImageConverter` 已注册的格式，无需额外维护。

## 变更范围

### 需要修改的文件

- **`converter/base.py`** 或 **`constants.py`**：定义 `IMAGE_ONLY_FORMATS` 集合
- **`workflow/core.py`**：
  - 图片 + 非 LLM 时 early return（跳过转换）
  - LLM 模式下不写 `.md` 文件（除非 `--keep-base`）
- **`cli/main.py`**：新增 `--keep-base` CLI 参数
- **`config.py`**：在 LLM 配置中加入 `keep_base: bool = False`
- **`cli/processors/file.py`**：处理跳过逻辑和终端提示信息
- **`cli/processors/batch.py`**：批量模式下的跳过逻辑和统计

### 不变的部分

- 非 LLM + 文档格式：行为完全不变
- `.markitai/assets/` 中的图片资源管理：不受影响
- LLM 处理流程本身（调用 LLM、缓存等）：不受影响
- stdout 模式下的内容输出逻辑：基本不变

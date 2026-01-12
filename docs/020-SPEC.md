# 任务批次 2026011202 - v0.2.0 深度实施建议

本版本聚焦于**本地 OCR 能力支持**与**底层架构升级调研**。以下是针对工程落地的深度分析与实施细则。

## 1. 本地 OCR 能力支持 (Local OCR - Phase 1)

### 1.1 技术背景与依赖分析

`pymupdf4llm` 是基于 `pymupdf` (MuPDF) 的高层封装。

当前版本的 `pymupdf4llm` 存在**图片避让机制 (Image Avoidance)**：即便是通过 OCR 注入了隐藏文本层（Invisible Text），只要这些文本位于图片区域内，`pymupdf4llm` 的版面分析算法 (`column_boxes`) 就会将其忽略。
因此，传统的 "Sandwich PDF"（保留原图，叠加文字）方案在 `pymupdf4llm` 上行不通。

### 1.2 实施方案：OCR 重构流 (Reconstruction Pipeline)

为确保 OCR 文字被 100% 提取，我们将采用 **Text-Only Reconstruction** 策略。

#### 1. 核心流程（待进一步分析可行性）

1.  **输入**: 原始 PDF (扫描件或图文混排)。
2.  **OCR 处理**: 使用 `PyMuPDF` 的 `get_textpage_ocr` 接口（调用 Tesseract）提取全页信息。
3.  **重构**: 在内存中构建一个新的临时 PDF。
    *   遍历 OCR 结果的 Blocks/Lines/Spans。
    *   将所有文本以可见字体 (`render_mode=0`) 绘制到新 PDF 的对应坐标。
    *   **关键点**: 新 PDF 中**不包含原始图片**。这消除了 `pymupdf4llm` 的避让干扰。
4.  **转换**: 将这个纯文字 PDF 交给 `pymupdf4llm` 转换为 Markdown。
5.  **输出**: 包含完美布局和 OCR 文字的 Markdown。

#### 2. 配置模型扩展 (`src/markit/config/settings.py`)

```python
class PDFConfig(BaseModel):
    engine: Literal["pymupdf4llm", "pymupdf", "pdfplumber", "markitdown"] = "pymupdf4llm"
    extract_images: bool = True

    # OCR 配置
    ocr_enabled: bool = False
    ocr_language: str = "" # 待明确

    # OCR 模式
    # text_priority: 重构 PDF，丢弃图片，优先保证文字提取 (适用于扫描件)
    # layout_priority: (预留) 尝试保留图片，但可能导致文字丢失
    ocr_mode: Literal["text_priority"] = "text_priority"
```

#### 3. 核心转换逻辑 (`src/markit/converters/pdf/pymupdf4llm.py`)

*   **新增方法 `_apply_ocr(self, doc: fitz.Document) -> fitz.Document`**:
    *   负责执行上述的 OCR 和重构逻辑。
    *   返回一个新的 `fitz.Document` 对象（内存中）。
*   **Tessdata 路径**:
    *   尝试自动探测。
    *   支持通过环境变量 `MARKIT_TESSDATA_PATH` 覆盖。

#### 4. 环境诊断工具 (`src/markit/cli/commands/check.py`)

*   **命令**: `markit check`
*   **检查项**:
    *   **Tesseract**: 二进制文件存在性、版本 (>5.0)、语言包 (`--list-langs`)。
    *   **PyMuPDF**: 验证是否支持 OCR (部分构建可能缺失 Tesseract 绑定)。

### 1.3 局限性说明

*   **图片丢失**: 在 `text_priority` 模式下，Markdown 结果将不包含任何图片引用 (`![](...)`)。对于纯文本扫描件这是优势（去除了噪点背景），但对于包含插图的文档是功能缺失。=> 需要进一步分析是否有更优方案，图片要尽可能保留。
*   **性能开销与并发控制**:
    *   OCR 是极高强度的 CPU 密集型操作。
    *   必须在 `ConcurrencyConfig` 中引入 `ocr_workers` (默认: CPU核数/2 或 1)，并使用 `asyncio.Semaphore` 严格限制并发数。
    *   避免在 `batch` 模式下因无限制并发导致系统假死。

---

## 2. LLM 架构演进调研 (LiteLLM Analysis)

参考 [LiteLLM 架构演进调研报告](./reference/litellm_analysis.md)

### 2.1 架构映射与差异

| 维度 | MarkIt 现状 | LiteLLM 方案 | 建议 |
| :--- | :--- | :--- | :--- |
| **Provider** | 手写 `OpenAIProvider`, `AnthropicProvider` 等 | `litellm.completion(model="gpt-4", ...)` | 替换手写 Provider。 |
| **Streaming** | 自定义 AsyncIterator，手动处理 chunks | 统一返回 `ModelResponse` chunk |  |
| **Tokenizer** | 强依赖 `tiktoken` | `litellm.encode()` (支持多种 tokenizer) | 切换到 LiteLLM 的 tokenizer 以获得更准确的非 OpenAI 模型统计。 |
| **Cost** | 简单的硬编码或配置表 | 内置模型价格表 + 自定义注册 | 利用 LiteLLM 的价格表，同时保留 MarkIt 的自定义覆盖能力。 |

### 2.2 深度重构验证 (POC)

#### 1. 依赖体积风险
LiteLLM 依赖极其庞大（为了适配上百种 Provider）。
**策略**: 将 LiteLLM 作为 `optional-dependency` (`pip install markit[litellm]`) 还是核心依赖？=> 核心依赖
**决策**: 鉴于 MarkIt 旨在做一个“开箱即用”的工具，建议将 LiteLLM 作为**核心依赖**引入，但需评估其对冷启动时间的影响。

#### 2. 重点验证场景

验证：
1.  **Ollama 本地调用**: 验证 `model="ollama/llama3"` 的连通性及 `api_base` 传递。
2.  **Tool Calling 流式透传**: 验证在使用 LiteLLM 时，复杂的 Tool Call chunk 是否能正确解析（为未来功能做准备）。
3.  **异常捕获**: 确保 LiteLLM 抛出的 `ContextWindowExceededError` 能被 MarkIt 的重试逻辑捕获。

### 2.3 演进路线

1.  **v0.2.0 (当前)**: 完成调研，全面切换至 LiteLLM，清理旧 Provider 代码。

---

## 3. 测试计划

### 3.1 自动化测试
*   **OCR Unit Test**: 构造一个微型纯图 PDF（内含 "Hello World"），在 CI 环境中（需安装 Tesseract）运行转换，断言输出包含 "Hello World"。
*   **Image Avoidance Test**: 验证重构 PDF 策略是否真正解决了图片避让问题。

### 3.2 性能基准测试
*   **Benchmark**: 对比 10 页扫描 PDF 在“开启 OCR”与“关闭 OCR”下的耗时。
*   **Memory**: 监控 PDF 重构过程中的内存峰值，确保处理大文件（100+页）时不会 OOM。

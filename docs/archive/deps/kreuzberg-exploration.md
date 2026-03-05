# Kreuzberg 深度调研报告（含事实核查）

> 调研日期：2026-03-05 | 目标版本：v4.4.1 | 许可证：MIT

## 一、项目基本信息

| 项目       | 详情                                                       | 核查状态 |
|------------|-----------------------------------------------------------|----------|
| 名称       | Kreuzberg                                                  | -        |
| 仓库       | github.com/kreuzberg-dev/kreuzberg                         | -        |
| 最新版本   | **v4.4.1**（PyPI 发布于 2026-02-28）                        | 已核查   |
| 作者       | Na'aman Hirschfeld                                          | 已核查   |
| 许可证     | MIT                                                        | 已核查   |
| GitHub Star| ~6,456（org 页面数据，2026-03-03 更新）                     | 已核查   |
| Fork       | ~298                                                       | 已核查   |
| 语言构成   | HTML 56.4%, Rust 23.4%, TypeScript 4.6%, 其余为多语言绑定   | 已核查   |

### 事实核查说明

前一轮分析中我提到的"最新版本 v4.3.8"**有误**。GitHub README 中提到的 v4.3.8 是该 tag 创建时的版本，但此后已连续发布 v4.4.0（R 语言绑定）和 v4.4.1。**截至 2026-03-05，PyPI 上的最新版本为 v4.4.1**。

---

## 二、架构与实现方式

### 2.1 核心架构：Rust Core + 多语言薄绑定

Kreuzberg 的核心架构是 **Rust-first, multi-language**。所有提取逻辑在 Rust 中实现，各语言通过薄绑定层接入：

- **Python** → PyO3
- **Node.js** → NAPI-RS
- **Ruby** → Magnus
- **Java** → Foreign Function & Memory API
- **Go** → CGO / FFI
- **C#** → .NET FFI
- **PHP** → FFI extension
- **Elixir** → Rustler NIF
- **R** → extendr（v4.4.0 新增）
- **WASM** → wasm-bindgen

Rust Core 内部的模块划分（来自官方架构文档）：

```
core/       → 提取编排、MIME 检测、配置加载
plugins/    → 插件系统，Registry 模式 + Trait 定义
extraction/ → 各格式的具体实现（PDF/pdfium, Excel/calamine, etc.）
extractors/ → 将格式处理器注册到插件系统的包装层
ocr/        → OCR 处理编排、Tesseract 集成、hOCR 解析、表格检测
text/       → Token reduction、质量评分、字符串处理
types/      → 共享类型定义（ExtractionResult, Metadata, Chunk 等）
error/      → 集中错误处理
```

### 2.2 格式分发机制

系统通过 MIME type 检测（覆盖 118+ 种文件扩展名）来分发到对应的 extractor，采用优先级选择策略。用户无需关心具体解析器，只需传入文件路径或字节流。

### 2.3 关键依赖演进

| 依赖           | v3 状态           | v4 状态                          | 核查状态 |
|----------------|-------------------|----------------------------------|----------|
| Pandoc         | 通过子进程调用     | **已移除**，改为原生 Rust 解析器   | 已核查   |
| LibreOffice    | 处理 .doc/.ppt    | **已移除**（v4.3.0），OLE/CFB 原生解析 | 已核查   |
| PDFium         | -                 | 原生绑定（当前 chromium/7678）     | 已核查   |
| ONNX Runtime   | 内置绑定          | 改为动态加载（需单独安装 1.24+）    | 已核查   |

### 2.4 HTML → Markdown 转换管线

Kreuzberg 团队独立维护了 `html-to-markdown` 项目（548 stars），作为内部 HTML 转换管线的核心：

- Rust 核心引擎，所有语言绑定输出完全一致
- 官方声称转换速度 150-280 MB/s（**自报数据，未独立验证**）
- 支持 CommonMark 规范
- 支持多种输出格式：Markdown、Djot、Plain Text
- 支持 visitor 模式进行自定义转换

**事实核查说明**：前一轮分析中我提到该转换器"使用 html5ever 进行 DOM 解析"。经核查，CHANGELOG 中早期版本确实使用 scraper + html5ever，但 lib.rs 页面描述为"using the astral-tl parser"。**该转换器的底层 HTML 解析器可能已更换或共存，具体以源码为准。**

---

## 三、全量功能清单

### 3.1 核心提取功能

**支持格式数量**：官方声称 75+（GitHub description 写 76+），跨 8 大类别。核查结论：**基本准确，文档中可逐项列举的格式覆盖面广泛。**

| 类别           | 格式                                                                 |
|----------------|----------------------------------------------------------------------|
| 文档           | PDF, DOCX, DOC, ODT, TXT, MD, Djot, MDX, RST, Org, RTF              |
| 电子表格       | XLSX, XLSM, XLSB, XLS, XLA, XLAM, XLTM, ODS, CSV, TSV              |
| 演示文稿       | PPTX, PPTM, PPSX                                                     |
| 图片（OCR）    | PNG, JPG, GIF, WebP, BMP, TIFF, JP2, JBIG2, PNM, SVG 等             |
| 邮件           | EML, MSG                                                              |
| Web/标记       | HTML, XML, XHTML, SVG, JSON, YAML, TOML                              |
| 归档           | ZIP, TAR, TGZ, GZ, 7Z                                                |
| 学术/科学      | BibTeX, RIS, NBIB, ENW, CSL, LaTeX, Typst, JATS, IPYNB, FB2, DocBook |

**提取能力**：

- 文本提取：支持多字节编码（UTF-8/16），Mojibake 检测与修正
- 表格提取：从 PDF、电子表格、Word 中提取结构化表格，支持合并单元格，输出 Markdown/JSON
- 元数据提取：标题、作者、创建日期、页数、MIME 类型等
- 图片提取：从 PDF 和 Office 文档中提取嵌入图片

### 3.2 OCR 支持

| OCR 后端        | 可用范围                    | 语言支持    | 备注                           |
|-----------------|-----------------------------|-------------|--------------------------------|
| Tesseract       | 所有绑定（含 WASM）         | 100+ 语言   | 原生集成，支持 hOCR 输出        |
| PaddleOCR       | 所有非 WASM 原生绑定        | 80+ 语言    | PP-OCRv5, ONNX Runtime, 11 字族 |
| EasyOCR         | 仅 Python                   | 80+ 语言    | 需 Python < 3.14，支持 GPU      |

OCR 附加功能：自动 fallback、强制 OCR 模式、结果缓存、图像预处理（对比度/倾斜校正/降噪）、多语言检测。

### 3.3 高级处理功能

| 功能              | 描述                                                        |
|-------------------|-------------------------------------------------------------|
| 语言检测          | 基于 fast-langdetect，60+ 语言，可配置置信度阈值             |
| 内容分块          | 递归/语义/token 感知三种策略，可配置 chunk 大小和 overlap     |
| 嵌入生成          | 通过 ONNX Runtime 本地生成，fast/balanced/quality 三档预设   |
| Token 压缩        | TF-IDF 句子评分 + 位置权重，light/moderate/aggressive 三档   |
| 质量处理          | Unicode 规范化、空白处理、Mojibake 修复                      |
| 关键词提取        | YAKE + RAKE 两种算法，支持 n-gram                            |
| 页面追踪          | 逐页内容提取，字节精确的页面边界，O(1) 页面查找              |
| PDF 层级检测      | K-means 聚类推断文档结构（标题/章节/段落等语义层级）          |
| 输出格式          | plain（默认）、markdown、djot、html                          |
| 批量处理          | 可配置并发度的并行提取                                       |
| 结果缓存          | 磁盘缓存提取结果                                             |

### 3.4 插件系统（v4.0.0+）

四种插件类型：

1. **Custom Document Extractor** — 注册自定义格式提取器
2. **PostProcessor** — 对提取结果做后处理
3. **Validator** — 验证提取结果质量
4. **Custom OCR Backend** — 接入自定义 OCR 引擎

插件跨语言工作：Rust 中定义的插件可被所有语言绑定调用；Python/TypeScript 可定义自定义插件并通过线程安全的回调接入 Rust 核心。

### 3.5 部署模式

| 模式           | 描述                                                     |
|----------------|----------------------------------------------------------|
| 库（Library）  | 直接在应用中调用                                          |
| CLI            | 跨平台二进制，支持 extract/batch/detect/cache 等子命令    |
| REST API       | 基于 Axum 的 HTTP 服务器，带 OpenAPI 文档                 |
| MCP Server     | Model Context Protocol 服务器，可接入 Claude Desktop 等   |
| Docker         | 官方镜像，Core 和 Full 两个版本                           |

---

## 四、事实核查汇总

| # | 前一轮声明 | 核查结果 | 状态 |
|---|-----------|---------|------|
| 1 | 最新版本 v4.3.8 | **实际为 v4.4.1**（2026-02-28 发布） | **有误，已修正** |
| 2 | 6.3k GitHub Stars | 约 6,456（org 页面），基本一致 | 略有偏差 |
| 3 | 75+ 种文件格式 | 官方文档和 README 均声称 75+/76+ | 一致 |
| 4 | Rust Core + 多语言绑定架构 | 架构文档详细描述了模块划分 | 准确 |
| 5 | v3 依赖 Pandoc，v4 移除 | CHANGELOG 和迁移指南均确认 | 准确 |
| 6 | v4.3.0 移除 LibreOffice | CHANGELOG v4.3.0 明确记载 | 准确 |
| 7 | 10-50x 性能提升 | **来自项目方自报数据**（PyPI 描述），无独立基准测试佐证 | 未独立验证 |
| 8 | 包体积 16-31MB | release 页面 CLI 二进制约 13.7-15.5MB，项目方声称 Python wheel 22MB | 基本一致 |
| 9 | Docling Docker 镜像 9.74GB | 独立来源 shekhargulati.com 确认相同数字 | 准确 |
| 10 | Unstructured ~146MB minimal | 来自项目方对比，**非独立验证** | 未独立验证 |
| 11 | html-to-markdown 150-280 MB/s | **项目方自报数据** | 未独立验证 |
| 12 | html-to-markdown 使用 html5ever | CHANGELOG 早期用 scraper+html5ever，lib.rs 现称 astral-tl parser | **可能有变动** |
| 13 | K-means 聚类做 PDF 层级检测 | Features 文档详细描述了算法流程 | 准确 |
| 14 | MIT 许可证 | GitHub 和 PyPI 均确认 | 准确 |
| 15 | 作者 Na'aman Hirschfeld | PyPI verified publisher 确认 | 准确 |
| 16 | 支持 output_format="markdown" | Python API Reference 确认该参数 | 准确 |
| 17 | 批处理 4-6x 吞吐提升 | **来自 CHANGELOG 中项目方声明** | 未独立验证 |

---

## 五、Python 集成方式

### 5.1 作为库直接使用

```python
# 安装
pip install kreuzberg

# 基本用法（异步）
import asyncio
from kreuzberg import extract_file, ExtractionConfig

async def main():
    config = ExtractionConfig(
        output_format="markdown",          # 输出 Markdown
        enable_quality_processing=True,    # 启用质量处理
    )
    result = await extract_file("document.pdf", config=config)
    print(result.content)       # Markdown 内容
    for table in result.tables:
        print(table.markdown)   # 表格 Markdown

asyncio.run(main())

# 同步用法
from kreuzberg import extract_file_sync
result = extract_file_sync("document.pdf", config=config)
```

### 5.2 批量处理

```python
from kreuzberg import batch_extract_files, ExtractionConfig

config = ExtractionConfig(output_format="markdown")
results = await batch_extract_files(
    ["doc1.pdf", "doc2.docx", "doc3.xlsx"],
    config=config,
)
```

### 5.3 仅使用 HTML → Markdown 转换

```python
pip install html-to-markdown

from html_to_markdown import convert, ConversionOptions
markdown = convert("<h1>Hello</h1><p>World</p>")
# 也支持 Djot 输出
djot = convert(html, ConversionOptions(output_format="djot"))
```

### 5.4 自定义插件示例

```python
from kreuzberg import register_document_extractor, ExtractionResult

class CustomExtractor:
    def name(self) -> str:
        return "my_custom"

    def supported_mime_types(self) -> list[str]:
        return ["application/x-myformat"]

    def extract(self, data: bytes, mime_type: str, config) -> ExtractionResult:
        return ExtractionResult(content="extracted text", mime_type=mime_type)

register_document_extractor(CustomExtractor())
```

---

## 六、对你项目的集成建议

### 推荐方案：直接集成 Kreuzberg 作为提取引擎

**理由**：

1. `pip install kreuzberg` 即装即用，预编译 Rust 二进制无需额外编译环境
2. `output_format="markdown"` 一键输出 Markdown
3. 内建批量处理，适合"多文档转 MD"场景
4. 插件系统允许你对特殊格式注册自定义提取器
5. MIT 许可证无商业使用限制

**注意事项**：

- OCR 功能需要额外安装 Tesseract 或者使用 PaddleOCR
- 嵌入生成功能需要单独安装 ONNX Runtime 1.24+
- Python wheel 约 22MB，如对包体积有要求需评估
- 当前 Python 绑定要求 Python 3.10+（cp310-abi3）

### 可参考的设计模式

如果你选择自建而非直接集成，以下 Kreuzberg 的设计模式值得借鉴：

1. **MIME 检测 + Registry 分发**：用 extractor 注册表替代 if-else 链
2. **两阶段流水线**：先提取为结构化中间表示（content + tables + metadata），再根据 output_format 渲染
3. **HTML 作为中间格式**：Office 文档 → HTML → Markdown 是经验证的可靠路径
4. **表格独立处理**：将表格作为独立对象返回，应用专门的 Markdown 渲染策略

---

## 七、生态系统

| 项目 | 描述 | Stars |
|------|------|-------|
| kreuzberg | 核心文档智能框架 | ~6,456 |
| html-to-markdown | 高性能 HTML→MD 转换器 | ~548 |
| haystack-core-integrations | Haystack 集成（fork） | - |
| langchain-kreuzberg | LangChain 集成 | ~4 |
| pdfium-render | PDFium Rust 包装（fork） | ~2 |
| kreuzberg.cloud | 托管云服务（规划中） | - |

---

*报告生成于 2026-03-05，基于公开可访问的 GitHub、PyPI、官方文档及第三方来源。性能数据标注为"未独立验证"的部分均为项目方自报，建议在实际场景中自行 benchmark。*

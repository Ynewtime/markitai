# MarkItDown PDF 处理深度调研报告

## 1. 概述

MarkItDown 是 Microsoft 开源的 Python 工具库，用于将各种文件格式转换为 Markdown。该库已在 GitHub 上获得超过 83k 星标，是 Microsoft AutoGen 团队的作品。

**核心目标**：将 PDF、Office 文档、图片、音频等多种格式转换为 LLM 友好的 Markdown 文本。

## 2. PDF 处理机制详解

### 2.1 基础 PDF 转换 (pdfminer)

MarkItDown 默认使用 **pdfminer.six** 库处理 PDF 文件。

**源码位置**：`markitdown/converters/_pdf_converter.py`

```python
class PdfConverter(DocumentConverter):
    """
    Converts PDFs to Markdown. Most style information is ignored, 
    so the results are essentially plain-text.
    """
    
    def convert(self, file_stream, stream_info, **kwargs):
        return DocumentConverterResult(
            markdown=pdfminer.high_level.extract_text(file_stream),
        )
```

**特点**：
- ✅ 简单直接，只需 4 行代码即可使用
- ✅ 支持可选文本的 PDF
- ❌ **不包含内置 OCR**：无法处理扫描版/图片 PDF
- ❌ **格式丢失**：提取的文本失去所有格式信息（标题、列表、加粗等）
- ❌ **无法识别图片中的文字**

### 2.2 Azure Document Intelligence 增强转换

对于需要更高质量转换的场景，MarkItDown 集成了 **Azure Document Intelligence** 服务。

**源码位置**：`markitdown/converters/_doc_intel_converter.py`

```python
class DocumentIntelligenceConverter(DocumentConverter):
    def __init__(self, endpoint, api_version="2024-07-31-preview", credential=None, file_types=...):
        self.doc_intel_client = DocumentIntelligenceClient(
            endpoint=endpoint,
            api_version=api_version,
            credential=credential,
        )
    
    def convert(self, file_stream, stream_info, **kwargs):
        poller = self.doc_intel_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=file_stream.read()),
            features=self._analysis_features(stream_info),
            output_content_format="markdown",
        )
        result = poller.result()
        return DocumentConverterResult(markdown=result.content)
```

**支持的功能**：
| 功能 | 说明 |
|------|------|
| `FORMULAS` | 数学公式提取为 LaTeX |
| `OCR_HIGH_RESOLUTION` | 高分辨率 OCR |
| `STYLE_FONT` | 字体样式提取 |

**支持的文件类型**：
- **需要 OCR**：PDF, JPEG, PNG, BMP, TIFF
- **不需要 OCR**：DOCX, PPTX, XLSX, HTML

## 3. LLM 集成能力分析

### 3.1 图像描述功能

MarkItDown 可以使用 LLM（如 gpt-5.2）为图片生成描述。

**源码位置**：`markitdown/converters/_image_converter.py`

```python
def _get_llm_description(self, file_stream, stream_info, *, client, model, prompt=None):
    if prompt is None:
        prompt = "Write a detailed caption for this image."
    
    # 将图片转换为 base64
    base64_image = base64.b64encode(file_stream.read()).decode("utf-8")
    data_uri = f"data:{content_type};base64,{base64_image}"
    
    # 调用 OpenAI 兼容的 API
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
    }]
    
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
```

### 3.2 ⚠️ 关键限制：LLM 不能直接分析 PDF

**重要发现**：MarkItDown 的 LLM 功能**仅限于图片和 PPTX 文件中的图片**，**不支持直接用 LLM 分析 PDF**。

引用源代码注释：
> "To use Large Language Models for image descriptions (currently only for pptx and image files)"

**原因分析**：
1. PDF 文件不是图片格式，无法直接发送给视觉模型
2. pdfminer 只提取文本，不处理 PDF 中的图像
3. Azure Document Intelligence 是云端服务，不使用 LLM

## 4. 架构设计

### 4.1 转换器优先级机制

```python
PRIORITY_SPECIFIC_FILE_FORMAT = 0.0   # 特定格式转换器（高优先级）
PRIORITY_GENERIC_FILE_FORMAT = 10.0   # 通用格式转换器（低优先级）
```

**转换器注册顺序**（从低到高优先级）：
1. PlainTextConverter, ZipConverter, HtmlConverter (通用)
2. RssConverter, WikipediaConverter, YouTubeConverter (特定)
3. DocxConverter, XlsxConverter, PptxConverter (Office)
4. ImageConverter, AudioConverter (媒体)
5. **PdfConverter** (PDF)
6. **DocumentIntelligenceConverter** (如果配置了 endpoint，最高优先级)

### 4.2 文件类型检测

使用 **magika** 库进行文件类型检测：

```python
self._magika = magika.Magika()
result = self._magika.identify_stream(file_stream)
```

## 5. 使用方法

### 5.1 基础 PDF 转换

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")
print(result.text_content)
```

### 5.2 使用 Azure Document Intelligence

```python
from markitdown import MarkItDown

# 方式 1: 使用环境变量 AZURE_API_KEY
md = MarkItDown(docintel_endpoint="https://your-endpoint.cognitiveservices.azure.com/")

# 方式 2: 使用显式凭证
from azure.core.credentials import AzureKeyCredential
md = MarkItDown(
    docintel_endpoint="https://your-endpoint.cognitiveservices.azure.com/",
    docintel_credential=AzureKeyCredential("your-api-key")
)

result = md.convert("document.pdf")
print(result.text_content)
```

### 5.3 使用 LLM 描述图片（非 PDF）

```python
from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI(api_key="your-api-key")
md = MarkItDown(
    llm_client=client,
    llm_model="gpt-5.2",
    llm_prompt="请详细描述这张图片"  # 可选
)

# 仅对图片文件有效
result = md.convert("image.jpg")
print(result.text_content)
```

## 6. 局限性总结

| 局限性 | 说明 |
|--------|------|
| 无内置 OCR | 扫描版 PDF 需要预处理或使用 Azure Document Intelligence |
| 格式丢失 | pdfminer 提取的文本无法区分标题、列表等 |
| LLM 不支持 PDF | LLM 功能仅限于图片和 PPTX 中的图片 |
| 图片提取 | 无法提取 PDF 中嵌入的图片 |
| 表格处理 | 基础模式下表格结构可能丢失 |

## 7. 替代方案

如果需要更强的 PDF 处理能力，可以考虑：

| 工具 | 特点 |
|------|------|
| **PyMuPDF4LLM** | 专为 LLM 设计，保留更多格式 |
| **MinerU** (opendatalab) | 支持 OCR、公式、表格，可输出 Markdown/JSON |
| **Marker** | 使用深度学习，支持 LLM 增强 |
| **Azure Document Intelligence** | 云服务，高精度 OCR 和布局分析 |

## 8. 结论

**MarkItDown 适用场景**：
- 快速提取可选文本 PDF 的内容
- 批量转换多种格式文档为 Markdown
- 与 Azure Document Intelligence 集成获得高质量 OCR

**不适用场景**：
- 需要保留 PDF 格式信息
- 处理扫描版 PDF（除非使用 Azure）
- 需要用 LLM 直接分析 PDF 内容（不支持）

---

*调研日期：2026年1月7日*
*MarkItDown 版本：0.1.3*
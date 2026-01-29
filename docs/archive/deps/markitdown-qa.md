> Question: 已经 2026 了，什么 PDF 工具最好？PyMuPDF 是不是过时了

## 2025-2026 Python PDF 工具现状

**PyMuPDF 没有过时**，它仍然是速度最快、功能最全面的底层 PDF 库之一。但现在有了更多专门面向 LLM/RAG 场景的上层工具。

## 当前主流工具对比

| 工具 | 速度 | 表格 | OCR | Markdown | 适用场景 |
|------|------|------|-----|----------|----------|
| **pymupdf4llm** | 0.12s | ✅ 好 | ❌ | ✅ | LLM/RAG 首选，速度与质量平衡 |
| **Docling** (IBM) | 4s/页 | ✅ 最佳 | ✅ | ✅ | 复杂表格、企业级 |
| **unstructured** | 1.3s | ✅ | ✅ | ✅ | RAG 语义分块 |
| **Marker** | 慢 | ✅ | ✅ Surya | ✅ | 多语言、科学文档 |
| **MinerU** | 中 | ✅ | ✅ Paddle | ✅ | 中文/金融/学术 |
| **pypdfium2** | 0.003s | ❌ | ❌ | ❌ | 纯速度，简单文本 |
| **pdfplumber** | 0.1s | ✅ 精准 | ❌ | ❌ | 表格坐标提取 |

## 推荐选择

### 1. 通用 LLM/RAG 场景 → **pymupdf4llm**
```python
import pymupdf4llm

# 基础用法
md = pymupdf4llm.to_markdown("doc.pdf")

# 带图片提取
result = pymupdf4llm.to_markdown(
    "doc.pdf",
    page_chunks=True,
    write_images=True,
    image_path="images/",
    dpi=300
)
```

### 2. 复杂表格/高精度 → **Docling**
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("doc.pdf")
md = result.document.export_to_markdown()
```

### 3. 需要 OCR → **Marker** 或 **MinerU**
```bash
# Marker
pip install marker-pdf
marker_single doc.pdf output/ --output_format markdown

# MinerU (中文友好)
pip install magic-pdf
```

### 4. 底层操作/图片提取 → **PyMuPDF**
```python
import fitz

doc = fitz.open("doc.pdf")
for page in doc:
    # 提取图片
    for img in page.get_images():
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        pix.save(f"img_{xref}.png")
```

## 结论

对于大多数文档处理需求，pymupdf4llm 在速度和质量之间取得了最佳平衡。如果需要更好的表格处理，Docling 质量最高但速度慢很多。PyMuPDF 本身作为底层库依然是最强的，pymupdf4llm 就是基于它构建的上层封装。


> Question: markitdown 支持提取哪些格式文件中的图片，也即 --keep-data-uris 会对哪些格式生效？

根据搜索结果，`--keep-data-uris` 支持的格式如下：

## 支持提取嵌入图片的格式

| 格式 | 支持 | 说明 |
|------|------|------|
| DOCX | ✅ | 内部转 HTML 后提取，使用 mammoth 库 |
| PPTX | ✅ | 直接从 shape.image 提取 |
| HTML | ✅ | 直接处理 `data:image/` URI |
| XLSX | ✅ | 通过 HTML 中间格式 |
| EPUB | ✅ | 通过 HTML 中间格式 |

## 不支持的格式

| 格式 | 原因 |
|------|------|
| PDF | 使用 pdfminer，只提取文本，不处理嵌入图片 |
| 旧版 DOC/PPT/XLS | 不支持，仅支持 Office Open XML 格式 |

## 工作原理

```
DOCX/PPTX/XLSX/EPUB
       ↓
   转换为 HTML (mammoth/pptx/pandas)
       ↓
   图片变成 base64 data URI
       ↓
   --keep-data-uris 保留完整 URI
       ↓
   输出 Markdown: ![alt](data:image/png;base64,...)
```

## 提取图片的后处理示例

```python
import re, base64
from pathlib import Path

# markitdown --keep-data-uris doc.docx > output.md
md_content = Path("output.md").read_text()

pattern = r'!\[([^\]]*)\]\((data:image/([a-zA-Z0-9]+);base64,([A-Za-z0-9+/=]+))\)'
img_dir = Path("images")
img_dir.mkdir(exist_ok=True)

for i, match in enumerate(re.finditer(pattern, md_content)):
    alt, full_uri, ext, b64_data = match.groups()
    img_path = img_dir / f"image_{i}.{ext}"
    img_path.write_bytes(base64.b64decode(b64_data))
    md_content = md_content.replace(full_uri, str(img_path))

Path("output_with_images.md").write_text(md_content)
```

**如果需要从 PDF 提取图片**，需要用其他工具如 PyMuPDF (fitz) 单独处理。


> Question: markitdown 是否支持直接将图片写入磁盘

**原生 markitdown 不直接支持**将图片提取并写入磁盘。这是一个被请求的功能（GitHub Issue #56），但目前尚未实现。

## 现有的替代方案

1. **`--keep-data-uris` + 后处理**
   ```bash
   markitdown --keep-data-uris input.docx
   ```
   然后用正则提取 base64 数据并自行保存：
   ```python
   import re, base64
   pattern = r'data:image/([a-zA-Z0-9]+);base64,([a-zA-Z0-9+/]+={0,2})'
   # 匹配后解码写入文件
   ```

2. **使用第三方扩展 `markitdown-chunker`**
   ```python
   from markitdown_chunker import MarkdownConverter
   
   converter = MarkdownConverter()
   result = converter.convert(
       file_path="document.pdf",
       output_dir="output/",
       save_images=True  # 自动提取图片到 output/images/
   )
   ```

3. **针对 DOCX 结合 mammoth**（Discussion #269 中提到的方案）
   ```python
   import mammoth
   from pathlib import Path
   
   img_dir = Path('images')
   img_dir.mkdir(exist_ok=True)
   
   def convert_image(image):
       with image.open() as image_bytes:
           img_name = Path(image_bytes.name).name
           with open(img_dir / img_name, "wb") as f:
               f.write(image_bytes.read())
       return {"src": str(img_dir / img_name)}
   
   with open("doc.docx", "rb") as f:
       mammoth.convert_to_markdown(f, convert_image=mammoth.images.img_element(convert_image))
   ```

如果你的需求是批量提取文档中的图片，`markitdown-chunker` 是最省事的选择。


> Question: rapidocr-onnxruntime 依赖的 onnxruntime，或者 markitdown 的某些底层依赖 是否已经支持 Python 3.13

## 总结

| 包                      | Python 3.13 支持 | 说明                                                      |
|-------------------------|------------------|-----------------------------------------------------------|
| onnxruntime             | ✅ 已支持         | v1.23.2 已提供 cp313 wheel                          |
| markitdown              | ✅ 已支持         | 官方声明支持 Python 3.10-3.13                       |
| rapidocr-onnxruntime    | ❌ 不支持         | 明确限制 `Python <3.13, >=3.6`                      |

### rapidocr 的替代方案

1. **使用新包 `rapidocr`**（不是 `rapidocr-onnxruntime`）
   - 声明 `Python <4, >=3.6`，理论上兼容 3.13
   - 这是 RapidAI 的新统一包

2. **使用 `rapidocr-web`**
   - 明确列出支持 Python 3.13

3. **通过 nixpkgs 安装**
   - nixpkgs 中已有 python3.13-rapidocr-onnxruntime 包

### 建议

如果你使用 Python 3.13：

```bash
# markitdown 可以直接安装
pip install markitdown

# rapidocr 用新包代替旧包
pip install rapidocr  # 而不是 rapidocr-onnxruntime
```
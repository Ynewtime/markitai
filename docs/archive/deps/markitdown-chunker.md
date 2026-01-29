---
source: https://raw.githubusercontent.com/Naveenkumarar/markitdown-chunker/refs/heads/main/README.md
---

> Question: markitdown 是否支持直接将图片写入磁盘

## 分析总结

| 指标             | 情况                                                       |
|------------------|-----------------------------------------------------------|
| 版本             | v0.1.0 (2025年10月发布)                                    |
| Stars/Forks      | 0 / 0                                                      |
| Commits          | 仅 7 次                                                    |
| 维护者           | 单人 (Naveenkumarar)                                       |
| 依赖             | markitdown + langchain-text-splitters + PyMuPDF/docx/pptx |

## 核心功能拆解

这个库本质上是个"胶水层"，把几个成熟库粘在一起：

```
markitdown-chunker
├── 文档转换: markitdown.convert()           # 直接调用
├── 文本分块: LangChain splitters            # 直接调用
│   ├── MarkdownHeaderTextSplitter
│   └── RecursiveCharacterTextSplitter
├── 图片提取: PyMuPDF / python-docx / pptx   # 自己实现
└── JSON 导出: 简单序列化                     # 自己实现
```

## 建议：自己开发

**理由：**

1. **核心代码量很小** — 自己写大约 100-200 行就能覆盖全部功能
2. **无社区验证** — 零采用率意味着潜在 bug 无人发现
3. **依赖风险** — 单人维护的小项目容易被弃坑
4. **可控性** — 自己写可以精确适配你的需求（chunking 策略、元数据格式等）

**自己实现的核心代码示例：**

```python
from markitdown import MarkItDown
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def convert_and_chunk(file_path: str, chunk_size=1000, overlap=200):
    # 1. 转换
    md = MarkItDown()
    result = md.convert(file_path)
    
    # 2. 按标题分块
    headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    header_chunks = header_splitter.split_text(result.text_content)
    
    # 3. 再按大小分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap
    )
    final_chunks = text_splitter.split_documents(header_chunks)
    
    return final_chunks
```

**如果只是快速验证想法**，可以先用这个库跑通 POC，但生产环境建议自己封装。

---

# MarkitDown Chunker

A powerful Python package that converts documents to markdown, intelligently chunks them, and exports structured data. Built as an add-on to the [markitdown](https://github.com/microsoft/markitdown) package with advanced chunking capabilities using LangChain.

## ✨ Features

- 📄 **Multi-format Support**: Convert PDF, DOCX, PPTX, XLSX, HTML, RTF, ODT, and more to markdown
- 🖼️ **Image Extraction**: Automatically extract images from PDF, DOCX, PPTX files ([requires optional dependencies](docs/IMAGE_EXTRACTION.md))
- 🎨 **Image Summarization**: Optional AI-powered image descriptions for better context
- ✂️ **Smart Chunking**: Markdown-aware text splitting that respects document structure
- 📊 **Structured Export**: Export chunks with metadata to JSON format
- 🔧 **Flexible Pipeline**: Run individual steps or complete pipeline as needed
- 🎯 **CLI & Python API**: Use from command line or integrate into your Python applications

## 📦 Installation

### Basic Installation

```bash
pip install markitdown-chunker
```

### With Image Extraction Support

To extract images from PDF, DOCX, and PPTX files:

```bash
pip install "markitdown-chunker[images]"
```

See [Image Extraction Guide](docs/IMAGE_EXTRACTION.md) for details.

### From Source

```bash
git clone https://github.com/Naveenkumarar/markitdown-chunker.git
cd markitdown-chunker
pip install -e .
# Or with image support:
pip install -e ".[images]"
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Convert, chunk, and export (full pipeline)
markitdown-chunker input.pdf output/

# Convert only
markitdown-chunker document.docx output/ --convert-only

# Chunk existing markdown
markitdown-chunker document.md output/ --chunk-only

# Custom chunk size and overlap
markitdown-chunker input.pdf output/ --chunk-size 2000 --overlap 400

# List supported formats
markitdown-chunker --list-formats
```

### Python API

#### Complete Pipeline

```python
from markitdown_chunker import MarkitDownProcessor

# Initialize processor with custom settings
processor = MarkitDownProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    use_markdown_splitter=True
)

# Process a document (all steps)
result = processor.process(
    file_path="document.pdf",
    output_dir="output/"
)

print(f"Markdown saved to: {result['conversion']['markdown_path']}")
print(f"Created {len(result['chunking']['chunks'])} chunks")
print(f"JSON exported to: {result['export']['json_path']}")
```

#### Step-by-Step Processing

```python
from markitdown_chunker import MarkdownConverter, DocumentChunker, JSONExporter

# Step 1: Convert to Markdown
converter = MarkdownConverter()
conversion_result = converter.convert(
    file_path="document.pdf",
    output_dir="output/",
    save_images=True
)

# Step 2: Chunk the markdown
chunker = DocumentChunker(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = chunker.chunk_file(
    markdown_path=conversion_result['markdown_path']
)

# Step 3: Export to JSON
exporter = JSONExporter()
json_path = exporter.export(
    chunks=chunks,
    output_path="output/chunks.json"
)
```

## 📚 Supported File Formats

- **Documents**: PDF, DOCX, DOC, RTF, ODT, TXT, MD
- **Presentations**: PPTX, PPT, ODP
- **Spreadsheets**: XLSX, XLS, ODS
- **Web**: HTML, HTM

> **Note**: Audio/video files (MP3, MP4, etc.) require ffmpeg. See [docs/FFMPEG_AUDIO.md](docs/FFMPEG_AUDIO.md) for details.

## 📂 Output Directory Structure

After processing a document, the output directory will contain:

```
output/
├── document.md                    # Converted markdown file
├── document_chunks.json           # Chunks with metadata and statistics
└── images/                        # Extracted images (if any)
    ├── page1_img1.png
    ├── page2_img1.jpg
    ├── page3_img1.png
    └── page3_img2.jpg
```

### Example Output Files

**`document.md`** - Markdown conversion with image references:
```markdown
# Document Title

Document content converted to markdown format...

## Extracted Images

![Page 1 Image 1](images/page1_img1.png)

![Page 2 Image 1](images/page2_img1.jpg)
```

**`document_chunks.json`** - Structured chunk data:
```json
{
  "source_info": {
    "source_file": "document.pdf",
    "markdown_file": "output/document.md",
    "images_dir": "output/images"
  },
  "chunks": [
    {
      "text": "Document content chunk...",
      "metadata": {
        "Header 1": "Introduction",
        "chunk_index": 0,
        "source_file": "output/document.md"
      }
    }
  ],
  "total_chunks": 42,
  "statistics": {
    "total_characters": 48392,
    "avg_chunk_size": 1152.19,
    "min_chunk_size": 234,
    "max_chunk_size": 1000
  },
  "exported_at": "2025-10-10T10:30:45.123456"
}
```

**`images/`** - Extracted images with organized naming:
- PDF images: `page{N}_img{M}.{ext}` (e.g., `page1_img1.png`)
- DOCX images: `docx_img{N}.{ext}` (e.g., `docx_img1.jpg`)
- PPTX images: `slide{N}_img{M}.{ext}` (e.g., `slide1_img1.png`)

> 💡 **Tip**: The images directory is only created if the document contains images and `save_images=True` (default).

## 🎛️ Configuration Options

### Chunking Parameters

```python
processor = MarkitDownProcessor(
    chunk_size=1000,           # Maximum characters per chunk
    chunk_overlap=200,          # Overlap between consecutive chunks
    use_markdown_splitter=True, # Use markdown-aware splitting
    json_indent=2              # JSON formatting
)
```

### Processing Options

```python
result = processor.process(
    file_path="input.pdf",
    output_dir="output/",
    save_images=True,                    # Save extracted images
    include_image_summaries=False,       # Add image summaries to chunks
    image_summarizer=my_summarizer_func, # Custom image summarizer
    skip_conversion=False,               # Skip if already markdown
    skip_chunking=False,                 # Only convert
    skip_export=False                    # Don't export JSON
)
```

## 🔬 Advanced Usage

### Custom Image Summarization

```python
def summarize_image(image_path: str) -> str:
    """Your custom image summarization logic."""
    # Example: Use vision AI model
    from my_vision_model import analyze_image
    return analyze_image(image_path)

processor = MarkitDownProcessor()
result = processor.process(
    file_path="document.pdf",
    output_dir="output/",
    include_image_summaries=True,
    image_summarizer=summarize_image
)
```

### Batch Processing

```python
processor = MarkitDownProcessor()

files = ["doc1.pdf", "doc2.docx", "doc3.pptx"]
results = processor.process_batch(
    file_paths=files,
    output_dir="output/"
)

for result in results:
    if "error" in result:
        print(f"Failed: {result['input_file']} - {result['error']}")
    else:
        print(f"Success: {result['input_file']}")
```

### Individual Step Processing

```python
processor = MarkitDownProcessor()

# Only convert to markdown
conversion = processor.convert_only(
    file_path="document.pdf",
    output_dir="output/"
)

# Only chunk existing markdown
chunks = processor.chunk_only(
    markdown_path="document.md"
)

# Only export chunks
processor.export_only(
    chunks=chunks,
    output_path="output/chunks.json",
    source_info={"source": "document.md"}
)
```

### Custom Markdown Header Splitting

```python
from markitdown_chunker import DocumentChunker

chunker = DocumentChunker(
    chunk_size=1000,
    chunk_overlap=200,
    use_markdown_splitter=True,
    headers_to_split_on=[
        ("#", "Title"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Paragraph")
    ]
)

chunks = chunker.chunk_file("document.md")
```

## 📤 Output Format

### JSON Structure

```json
{
  "source_info": {
    "source_file": "document.pdf",
    "markdown_file": "output/document.md",
    "output_dir": "output/",
    "images_dir": "output/images"
  },
  "chunks": [
    {
      "text": "Chunk content here...",
      "metadata": {
        "Header 1": "Introduction",
        "Header 2": "Overview",
        "sub_chunk_index": 0,
        "total_sub_chunks": 1,
        "source_file": "output/document.md",
        "chunk_size_config": 1000,
        "chunk_overlap_config": 200
      }
    }
  ],
  "total_chunks": 42,
  "statistics": {
    "total_characters": 48392,
    "avg_chunk_size": 1152.19,
    "min_chunk_size": 234,
    "max_chunk_size": 1000
  },
  "exported_at": "2025-10-09T10:30:45.123456"
}
```

## 🛠️ CLI Reference

```bash
markitdown-chunker [-h] [--convert-only | --chunk-only | --no-export]
                    [--chunk-size CHUNK_SIZE] [--overlap OVERLAP]
                    [--no-markdown-splitter] [--no-images]
                    [--include-image-summaries] [--json-indent JSON_INDENT]
                    [--list-formats] [--version] [-v]
                    input output

Positional Arguments:
  input                 Input file path
  output                Output directory

Optional Arguments:
  -h, --help            Show help message
  --convert-only        Only convert to markdown
  --chunk-only          Only chunk existing markdown
  --no-export           Skip JSON export
  --chunk-size SIZE     Maximum chunk size (default: 1000)
  --overlap SIZE        Chunk overlap (default: 200)
  --no-markdown-splitter Disable markdown-aware splitting
  --no-images           Don't save extracted images
  --json-indent N       JSON indentation (default: 2)
  --list-formats        List supported formats
  --version             Show version
  -v, --verbose         Enable verbose output
```

## 🧪 Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/markitdown-chunker.git
cd markitdown-chunker
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
pytest --cov=markitdown_chunker tests/
```

### Code Formatting

```bash
black markitdown_chunker/
flake8 markitdown_chunker/
mypy markitdown_chunker/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [markitdown](https://github.com/microsoft/markitdown) by Microsoft
- Uses [LangChain](https://github.com/langchain-ai/langchain) text splitters for intelligent chunking

## 📞 Support

- 🐛 [Report a bug](https://github.com/Naveenkumarar/markitdown-chunker/issues)
- 💡 [Request a feature](https://github.com/Naveenkumarar/markitdown-chunker/issues)
- 📖 [Documentation](https://github.com/Naveenkumarar/markitdown-chunker)
- 🖼️ [Image Extraction Guide](docs/IMAGE_EXTRACTION.md)
- 🎵 [Audio/Video Processing Guide](docs/FFMPEG_AUDIO.md)

## 🗺️ Roadmap

- [ ] Support for more document formats
- [ ] Advanced chunking strategies (semantic, sentence-based)
- [ ] Integration with vector databases
- [ ] Web UI for document processing
- [ ] Cloud storage integration (S3, GCS, Azure)
- [ ] Parallel batch processing
- [ ] Custom output formats (CSV, Parquet, etc.)

---

Made with ❤️ by the MarkitDown Chunker community

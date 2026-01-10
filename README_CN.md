# MarkIt

文档转 Markdown 工具，支持 LLM 增强。

## 安装

```bash
# 使用 uv（推荐）
uv pip install markit

pip install markit
```

**系统依赖**（处理 .doc/.ppt/.xls）：

```bash
# Ubuntu/Debian
sudo apt install libreoffice-core

# macOS
brew install --cask libreoffice

# Windows
scoop install libreoffice
```

**可选**（更好性能）：

```bash
# Ubuntu/Debian
sudo apt install pandoc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install oxipng
# or: brew install pandoc oxipng

# macOS
brew install pandoc oxipng

# Windows
scoop install pandoc oxipng
#   或下载安装:
#   Pandoc: https://pandoc.org/installing.html
#   oxipng: https://github.com/shssoichiro/oxipng/releases
```

## 使用

```bash
# 基础转换
markit convert document.docx

# LLM 增强（格式清洗、添加 frontmatter）
markit convert document.docx --llm

# 图片分析
markit convert document.pdf --analyze-image

# 批量转换
markit batch ./docs -o ./output -r
```

## 命令

| 命令 | 说明 |
|------|------|
| `markit convert <file>` | 转换单个文件 |
| `markit batch <dir>` | 批量转换目录 |
| `markit config show` | 显示当前配置 |
| `markit provider add` | 添加 LLM 提供商凭证 |
| `markit provider test` | 测试 LLM 连接 |
| `markit provider list` | 列出已配置的凭证 |
| `markit provider fetch` | 获取提供商可用模型列表 |
| `markit model add` | 添加模型到配置 |
| `markit model list` | 列出已配置的模型 |

## 主要选项

| 选项 | 说明 |
|------|------|
| `-o, --output` | 输出目录 |
| `--llm` | 启用 LLM Markdown 增强 |
| `--analyze-image` | 用 LLM 生成图片 alt 文本 |
| `--analyze-image-with-md` | 同时生成 `.md` 描述文件 |
| `--pdf-engine` | PDF 引擎：pymupdf4llm, pymupdf, pdfplumber |
| `--llm-provider` | openai, anthropic, gemini, ollama, openrouter |
| `-r, --recursive` | 递归处理子目录（batch） |
| `--resume` | 恢复中断的批处理 |

## 支持格式

| 格式 | 扩展名 | 引擎 |
|------|--------|------|
| Word | .docx, .doc | MarkItDown（.doc 需 LibreOffice） |
| PowerPoint | .pptx, .ppt | MarkItDown（.ppt 需 LibreOffice） |
| Excel | .xlsx, .xls | MarkItDown（.xls 需 LibreOffice） |
| PDF | .pdf | PyMuPDF4LLM / PyMuPDF / pdfplumber |
| HTML | .html, .htm | MarkItDown |
| 图片 | .png, .jpg, .gif, .webp | LLM 分析 |

## 配置

直接运行 `markit config init` 创建 `markit.yaml`

```yaml
log_level: "INFO"

output:
  default_dir: "output"

image:
  enable_compression: true

# 新配置结构（推荐用于多模型场景）
# 将凭证和模型分开定义
llm:
  credentials:
    - id: "openai-main"
      provider: "openai"
      # api_key: "sk-..."  # 或使用 OPENAI_API_KEY 环境变量

  models:
    - name: "GPT-4o"
      model: "gpt-4o"
      credential_id: "openai-main"
      capabilities: ["text", "vision"]
```

**环境变量：**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## 输出结构

```
output/
  document.docx.md
  assets/
    document.docx.001.png
    document.docx.001.png.md    # 使用 --analyze-image-with-md 时
```

## 许可证

MIT

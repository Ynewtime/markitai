# Markit

文档转 Markdown 工具，原生支持 LLM 增强。

## 特性

- **多格式支持** - DOCX/DOC, PPTX/PPT, XLSX/XLS, PDF, TXT, MD, JPG/PNG/WebP
- **LLM 增强** - 格式清洗、元数据生成、图片分析
- **批量处理** - 并发转换、断点恢复
- **OCR 识别** - 扫描版 PDF 和图片文字提取

## 安装

```bash
# 需要 Python 3.13+
uv add markit
```

## 快速开始

```bash
# 基础转换
markit document.docx

# LLM 增强
markit document.docx --llm

# 使用预设
markit document.pdf --preset rich      # LLM + alt + desc + screenshot
markit document.pdf --preset standard  # LLM + alt + desc
markit document.pdf --preset minimal   # 仅基础转换

# 批量处理
markit ./docs -o ./output

# 断点恢复
markit ./docs -o ./output --resume
```

## 输出结构

```
output/
├── document.docx.md        # 基础 Markdown
├── document.docx.llm.md    # LLM 优化版
├── assets/
│   ├── document.docx.0001.jpg
│   └── assets.json         # 图片描述
```

## 配置

优先级：命令行 > 环境变量 > 配置文件 > 默认值

```bash
# 查看配置
markit config list

# 初始化配置文件
markit config init -o .
```

配置文件路径：`./markit.json` 或 `~/.markit/config.json`

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |

## 依赖

- [pymupdf4llm](https://github.com/pymupdf/RAG) - PDF 转换
- [markitdown](https://github.com/microsoft/markitdown) - Office 文档转换
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM 网关
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR 识别

## License

MIT

# Markitai

开箱即用的 Markdown 转换器，原生支持 LLM 增强。

## 特性

- **多格式支持** - DOCX/DOC, PPTX/PPT, XLSX/XLS, PDF, TXT, MD, JPG/PNG/WebP, URLs
- **LLM 增强** - 格式清洗、元数据生成、图片分析
- **批量处理** - 并发转换、断点恢复、进度显示
- **OCR 识别** - 扫描版 PDF 和图片文字提取
- **URL 转换** - 直接转换网页，支持 SPA 浏览器渲染

## 安装

```bash
# 需要 Python 3.11+
uv add markitai

# 或使用 pip
pip install markitai
```

## 快速开始

```bash
# 基础转换
markitai document.docx

# URL 转换
markitai https://example.com/article

# LLM 增强
markitai document.docx --llm

# 使用预设
markitai document.pdf --preset rich      # LLM + alt + desc + screenshot
markitai document.pdf --preset standard  # LLM + alt + desc
markitai document.pdf --preset minimal   # 仅基础转换

# 批量处理
markitai ./docs -o ./output

# 断点恢复
markitai ./docs -o ./output --resume

# URL 批量处理（自动识别 .urls 文件）
markitai urls.urls -o ./output
```

## 输出结构

```
output/
├── document.docx.md        # 基础 Markdown
├── document.docx.llm.md    # LLM 优化版
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # 图片描述
├── screenshots/            # 页面截图（--screenshot 时）
│   └── example_com.full.jpg
```

## 配置

优先级：命令行 > 环境变量 > 配置文件 > 默认值

```bash
# 查看配置
markitai config list

# 初始化配置文件
markitai config init -o .

# 查看缓存状态
markitai cache stats

# 清理缓存
markitai cache clear
```

配置文件路径：`./markitai.json` 或 `~/.markitai/config.json`

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `GEMINI_API_KEY` | Google Gemini API Key |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `JINA_API_KEY` | Jina Reader API Key（URL 转换） |

## 依赖

- [pymupdf4llm](https://github.com/pymupdf/RAG) - PDF 转换
- [markitdown](https://github.com/microsoft/markitdown) - Office 文档和 URL 转换
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM 网关
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR 识别

## 文档

- [快速开始](https://ynewtime.github.io/markitai/guide/getting-started)
- [配置说明](https://ynewtime.github.io/markitai/guide/configuration)
- [CLI 命令参考](https://ynewtime.github.io/markitai/guide/cli)

## License

MIT

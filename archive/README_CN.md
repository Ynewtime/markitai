# MarkIt

智能文档转 Markdown 工具，支持 LLM 增强。

## 目录

- [快速启动](#快速启动)
- [详细指导](#详细指导)
  - [安装](#安装)
  - [配置](#配置)
  - [命令参考](#命令参考)
  - [支持格式](#支持格式)
  - [LLM 配置](#llm-配置)
  - [常见问题](#常见问题)
- [开发](#开发)
- [许可证](#许可证)

---

## 快速启动

### 安装

```bash
# 使用 uv（推荐）
uv pip install markit

# 或使用 pip
pip install markit
```

**系统依赖**（用于旧版 Office 格式）：

```bash
# Ubuntu/Debian
sudo apt install libreoffice-core pandoc

# macOS
brew install --cask libreoffice
brew install pandoc

# Windows
# 从 https://www.libreoffice.org/ 下载安装 LibreOffice
# 从 https://pandoc.org/ 下载安装 Pandoc
```

### 基本用法

```bash
# 转换单个文件
markit convert document.docx

# 启用 LLM 增强
markit convert document.docx --llm

# 批量转换目录
markit batch ./documents -o ./output

# 递归转换子目录
markit batch ./documents -o ./output -r

# 显示帮助
markit -h
markit convert -h
markit batch -h
```

### 输出结构

转换后的文件以 `<原始文件名>.md` 格式保存：

```
input/
  report.docx
  data.xlsx

output/
  report.docx.md
  data.xlsx.md
  assets/
    image_001.png
    image_002.png
```

使用 `--analyze-image-with-md` 时，还会生成详细的图片描述文件：

```
output/
  report.pdf.md
  assets/
    image_001.png
    image_001.png.md    # 详细图片描述
    image_002.png
    image_002.png.md
```

---

## 详细指导

### 安装

#### Python 包

需要 Python 3.12+

```bash
# 使用 uv（推荐）
uv pip install markit

# 使用 pip
pip install markit

# 安装所有可选依赖
pip install "markit[all]"
```

#### 系统依赖

MarkIt 需要以下外部工具进行某些格式的转换：

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| LibreOffice | .doc, .ppt, .xls（旧格式） | 见下方 |
| Pandoc | 备用转换器 | 见下方 |

**Ubuntu/Debian：**

```bash
sudo apt update
sudo apt install -y libreoffice-core pandoc

# 图片压缩（可选）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install oxipng
```

**macOS：**

```bash
brew install --cask libreoffice
brew install pandoc oxipng
```

**Windows：**

1. 从 [LibreOffice 官网](https://www.libreoffice.org/download/download/) 下载安装
2. 从 [Pandoc 官网](https://pandoc.org/installing.html) 下载安装

#### 验证安装

```bash
# 检查 markit
markit --version

# 检查依赖
which soffice || where soffice
which pandoc || where pandoc
```

---

### 配置

MarkIt 可通过配置文件、环境变量或命令行参数进行配置。

**优先级（从高到低）：**
1. 命令行参数
2. 环境变量
3. 配置文件（`markit.toml`）
4. 默认值

#### 配置文件

在项目目录创建 `markit.toml`：

```bash
# 复制示例配置
cp markit.example.toml markit.toml
```

**基础配置：**

```toml
log_level = "INFO"

[output]
default_dir = "output"
on_conflict = "rename"

[image]
enable_compression = true
filter_small_images = true
```

**完整配置示例：** 参见 [markit.example.toml](markit.example.toml)

#### 环境变量

所有配置项都可通过 `MARKIT_` 前缀的环境变量覆盖：

```bash
# 日志
export MARKIT_LOG_LEVEL="DEBUG"
export MARKIT_LOG_FILE="logs/markit.log"

# LLM API 密钥
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."

# 并发设置
export MARKIT_CONCURRENCY__FILE_WORKERS=8
```

#### 日志文件

默认情况下，日志仅输出到控制台（stderr）。如需保存到文件：

```toml
# 在 markit.toml 中
log_file = "logs/markit.log"
```

或通过环境变量：

```bash
export MARKIT_LOG_FILE="logs/markit.log"
```

---

### 命令参考

#### `markit convert`

转换单个文档为 Markdown。

```bash
markit convert [选项] INPUT_FILE
```

**参数：**

| 参数 | 说明 |
|------|------|
| `INPUT_FILE` | 输入文档路径 |

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output PATH` | 输出目录 | `./output` |
| `--llm` | 启用 LLM Markdown 增强 | 禁用 |
| `--analyze-image` | 启用 LLM 图片分析（仅生成 alt 文本） | 禁用 |
| `--analyze-image-with-md` | 启用 LLM 图片分析并生成详细 `.md` 描述文件 | 禁用 |
| `--no-compress` | 禁用图片压缩 | 启用 |
| `--pdf-engine ENGINE` | PDF 引擎（pymupdf4llm, pymupdf, pdfplumber, markitdown） | `pymupdf4llm` |
| `--llm-provider PROVIDER` | LLM 提供商 | 第一个可用 |
| `--llm-model MODEL` | LLM 模型名称 | 提供商默认 |
| `-v, --verbose` | 详细输出 | 禁用 |
| `--dry-run` | 仅显示计划，不执行 | 禁用 |
| `-h, --help` | 显示帮助信息 | - |

**示例：**

```bash
# 基础转换
markit convert report.pdf

# 使用 LLM 增强
markit convert report.pdf --llm --llm-provider openai

# 启用图片分析（生成 alt 文本）
markit convert report.pdf --analyze-image

# 启用图片分析并生成详细描述文件
markit convert report.pdf --analyze-image-with-md

# 自定义输出目录
markit convert presentation.pptx -o ./converted

# 详细模式用于调试
markit convert document.docx -v
```

#### `markit batch`

批量转换目录中的文档。

```bash
markit batch [选项] INPUT_DIR
```

**参数：**

| 参数 | 说明 |
|------|------|
| `INPUT_DIR` | 输入目录路径 |

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output PATH` | 输出目录 | `INPUT_DIR/output` |
| `-r, --recursive` | 处理子目录 | 禁用 |
| `--include PATTERN` | 包含文件模式（glob） | 所有支持格式 |
| `--exclude PATTERN` | 排除文件模式（glob） | 无 |
| `--file-concurrency N` | 文件并发数 | 4 |
| `--image-concurrency N` | 图片并发数 | 8 |
| `--llm-concurrency N` | LLM 请求并发数 | 5 |
| `--on-conflict MODE` | 冲突处理（skip, overwrite, rename） | `rename` |
| `--resume` | 从上次中断处继续 | 禁用 |
| `--state-file PATH` | 状态文件路径 | `.markit-state.json` |
| `--llm` | 启用 LLM 增强 | 禁用 |
| `--analyze-image` | 启用图片分析（仅 alt 文本） | 禁用 |
| `--analyze-image-with-md` | 启用图片分析并生成详细 `.md` 描述文件 | 禁用 |
| `-v, --verbose` | 详细输出 | 禁用 |
| `--dry-run` | 仅显示计划，不执行 | 禁用 |
| `-h, --help` | 显示帮助信息 | - |

**示例：**

```bash
# 基础批量转换
markit batch ./documents

# 递归转换并指定输出目录
markit batch ./docs -o ./markdown -r

# 按文件类型过滤
markit batch ./docs --include "*.docx" --exclude "*draft*"

# 启用 LLM 增强和详细图片描述
markit batch ./docs --llm --analyze-image-with-md

# 恢复中断的批处理
markit batch ./docs --resume

# 高并发
markit batch ./docs --file-concurrency 8 --image-concurrency 16
```

#### `markit config`

配置管理命令。

```bash
markit config show    # 显示当前配置
markit config init    # 初始化配置文件
markit config validate  # 验证配置
```

---

### 支持格式

| 格式 | 扩展名 | 主引擎 | 备注 |
|------|--------|--------|------|
| Word | .docx | MarkItDown | 完整支持 |
| Word（旧版） | .doc | LibreOffice + MarkItDown | 需要 LibreOffice |
| PowerPoint | .pptx | MarkItDown | 完整支持 |
| PowerPoint（旧版） | .ppt | LibreOffice + MarkItDown | 需要 LibreOffice |
| Excel | .xlsx | MarkItDown | 保留表格 |
| Excel（旧版） | .xls | LibreOffice + MarkItDown | 需要 LibreOffice |
| PDF | .pdf | PyMuPDF / pdfplumber | 可配置引擎 |
| CSV | .csv | MarkItDown | 转换为表格 |
| HTML | .html, .htm | MarkItDown | 完整支持 |
| 文本 | .txt | 直接读取 | 直通 |
| 图片 | .png, .jpg, .gif, .webp, .bmp | LLM 分析 | 需 --analyze-image |

### 图片分析选项

MarkIt 提供两种级别的 LLM 图片分析：

#### `--analyze-image`（仅 Alt 文本）

为每张提取的图片生成简洁的 alt 文本，嵌入到 Markdown 中：

```markdown
![展示数据处理流程的流程图](assets/diagram_001.png)
```

#### `--analyze-image-with-md`（详细描述）

除了 alt 文本外，还会在 `assets/` 目录中为每张图片生成详细的 `.md` 描述文件：

```
output/
  document.pdf.md
  assets/
    diagram_001.png
    diagram_001.png.md    # 详细描述文件
```

描述文件包含：

```markdown
---
source_image: diagram_001.png
image_type: diagram
generated_at: 2026-01-08T12:00:00Z
---

# Image Description

## Alt Text

展示数据处理流程的流程图。

## Detailed Description

该图展示了一个三阶段的数据处理流程...

## Detected Text

"输入" -> "处理" -> "输出"
```

**注意：** 两个选项每张图片只调用一次 LLM。使用 `--analyze-image-with-md` 时会自动启用 `--analyze-image` 的行为。

---

### LLM 配置

MarkIt 支持多个 LLM 提供商用于 Markdown 增强和图片分析。

#### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
markit convert doc.pdf --llm --llm-provider openai --llm-model gpt-5.2
```

#### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="..."
markit convert doc.pdf --llm --llm-provider anthropic --llm-model claude-sonnet-4-5
```

#### Google Gemini

```bash
export GOOGLE_API_KEY="..."
markit convert doc.pdf --llm --llm-provider gemini --llm-model gemini-3-flash-preview
```

#### Ollama（本地）

```bash
# 启动 Ollama 服务
ollama serve

# 拉取模型
ollama pull llama3.2-vision

# 使用 markit
markit convert doc.pdf --llm --llm-provider ollama --llm-model llama3.2-vision
```

#### OpenRouter

```bash
export OPENROUTER_API_KEY="..."
markit convert doc.pdf --llm --llm-provider openrouter --llm-model google/gemini-3-flash-preview
```

#### 配置文件方式

在 `markit.toml` 中配置多个提供商：

```toml
[[llm.providers]]
provider = "openai"
model = "gpt-5.2"

[[llm.providers]]
provider = "anthropic"
model = "claude-sonnet-4-5"

[[llm.providers]]
provider = "ollama"
model = "llama3.2-vision"
base_url = "http://localhost:11434"
```

系统会使用第一个可用的提供商。如果某个提供商失败，会自动尝试下一个。

---

### 常见问题

#### LibreOffice 转换失败

**现象：** 旧格式（.doc, .ppt, .xls）转换失败，显示 "LibreOffice error" 错误。

**解决方案：**

1. 确认 LibreOffice 已安装：
   ```bash
   which soffice  # Linux/macOS
   where soffice  # Windows
   ```

2. 手动测试 LibreOffice：
   ```bash
   soffice --headless --convert-to docx --outdir /tmp test.doc
   ```

3. 检查 LibreOffice 版本（推荐 4.0+）：
   ```bash
   soffice --version
   ```

#### 并发转换不稳定

**现象：** 批量转换 .doc, .ppt, .xls 文件时间歇性失败。

**解决方案：** MarkIt 使用隔离的用户配置目录来避免 LibreOffice 锁冲突。此外，v1.0.1+ 修复了 LLM Provider 初始化过程中的竞态条件问题，该问题可能导致并发处理时出现冗余 API 调用。请确保使用最新版本。

#### LLM 连接错误

**现象：** "LLM provider not available" 或超时错误。

**解决方案：**

1. 确认 API 密钥已设置：
   ```bash
   echo $OPENAI_API_KEY
   ```

2. 测试连通性：
   ```bash
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

3. 检查防火墙/代理设置。

#### 内存不足

**现象：** 转换大文件时进程被终止。

**解决方案：**

1. 降低并发数：
   ```bash
   markit batch ./docs --file-concurrency 2 --image-concurrency 4
   ```

2. 单独处理文件。

#### 图片提取问题

**现象：** 转换后的输出中缺少图片。

**解决方案：**

1. 检查 `markit.toml` 中的图片过滤设置：
   ```toml
   [image]
   filter_small_images = false  # 禁用以保留所有图片
   ```

2. 确认 PDF 引擎支持图片提取：
   ```bash
   markit convert doc.pdf --pdf-engine pymupdf4llm
   ```

#### AutoShape/VML 图形无法转换

**现象：** Word 文档（.docx）或 PowerPoint 文件（.pptx）中的形状或图表未出现在 Markdown 输出中。

**说明：** MarkItDown（底层转换库）不支持 AutoShape/VML/DrawingML 图形。这是该库的已知限制。

**MarkIt 的处理方式：**
- 自动检测源文档中的 AutoShape
- 在 Markdown 文件末尾添加提示说明，表示发现了 AutoShape
- 提示用户参考原始源文件

**解决方法：** 对于包含重要图表的文档，请手动将其导出为图片并包含在 Markdown 中。

#### PowerPoint 页眉页脚出现在输出中

**现象：** 转换后的 Markdown 每页幻灯片都出现重复的页脚文本（公司名称、日期、页码等）。

**解决方案：**

1. **自动过滤（默认）：** MarkIt 会自动检测并移除 PPTX 文件中的页眉/页脚占位符。

2. **LLM 智能清理：** 使用 `--llm` 参数获得更智能的过滤效果：
   ```bash
   markit convert presentation.pptx --llm
   ```
   LLM 可以更准确地识别和移除重复的页脚内容。

#### LLM 初始化缓慢或出现冗余 API 调用

**现象：** 日志中出现多次 "Provider initialized successfully" 消息，或批处理期间出现大量并发 `/models` 端点请求。

**说明：** 这是 v1.0.1 之前版本的竞态条件 Bug，并发的 LLM 任务会各自触发 Provider 初始化。

**解决方案：** 升级到 v1.0.1+，该版本实现了正确的异步锁定机制，确保即使在高并发情况下 Provider 也只初始化一次。

---

## 开发

```bash
# 克隆仓库
git clone https://github.com/Ynewtime/markit
cd markit

# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=markit

# 运行代码检查
ruff check .

# 运行类型检查
mypy markit

# 格式化代码
ruff format .
```

---

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

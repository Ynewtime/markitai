# Markitai 用户旅程文档

> 本文档从用户实际使用视角出发，完整描述与 Markitai 交互的全流程体验。

---

## 目录

1. [初识与安装](#1-初识与安装)
2. [首次配置](#2-首次配置)
3. [基础转换：单文件](#3-基础转换单文件)
4. [URL 抓取与转换](#4-url-抓取与转换)
5. [管道集成：stdout 模式](#5-管道集成stdout-模式)
6. [LLM 增强处理](#6-llm-增强处理)
7. [图片处理与分析](#7-图片处理与分析)
8. [批量处理](#8-批量处理)
9. [高级功能](#9-高级功能)
10. [问题排查与维护](#10-问题排查与维护)
11. [典型工作流场景](#11-典型工作流场景)

---

## 1. 初识与安装

### 1.1 用户画像

Markitai 面向以下典型用户：

- **知识工作者**：需要将 PDF、Office 文档批量转为 Markdown 以便归档或检索
- **开发者**：在 CLI 管道中集成文档转换，或需要抓取网页内容
- **内容创作者**：利用 LLM 增强文档质量，自动生成摘要、标签和图片描述
- **研究人员**：批量抓取和处理在线文献资料

### 1.2 安装体验

用户通过一行命令完成安装：

```bash
# Linux / macOS
curl -fsSL https://markitai.dev/setup.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

或通过包管理器手动安装：

```bash
uv pip install markitai

# 安装全部可选依赖
uv pip install "markitai[all]"
```

**可选依赖组（按需安装）：**

| 依赖组 | 用途 | 典型场景 |
|--------|------|----------|
| `claude-agent` | Claude AI 支持 | 使用 Claude 订阅（Claude Code CLI）做 LLM 增强 |
| `copilot` | GitHub Copilot | 基于订阅的本地 LLM 增强 |
| `browser` | Playwright 浏览器 | 抓取 JS 渲染的网页 |
| `extra-fetch` | curl-cffi | 绕过 TLS 指纹检测，增强 HTTP 能力 |
| `kreuzberg` | 额外格式支持 | 扩展 XML/TSV/RTF/RST/ORG/TEX/ODT/ODS 等文件类型 |
| `svg` | cairosvg | SVG 文件高质量渲染 |
| `heif` | pillow-heif | HEIC/HEIF/AVIF 图片输入解码 |

**安装后验证：**

```bash
markitai --version
# markitai 0.18.0（版本号请以实际安装结果为准）
```

---

## 2. 首次配置

### 2.1 交互式初始化

首次使用时，用户运行 `init` 命令进入引导式配置：

```bash
markitai init
```

系统会自动完成以下步骤：
1. 检查系统依赖（Playwright、LibreOffice、FFmpeg、RapidOCR）
2. 检测可用的 LLM 提供商（Claude CLI、Copilot CLI、ChatGPT，以及 DeepSeek/Gemini/OpenAI/Anthropic/OpenRouter API Key）
3. 让用户选择配置文件保存位置（全局 `~/.markitai/config.json` 或本地 `./markitai.json`）
4. 写入配置文件

快速模式跳过交互提示，自动检测并生成配置：

```bash
markitai init -y
```

项目级配置可在当前目录生成：

```bash
markitai init --local
# 生成 ./markitai.json
```

### 2.2 配置优先级

这里有两条容易混淆的优先级链条：

**配置文件的查找顺序**（决定加载哪个文件）：

```
--config / -c 参数指定路径 > MARKITAI_CONFIG 环境变量 > ./markitai.json（项目级） > ~/.markitai/config.json（用户级）
```

**具体设置值的生效顺序**（决定同一个设置项谁说了算）：

```
命令行参数 > 环境变量（如 OPENAI_API_KEY） > 配置文件里的值 > 内置默认值
```

这意味着用户可以在全局配置中设置常用默认值，在项目配置中覆盖特定行为，并通过命令行参数做临时调整。

### 2.3 环境变量

常用环境变量：

```bash
export MODEL="anthropic/claude-sonnet-4-6"  # 指定 LLM 模型（需带 provider/ 前缀）
export MARKITAI_PURE=1                      # 启用纯净模式
export ANTHROPIC_API_KEY="sk-..."           # Anthropic 认证
export OPENAI_API_KEY="sk-..."              # OpenAI 认证
```

### 2.4 健康检查

配置完成后，用户可运行诊断命令验证环境：

```bash
markitai doctor
```

系统检查内容：
- 核心依赖是否就绪
- 可选依赖安装状态
- LLM 提供商认证是否有效
- 系统权限和路径

---

## 3. 基础转换：单文件

### 3.1 最简用法

将文件转换并输出到指定目录：

```bash
markitai report.pdf -o ./output
```

**用户看到的交互过程：**

1. 终端显示进度信息（文件检测、格式识别）
2. 转换引擎处理文件
3. 如果有嵌入图片，自动提取并存储到 `output/.markitai/assets/`
4. 生成 `output/report.pdf.md`（在完整输入文件名后追加 `.md`，保留原始扩展名，不同输入不会重名）

**输出文件结构：**

```
output/
├── report.pdf.md                # 转换后的 Markdown
└── .markitai/
    └── assets/
        ├── report.pdf.0001.jpg  # 提取的图片（命名为 <源文件名>.<4位序号>.<扩展名>）
        └── report.pdf.0002.jpg
```

### 3.2 支持的文件格式

内置转换器覆盖以下格式，无需额外配置：

| 类别 | 格式 |
|------|------|
| 文档 | PDF, DOCX, DOC, EPUB |
| 演示文稿 | PPTX, PPT |
| 电子表格 | XLSX, XLS, CSV, NUMBERS |
| 图片 | JPEG, PNG, WEBP, GIF, BMP, TIFF, SVG，以及 HEIC/HEIF/AVIF（需要 `markitai[heif]`） |
| 网页 | HTML, HTM, XHTML |
| 文本 | TXT, MD |
| 笔记本 | IPYNB (Jupyter) |
| 邮件 | EML, MSG |

以下格式需要额外安装 `uv pip install markitai[kreuzberg]`（`markitai[all]` 已包含）才能转换，否则会报"没有可用转换器"：

| 类别 | 格式 |
|------|------|
| 需要 kreuzberg | RTF, ODT, TSV, ODS, RST, ORG, TEX, XML |

### 3.3 输出行为

**带 `-o` 参数**：写入文件到指定目录

```bash
markitai document.docx -o ./converted
# → ./converted/document.docx.md
```

**不带 `-o` 参数**：内容输出到 stdout（详见第 5 节）

```bash
markitai document.docx
# Markdown 内容直接打印到终端
```

### 3.4 生成的 Markdown 结构

默认情况下，基础 `.md` 文件包含 basic frontmatter（`title`、`source`、`markitai_processed`）：

```markdown
---
title: 季度财务报告
source: report.pdf
markitai_processed: "2026-03-15T10:30:00"
---

# 季度财务报告

正文内容...

![图表](.markitai/assets/report.pdf.0001.jpg)
```

LLM 增强模式生成的 `.llm.md` 文件包含更丰富的 frontmatter（额外包含 `description`、`tags`），URL 转换还会包含 `fetch_strategy`。

> 注意：`--pure` 无论是否同时启用 `--llm`，都会跳过 frontmatter 生成——`--pure` 单独使用时直接输出无 frontmatter 的原始转换正文；`--pure --llm` 会把内容送入 LLM 做文本清洗，但同样不生成 frontmatter/描述/标签等元数据，只返回清洗后的正文。

```bash
markitai report.pdf -o ./output --pure
```

### 3.5 冲突处理

当输出目录已存在同名文件时，默认行为是重命名（添加版本号）：

```
output/report.pdf.md      # 已存在
output/report.pdf.v2.md   # 新生成
output/report.pdf.v3.md   # 再次生成
```

可通过配置更改策略：
- `skip`：跳过不覆盖
- `overwrite`：直接覆盖
- `rename`：自动重命名（默认）

---

## 4. URL 抓取与转换

### 4.1 单 URL 转换

```bash
markitai https://example.com/article -o ./output
```

**用户体验流程：**

1. 系统自动选择最佳抓取策略
2. 下载页面内容和关联图片
3. 清理 HTML，提取正文
4. 转换为结构化 Markdown
5. 保存到输出目录

### 4.2 抓取策略选择

系统默认使用 `auto` 策略（本地优先：static → playwright → defuddle → jina → cloudflare，已知 SPA/重 JS 域名则改为 playwright 优先），自动选择最优方案并在失败时回退。用户也可用 `-s/--strategy` 强制指定（`--static`/`--defuddle`/`--playwright`/`--jina`/`--cloudflare` 仍可用，但已弃用，会打印迁移提示）：

```bash
# 静态 HTTP 抓取（最快，适合简单页面）
markitai https://blog.example.com -s static -o ./output

# Defuddle API（最佳内容清洗，无需认证）
markitai https://news.example.com -s defuddle -o ./output

# Playwright 浏览器渲染（完整 JS 支持，适合 SPA）
markitai https://app.example.com -s playwright -o ./output

# Jina Reader API（云端处理）
markitai https://example.com -s jina -o ./output

# Cloudflare Browser Rendering
markitai https://example.com -s cloudflare -o ./output
```

**策略选择建议：**

| 场景 | 推荐策略 |
|------|----------|
| 静态博客、文档站 | `-s static` 或 `auto` |
| 新闻网站、内容站 | `-s defuddle` |
| 需要登录或 JS 渲染、SPA | `-s playwright` |
| 基础限流/常规反爬 | `-s cloudflare` |
| x.com/twitter.com 等**强反爬**站点 | `-s playwright` 或 `-s jina`（Cloudflare Browser Rendering 对这类站点常返回 400，详见 website/guide/configuration.md 的 Cloudflare Settings 一节） |
| 简单场景 | `auto`（默认） |

### 4.3 批量 URL 处理

创建 `.urls` 文件列出多个 URL：

```
# my-articles.urls
# 每行一个 URL，可选空格后跟自定义输出文件名
https://example.com/article-1
https://example.com/article-2 custom-name
https://example.com/article-3

# 空行和 # 开头的注释会被忽略
# 也支持 JSON 数组格式（见配置文档）
```

> 注意：URL 后的可选字段是输出文件名（output_name），不是标题。用空格分隔，不要使用管道符 `|`。

执行批量抓取：

```bash
markitai my-articles.urls -o ./articles
```

**用户看到的过程：**

1. 解析 `.urls` 文件，识别有效条目
2. 并发抓取（默认 5 个并发，可通过 `--url-concurrency` 调整）
3. 逐个显示转换进度和结果
4. 汇总报告成功/失败数量

---

## 5. 管道集成：stdout 模式

### 5.1 基本用法

不指定 `-o` 参数时，转换结果直接输出到 stdout：

```bash
markitai document.pdf
```

这使 Markitai 可以无缝融入 Unix 管道：

```bash
# 转换后搜索关键词
markitai report.pdf | grep "revenue"

# 转换后传入其他工具
markitai article.html | wc -w

# 重定向到文件
markitai notes.docx > notes.md
```

### 5.2 图片处理（stdout 模式）

stdout 模式下图片处理采用三层优先级策略：

**第一优先级 — 终端内联显示：**
如果检测到终端支持（Kitty 或 iTerm2 协议），图片会直接在终端中内联渲染：

```bash
# 在 Kitty 终端中
markitai article-with-images.pdf
# 图片直接显示在 Markdown 文本中间
```

**第二优先级 — 持久化资源存储：**
`image.stdout_persist` 默认就是 `true`，图片会保存到 `~/.markitai/assets/` 并生成 `file:///` URI 引用（设为 `false` 可关闭）：

```bash
markitai article.pdf
# 图片引用：![图表](file:///home/user/.markitai/assets/article.pdf.0001.jpg)
```

**第三优先级 — 占位符：**
默认回退方案，生成纯文本占位符：

```markdown
![image: chart.png]()
```

### 5.3 静默模式

stdout 模式下会自动进入静默控制台模式，控制台日志仅保留 ERROR 级别，确保管道输出干净：

```bash
markitai document.pdf | process_markdown
```

---

## 6. LLM 增强处理

### 6.1 启用 LLM

```bash
markitai document.pdf --llm -o ./output
```

或使用预设：

```bash
markitai document.pdf --preset standard -o ./output
```

### 6.2 预设系统

三个内置预设简化了功能组合：

| 预设 | 包含功能 | 适用场景 |
|------|----------|----------|
| `minimal` | 无增强 | 快速转换、大批量处理 |
| `standard` | LLM + alt 文本 + 图片描述 | 日常使用 |
| `rich` | LLM + alt 文本 + 图片描述 + 截图 | 最高质量输出 |

### 6.3 LLM 增强的效果

**原始转换结果（基础 `.md`，无 LLM）：**

```markdown
---
title: Q3 Financial Results
source: report.pdf
markitai_processed: "2026-03-15T10:30:00"
---

Q3 Financial Results
Revenue was $5.2M, up 23% YoY. Costs...
```

> 基础 `.md` 默认包含 basic frontmatter（`title`、`source`、`markitai_processed`）。如需完全跳过 frontmatter，使用 `--pure`。

**LLM 增强后（`.llm.md`）：**

```markdown
---
title: 2026年第三季度财务报告
source: report.pdf
description: 本报告概述了公司第三季度的财务表现，包括营收增长、成本控制和未来展望。
tags:
  - 财务报告
  - 季度总结
  - 营收分析
markitai_processed: "2026-03-15T10:30:00"
---

# 2026年第三季度财务报告

第三季度营收达到 520 万美元，同比增长 23%。成本...
```

### 6.4 双文件模式

启用 LLM 时，默认只生成增强后的 `.llm.md` 文件。使用 `--keep-base` 同时保留原始转换结果：

```bash
markitai report.pdf --llm --keep-base -o ./output
```

输出：
```
output/
├── report.pdf.md           # 原始转换
└── report.pdf.llm.md       # LLM 增强版
```

### 6.5 纯净模式

`--pure` 模式下 LLM 只做文本清洗，不生成 frontmatter，不做后处理：

```bash
markitai messy-doc.html --llm --pure -o ./output
```

适用于只需要 LLM 修正格式、清理噪音但不要额外元数据的场景。

### 6.6 缓存机制

LLM 调用结果会被自动缓存，重复处理同一文件时直接使用缓存结果：

```bash
# 首次处理（调用 LLM）
markitai report.pdf --llm -o ./output

# 再次处理（使用缓存，更快）
markitai report.pdf --llm -o ./output

# 跳过缓存强制重新处理
markitai report.pdf --llm --no-cache -o ./output

# 对特定文件/模式跳过缓存
markitai docs/ --llm --no-cache-for "*.pdf,report*" -o ./output
```

---

## 7. 图片处理与分析

### 7.1 Alt 文本生成

为文档中的图片自动生成描述性 alt 文本（需 LLM）：

```bash
markitai presentation.pptx --llm --alt -o ./output
```

效果：
```markdown
<!-- 处理前 -->
![](.markitai/assets/chart.png)

<!-- 处理后 -->
![2026年Q3营收趋势柱状图，显示逐月增长](.markitai/assets/chart.png)
```

### 7.2 图片描述文件

生成详细的图片元数据 JSON 文件（需 LLM 视觉能力）：

```bash
markitai document.pdf --llm --desc -o ./output
```

在 `.markitai/assets/` 目录下生成 `images.json`（使用 `--llm --desc` 时默认只写 `.llm.md`）：

```
output/
├── document.pdf.llm.md
└── .markitai/
    └── assets/
        ├── document.pdf.0001.jpg
        └── images.json          # 图片元数据
```

`images.json` 示例：
```json
{
  "version": "1.0",
  "created": "2026-03-15T10:30:00",
  "updated": "2026-03-15T10:30:00",
  "images": [
    {
      "path": "/absolute/path/to/.markitai/assets/document.pdf.0001.jpg",
      "alt": "2026年Q1-Q3月度营收趋势图",
      "desc": "营收趋势柱状图，显示2026年Q1至Q3的月度营收数据...",
      "text": "",
      "source": "/absolute/path/to/document.pdf",
      "created": "2026-03-15T10:30:00"
    }
  ]
}
```

### 7.3 图片压缩

默认启用图片压缩，减小输出体积：

```bash
# 禁用压缩（保持原始质量）
markitai document.pdf --no-compress -o ./output
```

可配置参数：
- 压缩质量（1-100，默认 75）
- 输出格式（jpeg / png / webp）
- 最大尺寸（默认宽 1920px）

### 7.4 图片过滤

自动过滤掉尺寸过小的装饰性图片（图标、分隔线等）：

- 最小宽度：50px
- 最小高度：50px
- 最小面积：5000px²
- 自动去重（基于内容哈希）

### 7.5 截图功能

为文档生成页面截图：

```bash
# PDF → 每页生成 JPEG 截图
markitai report.pdf --screenshot -o ./output

# URL → 全页面截图
markitai https://example.com --screenshot -o ./output

# 仅截图不转换（配合或不配合 LLM）
markitai presentation.pptx --screenshot-only -o ./output
```

---

## 8. 批量处理

### 8.1 目录批量转换

```bash
markitai ./documents/ -o ./converted
```

**用户体验流程：**

1. 递归扫描目录（默认深度 5 层）
2. 显示发现的文件数量和格式统计
3. 并发处理文件（默认 10 并发）
4. 实时显示进度（已完成/总数、成功/失败）
5. 输出保留原始目录结构

**输出示例：**
```
converted/
├── reports/
│   ├── q1-report.pdf.md
│   └── q2-report.pdf.md
├── presentations/
│   └── kickoff.pptx.md
└── notes.txt.md
```

### 8.2 文件过滤

使用 glob 模式筛选要处理的文件：

```bash
# 只处理 PDF 文件
markitai ./docs/ -g "*.pdf" -o ./output

# 处理多种格式
markitai ./docs/ -g "*.pdf" -g "*.docx" -o ./output

# 排除特定模式（! 前缀）
markitai ./docs/ -g "*.pdf" -g "!*draft*" -o ./output
```

### 8.3 深度控制

```bash
# 只处理当前目录（不递归）
markitai ./docs/ --max-depth 0 -o ./output

# 最多递归 2 层
markitai ./docs/ --max-depth 2 -o ./output
```

### 8.4 并发控制

```bash
# 文件处理并发数
markitai ./docs/ -j 20 -o ./output

# LLM 请求并发数
markitai ./docs/ --llm --llm-concurrency 5 -o ./output

# URL 抓取并发数
markitai urls.urls --url-concurrency 10 -o ./output
```

### 8.5 断点续传

大批量处理时支持中断后恢复：

```bash
# 开始批量处理
markitai ./large-dataset/ --llm -o ./output
# 按 Ctrl+C 中断...

# 从上次中断处继续
markitai ./large-dataset/ --llm --resume -o ./output
```

系统通过 `.markitai/states/markitai.<hash>.state.json` 跟踪每个文件的处理状态（pending → in_progress → completed/failed）。续传时：已完成的文件直接跳过；上次中断时处于 in_progress、或标记为 failed 的文件会重新入队；本次新发现的文件/URL 也会一并纳入——终端会显示 `Resuming batch: N completed, M remaining`。`.urls` 批量 URL 输入同样支持 `--resume`，逻辑与文件批量对称。

> 注意：`--resume` 仅对批量（目录或 `.urls`）输入生效；单个文件/URL 转换会忽略该参数。

### 8.6 预览模式

在正式处理前预览将要执行的操作：

```bash
markitai ./docs/ --dry-run -o ./output
```

显示：
- 将要处理的文件列表
- 检测到的格式
- 预计输出路径
- 不实际写入任何文件

---

## 9. 高级功能

### 9.1 认证管理

认证命令按提供商分组，语法为 `markitai auth <provider> <action>`：

```bash
# 查看特定提供商认证状态
markitai auth claude status
markitai auth copilot status
markitai auth chatgpt status

# 登录特定提供商
markitai auth claude login
markitai auth copilot login

# JSON 格式输出（便于脚本集成）
markitai auth claude status --json
```

> 支持的提供商：`claude`、`copilot`、`chatgpt`。每个提供商支持 `status` 和 `login` 子命令。Gemini 通过直连 API Key 或 OpenRouter 接入，不走本命令（见配置章节）。

### 9.2 缓存管理

```bash
# 查看缓存统计（大小、条目数）
markitai cache stats

# 详细查看缓存条目
markitai cache stats -v

# 清除所有缓存
markitai cache clear

# 管理 SPA 域名缓存
markitai cache spa-domains
```

### 9.3 配置管理

```bash
# 查看当前生效配置
markitai config list

# JSON 格式输出
markitai config list -f json

# 查看配置文件路径
markitai config path

# 验证配置文件
markitai config validate ./markitai.json

# 获取/设置配置值
markitai config get llm.enabled
markitai config set llm.enabled true

# 交互式编辑配置（搜索、浏览、修改各项设置）
markitai config edit
```

### 9.4 交互模式

对命令行不熟悉的用户可使用交互式引导（`-I` 会进入独立的交互流程，无需预先指定输入）：

```bash
markitai -I
```

系统引导用户逐步选择：
- 输入类型（文件/目录/URL）和输入路径
- 是否启用 LLM
- 图片处理方式
- 输出位置
- 确认后执行

### 9.5 自定义 Prompt

高级用户可自定义 LLM 使用的提示词：

```bash
# 在 ~/.markitai/prompts/ 下放置自定义提示词
~/.markitai/prompts/
├── cleaner_system.md        # 自定义 Markdown 清洗提示
├── image_caption_system.md  # 自定义 alt 文本生成提示
└── image_description_system.md  # 自定义图片描述提示
```

### 9.6 域名配置文件

为特定域名定制抓取行为，可配置字段包括 `wait_for_selector`、`wait_for`、`extra_wait_ms`、`prefer_strategy`、`strategy_priority`、`skip_auto_scroll`、`reject_resource_patterns`：

```json
// markitai.json
{
  "fetch": {
    "domain_profiles": {
      "spa-app.example.com": {
        "prefer_strategy": "playwright",
        "wait_for": "networkidle",
        "wait_for_selector": "#app-root"
      }
    }
  }
}
```

> 注意：Markitai 内置了 `x.com`/`twitter.com` 和 `github.com` 的域名配置。若在 `domain_profiles` 中为同一域名重新配置，会**整体替换**内置配置而非逐字段合并——例如自行覆盖 `github.com` 时，需要自己重新声明内置的 `wait_for_selector: ".markdown-body"` 等字段，否则会丢失。

---

## 10. 问题排查与维护

### 10.1 常见问题与解决

**问题：转换后乱码或格式丢失**
```bash
# 尝试使用 kreuzberg 转换器
markitai document.pdf -b kreuzberg -o ./output
```

**问题：URL 抓取失败**
```bash
# 开启详细日志查看原因
markitai https://example.com -v -o ./output

# 尝试切换抓取策略
markitai https://example.com -s playwright -o ./output
```

**问题：LLM 调用失败**
```bash
# 检查认证状态（以 Claude 为例）
markitai auth claude status

# 检查完整环境
markitai doctor
```

### 10.2 日志

日志配置选项：
- 日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
- 日志目录：可自定义（默认无文件日志）
- 日志轮转：默认 10MB 轮转，保留 7 天
- 日志格式：text 或 json

```bash
# 详细输出模式
markitai document.pdf -v -o ./output

# 静默模式（只显示错误）
markitai document.pdf -q -o ./output
```

---

## 11. 典型工作流场景

### 场景 A：研究员批量处理论文

```bash
# 1. 将下载的论文批量转为 Markdown
markitai ./papers/ -g "*.pdf" --llm --alt --preset standard -o ./papers-md

# 2. 中断后继续
markitai ./papers/ -g "*.pdf" --llm --alt --preset standard --resume -o ./papers-md
```

### 场景 B：开发者抓取技术文档

```bash
# 1. 创建 URL 列表
cat > docs.urls << 'EOF'
https://docs.example.com/api/v2/auth
https://docs.example.com/api/v2/users
https://docs.example.com/api/v2/resources
EOF

# 2. 批量抓取
markitai docs.urls -s defuddle -o ./api-docs
```

### 场景 C：在脚本中集成

```bash
#!/bin/bash
# 将文档转换后喂给另一个工具
markitai report.pdf --pure | my-analysis-tool --input -

# 或存储到变量
content=$(markitai page.html)
echo "$content" | wc -w
```

### 场景 D：内容创作者增强文档

```bash
# 完整增强流水线
markitai draft-article.docx \
  --llm \
  --alt \
  --desc \
  --screenshot \
  --keep-base \
  -o ./enhanced

# 对比原始和增强版本
diff enhanced/draft-article.docx.md enhanced/draft-article.docx.llm.md
```

### 场景 E：快速查看文件内容

```bash
# 在终端中快速阅读 PDF（支持内联图片的终端体验更佳）
markitai report.pdf

# 快速查看网页正文
markitai https://news.example.com/article
```

---

## 附录：命令速查表

| 操作 | 命令 |
|------|------|
| 单文件转换 | `markitai file.pdf -o ./out` |
| stdout 输出 | `markitai file.pdf` |
| URL 转换 | `markitai https://... -o ./out` |
| 批量目录 | `markitai ./dir/ -o ./out` |
| 批量 URL | `markitai urls.urls -o ./out` |
| LLM 增强 | `markitai file.pdf --llm -o ./out` |
| 使用预设 | `markitai file.pdf --preset rich -o ./out` |
| 生成 alt 文本 | `markitai file.pdf --llm --alt -o ./out` |
| 图片描述 | `markitai file.pdf --llm --desc -o ./out` |
| 截图模式 | `markitai file.pdf --screenshot -o ./out` |
| OCR 扫描件 | `markitai scan.pdf --ocr -o ./out` |
| 纯净模式 | `markitai file.pdf --llm --pure -o ./out` |
| 断点续传 | `markitai ./dir/ --resume -o ./out` |
| 预览模式 | `markitai ./dir/ --dry-run -o ./out` |
| 交互模式 | `markitai file.pdf -I` |
| 健康检查 | `markitai doctor` |
| 查看配置 | `markitai config list` |
| 交互式编辑配置 | `markitai config edit` |
| 认证状态 | `markitai auth claude status` |
| 缓存统计 | `markitai cache stats` |

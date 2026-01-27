# CLI 命令参考

## 基本用法

```bash
markitai <input> [options]
```

`<input>` 可以是：
- 文件路径 (`document.docx`)
- 目录路径 (`./docs`)
- URL (`https://example.com`)

## 转换选项

### `--llm`

启用 LLM 驱动的格式清洗和优化。

```bash
markitai document.docx --llm
```

### `--preset <name>`

使用预定义的配置预设。

| 预设 | 说明 |
|------|------|
| `rich` | LLM + alt + desc + screenshot |
| `standard` | LLM + alt + desc |
| `minimal` | 仅基础转换 |

```bash
markitai document.pdf --preset rich
```

### `--alt`

使用 AI 生成图片的 alt 文本。

```bash
markitai document.pdf --alt
```

### `--desc`

生成图片的详细描述。

```bash
markitai document.pdf --desc
```

### `--screenshot`

启用截图捕获：
- **PDF/PPTX**: 将页面/幻灯片渲染为 JPEG 图片
- **URL**: 使用 agent-browser 捕获全页面截图

```bash
# 文档截图
markitai document.pdf --screenshot
markitai presentation.pptx --screenshot

# URL 截图
markitai https://example.com --screenshot
```

::: tip
对于 URL，`--screenshot` 会在需要时自动将抓取策略升级为 `browser`。截图将保存为 `screenshots/{域名}_路径.full.jpg`。
:::

### `--ocr`

为扫描文档启用 OCR。

```bash
markitai scanned.pdf --ocr
```

### `--no-compress`

禁用图片压缩。

```bash
markitai document.pdf --no-compress
```

## 输出选项

### `-o, --output <path>`

指定输出目录。

```bash
markitai document.docx -o ./output
```

### `--resume`

恢复中断的批量处理。

```bash
markitai ./docs -o ./output --resume
```

## 并发选项

### `--llm-concurrency <n>`

LLM 并发请求数。

```bash
markitai ./docs --llm --llm-concurrency 10
```

### `-j, --batch-concurrency <n>`

文件处理并发数（默认：10）。

```bash
markitai ./docs -o ./output -j 4
```

::: tip
对于文件和 URL 混合的批处理，使用 `--url-concurrency` 单独控制 URL 抓取。这样可以防止慢速 URL 阻塞文件处理。
:::

## 缓存选项

### `--no-cache`

禁用 LLM 结果缓存（强制重新调用 API）。

```bash
markitai document.docx --llm --no-cache
```

### `--no-cache-for <patterns>`

对特定文件或模式禁用缓存（逗号分隔）。

```bash
# 单个文件
markitai ./docs --no-cache-for file1.pdf

# Glob 模式
markitai ./docs --no-cache-for "*.pdf"

# 多个模式
markitai ./docs --no-cache-for "*.pdf,reports/**"
```

## URL 选项

### `.urls` 文件支持

当输入为 `.urls` 文件时，Markitai 自动将其作为 URL 批量任务处理。

```bash
markitai urls.urls -o ./output
```

`.urls` 文件格式：
```
# 以 # 开头的是注释
https://example.com/page1
https://example.com/page2
```

### `--url-concurrency <n>`

URL 抓取并发数（默认：5）。与 `--batch-concurrency` 独立，防止慢速 URL 阻塞文件处理。

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `--agent-browser`

强制使用浏览器渲染抓取 URL。适用于 JavaScript 重度依赖的 SPA 网站（如 x.com、动态 Web 应用）。

```bash
markitai https://x.com/user/status/123 --agent-browser
```

::: tip
需要安装 `agent-browser`：
```bash
pnpm add -g agent-browser
agent-browser install           # 下载 Chromium
agent-browser install --with-deps  # Linux: 同时安装系统依赖
```
详见 [agent-browser 文档](https://github.com/vercel-labs/agent-browser)。
:::

### `--jina`

强制使用 Jina Reader API 抓取 URL。当浏览器渲染不可用时的云端替代方案。

```bash
markitai https://example.com --jina
```

::: warning
`--agent-browser` 和 `--jina` 互斥，同时只能使用一个。
:::

## 配置命令

### `markitai config list`

显示所有配置设置。

```bash
markitai config list
markitai config list --json
```

### `markitai config init`

创建新的配置文件。

```bash
markitai config init
markitai config init -o ~/.markitai/
```

### `markitai config get <key>`

获取特定配置值。

```bash
markitai config get llm.enabled
markitai config get cache.enabled
```

### `markitai config set <key> <value>`

设置配置值。

```bash
markitai config set llm.enabled true
markitai config set cache.enabled false
```

### `markitai config path`

显示配置文件路径。

```bash
markitai config path
```

### `markitai config validate`

验证配置文件。

```bash
markitai config validate
```

## 缓存命令

### `markitai cache stats`

显示缓存统计信息。

```bash
markitai cache stats
markitai cache stats -v           # 详细模式
markitai cache stats --json       # JSON 输出
markitai cache stats --scope project  # 仅项目缓存
```

### `markitai cache clear`

清除缓存数据。

```bash
markitai cache clear
markitai cache clear --scope project  # 只清除项目缓存
markitai cache clear --scope global   # 只清除全局缓存
markitai cache clear --include-spa-domains  # 同时清除已学习的 SPA 域名
```

### `markitai cache spa-domains`

查看或管理已学习的 SPA 域名。这些是自动检测到需要浏览器渲染的域名。

```bash
markitai cache spa-domains             # 列出已学习的域名
markitai cache spa-domains --json      # JSON 输出
markitai cache spa-domains --clear     # 清除所有已学习的域名
```

::: tip
SPA 域名会在静态抓取检测到 JavaScript 依赖时自动学习。这可以加速后续请求，避免浪费的静态抓取尝试。
:::

## 诊断命令

### `markitai check-deps`

检查所有可选依赖及其状态。用于诊断安装问题。

```bash
markitai check-deps
markitai check-deps --json    # JSON 输出
```

此命令验证：
- **agent-browser**: 用于动态 URL 抓取（SPA 渲染）
- **LibreOffice**: 用于 Office 文档转换（doc, docx, xls, xlsx, ppt, pptx）
- **Tesseract OCR**: 用于扫描文档处理（可选，RapidOCR 已内置）
- **LLM API**: 配置和连接状态

## 其他选项

### `--verbose`

启用详细输出。

```bash
markitai document.docx --verbose
```

### `--dry-run`

预览转换，不写入文件。

```bash
markitai document.docx --dry-run
```

### `-c, --config <path>`

指定配置文件路径。

```bash
markitai document.docx -c ./my-config.json
```

### `-v, --version`

显示版本信息。

```bash
markitai -v
```

### `-h, --help`

显示帮助信息。

```bash
markitai -h
markitai config -h
markitai cache -h
```

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

启用 LLM 驱动的格式清洗和优化。默认只写入 `.llm.md`（跳过基础 `.md`）。使用 `--keep-base` 可以同时写入两个文件。

```bash
markitai document.docx --llm
```

社媒帖（由站点提取器整理、标记为 `content_profile: social_post` 的内容，如 X/Twitter 帖子）的正文会原样直通。LLM 仅生成 frontmatter 元数据，帖子结构和原文措辞不会被改动。

**目录批量转换**可用 `--llm-batch` 走服务商的 Batch API，价格为实时调用的一半：

```bash
markitai docs/ --llm --llm-batch -o out/       # 挂等到 --llm-batch-timeout（默认 1 小时），超时转两段式
markitai --llm-batch-collect <batch-id> -o out/  # 稍后收取已转交的 batch
```

只增强本次成功转换的文件，输出目录中已有的无关文件不会被纳入。需要单模型的 OpenAI 或 Anthropic 池。命中缓存的文档立即完成；失败的 batch 请求按实时价格重试，增强失败时仍保留基础输出。OpenAI 侧的推理模型在 batch 中以关闭推理运行——batch 部署对 function tools 的硬性要求。

`--alt`/`--desc` 与 `--screenshot` 都与文档增强同批提交，图片分析和页图增强同样享受折扣；页数超过单次调用上限的文档改为实时增强，因为批处理多一轮就多一次等待。暂不能与 `--ocr` 组合——它的页图从不渲染：batch 第一阶段关闭 LLM 转换，而那正是用本地 OCR 读扫描页的分支。

::: tip
`--llm`、`--alt`、`--desc`、`--ocr`、`--screenshot` 都有对应的 `--no-*` 反义参数（`--no-llm`、`--no-alt`、`--no-desc`、`--no-ocr`、`--no-screenshot`），可用来显式关闭某个预设本会启用的特性，例如 `--preset rich --no-desc`。
:::

### `-p, --preset <name>`

使用预定义的配置预设。

| 预设 | 说明 |
|------|------|
| `rich` | LLM + alt + desc + screenshot |
| `standard` | LLM + alt + desc |
| `minimal` | 仅基础转换 |

```bash
markitai document.pdf --preset rich
markitai document.pdf --preset rich --no-desc   # rich 但不生成 desc，任何预设特性都可用 --no-* 单独关闭
```

### `--profile <name>`

为下游消费者塑形输出。与 `--preset` 正交：预设决定运行哪些功能，profile 决定输出长什么样。不带 `--profile` 时输出不变。

| Profile | 效果 |
|---------|------|
| `rag` | 可见的 `assets/` 目录（而非隐藏的 `.markitai/assets/`）、PDF 注入 `<!-- page: N -->` 页标记、管道表列数校验 |
| `obsidian` | 可见的 `assets/` 目录；可通过 `output.wikilinks` 启用 wikilink 图片引用 |
| `okf` | frontmatter 对齐 [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog) 规范 |

```bash
markitai document.pdf --profile rag -o out/
markitai document.pdf --preset rich --profile rag -o out/   # 功能 + 形态
```

详见[输出 Profile](./output-profiles.md)。

### `--alt`

使用 AI 生成图片的 alt 文本。需要 `--llm`，未启用时会跳过图片分析并给出提示。

```bash
markitai document.pdf --llm --alt
```

### `--desc`

生成图片的详细描述。需要 `--llm`，未启用时会跳过图片分析并给出提示。

```bash
markitai document.pdf --llm --desc
```

### `--screenshot`

启用截图捕获：
- **PDF/PPTX**: 将页面/幻灯片渲染为 JPEG 图片
- **URL**: 使用 Playwright 捕获全页面截图

```bash
# 文档截图
markitai document.pdf --screenshot
markitai presentation.pptx --screenshot

# URL 截图
markitai https://example.com --screenshot
```

::: tip
对于 URL，`--screenshot` 会在需要时自动将抓取策略升级为 `playwright`。截图将保存为 `.markitai/screenshots/{域名}_路径.full.jpg`。
:::

### `--screenshot-only`

仅捕获截图，不提取内容。**对 URL 输入而言**，行为取决于是否启用 `--llm`：

| 命令 | 输出 |
|------|------|
| `--screenshot-only` | 仅截图（不生成 .md 文件） |
| `--llm --screenshot-only` | `.llm.md` + 截图（LLM 从截图提取内容）；加 `--keep-base` 可同时得到 `.md` |

```bash
# 仅捕获截图
markitai https://example.com --screenshot-only

# LLM 纯粹从截图提取内容
markitai https://example.com --llm --screenshot-only
```

::: tip
当传统内容提取失败时（如重 JavaScript 网站、社交媒体），使用 `--llm --screenshot-only` 模式。
:::

::: warning
对于**文件输入**（PDF/PPTX），不加 `--llm` 的 `--screenshot-only` **不会**跳过 `.md`，仍会照常写入提取出的文本 Markdown，只是额外附带截图。上面“不生成 `.md` 文件”的保证仅适用于 URL 输入。
:::

当配置文件启用了该模式时，用 `--no-screenshot-only` 关掉它。

### `--ocr`

为扫描文档启用 OCR。

```bash
markitai scanned.pdf --ocr
```

- 不带 `--llm`：`--ocr` 使用本地 RapidOCR。
- 带 `--llm`（`--ocr --llm`）：由视觉模型直接读取页面图像（VLM OCR），不再走 RapidOCR——无需本地 OCR 后端，但页面图像会发送给远端模型；设置 `MARKITAI_NO_VLM_OCR=1` 可强制走本地 RapidOCR。

处理单张图片时，请使用 `--ocr` 提取文字，或使用 `--llm` 分析图片。如果两者都未启用，Markitai 会以状态码 1 退出，不会在没有任何输出时仍报告转换成功。

#### PDF 里的数学公式

PDF 文本抽取本身不认识「公式」这个概念，所以公式能保留多少，取决于模型能看到多少页面：

| 运行方式 | 公式的下场 |
|---------|-----------|
| `--ocr --llm` | 行内公式写成 `$...$` LaTeX，被抽取噪声吞掉的正文也会一并还原 |
| `--alt` / `--desc` | 独立成行的公式是以图片形式进入 markitai 的，其 LaTeX 会被还原进 `images.json` 的 `text` 字段，alt 文本给出概述 |
| 都不带 | 独立公式保留为图片引用——内容不丢，但也不是文本；行内公式维持抽取噪声原样 |

网页不同：MathJax 和 MathML 无需模型参与，直接转换为 `$...$` / `$$...$$`。

### `--pure`

透明直通模式：LLM 仅做文本清理，不生成 frontmatter 或后处理。

```bash
# 不带 --llm：输出原始 markdown，不含 frontmatter
markitai document.docx --pure

# 带 --llm：通过 LLM 仅做文本清理
markitai document.docx --llm --pure

# 带 --preset：preset 控制功能，--pure 控制输出格式
markitai document.pdf --preset rich --pure
```

::: tip
`--pure` 和 `--llm` 是独立的标志。`--pure` 单独使用时跳过 frontmatter 生成；`--pure --llm` 将内容发送给 LLM 做文本清理，但返回原始输出，不包含生成的元数据（description、tags 等）。
:::

::: warning
`--pure` 会静默覆盖 `--alt`、`--desc` 和 `--screenshot`。同时使用这些标志时会显示警告。
:::

当配置文件启用了纯净模式时，用 `--no-pure` 恢复 frontmatter 和后处理。

### `--keep-base`

在 LLM 模式下仍写入基础 `.md` 文件。默认情况下 `--llm` 只输出 `.llm.md` 以避免冗余文件。

```bash
# 默认：只写入 .llm.md
markitai document.docx --llm

# 同时保留 .md 和 .llm.md
markitai document.docx --llm --keep-base
```

### `--no-compress`

禁用图片压缩。

```bash
markitai document.pdf --no-compress
```

当配置文件禁用了压缩时，用 `--compress` 重新开启。

## 输出选项

### `-o, --output <path>`

指定输出位置。单个文件或 URL 输入时，`-o` 也可以直接写一个 `.md` 文件路径（如 `-o result.md`）。批量（目录或 `.urls` 列表输入）必须传目录：`.md` 值会被拒绝并报用法错误，而不会变成目录。省略时，单文件/URL 转换会输出到 stdout；目录批量与 `.urls` 列表输入则必须指定 `-o`。

```bash
markitai document.docx -o ./output
markitai document.docx -o ./result.md
```

### `--json`

在 stdout 打印一份机器可读的 JSON 结果，并关闭进度输出。必须配合 `-o`（stdout 归 JSON），与 `--llm-batch-collect` 及 `--dry-run` 不兼容（dry-run 不产生任何可由信封承载的 item）。

文档结构为 `{version, ok, error, items[], totals}`；其中 `version` 是信封 schema 版本，不是 markitai 发行版本。

- `items[]`：每个工作项一条，含 `kind`、`source`、`status`（`completed` / `failed` / `skipped`）、`output`、`error`、`skip_reason`、`images`、`screenshots`、`cost_usd`、`duration_s`、缓存标志、`fetch_strategy` 与 `llm_usage`。
- `error`：没有产生任何 item 的运行级失败（输入路径不存在、URL scheme 被拒、配置文件无法读取或格式非法、被中断），否则为 `null`。
- `totals`：`total`、`completed`、`failed`、`skipped`、`cost_usd`，以及 `duration_s`（各项耗时之和；并发批次的实际墙钟时间会短于该值）。
- `ok`：任一 item 失败或存在运行级 `error` 时为 `false`。

```bash
markitai ./docs -o ./output --json
markitai document.pdf -o ./output --json | jq '.items[] | select(.status == "failed")'
```

退出码语义不变（见[退出码](#退出码)）；脚本还应同时读取 `ok`，因为部分失败的批量会以 `10` 退出但仍输出 JSON。参数与用法错误（未知参数、`--json` 未配 `-o`、`-c` 指向不存在的路径）以非零状态退出，只写 stderr，不输出 JSON。运行时配置错误，包括非法 JSON、非对象根节点或无效 UTF-8，写入 JSON 的 `error` 字段。交互模式（`-I`）会在子进程中重跑收集到的命令且不带 `--json`，因此两者只在人工会话里搭配使用。

配合 `--llm-batch` 时，JSON 包含最终增强文件路径及累计 usage/费用。Batch 等待超时后，未完成增强的项以失败状态说明仍在等待，进程退出 `2`；使用打印的收取命令继续。收取命令本身不支持 `--json`。

### `--resume`

恢复中断的批量处理。已完成的文件会跳过，`FAILED`/中断时处于 `IN_PROGRESS` 的文件会重试，本次新增的文件也会被纳入，并报告 `Resuming batch: N completed, M remaining`。仅对批量输入（目录/`.urls`）生效，单个文件/URL 输入时会被忽略。

```bash
markitai ./docs -o ./output --resume
```

### `--record-history` {#record-history}

将本次完成的运行记录为 `markitai serve` 历史中的任务（保存在 `~/.markitai/serve/jobs/` 下，输出与引用的资源会一并复制），使其无需重启服务即可实时出现在网页界面的历史页面中。CLI 记录的条目带有「CLI」徽标，并与网页创建的任务共享七天清理、删除与归档下载。

```bash
markitai document.docx -o ./output --record-history
```

优先级：`--record-history` / `--no-record-history` > `MARKITAI_RECORD_HISTORY`（真值：`1`/`true`/`yes`/`on`；已设置但为假值表示显式关闭）> 配置项 `history.record` > 默认（关闭）。stdout/管道模式下跳过记录；记录完全容错，绝不会导致转换失败，完成后在 stderr 打印一行确认信息（可被 `--quiet` 抑制）。参见[配置 → 环境变量](/zh/guide/configuration#markitai-设置)。

## 并发选项

### `--llm-concurrency <n>`

LLM 并发请求数（默认：10）。

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

当配置文件禁用了缓存时，用 `--cache` 重新允许读取缓存。

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

当输入为 `.urls` 文件时，Markitai 自动将其作为 URL 批量任务处理。目录批量输入也会自动发现扫描树内的 `.urls` 文件并处理（遵循相同的 `--glob`/`--max-depth` 规则），将其中的 URL 与常规文件合并到同一次批量任务中。

```bash
markitai urls.urls -o ./output
```

`.urls` 文件支持三种格式：

纯文本：每行一个 URL，可在 URL 后加空白与自定义输出文件名：
```text
# 以 # 开头的是注释
https://example.com/page1
https://example.com/page2 custom_name
```

URL 字符串组成的 JSON 数组：
```json
["https://example1.com", "https://example2.com"]
```

对象组成的 JSON 数组，可选 `output_name`：
```json
[
  {"url": "https://example1.com"},
  {"url": "https://example2.com", "output_name": "custom"}
]
```

如果批次中有 URL 失败，已成功的 URL 结果仍会保留。部分成功的 `.urls` 任务以状态码 10 退出，脚本和 CI 可以据此区分完整成功与部分失败。

### `--glob, -g <pattern>`

限制目录批量扫描的匹配模式。可重复指定多个模式。使用 `!` 前缀排除。

```bash
# 只处理 PDF 文件
markitai ./docs -o ./output -g "*.pdf"

# 处理 PDF 和 DOCX 文件
markitai ./docs -o ./output -g "*.pdf" -g "*.docx"

# 排除子目录
markitai ./docs -o ./output -g '!drafts/**'
```

::: tip
仅适用于目录输入。在有历史扩展的 shell（如 zsh）中使用 `!` 前缀时请使用单引号。
:::

### `--max-depth <n>`

覆盖目录递归扫描深度（默认：5）。`0` 表示仅扫描输入目录本身（不递归）。

```bash
markitai ./docs -o ./output --max-depth 2
```

### `--url-concurrency <n>`

URL 抓取并发数（默认：5）。与 `--batch-concurrency` 独立，防止慢速 URL 阻塞文件处理。

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `-s, --strategy <name>`

选择 URL 抓取策略。这是抓取 URL 的主要参数，与下方的 `-b/--backend` 正交。

| 取值 | 说明 |
|------|------|
| `auto`（默认） | 按策略优先级依次尝试，失败自动回退 |
| `static` | 使用内置 webextract 的静态 HTTP 抓取，快速、无需 JS、无需外部 API |
| `playwright` | Playwright 浏览器渲染，适用于 JavaScript 重度依赖的 SPA 网站（如 x.com） |
| `defuddle` | Defuddle API，免费、无需认证，内容清洗效果优秀 |
| `jina` | Jina Reader API，浏览器渲染不可用时的云端替代方案 |
| `cloudflare` | Cloudflare Browser Rendering `/content` API，同时会启用 Workers AI `toMarkdown` 文件转换（见 `-b/--backend`） |

```bash
markitai https://example.com -s defuddle
markitai https://x.com/user/status/123 -s playwright
```

::: tip
`-s cloudflare` 需要 `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID`（环境变量或 `markitai.json` 中配置）。在 [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens) 创建 Token，添加 *Browser Rendering: Edit* 和 *Workers AI: Read* 权限。详见[配置说明 → Cloudflare 设置](/zh/guide/configuration#cloudflare-设置)。
:::

::: tip
如需预先安装 Playwright 浏览器：
```bash
uv run playwright install chromium
# Linux 还需安装系统依赖：
uv run playwright install-deps chromium
```
:::

### `-b, --backend <name>`

选择文件转换后端，与 `-s/--strategy`（仅影响 URL 抓取）正交。

| 取值 | 说明 |
|------|------|
| `native`（默认） | 内置转换器（DOCX、PDF、图片等） |
| `cloudflare` | Cloudflare Workers AI `toMarkdown`，需要 CF 凭据 |

```bash
markitai document.pdf -b cloudflare
markitai https://example.com -s playwright -b cloudflare   # -s 与 -b 可自由组合
```

::: tip
Cloudflare Browser Rendering 在 Free 计划上可用。Workers AI `toMarkdown` 对 PDF/Office/CSV/XML 免费；图片转换使用 Neurons 配额。对于有本地转换器的格式，内置转换器通常输出质量更高。存在更优本地转换器时，`-b cloudflare` 会给出提示。
:::

### 已移除的旧后端参数

下面 6 个后端别名已在 1.0.0 **移除**。继续传入会得到一条指明替代写法的用法错误，因此旧脚本会直接报错，而不是悄悄用错引擎完成转换：

| 已移除参数 | 改用 |
|-----------|--------|
| `--playwright` | `-s playwright` |
| `--defuddle` | `-s defuddle` |
| `--static` | `-s static` |
| `--jina` | `-s jina` |
| `--cloudflare` | `-s cloudflare`（需要 CF 文件转换时再加 `-b cloudflare`） |
| `--kreuzberg` | *（无——`.rtf` 自 1.0.0 起原生转换）* |

```bash
markitai https://example.com -s defuddle   # 替代旧的 --defuddle
```

这些别名带来的互斥规则也随之取消：`-s/--strategy` 与 `-b/--backend` 相互正交，可自由组合。

### `--no-remote-fetch`

绝不把 URL 交给远程抽取服务（Defuddle、Jina、Cloudflare）。效果等同于 `MARKITAI_NO_REMOTE_FETCH=1`，但写在命令行上，因此这是一个显式的隐私选择，而不是从 `--quiet` 继承来的副作用。

```bash
markitai https://example.com -o ./output --no-remote-fetch
```

`--quiet` 会抑制远程抓取同意提示，因此在 `fetch.remote_consent=ask` 下，静默运行会跳过所有远程策略。此时 CLI 会在 stderr 打印一条提示，并建议改用 `--no-remote-fetch`。详见[抓取策略 → 远程回退与仅本地 URL](/zh/guide/fetch-policy#remote-fallback-and-local-only-urls)。

## 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 成功（包括 `--dry-run`） |
| `1` | 单项失败或运行时错误（包括 `config validate` 校验失败、`cache stats --json` 读到损坏的缓存） |
| `2` | 参数/用法错误，或 Batch API 等待超时、需用 `--llm-batch-collect` 继续 |
| `10` | 批量运行部分失败（目录批处理或 URL 批量）；成功的条目会被保留 |

`--json` 不改变这些退出码：部分失败的批量仍以 `10` 退出，同时打印 JSON 文档，因此脚本应同时检查 `ok` 字段。

## 初始化命令

### `markitai init`

交互式配置向导，检查依赖项、检测 LLM 提供商（包括 ChatGPT OAuth、Claude/Copilot CLI，以及 `GEMINI_API_KEY` 环境变量）并生成配置文件。

```bash
# 交互式配置向导
markitai init

# 快速模式（不询问直接生成默认配置）
markitai init --yes
markitai init -y

# 生成本地项目配置（./markitai.json）
markitai init --local

# 指定输出路径
markitai init -o ./markitai.json
```

### `-I, --interactive`

进入交互模式，引导式文件转换设置。

```bash
markitai -I
```

## 配置命令

### `markitai config list`

显示当前生效的配置。默认会递归遮罩秘密值，包括提供商、抓取或认证设置中的嵌套字段。

```bash
markitai config list                        # 默认格式：json
markitai config list --format table         # 简洁表格视图
markitai config list -f yaml                # 需要：uv add pyyaml
markitai config list --show-secrets         # 明确要求显示原始值
```

::: warning
`--show-secrets` 仅供本机检查。不要把它的完整输出贴到 issue、聊天、CI 日志或其他共享渠道。
:::

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

### `markitai config edit`

交互式编辑配置设置，通过引导式菜单操作。

```bash
markitai config edit
```

### `markitai config validate`

验证配置文件。

```bash
markitai config validate
markitai config validate ./markitai.json    # 验证指定文件
```

## 缓存命令

### `markitai cache stats`

显示缓存统计信息。

```bash
markitai cache stats
markitai cache stats -v           # 详细模式（等同于 --verbose）
markitai cache stats --json       # JSON 输出
markitai cache stats --verbose --limit 50   # 限制显示条目数（默认：20）
```

### `markitai cache clear`

清除缓存数据。

```bash
markitai cache clear
markitai cache clear -y                       # 跳过确认
markitai cache clear --include-spa-domains    # 同时清除已学习的 SPA 域名
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

### `markitai doctor`

检查核心状态、可选能力和认证状态。缺少未启用的可选工具只会显示警告，不代表基础安装失败。当已配置的 Playwright 工作流无法启动、活跃 API 模型引用了缺失的环境变量、已启用的本地 LLM Provider 无法加载或认证，或明确请求的自动修复未成功时，命令会以非零状态码退出，因此脚本和 CI 可以依赖该结果。

```bash
markitai doctor
markitai doctor --fix     # Playwright 包已存在时，安全安装并重新检查 Chromium
markitai doctor --json    # JSON 输出
markitai doctor --suggest-extras   # 输出适合 `uv tool install "markitai[...]"` 的逗号分隔 extras 列表，包含 browser/extra-fetch/svg/heif/ocr 及检测到的提供商 extra
```

每一项都是能力报告，没有启用的能力不会让整次检查失败：

- **可选：RapidOCR**，用于对扫描件和图片执行 `--ocr`。它随 `ocr` extra 提供，不在核心安装内，因此这里显示「未安装」只表示 OCR 未开启，而不是安装损坏
- **可选：VLM OCR**，用于对扫描件执行 `--ocr --llm`：由视觉模型读取页面图像，替代 RapidOCR。配置了视觉模型即可用；设置 `MARKITAI_NO_VLM_OCR=1` 可强制走本地 RapidOCR
- **未配置时可选：Playwright**，用于动态 URL 抓取（SPA 渲染）；当 `fetch.strategy` 为 `playwright` 或 `screenshot.enabled` 为 true 时，它会成为阻断检查
- **可选：LibreOffice**，用于 PPTX 幻灯片渲染（macOS 上未安装时会回退到已装的 Microsoft PowerPoint）
- **LLM API**：配置和模型状态
- **Vision Model**：用于图像分析（从 litellm 自动检测）
- **本地 Provider 认证**：Claude Agent、GitHub Copilot 和 ChatGPT 的认证状态（如果已配置）

普通 doctor 每次发现 Chromium 文件存在时，都会执行隔离且有超时限制的 headless 启动测试，因此过期 marker 或缺少 Linux 系统库不会得到绿灯。`doctor --fix` 不会向当前项目添加 Python 包。

Playwright 包存在但 Chromium 缺失或不可用时，它会使用 Markitai 自身的解释器安装 Chromium，并重新执行运行时检查。启动失败仍会以非零状态退出；Linux 下会附上 `playwright install-deps chromium` 修复命令。如果 Playwright 包本身缺失，命令会安全退出，并提示使用 `uv tool install 'markitai[browser]' --force` 或对应的 pipx 命令替换隔离安装。

`--json` 与 `--fix` 不能同时使用：JSON 是只读的健康状态快照，修复则是面向人的交互操作。

输出示例：
```text
◆ 系统检查

  • 配置文件：~/.markitai/config.json

可选能力
  ⚠ RapidOCR: not installed — --ocr unavailable (everything else works)
  ⚠ Playwright: Playwright not installed
  ⚠ LibreOffice: Not installed

LLM
  ✓ LLM API: 1 active model(s) configured
  ✓ Vision Model: 1 detected: copilot/claude-haiku-4.5
  ✓ GitHub Copilot SDK: SDK + CLI installed

认证状态
  ✓ Copilot Auth: Authenticated

⚠ 核心检查通过（3 项必需或已配置检查通过，3 项非阻断警告）
```

::: tip
`LLM` 分组中的部分提供商和模型消息仍会显示为英文；分组标题、修复提示和总结行已经本地化。`LLM API` 行里的 `API provider(s): ...` 仅在启用了真正的远程 API 模型（非 `claude-agent/`、`copilot/` 或 `chatgpt/`）时才会出现。doctor 会解析活跃 API 模型中显式的 `env:` 引用；缺失变量会作为已配置失败，而不是只按模型数量显示绿灯。
:::

::: tip
当使用本地 provider（`claude-agent/` 或 `copilot/`）时，doctor 命令还会检查认证状态，如果认证失败会提供解决方案提示。
:::

## 认证命令

### `markitai auth`

本地提供商（Copilot、Claude、ChatGPT）的认证辅助工具。Gemini 通过直接 API Key 或 OpenRouter 访问（见[配置说明](/zh/guide/configuration#模型命名)），不通过此命令管理。不带子命令运行时，会显示三个 provider 各自登录状态的一览。

```bash
markitai auth                   # 查看所有 provider 概览
```

### `markitai auth copilot status`

显示 GitHub Copilot CLI 认证状态。

```bash
markitai auth copilot status
markitai auth copilot status --json    # JSON 输出
```

### `markitai auth copilot login`

运行 GitHub Copilot CLI 认证。

```bash
markitai auth copilot login
```

### `markitai auth claude status`

显示 Claude Code CLI 认证状态。

```bash
markitai auth claude status
markitai auth claude status --json     # JSON 输出
```

### `markitai auth claude login`

运行 Claude Code CLI 认证。

```bash
markitai auth claude login
```

### `markitai auth chatgpt status`

显示 ChatGPT OAuth 认证状态。

```bash
markitai auth chatgpt status
markitai auth chatgpt status --json    # JSON 输出
```

### `markitai auth chatgpt login`

运行 ChatGPT OAuth Device Code Flow 认证。

```bash
markitai auth chatgpt login
```

::: tip
也可以使用 `markitai doctor` 一次性检查所有已配置提供商的认证状态。
:::

## 服务与 Agent 命令

### `markitai serve`

启动本地网页工作台及其 REST + SSE API。它需要 `serve` 附加组件（`fastapi`、`uvicorn`、`python-multipart`）：

```bash
uv tool install "markitai[serve]" --force
markitai serve                    # http://127.0.0.1:3600，自动打开浏览器
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host <interface>` | `127.0.0.1` | 绑定的主机接口。默认值仅本机可访问；改为其他值会把 API 发布给所有能连上该地址的主机，这些请求需要启动时打印的访问令牌（除非使用 `--no-auth`） |
| `--port <n>` | `3600` | 监听端口 |
| `--no-open` | 关闭 | 启动后不打开浏览器 |
| `--no-auth` | 关闭 | 禁用访问令牌。其他机器此时无需凭据，URL 目标限定为公网地址且不能访问 LLM 设置，但仍可上传文件、访问历史、下载和删除；回环地址仍保持完整权限 |
| `--allowed-host <hostname>` | — | 额外允许出现在 `Host`/`Origin` 头中的主机名（可重复传入）。`localhost` 与 IP 字面量始终接受，其他名称一律拒绝以阻止 DNS 重绑定。这只是名称过滤，不是身份认证 |

启动时会生成访问令牌并打印形如 `http://host:port/#token=…` 的可直接打开的 URL，也可用 `MARKITAI_SERVE_TOKEN` 固定。本机请求永远不需要令牌；其他机器的请求必须携带——网页界面从 URL fragment 读取令牌并存入 `sessionStorage`（浏览器会把它从地址栏抹掉），脚本则发送 `Authorization: Bearer <token>`（下载/SSE 链接也可用 `?token=`）。令牌就是凭据：持有它的人可发起转换、查看历史、下载、删除并管理 LLM 设置。工作台本身与完整 API 列表见[网页工作台指南](/zh/guide/serve)。

### `markitai mcp`

通过 stdio 为 AI Agent 启动随包发布的 MCP 服务器：

```bash
markitai mcp
```

它与 `markitai-mcp` 命令是同一个服务器，因此注册表与客户端可以通过主 CLI 包启动它，而无需知道第二个可执行文件名（`uvx --from "markitai[mcp]" markitai mcp`）。它暴露 `convert_document`、`convert_url`、`batch_convert` 与 `job_status` 四个工具；客户端配置、LLM 设置与各能力所需附加组件见 [MCP 指南](/zh/guide/mcp)。

## 其他选项

### `--quiet, -q`

隐藏进度和一般提示，错误仍写入 stderr。单次转换原本会输出到 stdout 的 Markdown 在 `--quiet` 下仍会保留；静默模式不会把实际结果变成空输出。当远程服务可能收到公网 URL 时，一次性的隐私揭露也会刻意保留在 stderr。

```bash
markitai document.docx --quiet
```

### `-v, --verbose`

启用详细输出。

```bash
markitai document.docx --verbose
```

### `--log-level <level>`

**日志文件**的最低级别（`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`），覆盖配置里的 `log.level`。未配置 `log.dir` 时文件日志本身是关闭的，因此该标志在那种情况下不生效；它只作用于转换运行（子命令有自己的输出）。终端输出仍由 `--verbose` / `--quiet` 控制；这个标志不会让终端变吵。

```bash
markitai ./docs -o out --log-level WARNING
```

### `--dry-run`

预览转换，不写入文件。

```bash
markitai document.docx --dry-run
```

### `-c, --config <path>`

指定配置文件路径。

```bash
markitai document.docx --config ./my-config.json
```

### `--config-json <json>`

内联 JSON 配置覆盖，与配置文件深度合并（显式 CLI 参数仍然优先）。适合 agent/CI 场景。

```bash
markitai document.docx --config-json '{"llm": {"concurrency": 4}}'
```

### `-V, --version`

显示版本信息。

```bash
markitai -V
```

### `-h, --help`

显示帮助信息。

```bash
markitai -h
markitai config -h
markitai cache -h
```

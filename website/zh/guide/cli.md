# CLI 命令参考

## 基本用法

```bash
markitai <input> [options]
```

输入可以是文件（`document.docx`）、目录（`./docs`）或 URL（`https://example.com`）。`mkai` 是同一个命令的短别名。

## 转换选项

### `--llm`

用 LLM 清洗格式并生成 frontmatter。默认只写 `.llm.md`，加 `--keep-base` 可以同时保留原始 `.md`。

```bash
markitai document.docx --llm
```

社交帖子（X/Twitter 等）正文原样保留，LLM 只写 frontmatter。

`--llm`、`--alt`、`--desc`、`--ocr`、`--screenshot` 各有一个 `--no-*` 对应项（`--no-llm`、`--no-alt`、`--no-desc`、`--no-ocr`、`--no-screenshot`），用来关掉预设或配置文件打开的功能，例如 `--preset rich --no-desc`。

#### Batch API

目录批量转换时，`--llm-batch` 把增强交给提供商的 Batch API，价格是标价的一半：

```bash
markitai docs/ --llm --llm-batch -o out/         # 最多等 --llm-batch-timeout（1 小时），超时后转为后台
markitai --llm-batch-collect <batch-id> -o out/  # 稍后收取转为后台的批次
```

需要单个 OpenAI 或 Anthropic 模型。图片分析和截图走同一个批次。`--ocr` 暂不支持批量模式。增强失败时基础输出照常写出。

### `-p, --preset <name>`

用一组打包好的常用参数：

| 预设 | LLM | alt | desc | screenshot | OCR |
|------|:---:|:---:|:----:|:----------:|:---:|
| `minimal` | – | – | – | – | – |
| `standard` | ✓ | ✓ | ✓ | – | – |
| `rich` | ✓ | ✓ | ✓ | ✓ | – |

没有预设会开 OCR，`--ocr` 总是显式指定。自定义预设可在[配置文件](/zh/guide/configuration#预设)里定义。

```bash
markitai document.pdf --preset rich
markitai document.pdf --preset rich --no-desc
```

### `--profile <name>`

为下游消费者塑形输出文件。预设决定跑哪些功能，profile 决定文件长什么样。不带 `--profile` 时输出不变。

| Profile | 效果 |
|---------|------|
| `rag` | 可见的 `assets/` 目录，PDF 带 `<!-- page: N -->` 标记，管道表格列数检查 |
| `obsidian` | 可见的 `assets/` 目录，可选的 wikilink 图片引用 |
| `okf` | frontmatter 对齐 [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog) |

```bash
markitai document.pdf --profile rag -o out/
markitai document.pdf --preset rich --profile rag -o out/
```

详见[输出 Profile](./output-profiles.md)。

### `--alt`

为图片生成 alt 文本。需要 `--llm`。

```bash
markitai document.pdf --llm --alt
```

### `--desc`

为图片生成详细描述，写入 `images.json`。需要 `--llm`。

```bash
markitai document.pdf --llm --desc
```

### `--screenshot`

把 PDF 页面和 PPTX 幻灯片渲染成 JPEG，或用 Playwright 给 URL 截整页。截图放在 `.markitai/screenshots/`。

```bash
markitai document.pdf --screenshot
markitai https://example.com --screenshot
```

对 URL，`--screenshot` 会在需要时把抓取策略切到 `playwright`。

### `--screenshot-only`

只截图，跳过内容抽取。对 URL：

| 命令 | 输出 |
|------|------|
| `--screenshot-only` | 只有截图，没有 `.md` |
| `--llm --screenshot-only` | LLM 从截图里读出内容写成 `.llm.md`，外加截图 |

```bash
markitai https://example.com --screenshot-only
markitai https://example.com --llm --screenshot-only
```

`--llm --screenshot-only` 是文本抽取失败时的兜底，比如重 JavaScript 的站点。对 PDF 和 PPTX 文件，不带 `--llm` 的 `--screenshot-only` 仍会在截图旁写出正常抽取的 `.md`。配置文件开了这个模式时，用 `--no-screenshot-only` 关掉。

### `--ocr`

识别扫描版 PDF 和图片。

```bash
markitai scanned.pdf --ocr
```

不带 `--llm` 时，用 RapidOCR 在本地识别（需要 `markitai[ocr]`）。带 `--llm` 时改由视觉模型直接读页面图片，不需要 OCR extra，但页面会发给模型。`MARKITAI_NO_VLM_OCR=1` 强制走本地。

单张图片作为输入时需要 `--ocr` 或 `--llm`，两者都没有时 markitai 以状态码 1 退出，而不是报告一次空成功。

#### PDF 里的数学公式

| 运行方式 | 公式的下场 |
|----------|-----------|
| `--ocr --llm` | 行内公式变成 `$...$` LaTeX |
| `--alt` 或 `--desc` | 独立公式是一张图片，它的 LaTeX 进 `images.json` |
| 都不带 | 独立公式保持图片引用，行内公式留作抽取噪声 |

网页不需要模型：MathJax 和 MathML 直接转成 `$...$` 和 `$$...$$`。

### `--pure`

输出不带 frontmatter 的纯 Markdown。带 `--llm` 时，模型只清洗文本，不加任何元数据。

```bash
markitai document.docx --pure
markitai document.docx --llm --pure
```

::: warning
`--pure` 会覆盖 `--alt`、`--desc` 和 `--screenshot`。同时使用时会打印警告。
:::

配置文件开了 pure 模式时，用 `--no-pure` 恢复 frontmatter。

### `--keep-base`

LLM 模式下同时写出原始 `.md` 和 `.llm.md`。

```bash
markitai document.docx --llm --keep-base
```

### `--no-compress`

图片保持原始尺寸和格式。配置文件关了压缩时，用 `--compress` 强制打开。

```bash
markitai document.pdf --no-compress
```

## 输出选项

### `-o, --output <path>`

写到哪里。目录对任何输入都可以；单个文件或 URL 也可以直接指定一个 `.md` 文件（`-o result.md`）。不带 `-o` 时，单个文件或 URL 打印到 stdout。批量（目录和 `.urls` 文件）必须给目录。

```bash
markitai document.docx -o ./output
markitai document.docx -o ./result.md
```

### `--json`

在 stdout 打印一份机器可读的 JSON 结果，并关闭进度输出。需要 `-o`。

```bash
markitai ./docs -o ./output --json
markitai document.pdf -o ./output --json | jq '.items[] | select(.status == "failed")'
```

文档结构是 `{version, ok, error, items[], totals}`：

- `items[]`：每个输入一条，含 `source`、`status`（`completed`、`failed`、`skipped`）、`output`、`error`、`cost_usd`、`duration_s`、`fetch_strategy` 和 `llm_usage`。
- `error`：没产生任何条目的运行级错误，否则为 `null`。
- `totals`：按状态计数，加 `cost_usd` 和 `duration_s`。
- `ok`：任一条目失败或设了 `error` 时为 `false`。

退出码含义不变（见[退出码](#退出码)）；部分失败的批量仍以 `10` 退出并打印 JSON，脚本要同时看 `ok`。用法错误只在 stderr 报，不输出 JSON。`--json` 不能和 `--dry-run` 或 `--llm-batch-collect` 一起用。

### `--resume`

继续被中断的批量。已完成的跳过，失败和中断的重试，新增的文件会被捡起。只对批量输入有效。

```bash
markitai ./docs -o ./output --resume
```

### `--record-history` {#record-history}

把这次运行记入[网页工作台](/zh/guide/serve#历史记录)的历史，带 CLI 徽标，保留七天。

```bash
markitai document.docx -o ./output --record-history
```

优先级：`--record-history` / `--no-record-history`，其次环境变量 `MARKITAI_RECORD_HISTORY`，再次配置项 `history.record`，默认关闭。stdout 模式不记录，记录失败也不影响转换。

## 并发选项

### `--llm-concurrency <n>`

同时发出的 LLM 请求数（默认 10）。

```bash
markitai ./docs --llm --llm-concurrency 10
```

### `-j, --batch-concurrency <n>`

同时转换的文件数（默认 10）。URL 有自己的并发池，见 `--url-concurrency`。

```bash
markitai ./docs -o ./output -j 4
```

## 缓存选项

### `--no-cache`

跳过缓存的 LLM 结果，重新调用 API。配置文件关了缓存读取时，用 `--cache` 重新打开。

```bash
markitai document.docx --llm --no-cache
```

### `--no-cache-for <patterns>`

对特定文件或 glob 跳过缓存，逗号分隔。

```bash
markitai ./docs --no-cache-for "*.pdf,reports/**"
```

## URL 选项

### `.urls` 文件支持

`.urls` 文件按 URL 批量转换。目录批量也会捡起目录树里的 `.urls` 文件。

```bash
markitai urls.urls -o ./output
```

文件是纯文本，一行一个 URL，空格后可以跟自定义输出名；也可以是 JSON 数组，元素为字符串或 `{"url", "output_name"}` 对象。`#` 开头的行是注释。

```text
https://example.com/page1
https://example.com/page2 custom_name
```

有 URL 失败时成功的照样保留；部分成功的运行以状态码 10 退出。

### `--glob, -g <pattern>`

把目录批量限制在匹配的相对路径上。可重复传多个模式；`!` 前缀表示排除。

```bash
markitai ./docs -o ./output -g "*.pdf" -g "*.docx"
markitai ./docs -o ./output -g '!drafts/**'
```

在有历史扩展的 shell 里，`!` 模式要用单引号包起来。

### `--max-depth <n>`

目录扫描深度（默认 5）。`0` 只扫描目录本身。

```bash
markitai ./docs -o ./output --max-depth 2
```

### `--url-concurrency <n>`

同时抓取的 URL 数（默认 5），和文件转换分开，慢页面不会拖住本地文件。

```bash
markitai ./docs -o ./output --url-concurrency 5
```

### `-s, --strategy <name>`

怎么抓 URL：

| 取值 | 说明 |
|------|------|
| `auto`（默认） | 按策略顺序逐个尝试，本机优先 |
| `static` | 普通 HTTP 抓取加内置抽取器。快，不跑 JavaScript，不出本机 |
| `playwright` | 浏览器渲染，用于重 JavaScript 站点。需要 `markitai[browser]` |
| `defuddle` | Defuddle API，免费，无需 key |
| `jina` | Jina Reader API，需要 `JINA_API_KEY` |
| `cloudflare` | Cloudflare Browser Rendering，需要 `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID` |

```bash
markitai https://example.com -s defuddle
markitai https://x.com/user/status/123 -s playwright
```

`auto` 的顺序见[抓取策略](/zh/guide/fetch-policy)，token 获取见 [Cloudflare 设置](/zh/guide/configuration#cloudflare-设置)。

### `-b, --backend <name>`

怎么转文件。和 `-s` 无关，`-s` 只影响 URL。

| 取值 | 说明 |
|------|------|
| `native`（默认） | 内置转换器 |
| `cloudflare` | Cloudflare Workers AI `toMarkdown`。需要 Cloudflare 凭据 |

```bash
markitai document.pdf -b cloudflare
```

内置转换器支持的格式通常质量更好，这种情况下 `-b cloudflare` 会给出提醒。

### 已移除的旧后端参数

1.0.0 移除了六个别名。传入会报用法错误并给出替代写法：

| 已移除 | 改用 |
|--------|------|
| `--playwright` | `-s playwright` |
| `--defuddle` | `-s defuddle` |
| `--static` | `-s static` |
| `--jina` | `-s jina` |
| `--cloudflare` | `-s cloudflare`（文件转换再加 `-b cloudflare`） |
| `--kreuzberg` | 无；`.rtf` 已由内置转换器处理 |

### `--no-remote-fetch`

绝不把 URL 发给远程服务（Defuddle、Jina、Cloudflare）。等同于 `MARKITAI_NO_REMOTE_FETCH=1`。

```bash
markitai https://example.com -o ./output --no-remote-fetch
```

`--quiet` 会压掉同意询问，所以在 `fetch.remote_consent=ask` 下，静默运行会跳过所有远程服务并在 stderr 说明。见[抓取策略](/zh/guide/fetch-policy#remote-fallback-and-local-only-urls)。

## 退出码

| 码 | 含义 |
|----|------|
| `0` | 成功，包括 `--dry-run` |
| `1` | 单个条目失败，或运行时错误 |
| `2` | 用法错误，或 Batch API 等待超时、需要 `--llm-batch-collect` |
| `10` | 批量部分失败；成功的条目保留 |

## 初始化命令

### `markitai init`

引导式配置：检查依赖、检测 LLM 提供商、写出配置文件。

```bash
markitai init              # 交互式
markitai init --yes        # 全用默认值，不询问（-y）
markitai init --local      # 写 ./markitai.json 而不是 ~/.markitai/config.json
markitai init -o ./markitai.json
```

### `-I, --interactive`

引导式转换：markitai 询问输入、输出和选项，然后运行。

```bash
markitai -I
```

## 配置命令

### `markitai config list`

显示生效的配置。密钥会被打码。

```bash
markitai config list                    # JSON
markitai config list --format table
markitai config list -f yaml            # 需要 pyyaml
markitai config list --show-secrets
```

::: warning
`--show-secrets` 只用于本机查看。不要把它的输出贴到 issue、聊天或 CI 日志里。
:::

### `markitai config get <key>`

```bash
markitai config get llm.enabled
```

### `markitai config set <key> <value>`

```bash
markitai config set llm.enabled true
```

### `markitai config path`

显示配置文件的位置。

### `markitai config edit`

通过引导菜单编辑设置。

### `markitai config validate`

校验配置文件，可以指定某一个。

```bash
markitai config validate
markitai config validate ./markitai.json
```

## 缓存命令

### `markitai cache stats`

```bash
markitai cache stats
markitai cache stats --verbose --limit 50   # 按模型列出条目（-v；默认显示 20 条）
markitai cache stats --json
```

### `markitai cache clear`

```bash
markitai cache clear
markitai cache clear -y                       # 跳过确认
markitai cache clear --include-spa-domains    # 同时忘掉学到的 SPA 域名
```

### `markitai cache spa-domains`

markitai 学到的需要用浏览器渲染的域名（见 [SPA 学习](/zh/guide/fetch-policy#spa-学习)）。

```bash
markitai cache spa-domains
markitai cache spa-domains --json
markitai cache spa-domains --clear
```

## 诊断命令

### `markitai doctor`

检查安装、可选能力、LLM 配置和提供商登录状态。缺少可选工具只是警告，不算失败。

```bash
markitai doctor
markitai doctor --fix              # 已装 Playwright 包时安装 Chromium
markitai doctor --json
markitai doctor --suggest-extras   # 列出这台机器能用上的 extras
```

你配置了但跑不起来的东西会让命令以非零退出：无法启动的 Playwright 策略、API key 环境变量缺失的模型、没登录的本地提供商。脚本和 CI 可以依赖这一点。

`--fix` 不会安装任何 Python 包。Playwright 本身缺失时，它会提示你带 `markitai[browser]` 重装。`--json` 和 `--fix` 不能同时用。

## 认证命令

### `markitai auth`

订阅制提供商的登录助手。不带子命令时显示状态总览。

```bash
markitai auth
```

### `markitai auth copilot status`

```bash
markitai auth copilot status          # 加 --json 输出机器可读格式
```

### `markitai auth copilot login`

```bash
markitai auth copilot login
```

### `markitai auth claude status`

```bash
markitai auth claude status
```

### `markitai auth claude login`

```bash
markitai auth claude login
```

### `markitai auth chatgpt status`

```bash
markitai auth chatgpt status
```

### `markitai auth chatgpt login`

用设备码 OAuth 登录 ChatGPT 订阅。

```bash
markitai auth chatgpt login
```

Gemini 没有登录流程，用 API key 或 OpenRouter（见[模型命名](/zh/guide/configuration#模型命名)）。

## 服务与 Agent 命令

### `markitai serve`

启动[网页工作台](/zh/guide/serve)。需要 `markitai[serve]`。

```bash
markitai serve                    # http://127.0.0.1:3600，自动打开浏览器
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host <interface>` | `127.0.0.1` | 绑定的网卡。其他设备访问用 `0.0.0.0`，它们需要启动时打印的访问令牌 |
| `--port <n>` | `3600` | 端口 |
| `--no-open` | 关闭 | 不打开浏览器 |
| `--no-auth` | 关闭 | 禁用访问令牌。远程客户端此时只能提交公网 URL，且不能改 LLM 设置 |
| `--allowed-host <hostname>` | — | 用域名访问时额外放行的主机名（可重复） |

### `markitai mcp`

通过 stdio 启动 [MCP 服务器](/zh/guide/mcp)。需要 `markitai[mcp]`。

```bash
markitai mcp
```

## 其他选项

### `--quiet, -q`

隐藏进度和提示信息。错误、stdout 里的 Markdown 和一次性的远程抓取提示仍会显示。

### `-v, --verbose`

显示更多细节。

### `--log-level <level>`

日志文件的最低级别（`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`）。只在设了 `log.dir` 时生效；终端输出由 `--verbose` 和 `--quiet` 控制。

```bash
markitai ./docs -o out --log-level WARNING
```

### `--dry-run`

只显示会转换哪些内容，不写任何文件。

```bash
markitai ./docs --dry-run
```

### `-c, --config <path>`

使用指定的配置文件。

```bash
markitai document.docx --config ./my-config.json
```

### `--config-json <json>`

内联配置覆盖，合并到配置文件之上。显式参数仍然优先。适合 Agent 和 CI。

```bash
markitai document.docx --config-json '{"llm": {"concurrency": 4}}'
```

### `-V, --version`

```bash
markitai -V
```

### `-h, --help`

```bash
markitai -h
markitai config -h
```

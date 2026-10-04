# 配置

Markitai 的配置是一个 UTF-8 JSON 对象，只需写出与默认值不同的字段。CLI、`serve`、`mcp` 和各语言绑定共用同一套键、默认值和校验规则。配置描述用户期望；转换层决定能否兑现某项能力。浏览器、OCR、模型路由等字段在结构上有效，并不代表对应功能已经可用；尚未实现的运行路径给出明确错误。

## 快速参考

### 使用哪个文件

每次运行只选中一个文件，不会合并多个文件：

1. `-c/--config PATH`（必须存在；`config set/edit` 可以创建）
2. 环境变量 `MARKITAI_CONFIG`
3. 当前目录的 `markitai.json`
4. 用户配置 `MARKITAI_HOME/config.json`，未设置 `MARKITAI_HOME` 时为 `~/.markitai/config.json`
5. 都不存在时使用内建默认值

`--config-json '{...}'` 在选中的文件之上深度合并，仅对本次运行有效；`-p/--preset` 再覆盖功能开关，命令行显式参数最后生效。根级 `-c` 与 `--config-json` 对子命令同样有效。`markitai config path` 显示当前选中的文件。

### 常用命令

```sh
markitai init --yes                  # 在用户目录创建配置；检测到 API 密钥环境变量时写入模型名
markitai init --local --yes          # 在当前目录创建 markitai.json
markitai config path
markitai config list -f table        # json（默认）、yaml 或 table；默认隐藏密钥
markitai config get output.on_conflict
markitai config set output.on_conflict overwrite
markitai config set 'llm.model_list[0].litellm_params.model' openai/gpt-4.1-mini
markitai config validate markitai.json
markitai config edit                 # 终端中的交互编辑器
```

`config set` 按字段声明解析值，保存前完整校验，只写改动的路径并保留文件中的未知键；未知键（如 `image.qualty`）和越界的数组下标都会报错而不修改文件。`init` 生成的配置默认关闭 LLM，且不保存密钥明文。它也检查 PATH 或 `COPILOT_CLI_PATH`、`CLAUDE_CLI_PATH`、`CODEX_CLI_PATH` 指定的可执行文件，提示已安装的 Copilot CLI、Claude Code 和 Codex CLI。检查不会启动这些程序，也不读取它们的登录配置；文件存在不代表版本受支持或已登录。请按提示运行 `markitai auth <provider> status`，再参照[订阅指南](subscriptions.md)配置受支持的模型。初始化不会为订阅运行时猜测模型名。

### 示例

```json
{
  "llm": {
    "enabled": true,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "env:OPENAI_API_KEY"}
      }
    ]
  },
  "output": {"on_conflict": "overwrite"},
  "ocr": {"lang": "zh"},
  "log": {"dir": "~/.markitai/logs"}
}
```

`env:NAME` 在需要时才读取环境变量，配置文件里不必出现密钥。`~/.markitai/...` 路径在设置 `MARKITAI_HOME` 时自动落到该目录下。

### 常用设置

| 键 | 默认值 | 作用 |
|---|---|---|
| `output.dir` | `null` | 目录/URL 列表未给 `-o` 时的输出目录；单文件仍输出到 stdout |
| `output.on_conflict` | `rename` | 目标已存在时 `rename`（生成 `.v2`）、`overwrite` 或 `skip` |
| `output.profile` | `null` | `rag`、`obsidian` 或 `okf` 输出形态，同 `--profile` |
| `output.report` | `null` | 批量报告；`null` 表示目录/URL 列表开启、单项关闭 |
| `llm.enabled` | `false` | 模型增强，同 `--llm` |
| `llm.model_list` | `[]` | 模型部署；为空时按 `MODEL` 和供应商密钥自动选择。前缀见 [LLM：供应商](llm.md#providers-and-request-parameters)，其中包括 `groq/`、`mistral/`、`xai/`、`together_ai/` 等兼容 OpenAI 的前缀 |
| `llm.model_list[].model_info.max_input_tokens` | `null` | 模型的输入窗口（token）。设置后文档分块还须放进所有启用部署中最小的窗口（扣除提示词后按保守估算取 90%），只会变小、不会超过 32,000 字符；见 [LLM：长文本](llm.md#structured-documents-and-complete-long-text) |
| `llm.keep_base` | `false` | 增强后同时保留基础 Markdown，同 `--keep-base` |
| `llm.on_failure` | `fallback` | 增强失败时保留基础结果并警告；`fail` 判为失败 |
| `llm.concurrency` | `10` | 同时进行的模型请求上限 |
| `llm.max_requests_per_document` | `50` | 每个文档的模型请求上限，`0` 不限 |
| `llm.max_cost_per_document_usd` | `0` | 每个文档的美元上限，`0` 不限；见[定价](pricing.md) |
| `image.compress` / `image.format` / `image.quality` | `true` / `jpeg` / `75` | 资产图片压缩 |
| `image.max_width` | `1920` | 资产图片最大宽度 |
| `image.alt_enabled` / `image.desc_enabled` | `false` | 图片说明与描述（需要 LLM），同 `--alt`/`--desc` |
| `image.stdout_persist` | `true` | 未给 `-o` 输出到 stdout 时保存引用的图片并以 `file://` 链接；`false` 保留相对引用并警告，见[stdout 中的图片](images.md#images-on-stdout) |
| `image.stdout_persist_dir` | `~/.markitai/assets` | stdout 图片库目录（文件在其下 `blobs/`）；默认值跟随 `MARKITAI_HOME`，自定义路径保持原义 |
| `image.stdout_fetch_external` | `false` | 参考版用于终端内联显示远程图片；本构建无终端图片输出，可设置但无作用 |
| `ocr.enabled` / `ocr.lang` | `false` / `en` | OCR 开关与语言。默认 `en` 的多语言策略依后端而定：Vision 先英文再按需重试中日韩，Paddle 先多语言识别再按需重试韩文；`en-US` 在 Vision 中只读英文，在 Paddle 中仍使用多语言模型但不追加默认韩文回退。见[本地 OCR](ocr.md#the-default-language) |
| `security.pdf_sanitize` | `warn` | PDF 提取正文的隐藏文字策略：`off` 关闭安全提示，`warn` 保留原正文并提示，`remove` 有边界地过滤可疑文字；不修改原文件或资产，限制见[PDF](pdf.md#hidden-text-policy) |
| `screenshot.enabled` | `false` | 页面截图，同 `--screenshot` |
| `batch.concurrency` / `batch.url_concurrency` | `10` / `5` | 文件与 URL 并发，同 `-j`/`--url-concurrency` |
| `batch.scan_max_depth` / `batch.scan_max_files` | `5` / `10000` | 目录扫描深度与文件数上限 |
| `cache.enabled` | `true` | 模型答案与网页缓存，见[缓存](cache.md) |
| `cache.fetch_ttl_seconds` | `86400` | 无验证头网页的复用时间 |
| `fetch.strategy` | `auto` | URL 策略，同 `-s` |
| `fetch.remote_consent` | `always` | 是否允许把 URL 交给远程抽取服务（`ask`/`always`/`never`）。默认值 `always` 只让显式选择的远程策略运行；`auto` 只有在你自己写出 `always`（配置文件或 `--config-json`）或选 `ask` 并在终端回答同意后，才会在本地策略失败时回退到远程服务，见[抓取：策略顺序与远程回退](fetch.md#strategy-order-and-remote-fallback) |
| `fetch.policy.strategy_priority` / `max_strategy_hops` | `null` / `5` | `auto` 的策略顺序与最多尝试的策略数；`fetch.domain_profiles."<host[:port]>".strategy_priority`/`prefer_strategy` 按域名覆盖 |
| `fetch.policy.local_only_patterns` | `[]` | 永不发送给远程服务的主机（`NO_PROXY` 语法；`inherit_no_proxy` 默认把 `NO_PROXY` 也算进来） |
| `fetch.fallback_patterns` | X、Instagram 等 6 个域名 | 你自己写出的列表中的域名及其子域名在 `auto` 中先用本地浏览器，再静态抓取。默认列表不生效：X 帖子由静态抓取加 X 帖子阅读器读取，更快也更稳 |
| `fetch.cloudflare.api_token` / `account_id` | `null` | `-s cloudflare` 与 `-b cloudflare` 使用的你自己的 Cloudflare 凭据，可写 `env:NAME`，否则读 `CLOUDFLARE_API_TOKEN`/`CLOUDFLARE_ACCOUNT_ID` |
| `log.dir` / `log.level` | `null` / `INFO` | 文件日志目录与级别，默认不写日志 |
| `history.record` | `false` | 记录供 `serve` 查看的历史，同 `--record-history` |

内建 preset：`minimal` 全部关闭；`standard` 开启 LLM、alt 和 desc；`rich` 再开启截图；都不开启 OCR。`markitai config list` 列出完整默认值。

### 配置键的运行状态

`config validate` 先验证类型、枚举和范围；无效配置仍是错误并退出非零。
对于结构合法、但当前没有运行效果的兼容键，它在 stderr 输出 `Warning:`，
stdout 仍为“配置有效”且退出 0。只检查所选文件与 `--config-json` 的原始内容，
不把自动填充的默认键当成用户设置；普通转换、`config list` 和库加载不显示这些警告。
显式写出一个无效用键，即使值等于默认值，也会得到提示。

| 键 | 状态 | 实际行为与替代方式 |
|---|---|---|
| `batch.heavy_task_limit` | 无运行效果；显式配置时警告 | 不提供额外的“重任务”限额。用 `batch.concurrency`/`-j` 限制文件转换并发；模型请求用 `llm.concurrency`，URL 用 `batch.url_concurrency` |
| `office.macos_fallback` | 无运行效果；显式配置时警告 | 不自动操作 PowerPoint 等 macOS 应用。Office 文本用原生解析；页面截图所需后端见 [Office 渲染](office-rendering.md) |
| `image.stdout_fetch_external` | 无运行效果；显式配置时警告 | 不提供终端内联图片显示。stdout 的已提取图片仍由 `image.stdout_persist` 保存，或用 `-o DIR` 保留资产 |
| `image.quality` | JPEG 有效；WebP 不使用 | JPEG 按设置编码；WebP 使用无损编码，不受质量值影响；PNG 也不使用有损质量值，见[图片](images.md) |
| `llm.model_list[].model_info.max_input_tokens` | 有效 | 限制文档分块的输入窗口，不是兼容性占位键；详见上表和[长文本](llm.md#structured-documents-and-complete-long-text) |
| `fetch.fallback_patterns` | 显式配置时有效 | CLI 的 `auto` 使用用户写出的列表；内建默认列表不改变策略顺序，`serve`、`mcp` 和绑定不应用列表 |
| `fetch.remote_consent` | 依调用方式生效 | 远程策略明确选用时按值运行；CLI `auto` 回退需要显式同意，其他接口的 `auto` 不远程回退；见[抓取](fetch.md#strategy-order-and-remote-fallback) |

这张表列出容易与参考版行为混淆的键；其余配置的类型和默认值可用
`markitai config list` 查看。值通过结构验证，并不会安装可选后端、确认模型登录，
或保证外部服务可用。默认值本身不产生无效用键的提示。

### 环境变量

| 变量 | 作用 |
|---|---|
| `MARKITAI_HOME` | 替代 `~/.markitai`：用户配置、`.env`、缓存、浏览器安装、历史与 stdout 图片库（`assets/`）都放在这里 |
| `MARKITAI_CONFIG` | 配置文件路径（优先级低于 `-c`） |
| `MARKITAI_PURE` | `1`/`true`/`yes` 时等同 `--pure` |
| `MARKITAI_RECORD_HISTORY` | `1`/`true`/`yes`/`on`（不分大小写）开启历史，其他非空值关闭；命令行开关优先 |
| `MARKITAI_NO_REMOTE_FETCH` | `1`/`true`/`yes`/`on` 时禁止远程抽取服务（包括显式选择的远程策略和 `-b cloudflare`） |
| `MARKITAI_NO_VLM_OCR` | 非空且不是 `0`/`false`/`no` 时，LLM 开启的 OCR 先本地识别再只发送文字 |
| `MARKITAI_LOG_DIR` / `MARKITAI_LOG_FORMAT` | 覆盖 `log.dir` 与 `log.format`（`text`/`json`） |
| `MARKITAI_SERVE_TOKEN` | `serve` 的进程环境访问令牌（不从 `.env` 读取；含回环地址请求；未设置或空白时自动生成），见 [REST 服务](serve.md) |
| `MARKITAI_LANG` | `doctor`、`cache`、`config path/validate`、`init`、转换与批量运行的 stderr 提示以及 `--help` 的终端语言：以 `zh` 开头为中文，其他值为英文；为空时依次看 `LC_ALL`、`LC_MESSAGES`、`LANG`（跳过 `C`/`POSIX`），见 [CLI](cli.md#终端语言) |
| `MARKITAI_BROWSER_EXECUTABLE` | 指定 Chrome/Chromium 可执行文件 |
| `PLAYWRIGHT_BROWSERS_PATH` | 额外搜索的 Playwright 浏览器缓存目录 |
| `MODEL` | 未配置 `llm.model_list` 时使用的模型，如 `openai/gpt-4.1-mini` |
| `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY` | 供应商密钥；未配置模型时据此自动选择模型 |
| `AZURE_API_KEY`、`AZURE_API_VERSION`、`OLLAMA_API_KEY` | Azure 与 Ollama 部署的凭据和 API 版本 |
| `GROQ_API_KEY`、`MISTRAL_API_KEY`、`XAI_API_KEY`、`TOGETHER_API_KEY` 等 | 兼容 OpenAI 的前缀的密钥（完整列表及 LiteLLM 的别名见 [LLM：兼容 OpenAI 的前缀](llm.md#openai-compatible-prefixes)）；不参与自动选择，需用 `MODEL` 或 `llm.model_list` 指定模型。`hosted_vllm/`、`lm_studio/` 无需密钥 |
| `<PREFIX>_API_BASE`、`OPENAI_BASE_URL` | 部署未设置 `api_base` 时的端点；内置供应商为 `<PREFIX>_API_BASE`（OpenAI 另读 `OPENAI_BASE_URL`，`ollama_chat` 另读 `OLLAMA_API_BASE`），兼容前缀的变量名见上述表格。`hosted_vllm/` 没有默认端点，须设置 `api_base` 或 `HOSTED_VLLM_API_BASE` |
| `JINA_API_KEY` | Jina Reader 的可选密钥（未设置 `fetch.jina.api_key` 时使用） |
| `CLOUDFLARE_API_TOKEN`、`CLOUDFLARE_ACCOUNT_ID` | 未设置 `fetch.cloudflare.api_token`/`account_id` 时 Cloudflare 使用的令牌与账户 ID |
| `COPILOT_CLI_PATH`、`COPILOT_HOME`、`COPILOT_CACHE_HOME`、`COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`GITHUB_TOKEN`、`CLAUDE_CLI_PATH`、`CLAUDE_CONFIG_DIR`、`CODEX_CLI_PATH`、`CODEX_HOME` | 订阅运行时的可执行文件、状态目录与令牌，见[订阅](subscriptions.md) |
| `HTTPS_PROXY`、`HTTP_PROXY`、`ALL_PROXY`、`NO_PROXY` | 代理，见[抓取](fetch.md#proxies) |

配置解析使用环境快照：依次读取进程环境、当前目录 `.env`、`MARKITAI_HOME/.env`（`~/.markitai/.env`），已存在的值优先，不修改宿主进程环境。`MARKITAI_HOME` 与 `MARKITAI_SERVE_TOKEN` 直接读取进程环境，应在启动前导出，不能靠 `.env` 设置。参考版本的 `MARKITAI_PDF_WORKERS`、`MARKITAI_STATIC_HTTP` 在本构建中不读取。

`fetch.remote_consent` 的默认值 `always` 来自参考版本，`config list` 也这样显示；但运行时拿到的是填好默认值的配置，分不出这个 `always` 是默认值还是你写的。因此本构建中 `auto` 的远程回退只认 CLI 从所选配置文件和 `--config-json` 的原始内容中读到的、你亲手写出的 `always`（第一次回退时显示一次说明，每个 `MARKITAI_HOME` 只显示一次），或 `ask`（每次运行在终端询问一次；没有终端或使用 `--quiet` 时视为 `never` 并提示一次）。`serve`、`mcp` 和语言绑定中的 `auto` 从不回退到远程服务。`fetch.fallback_patterns` 同理：只有你在配置文件或 `--config-json` 中写出的列表才让其中的域名先用浏览器；默认列表（X、Instagram 等）虽在 `config list` 中显示但不生效，`serve`、`mcp` 和语言绑定也不应用任何列表。

同一套一次性提示（`MARKITAI_HOME/notices` 下的空标记文件）还用于两种情况：选定的远程抓取策略（`-s defuddle`/`jina`/`cloudflare` 或 `fetch.strategy`）第一次把 URL 发给该服务之前，每个服务提示一次；以及第一次把图像（页面渲染、截图、`--alt`/`--desc` 的图片、OCR 页面）发给不在本机的模型之前（回环地址上的模型不算）。`--quiet` 不显示也不记录，下次运行再显示；`serve`、`mcp` 和语言绑定不显示。见 [LLM：隐私提示](llm.md#privacy-notices)。

## 默认值与配置选择

`config list` 显示填充后的完整默认值；只有一个配置文件入选，临时 override
递归合并对象，数组和标量整体替换。配置损坏、无法读取或字段无效会报错。
CLI 缺失 `-c` 文件是用法错误；核心加载器找不到显式/环境指定文件时警告并
使用默认值，不继续读取低优先级文件。

`MARKITAI_HOME` 隔离默认的 `~/.markitai/...` 状态路径；显式自定义路径保持原义。
测试还应为子进程隔离 `HOME`，在项目测试目录中运行，不读取真实用户配置或凭据。

## 规范化与验证

`config validate` 检查类型、枚举和范围，不验证服务登录或安装后端。
未声明字段在加载时忽略；`config set` 则拒绝未知键，便于发现拼写错误。
模型部署须含 `model_name` 与 `litellm_params.model`，供应商连接须含 `id` 与
`provider`。配置对象省略的嵌套字段按默认值补齐。

Python 的配置对象在转换前也会规范化，但赋值本身不一定立即验证；
详见[语言绑定](bindings.md#python)。

## 定点编辑

键采用 `llm.model_list[0].litellm_params.weight` 这样的路径。数组下标必须已存在，
不能靠 `config set` 隐式创建模型。值按字段类型解析；字符串字段里的数字或
`true` 仍是字符串。更新先完整校验，再原子保存；失败不改变原文件。

`set/edit` 保留文件中的未知扩展键，但拒绝与临时 `--config-json` 合用。
配置路径是符号链接时更新它的目标并保留链接；输出资产的链接策略与此独立。
`config edit` 需要终端，复杂数组/映射用 `config set` 或直接编辑 JSON。

`list/get/set` 默认隐藏密钥、token、密码和所有 HTTP header 值，保留 `env:VAR`
引用名称。API base 只显示 scheme/host/端口。`--show-secrets` 会显示原值，
不要把这种输出附在问题报告里。脱敏不改写保存的配置。

## 环境引用

配置文件中的 `env:VARIABLE` 在结构加载时保留原样，不要求变量存在。浏览器设置接口不接受客户端提交的 `env:` 密钥引用；已保存密钥也不能用于任意改写的端点，详见[服务设置](service-settings.md)。运行到需要凭据的路径时，调用 `resolve_env_value` 或 `resolve_optional`，传入环境快照与 strict 标志：strict 缺失变量报错，非 strict 返回 None。显式引用变量存在但为空字符串时仍算已找到。

Jina/Cloudflare 一类可选凭据的 fallback 环境变量（`JINA_API_KEY`、`CLOUDFLARE_API_TOKEN`、`CLOUDFLARE_ACCOUNT_ID`）仅在配置值缺失或为空时使用。显式 `env:MISSING` 在非 strict 模式解析失败后，不再回退到另一个变量；Jina 此时不发送密钥，`auto` 跳过 Cloudflare，而 `-s cloudflare` 与 `-b cloudflare` 以 strict 模式解析，报出缺失的变量名。读取 `.env` 的顺序为进程环境、当前目录文件、隔离用户目录文件；已存在值优先，不修改宿主进程环境。

## 兼容边界

配置整数限于 64 位可表示范围，非有限浮点数拒绝。错误按第一个字段给出，
不复刻 Pydantic 的聚合错误格式。Python 配置适配器使用同一套 schema；
具体 API 差异见[语言绑定](bindings.md)。

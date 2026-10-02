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

`config set` 按字段声明解析值，保存前完整校验，只写改动的路径并保留文件中的未知键；未知键（如 `image.qualty`）和越界的数组下标都会报错而不修改文件。`init` 生成的配置默认关闭 LLM，且不保存密钥明文。

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
| `llm.model_list` | `[]` | 模型部署；为空时按 `MODEL` 和供应商密钥自动选择 |
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
| `ocr.enabled` / `ocr.lang` | `false` / `en` | OCR 开关与语言。默认 `en` 先按英文识别，英文读不出时再试中文、韩文和日文；写成 `en-US` 只读英文，其余值只读所写的那一种语言。见[本地 OCR](ocr.md#the-default-language) |
| `screenshot.enabled` | `false` | 页面截图，同 `--screenshot` |
| `batch.concurrency` / `batch.url_concurrency` | `10` / `5` | 文件与 URL 并发，同 `-j`/`--url-concurrency` |
| `batch.scan_max_depth` / `batch.scan_max_files` | `5` / `10000` | 目录扫描深度与文件数上限 |
| `cache.enabled` | `true` | 模型答案与网页缓存，见[缓存](cache.md) |
| `cache.fetch_ttl_seconds` | `86400` | 无验证头网页的复用时间 |
| `fetch.strategy` | `auto` | URL 策略，同 `-s` |
| `fetch.remote_consent` | `always` | 是否允许把 URL 交给远程抽取服务（`ask`/`always`/`never`） |
| `log.dir` / `log.level` | `null` / `INFO` | 文件日志目录与级别，默认不写日志 |
| `history.record` | `false` | 记录供 `serve` 查看的历史，同 `--record-history` |

内建 preset：`minimal` 全部关闭；`standard` 开启 LLM、alt 和 desc；`rich` 再开启截图；都不开启 OCR。`markitai config list` 列出完整默认值。

### 环境变量

| 变量 | 作用 |
|---|---|
| `MARKITAI_HOME` | 替代 `~/.markitai`：用户配置、`.env`、缓存、浏览器安装、历史与 stdout 图片库（`assets/`）都放在这里 |
| `MARKITAI_CONFIG` | 配置文件路径（优先级低于 `-c`） |
| `MARKITAI_PURE` | `1`/`true`/`yes` 时等同 `--pure` |
| `MARKITAI_RECORD_HISTORY` | `1`/`true`/`yes`/`on`（不分大小写）开启历史，其他非空值关闭；命令行开关优先 |
| `MARKITAI_NO_REMOTE_FETCH` | `1`/`true`/`yes`/`on` 时禁止远程抽取服务 |
| `MARKITAI_NO_VLM_OCR` | 非空且不是 `0`/`false`/`no` 时，LLM 开启的 OCR 先本地识别再只发送文字 |
| `MARKITAI_LOG_DIR` / `MARKITAI_LOG_FORMAT` | 覆盖 `log.dir` 与 `log.format`（`text`/`json`） |
| `MARKITAI_SERVE_TOKEN` | `serve` 远程访问令牌，见 [REST 服务](serve.md) |
| `MARKITAI_LANG` | `doctor`、`cache`、`config path/validate`、`init`、转换与批量运行的 stderr 提示以及 `--help` 的终端语言：以 `zh` 开头为中文，其他值为英文；为空时依次看 `LC_ALL`、`LC_MESSAGES`、`LANG`（跳过 `C`/`POSIX`），见 [CLI](cli.md#终端语言) |
| `MARKITAI_BROWSER_EXECUTABLE` | 指定 Chrome/Chromium 可执行文件 |
| `PLAYWRIGHT_BROWSERS_PATH` | 额外搜索的 Playwright 浏览器缓存目录 |
| `MODEL` | 未配置 `llm.model_list` 时使用的模型，如 `openai/gpt-4.1-mini` |
| `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY` | 供应商密钥；未配置模型时据此自动选择模型 |
| `AZURE_API_KEY`、`AZURE_API_VERSION`、`OLLAMA_API_KEY` | Azure 与 Ollama 部署的凭据和 API 版本 |
| `<PROVIDER>_API_BASE`、`OPENAI_BASE_URL` | 部署未设置 `api_base` 时的端点 |
| `JINA_API_KEY` | `-s jina` 的可选密钥 |
| `COPILOT_CLI_PATH`、`COPILOT_HOME`、`COPILOT_CACHE_HOME`、`COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`GITHUB_TOKEN`、`CLAUDE_CLI_PATH`、`CLAUDE_CONFIG_DIR`、`CODEX_CLI_PATH`、`CODEX_HOME` | 订阅运行时的可执行文件、状态目录与令牌，见[订阅](subscriptions.md) |
| `HTTPS_PROXY`、`HTTP_PROXY`、`ALL_PROXY`、`NO_PROXY` | 代理，见[抓取](fetch.md#proxies) |

CLI 启动时依次从进程环境、当前目录 `.env`、`MARKITAI_HOME/.env`（`~/.markitai/.env`）读取变量，已存在的值优先，不修改宿主进程环境。参考版本的 `MARKITAI_PDF_WORKERS`、`MARKITAI_STATIC_HTTP` 以及 Cloudflare 凭据在本构建中不读取。

## 默认值与配置选择

以下各节说明实现与契约细节，供集成和维护参考。

`config::defaults()` 返回与参考版本 1.2.0 全量模型一致的 JSON 对象：14 个顶层配置组，27 个模型共 169 个声明字段（含嵌套模型定义）。完整默认快照作为测试夹具跟踪，核心只嵌入约 13 KiB 的类型、枚举、边界和默认值事实。数据首次使用时解析，随后复用不可变元数据；没有引入完整 JSON Schema 引擎或 Python 运行时。

文件选择顺序为显式路径、`MARKITAI_CONFIG`、当前目录 `markitai.json`、用户目录 `config.json`。仅一个文件入选；不会合并项目和用户文件。其上可以递归应用临时 override，再执行规范化。数组和标量替换原值。

配置加载器选中了不存在的显式或环境路径时会在 stderr 警告并使用默认值，不继续选择低优先级文件；CLI 在此之前单独将缺失 `-c` 判为用法错误。已存在但无法读取、不是 UTF-8 JSON 对象、内容损坏或字段无效的文件都返回错误，不降级为默认值。

`MARKITAI_HOME` 替换通常的用户 `.markitai` 目录，用于隔离试用、测试和多实例。序列化默认仍保留旧路径字面值；运行时使用 `config::state_path` 将默认 `~/.markitai/...` 路径解析到隔离目录。显式自定义路径不会被重定向。不要为了测试修改进程的 HOME。

## 规范化与验证

`config::normalize(&Value)` 返回有效配置，递归填入缺失默认值，包括 `model_list`、`providers`、自定义 preset 和域名配置中的条目。返回对象与输入不共享可变状态。`config::validate(&Value)` 只报告是否可规范化，不改变原对象；调用者需要使用规范化结果才能获得类型转换和默认值。

结构规则包括：

- bool、整数、浮点数、字符串、nullable、数组和映射的元素类型；字符串字段不会把数字或布尔值转换成文本。
- 旧模型支持的常见标量转换，如 `"yes"`→true、`"1_000"`→1000、整数值浮点数→整数；字符串 `" true "` 仍不是合法 bool。
- 所有声明的 Literal 枚举及 minimum/maximum 边界。未声明的范围不会自行收紧，例如结构层不为批量并发添加旧版不存在的 1024 上限。
- 模型部署必须含 `model_name` 和 `litellm_params.model`；供应商连接必须含 `id` 与 `provider`。
- fetch strategy priority 必须非空、没有重复且只含实际策略名，不能包含 auto。local-only 模式不能为空，带 `/` 的值必须为有效 IPv4/IPv6 网络。
- 已知模型中的未知字段在加载时忽略，与原模型默认规则一致；任意键映射仍按其声明的值类型验证。

内部 `output.filename` 和 `output.reserved_stem` 用于 CLI 输出路径和同批名称预留，会在规范化时保留，但不出现在公开默认值中，也不能通过公开 `config set` 创建。

## 定点编辑

`config::key_pointer` 处理 `llm.model_list[0].litellm_params.weight` 一类键。`config::parse_cli_value` 根据字段声明保留字符串，因此数字形密钥和名为 `true` 的目录不会变成数值/bool。`config::set_value` 验证已声明键及数组索引，检查新配置后才更新原始对象。失败不修改内容；成功只写变化的路径，保留原文件里的未知扩展键，不输出全量默认配置。

未知静态叶子（例如 `image.qualty`）是错误；类型化动态映射可以增加条目，例如 `presets.custom`。数组下标不能越界，也不能隐式创建部署。`--config-json` 为临时读取 override；与 config set/edit 同用会报用法错误，避免把临时值误写入文件。

CLI 保存采用同目录临时文件和原子替换。配置路径本身是符号链接时先解析目标，更新目标文件，保留链接。输出文件的符号链接策略与配置保存独立。

`config validate <path>` 的显式位置参数必须存在，否则返回用法错误 2。配置显示在模型对象中省略 null 字段，保留任意映射中的显式 null；直接 `config get` 查询 nullable 字段仍显示 `null`。

`config list/get/set` 的默认显示通过同一脱敏函数处理：密钥/token/密码等字段隐藏，`env:VAR` 保留引用名称，API base 只显示 scheme、host 和端口，丢弃用户信息、路径、query 和 fragment。HTTP header 的名称保留、所有值隐藏，形似 `env:VAR` 的 header 也按实际内联内容隐藏。单键读取先定位值再按键路径脱敏，因此敏感容器的子键仍可查询。脱敏不会改写配置文件；`--show-secrets` 明确请求原值。

## 环境引用

`env:VARIABLE` 在结构加载时保留原样，不要求变量存在。运行到需要凭据的路径时，调用 `resolve_env_value` 或 `resolve_optional`，传入环境快照与 strict 标志：strict 缺失变量报错，非 strict 返回 None。显式引用变量存在但为空字符串时仍算已找到。

Jina/Cloudflare 一类可选凭据的 fallback 环境变量仅在配置值缺失或为空时使用。显式 `env:MISSING` 在非 strict 模式解析失败后，不再回退到另一个变量。读取 `.env` 的顺序为进程环境、当前目录文件、隔离用户目录文件；已存在值优先，不修改宿主进程环境。

## 证据与边界

`tests/derive_config_contract.py` 使用明确传入的参考 checkout、其 Python 环境和 Pydantic 模型重新计算事实。它仅提取默认值和 schema 约束，不复制旧文档、注释或实现。当前事实对照参考提交 `ba374322f884b0e720b45466cc1196f4574a3da5`。

```sh
/path/to/reference/.venv/bin/python \
  crates/markitai-core/tests/derive_config_contract.py /path/to/reference
```

测试覆盖全量默认快照、文件选择优先级、缺失/损坏文件、override、嵌套部署默认值、类型和数值边界、策略/CIDR、未知字段、稀疏编辑回滚、字符串输入、环境引用、单独模型验证、脱敏和配置符号链接保存。Python 配置适配器消费相同的嵌入式 schema，并通过 `normalize_model` 构造顶层或嵌套模型。调度中心记录实际 Cargo/差分测试结果；生成事实文件成功不能替代 Rust 编译和运行测试。

目前明确的结构边界：JSON 整数保存为 i64/u64，超出 64 位可表示范围的整数拒绝；非有限浮点数拒绝。Python 的任意精度整数以及个别非标准 Infinity 输入不能逐值等价。错误按第一个字段给出，未复制 Pydantic 的聚合错误格式。dotenv 插值在不修改宿主环境的约束下仍需跨文件差分测试。运行能力清单仍以 [CLI 状态](cli.md) 和统一调度中心为准。

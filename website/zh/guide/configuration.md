# 配置说明

## 配置优先级

从高到低：

1. 命令行参数
2. 环境变量
3. 配置文件
4. 内置默认值

## 配置文件

markitai 按下面的顺序找到第一个配置文件就用：

1. `--config` 指定的路径
2. `MARKITAI_CONFIG`
3. 当前目录的 `./markitai.json`
4. `~/.markitai/config.json`

### 初始化配置

```bash
markitai init            # 引导式配置
markitai init --yes      # 全用默认值，不询问
markitai init --local    # 写 ./markitai.json
```

### 查看配置

```bash
markitai config list                  # 生效的设置，密钥已打码
markitai config get llm.enabled
markitai config set llm.enabled true
markitai config edit                  # 引导菜单
markitai config validate
```

`config list` 会给密钥打码，包括嵌套的 API key、token、cookie 和自定义请求头。`--show-secrets` 显示原值，输出只留在本机。

### 完整配置示例

每个设置及其默认值：

:::: details 含全部默认值的 markitai.json

```json
{
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-flash-lite-latest",
          "api_key": "env:GEMINI_API_KEY"
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120
    },
    "concurrency": 10,
    "max_requests_per_document": 50,
    "max_cost_per_document_usd": 0,
    "max_vision_pages_per_document": 0,
    "pure": false,
    "keep_base": false
  },
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 75,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 99999,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 5000,
      "deduplicate": true
    },
    "stdout_persist": true,
    "stdout_persist_dir": "~/.markitai/assets",
    "stdout_fetch_external": false
  },
  "ocr": {
    "enabled": false,
    "lang": "en",
    "per_page_routing": true
  },
  "office": {
    "macos_fallback": true
  },
  "screenshot": {
    "enabled": false,
    "screenshot_only": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  },
  "cache": {
    "enabled": true,
    "no_cache": false,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  },
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "scan_max_depth": 5,
    "scan_max_files": 10000,
    "state_flush_interval_seconds": 10,
    "heavy_task_limit": 0
  },
  "fetch": {
    "strategy": "auto",
    "remote_consent": "always",
    "defuddle": {
      "timeout": 30,
      "rpm": 20
    },
    "playwright": {
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 3000,
      "wait_for_selector": null,
      "cookies": null,
      "reject_resource_patterns": null,
      "extra_http_headers": null,
      "user_agent": null,
      "http_credentials": null,
      "session_mode": "isolated",
      "session_ttl_seconds": 600
    },
    "jina": {
      "api_key": null,
      "timeout": 30,
      "rpm": 20,
      "no_cache": false,
      "target_selector": null,
      "wait_for_selector": null
    },
    "cloudflare": {
      "api_token": null,
      "account_id": null,
      "timeout": 30000,
      "wait_until": "networkidle0",
      "cache_ttl": 0,
      "reject_resource_patterns": null,
      "convert_enabled": false,
      "user_agent": null,
      "cookies": null,
      "wait_for_selector": null,
      "http_credentials": null
    },
    "policy": {
      "enabled": true,
      "max_strategy_hops": 5,
      "strategy_priority": null,
      "local_only_patterns": [],
      "inherit_no_proxy": true
    },
    "domain_profiles": {},
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  },
  "output": {
    "dir": null,
    "on_conflict": "rename",
    "allow_symlinks": false,
    "report": null
  },
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": null,
    "rotation": "10 MB",
    "retention": "7 days"
  },
  "security": {
    "pdf_sanitize": "warn"
  },
  "history": {
    "record": false
  },
  "prompts": {
    "dir": "~/.markitai/prompts"
  }
}
```

::::

任何字符串值都可以用 `env:VAR_NAME` 引用环境变量。`JINA_API_KEY`、`CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID` 不写进配置也会从环境里自动读取。

## 环境变量

### API 密钥

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic（Claude） |
| `GEMINI_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `OPENROUTER_API_KEY` | OpenRouter |
| `JINA_API_KEY` | Jina Reader |
| `CLOUDFLARE_API_TOKEN` | Cloudflare（Browser Rendering、Workers AI） |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare 账户 ID |

### Markitai 设置

| 变量 | 说明 |
|------|------|
| `MODEL` | 未配置 `model_list` 时使用的模型 |
| `MARKITAI_CONFIG` | 配置文件路径 |
| `MARKITAI_LOG_DIR` | 日志文件目录 |
| `MARKITAI_LOG_FORMAT` | `text` 或 `json` |
| `MARKITAI_LANG` | CLI 语言，`en` 或 `zh` |
| `MARKITAI_PURE` | 开启 pure 模式（`1`、`true`、`yes`） |
| `MARKITAI_RECORD_HISTORY` | 把 CLI 运行记入网页工作台历史（`1`、`true`、`yes`、`on`） |
| `MARKITAI_NO_REMOTE_FETCH` | 绝不把 URL 发给远程服务，显式 `-s` 也不行（`1`、`true`、`yes`） |
| `MARKITAI_NO_VLM_OCR` | `--ocr --llm` 时用本地 RapidOCR 而不是视觉模型（`1`、`true`、`yes`） |
| `MARKITAI_STATIC_HTTP` | 静态抓取客户端：`httpx`（默认）或 `curl_cffi` |
| `MARKITAI_SERVE_TOKEN` | `markitai serve` 的固定访问令牌 |
| `MARKITAI_INSTALL_OPTIONAL` | 安装脚本：不询问直接装可选组件 |
| `MARKITAI_USE_MIRROR` | 安装脚本：`1` 总是提供镜像，`0` 从不询问 |
| `MARKITAI_VERSION` | 安装脚本：要安装的版本 |

### `.env` 文件加载

markitai 先加载 `./.env`，再加载 `~/.markitai/.env`。先加载的值优先，所以项目可以覆盖全局设置。

## LLM 配置

### 支持的提供商

任何 [LiteLLM](https://docs.litellm.ai/) 提供商都可以用 API key 接入：OpenAI、Anthropic、Google、DeepSeek、OpenRouter、Ollama 等。

三家订阅制提供商改用各自的 CLI 或 OAuth 登录：

| 提供商 | 前缀 | 登录方式 | Extra |
|--------|------|----------|-------|
| Claude Code | `claude-agent/` | `markitai auth claude login` | `markitai[claude-agent]` |
| GitHub Copilot | `copilot/` | `markitai auth copilot login` | `markitai[copilot]` |
| ChatGPT | `chatgpt/` | 首次使用时走 OAuth 设备码 | — |

Claude Code 和 Copilot 的 CLI 要先装好：`curl -fsSL https://claude.ai/install.sh | bash` 和 `curl -fsSL https://gh.io/copilot-install | bash`（Windows：`irm https://claude.ai/install.ps1 | iex` 和 `winget install GitHub.Copilot`）。

Gemini 没有订阅登录。用 API key（`gemini/`）或走 OpenRouter（`openrouter/google/...`）。

### 模型命名

模型按 LiteLLM 的 `provider/model` 命名：

- `openai/gpt-5.6`
- `anthropic/claude-sonnet-4-6`
- `gemini/gemini-flash-lite-latest`
- `deepseek/deepseek-v4-flash`
- `ollama/llama3.2`
- `claude-agent/sonnet`、`copilot/gpt-5.6`、`chatgpt/gpt-5.6`（订阅制提供商）

#### markitai 自动选用的默认模型

`markitai init` 和自动检测 key 时，会选所找到的提供商里便宜、快速的那一档：

| 提供商 | 默认模型 |
|---|---|
| Claude Code | `claude-agent/sonnet` |
| GitHub Copilot | `copilot/claude-haiku-4.5` |
| ChatGPT | `chatgpt/gpt-5.6` |
| Anthropic | `anthropic/claude-haiku-4-5` |
| OpenAI | `openai/gpt-5.6-luna` |
| Gemini | `gemini/gemini-flash-lite-latest` |
| DeepSeek | `deepseek/deepseek-v4-flash` |
| OpenRouter | `openrouter/google/gemini-3.1-flash-lite` |

想用别的就设 `model_list`。配了已被提供商下线的模型只会在启动时警告一句，markitai 不会替你改。

图片分析（`--alt`、`--desc`）需要支持视觉的模型。订阅制提供商通过文件附件支持。

常见错误：

| 错误 | 处理 |
|------|------|
| "SDK not installed" | 安装 `markitai[copilot]` 或 `markitai[claude-agent]` |
| "CLI not found" | 安装 [Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli) 或 [Claude Code](https://claude.ai/code) |
| "Not authenticated" | 运行 `markitai auth copilot login` 或 `markitai auth claude login`。ChatGPT 首次使用时登录 |
| "Rate limit" | 稍等再试，或检查订阅额度 |

`markitai doctor` 会报告登录状态并给出提示。

### 自定义 API 端点

`api_base` 把提供商指向另一个端点：自建推理服务、地区代理或 API 网关。和 `api_key` 一样支持 `env:VAR_NAME`：

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "openai/your-model-name",
          "api_key": "env:YOUR_API_KEY",
          "api_base": "https://your-api-endpoint.com/v1"
        }
      }
    ]
  }
}
```

另外两种写法：

```json
// 本地 Ollama
{ "model": "ollama/llama3.2", "api_base": "http://localhost:11434" }

// Azure OpenAI：model 填你的部署名，不是模型 ID
{
  "model": "azure/your-deployment-name",
  "api_key": "env:AZURE_API_KEY",
  "api_base": "https://your-resource.openai.azure.com",
  "api_version": "2025-02-01-preview"
}
```

`api_base` 对订阅制提供商无效。Claude Code 认 `ANTHROPIC_BASE_URL`，Copilot 和 ChatGPT 自己管理端点。

### Vision 模型

视觉能力会从 LiteLLM 自动检测。要手动指定，在模型条目上设 `model_info.supports_vision`：

```json
{
  "model_name": "default",
  "litellm_params": { "model": "gemini/gemini-flash-lite-latest", "api_key": "env:GEMINI_API_KEY" },
  "model_info": { "supports_vision": true }
}
```

### 模型 Token 上限

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `litellm_params.max_tokens` | `null` | 每次调用请求的最大输出 token |
| `model_info.max_tokens` | `null` | 最大输出 token 元数据；省略时从 LiteLLM 检测 |
| `model_info.max_input_tokens` | `null` | 最大上下文 token 元数据；省略时从 LiteLLM 检测 |

### 路由设置

`llm.router_settings` 和它旁边的几项控制请求怎样分配到多个模型，以及一份文档最多消耗多少：

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `routing_strategy` | `simple-shuffle` | `simple-shuffle`、`least-busy`、`usage-based-routing` 或 `latency-based-routing`。订阅制提供商始终按权重随机 |
| `num_retries` | `2` | 每个请求的重试次数 |
| `timeout` | `120` | 请求超时（秒） |
| `fallbacks` | `[]` | 分组回退，如 `[{"default": ["backup"]}]`。不在回退组里的模型只通过回退接收流量 |
| `concurrency` | `10` | 同时发出的 LLM 请求数 |
| `max_requests_per_document` | `50` | 一份文档的请求数到这个值就停止增强，保留基础输出。`0` 不限 |
| `max_cost_per_document_usd` | `0` | 一份文档花到这个金额就停止增强。`0` 不限 |
| `max_vision_pages_per_document` | `0` | 一份文档最多发给视觉模型的页面图片数。超出的文档不做视觉增强。`0` 不限 |

#### 模型权重

`model_list` 里每个条目都可以在 `litellm_params` 里设 `weight`。`1` 是正常，`10` 被选中的概率是十倍，`0` 禁用该模型但保留配置。至少要有一个模型权重大于零；这在首次使用时检查，`config validate` 不查。

```json
{
  "model_name": "default",
  "litellm_params": { "model": "gemini/gemini-flash-lite-latest", "api_key": "env:GEMINI_API_KEY", "weight": 10 }
}
```

### 自适应超时

订阅制提供商的请求超时会随提示词长度、图片数量和预期输出自动伸缩，在 60 到 600 秒之间，大文档不会超时，短请求也不会干等。

### 提示缓存（Claude Agent）

Claude Code 会自动缓存 4 KB 以上的系统提示词。无需配置，`markitai cache stats --verbose` 能看到数字。

## 图片配置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `alt_enabled` | `false` | 用 LLM 生成 alt 文本 |
| `desc_enabled` | `false` | 生成图片描述 |
| `compress` | `true` | 压缩抽取出的图片 |
| `quality` | `75` | JPEG/WebP 质量（1–100） |
| `format` | `jpeg` | `jpeg`、`png` 或 `webp` |
| `max_width` | `1920` | 最大宽度（像素） |
| `max_height` | `99999` | 最大高度（像素） |
| `filter.min_width` | `50` | 跳过更窄的图片 |
| `filter.min_height` | `50` | 跳过更矮的图片 |
| `filter.min_area` | `5000` | 跳过更小的图片 |
| `filter.deduplicate` | `true` | 去掉重复图片 |
| `stdout_persist` | `true` | stdout 模式下把图片存进持久目录 |
| `stdout_persist_dir` | `~/.markitai/assets` | 持久目录位置 |
| `stdout_fetch_external` | `false` | stdout 模式下下载外部图片 URL |

## 截图配置

截图把 PDF 页面和 PPTX 幻灯片渲染成 JPEG，并用 Playwright 给 URL 截整页。文件放在 `.markitai/screenshots/`。

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 等同 `--screenshot` |
| `screenshot_only` | `false` | 等同 `--screenshot-only` |
| `viewport_width` | `1920` | URL 截图的浏览器视口宽度 |
| `viewport_height` | `1080` | URL 截图的浏览器视口高度 |
| `quality` | `75` | JPEG 质量（1–100） |
| `tile_height` | `2000` | 很长的 URL 截图按这个高度切片，视觉模型才读得清 |
| `max_height` | `10000` | 只在 `tile_height` 为 `0` 时使用的高度上限 |

## 预设

内置三个预设（`minimal`、`standard`、`rich`）。在 `presets` 下定义自己的：

```json
{
  "presets": {
    "my-preset": { "llm": true, "ocr": false, "alt": true, "desc": false, "screenshot": true }
  }
}
```

五个键默认都是 `false`。用 `markitai document.pdf --preset my-preset` 调用。

## OCR 配置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 等同 `--ocr` |
| `lang` | `en` | 语言：`en`、`zh`、`ja`、`ko`、`ar`、`th` 或 `latin` |
| `per_page_routing` | `true` | 看起来正常的页面保留原生文本层，只对其余页面做 OCR。`false` 则每页都 OCR |

本地 OCR 用 `ocr` extra 里的 [RapidOCR](https://github.com/RapidAI/RapidOCR)：

```bash
uv tool install "markitai[ocr]" --force
```

`--ocr --llm` 加一个支持视觉的模型就不需要 extra：模型直接读页面图片。`MARKITAI_NO_VLM_OCR=1` 强制走本地。

## Office 配置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `macos_fallback` | `true` | 没装 LibreOffice 的 Mac 上，调用已装的 PowerPoint 渲染 PPTX 幻灯片 |

首次渲染会弹一次 macOS 授权对话框。无头 Mac（SSH、CI）上没人能点，把它设成 `false`。

## 批处理配置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `concurrency` | `10` | 同时转换的文件数 |
| `url_concurrency` | `5` | 同时抓取的 URL 数，单独计数，慢页面不会拖住文件 |
| `scan_max_depth` | `5` | 目录扫描深度 |
| `scan_max_files` | `10000` | 每次运行的最大文件数 |
| `state_flush_interval_seconds` | `10` | 多久保存一次供 `--resume` 用的批量状态 |
| `heavy_task_limit` | `0` | CPU 密集任务的上限；`0` 按可用内存自动决定 |

## URL 抓取配置

```json
{
  "fetch": {
    "strategy": "auto",
    "remote_consent": "always",
    "playwright": { "timeout": 30000, "wait_for": "domcontentloaded", "extra_wait_ms": 3000 },
    "jina": { "api_key": "env:JINA_API_KEY" },
    "cloudflare": { "api_token": "env:CLOUDFLARE_API_TOKEN", "account_id": "env:CLOUDFLARE_ACCOUNT_ID" },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### 抓取策略

| 策略 | 说明 |
|------|------|
| `auto` | 本机优先：static，然后 Playwright，再 Defuddle、Jina、Cloudflare。已知重 JavaScript 的域名从 Playwright 开始。见[抓取策略](/zh/guide/fetch-policy) |
| `static` | 普通 HTTP 加内置抽取器。快，不跑 JavaScript |
| `playwright` | 浏览器渲染 JavaScript 页面 |
| `defuddle` | Defuddle API，免费，无需 key |
| `jina` | Jina Reader API |
| `cloudflare` | Cloudflare Browser Rendering；渲染后的 HTML 在本地抽取 |

### 远程抓取同意

| 设置 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `fetch.remote_consent` | `always`、`ask`、`never` | `always` | `always`：公开 URL 可回退到远程服务，首次尝试前在 stderr 提示。`ask`：有终端时每个进程问一次，否则跳过所有远程服务。`never`：只用本地策略 |

私有、内网和带凭据的 URL 无论怎么设都不会发给远程服务。`fetch.policy.local_only_patterns` 和 `NO_PROXY` 里的域名在 `auto` 链里只走本地。对公开 URL，显式的 `-s defuddle`、`-s jina` 或 `-s cloudflare` 会覆盖 `never` 和模式规则；`MARKITAI_NO_REMOTE_FETCH=1` 连这个也拦住。

X/Twitter 的补充抓取（FxTwitter、Twitter oEmbed）和其他远程服务遵循同一个同意决定。

### Playwright 设置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | `30000` | 页面加载超时（毫秒） |
| `wait_for` | `domcontentloaded` | `load`、`domcontentloaded` 或 `networkidle` |
| `extra_wait_ms` | `3000` | 加载事件后再等 JavaScript 的时间 |
| `session_mode` | `isolated` | `isolated`（每个请求新上下文）或 `domain_persistent`（按域名复用） |
| `session_ttl_seconds` | `600` | 持久会话的寿命 |
| `wait_for_selector` | `null` | 等待的 CSS 选择器 |
| `cookies` | `null` | `[{name, value, domain, path}]` |
| `reject_resource_patterns` | `null` | 拦截匹配的请求，如 `["**/*.css"]` |
| `extra_http_headers` | `null` | `{"Accept-Language": "zh-CN"}` |
| `user_agent` | `null` | 自定义 User-Agent |
| `http_credentials` | `null` | HTTP 认证的 `{username, password}` |

### Jina 设置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | `null` | Jina Reader API key（支持 `env:`） |
| `timeout` | `30` | 请求超时（秒） |
| `rpm` | `20` | 每分钟请求数 |
| `no_cache` | `false` | 绕过 Jina 的服务端缓存 |
| `target_selector` | `null` | 要抽取内容的 CSS 选择器 |
| `wait_for_selector` | `null` | 等待的 CSS 选择器 |

### Defuddle 设置

[Defuddle](https://defuddle.md) 抽取干净的文章正文，返回带丰富 frontmatter 的 Markdown。免费，不需要 key。

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `timeout` | `30` | 请求超时（秒） |
| `rpm` | `20` | 每分钟请求数 |

### Cloudflare 设置

Cloudflare 提供两样东西，分别选用：**Browser Rendering**（`-s cloudflare`）为 URL 抓取渲染后的 HTML，**Workers AI toMarkdown**（`-b cloudflare`）转换文件。

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `api_token` | `null` | API token（支持 `env:`） |
| `account_id` | `null` | 账户 ID（支持 `env:`） |
| `timeout` | `30000` | Browser Rendering 超时（毫秒） |
| `wait_until` | `networkidle0` | `load`、`domcontentloaded` 或 `networkidle0` |
| `cache_ttl` | `0` | Browser Rendering 缓存 TTL（秒） |
| `reject_resource_patterns` | `null` | 拦截匹配的请求，如 `["/\\.css$/"]` |
| `user_agent` | `null` | 自定义 User-Agent |
| `cookies` | `null` | `[{"name": "k", "value": "v", "url": "..."}]` |
| `wait_for_selector` | `null` | 等待的 CSS 选择器 |
| `http_credentials` | `null` | `{"username": "u", "password": "p"}` |
| `convert_enabled` | `false` | 开启 Workers AI toMarkdown 文件转换 |

获取凭据：

1. **账户 ID**：在[控制台](https://dash.cloudflare.com/)的 URL 里，`dash.cloudflare.com/<account_id>/...`。
2. **API token**：[My Profile → API Tokens](https://dash.cloudflare.com/profile/api-tokens)，*Create Token*，自定义 token，给你的账户加上 *Browser Rendering: Edit* 和 *Workers AI: Read* 权限。
3. **启用 Browser Rendering**：在 *Workers & Pages → Browser Rendering* 下开启。免费套餐可用。

```bash
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
```

免费套餐允许两个并发浏览器会话，markitai 会串行发请求并在限流时重试。反爬严格的站点（比如 x.com）走 Cloudflare 可能失败，改用 `-s playwright` 或 `-s jina`。文件转换方面内置转换器通常效果更好，toMarkdown 主要用于 markitai 本地转不了的格式。

### 抓取策略、域名配置与回退模式 {#fetch-policy-domain-profiles}

策略引擎按域名排定策略顺序，并记住哪些域名需要浏览器。它的选项、域名配置字段和内置配置都写在[抓取策略](/zh/guide/fetch-policy#配置)一页。

两条容易踩坑的规则：自定义的 `domain_profiles` 条目会整个替换该域名的内置配置，而不是合并；`auto` 把 `fallback_patterns` 里的域名都当成重 JavaScript，直接从浏览器开始。

### 代理

`HTTPS_PROXY`、`HTTP_PROXY` 和 `ALL_PROXY` 都生效，`NO_PROXY` 是绕过列表。都没设时用操作系统代理：Windows 的 Internet 设置、macOS 的网络设置，以及 Linux 上 GNOME 或 KDE 桌面的手动 HTTP 代理。PAC、纯 SOCKS 和带认证的桌面代理不会导入，请改设环境变量。

## 缓存配置

LLM 结果缓存在 `~/.markitai/cache.db`，同一份文档再转一次不花钱。

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 缓存 LLM 结果 |
| `no_cache` | `false` | 跳过读取但照常写入（同 `--no-cache`） |
| `no_cache_patterns` | `[]` | 绕过缓存的 glob |
| `max_size_bytes` | `536870912` | 缓存上限（512 MB） |
| `global_dir` | `~/.markitai` | 缓存目录 |

```bash
markitai cache stats --verbose         # 缓存了什么，按模型列出
markitai cache clear
markitai document.pdf --no-cache       # 这一次绕过缓存
markitai ./docs --no-cache-for "*.pdf"
```

## 输出配置

| 设置 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `dir` | — | `null` | 输出目录 |
| `on_conflict` | `rename`、`overwrite`、`skip` | `rename` | 输出文件已存在时怎么办 |
| `allow_symlinks` | — | `false` | 允许输出路径里有符号链接 |
| `report` | `true`、`false`、`null` | `null` | 写 JSON 报告。`null` 只在批量运行时写 |
| `profile` | `rag`、`obsidian`、`okf`、`null` | `null` | [输出 Profile](./output-profiles.md) |
| `wikilinks` | `true`、`false` | `false` | 用 `obsidian` 时把图片链接写成 `![[assets/x.png]]` |

## 日志配置

设了 `dir` 才会写日志文件。

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `level` | `INFO` | `DEBUG`、`INFO`、`WARNING`、`ERROR` 或 `CRITICAL` |
| `format` | `text` | `text` 或 `json` |
| `dir` | `null` | 日志目录 |
| `rotation` | `10 MB` | 文件超过这个大小就轮转 |
| `retention` | `7 days` | 删除更早的日志 |

## 安全配置

PDF 里可能藏着看不见的文字（白底白字、零字号、页面之外），它们会悄悄进入 Markdown，也会进入由此生成的 LLM 提示词。

| 设置 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `pdf_sanitize` | `off`、`warn`、`remove` | `warn` | `warn` 记录哪些页面有隐藏文字，`remove` 顺便删掉，`off` 不检查 |

## 自定义提示词

每个 LLM 任务都有一条 system 提示词（角色和规则）和一条 user 提示词（内容模板）。往提示词目录里放 Markdown 文件，或在配置里指向某个文件，就能覆盖：

```text
~/.markitai/prompts/
├── cleaner_system.md            # 文档清洗
├── cleaner_user.md
├── image_caption_system.md      # alt 文本
├── image_description_system.md  # 图片描述
├── document_process_system.md   # 文档处理
└── url_enhance_system.md        # URL 增强
```

```json
{
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner_system": "/path/to/my-cleaner-system.md"
  }
}
```

可用的键有 `cleaner`、`image_caption`、`image_description`、`image_analysis`、`document_process`、`document_vision` 和 `url_enhance`，各带 `_system` 和 `_user` 两个变体。

## 中国大陆用户指南

### 安装脚本镜像加速

安装脚本会检测代理环境变量（`HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`）。没有代理时，它会问你要不要用国内镜像：

| 镜像源 | PyPI | npm | 推荐地域 |
|--------|------|-----|----------|
| **清华 TUNA**（默认） | `pypi.tuna.tsinghua.edu.cn` | `registry.npmmirror.com` | 北方 / 通用 |
| **阿里云** | `mirrors.aliyun.com` | `registry.npmmirror.com` | 东部 |
| **腾讯云** | `mirrors.cloud.tencent.com` | `mirrors.cloud.tencent.com` | 南方 |
| **华为云** | `repo.huaweicloud.com` | `mirrors.huaweicloud.com` | 北方 |

Playwright 浏览器统一走 npmmirror CDN（`cdn.npmmirror.com`）。`MARKITAI_USE_MIRROR=1` 总是提供镜像选择，`0` 从不询问。

也可以在运行脚本前手动设置（以清华 TUNA 为例）：

::: code-group
```bash [macOS / Linux]
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PLAYWRIGHT_DOWNLOAD_HOST="https://cdn.npmmirror.com/binaries/playwright"
export NPM_CONFIG_REGISTRY="https://registry.npmmirror.com"
```

```powershell [Windows]
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:PLAYWRIGHT_DOWNLOAD_HOST = "https://cdn.npmmirror.com/binaries/playwright"
$env:NPM_CONFIG_REGISTRY = "https://registry.npmmirror.com"
```
:::

### LLM API 访问

| 提供商 | 可用性 | 说明 |
|--------|--------|------|
| **DeepSeek** | 直连 | 直接用 `deepseek/deepseek-v4-flash` |
| **Ollama** | 离线 | 本地模型，如 `ollama/llama3.2` |
| **API 中转** | 通过中转 | 用 `api_base` 指向第三方中转服务，写法见[自定义 API 端点](#自定义-api-端点) |
| **OpenAI / Claude / Gemini** | 需代理 | 走代理或 `api_base` 中转 |

### 代理配置

已有代理时，设环境变量即可对所有网络请求生效（导入规则见[代理](#代理)一节）：

::: code-group
```bash [macOS / Linux]
export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
```

```powershell [Windows]
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:HTTP_PROXY = "http://127.0.0.1:7890"
```
:::

设了代理环境变量后，安装脚本会跳过镜像配置。

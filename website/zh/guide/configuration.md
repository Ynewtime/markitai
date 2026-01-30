# 配置说明

## 配置优先级

Markitai 使用以下优先级顺序（从高到低）：

1. 命令行参数
2. 环境变量
3. 配置文件
4. 默认值

## 配置文件

Markitai 按以下顺序查找配置文件：

1. `--config` 参数指定的路径
2. `MARKITAI_CONFIG` 环境变量
3. `./markitai.json`（当前目录）
4. `~/.markitai/config.json`（用户主目录）

### 初始化配置

```bash
# 在当前目录创建配置文件
markitai config init

# 在指定位置创建
markitai config init -o ~/.markitai/
```

### 查看配置

```bash
# 列出所有设置
markitai config list

# 获取特定值
markitai config get llm.enabled

# 设置值
markitai config set llm.enabled true
```

### 完整配置示例

```json
{
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY"
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120
    },
    "concurrency": 10
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
    }
  },
  "ocr": {
    "enabled": false,
    "lang": "en"
  },
  "screenshot": {
    "enabled": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  },
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  },
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "scan_max_depth": 5,
    "scan_max_files": 10000
  },
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 1000
    },
    "jina": {
      "api_key": null,
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  },
  "output": {
    "on_conflict": "rename"
  },
  "log": {
    "level": "INFO",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  },
  "prompts": {
    "dir": "~/.markitai/prompts"
  }
}
```

::: tip
使用 `env:VAR_NAME` 语法在配置文件中引用环境变量。
:::

## 环境变量

### API 密钥

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API 密钥 |
| `GEMINI_API_KEY` | Google Gemini API 密钥 |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 |
| `JINA_API_KEY` | Jina Reader API 密钥 |

### Markitai 设置

| 变量 | 说明 |
|------|------|
| `MARKITAI_CONFIG` | 配置文件路径 |
| `MARKITAI_LOG_DIR` | 日志文件目录 |

## LLM 配置

### 支持的提供商

Markitai 通过 [LiteLLM](https://docs.litellm.ai/) 支持多个 LLM 提供商：

- OpenAI (GPT-5.2, GPT-5-mini)
- Anthropic (Claude 3.5/4)
- Google (Gemini 2.x)
- DeepSeek
- OpenRouter
- Ollama（本地模型）

#### 本地提供商（基于订阅）

Markitai 还支持使用 CLI 认证和订阅额度的本地提供商：

- **Claude Agent** (`claude-agent/`): 使用 [Claude Agent SDK](https://github.com/anthropics/claude-code) 通过 Claude Code CLI 认证
- **GitHub Copilot** (`copilot/`): 使用 [GitHub Copilot SDK](https://github.com/github/copilot-sdk) 通过 Copilot CLI 认证

这些提供商需要：
1. 安装并认证对应的 CLI 工具
2. 可选 SDK 包：`pip install markitai[claude-agent]` 或 `pip install markitai[copilot]`

### 模型命名

使用 LiteLLM 模型命名规范：

```
provider/model-name
```

示例：
- `openai/gpt-4o`
- `anthropic/claude-sonnet-4-20250514`
- `gemini/gemini-2.5-flash`
- `deepseek/deepseek-chat`
- `ollama/llama3.2`
- `claude-agent/sonnet`（本地，需要 Claude Code CLI）
- `copilot/gpt-5.2`（本地，需要 Copilot CLI）

Claude Agent SDK 支持的模型：
- 别名（推荐）：`sonnet`、`opus`、`haiku`、`inherit`
- 完整模型字符串：`claude-sonnet-4-5-20250929`、`claude-opus-4-5-20251101`、`claude-opus-4-1-20250805`

GitHub Copilot SDK 支持的模型：
- OpenAI: `gpt-5.2`、`gpt-5.1`、`gpt-5-mini`、`gpt-5.1-codex`
- Anthropic: `claude-sonnet-4.5`、`claude-opus-4.5`、`claude-haiku-4.5`
- Google: `gemini-2.5-pro`、`gemini-3-flash`
- 可用性取决于您的 Copilot 订阅

::: warning 模型下线通知
以下模型将于 **2025年2月13日** 下线：
- `gpt-4o`、`gpt-4.1`、`gpt-4.1-mini`、`o4-mini`、`gpt-5`

请在截止日期前迁移到 `gpt-5.2` 或其他支持的模型。
:::

::: tip 本地提供商支持 Vision
本地提供商（`claude-agent/`、`copilot/`）通过文件附件支持图片分析（`--alt`、`--desc`）。请确保使用支持 vision 的模型（如 `copilot/gpt-5.2`、`copilot/claude-sonnet-4.5`）。
:::

::: tip 本地提供商故障排除
常见错误和解决方案：

| 错误 | 解决方案 |
|------|----------|
| "SDK 未安装" | `pip install markitai[copilot]` 或 `pip install markitai[claude-agent]` |
| "CLI 未找到" | 安装并认证 CLI 工具（[Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli)、[Claude Code](https://claude.ai/code)） |
| "未认证" | 运行 `copilot auth login` 或 `claude auth login` |
| "速率限制" | 等待后重试，或检查订阅额度 |
:::

### Vision 模型

对于图片分析（`--alt`、`--desc`），Markitai 自动路由到支持视觉的模型。视觉能力默认**自动检测**自 litellm，大多数模型无需手动配置。

如需显式覆盖自动检测，设置 `supports_vision`：

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY"
        },
        "model_info": {
          "supports_vision": true  // 可选：省略时自动检测
        }
      }
    ]
  }
}
```

### 路由设置

配置 Markitai 如何在多个模型间分发请求：

```json
{
  "llm": {
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120,
      "fallbacks": []
    },
    "concurrency": 10
  }
}
```

| 设置 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `routing_strategy` | `simple-shuffle`, `least-busy`, `usage-based-routing`, `latency-based-routing` | `simple-shuffle` | 模型选择策略 |
| `num_retries` | 0-10 | 2 | 失败重试次数 |
| `timeout` | 秒 | 120 | 请求超时时间 |
| `concurrency` | 1-20 | 10 | 最大并发 LLM 请求数 |

## 图片配置

控制图片处理和压缩：

```json
{
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
    }
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `alt_enabled` | `false` | 通过 LLM 生成 alt 文本 |
| `desc_enabled` | `false` | 生成图片描述文件 |
| `compress` | `true` | 压缩图片 |
| `quality` | `75` | JPEG/WebP 质量 (1-100) |
| `format` | `jpeg` | 输出格式：`jpeg`, `png`, `webp` |
| `max_width` | `1920` | 最大宽度（像素） |
| `max_height` | `99999` | 最大高度（像素，实际无限制） |
| `filter.min_width` | `50` | 跳过宽度小于此值的图片 |
| `filter.min_height` | `50` | 跳过高度小于此值的图片 |
| `filter.min_area` | `5000` | 跳过面积小于此值的图片 |
| `filter.deduplicate` | `true` | 去除重复图片 |

## 截图配置

为文档和 URL 启用截图捕获：

```json
{
  "screenshot": {
    "enabled": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  }
}
```

启用后（`--screenshot` 或 `--preset rich`）：

- **PDF/PPTX**: 将每个页面/幻灯片渲染为 JPEG 图片
- **URL**: 使用 agent-browser 捕获全页面截图

| 设置 | 默认值 | 描述 |
|------|--------|------|
| `enabled` | `false` | 启用截图捕获 |
| `viewport_width` | `1920` | URL 截图的浏览器视口宽度 |
| `viewport_height` | `1080` | URL 截图的浏览器视口高度 |
| `quality` | `75` | JPEG 压缩质量 (1-100) |
| `max_height` | `10000` | 截图最大高度（像素） |

截图保存在 `output/screenshots/` 目录。

::: tip
对于 URL，启用 `--screenshot` 会在需要时自动将抓取策略升级为 `browser`，确保页面完全渲染后再捕获。
:::

## OCR 配置

配置扫描文档的光学字符识别。Markitai 使用 [RapidOCR](https://github.com/RapidAI/RapidOCR)（ONNX Runtime + OpenCV）进行 OCR 处理。

```json
{
  "ocr": {
    "enabled": false,
    "lang": "en"
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 为 PDF 启用 OCR |
| `lang` | `en` | RapidOCR 语言代码 |

支持的语言代码：
- `en` - 英语
- `zh` / `ch` - 中文（简体）
- `ja` / `japan` - 日语
- `ko` / `korean` - 韩语
- `ar` / `arabic` - 阿拉伯语
- `th` - 泰语
- `latin` - 拉丁语系

::: tip
RapidOCR 已作为依赖包含，开箱即用，无需额外安装。
:::

## 批处理配置

控制并行处理：

```json
{
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "scan_max_depth": 5,
    "scan_max_files": 10000
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `concurrency` | `10` | 最大并发文件转换数 |
| `url_concurrency` | `5` | 最大并发 URL 抓取数（与文件分离） |
| `scan_max_depth` | `5` | 最大目录扫描深度 |
| `scan_max_files` | `10000` | 单次运行最大处理文件数 |

::: tip
URL 抓取使用独立的并发池，因为 URL 可能有较高延迟（如浏览器渲染的页面）。这可以防止慢速 URL 阻塞本地文件处理。
:::

## URL 抓取配置

配置 URL 的抓取方式：

```json
{
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 1000
    },
    "jina": {
      "api_key": "env:JINA_API_KEY",
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### 抓取策略

| 策略 | 说明 |
|------|------|
| `auto` | 自动检测：对 `fallback_patterns` 中的模式使用浏览器，否则使用静态 |
| `static` | 使用 MarkItDown 内置的 URL 转换器（快速，无 JS） |
| `browser` | 使用 agent-browser 处理 JS 渲染的页面（支持 SPA） |
| `jina` | 使用 Jina Reader API |

### 浏览器设置

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `command` | `agent-browser` | agent-browser 路径 |
| `timeout` | `30000` | 页面加载超时（毫秒） |
| `wait_for` | `domcontentloaded` | 等待条件：`load`, `domcontentloaded`, `networkidle` |
| `extra_wait_ms` | `1000` | JS 渲染额外等待时间 |

### 回退模式

匹配这些模式的网站自动使用浏览器策略：

```json
{
  "fetch": {
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

## 缓存配置

Markitai 使用全局缓存，存储在 `~/.markitai/cache.db`。

```json
{
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 启用 LLM 结果缓存 |
| `no_cache_patterns` | `[]` | 跳过缓存的 glob 模式 |
| `max_size_bytes` | `536870912` (512MB) | 最大缓存大小 |
| `global_dir` | `~/.markitai` | 全局缓存目录 |

### 缓存命令

```bash
# 查看缓存统计
markitai cache stats

# 查看详细统计（条目、按模型分组）
markitai cache stats -v

# 指定显示数量
markitai cache stats -v --limit 50

# 清除缓存
markitai cache clear
markitai cache clear -y  # 跳过确认
```

### 禁用缓存

```bash
# 整次运行禁用
markitai document.pdf --no-cache

# 对特定文件/模式禁用
markitai ./docs --no-cache-for "*.pdf"
markitai ./docs --no-cache-for "file1.pdf,reports/**"
```

## 输出配置

控制输出文件处理：

```json
{
  "output": {
    "on_conflict": "rename"
  }
}
```

| 设置 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `on_conflict` | `rename`, `overwrite`, `skip` | `rename` | 处理已存在文件的方式 |

## 日志配置

配置日志行为：

```json
{
  "log": {
    "level": "INFO",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `level` | `INFO` | 日志级别：`DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `dir` | `~/.markitai/logs` | 日志文件目录 |
| `rotation` | `10 MB` | 文件超过此大小时轮转 |
| `retention` | `7 days` | 删除早于此时间的日志 |

## 自定义提示词

自定义不同任务的 LLM 提示词。每个提示词拆分为 **system**（角色定义）和 **user**（内容模板）两部分：

```json
{
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner_system": null,
    "cleaner_user": null,
    "frontmatter_system": null,
    "frontmatter_user": null,
    "image_caption_system": null,
    "image_caption_user": null,
    "image_description_system": null,
    "image_description_user": null,
    "document_process_system": null,
    "document_process_user": null
  }
}
```

在提示词目录创建自定义提示词文件：

```
~/.markitai/prompts/
├── cleaner_system.md          # 文档清理角色和规则
├── cleaner_user.md            # 文档清理内容模板
├── frontmatter_system.md      # 元数据提取角色
├── frontmatter_user.md        # 元数据提取模板
├── image_caption_system.md    # Alt 文本生成角色
├── image_caption_user.md      # Alt 文本内容模板
└── document_enhance_system.md # 视觉增强角色
```

指定特定的提示词文件路径：

```json
{
  "prompts": {
    "cleaner_system": "/path/to/my-cleaner-system.md",
    "cleaner_user": "/path/to/my-cleaner-user.md"
  }
}
```

::: tip
system/user 拆分可以防止 LLM 意外地将提示词指令包含在其输出中。system 提示词定义角色和规则，而 user 提示词包含实际要处理的内容。
:::

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
      "num_retries": 3,
      "timeout": 120
    },
    "concurrency": 5
  },
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 85,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 1080,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 2500,
      "deduplicate": true
    }
  },
  "ocr": {
    "enabled": false,
    "lang": "eng+chi_sim"
  },
  "screenshot": {
    "enabled": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 85,
    "max_height": 10000
  },
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 1073741824,
    "global_dir": "~/.markitai"
  },
  "batch": {
    "concurrency": 10,
    "url_concurrency": 3,
    "scan_max_depth": 10,
    "scan_max_files": 10000
  },
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 2000
    },
    "jina": {
      "api_key": null,
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com"]
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
| `MARKITAI_PROMPT_DIR` | 自定义提示词目录 |

## LLM 配置

### 支持的提供商

Markitai 通过 [LiteLLM](https://docs.litellm.ai/) 支持多个 LLM 提供商：

- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5/4)
- Google (Gemini 2.x)
- DeepSeek
- OpenRouter
- Ollama（本地模型）

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
      "num_retries": 3,
      "timeout": 120,
      "fallbacks": []
    },
    "concurrency": 5
  }
}
```

| 设置 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `routing_strategy` | `simple-shuffle`, `least-busy`, `usage-based-routing`, `latency-based-routing` | `simple-shuffle` | 模型选择策略 |
| `num_retries` | 0-10 | 3 | 失败重试次数 |
| `timeout` | 秒 | 120 | 请求超时时间 |
| `concurrency` | 1-20 | 5 | 最大并发 LLM 请求数 |

## 图片配置

控制图片处理和压缩：

```json
{
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 85,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 1080,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 2500,
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
| `quality` | `85` | JPEG/WebP 质量 (1-100) |
| `format` | `jpeg` | 输出格式：`jpeg`, `png`, `webp` |
| `max_width` | `1920` | 最大宽度（像素） |
| `max_height` | `1080` | 最大高度（像素） |
| `filter.min_width` | `50` | 跳过宽度小于此值的图片 |
| `filter.min_height` | `50` | 跳过高度小于此值的图片 |
| `filter.min_area` | `2500` | 跳过面积小于此值的图片 |
| `filter.deduplicate` | `true` | 去除重复图片 |

## 截图配置

为文档和 URL 启用截图捕获：

```json
{
  "screenshot": {
    "enabled": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 85,
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
| `quality` | `85` | JPEG 压缩质量 (1-100) |
| `max_height` | `10000` | 截图最大高度（像素） |

截图保存在 `output/screenshots/` 目录。

::: tip
对于 URL，启用 `--screenshot` 会在需要时自动将抓取策略升级为 `browser`，确保页面完全渲染后再捕获。
:::

## OCR 配置

配置扫描文档的光学字符识别：

```json
{
  "ocr": {
    "enabled": false,
    "lang": "eng+chi_sim"
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 为 PDF 启用 OCR |
| `lang` | `eng+chi_sim` | Tesseract 语言代码 |

常用语言代码：
- `eng` - 英语
- `chi_sim` - 简体中文
- `chi_tra` - 繁体中文
- `jpn` - 日语
- `kor` - 韩语

::: warning
OCR 需要安装 Tesseract。参见[快速开始](/zh/guide/getting-started#可选依赖)。
:::

## 批处理配置

控制并行处理：

```json
{
  "batch": {
    "concurrency": 10,
    "url_concurrency": 3,
    "scan_max_depth": 10,
    "scan_max_files": 10000
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `concurrency` | `10` | 最大并发文件转换数 |
| `url_concurrency` | `3` | 最大并发 URL 抓取数（与文件分离） |
| `scan_max_depth` | `10` | 最大目录扫描深度 |
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
      "extra_wait_ms": 2000
    },
    "jina": {
      "api_key": "env:JINA_API_KEY",
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com"]
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
| `extra_wait_ms` | `2000` | JS 渲染额外等待时间 |

### 回退模式

匹配这些模式的网站自动使用浏览器策略：

```json
{
  "fetch": {
    "fallback_patterns": ["x.com", "twitter.com", "spa-site.com"]
  }
}
```

## 缓存配置

Markitai 使用双层缓存系统：

- **项目缓存**：当前目录的 `.markitai/cache/`
- **全局缓存**：`~/.markitai/cache/`

```json
{
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 1073741824,
    "global_dir": "~/.markitai"
  }
}
```

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 启用 LLM 结果缓存 |
| `no_cache_patterns` | `[]` | 跳过缓存的 glob 模式 |
| `max_size_bytes` | `1073741824` (1GB) | 最大缓存大小 |
| `global_dir` | `~/.markitai` | 全局缓存目录 |

### 缓存命令

```bash
# 查看缓存统计
markitai cache stats

# 查看详细统计（条目、按模型分组）
markitai cache stats -v

# 指定显示数量
markitai cache stats -v --limit 50

# 清除所有缓存
markitai cache clear

# 只清除项目缓存
markitai cache clear --scope project
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

自定义不同任务的 LLM 提示词：

```json
{
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner": null,
    "frontmatter": null,
    "image_caption": null,
    "image_description": null,
    "image_analysis": null,
    "page_content": null,
    "document_enhance": null
  }
}
```

在提示词目录创建自定义提示词文件：

```
~/.markitai/prompts/
├── cleaner.md          # 文档清理提示词
├── frontmatter.md      # 元数据提取提示词
├── image_caption.md    # Alt 文本生成
├── image_description.md # 图片描述
└── document_enhance.md # 基于视觉的增强
```

指定特定的提示词文件路径：

```json
{
  "prompts": {
    "cleaner": "/path/to/my-cleaner.md"
  }
}
```

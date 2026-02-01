# AI 工具配置指南

本文档介绍如何配置 Markitai 使用各种 AI 工具进行 LLM 增强。

---

## 概述

Markitai 支持多种 LLM 提供商：

| 提供商 | 前缀 | 计费方式 | 说明 |
|--------|------|----------|------|
| Claude Code CLI | `claude-agent/` | 订阅制 | 使用 Claude 订阅 |
| GitHub Copilot CLI | `copilot/` | 订阅制 | 使用 Copilot 订阅 |
| OpenAI | `openai/` 或直接使用 | API 按量计费 | 标准 API |
| Anthropic | `anthropic/` | API 按量计费 | 标准 API |
| 其他 100+ 模型 | 参见 LiteLLM | 各有不同 | 通过 LiteLLM 支持 |

---

## Claude Code CLI

### 安装

1. **安装 Claude Code CLI**
   ```bash
   # macOS/Linux
   curl -fsSL https://claude.ai/install.sh | sh

   # Windows
   irm https://claude.ai/install.ps1 | iex
   ```

2. **认证**
   ```bash
   claude login
   ```

3. **安装 Python SDK**
   ```bash
   uv sync --extra claude-agent
   # 或
   uv add claude-agent-sdk
   ```

### 配置

在 `markitai.json` 中配置：

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "claude-agent/sonnet"
        }
      }
    ]
  }
}
```

### 支持的模型别名

| 别名 | 解析为 |
|------|--------|
| `claude-agent/sonnet` | 最新 Sonnet 模型 |
| `claude-agent/opus` | 最新 Opus 模型 |
| `claude-agent/haiku` | 最新 Haiku 模型 |
| `claude-agent/inherit` | 继承 CLI 设置 |

### 使用

```bash
# 使用 Claude Agent 进行 LLM 增强
markitai document.pdf --llm
```

### 成本追踪

Claude Code 使用订阅制，成本按估算的 token 数量记录：

```
LLM 成本报告:
├── 总 token: 15,234 (输入: 12,500, 输出: 2,734)
├── 估算成本: $0.00 (订阅制)
└── 模型: claude-agent/sonnet
```

---

## GitHub Copilot CLI

### 安装

1. **安装 GitHub Copilot CLI**
   ```bash
   # 确保已安装 GitHub CLI
   gh extension install github/gh-copilot
   ```

2. **认证**
   ```bash
   gh auth login
   gh copilot --version  # 验证安装
   ```

3. **安装 Python SDK**
   ```bash
   uv sync --extra copilot
   # 或
   uv add github-copilot-sdk
   ```

### 配置

在 `markitai.json` 中配置：

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "copilot/gpt-4.1"
        }
      }
    ]
  }
}
```

### 支持的模型

| 模型 | 说明 |
|------|------|
| `copilot/gpt-4.1` | GPT-4.1 |
| `copilot/gpt-4o` | GPT-4o |
| `copilot/claude-sonnet-4.5` | Claude Sonnet 4.5 |
| `copilot/o1` | OpenAI o1 |

### 使用

```bash
markitai document.pdf --llm
```

---

## 标准 API 提供商

### OpenAI

1. **获取 API 密钥**

   访问 [platform.openai.com](https://platform.openai.com/api-keys)

2. **配置环境变量**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   或在配置文件中使用 `env:` 语法：
   ```json
   {
     "llm": {
       "model_list": [
         {
           "model_name": "default",
           "litellm_params": {
             "model": "gpt-4o-mini",
             "api_key": "env:OPENAI_API_KEY"
           }
         }
       ]
     }
   }
   ```

### Anthropic

1. **获取 API 密钥**

   访问 [console.anthropic.com](https://console.anthropic.com/)

2. **配置**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

   ```json
   {
     "llm": {
       "model_list": [
         {
           "model_name": "default",
           "litellm_params": {
             "model": "anthropic/claude-sonnet-4-20250514"
           }
         }
       ]
     }
   }
   ```

### Azure OpenAI

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "azure/gpt-4o",
          "api_base": "https://your-resource.openai.azure.com",
          "api_key": "env:AZURE_API_KEY",
          "api_version": "2024-02-15-preview"
        }
      }
    ]
  }
}
```

---

## 配置优先级

Markitai 按以下优先级加载配置：

```
1. 配置文件 (markitai.json)
2. 默认值
```

### 配置文件位置

按以下顺序搜索：

1. `--config` 指定的路径
2. `MARKITAI_CONFIG` 环境变量
3. 当前目录 `./markitai.json`
4. 用户主目录 `~/.markitai/config.json`

---

## 模型路由

可以配置多个模型进行负载均衡或后备：

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "claude-agent/sonnet"
        }
      },
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gpt-4o-mini"
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle"
    }
  }
}
```

---

## 常见问题

### Claude Agent SDK 未安装

```
错误: claude-agent/ 模型需要 Claude Agent SDK
```

**解决方案**:
```bash
uv sync --extra claude-agent
# 或
uv add claude-agent-sdk
```

### Claude Code CLI 未认证

```
错误: Claude Code CLI 未认证
```

**解决方案**:
```bash
claude login
```

### Copilot SDK 未安装

```
错误: copilot/ 模型需要 GitHub Copilot SDK
```

**解决方案**:
```bash
uv sync --extra copilot
# 或
uv add github-copilot-sdk
```

### API 密钥无效

```
错误: API key is invalid
```

**解决方案**:
1. 检查环境变量是否正确设置
2. 确认 API 密钥未过期
3. 验证密钥权限

### 模型不存在

```
错误: Model not found
```

**解决方案**:
1. 检查模型名称拼写
2. 确认账户有权访问该模型
3. 对于 Claude Agent，尝试使用别名 (`sonnet`, `opus`, `haiku`)

---

## 调试

启用详细日志：

```bash
markitai document.pdf --llm --verbose
```

查看 LLM 调用详情：

```bash
export LITELLM_LOG=DEBUG
markitai document.pdf --llm
```

---

## 相关文档

- [配置指南](../website/guide/configuration.md)
- [CLI 参考](../website/guide/cli.md)
- [LiteLLM 文档](https://docs.litellm.ai/)

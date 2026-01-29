---
source:
- https://docs.github.com/en/copilot/reference/ai-models/supported-models
- https://github.blog/changelog/2025-10-03-github-copilot-cli-enhanced-model-selection-image-support-and-streamlined-ui/
updated: 2026-01-27
---

# GitHub Copilot 支持的模型

> 模型可用性取决于您的 Copilot 订阅计划。

## OpenAI 模型

| 模型标识符 | 状态 | 说明 |
|-----------|------|------|
| `gpt-4.1` | GA | 推荐用于一般任务 |
| `gpt-5` | ⚠️ 2026-02-17 关闭 | 请迁移到 gpt-5.1+ |
| `gpt-5-mini` | GA | 轻量级，速度快 |
| `gpt-5.1` | GA | |
| `gpt-5.1-codex` | GA | 代码优化版本 |
| `gpt-5.1-codex-mini` | Preview | |
| `gpt-5.1-codex-max` | GA | |
| `gpt-5.2` | GA | 最新版本 |
| `gpt-5.2-codex` | GA | 代码优化版本 |

## Anthropic 模型

| 模型标识符 | 状态 | 说明 |
|-----------|------|------|
| `claude-haiku-4.5` | GA | 快速、经济 |
| `claude-opus-4.1` | ⚠️ 2026-02-17 关闭 | 请迁移到 opus-4.5 |
| `claude-opus-4.5` | GA | 最强大 |
| `claude-sonnet-4` | GA | 平衡性能与速度 |
| `claude-sonnet-4.5` | GA | 推荐用于代码任务 |

## Google 模型

| 模型标识符 | 状态 | 说明 |
|-----------|------|------|
| `gemini-2.5-pro` | GA | |
| `gemini-3-flash` | Preview | 快速响应 |
| `gemini-3-pro` | Preview | |

## 其他模型

| 模型标识符 | 提供商 | 状态 |
|-----------|--------|------|
| `grok-code-fast-1` | xAI | GA |
| `raptor-mini` | GitHub (微调 GPT-5 mini) | Preview |

## 在 markitai 中使用

### 方式 1: 配置文件 (推荐)

创建 `markitai.json`:

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "copilot/claude-sonnet-4.5"
        }
      }
    ]
  }
}
```

然后运行:

```bash
markitai doc.pdf --llm
```

### 方式 2: 指定配置文件路径

```bash
# 创建临时配置
cat > /tmp/copilot.json << 'EOF'
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
EOF

# 使用配置文件
markitai doc.pdf --llm -c /tmp/copilot.json
```

## 运行时获取模型列表

### 通过 CLI

```bash
copilot /model
```

### 通过 SDK

```python
import asyncio
from copilot import CopilotClient

async def list_models():
    client = CopilotClient()
    await client.start()

    # SDK 提供运行时获取可用模型的方法
    models = await client.get_available_models()
    for m in models:
        print(m)

    await client.stop()

asyncio.run(list_models())
```

## 注意事项

1. **订阅限制**: 部分模型可能需要特定的 Copilot 订阅计划
2. **区域可用性**: 某些模型可能在特定区域不可用
3. **配额限制**: Premium 模型可能有使用配额限制
4. **版本更新**: 模型列表会随时间更新，请参考官方文档获取最新信息
5. **Vision 支持**: Copilot provider 支持图片附件 (`--alt`, `--desc`)，但需要选择支持 vision 的模型

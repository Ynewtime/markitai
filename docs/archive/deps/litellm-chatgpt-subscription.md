# LiteLLM × ChatGPT Subscription 完整技术报告

> **版本信息：** litellm v1.82.0 · 数据来源：PyPI 安装包源码 + 官方文档 · 2026 年 3 月

---

## 目录

1. [背景与上下文](#1-背景与上下文)
2. [架构概览](#2-架构概览)
3. [认证：OAuth Device Code Flow](#3-认证oauth-device-code-flow)
4. [API 覆盖与参数支持](#4-api-覆盖与参数支持)
5. [HTTP 请求细节](#5-http-请求细节)
6. [使用指南](#6-使用指南)
7. [限制与注意事项](#7-限制与注意事项)
8. [事实核查](#8-事实核查)
9. [总结](#9-总结)

---

## 1. 背景与上下文

ChatGPT Plus/Pro 订阅（$20–$200/月）与 OpenAI Platform API 是**完全独立的两套计费系统**。订阅用户无法通过标准 OpenAI SDK 的 `api.openai.com` 端点调用模型——API 使用始终需要单独付费的 API Key。这对已经在为 ChatGPT 订阅付费的开发者来说是长期痛点。

2026 年 1 月，LiteLLM 推出了解决方案：专属的 `chatgpt/` provider（PR #19030，v1.81.3，2026-01-26 发布），通过 OpenAI 自己的 **OAuth Device Code 协议**将请求路由到 ChatGPT 订阅——这与官方 Codex CLI 使用的协议完全相同。

> **重要：** 这不是逆向工程。LiteLLM 调用的 OAuth 端点、使用的 `client_id`（`app_EMoamEEZ73f0CkXaXp7hrann`），均来自官方 Codex CLI 的认证流程。

---

## 2. 架构概览

### 2.1 请求流程

```
Your Code
   ↓  litellm.completion(model="chatgpt/gpt-5.2", messages=[...])
LiteLLM Router
   ↓  检测到 "chatgpt/" 前缀 → 路由到 ChatGPTConfig
ChatGPTConfig.validate_environment()
   ↓  调用 Authenticator.get_access_token()
Authenticator
   ↓  读取 ~/.config/litellm/chatgpt/auth.json
   ↓  token 有效 → 直接返回 access_token
   ↓  token 过期 → 通过 https://auth.openai.com/oauth/token 刷新
   ↓  无 token  → 触发 OAuth Device Code Flow（交互式）
HTTP 请求
   ↓  POST https://chatgpt.com/backend-api/codex/responses
   ↓  Authorization: Bearer <access_token>
   ↓  originator: codex_cli_rs
   ↓  ChatGPT-Account-Id: <account_id>
响应 → 规范化为 OpenAI 格式 → 返回给调用方
```

### 2.2 模块结构（源码实测）

| 文件 | 职责 |
|------|------|
| `litellm/llms/chatgpt/authenticator.py` | OAuth 流程：device code、token 刷新、auth.json 读写 |
| `litellm/llms/chatgpt/common_utils.py` | 常量（URLs、client_id）、headers 构建、session 工具 |
| `litellm/llms/chatgpt/responses/transformation.py` | Responses API 处理器（Codex 模型的原生路径） |
| `litellm/llms/chatgpt/chat/transformation.py` | Chat Completions 处理器（桥接到 Responses API） |

---

## 3. 认证：OAuth Device Code Flow

### 3.1 完整流程

**Step 1 — 请求 device code**

```http
POST https://auth.openai.com/api/accounts/deviceauth/usercode
Content-Type: application/json

{"client_id": "app_EMoamEEZ73f0CkXaXp7hrann"}
```

**Step 2 — 终端打印验证码**

```
Sign in with ChatGPT using device code:
1) Visit https://auth.openai.com/codex/device
2) Enter code: XXXX-XXXX
Device codes are a common phishing target. Never share this code.
```

**Step 3 — 轮询 authorization code**

```http
POST https://auth.openai.com/api/accounts/deviceauth/token

{"device_auth_id": "...", "user_code": "XXXX-XXXX"}
```

每隔 5 秒轮询一次，最长等待 15 分钟。

**Step 4 — 交换 token**

```http
POST https://auth.openai.com/oauth/token

{
  "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
  "grant_type": "authorization_code",
  "code": "...",
  "code_verifier": "..."
}
```

返回 `access_token`、`refresh_token`、`id_token`。

**Step 5 — 持久化存储**

写入 `~/.config/litellm/chatgpt/auth.json`，后续请求复用，过期前自动刷新。

### 3.2 Token 生命周期

| 状态 | 行为 | 触发条件 |
|------|------|----------|
| Token 有效 | 直接使用，无网络请求 | JWT exp > now + 60s skew |
| Token 过期 | 自动用 refresh_token 刷新 | JWT exp ≤ now + 60s skew |
| 刷新失败 | 重新触发 Device Code Flow（交互式） | HTTP 错误或响应缺少字段 |
| 无 token 文件 | 触发 Device Code Flow（交互式） | auth.json 不存在或 JSON 损坏 |
| Device Code 冷却 | 静默等待最多 5 分钟再重试 | 防止频繁请求 device auth 端点 |

### 3.3 关键常量（源码实测）

```python
# litellm/llms/chatgpt/common_utils.py

CHATGPT_AUTH_BASE         = "https://auth.openai.com"
CHATGPT_DEVICE_CODE_URL   = ".../api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL  = ".../api/accounts/deviceauth/token"
CHATGPT_OAUTH_TOKEN_URL   = ".../oauth/token"
CHATGPT_DEVICE_VERIFY_URL = ".../codex/device"
CHATGPT_API_BASE          = "https://chatgpt.com/backend-api/codex"
CHATGPT_CLIENT_ID         = "app_EMoamEEZ73f0CkXaXp7hrann"

# Token 存储
DEFAULT_DIR  = ~/.config/litellm/chatgpt/
DEFAULT_FILE = auth.json

# 可通过环境变量覆盖
CHATGPT_TOKEN_DIR, CHATGPT_AUTH_FILE
```

---

## 4. API 覆盖与参数支持

### 4.1 端点支持

| 端点 | 状态 | 说明 |
|------|------|------|
| `/responses`（Responses API） | ✅ 原生支持 | 所有 chatgpt/ 模型的推荐路径，`store=false` 和 `stream=true` 被强制设置 |
| `/chat/completions`（Chat） | ✅ 桥接支持 | LiteLLM 将 `messages[]` 格式转换为 Responses API 格式后发送 |
| `/embeddings` | ❌ 不支持 | ChatGPT 订阅后端未暴露 embeddings 端点 |
| `/images` | ❌ 不支持 | 图像生成不可用 |
| `/audio` | ❌ 不支持 | 音频端点不可用 |

### 4.2 请求参数支持

后端强制执行严格的参数白名单，LiteLLM 会自动剔除不支持的字段：

| 参数 | 状态 | 说明 |
|------|------|------|
| `model` | ✅ | 必填，如 `gpt-5.2`、`gpt-5.2-codex` |
| `input` / `messages` | ✅ | Chat 路径的 `messages[]` 自动转换为 `input` |
| `instructions` | ✅ | LiteLLM 会在用户 instructions 前**自动注入 Codex 默认系统提示** |
| `tools` | ✅ | 函数调用 / 工具定义 |
| `tool_choice` | ✅ | 工具选择控制 |
| `reasoning` | ✅ | 推理力度（`low`/`medium`/`high`/`xhigh`） |
| `previous_response_id` | ✅ | Responses API 多轮对话 |
| `truncation` | ✅ | 上下文截断设置 |
| `stream` | ✅ | `/chat/completions` 遵守调用方的 stream 参数；后端始终使用流式 |
| `include` | ✅（自动） | LiteLLM 自动追加 `reasoning.encrypted_content` |
| **`max_tokens`** | ❌ **被剔除** | 后端拒绝此字段，LiteLLM 静默移除 |
| **`max_output_tokens`** | ❌ **被剔除** | 同上 |
| **`max_completion_tokens`** | ❌ **被剔除** | 同上 |
| **`metadata`** | ❌ **被剔除** | 后端拒绝 metadata，LiteLLM 静默移除 |
| `temperature` | ❌ 不在白名单 | 被剔除 |
| `top_p` | ❌ 不在白名单 | 被剔除 |

### 4.3 支持的模型（v1.82.0 实测）

```python
# 通过 litellm.model_cost 验证，所有模型 litellm_provider="chatgpt"，mode="responses"

chatgpt/gpt-5.2            context=64K    input_cost=None  output_cost=None
chatgpt/gpt-5.2-codex      context=128K   input_cost=None  output_cost=None
chatgpt/gpt-5.1-codex-max  context=128K   input_cost=None  output_cost=None
chatgpt/gpt-5.1-codex-mini context=64K    input_cost=None  output_cost=None
```

> **注意：** `input_cost_per_token` 和 `output_cost_per_token` 均为 `None`——使用订阅无需按 token 计费，LiteLLM 不会记录这些模型的费用。

> **更新型号：** `gpt-5.3-codex`（2026 年 3 月最新）仅通过 OAuth 认证可用，不在 OpenAI Platform API 上。若订阅包含此模型，可以尝试 `chatgpt/gpt-5.3-codex`，即使它不在 LiteLLM 的 cost map 中。

---

## 5. HTTP 请求细节

### 5.1 请求 Headers

```http
Authorization:      Bearer <access_token>
content-type:       application/json
accept:             text/event-stream
originator:         codex_cli_rs
user-agent:         codex_cli_rs/<litellm_version> (<OS> <version>; <arch>) <terminal>
ChatGPT-Account-Id: <account_id，从 JWT 中提取>
session_id:         <每次请求的 uuid4，或来自 litellm_params>
```

### 5.2 请求 Body（Responses API 路径）

```json
{
  "model": "gpt-5.2-codex",
  "input": [...],
  "instructions": "<Codex 默认系统提示 + 用户 instructions>",
  "stream": true,
  "store": false,
  "include": ["reasoning.encrypted_content"],
  "tools": [...],
  "tool_choice": ...,
  "reasoning": ...
}

// 以下字段不会被发送（LiteLLM 自动剔除）：
// max_tokens, max_output_tokens, max_completion_tokens, metadata
// temperature, top_p, frequency_penalty 等所有标准 OpenAI 参数
```

### 5.3 API Endpoint

```
Base URL:  https://chatgpt.com/backend-api/codex
Endpoint:  POST /responses
Full URL:  https://chatgpt.com/backend-api/codex/responses

环境变量覆盖：CHATGPT_API_BASE 或 OPENAI_CHATGPT_API_BASE
```

---

## 6. 使用指南

### 6.1 安装

```bash
pip install litellm>=1.81.3

# 首次调用会触发交互式 OAuth（需要浏览器登录）
# 后续调用复用缓存的 token，自动刷新
```

### 6.2 Python SDK — Responses API（推荐）

```python
import litellm

response = litellm.responses(
    model="chatgpt/gpt-5.2-codex",
    input="Write a Python hello world"
)
print(response)
```

### 6.3 Python SDK — Chat Completions（桥接）

```python
import litellm

response = litellm.completion(
    model="chatgpt/gpt-5.2",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### 6.4 流式输出

```python
response = litellm.completion(
    model="chatgpt/gpt-5.2",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### 6.5 LiteLLM Proxy / AI Gateway

```yaml
# config.yaml
model_list:
  - model_name: chatgpt/gpt-5.2
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.2

  - model_name: chatgpt/gpt-5.2-codex
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.2-codex
```

```bash
litellm --config config.yaml
```

```python
# 之后任何 OpenAI 兼容客户端均可调用
import openai
client = openai.OpenAI(base_url="http://localhost:4000", api_key="anything")
response = client.chat.completions.create(
    model="chatgpt/gpt-5.2",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 6.6 环境变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `CHATGPT_TOKEN_DIR` | `~/.config/litellm/chatgpt` | auth.json 存储目录 |
| `CHATGPT_AUTH_FILE` | `auth.json` | token 文件名 |
| `CHATGPT_API_BASE` | `https://chatgpt.com/backend-api/codex` | 覆盖 API 端点（如用于测试） |
| `OPENAI_CHATGPT_API_BASE` | （同上的别名） | `CHATGPT_API_BASE` 的别名 |
| `CHATGPT_ORIGINATOR` | `codex_cli_rs` | 覆盖 `originator` header |
| `CHATGPT_USER_AGENT` | （自动生成） | 覆盖 `User-Agent` header |
| `CHATGPT_USER_AGENT_SUFFIX` | （空） | 追加到自动生成的 User-Agent 后缀 |
| `CHATGPT_DEFAULT_INSTRUCTIONS` | （Codex 默认提示） | 覆盖注入到所有请求的默认系统指令 |

---

## 7. 限制与注意事项

### 7.1 参数限制（硬约束）

> ⚠️ **无法控制输出长度。** 后端会拒绝 `max_tokens`、`max_output_tokens`、`max_completion_tokens`，LiteLLM 静默剔除这些字段。模型会一直生成到自然结束。

| 限制 | 影响 |
|------|------|
| 无 max_tokens / 长度控制 | 模型自行决定何时停止，复杂 prompt 可能产生极长输出 |
| 无 temperature / top_p | 输出的随机性不可配置 |
| 无 metadata 字段 | 无法为请求附加自定义元数据 |
| `store=false` 强制 | 响应不会存储在 ChatGPT 服务器上（有利于隐私，但无历史记录） |
| 仅限 Codex 系列模型 | 无法通过订阅路由使用 gpt-4o、gpt-4.1、o1、o3、o4-mini |
| 无 embeddings | 嵌入 API 需要单独的 API Key |
| 首次登录需交互 | 第一次调用需要浏览器完成 OAuth，不能完全无人值守 |
| 订阅套餐限制 | gpt-5.3-codex 需要 Pro/Max；Plus 用户可能遇到"模型不可用"错误 |

### 7.2 系统指令注入

LiteLLM 会在每次请求的 `instructions` 字段前**自动注入 Codex CLI 的默认系统提示**（约 1500 字符），内容涵盖代码编辑规范、git 行为、响应格式和前端设计指南。如果你设置了自定义 instructions，它们会被**追加**在默认提示之后，而不是替换。

如需完全替换默认提示，设置环境变量：

```bash
export CHATGPT_DEFAULT_INSTRUCTIONS="Your custom instructions here"
```

### 7.3 ToS 风险

> ⚠️ OpenAI 的 ChatGPT 订阅服务条款并未明确授权通过 Codex 后端 API 进行编程访问（官方 Codex CLI 工具除外）。LiteLLM 使用了与 Codex CLI 相同的 OAuth client_id 和端点——Codex CLI 是官方工具——但在生产服务或商业产品中使用此方案，或进行大规模自动化调用，仍可能违反 OpenAI 的使用政策。**建议将此功能定位为开发者个人用途。**

---

## 8. 事实核查

以下所有声明均已通过 litellm v1.82.0（从 PyPI 安装）的源码逐一验证。

**最关键的实测验证：** 在运行 `get_llm_provider("chatgpt/gpt-5.2")` 时，LiteLLM 直接发起了真实的 OAuth 流程并打印：

```
Sign in with ChatGPT using device code:
1) Visit https://auth.openai.com/codex/device
2) Enter code: EOIF-YKZXL
Device codes are a common phishing target. Never share this code.
```

这证明整个 provider 是真实可用的，并非文档占位符。

| # | 声明 | 结论 | 验证依据 |
|---|------|------|----------|
| 1 | 功能在 v1.81.3 通过 PR #19030 合入 | ✅ 确认 | v1.81.3 release notes 明确列出 "Adds support for calling chatgpt subscription via LiteLLM - PR #19030" |
| 2 | Provider 前缀为 `chatgpt/` | ✅ 确认 | `litellm/llms/chatgpt/` 目录在 v1.82.0 中存在；`get_llm_provider("chatgpt/gpt-5.2")` 正确路由到 chatgpt provider |
| 3 | 使用 OAuth Device Code Flow | ✅ 确认 | `authenticator.py` 实现了完整 device code 流程；实测触发了真实 OAuth，打印了 `Visit https://auth.openai.com/codex/device` 和用户码 |
| 4 | Token 缓存到 auth.json | ✅ 确认 | Authenticator 读写 `~/.config/litellm/chatgpt/auth.json`，包含 JWT exp 解析和 refresh_token 复用逻辑 |
| 5 | API base 是 `chatgpt.com/backend-api/codex` | ✅ 确认 | `common_utils.py` 中 `CHATGPT_API_BASE = "https://chatgpt.com/backend-api/codex"`；`get_complete_url` 返回 `f"{api_base}/responses"` |
| 6 | `max_tokens` 被剔除 | ✅ 确认 | `transform_responses_api_request` 使用白名单 `allowed_keys`；`max_tokens`、`max_output_tokens`、`max_completion_tokens`、`metadata` 均不在白名单中 |
| 7 | 4 个模型已注册（gpt-5.2, gpt-5.2-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini） | ✅ 确认 | `litellm.model_cost` 验证：4 个模型均有 `litellm_provider="chatgpt"`、`mode="responses"`、`input_cost_per_token=None` |
| 8 | `stream=True` 在内部被强制设置 | ✅ 确认 | `responses/transformation.py` 无条件设置 `request["stream"] = True`；`/chat/completions` 在响应组装时遵守调用方的 stream 参数 |
| 9 | `store=False` 被强制设置 | ✅ 确认 | `responses/transformation.py` 无条件设置 `request["store"] = False` |
| 10 | 使用官方 Codex CLI 的 client_id | ✅ 确认 | `CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"`；与 openai/codex 官方 CLI 使用的 client_id 相同 |
| 11 | 默认系统指令是 Codex CLI 的提示 | ✅ 确认 | `common_utils.py` 中的 `CHATGPT_DEFAULT_INSTRUCTIONS` 是完整的 Codex CLI 系统提示（约 1500 字符，涵盖编码任务、文件编辑、git 行为） |
| 12 | 支持 gpt-5.1-codex 系列 | ✅ 确认 | issue #18753 中列出了 gpt-5.1-codex-max 和 gpt-5.1-codex-mini；两者均在 v1.82.0 的 model_cost map 中 |

---

## 9. 总结

### 适合使用的场景

- 已有 ChatGPT Plus/Pro/Max 订阅，想在代码中使用 Codex/GPT-5.x 模型而不额外支付 API 费用
- 个人开发工具或自动化脚本
- 需要运行与 Codex CLI 相同模型的私人 pipeline

### 不适合使用的场景

- 构建生产服务或商业产品（ToS 风险明确）
- 需要 gpt-4o、o3、o4-mini 或 Codex/GPT-5.x 系列以外的模型
- 需要 embeddings、图像生成或音频功能
- 需要控制输出长度（不支持 max_tokens）
- 需要完全无人值守部署（首次 OAuth 需要浏览器交互）

### 一句话结论

LiteLLM 的 `chatgpt/` provider 是经源码验证的合法实现，使用了 OpenAI 官方 OAuth 协议，功能与文档描述完全一致。核心权衡是：模型仅限 Codex/GPT-5.x 系列、无 token 长度控制、首次登录需交互、非个人用途存在 ToS 风险。

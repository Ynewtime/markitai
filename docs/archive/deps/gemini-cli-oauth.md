# 使用 Google 账户订阅调用 Gemini 的完整技术报告

> 调研范围：Google OAuth 认证路线 · Code Assist 内部 API · Python 集成方案
> 参考实现：pi agent (badlogic/pi-mono) · Cline · LiteLLM
> 更新时间：2026-03-05

---

## 目录

1. [核心架构：两条截然不同的路线](#1-核心架构两条截然不同的路线)
2. [完整认证方式详解](#2-完整认证方式详解)
3. [API 规格：Code Assist 内部端点](#3-api-规格code-assist-内部端点)
4. [pi agent 的实现剖析](#4-pi-agent-的实现剖析)
5. [Python 集成方案](#5-python-集成方案)
6. [事实核查](#6-事实核查)
7. [综合建议](#7-综合建议)

---

## 1. 核心架构：两条截然不同的路线

Google 账户登录的 Gemini CLI **不走公共 Gemini API**，而是通过 Google 内部的 **Cloud Code Assist** 服务代理请求。这是理解整个技术体系的关键前提。

### 1.1 路线对比

| 维度 | 公共 API 路线 | OAuth / Code Assist 路线 |
|---|---|---|
| 认证方式 | `GEMINI_API_KEY` 环境变量 | Google OAuth 2.0 |
| 后端端点 | `generativelanguage.googleapis.com` | `cloudcode-pa.googleapis.com/v1internal` |
| 请求格式 | 标准 `GenerateContentRequest` | 自定义封装格式 `CAGenerateContentRequest` |
| 官方 SDK 可用 | ✅ `google-genai` 直接支持 | ❌ 需手动实现，SDK 不支持 |
| 计费方式 | 按 token 计费 | 包含在 Google 订阅配额内 |
| 免费配额 | 有限（取决于模型） | 每分钟 60 次 / 每天 1000 次（个人账户） |

### 1.2 两个 Provider 的区分（来自 pi agent）

pi agent 明确区分了两个不同的 Google 订阅 provider：

| Provider 名称 | pi 内部标识符 | 后端说明 | 可用模型 |
|---|---|---|---|
| Gemini CLI | `google-gemini-cli` | Code Assist 标准通道 | 标准 Gemini 系列 |
| Antigravity | `google-antigravity` | Google 内部沙盒 | Gemini 3、Claude、GPT-OSS |

两者均通过相同的 `cloudcode-pa.googleapis.com` 域名，但路由和配额池不同。

---

## 2. 完整认证方式详解

Gemini CLI 支持五种认证方式，按使用场景从简到复：

### 方式 A：Login with Google（OAuth 个人账户）— **本文重点**

**适用人群**：个人 Google 账户、Google AI Pro/Ultra 订阅用户
**认证流程**：

```
1. 运行 gemini CLI，选择 "Login with Google"
2. CLI 启动本地 HTTP 服务器监听 localhost:45289
3. 浏览器打开 Google 授权页面
4. 用户授权后，浏览器重定向到 localhost:45289/oauth2callback
5. CLI 接收 authorization code，换取 access_token + refresh_token
6. 凭据缓存至本地文件
```

**OAuth App 信息**（来自 Gemini CLI 源码，公开硬编码）：

```
client_id:     <GEMINI_CLI_CLIENT_ID — see Gemini CLI source>
client_secret: <GEMINI_CLI_CLIENT_SECRET — see Gemini CLI source>
redirect_uri:  http://localhost:45289
```

**OAuth Scopes**：

```
https://www.googleapis.com/auth/cloud-platform
https://www.googleapis.com/auth/userinfo.email
https://www.googleapis.com/auth/userinfo.profile
```

**凭据存储路径**：

| 平台 | 路径 |
|---|---|
| macOS / Linux | `~/.gemini/oauth_creds.json` |
| Windows | `C:\Users\USERNAME\.gemini\oauth_creds.json` |

**凭据文件格式**：

```json
{
    "access_token": "ya29.xxxxxx",
    "refresh_token": "1//xxxxxx",
    "scope": "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile",
    "token_type": "Bearer",
    "id_token": "eyJxxxxxx",
    "expiry_date": 1753710424847
}
```

> **注意**：`expiry_date` 为毫秒级 Unix 时间戳（与标准 Python `time.time()` 的秒级不同）。

**另一个相关文件** `~/.gemini/google_accounts.json` 用于存储已登录的账户信息（多账户场景），格式不同，不含 token，仅记录账户元数据和活跃账户标识。

---

### 方式 B：Gemini API Key

**适用人群**：不想用 Google 账户、或需要在无浏览器环境中使用
**配置方式**：

```bash
export GEMINI_API_KEY="your-api-key-from-aistudio"
```

此路线调用的是公共 `generativelanguage.googleapis.com` 端点，走标准 API 计费，与本文关注的 OAuth 路线完全不同。

---

### 方式 C：Vertex AI + ADC（Application Default Credentials）

**适用人群**：拥有 GCP 项目的开发者、CI/CD 环境
**配置方式**：

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI=true
```

此路线后端为 `aiplatform.googleapis.com`，走 Vertex AI 计费，与 Code Assist 路线也不同。

---

### 方式 D：Vertex AI + 服务账户 JSON Key

**适用人群**：CI/CD 流水线、组织限制 ADC 的场景
**配置方式**：

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

---

### 方式 E：Vertex AI + Google Cloud API Key

**适用人群**：不想配置服务账户的 Vertex AI 用户
**配置方式**：

```bash
export GOOGLE_API_KEY="your-cloud-api-key"
export GOOGLE_GENAI_USE_VERTEXAI=true
```

---

### 认证方式汇总

| 方式 | 后端 | Python SDK 支持 | 无浏览器可用 | 免费配额 |
|---|---|---|---|---|
| A. Google OAuth | `cloudcode-pa.googleapis.com` | ❌（需手动实现） | 需预登录后复用凭据 | ✅ |
| B. Gemini API Key | `generativelanguage.googleapis.com` | ✅ `google-genai` | ✅ | 有限 |
| C. Vertex AI + ADC | `aiplatform.googleapis.com` | ✅ `google-genai` | ✅（gcloud） | ❌ 按量计费 |
| D. Vertex AI + SA Key | `aiplatform.googleapis.com` | ✅ `google-genai` | ✅ | ❌ 按量计费 |
| E. Vertex AI + API Key | `aiplatform.googleapis.com` | ✅ `google-genai` | ✅ | ❌ 按量计费 |

---

## 3. API 规格：Code Assist 内部端点

### 3.1 基本信息

| 属性 | 值 |
|---|---|
| Base URL | `https://cloudcode-pa.googleapis.com` |
| API 版本 | `v1internal` |
| 协议 | HTTPS + SSE（流式）/ 标准 JSON（非流式） |
| 认证 | `Authorization: Bearer <access_token>` |

### 3.2 可用方法

| 方法 | 路径 | 说明 |
|---|---|---|
| 非流式生成 | `POST /v1internal:generateContent` | 返回完整响应 |
| 流式生成 | `POST /v1internal:streamGenerateContent?alt=sse` | SSE 流式输出 |
| Token 计数 | `POST /v1internal:countTokens` | 预估 token 用量 |
| 加载 Code Assist | `POST /v1internal:loadCodeAssist` | 初始化/检查订阅状态 |

### 3.3 请求体格式（CAGenerateContentRequest）

与标准 Gemini API 的关键区别：请求被包裹在一个自定义信封中。

```json
{
    "model": "gemini-2.5-pro",
    "project": "your-gcp-project-id",
    "user_prompt_id": "optional-uuid",
    "request": {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "你好，请介绍一下自己"}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": "You are a helpful assistant."}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
            "topP": 0.9
        },
        "tools": [],
        "safetySettings": []
    }
}
```

> **重要**：`project` 字段为 GCP 项目 ID（非项目编号）。可从 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 查询，也可通过 `loadCodeAssist` 接口自动发现。

### 3.4 流式响应格式（SSE）

流式端点返回 `text/event-stream`，每个事件格式为标准 SSE：

```
data: {"candidates": [{"content": {"parts": [{"text": "你"}], "role": "model"}, "finishReason": null}], ...}

data: {"candidates": [{"content": {"parts": [{"text": "好"}], "role": "model"}, "finishReason": null}], ...}

data: {"candidates": [{"content": {"parts": [{"text": "！"}], "role": "model"}, "finishReason": "STOP"}], "usageMetadata": {...}}
```

### 3.5 已知限制

- **思维链（Thinking）字段被截断**：`converter.ts` 源码显示，`thought` 字段内容被合并进 `text`，结构信息丢失。这是 OAuth 路线当前的一个已知缺陷。
- **不支持部分高级特性**：`automaticFunctionCalling`、`imageConfig`、`modelArmorConfig` 等仅限公共 API 的字段在此路线下不可用。
- **Project ID 强制要求**：Google Workspace 账户、Code Assist 订阅用户必须显式传入 `project` 字段，个人 Google 账户可能会自动分配临时 project。

### 3.6 配额信息

| 账户类型 | RPM | RPD | 上下文窗口 |
|---|---|---|---|
| 个人 Google 账户（免费） | 60 | 1,000 | 1M tokens |
| Google AI Pro | 高于免费 | 高于免费 | 1M tokens |
| Google AI Ultra | 更高 | 更高 | 1M tokens |
| Code Assist Standard | 不适用 | ~1,500 | 1M tokens |

---

## 4. pi agent 的实现剖析

pi agent（`badlogic/pi-mono`）的 `@mariozechner/pi-ai` 包是目前最完整的 Google OAuth + Code Assist 路线开源实现，虽然是 TypeScript，但架构思路可直接用于 Python 移植。

### 4.1 核心函数

```typescript
import {
    loginGeminiCli,       // 发起 OAuth 登录流程，返回凭据
    loginAntigravity,     // 发起 Antigravity OAuth 流程
    refreshOAuthToken,    // (provider, credentials) => 刷新后的凭据
    getOAuthApiKey,       // (provider, credentialsMap) => { newCredentials, apiKey }
    type OAuthProvider,   // 'google-gemini-cli' | 'google-antigravity' | ...
    type OAuthCredentials,
} from '@mariozechner/pi-ai';
```

**关键设计**：`getOAuthApiKey` 返回的 `apiKey` 是一个 **JSON 字符串**，同时包含 `access_token` 和 `project`，由内部自动序列化。这允许用统一的 `apiKey: string` 接口传递 OAuth 凭据，无需修改通用流处理代码。

### 4.2 凭据存储位置

```
~/.pi/agent/auth.json
```

结构示例：

```json
{
    "google-gemini-cli": {
        "type": "oauth",
        "access_token": "ya29.xxx",
        "refresh_token": "1//xxx",
        "expiry_date": 1234567890000,
        "scope": "...",
        "id_token": "..."
    },
    "google-antigravity": {
        "type": "oauth",
        ...
    }
}
```

### 4.3 与 Gemini CLI 凭据文件的关系

pi agent 和 Gemini CLI **使用不同的凭据文件**：

- Gemini CLI 的 OAuth 凭据 → `~/.gemini/oauth_creds.json`
- pi agent 的 OAuth 凭据 → `~/.pi/agent/auth.json`

两者格式相似，但 pi agent 通过 `/login` 命令触发自己的 OAuth 流程（使用相同的 client_id 和 secret），独立存储。

### 4.4 双配额池策略（Antigravity 插件）

opencode-antigravity-auth 项目揭示了一个实用技巧：可同时持有 `google-gemini-cli` 和 `google-antigravity` 两个 OAuth token，对 Gemini 模型请求在两个配额池之间自动轮转，叠加可用配额。

---

## 5. Python 集成方案

### 方案一：复用已有 Gemini CLI 凭据（推荐·最简）

**前提**：用户已在本地运行过 `gemini` CLI 并完成登录。
**依赖**：`pip install google-auth httpx`

```python
import json
import time
from pathlib import Path
import httpx
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

OAUTH_CLIENT_ID = "<GEMINI_CLI_CLIENT_ID — see Gemini CLI source>"
OAUTH_CLIENT_SECRET = "<GEMINI_CLI_CLIENT_SECRET — see Gemini CLI source>"

CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
CODE_ASSIST_API_VERSION = "v1internal"


def load_gemini_credentials() -> Credentials:
    """加载并自动刷新 Gemini CLI 的 OAuth 凭据。"""
    creds_path = Path.home() / ".gemini" / "oauth_creds.json"
    if not creds_path.exists():
        raise FileNotFoundError(
            "未找到 ~/.gemini/oauth_creds.json，请先运行 gemini CLI 并完成 Google 登录"
        )

    raw = json.loads(creds_path.read_text())

    # expiry_date 是毫秒级时间戳，需转换为秒
    from datetime import datetime, timezone
    expiry = datetime.fromtimestamp(raw["expiry_date"] / 1000, tz=timezone.utc)

    creds = Credentials(
        token=raw["access_token"],
        refresh_token=raw["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        scopes=raw["scope"].split(),
        expiry=expiry,
    )

    # token 过期时自动刷新
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

        # 将刷新后的 token 写回文件
        raw["access_token"] = creds.token
        raw["expiry_date"] = int(creds.expiry.timestamp() * 1000)
        creds_path.write_text(json.dumps(raw, indent=2))

    return creds


def generate_content(
    prompt: str,
    model: str = "gemini-2.5-pro",
    project_id: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """调用 Code Assist 端点生成内容。"""
    creds = load_gemini_credentials()

    url = f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:generateContent"

    # 构造内部封装格式（CAGenerateContentRequest）
    inner_request: dict = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_prompt:
        inner_request["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    payload: dict = {
        "model": model,
        "request": inner_request,
    }

    # project 字段仅在 Workspace / Code Assist 订阅时需要
    if project_id:
        payload["project"] = project_id

    response = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {creds.token}"},
        timeout=120.0,
    )
    response.raise_for_status()

    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def stream_generate_content(
    prompt: str,
    model: str = "gemini-2.5-pro",
    project_id: str | None = None,
) -> None:
    """流式调用，打印 SSE 输出。"""
    creds = load_gemini_credentials()

    url = (
        f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}"
        ":streamGenerateContent?alt=sse"
    )

    payload = {
        "model": model,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        },
    }
    if project_id:
        payload["project"] = project_id

    with httpx.stream(
        "POST",
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Accept": "text/event-stream",
        },
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                try:
                    text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                    print(text, end="", flush=True)
                except (KeyError, IndexError):
                    pass
    print()  # 换行


# --- 使用示例 ---
if __name__ == "__main__":
    # 非流式
    result = generate_content("用一句话介绍量子计算")
    print(result)

    # 流式
    stream_generate_content("写一首关于秋天的短诗")
```

---

### 方案二：程序化发起 OAuth 登录（无需预装 gemini CLI）

**依赖**：`pip install google-auth google-auth-oauthlib httpx`

```python
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

OAUTH_CLIENT_ID = "<GEMINI_CLI_CLIENT_ID — see Gemini CLI source>"
OAUTH_CLIENT_SECRET = "<GEMINI_CLI_CLIENT_SECRET — see Gemini CLI source>"

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

CLIENT_CONFIG = {
    "installed": {
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "redirect_uris": ["http://localhost:45289"],
        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"


def login_google() -> dict:
    """
    发起 Google OAuth 登录流程。
    打开浏览器完成授权后，凭据自动保存至 ~/.gemini/oauth_creds.json
    """
    flow = InstalledAppFlow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        # 使用与 Gemini CLI 相同的端口
    )

    # 启动本地服务器监听 OAuth 回调
    creds = flow.run_local_server(
        port=45289,
        prompt="consent",
        access_type="offline",  # 确保获取 refresh_token
    )

    token_data = {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "scope": " ".join(creds.scopes) if creds.scopes else " ".join(SCOPES),
        "token_type": "Bearer",
        "id_token": creds.id_token,
        "expiry_date": int(creds.expiry.timestamp() * 1000),  # 毫秒
    }

    # 保存至与 Gemini CLI 兼容的路径
    CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDS_PATH.write_text(json.dumps(token_data, indent=2))
    print(f"凭据已保存至 {CREDS_PATH}")

    return token_data


def auto_discover_project_id(access_token: str) -> str | None:
    """通过 loadCodeAssist 接口自动发现 GCP Project ID。"""
    import httpx

    resp = httpx.post(
        "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
        headers={"Authorization": f"Bearer {access_token}"},
        json={},
        timeout=30.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("cloudProject") or data.get("projectId")
    return None


if __name__ == "__main__":
    token_data = login_google()
    project_id = auto_discover_project_id(token_data["access_token"])
    print(f"自动发现 Project ID: {project_id}")
```

---

### 方案三：通过 LiteLLM 代理（兼容 OpenAI API）

适合希望用标准 OpenAI 格式、或需要多账户负载均衡的场景。

**Step 1：准备配置文件 `config.yaml`**

```yaml
model_list:
  - model_name: gemini-cli-pro
    litellm_params:
      model: gemini-2.5-pro
      api_base: https://cloudcode-pa.googleapis.com/v1internal
      vertex_project: your-gcp-project-id  # 从 aistudio.google.com/apikey 获取
      vertex_credentials: |
        {
            "access_token": "ya29.xxx",
            "refresh_token": "1//xxx",
            "scope": "https://www.googleapis.com/auth/cloud-platform ...",
            "token_type": "Bearer",
            "expiry_date": 1753710424847
        }

  # 可添加多个账户实现负载均衡（使用相同 model_name）
  - model_name: gemini-cli-pro
    litellm_params:
      model: gemini-2.5-pro
      api_base: https://cloudcode-pa.googleapis.com/v1internal
      vertex_project: your-second-project-id
      vertex_credentials: |
        { ... second account credentials ... }

general_settings:
  master_key: sk-your-proxy-key
```

**Step 2：启动代理**

```bash
docker run \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml
```

**Step 3：Python 调用**

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-proxy-key",
    base_url="http://localhost:4000",
)

response = client.chat.completions.create(
    model="gemini-cli-pro",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

---

### 方案对比

| 维度 | 方案一（复用凭据） | 方案二（程序化登录） | 方案三（LiteLLM） |
|---|---|---|---|
| 需要预装 gemini CLI | ✅ 是 | ❌ 否 | ❌ 否 |
| 需要浏览器 | ❌ 否（已登录） | ✅ 一次性 | ✅ 一次性 |
| OpenAI 格式兼容 | ❌ | ❌ | ✅ |
| 多账户轮转 | 需手动实现 | 需手动实现 | ✅ 内置 |
| 额外依赖 | `google-auth httpx` | `google-auth-oauthlib httpx` | Docker + LiteLLM |
| 维护复杂度 | 低 | 低 | 中（需运行代理） |

---

## 6. 事实核查

以下逐条核查本报告中的关键技术声明：

### ✅ 已验证

| 声明 | 来源 | 状态 |
|---|---|---|
| 内部端点为 `cloudcode-pa.googleapis.com/v1internal` | gemini-cli 源码 `server.ts:62-63`；多个 GitHub issue 中的错误信息均包含此 URL | ✅ 确认 |
| OAuth Client ID 为 `681255809395-...` | Roo-Code issue #5134、gemini-cli auth URL 在多个 issue 中可见 | ✅ 确认 |
| 凭据文件路径为 `~/.gemini/oauth_creds.json` | LiteLLM 官方文档、Cline 实现、多个第三方工具均使用此路径 | ✅ 确认 |
| 请求格式为 `{model, project, request: {...}}` 而非标准格式 | gemini-cli 源码 `converter.ts:125`；GitHub issue #19200 详细分析 | ✅ 确认 |
| 个人账户免费配额为 60 RPM / 1000 RPD | Gemini CLI 官方文档 | ✅ 确认 |
| `expiry_date` 为毫秒时间戳 | LiteLLM 文档中凭据格式示例 | ✅ 确认 |
| pi agent 区分 `google-gemini-cli` 和 `google-antigravity` 两个 provider | pi-mono `packages/ai/README.md` 和 `packages/coding-agent/docs/providers.md` | ✅ 确认 |
| Thinking/思维链字段在 OAuth 路线下被截断 | gemini-cli issue #19200，`converter.ts` 源码分析 | ✅ 确认 |

### ⚠️ 需要注意的事项

| 声明 | 状态 | 说明 |
|---|---|---|
| OAuth Client Secret 可安全使用 | ⚠️ 注意 | 这是 Google 自己的 OAuth App 凭据，属于已公开的硬编码值，用于代表 Gemini CLI 应用申请用户授权，并非用户私密凭据。但滥用可能违反 Google ToS。 |
| Antigravity 可免费访问 Claude 模型 | ⚠️ 不稳定 | 有用户报告账户被封禁；Antigravity ToS 明确禁止第三方使用。Gemini CLI（Code Assist）路线相对宽松，但也存在 ToS 风险。 |
| `loadCodeAssist` 可自动发现 Project ID | ⚠️ 部分情况适用 | 个人 Google 账户可能自动获得临时 project；Workspace 账户或订阅用户必须显式提供 `GOOGLE_CLOUD_PROJECT`，否则可能沿用免费配额。 |
| Python 子进程调用 gemini CLI 可复用 OAuth 凭据 | ⚠️ 有 Bug | GitHub issue #12042 显示，在 `--experimental-acp` 模式下从 Python 子进程启动 CLI 时，即使本地有缓存凭据，CLI 仍可能重新要求登录。 |

### ❌ 已证伪

| 声明 | 状态 | 说明 |
|---|---|---|
| 可以用 `google-genai` SDK 直接对接 OAuth 路线 | ❌ 不正确 | `google-genai` 指向公共 API 端点，无法路由到 `cloudcode-pa.googleapis.com`，请求格式也不兼容。 |
| OAuth 路线和 API Key 路线的后端是同一个服务 | ❌ 不正确 | 两者后端完全不同，配额池、定价、延迟、功能支持均有差异。 |

---

## 7. 综合建议

### 场景决策树

```
想用 Google 账户订阅（免费配额）？
│
├─ 已安装 gemini CLI 且已登录
│   └─ → 方案一：直接读取 ~/.gemini/oauth_creds.json
│
├─ 没有 gemini CLI，但愿意一次性浏览器授权
│   └─ → 方案二：程序化 OAuth 登录
│
└─ 需要 OpenAI 兼容接口 / 多账户负载均衡
    └─ → 方案三：LiteLLM 代理
```

### 生产环境注意事项

1. **Token 刷新**：`access_token` 约 1 小时过期，必须实现自动刷新逻辑（使用 `refresh_token` 调用 `oauth2.googleapis.com/token`）。

2. **Project ID 的获取**：优先通过 `loadCodeAssist` 接口自动发现；若为 Code Assist 订阅用户，必须手动配置 `GOOGLE_CLOUD_PROJECT`，否则只能用免费配额。

3. **ToS 合规性**：Code Assist 个人免费用途相对宽松；Antigravity 和商业用途建议咨询 Google 相关条款。

4. **错误处理**：`v1internal` 端点返回标准 Google API 错误码，429 表示配额耗尽，需实现退避重试。

### 最小依赖方案

```bash
# 仅需两个包
pip install google-auth httpx
```

配合「方案一」即可完成完整的认证 + 调用，无需其他依赖。

---

*本报告基于 2026-03-05 调研的公开信息，所有技术规格均来自 Google 官方源码、文档及社区 issue，不含私有或未公开信息。*

# MODEL env var 零配置 + 订阅制 Provider 设计

> 2026-03-05 | 状态：待实施

---

## 1. 总览

四个模块：

1. **MODEL env var 零配置** — `MODEL=openai/gpt-4o markitai doc.pdf --llm`
2. **gemini-cli/ provider** — Google 订阅通过 Code Assist 内部端点
3. **chatgpt/ provider wrapper** — ChatGPT 订阅通过 Responses API
4. **全局 local provider 错误 UX** — 统一 auth 异常捕获 + resolution hint

---

## 2. MODEL env var 零配置

### 用户体验

```bash
# 标准 API（LiteLLM 自动读对应 env var 的 key）
OPENAI_API_KEY=sk-xxx MODEL=openai/gpt-4o markitai doc.pdf --llm
GEMINI_API_KEY=xxx MODEL=gemini/gemini-2.5-flash markitai doc.pdf --llm

# 订阅制（无需 API key）
MODEL=chatgpt/gpt-5.2 markitai doc.pdf --llm
MODEL=gemini-cli/gemini-2.5-pro markitai doc.pdf --llm
MODEL=claude-agent/sonnet markitai doc.pdf --llm
```

### 触发条件

- `--llm` 已启用
- `MODEL` 环境变量已设置
- config 文件中**没有** `model_list`（或为空）

如果 `markitai.json` 已有 model_list，`MODEL` env var 被忽略。

### 实现

位置：`cli/main.py` 的 config merge 阶段。

```python
if config.llm.enabled and not config.llm.model_list:
    model_env = os.environ.get("MODEL")
    if model_env:
        config.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model=model_env),
            )
        ]
```

### 错误处理

不做显式 key 校验，依赖 LiteLLM 报错。在 processor 层捕获 `AuthenticationError` 并展示友好提示，包含缺失的 env var 名称。

---

## 3. gemini-cli/ Provider

### 架构

新建 `providers/gemini_cli.py`，实现 `GeminiCLIProvider(CustomLLM)`。

### 认证（双模式）

1. **优先读凭据**：`~/.gemini/oauth_creds.json`（Gemini CLI 登录后生成）
2. **Fallback OAuth**：如果凭据文件不存在，使用 `google-auth-oauthlib` 的 `InstalledAppFlow` 触发浏览器 OAuth 登录（与 Gemini CLI 使用相同的 client_id `681255809395-...` 和端口 45289）
3. **Token 自动刷新**：`google-auth` 的 `Credentials.refresh(Request())` 处理过期刷新

### OAuth App 信息（Gemini CLI 公开硬编码）

```
client_id:     <GEMINI_CLI_CLIENT_ID — see Gemini CLI source>
client_secret: <GEMINI_CLI_CLIENT_SECRET — see Gemini CLI source>
redirect_uri:  http://localhost:45289
scopes:        cloud-platform, userinfo.email, userinfo.profile
```

### API 调用

- 端点：`https://cloudcode-pa.googleapis.com/v1internal:generateContent`（非流式）
- 端点：`https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse`（流式）
- 请求格式：`CAGenerateContentRequest`（model, project, request 三层封装）
- 认证头：`Authorization: Bearer <access_token>`

### 请求格式

```json
{
    "model": "gemini-2.5-pro",
    "project": "<auto-discovered-or-optional>",
    "request": {
        "contents": [{"role": "user", "parts": [{"text": "..."}]}],
        "systemInstruction": {"parts": [{"text": "..."}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}
    }
}
```

### Project ID

- 通过 `loadCodeAssist` 接口自动发现
- 个人 Google 账户通常自动分配临时 project，可不传
- 缓存发现结果

### 模型前缀

`gemini-cli/<model-name>`，例如 `gemini-cli/gemini-2.5-pro`、`gemini-cli/gemini-2.5-flash`。

### Vision 支持

Code Assist 端点支持 `inlineData` parts（base64 图片），与标准 Gemini API 格式一致。所有 Gemini 模型均支持 vision。

### 配额

| 账户类型 | RPM | RPD |
|----------|-----|-----|
| 个人 Google 账户（免费） | 60 | 1,000 |
| Google AI Pro | 更高 | 更高 |
| Google AI Ultra | 更高 | 更高 |

### 依赖

- `google-auth`：token 刷新（已在项目依赖链中，通过 google-cloud 生态）
- `google-auth-oauthlib`：OAuth 登录流程（新增可选依赖）
- `httpx`：HTTP 请求（已有）

作为可选依赖组：`uv add markitai[gemini-cli]`

---

## 4. chatgpt/ Provider Wrapper

### 问题

LiteLLM v1.82.0 的 `chatgpt/` provider 有 `mode=responses`，但通过 `litellm.acompletion()` 调用时走 Chat Completions handler（`ChatGPTConfig(OpenAIConfig)`），拼出 `/chat/completions` 端点 → 403。正确端点是 `/responses`。

### 方案

新建 `providers/chatgpt.py`，实现 `ChatGPTProvider(CustomLLM)`：

1. **认证**：直接复用 LiteLLM 的 `Authenticator` 类（`litellm.llms.chatgpt.authenticator`）
   - 读 `~/.config/litellm/chatgpt/auth.json`
   - Token 过期自动 refresh
   - 无凭据时自动触发 Device Code Flow（终端打印验证码 + URL）
2. **请求**：用 httpx 直接调 `https://chatgpt.com/backend-api/codex/responses`
   - 将 messages 转为 Responses API 的 input 格式
   - 设置必要 headers（originator, ChatGPT-Account-Id, session_id）
   - 强制 `store=false`, `stream=true`
   - 白名单参数过滤（与 LiteLLM 的 `ChatGPTResponsesAPIConfig` 一致）
3. **响应**：解析 SSE stream，提取 `response.completed` 事件，转为 `ModelResponse` 格式

### 注册

在 `custom_provider_map` 中注册 `chatgpt` 前缀，截获请求使其不走 LiteLLM 的 broken Chat Completions 路径。

### 模型前缀

`chatgpt/<model-name>`，例如 `chatgpt/gpt-5.2`、`chatgpt/gpt-5.2-codex`。

### 限制（继承自 ChatGPT 后端）

- 不支持 `max_tokens`、`temperature`、`top_p`（后端拒绝）
- 仅限 Codex/GPT-5.x 系列模型
- 首次使用需浏览器交互完成 OAuth

### 依赖

无新增依赖。`litellm` 和 `httpx` 已有。LiteLLM 的 Authenticator 作为内部实现依赖使用。

---

## 5. 全局 Local Provider 错误 UX

### 当前问题

| Provider | 无凭据时的行为 |
|----------|---------------|
| claude-agent/ | SDK 异常 → RuntimeError，无引导 |
| copilot/ | CLI 子进程失败 → FileNotFoundError，无引导 |
| chatgpt/ | 自动 Device Code Flow（已友好） |
| gemini-cli/ | 自动 OAuth 登录（设计中） |

### 改善方案

在每个 provider 的 `acompletion()` 方法中，捕获 auth 相关异常，包装为带 resolution hint 的错误消息：

```python
# claude-agent
except Exception as e:
    if is_auth_error(e):
        hint = get_auth_resolution_hint("claude-agent")
        raise RuntimeError(f"Claude Agent authentication failed: {e}\n\n{hint}") from e
    raise

# copilot
except (FileNotFoundError, OSError) as e:
    hint = get_auth_resolution_hint("copilot")
    raise RuntimeError(f"Copilot authentication failed: {e}\n\n{hint}") from e
```

Resolution hint 已在 `auth.py` 中实现（包含 env var 替代方案）。

### 全局 LLM 错误处理（processor 层）

在 `LLMProcessor` 的 `_call_with_retry()` 中，对 `AuthenticationError` 做特殊处理：

```
[LLM] Authentication failed for model 'openai/gpt-4o':
  OpenAI API key not found. Set OPENAI_API_KEY environment variable.

  Hint: Use MODEL=<model> with the corresponding API key env var,
  or run 'markitai init' to configure interactively.
```

---

## 6. Auth 预检和 Interactive 模式

### AuthManager 新增

| Provider | 检查方法 | 认证标志 |
|----------|---------|---------|
| gemini-cli | `~/.gemini/oauth_creds.json` 存在且有 access_token | authenticated |
| chatgpt | `~/.config/litellm/chatgpt/auth.json` 存在且有 access_token | authenticated |

### interactive.py 探测顺序更新

```python
DETECTION_ORDER = [
    # 订阅制（免费/包含在订阅中）
    ("claude-agent", "claude-agent/sonnet"),
    ("copilot", "copilot/gpt-5.2"),
    ("chatgpt", "chatgpt/gpt-5.2"),        # 新增
    ("gemini-cli", "gemini-cli/gemini-2.5-pro"),  # 新增
    # 标准 API
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-20250514"),
    ("OPENAI_API_KEY", "openai/gpt-4o"),
    ("GEMINI_API_KEY", "gemini/gemini-2.5-flash"),
    ("DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ("OPENROUTER_API_KEY", "openrouter/..."),
]
```

### markitai doctor

新增 gemini-cli 和 chatgpt 的认证状态检查。

---

## 7. 新增依赖

| 依赖 | 用途 | 类型 |
|------|------|------|
| `google-auth` | Gemini token 刷新 | 可选（gemini-cli extra） |
| `google-auth-oauthlib` | Gemini OAuth 登录 | 可选（gemini-cli extra） |

在 `pyproject.toml` 中新增 extra：

```toml
[project.optional-dependencies]
gemini-cli = ["google-auth>=2.0", "google-auth-oauthlib>=1.0"]
```

---

## 8. 文件清单

### 新建

| 文件 | 说明 |
|------|------|
| `providers/gemini_cli.py` | Gemini CLI provider（~300 行） |
| `providers/chatgpt.py` | ChatGPT subscription provider（~250 行） |
| `tests/unit/test_provider_gemini_cli.py` | Gemini CLI 单元测试 |
| `tests/unit/test_provider_chatgpt.py` | ChatGPT 单元测试 |
| `tests/integration/test_subscription_providers.py` | 集成测试 |

### 修改

| 文件 | 变更 |
|------|------|
| `cli/main.py` | MODEL env var 检测（~15 行） |
| `providers/__init__.py` | 注册 gemini-cli + chatgpt（~40 行） |
| `providers/auth.py` | gemini-cli + chatgpt auth 预检（~60 行） |
| `providers/claude_agent.py` | 改善 auth 错误提示（~10 行） |
| `providers/copilot.py` | 改善 auth 错误提示（~10 行） |
| `cli/interactive.py` | 新增探测（~20 行） |
| `llm/processor.py` | LLM auth 错误处理（~15 行） |
| `constants.py` | 新增常量（~10 行） |
| `pyproject.toml` | gemini-cli extra 依赖 |

---

## 9. 实施顺序

1. **Phase 1**：MODEL env var 零配置 + processor 错误处理
2. **Phase 2**：chatgpt/ provider wrapper
3. **Phase 3**：gemini-cli/ provider
4. **Phase 4**：auth 预检 + interactive 探测 + 错误 UX 统一
5. **Phase 5**：文档更新

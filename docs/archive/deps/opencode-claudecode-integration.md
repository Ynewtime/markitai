# OpenCode 集成 Claude Code 订阅：深度技术方案拆解

## 项目概述

**OpenCode** 是由 Anomaly 团队开发的开源 AI 编程代理，仓库位于 [github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)，目前拥有 **79,000+ Stars**。项目定位为 Claude Code 的开源替代方案，核心特点包括：100% 开源、Provider 无关（支持多种 LLM）、原生 LSP 支持、以及 TUI 优先的设计理念。

其最具争议性的功能是通过**逆向工程 Claude Code 的 OAuth 认证机制**，允许用户复用 Claude Pro/Max 订阅，而无需单独购买 API 额度。

---

## 核心架构：插件化认证系统

OpenCode 的 Claude 订阅集成**不在主仓库中**，而是通过独立的认证插件实现：

```
anomalyco/opencode                    # 主项目
anomalyco/opencode-anthropic-auth     # Anthropic OAuth 认证插件
anomalyco/opencode-copilot-auth       # GitHub Copilot 认证插件
```

插件通过 npm 分发，在 `packages/opencode/src/plugin/index.ts` 中定义为内置插件：

```typescript
const BUILTIN = [
  "opencode-copilot-auth@0.0.9",
  "opencode-anthropic-auth@0.0.8"
]
```

---

## 技术方案一：OAuth 认证流程复用

### 背景：Claude Code 的认证机制

**Claude Code** 是 Anthropic 官方的 CLI 工具。它使用 **OAuth 2.0 + PKCE** 流程让用户通过 Claude Pro/Max 订阅进行认证，无需 API Key。

OpenCode 通过逆向工程获取了 Claude Code 使用的 OAuth 参数，并在 `opencode-anthropic-auth` 插件中实现了相同的认证流程。

### 关键 OAuth 参数

```javascript
// 从 Claude Code 逆向获取的 Client ID
const CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

// OAuth 端点
const AUTH_URL = "https://claude.ai/oauth/authorize"        // 授权端点
const TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"  // Token 交换端点
const REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"

// OAuth Scope
const SCOPE = "org:create_api_key+user:profile+user:inference"
```

### 认证流程实现

`opencode-anthropic-auth/index.mjs` 中的 Token 交换函数：

```javascript
async function exchange(code, verifier) {
  // Anthropic 授权码格式: "code#state"
  const splits = code.split("#")

  const result = await fetch("https://console.anthropic.com/v1/oauth/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      code: splits[0],
      state: splits[1],
      grant_type: "authorization_code",
      client_id: CLIENT_ID,
      redirect_uri: "https://console.anthropic.com/oauth/code/callback",
      code_verifier: verifier,  // PKCE 验证器
    }),
  });

  const json = await result.json();
  return {
    type: "success",
    refresh: json.refresh_token,
    access: json.access_token,
    expires: Date.now() + json.expires_in * 1000,
  };
}
```

### 凭据存储

认证成功后，Token 存储在本地：

| 平台 | 路径 |
|------|------|
| Linux | `~/.local/share/opencode/auth.json` |
| macOS | `~/Library/Application Support/opencode/auth.json` |

存储结构：
```json
{
  "anthropic": {
    "type": "oauth",
    "access": "sk-ant-oat01-...",
    "refresh": "...",
    "expires": 1234567890000
  }
}
```

---

## 技术方案二：API 请求伪装

Anthropic 在 2026 年 1 月 9 日实施技术封锁后，仅凭有效的 OAuth Token 已无法通过认证。OpenCode 必须**精确模拟 Claude Code CLI 的请求特征**才能使用订阅凭据。

### 必需的 HTTP Headers

```javascript
// 在 fetch 拦截器中设置
requestHeaders.set("authorization", `Bearer ${auth.access}`);
requestHeaders.set("user-agent", "claude-cli/2.1.2 (external, cli)");
requestHeaders.set("anthropic-beta", "oauth-2025-04-20,interleaved-thinking-2025-05-14");

// 必须移除 API Key header
requestHeaders.delete("x-api-key");
```

**关键发现**：`/v1/messages` 端点需要添加 `?beta=true` URL 参数才能被接受。

### 请求体修改

插件对请求体进行多项修改以匹配 Claude Code 的预期格式：

```javascript
// 解析请求体
const parsed = JSON.parse(body);

// 1. 删除 tool_choice 字段（Claude Code 不发送此参数）
delete parsed.tool_choice;

// 2. 注入 metadata.user_id（从 ~/.claude.json 读取）
parsed.metadata = {
  ...parsed.metadata,
  user_id: getUserIdFromClaudeConfig()
};
```

---

## 技术方案三：工具名称混淆

Anthropic 服务器会验证 OAuth Token 使用的工具名称是否在 Claude Code 的白名单中。OpenCode 开发了多种绕过方案：

### 当前有效方案：`mcp_` 前缀 (PR #14)

```javascript
const TOOL_PREFIX = "mcp_";

// 发送请求时：添加前缀
if (parsed.tools && Array.isArray(parsed.tools)) {
  parsed.tools = parsed.tools.map(tool => ({
    ...tool,
    name: `${TOOL_PREFIX}${tool.name}`  // read_file → mcp_read_file
  }));
}

// 处理响应时：移除前缀
const transformed = text.replace(/mcp_([a-z_]+)/g, '$1');
```

### 历史方案演进

| 时间 | 方案 | 状态 |
|------|------|------|
| 2026-01-09 | `oc_` 前缀 (PR #10) | 已被封锁 |
| 2026-01-09 | PascalCase 工具名 | 部分有效 |
| 2026-01-09 | `mcp_` 前缀 (PR #14) | 当前有效 |
| 社区方案 | TTL 缓存随机后缀 | 备用方案 |

### PR #15 的深度伪装尝试

PR #15 尝试更完整地模拟 Claude Code：

```javascript
// Claude Code 的原生工具名称
const CLAUDE_CODE_TOOL_NAMES = [
  "Read", "Write", "Edit", "MultiEdit", "Bash",
  "Glob", "Grep", "LS", "Task", "WebFetch", "WebSearch"
];

// 将工具名转换为 PascalCase 并匹配 Claude Code 格式
// read_file → Read
// execute_command → Bash
```

---

## 技术方案四：系统提示净化

Anthropic 服务器会检测系统提示中的特定字符串。插件实现了净化逻辑：

```javascript
// experimental.chat.system.transform hook
if (parsed.system && Array.isArray(parsed.system)) {
  parsed.system = parsed.system.map(item => {
    if (item.type === 'text' && item.text) {
      return {
        ...item,
        // 移除或替换 "OpenCode" 字符串
        text: item.text.replace(/OpenCode/gi, 'Claude Code')
      };
    }
    return item;
  });
}
```

主仓库中的 `PROMPT_ANTHROPIC_SPOOF` 会在系统提示开头注入 Claude Code 的身份标识：

```javascript
// packages/opencode/src/session/system.ts
"experimental.chat.system.transform": (input, output) => {
  const prefix = "You are Claude Code, Anthropic's official CLI for Claude.";
  if (input.model?.providerID === "anthropic") {
    output.system.unshift(prefix);
  }
}
```

---

## Token 生命周期管理

### 自动刷新机制

OAuth Access Token 有效期约 **1 小时**。插件在到期前自动刷新：

```javascript
// packages/opencode/src/auth/anthropic.ts
async function refreshToken(refreshToken) {
  const result = await fetch("https://console.anthropic.com/v1/oauth/token", {
    method: "POST",
    body: JSON.stringify({
      grant_type: "refresh_token",
      refresh_token: refreshToken,
      client_id: CLIENT_ID,
    }),
  });
  // 返回新的 access_token
}
```

### Keep-Alive 机制 (Issue #9111)

Anthropic 的 Refresh Token 在长时间不活动后会失效。社区提议实现 Keep-Alive：

```javascript
// packages/opencode/src/auth/keepalive.ts（提议）
// 每小时 ping /api/oauth/usage 端点
// 使用 Auth.OAuthPool.fetchAnthropicUsage
// 不消耗 token，仅保持会话活跃
```

---

## 主仓库中的 Provider 实现

### provider.ts 中的 OAuth 处理

`packages/opencode/src/provider/provider.ts` 包含凭据加载逻辑：

```typescript
// 第 823-832 行
// 加载存储的认证凭据
for (const [providerID, provider] of Object.entries(await Auth.all())) {
  if (disabled.has(providerID)) continue
  if (provider.type === "api") {
    mergeProvider(providerID, {
      source: "api",
      key: provider.key,
    })
  }
}
```

**注意**：存储的 OAuth 凭据会覆盖配置文件中的 API Key（Issue #10950）。

### 自定义 Fetch 函数

`provider.ts` 第 42-71 行实现了 OAuth 认证的自定义 fetch：

```typescript
// 当检测到 OAuth 认证时
if (auth.type === "oauth") {
  // 使用 Bearer Token 而非 API Key
  // 设置必要的 headers
  // 应用请求体转换
}
```

---

## Anthropic 的封锁时间线

| 日期 | 事件 |
|------|------|
| 2025-06-26 | Issue #417 首次报告 OAuth 限制 |
| 2026-01-05 | Issue #6930 用户因 OAuth 使用被封号 |
| 2026-01-09 02:20 UTC | Anthropic 官方宣布收紧检测 |
| 2026-01-09 | PR #10, #11, #14 快速修复 |
| 2026-01-28 | Issue #10937 报告 v0.0.8 插件失效 |

### Anthropic 的错误信息

封锁后，使用 OAuth Token 的非官方工具会收到：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "This credential is only authorized for use with Claude Code and cannot be used for other API requests."
  }
}
```

---

## 技术方案总结

OpenCode 的 Claude Code 订阅集成是一个**多层次的逆向工程方案**：

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenCode 主程序                          │
├─────────────────────────────────────────────────────────────┤
│  packages/opencode/src/provider/provider.ts                 │
│  - OAuth 凭据加载                                            │
│  - 自定义 fetch 函数                                         │
│  - PROMPT_ANTHROPIC_SPOOF 系统提示注入                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              opencode-anthropic-auth 插件                   │
├─────────────────────────────────────────────────────────────┤
│  index.mjs                                                  │
│  - OAuth 2.0 + PKCE 认证流程                                │
│  - Token 交换与刷新                                          │
│  - HTTP Headers 伪装 (User-Agent, anthropic-beta)           │
│  - 请求体修改 (删除 tool_choice, 注入 metadata)              │
│  - 工具名称混淆 (mcp_ 前缀)                                  │
│  - 系统提示净化 (移除 "OpenCode" 字符串)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Anthropic API 服务器                           │
├─────────────────────────────────────────────────────────────┤
│  验证层：                                                    │
│  - OAuth Token 有效性                                        │
│  - User-Agent 检测                                          │
│  - 工具名称白名单                                            │
│  - 系统提示内容检测                                          │
│  - 请求格式匹配                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 风险警示

根据 Issue #6930 的讨论，Anthropic 工程师明确表示：

> "Using your Claude Pro/Max subscription in OpenCode is not officially supported by Anthropic."

使用此方案的风险包括：
1. **随时失效** — Anthropic 持续更新检测机制
2. **违反 ToS** — 可能导致账号封禁
3. **无官方支持** — 遇到问题无法获得 Anthropic 帮助

**推荐替代方案**：使用 Anthropic API Key 按量计费，或通过 OpenCode Zen 获取官方支持的模型访问。

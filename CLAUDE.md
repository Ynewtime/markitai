# 说明

此目录是 Markit 项目的 dev worktree+branch。

## 开发偏好

1. Keep plan mode concise, remove unnecessary grammar.
2. Ask me questions about design/implementation decisions before coding.
3. Use only English for ASCII arts.
4. 请尽可能使用最新的依赖，如果对依赖是否是最新版本（结合当下的时间戳）有疑义，请调用 agent-browser 查询最新版本信息。

## Browser Automation

> source: https://github.com/vercel-labs/agent-browser

Run `agent-browser --help` for all commands.

Use `agent-browser` for web automation.

Core workflow:
1. `agent-browser open <url>` - Navigate to page
2. `agent-browser snapshot -i` - Get interactive elements with refs (@e1, @e2)
3. `agent-browser click @e1` / `fill @e2 "text"` - Interact using refs
4. Re-snapshot after page changes

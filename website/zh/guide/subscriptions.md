# Claude 与 Copilot 订阅

markitai 可以用 Claude Code 或 GitHub Copilot 订阅代替按 token 计费的 API 密钥，经由 markitai（或 `markitai serve`）所在机器上的厂商 CLI。

```mermaid
flowchart LR
  M[markitai] --> E[Python extra] --> C[厂商 CLI] --> S[你的订阅]
```

| | Claude | Copilot |
|---|---|---|
| 模型名 | `claude-agent/<模型>`，例如 `claude-agent/haiku` | `copilot/<模型>`，例如 `copilot/auto` |
| Python extra | `claude-agent` | `copilot` |
| CLI | Claude Code 2.1.284 或更新 | Copilot CLI 1.0.90 或更新 |
| 登录 | `markitai auth claude login` | `markitai auth copilot login` |
| 检查 | `markitai auth claude status` | `markitai auth copilot status` |

## 安装

安装脚本问到“安装 Claude Code CLI?”或“安装 Copilot CLI?”时回答是（默认否），就会装好 CLI 和对应的 extra；已装好的 CLI 只补上 extra。

<!--@include: ../../snippets/setup.md-->

::: details 手动安装
```bash
# Claude
uv tool install "markitai[claude-agent]" --force    # pip：pip install "markitai[claude-agent]"
npm install -g @anthropic-ai/claude-code@2.1.284    # 或：curl -fsSL https://claude.ai/install.sh | bash
                                                    # Windows：irm https://claude.ai/install.ps1 | iex
# Copilot
uv tool install "markitai[copilot]" --force
npm install -g @github/copilot@1.0.90-2             # 或：curl -fsSL https://gh.io/copilot-install | bash
                                                    # Windows：winget install GitHub.Copilot
```
:::

## 登录

在运行 markitai 的机器上登录。工作台提示订阅“尚未安装”或“尚未登录”时，在服务器所在的机器上处理，再重新加载模型列表。

- **Claude：**已登录订阅时，markitai 对 Claude Code 隐藏 `ANTHROPIC_API_KEY`，以免改按该密钥计费。
- **Copilot：**带“Copilot Requests”权限的 `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` 可代替登录。

## ChatGPT

不需要 CLI，也不需要 extra：见 [ChatGPT 订阅](/zh/guide/subscription-chatgpt)。

# Claude 与 Copilot 订阅

markitai 可以用 Claude Code 或 GitHub Copilot 订阅代替按 token 计费的 API 密钥，经由 markitai（或 `markitai serve`）所在机器上的厂商官方 CLI。

```mermaid
flowchart LR
  M[markitai] --> C[厂商 CLI] --> S[你的订阅]
```

| | Claude | Copilot |
|---|---|---|
| 模型名 | `claude-agent/<模型>`，例如 `claude-agent/haiku` | `copilot/<模型>`，例如 `copilot/auto` |
| CLI | Claude Code 2.1.284 或更新 | Copilot CLI 1.0.90-2 或更新 |
| 登录 | `markitai auth claude login` | `markitai auth copilot login` |
| 检查 | `markitai auth claude status` | `markitai auth copilot status` |

## 安装

CLI 需要自己安装；markitai 在 `PATH` 中查找（或 `CLAUDE_CLI_PATH`、`COPILOT_CLI_PATH` 指定的位置）：

```bash
# Claude
npm install -g @anthropic-ai/claude-code@2.1.284    # 或：curl -fsSL https://claude.ai/install.sh | bash
                                                    # Windows：irm https://claude.ai/install.ps1 | iex
# Copilot
npm install -g @github/copilot@1.0.90-2             # 或：curl -fsSL https://gh.io/copilot-install | bash
                                                    # Windows：winget install GitHub.Copilot
```

CLI 会自行更新：更新的版本照常使用，并提示一次；更旧的版本会被拒绝。

## 登录

在 markitai 运行的机器上登录。单独运行 `markitai auth` 会列出所有订阅；脚本用 `status --json`。工作台提示订阅“未安装”或“未登录”时，到服务器所在机器上处理，再重新加载模型列表。markitai 不会在转换过程中发起登录。

- **Claude：** 只认订阅登录。markitai 不把 `ANTHROPIC_API_KEY` 等 API 设置传给 Claude Code，因此不会改为按密钥计费。
- **Copilot：** 带“Copilot Requests”权限的 `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` 可以代替登录。

没有 `llm.model_list` 也没有 `MODEL` 时，已登录的订阅会加入自动模型池（`claude-agent/haiku`、`copilot/auto`）。订阅模型不接受 `api_key`、`api_base`、`max_tokens` 或 `reasoning_effort`，费用显示为未知；`llm.max_cost_per_document_usd` 为正数时会拒绝它们（[价格](/zh/guide/pricing)）。

## ChatGPT

通过 Codex CLI：见 [ChatGPT 订阅](/zh/guide/subscription-chatgpt)。

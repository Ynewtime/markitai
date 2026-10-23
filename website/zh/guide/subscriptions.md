# Claude 与 Copilot 订阅

markitai 可以通过 Claude Code 或 GitHub Copilot 订阅发送请求，不必使用按 token 计费的 API 密钥。两者都经由厂商自己的命令行，运行在 markitai（或 `markitai serve`）所在的机器上。

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

运行安装脚本，在“安装 Claude Code CLI?”或“安装 Copilot CLI?”处回答是（两者默认都是否）。它用厂商的安装程序装好 CLI（失败时改用 bun、pnpm 或 npm；Windows 上 Copilot 先用 winget），并加上对应的 extra。已经装好的 CLI 只补上 extra。

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

在运行 markitai 的机器上执行表中的登录命令，再用检查命令或 `markitai doctor` 确认。网页工作台提示某个订阅“尚未安装”或“尚未登录”时，在服务器所在的机器上完成安装或登录，然后重新加载模型列表。

- **Claude：**设置了 `ANTHROPIC_API_KEY` 时，Claude Code 会按该密钥计费而不是用订阅；已登录订阅时，markitai 会把这个变量对 Claude Code 隐藏。
- **Copilot：**带有“Copilot Requests”权限的 `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` 可以代替登录。

## ChatGPT

不需要 CLI，也不需要 extra：见 [ChatGPT 订阅](/zh/guide/subscription-chatgpt)。

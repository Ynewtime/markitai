# Claude and Copilot subscriptions

markitai can use a Claude Code or GitHub Copilot subscription instead of a per-token API key, through the vendor's CLI on the machine where markitai (or `markitai serve`) runs.

```mermaid
flowchart LR
  M[markitai] --> E[Python extra] --> C[Vendor CLI] --> S[Your subscription]
```

| | Claude | Copilot |
|---|---|---|
| Model names | `claude-agent/<model>`, e.g. `claude-agent/haiku` | `copilot/<model>`, e.g. `copilot/auto` |
| Python extra | `claude-agent` | `copilot` |
| CLI | Claude Code 2.1.284 or later | Copilot CLI 1.0.90 or later |
| Sign in | `markitai auth claude login` | `markitai auth copilot login` |
| Check | `markitai auth claude status` | `markitai auth copilot status` |

## Install

The setup script installs each CLI and its extra when you answer yes to "Install Claude Code CLI?" or "Install Copilot CLI?" (default no); a CLI already installed only gets its extra.

<!--@include: ../snippets/setup.md-->

::: details Manual install
```bash
# Claude
uv tool install "markitai[claude-agent]" --force    # pip: pip install "markitai[claude-agent]"
npm install -g @anthropic-ai/claude-code@2.1.284    # or: curl -fsSL https://claude.ai/install.sh | bash
                                                    # Windows: irm https://claude.ai/install.ps1 | iex
# Copilot
uv tool install "markitai[copilot]" --force
npm install -g @github/copilot@1.0.90-2             # or: curl -fsSL https://gh.io/copilot-install | bash
                                                    # Windows: winget install GitHub.Copilot
```
:::

## Sign in

Sign in where markitai runs. When the workbench says a subscription "is not installed" or "is not signed in", fix it on the server's machine and reload the model list.

- **Claude:** markitai hides `ANTHROPIC_API_KEY` from Claude Code while a subscription is signed in, so the key is not billed instead.
- **Copilot:** a `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` with the "Copilot Requests" permission replaces signing in.

## ChatGPT

Needs no CLI and no extra: see [ChatGPT subscription](/guide/subscription-chatgpt).

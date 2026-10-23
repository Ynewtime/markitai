# Claude and Copilot subscriptions

markitai can send requests through a Claude Code or GitHub Copilot subscription instead of a per-token API key. Each goes through the vendor's command line on the machine where markitai (or `markitai serve`) runs.

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

Run the setup script and answer yes to "Install Claude Code CLI?" or "Install Copilot CLI?" (both default to no). It installs the CLI with the vendor's installer (bun, pnpm or npm as a fallback; winget for Copilot on Windows) and adds the matching extra. A CLI that is already installed only gets its extra.

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

Run the sign-in command from the table on the machine where markitai runs, then check it with the status command or `markitai doctor`. When the web workbench says a subscription "is not installed" or "is not signed in", run the install or sign-in on the server's machine and reload the model list.

- **Claude:** with `ANTHROPIC_API_KEY` set, Claude Code would bill that key instead of the subscription; markitai hides the variable from Claude Code while a subscription is signed in.
- **Copilot:** a `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` with the "Copilot Requests" permission works instead of signing in.

## ChatGPT

Needs no CLI and no extra: see [ChatGPT subscription](/guide/subscription-chatgpt).

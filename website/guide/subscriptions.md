# Claude and Copilot subscriptions

markitai can use a Claude Code or GitHub Copilot subscription instead of a per-token API key, through the vendor's official CLI on the machine where markitai (or `markitai serve`) runs.

```mermaid
flowchart LR
  M[markitai] --> C[Vendor CLI] --> S[Your subscription]
```

| | Claude | Copilot |
|---|---|---|
| Model names | `claude-agent/<model>`, e.g. `claude-agent/haiku` | `copilot/<model>`, e.g. `copilot/auto` |
| CLI | Claude Code 2.1.284 or later | Copilot CLI 1.0.90-2 or later |
| Sign in | `markitai auth claude login` | `markitai auth copilot login` |
| Check | `markitai auth claude status` | `markitai auth copilot status` |

## Install

Install the CLI yourself; markitai finds it on `PATH` (or at `CLAUDE_CLI_PATH`, `COPILOT_CLI_PATH`):

```bash
# Claude
npm install -g @anthropic-ai/claude-code@2.1.284    # or: curl -fsSL https://claude.ai/install.sh | bash
                                                    # Windows: irm https://claude.ai/install.ps1 | iex
# Copilot
npm install -g @github/copilot@1.0.90-2             # or: curl -fsSL https://gh.io/copilot-install | bash
                                                    # Windows: winget install GitHub.Copilot
```

The CLIs update themselves: a newer release is used, with a one-time warning; an older one is refused.

## Sign in

Sign in where markitai runs. `markitai auth` alone shows every subscription; `status --json` is for scripts. When the workbench says a subscription "is not installed" or "is not signed in", fix it on the server's machine and reload the model list. markitai never starts a sign-in during a conversion.

- **Claude:** only a subscription sign-in counts. markitai keeps `ANTHROPIC_API_KEY` and other API settings away from Claude Code, so the key is never billed instead.
- **Copilot:** a `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` with the "Copilot Requests" permission replaces signing in.

With no `llm.model_list` and no `MODEL`, a signed-in subscription joins the automatic model pool (`claude-agent/haiku`, `copilot/auto`). Subscription models take no `api_key`, `api_base`, `max_tokens` or `reasoning_effort`, their cost shows as unknown, and a positive `llm.max_cost_per_document_usd` refuses them ([Prices](/guide/pricing)).

## ChatGPT

Through the Codex CLI: see [ChatGPT subscription](/guide/subscription-chatgpt).

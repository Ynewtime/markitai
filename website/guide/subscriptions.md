# Claude and Copilot subscriptions

markitai can send requests through a Claude Code or GitHub Copilot subscription instead of a per-token API key. Both go through the vendor's own command line, which markitai finds on the machine where it runs (`markitai serve` included). If the web workbench says a subscription "is not installed" or "is not signed in", this page has the command to run on that machine; reload the model list afterwards.

## Claude

Models are named `claude-agent/<model>`, for example `claude-agent/haiku`.

1. Install the Python extra: `uv tool install "markitai[claude-agent]" --force` (pip: `pip install "markitai[claude-agent]"`).
2. Install the Claude Code CLI, release 2.1.284 or later: `npm install -g @anthropic-ai/claude-code@2.1.284` (macOS and Linux also: `curl -fsSL https://claude.ai/install.sh | bash`; Windows: `irm https://claude.ai/install.ps1 | iex`).
3. Sign in: `markitai auth claude login`.

Check it with `markitai auth claude status` and `markitai doctor`.

With `ANTHROPIC_API_KEY` set, Claude Code bills that key instead of the subscription; markitai hides the variable from Claude Code while a subscription is signed in.

## Copilot

Models are named `copilot/<model>`, for example `copilot/auto`.

1. Install the Python extra: `uv tool install "markitai[copilot]" --force`.
2. Install the Copilot CLI, release 1.0.90 or later: `npm install -g @github/copilot@1.0.90-2` (or `curl -fsSL https://gh.io/copilot-install | bash`; Windows: `winget install GitHub.Copilot`).
3. Sign in: `markitai auth copilot login`. A `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` with the "Copilot Requests" permission works too.

## ChatGPT

See [ChatGPT subscription](/guide/subscription-chatgpt).

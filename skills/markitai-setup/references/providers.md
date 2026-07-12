# LLM provider configuration

Long-form docs: <https://markitai.dev/guide/configuration>. Models follow the LiteLLM naming convention `provider/model`, so any LiteLLM-supported provider works.

## Model list in `markitai.json`

```json
{
  "llm": {
    "enabled": true,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-3.1-flash-lite-preview",
          "api_key": "env:GEMINI_API_KEY"
        }
      }
    ],
    "router_settings": { "routing_strategy": "simple-shuffle", "num_retries": 2, "timeout": 120 },
    "concurrency": 10
  }
}
```

Multiple entries with the same `model_name` load-balance through the litellm router (`routing_strategy`: `simple-shuffle` default, also `least-busy`, `usage-based-routing`, `latency-based-routing`). A per-model `weight` in `litellm_params` skews traffic; `weight: 0` disables a model without deleting it (enforced at first LLM use, not by `config validate`). `env:VAR` works for `api_key` and `api_base`. With no `model_list` at all, the `MODEL` env var acts as a single-model override.

## Naming examples

| Provider | Model string | Auth |
|---|---|---|
| Google Gemini | `gemini/gemini-3.1-flash-lite-preview` | `GEMINI_API_KEY` |
| OpenAI | `openai/gpt-5.4` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| OpenRouter | `openrouter/google/gemini-3.1-pro` | `OPENROUTER_API_KEY` |
| Ollama (local server) | `ollama/llama3.2` + `"api_base": "http://localhost:11434"` | none |
| Claude subscription | `claude-agent/sonnet` (aliases: `sonnet` `opus` `haiku` `inherit`, or full model strings) | Claude Code CLI login; extra: `claude-agent` |
| Copilot subscription | `copilot/gpt-5.4`, `copilot/claude-sonnet-4.6`, … (plan-dependent; no o1/o3) | Copilot CLI login; extra: `copilot` |
| ChatGPT subscription | `chatgpt/gpt-5.4`, `chatgpt/gpt-5.4-codex`, `chatgpt/codex-mini` | OAuth device flow on first use; no SDK needed |

Gemini has no CLI/subscription route — direct API key or OpenRouter only.

Retired 2025-02-13 (rejected upstream): `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`, `gpt-5`, `gpt-5.1`, `gpt-5.2` — use `gpt-5.4`.

## Custom endpoints (`api_base`)

```json
// Azure OpenAI — "azure/<name>" is your deployment alias from Azure Portal, not a model ID
{ "model": "azure/your-deployment-name", "api_key": "env:AZURE_API_KEY",
  "api_base": "https://your-resource.openai.azure.com", "api_version": "2025-02-01-preview" }

// Any OpenAI-compatible gateway
{ "model": "openai/served-model-name", "api_key": "env:GATEWAY_KEY",
  "api_base": "https://gateway.example.com/v1" }
```

## Vision (`--alt` / `--desc`)

Image analysis needs a vision-capable model. Remote API models are auto-detected via litellm; local providers pass images as file attachments, so pick a vision-capable model there too (e.g. `copilot/gpt-5.4`, `chatgpt/gpt-5.4`, `claude-agent/sonnet`). `markitai doctor` reports which vision model it detected.

## Local provider errors

| Error | Resolution |
|---|---|
| "SDK not installed" | install the `claude-agent` or `copilot` extra into the same environment as markitai |
| "CLI not found" | install Claude Code (`curl -fsSL https://claude.ai/install.sh \| bash`) or Copilot CLI (`curl -fsSL https://gh.io/copilot-install \| bash`) |
| "Not authenticated" | `markitai auth claude\|copilot login`; env alternatives: `COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`GITHUB_TOKEN`, `CLAUDE_CODE_USE_BEDROCK=1`/`CLAUDE_CODE_USE_VERTEX=1`/`CLAUDE_CODE_USE_FOUNDRY=1`; ChatGPT re-triggers OAuth on next use |
| "Rate limit" | subscription quota — wait or switch models |
| "Request timeout" | adaptive; very large documents legitimately take longer |

`markitai doctor` aggregates auth status for all three local providers and prints resolution hints.

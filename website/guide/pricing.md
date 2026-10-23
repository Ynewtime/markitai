# Prices and the cost shown

markitai reports what each conversion cost from the number of tokens a model used and a price list it ships. A request whose model has **no reviewed price** is counted, but its cost is **unknown**, not $0: the cost shown is the known part (the priced requests), and the total says it is incomplete (`! Cost is incomplete: …`).

| Model | Price |
|-------|-------|
| Listed OpenAI, Anthropic, Gemini and DeepSeek models, at the provider's own address | From the list |
| OpenRouter | The charge OpenRouter reports |
| A server of your own (`ollama/`, `lm_studio/`, `hosted_vllm/`, ...) | $0, at any address |
| A gateway or proxy (`openai/` with an `api_base`, `litellm_proxy/`), Bedrock, Vertex AI, other providers | Unknown |
| Subscriptions (`claude-agent/`, `copilot/`, `chatgpt/`) | Unknown: billed by your plan |

To get a cost:

- **Use a listed model**, such as the [defaults](/guide/configuration#defaults-markitai-picks-for-you).
- **Remove a custom `api_base`** (a proxy or relay): its prices are not the provider's.
- **Pin an exact model name** in `litellm_params.model` when the provider answers with a different model than you asked for.
- **Update markitai** if a newer release lists the model.

markitai has no setting for your own prices. Costs appear in the summary, `--json`, batch reports, the web workspace and MCP results.

## Spending limit

`llm.max_cost_per_document_usd` (`0`: no limit) stops a document's further requests once its known cost passes the cap. With a cap set, a model whose price is unknown, a subscription included, is refused before anything is sent. Requests already running still finish, so a document can end slightly above the cap. `llm.max_requests_per_document` (50) limits every model.

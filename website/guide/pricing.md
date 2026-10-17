# Prices and the cost shown

markitai reports what each conversion cost from the number of tokens a model used and a price list it ships. A request whose model has **no reviewed price** is counted, but its cost is **unknown**, not $0: the cost shown is the known part (the priced requests), and the total says it is incomplete.

Prices are known only for models markitai can match to its list, answered by the provider's own API address. To get a cost:

- **Use a listed model.** The list covers OpenAI, Anthropic, Gemini, DeepSeek and OpenRouter models, plus the subscription providers (which have no per-token charge).
- **Remove a custom `api_base`** (a proxy or relay): its prices are not the provider's.
- **Pin an exact model name** in `litellm_params.model` when the provider answers with a different model than you asked for.
- **Update markitai** if a newer release lists the model.

markitai has no setting for your own prices. `llm.max_cost_per_document_usd` cannot limit a model whose price is unknown; `llm.max_requests_per_document` still does.

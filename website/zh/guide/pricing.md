# 价格与费用显示

markitai 按模型用掉的 token 数和随包的价格表报告每次转换的费用。**没有审核过价格**的模型，其请求会被计数，但费用是**未知**，而不是 $0：显示的费用只是已知的那部分（有价格的请求），总额会标明不完整（`! Cost is incomplete: …`）。

| 模型 | 价格 |
|-------|-------|
| 价格表里的 OpenAI、Anthropic、Gemini 和 DeepSeek 模型，使用服务商自己的地址 | 按价格表 |
| OpenRouter | OpenRouter 报告的扣费 |
| 你自己的服务器（`ollama/`、`lm_studio/`、`hosted_vllm/`……） | $0，地址不限 |
| 网关或代理（带 `api_base` 的 `openai/`、`litellm_proxy/`）、Bedrock、Vertex AI、其他服务商 | 未知 |
| 订阅（`claude-agent/`、`copilot/`、`chatgpt/`） | 未知：按你的套餐计费 |

要得到费用：

- **使用价格表里的模型**，例如[默认模型](/zh/guide/configuration#defaults-markitai-picks-for-you)。
- **去掉自定义的 `api_base`**（代理或中转）：它的价格不是服务商的价格。
- **服务商返回的模型与请求的不同时**，在 `litellm_params.model` 中写明确切的模型名。
- **升级 markitai**，新版本可能已收录该模型。

markitai 没有设置自己价格的选项。费用显示在汇总、`--json`、批量报告、网页工作台和 MCP 结果中。

## 费用上限

`llm.max_cost_per_document_usd`（`0`：不限）在一个文档的已知费用超过上限后，停止它后续的请求。设了上限时，价格未知的模型（包括订阅）在发送任何内容之前就会被拒绝。已经在进行的请求仍会完成，所以一个文档最终可能略超上限。`llm.max_requests_per_document`（50）对所有模型都有效。

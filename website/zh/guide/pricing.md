# 价格与费用显示

markitai 按模型用掉的 token 数和随包的价格表报告每次转换的费用。**没有审核过价格**的模型，其请求会被计数，但费用是**未知**，而不是 $0：显示的费用只是已知价格的那部分请求，总额会标明不完整。

只有 markitai 能在价格表中匹配到的模型、并且由服务商自己的 API 地址应答时，价格才是已知的。要得到费用：

- **使用价格表里的模型。** 价格表涵盖 OpenAI、Anthropic、Gemini、DeepSeek 和 OpenRouter 的模型，以及订阅类服务商（它们没有按 token 的费用）。
- **去掉自定义的 `api_base`**（代理或中转）：它的价格不是服务商的价格。
- **服务商返回的模型与请求的不同时**，在 `litellm_params.model` 中写明确切的模型名。
- **升级 markitai**，新版本可能已收录该模型。

markitai 没有设置自己价格的选项。`llm.max_cost_per_document_usd` 无法限制价格未知的模型；`llm.max_requests_per_document` 仍然有效。

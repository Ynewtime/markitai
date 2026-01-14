# LiteLLM 架构演进调研报告

## 1. 概述
本报告旨在评估将 MarkIt 的底层 LLM 交互层迁移至 `litellm` 的可行性与收益。LiteLLM 是一个通用的 I/O 库，支持 100+ LLM 提供商，提供统一的 OpenAI 兼容接口。

## 2. 接口映射分析

### 2.1 核心调用
| 功能 | MarkIt 现状 | LiteLLM 方案 | 迁移建议 |
| :--- | :--- | :--- | :--- |
| **Provider 抽象** | `BaseLLMProvider` 子类 (OpenAI, Anthropic, Gemini...) | 无需子类，通过 `model="provider/model_name"` 区分 | 引入 `LiteLLMAdapter` 统一代理 |
| **Completion** | `provider.complete(messages, ...)` | `litellm.completion(model, messages, ...)` | 直接替换 |
| **Streaming** | `provider.stream(messages, ...)` 返回 `AsyncIterator[str]` | `litellm.completion(stream=True)` 返回 `ModelResponse` chunk | 编写 `stream_adapter` 提取 `delta.content` |
| **Image Analysis** | 各 Provider 独立实现图片构建逻辑 | 标准 OpenAI 多模态格式 (`{"type": "image_url", ...}`) | 统一使用 OpenAI 格式构建 Message |

### 2.2 配置管理
| 配置项 | MarkIt 现状 | LiteLLM 方案 |
| :--- | :--- | :--- | :--- |
| **API Key** | `LLMCredentialConfig` 中管理 | 环境变量或参数传递 |
| **Base URL** | 支持自定义 (如 Ollama) | 支持 `api_base` 参数 |
| **Model Name** | 用户指定 (如 `gpt-4o`) | 需加前缀 (如 `openai/gpt-4o`) |

### 2.3 成本与统计
| 功能 | MarkIt 现状 | LiteLLM 方案 | 迁移建议 |
| :--- | :--- | :--- | :--- |
| **Token 计算** | 强依赖 `tiktoken` | `litellm.encode()` / `token_counter()` | 切换至 LiteLLM 以支持更多模型 |
| **成本估算** | 简单的硬编码或配置表 | 内置模型价格表 + `completion_cost()` | 利用 LiteLLM 价格表，保留自定义覆盖能力 |

## 3. 关键差距与风险

### 3.1 依赖体积
LiteLLM 依赖较大，包含 `tiktoken`, `tokenizers`, `aiohttp`, `requests` 等。需评估对 CLI 启动速度的影响。

### 3.2 异常处理
MarkIt 封装了统一的 `LLMError` 体系。LiteLLM 有自己的异常体系 (`ContextWindowExceededError`, `RateLimitError`)。

### 3.3 Ollama 支持
LiteLLM 对 Ollama 的支持依赖于 `api_base` 的正确设置。需在 POC 中重点验证本地 Ollama 的连通性。

## 4. POC 验证计划

### 4.1 验证场景
1.  **OpenAI Compatible**: 验证 DeepSeek/OpenRouter 等兼容接口。
2.  **Native Providers**: 验证 Anthropic/Gemini 的原生调用。
3.  **Local LLM**: 验证 Ollama 的流式输出。
4.  **Cost Tracking**: 验证 `completion_cost` 的准确性。

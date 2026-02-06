# LiteLLM 深度调研报告 by Claude Opus 4.5

## 1. 概述

LiteLLM 是一个开源的 Python 库和 AI 网关，提供统一接口调用 **100+ LLM Provider** 的 API。当前最新稳定版本为 **v1.80.13** (2026年1月)。

核心价值：

- **统一 API**：使用 OpenAI 格式调用任意 LLM，包括最新的 GPT-5、Claude 4、Gemini 3 等
- **成本追踪**：内置精确的 token 计算和成本统计
- **负载均衡**：支持多 Provider 路由、故障转移和智能调度
- **MCP 网关**：原生支持 Model Context Protocol，统一工具调用
- **Agent 网关 (A2A)**：支持 LangGraph、Pydantic AI 等 Agent 框架的统一访问

```bash
pip install litellm==1.80.13
```

## 2. 最新版本特性 (v1.80.x)

### 2.1 v1.80.13 (2026年1月)

- Gemini 3 Flash Preview 完整支持
- Minimax 聊天补全和 TTS 支持
- Azure Sentinel 日志集成
- 5 个新 AI Provider 通过 openai_like 添加

### 2.2 v1.80.10 (2025年12月)

- **Agent (A2A) Gateway**：支持 Agent 成本追踪
- **GPT-5.2 系列**：完整支持 GPT-5.2、GPT-5.2-pro
- **227 个 Fireworks AI 模型**：大规模模型覆盖
- **MCP 支持 /chat/completions**：直接在聊天端点使用 MCP

### 2.3 v1.80.5 (2025年11月)

- **Gemini 3**：Day-0 支持 Gemini 3 模型和 thought signatures
- **Prompt Studio**：完整的提示词版本管理 UI
- **MCP Hub**：组织内 MCP 服务器发布和发现
- **Model Compare UI**：并排模型比较界面

### 2.4 v1.80.0 (2025年11月)

- **Agent Hub**：注册和发布 Agent 供组织使用
- **GPT-5.1 系列**：支持 OpenAI gpt-5.1 和 gpt-5.1-codex
- **RunwayML 集成**：视频生成、图像生成、TTS 完整支持
- **Prometheus 开源版**：监控指标现已开源

## 3. 支持的 Provider (100+)

### 3.1 主流商业云服务

| Provider | 前缀 | 说明 |
|----------|------|------|
| OpenAI | 无/openai/ | GPT-5.x, GPT-4o, o3, o1 系列 |
| Anthropic | 无/anthropic/ | Claude 4, Claude 3.5/3 系列 |
| Google Gemini | gemini/ | Gemini 3, Gemini 2.x, 1.5 系列 |
| Azure OpenAI | azure/ | 企业级 OpenAI 部署 |
| AWS Bedrock | bedrock/ | Claude, Llama, Titan 等 |
| Vertex AI | vertex_ai/ | Google Cloud 上的模型 |

### 3.2 开源推理平台

| Provider | 前缀 | 说明 |
|----------|------|------|
| Ollama | ollama/ | 本地模型推理 |
| vLLM | vllm/ | 高性能推理引擎 |
| LM Studio | lm_studio/ | 本地 GUI 推理 |
| Llamafile | llamafile/ | 单文件模型运行 |
| HuggingFace | huggingface/ | HF 推理端点 |

### 3.3 聚合平台

| Provider | 前缀 | 说明 |
|----------|------|------|
| OpenRouter | openrouter/ | 多模型聚合 |
| Groq | groq/ | 超低延迟推理 |
| Together AI | together_ai/ | 开源模型托管 |
| Fireworks AI | fireworks_ai/ | 227+ 模型支持 |
| DeepInfra | deepinfra/ | 高性价比推理 |

### 3.4 国产/区域服务

| Provider | 前缀 | 说明 |
|----------|------|------|
| DeepSeek | deepseek/ | 深度求索模型 |
| Dashscope (通义千问) | dashscope/ | 阿里云 Qwen API |
| Volcengine (火山引擎) | volcengine/ | 字节跳动 |
| Moonshot AI | moonshot/ | Kimi 模型 |
| Z.AI (智谱AI) | zai/ | GLM 系列 |
| Xiaomi MiMo | xiaomi_mimo/ | 小米模型 |
| MiniMax | minimax/ | MiniMax 模型 |

### 3.5 企业服务

| Provider | 前缀 | 说明 |
|----------|------|------|
| Databricks | databricks/ | 企业数据平台 |
| Snowflake | snowflake/ | 数据云 |
| SAP Gen AI Hub | sap/ | SAP 企业 AI |
| WatsonX | watsonx/ | IBM 企业 AI |
| Oracle OCI | oci/ | Oracle 云 |

### 3.6 新增 Provider (2025)

- **RunwayML**: 视频/图像生成
- **Fal AI**: 快速图像生成
- **Recraft**: 图像生成
- **LangGraph**: Agent 框架
- **Pydantic AI Agents**: A2A 网关
- **Manus**: AI Agent
- **GitHub Copilot**: 代码助手
- **Lemonade**: AMD GPU 本地推理

## 4. 核心功能

### 4.1 基础调用

```python
from litellm import completion

# OpenAI GPT-5
response = completion(
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic Claude 4 Sonnet
response = completion(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Google Gemini 3
response = completion(
    model="gemini/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Hello!"}]
)

# DeepSeek
response = completion(
    model="deepseek/deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Ollama 本地
response = completion(
    model="ollama/llama3.2",
    api_base="http://localhost:11434",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 4.2 Responses API (推理模型)

对于支持推理的模型 (GPT-5, o3 等)，使用 `responses()`:

```python
from litellm import responses

response = responses(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="medium"  # low, medium, high
)

print(response.choices[0].message.content)  # 回答
print(response.choices[0].message.reasoning_content)  # 推理过程
```

### 4.3 异步和流式

```python
from litellm import acompletion, completion
import asyncio

# 异步调用
async def main():
    response = await acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    return response

# 流式输出
response = completion(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## 5. Token 计算与成本追踪

### 5.1 核心函数

```python
from litellm import token_counter, cost_per_token, completion_cost, get_model_info

# 1. Token 计数
messages = [{"role": "user", "content": "Hello, how are you?"}]
token_count = token_counter(model="gpt-5", messages=messages)

# 2. 单价查询
prompt_cost, completion_cost = cost_per_token(
    model="gpt-5",
    prompt_tokens=100,
    completion_tokens=50
)

# 3. 请求成本计算
response = completion(model="gpt-5", messages=messages)
cost = completion_cost(completion_response=response)
print(f"Cost: ${cost:.6f}")

# 4. 模型信息查询
info = get_model_info("claude-sonnet-4-20250514")
print(f"Input: ${info['input_cost_per_token'] * 1_000_000:.2f}/1M tokens")
print(f"Output: ${info['output_cost_per_token'] * 1_000_000:.2f}/1M tokens")
print(f"Max context: {info['max_input_tokens']} tokens")
```

### 5.2 成本数据来源

LiteLLM 维护一个持续更新的定价数据库：
- 文件: `model_prices_and_context_window.json`
- 在线 API: `api.litellm.ai`
- 社区维护，欢迎贡献

### 5.3 自定义定价

```yaml
# config.yaml
model_list:
  - model_name: my-azure-model
    litellm_params:
      model: azure/gpt-4-deployment
      api_key: os.environ/AZURE_API_KEY
    model_info:
      input_cost_per_token: 0.00001   # 自定义输入价格
      output_cost_per_token: 0.00003  # 自定义输出价格
      cache_read_input_token_cost: 0.000001  # 缓存读取价格
```

### 5.4 Proxy 成本追踪

```bash
# 查询用户每日花费明细
curl -X GET 'http://localhost:4000/user/daily/activity?start_date=2026-01-01&end_date=2026-01-12' \
  -H 'Authorization: Bearer sk-...'
```

响应示例：

```json
{
  "results": [{
    "date": "2026-01-12",
    "metrics": {
      "spend": 0.0177,
      "prompt_tokens": 111,
      "completion_tokens": 1711,
      "total_tokens": 1822,
      "api_requests": 11
    },
    "breakdown": {
      "models": {
        "gpt-5-mini": {"spend": 0.01, "total_tokens": 1000}
      }
    }
  }]
}
```

### 5.5 主流模型成本对比 (2026年1月 官方最新)

> 数据来源：各厂商官方定价页面 (2026年1月12日更新)

#### OpenAI 模型定价

| Model | Input ($/1M) | Output ($/1M) | Cached Input | Context | 说明 |
|-------|--------------|---------------|--------------|---------|------|
| **GPT-5.2** | $1.75 | $14.00 | $0.18 | 400K | 最新旗舰，代码/Agent 最强 |
| GPT-5 | $1.25 | $10.00 | $0.125 | 256K | 旗舰模型 |
| GPT-5-mini | $0.25 | $2.00 | $0.025 | 128K | 性价比版 |
| GPT-5-nano | $0.05 | $0.40 | $0.005 | 128K | 超低成本 |
| GPT-4.1 | $2.00 | $8.00 | $0.50 | 1M | 非推理最强 |
| GPT-4.1-mini | $0.40 | $1.60 | $0.10 | 1M | 指令遵循优秀 |
| GPT-4o | $2.50 | $10.00 | $0.625 | 128K | 多模态旗舰 |
| GPT-4o-mini | $0.15 | $0.60 | $0.075 | 128K | 超低成本多模态 |
| **o3** | $2.00 | $8.00 | $0.50 | 200K | 推理模型降价版 |
| o4-mini | $1.10 | $4.40 | $0.275 | 200K | 推理性价比版 |
| o1 | $15.00 | $60.00 | $3.75 | 200K | 深度推理 |
| o1-pro | $150.00 | $600.00 | - | 200K | 专业版 |

#### Anthropic Claude 定价 (官方)

| Model | Input ($/1M) | Output ($/1M) | Cache Write | Cache Hit | Context |
|-------|--------------|---------------|-------------|-----------|---------|
| **Claude Opus 4.6** | $5.00 | $25.00 | $6.25 | $0.50 | 200K/1M* |
| Claude Opus 4.5 | $5.00 | $25.00 | $6.25 | $0.50 | 200K |
| Claude Opus 4.1 | $15.00 | $75.00 | $18.75 | $1.50 | 200K |
| Claude Opus 4 | $15.00 | $75.00 | $18.75 | $1.50 | 200K |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | $3.75 | $0.30 | 200K/1M* |
| Claude Sonnet 4 | $3.00 | $15.00 | $3.75 | $0.30 | 200K/1M* |
| **Claude Haiku 4.5** | $1.00 | $5.00 | $1.25 | $0.10 | 200K |
| Claude Haiku 3.5 | $0.80 | $4.00 | $1.00 | $0.08 | 200K |
| Claude Haiku 3 | $0.25 | $1.25 | $0.30 | $0.03 | 200K |

*Sonnet 4/4.5 支持 1M 上下文 beta (>200K 输入按 $6/$22.50 计费)

#### Google Gemini 定价 (官方)

| Model | Input ($/1M) | Output ($/1M) | Cache | Context | 说明 |
|-------|--------------|---------------|-------|---------|------|
| **Gemini 3 Pro Preview** | $2.00 / $4.00* | $12.00 / $18.00* | $0.20 | 2M | 最新旗舰 |
| **Gemini 3 Flash Preview** | $0.50 | $3.00 | $0.05 | 1M | 高性价比 |
| Gemini 2.5 Pro | $1.25 / $2.50* | $10.00 / $15.00* | $0.125 | 2M | 主力Pro模型 |
| Gemini 2.5 Flash | $0.30 | $2.50 | $0.03 | 1M | 混合推理 |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.01 | 1M | 超低成本 |
| Gemini 2.0 Flash | $0.10 | $0.40 | $0.025 | 1M | 多模态Agent |
| Gemini 2.0 Flash-Lite | $0.075 | $0.30 | - | 1M | 最低成本 |

*价格前为 ≤200K tokens，后为 >200K tokens

#### DeepSeek 定价 (官方 V3.2)

| Model | Input Cache Hit | Input Cache Miss | Output | Context | 说明 |
|-------|-----------------|------------------|--------|---------|------|
| **deepseek-chat** | $0.028 | $0.28 | $0.42 | 128K | V3.2 非思考模式 |
| **deepseek-reasoner** | $0.028 | $0.28 | $0.42 | 128K | V3.2 思考模式 |

> DeepSeek V3.2 于 2025年12月1日发布，定价大幅下调。支持 JSON Output、Tool Calls 等功能。

#### xAI Grok 定价

| Model | Input ($/1M) | Output ($/1M) | Cached Input | Context | 说明 |
|-------|--------------|---------------|--------------|---------|------|
| **Grok 4** | $3.00 | $15.00 | $0.75 | 256K | 旗舰推理 |
| Grok 4.1 Fast (Reasoning) | $0.20 | $0.50 | - | 2M | 超高性价比 |
| Grok 4.1 Fast (Non-Reasoning) | $0.20 | $0.50 | - | 2M | 非推理版 |
| Grok 3 | $3.00 | $15.00 | - | 131K | 旧版 |
| Grok 3 Mini | $0.30 | $0.50 | - | 131K | 小模型 |

#### 成本效益对比总结

| 使用场景 | 推荐模型 | 成本/1M tokens | 理由 |
|----------|----------|----------------|------|
| **极致低成本** | DeepSeek V3.2 | $0.028-$0.70 | 业界最低价，能力接近 GPT-4 |
| **高性价比推理** | Grok 4.1 Fast | $0.20-$0.50 | 2M 上下文，推理能力强 |
| **通用低成本** | Gemini 2.5 Flash-Lite | $0.10-$0.40 | Google 最便宜，功能全面 |
| **OpenAI 低成本** | GPT-4o-mini | $0.15-$0.60 | OpenAI 生态最便宜 |
| **多模态性价比** | Gemini 2.0 Flash | $0.10-$0.40 | 支持图像视频音频 |
| **顶级推理** | Claude Opus 4.6 | $5-$25 | 最强推理+128K输出+1M上下文 |
| **均衡选择** | Claude Sonnet 4.5 | $3-$15 | SWE-bench 第一 |
| **代码开发** | GPT-5.2 | $1.75-$14 | 代码/Agent 最强 |

## 6. Router 负载均衡

### 6.1 基本配置

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-4",  # 用户请求的名称
        "litellm_params": {
            "model": "gpt-5-mini",
            "api_key": "sk-openai-key"
        }
    },
    {
        "model_name": "gpt-4",  # 同名 = 负载均衡
        "litellm_params": {
            "model": "azure/gpt-4",
            "api_base": "https://xxx.openai.azure.com",
            "api_key": "azure-key"
        }
    }
]

router = Router(model_list=model_list)
response = router.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 6.2 路由策略

```yaml
# config.yaml
router_settings:
  routing_strategy: simple-shuffle  # 默认：随机分配
  # 可选值:
  # - simple-shuffle: 随机分配
  # - least-busy: 最少繁忙
  # - latency-based-routing: 基于延迟
  # - usage-based-routing: 基于使用量
  # - cost-based-routing: 选择最便宜
```

### 6.3 优先级和故障转移

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-primary
      api_key: os.environ/AZURE_API_KEY
      order: 1  # 最高优先级

  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-fallback
      api_key: os.environ/AZURE_API_KEY_2
      order: 2  # 备用

router_settings:
  fallbacks: [{"gpt-4": ["claude-sonnet-4"]}]  # 模型组故障转移
  context_window_fallbacks: [{"gpt-4": ["gpt-4-32k"]}]  # 上下文超限时
  num_retries: 3
  timeout: 60
```

### 6.4 权重配置

```yaml
model_list:
  - model_name: chat
    litellm_params:
      model: gpt-5-mini
    weight: 0.7  # 70% 流量

  - model_name: chat
    litellm_params:
      model: claude-sonnet-4-20250514
    weight: 0.3  # 30% 流量
```

## 7. MCP 集成 (Model Context Protocol)

### 7.1 概述

LiteLLM 提供 MCP Gateway，让所有支持的模型都能使用 MCP 工具：

- 统一端点访问所有 MCP 工具
- 按 Key/Team 控制 MCP 访问权限
- MCP Hub：组织内 MCP 服务器发现

### 7.2 配置 MCP 服务器

```yaml
# config.yaml
general_settings:
  store_model_in_db: true

model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY
```

### 7.3 通过 UI 添加 MCP

1. 导航到 LiteLLM UI -> "MCP Servers"
2. 点击 "Add New MCP Server"
3. 输入 MCP Server URL 和传输类型 (HTTP/SSE/stdio)
4. 支持 OAuth 2.0 认证

### 7.4 使用 MCP 工具

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

response = client.responses.create(
    model="gpt-5",
    input=[{
        "role": "user",
        "content": "Summarize the latest PR in BerriAI/litellm",
        "type": "message"
    }],
    tools=[{
        "type": "mcp",
        "server_label": "github_mcp",
        "server_url": "litellm_proxy/mcp/github",
        "require_approval": "never"
    }],
    stream=True
)
```

### 7.5 Cursor IDE 集成

```json
{
  "mcpServers": {
    "LiteLLM": {
      "url": "http://localhost:4000/mcp",
      "headers": {
        "x-litellm-api-key": "Bearer sk-1234"
      }
    }
  }
}
```

## 8. Agent Gateway (A2A)

### 8.1 支持的 Agent 框架

- LangGraph Agents
- Azure AI Foundry Agents
- Pydantic AI Agents
- Bedrock AgentCore
- Vertex AI Agent Engine

### 8.2 调用 Agent

```python
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4
import httpx

base_url = "http://localhost:4000/a2a/my-agent"
headers = {"Authorization": "Bearer sk-1234"}

async with httpx.AsyncClient(headers=headers) as httpx_client:
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello!"}],
                "messageId": uuid4().hex,
            }
        )
    )
    response = await client.send_message(request)
```

### 8.3 Agent 成本追踪

v1.80.10 新增 Agent 级别的成本追踪：

- 每个查询的成本
- 每 Token 定价
- 在仪表盘查看 Agent 使用情况

## 9. LiteLLM Proxy Server

### 9.1 快速启动

```bash
# Docker (推荐)
docker run \
  -e STORE_MODEL_IN_DB=True \
  -p 4000:4000 \
  docker.litellm.ai/berriai/litellm:v1.80.13-stable

# Pip
pip install 'litellm[proxy]'
litellm --config config.yaml
```

### 9.2 完整配置示例

```yaml
# config.yaml
model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini
    litellm_params:
      model: gemini/gemini-3-flash-preview
      api_key: os.environ/GEMINI_API_KEY

  - model_name: local
    litellm_params:
      model: ollama/llama3.2
      api_base: http://localhost:11434

litellm_settings:
  drop_params: true
  set_verbose: false

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
  timeout: 60
  redis_host: localhost  # 分布式部署时使用
  redis_port: 6379

general_settings:
  master_key: sk-1234
  database_url: postgresql://user:pass@localhost/litellm
  store_model_in_db: true
```

### 9.3 Prompt Studio

v1.80.5 引入的提示词管理解决方案：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="sk-1234")

response = client.chat.completions.create(
    model="gpt-5",
    extra_body={
        "prompt_id": "your-prompt-id",
        "prompt_version": 2,  # 可选：指定版本
        "prompt_variables": {"name": "value"}  # 可选：变量
    }
)
```

功能：

- 创建和测试提示词
- 动态变量支持 `{{variable_name}}`
- 自动版本控制
- 版本历史和回滚

## 10. 可观测性

### 10.1 支持的平台 (50+)

| 类别 | 平台 |
|------|------|
| LLM 可观测 | Langfuse, Langsmith, Helicone, Arize, Braintrust, Galileo |
| 通用监控 | Prometheus, Datadog, OpenTelemetry, Azure Sentinel |
| 日志存储 | S3, GCS, Azure Storage, SumoLogic |
| 告警 | PagerDuty, Slack, Email (新增预算告警) |
| 云原生 | Cloudzero (UI 直接配置) |

### 10.2 配置回调

```python
import litellm

# 单个平台
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# 多平台
litellm.success_callback = ["langfuse", "prometheus", "datadog"]
```

### 10.3 Prometheus 指标 (开源)

v1.80.0 后 Prometheus 指标完全开源：

```yaml
# 暴露的指标
# - litellm_requests_total
# - litellm_request_duration_seconds
# - litellm_tokens_total
# - litellm_cost_total
# - litellm_errors_total
```

## 11. Guardrails (内容安全)

### 11.1 内置 Guardrails

v1.80.10 新增内置内容过滤器：

- 有害内容检测
- 偏见检测
- 图像内容过滤

### 11.2 Guardrail 负载均衡

支持在多个 Guardrail 提供商之间负载均衡。

### 11.3 集成第三方

- EnkryptAI Guardrails (v1.78.0)
- 自定义 Guardrail

## 12. 性能优化

### 12.1 延迟改进

- v1.80.13：懒加载 109 个组件，大幅减少冷启动时间
- v1.80.0：/embeddings API P95 延迟降低 92%
- v1.78.0：P99 延迟降低 70%
- v1.75.5：Redis 启用时 P99 延迟降低 50%

### 12.2 性能基准

官方数据：**8ms P95 延迟 @ 1k RPS**

### 12.3 最佳实践

```yaml
# 高性能配置
litellm_settings:
  request_timeout: 60
  num_retries: 3

router_settings:
  redis_host: localhost  # 启用 Redis 分布式状态
  routing_strategy: least-busy  # 最少繁忙路由
```

## 13. 使用场景

### 13.1 多云 LLM 网关

```yaml
model_list:
  # 主要：OpenAI
  - model_name: main
    litellm_params:
      model: gpt-5-mini
      order: 1

  # 备用：Azure
  - model_name: main
    litellm_params:
      model: azure/gpt-5
      order: 2

  # 故障转移：Anthropic
  - model_name: fallback
    litellm_params:
      model: claude-sonnet-4-20250514

router_settings:
  fallbacks: [{"main": ["fallback"]}]
```

### 13.2 成本优化

```yaml
router_settings:
  routing_strategy: cost-based-routing  # 自动选择最便宜的
```

### 13.3 本地 + 云端混合

```yaml
model_list:
  # 敏感数据：本地 Ollama
  - model_name: private
    litellm_params:
      model: ollama/llama3.2
      api_base: http://localhost:11434

  # 一般任务：云端
  - model_name: cloud
    litellm_params:
      model: gpt-5-mini
```

### 13.4 企业级部署

- 多租户成本追踪
- 按项目/团队预算限制
- SSO 集成 (Okta, Azure AD) + SCIM 自动同步
- Virtual Keys 安全访问控制
- 审计日志

## 14. 与其他工具集成

### 14.1 OpenAI Agents SDK

```python
from agents.extensions.models.litellm_model import LitellmModel
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    model=LitellmModel(model="anthropic/claude-sonnet-4", api_key="..."),
    tools=[...]
)

result = await Runner.run(agent, "What's the weather?")
```

### 14.2 DSPy

```python
# DSPy 内部使用 LiteLLM
import dspy
lm = dspy.LM("anthropic/claude-3-opus-20240229")
```

### 14.3 LangChain

```python
from langchain_community.llms import LiteLLM

llm = LiteLLM(model="gpt-5-mini")
```

## 15. 最佳实践

### 15.1 错误处理

```python
from litellm import completion
from litellm.exceptions import (
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    BudgetExceededError
)

try:
    response = completion(model="gpt-5", messages=[...])
except RateLimitError:
    # 等待并重试，或切换 Provider
    pass
except BudgetExceededError:
    # 预算超限
    pass
```

### 15.2 环境变量

```bash
# 主流 Provider
export OPENAI_API_KEY="sk-xxx"
export ANTHROPIC_API_KEY="sk-ant-xxx"
export GEMINI_API_KEY="xxx"
export DEEPSEEK_API_KEY="xxx"
export OPENROUTER_API_KEY="xxx"

# Azure
export AZURE_API_KEY="xxx"
export AZURE_API_BASE="https://xxx.openai.azure.com"
export AZURE_API_VERSION="2024-02-01"

# AWS Bedrock (使用 AWS 凭证)
export AWS_ACCESS_KEY_ID="xxx"
export AWS_SECRET_ACCESS_KEY="xxx"
export AWS_REGION_NAME="us-east-1"
```

### 15.3 安全建议

1. 使用环境变量存储 API Key
2. 在 Proxy 模式使用 Virtual Key，不暴露真实 Key
3. 配置预算限制防止意外高额账单
4. 启用审计日志
5. 使用 Guardrails 过滤有害内容

## 16. 总结

### 核心优势

| 特性 | 价值 |
|------|------|
| 统一接口 | 100+ Provider，一套代码 |
| 成本追踪 | 精确的 token 和费用计算 |
| 负载均衡 | 多策略路由，故障自动转移 |
| MCP 网关 | 统一工具调用接口 |
| Agent 网关 | 支持多种 Agent 框架 |
| Prompt Studio | 提示词版本管理 |
| 企业特性 | 多租户、预算、SSO |

### 适用场景

- 需要对接多个 LLM Provider 的应用
- 需要精确成本追踪和预算控制
- 需要高可用性和故障转移
- 需要统一的 API 网关管理
- 企业级 LLM 平台建设

### 不适用场景

- 只使用单一 Provider 且无特殊需求
- 对延迟极度敏感 (增加约 8ms)

## 参考资源

- **GitHub**: https://github.com/BerriAI/litellm
- **文档**: https://docs.litellm.ai
- **PyPI**: https://pypi.org/project/litellm
- **模型定价**: https://models.litellm.ai
- **Release Notes**: https://docs.litellm.ai/release_notes
- **Discord/Slack**: https://www.litellm.ai/support


---


# LiteLLM 深度调研报告：AI 网关与多模型统一架构 by Google Gemini

## 1. 核心定位：LLM 时代的 "TCP/IP 协议层"

LiteLLM 已不仅仅是一个 Python SDK，它已演变为企业级 AI 基础设施的标准网关。

* **核心价值**：**"Write once, call 100+ LLMs"**。它将所有模型（OpenAI, Anthropic, DeepSeek, Gemini, Bedrock 等）的差异化 API 强行抹平为 **OpenAI 兼容格式**。
* **最新趋势**：全面支持"思考型"模型（Reasoning Models）的参数统一，解决各家推理参数（Thinking/Reasoning Effort）不一致的痛点。

---

## 2. 核心功能与代码实战

### 2.1 统一调用范式 (The Universal Call)

无论后端是闭源模型还是本地 Ollama，调用方式完全一致。

```python
from litellm import completion
import os

# 统一入口，自动处理格式转换
response = completion(
    model="os.environ/MODEL_NAME", # 支持从环境变量读取模型名
    messages=[{"role": "user", "content": "你好，LiteLLM"}]
)

```

### 2.2 🔥 DeepSeek 与推理模型支持 (2026 重点)

LiteLLM 现已完美支持 **DeepSeek V3 (Chat)** 和 **DeepSeek R1 (Reasoner)**，并实现了跨厂商的"思考参数"归一化。

* **DeepSeek R1 (推理模型) - 推荐写法**
LiteLLM 允许你使用 OpenAI 的 `reasoning_effort` 参数来控制 DeepSeek R1，实现代码无缝迁移。
```python
response = completion(
    model="deepseek/deepseek-reasoner",
    api_key="sk-...",
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个大？"}],
    # LiteLLM 黑科技：自动将 reasoning_effort 映射为 DeepSeek/Gemini 的对应参数
    reasoning_effort="medium" # 可选: low, medium, high
)

# 获取思维链 (Chain of Thought)
# DeepSeek 返回在 reasoning_content，LiteLLM 统一封装
print("思考过程:", response.choices[0].message.reasoning_content)
print("最终答案:", response.choices[0].message.content)

```



### 2.3 多大厂模型集成 (主流配置)

| 厂商 | 模型标识 (Model String) | 2026 新特性支持 |
| --- | --- | --- |
| **Google** | `gemini/gemini-2.0-flash-exp` | 支持 `thinking_level` (通过 `reasoning_effort` 映射) |
| **OpenAI** | `gpt-4o`, `o1`, `o3-mini` | 原生支持，自动处理 o1 系列的 `streaming` 限制 |
| **Anthropic** | `anthropic/claude-3-5-sonnet` | 自动处理 System Prompt 剥离，支持 Prompt Caching |
| **AWS** | `bedrock/us.anthropic.claude-3-5...` | 支持 Bedrock 的 `/converse` 新接口，延迟更低 |
| **Ollama** | `ollama/llama3.2` | 自动处理本地 API Base，支持 JSON Mode |

---

## 3. Token 计算与成本风控 (Enterprise Ready)

LiteLLM 的成本管理已进化为**实时风控系统**，不再依赖简单的本地字典。

### 3.1 动态价格同步

LiteLLM 维护了一个每日更新的模型价格注册表（GitHub Reop），确保新模型（如 DeepSeek V3）发布后，无需发版即可更新价格。

### 3.2 成本计算实战

```python
from litellm import completion

res = completion(
    model="deepseek/deepseek-chat", # 极低成本模型
    messages=[{"role": "user", "content": "写首诗"}]
)

# 隐藏参数中包含精确的成本分析
usage_data = res._hidden_params
print(f"输入Token: {usage_data['input_tokens']}")
print(f"输出Token: {usage_data['output_tokens']}")
print(f"本次花费(USD): ${usage_data['response_cost']}")
# 输出示例: 本次花费(USD): $0.0000002

```

### 3.3 预算管理 (Budget Manager)

在 Proxy 模式下，支持多级预算控制，防止 Token 爆炸。

* **用户级预算**: `user_id="user_123", max_budget="1.00"` (1美元封顶)
* **Key 级预算**: 为某个 API Key 设置月度限额。
* **Tag 级预算**: 针对项目（如 `tags=["project_alpha"]`）设置总预算。

---

## 4. 生产环境架构：Router 与 Proxy

在生产环境中，强烈建议使用 **LiteLLM Proxy**（独立服务）而非仅作为 SDK 使用。

### 4.1 智能路由 (Router)

解决"OpenAI 经常 500 报错"或"DeepSeek 偶尔限流"的问题。

```python
from litellm import Router

model_list = [
    { # 优先路由：DeepSeek (便宜)
        "model_name": "smart-model",
        "litellm_params": {"model": "deepseek/deepseek-chat", "api_key": "sk-deepseek..."}
    },
    { # 故障转移/兜底：OpenAI (稳定)
        "model_name": "smart-model",
        "litellm_params": {"model": "gpt-4o", "api_key": "sk-openai..."}
    }
]

# 策略：usage-based-routing (基于负载), latency-based-routing (基于延迟)
router = Router(model_list=model_list, routing_strategy="latency-based-routing")

# 调用别名 "smart-model"，自动选择最快的线路
resp = await router.acompletion(model="smart-model", messages=[...])

```

### 4.2 Proxy Server (独立网关)

通过 Docker 启动一个兼容 OpenAI 接口的网关服务器。

**核心优势**：

1. **秘钥隔离**：开发者只需持有 Proxy 的虚拟 Key（`sk-proxy-123`），无需接触真实的 `sk-openai/sk-deepseek`。
2. **协议转换**：后端可以是 Ollama、Azure、Bedrock，前端统一暴露为标准的 `https://proxy/v1/chat/completions`。
3. **护栏 (Guardrails)**：集成 LLM Guard，自动拦截 PII (敏感信息) 或 攻击性 Prompt。

**Config.yaml 配置示例 (2026 版):**

```yaml
model_list:
  - model_name: gpt-4-prod
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 1000 # 限制每分钟请求数

  - model_name: deepseek-r1
    litellm_params:
      model: deepseek/deepseek-reasoner
      api_key: os.environ/DEEPSEEK_API_KEY
      # 强制参数覆盖
      extra_body:
        reasoning_effort: "medium"

litellm_settings:
  # 开启数据合规日志 (不记录具体 Content，只记元数据)
  send_instacart_logs: true
  callbacks: ["langfuse"] # 原生集成 Langfuse 监控

```

---

## 5. 总结：如何选择集成方式？

| 场景 | 推荐方式 | 理由 |
| --- | --- | --- |
| **Python 脚本 / 个人开发** | **Python SDK** | `pip install litellm`，极其轻量，立刻支持 DeepSeek/Gemini。 |
| **企业后端 / 微服务架构** | **LiteLLM Proxy** | 集中管理 Key，统一计费，统一鉴权。业务服务只需请求 Proxy。 |
| **高可用 / 跨境业务** | **Router SDK** | 通过配置多个 Azure/AWS 区域的 Endpoint，实现 99.99% 可用性。 |
| **本地离线 Agent** | **SDK + Ollama** | 利用 LiteLLM 自动处理 Prompt Template，无缝切换云端/本地模型。 |

**一句话建议**：现在就开始使用 `reasoning_effort` 参数统一你的推理模型调用，通过 LiteLLM Proxy 统一管理你的 DeepSeek 和 OpenAI 流量。

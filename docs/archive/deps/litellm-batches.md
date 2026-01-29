---
source: https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/docs/my-website/docs/batches.md
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /batches

Covers Batches, Files

| Feature | Supported | Notes |
|-------|-------|-------|
| Supported Providers | OpenAI, Azure, Vertex, Bedrock, vLLM | - |
| ✨ Cost Tracking | ✅ | LiteLLM Enterprise only |
| Logging | ✅ | Works across all logging integrations |

## Quick Start

- Create File for Batch Completion

- Create Batch Request

- List Batches

- Retrieve the Specific Batch and File Content


<Tabs>
<TabItem value="proxy" label="LiteLLM PROXY Server">

```bash
$ export OPENAI_API_KEY="sk-..."

$ litellm

# RUNNING on http://0.0.0.0:4000
```

**Create File for Batch Completion**

```shell
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Create Batch Request**

```bash
curl http://localhost:4000/v1/batches \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "input_file_id": "file-abc123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
    }'
```

**Retrieve the Specific Batch**

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
```


**List Batches**

```bash
curl http://localhost:4000/v1/batches \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
```

</TabItem>
<TabItem value="sdk" label="SDK">

**Create File for Batch Completion**

```python
import litellm
import os
import asyncio

os.environ["OPENAI_API_KEY"] = "sk-.."

file_name = "openai_batch_completions.jsonl"
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = await litellm.acreate_file(
    file=open(file_path, "rb"),
    purpose="batch",
    custom_llm_provider="openai",
)
print("Response from creating file=", file_obj)
```

**Create Batch Request**

```python
import litellm
import os
import asyncio

create_batch_response = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id,
    custom_llm_provider="openai",
    metadata={"key1": "value1", "key2": "value2"},
)

print("response from litellm.create_batch=", create_batch_response)
```

**Retrieve the Specific Batch and File Content**

```python
    # Maximum wait time before we give up
    MAX_WAIT_TIME = 300

    # Time to wait between each status check
    POLL_INTERVAL = 5

    #Time waited till now
    waited = 0

    # Wait for the batch to finish processing before trying to retrieve output
    # This loop checks the batch status every few seconds (polling)

    while True:
        retrieved_batch = await litellm.aretrieve_batch(
            batch_id=create_batch_response.id,
            custom_llm_provider="openai"
        )

        status = retrieved_batch.status
        print(f"⏳ Batch status: {status}")

        if status == "completed" and retrieved_batch.output_file_id:
            print("✅ Batch complete. Output file ID:", retrieved_batch.output_file_id)
            break
        elif status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"❌ Batch failed with status: {status}")

        await asyncio.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        if waited > MAX_WAIT_TIME:
            raise TimeoutError("❌ Timed out waiting for batch to complete.")

print("retrieved batch=", retrieved_batch)
# just assert that we retrieved a non None batch

assert retrieved_batch.id == create_batch_response.id

# try to get file content for our original file

file_content = await litellm.afile_content(
    file_id=batch_input_file_id, custom_llm_provider="openai"
)

print("file content = ", file_content)
```

**List Batches**

```python
list_batches_response = litellm.list_batches(custom_llm_provider="openai", limit=2)
print("list_batches_response=", list_batches_response)
```

</TabItem>

</Tabs>


## Multi-Account / Model-Based Routing

Route batch operations to different provider accounts using model-specific credentials from your `config.yaml`. This eliminates the need for environment variables and enables multi-tenant batch processing.

### How It Works

**Priority Order:**
1. **Encoded Batch/File ID** (highest) - Model info embedded in the ID
2. **Model Parameter** - Via header (`x-litellm-model`), query param, or request body
3. **Custom Provider** (fallback) - Uses environment variables

### Configuration

```yaml
model_list:
  - model_name: gpt-4o-account-1
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-account-1-key
      api_base: https://api.openai.com/v1

  - model_name: gpt-4o-account-2
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-account-2-key
      api_base: https://api.openai.com/v1

  - model_name: azure-batches
    litellm_params:
      model: azure/gpt-4
      api_key: azure-key-123
      api_base: https://my-resource.openai.azure.com
      api_version: "2024-02-01"
```

### Usage Examples

#### Scenario 1: Encoded File ID with Model

When you upload a file with a model parameter, LiteLLM encodes the model information in the file ID. All subsequent operations automatically use those credentials.

```bash
# Step 1: Upload file with model
curl http://localhost:4000/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: gpt-4o-account-1" \
  -F purpose="batch" \
  -F file="@batch.jsonl"

# Response includes encoded file ID:
# {
#   "id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
#   ...
# }

# Step 2: Create batch - automatically routes to gpt-4o-account-1
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# Batch ID is also encoded with model:
# {
#   "id": "batch_bGl0ZWxsbTpiYXRjaF82OTIwM2IzNjg0MDQ4MTkwYTA3ODQ5NDY3YTFjMDJkYTttb2RlbCxncHQtNG8tYWNjb3VudC0x",
#   "input_file_id": "file-bGl0ZWxsbTpmaWxlLUxkaUwzaVYxNGZRVlpYcU5KVEdkSjk7bW9kZWwsZ3B0LTRvLWFjY291bnQtMQ",
#   ...
# }

# Step 3: Retrieve batch - automatically routes to gpt-4o-account-1
curl http://localhost:4000/v1/batches/batch_bGl0ZWxsbTpiYXRjaF82OTIwM2IzNjg0MDQ4MTkwYTA3ODQ5NDY3YTFjMDJkYTttb2RlbCxncHQtNG8tYWNjb3VudC0x \
  -H "Authorization: Bearer sk-1234"
```

**✅ Benefits:**
- No need to specify model on every request
- File and batch IDs "remember" which account created them
- Automatic routing for retrieve, cancel, and file content operations

#### Scenario 2: Model via Header/Query Parameter

Specify the model for each request without encoding it in the ID.

```bash
# Create batch with model header
curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: gpt-4o-account-2" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# Or use query parameter
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'

# List batches for specific model
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2" \
  -H "Authorization: Bearer sk-1234"
```

**✅ Use Case:**
- One-off batch operations
- Different models for different operations
- Explicit control over routing

#### Scenario 3: Environment Variables (Fallback)

Traditional approach using environment variables when no model is specified.

```bash
export OPENAI_API_KEY="sk-env-key"

curl http://localhost:4000/v1/batches \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

**✅ Use Case:**
- Backward compatibility
- Simple single-account setups
- Quick prototyping

### Complete Multi-Account Example

```bash
# Upload file to Account 1
FILE_1=$(curl -s http://localhost:4000/v1/files \
  -H "x-litellm-model: gpt-4o-account-1" \
  -F purpose="batch" \
  -F file="@batch1.jsonl" | jq -r '.id')

# Upload file to Account 2
FILE_2=$(curl -s http://localhost:4000/v1/files \
  -H "x-litellm-model: gpt-4o-account-2" \
  -F purpose="batch" \
  -F file="@batch2.jsonl" | jq -r '.id')

# Create batch on Account 1 (auto-routed via encoded file ID)
BATCH_1=$(curl -s http://localhost:4000/v1/batches \
  -d "{\"input_file_id\": \"$FILE_1\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}" | jq -r '.id')

# Create batch on Account 2 (auto-routed via encoded file ID)
BATCH_2=$(curl -s http://localhost:4000/v1/batches \
  -d "{\"input_file_id\": \"$FILE_2\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}" | jq -r '.id')

# Retrieve both batches (auto-routed to correct accounts)
curl http://localhost:4000/v1/batches/$BATCH_1
curl http://localhost:4000/v1/batches/$BATCH_2

# List batches per account
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-1"
curl "http://localhost:4000/v1/batches?model=gpt-4o-account-2"
```

### SDK Usage with Model Routing

```python
import litellm
import asyncio

# Upload file with model routing
file_obj = await litellm.acreate_file(
    file=open("batch.jsonl", "rb"),
    purpose="batch",
    model="gpt-4o-account-1",  # Route to specific account
)

print(f"File ID: {file_obj.id}")
# File ID is encoded with model info

# Create batch - automatically uses gpt-4o-account-1 credentials
batch = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=file_obj.id,  # Model info embedded in ID
)

print(f"Batch ID: {batch.id}")
# Batch ID is also encoded

# Retrieve batch - automatically routes to correct account
retrieved = await litellm.aretrieve_batch(
    batch_id=batch.id,  # Model info embedded in ID
)

print(f"Batch status: {retrieved.status}")

# Or explicitly specify model
batch2 = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id="file-regular-id",
    model="gpt-4o-account-2",  # Explicit routing
)
```

### How ID Encoding Works

LiteLLM encodes model information into file and batch IDs using base64:

```
Original:  file-abc123
Encoded:   file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8tdGVzdA
           └─┬─┘ └──────────────────┬──────────────────────┘
          prefix      base64(litellm:file-abc123;model,gpt-4o-test)

Original:  batch_xyz789
Encoded:   batch_bGl0ZWxsbTpiYXRjaF94eXo3ODk7bW9kZWwsZ3B0LTRvLXRlc3Q
           └──┬──┘ └──────────────────┬──────────────────────┘
           prefix       base64(litellm:batch_xyz789;model,gpt-4o-test)
```

The encoding:
- ✅ Preserves OpenAI-compatible prefixes (`file-`, `batch_`)
- ✅ Is transparent to clients
- ✅ Enables automatic routing without additional parameters
- ✅ Works across all batch and file endpoints

### Supported Endpoints

All batch and file endpoints support model-based routing:

| Endpoint | Method | Model Routing |
|----------|--------|---------------|
| `/v1/files` | POST | ✅ Via header/query/body |
| `/v1/files/{file_id}` | GET | ✅ Auto from encoded ID + header/query |
| `/v1/files/{file_id}/content` | GET | ✅ Auto from encoded ID + header/query |
| `/v1/files/{file_id}` | DELETE | ✅ Auto from encoded ID |
| `/v1/batches` | POST | ✅ Auto from file ID + header/query/body |
| `/v1/batches` | GET | ✅ Via header/query |
| `/v1/batches/{batch_id}` | GET | ✅ Auto from encoded ID |
| `/v1/batches/{batch_id}/cancel` | POST | ✅ Auto from encoded ID |

## **Supported Providers**:
### [Azure OpenAI](./providers/azure#azure-batches-api)
### [OpenAI](#quick-start)
### [Vertex AI](./providers/vertex#batch-apis)
### [Bedrock](./providers/bedrock_batches)
### [vLLM](./providers/vllm_batches)


## How Cost Tracking for Batches API Works

LiteLLM tracks batch processing costs by logging two key events:

| Event Type | Description | When it's Logged |
|------------|-------------|------------------|
| `acreate_batch` | Initial batch creation | When batch request is submitted |
| `batch_success` | Final usage and cost | When batch processing completes |

Cost calculation:

- LiteLLM polls the batch status until completion
- Upon completion, it aggregates usage and costs from all responses in the output file
- Total `token` and `response_cost` reflect the combined metrics across all batch responses

## [Swagger API Reference](https://litellm-api.up.railway.app/#/batch)


---


# LiteLLM Batch API 支持深度调研报告 by Claude Opus 4.5

> 调研时间：2026年1月
> 基于 LiteLLM 官方文档、GitHub Issues/Discussions 及社区最新信息

---

## 1. 概述

### 1.1 什么是 Batch API

Batch API 是各大 LLM Provider 提供的异步批量推理接口，相比实时 API 通常能获得 **50% 的成本优惠**，适用于不需要即时响应的大规模数据处理场景。

### 1.2 LiteLLM 的两种"Batch"概念

LiteLLM 中存在两个容易混淆的概念：

| 功能 | 说明 | 是否节省成本 |
|------|------|-------------|
| `batch_completion()` | 并发调用多个实时 API，本质是多线程同步请求 | 否 |
| `/v1/batches` Endpoint | 真正的异步 Batch API，对接 Provider 原生 Batch 接口 | **是 (约50%)** |

**本报告聚焦后者——真正的 Provider Batch API 支持。**

---

## 2. Provider 支持矩阵

| Provider | 支持状态 | 成熟度 | 特殊要求 | 备注 |
|----------|----------|--------|----------|------|
| **OpenAI** | Full | Stable | 无 | 最完整的支持 |
| **Azure OpenAI** | Full | Stable | 需配置 `mode: batch` | 支持负载均衡 |
| **AWS Bedrock** | Partial | Beta | S3 Bucket + IAM Role | 仅支持 Anthropic 模型 |
| **Vertex AI (Google)** | Partial | Beta | GCS Bucket | 存在一些已知问题 |
| **vLLM** | Full | Stable | 自托管 vLLM 服务 | v1.80+ 新增 |
| **Anthropic (直连)** | Passthrough | Stable | 需使用 Passthrough 端点 | 非统一 API |

---

## 3. 各 Provider 详细分析

### 3.1 OpenAI

**支持状态：完全支持 ✅**

OpenAI 是 LiteLLM Batch API 支持最完整的 Provider，所有标准操作均可正常工作。

#### 配置示例

```yaml
model_list:
  - model_name: "gpt-4o-batch"
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: batch
```

#### 使用方式

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",  # LiteLLM Proxy Key
    base_url="http://localhost:4000"
)

# 1. 上传文件
file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch",
    extra_body={"model": "gpt-4o-batch"}
)

# 2. 创建 Batch
batch = client.batches.create(
    input_file_id=file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 3. 查询状态
status = client.batches.retrieve(batch.id)
```

#### 支持的操作

- `POST /v1/files` - 上传文件
- `POST /v1/batches` - 创建 Batch 任务
- `GET /v1/batches/{id}` - 获取状态
- `GET /v1/batches/{id}/cancel` - 取消任务
- `GET /v1/batches` - 列出所有 Batch
- `GET /v1/files/{id}/content` - 下载结果

---

### 3.2 Azure OpenAI

**支持状态：完全支持 ✅**

Azure OpenAI 的 Batch API 支持同样成熟，且支持**跨多个 Azure 部署的负载均衡**。

#### 配置示例

```yaml
model_list:
  - model_name: "gpt-4o-batch"
    litellm_params:
      model: azure/gpt-4o-mini-deployment-1
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
    model_info:
      mode: batch  # 关键配置
  - model_name: "gpt-4o-batch"  # 同名配置实现负载均衡
    litellm_params:
      model: azure/gpt-4o-mini-deployment-2
      api_base: os.environ/AZURE_API_BASE_2
      api_key: os.environ/AZURE_API_KEY_2
    model_info:
      mode: batch

litellm_settings:
  enable_loadbalancing_on_batch_endpoints: true  # 启用负载均衡
```

#### 特性

- **负载均衡**：可配置多个 Azure 部署，LiteLLM 自动选择
- **Managed Files**：无需知道具体 Azure 部署名，使用虚拟模型名即可
- **自动转换**：LiteLLM 会将 JSONL 中的模型名替换为实际部署名

#### 使用方式

```python
# 通过 custom-llm-provider header 指定
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    extra_headers={"custom-llm-provider": "azure"}
)
```

---

### 3.3 AWS Bedrock

**支持状态：部分支持 ⚠️ (Beta)**

Bedrock Batch API 在 v1.77.2 (2025年9月) 引入，但有一些限制。

#### 限制

> **重要**: LiteLLM 目前仅支持 Bedrock 上的 **Anthropic Claude 模型** 进行 Batch 处理。其他 Bedrock 模型暂不支持。

#### 前置要求

1. **S3 Bucket** - 用于存储输入/输出文件
2. **IAM Role** - 需要 Bedrock 执行角色 (batch role ARN)
3. **可选: KMS 密钥** - 用于 S3 加密

#### 配置示例

```yaml
model_list:
  - model_name: "bedrock-batch-claude"
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      # Batch 特有配置
      s3_bucket_name: my-batch-bucket
      s3_region_name: us-west-2
      s3_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      s3_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_batch_role_arn: arn:aws:iam::123456789:role/BedrockBatchRole
      # 可选: KMS 加密
      # s3_encryption_key_id: arn:aws:kms:us-west-2:123456789:key/xxx
    model_info:
      mode: batch
```

#### 已知问题

1. **files/{file_id}/content 端点不完整** - 获取结果可能需要直接从 S3 下载 ([Issue #16186](https://github.com/BerriAI/litellm/issues/16186))
2. **实现逻辑曾缺失** - v1.77.2 之前版本的 `create_batch` 逻辑不完整 ([Issue #15563](https://github.com/BerriAI/litellm/issues/15563))

#### 输出格式

Bedrock 的输出格式与 OpenAI 略有不同：

```json
{
  "recordId": "request-1",
  "modelInput": { "messages": [...], "max_tokens": 1000 },
  "modelOutput": {
    "content": [...],
    "id": "msg_abc123",
    "model": "claude-3-5-sonnet-20240620-v1:0",
    "role": "assistant",
    "stop_reason": "end_turn",
    "usage": { "input_tokens": 15, "output_tokens": 10 }
  }
}
```

---

### 3.4 Vertex AI (Google)

**支持状态：部分支持 ⚠️ (Beta)**

Vertex AI Batch API 支持是较新的功能，存在一些已知问题。

#### 前置要求

1. **GCS Bucket** - 用于存储批处理文件
2. **Service Account** - 需要 GCS 和 Vertex AI 权限

#### 环境变量配置

```bash
# GCS 配置
export GCS_BUCKET_NAME="my-batch-bucket"
export GCS_PATH_SERVICE_ACCOUNT="/path/to/service_account.json"

# Vertex AI 配置
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
export VERTEXAI_LOCATION="us-central1"
export VERTEXAI_PROJECT="my-project"
```

#### 已知问题

1. **File ID 过长问题** ([Issue #16876](https://github.com/BerriAI/litellm/issues/16876))
   - 上传文件返回的 GCS 路径可能超过 64 字符限制
   - 需要 URL 编码处理

2. **缺少 custom_id 支持** ([Issue #14044](https://github.com/BerriAI/litellm/issues/14044))
   - Vertex AI 原生不支持 `custom_id`
   - LiteLLM 需要额外维护请求映射

3. **结果获取复杂**
   - File ID 需要 URL 编码
   - 示例：`gs://bucket/path` -> `gs%3A%2F%2Fbucket%2Fpath`

#### 使用示例

```python
# 上传文件
file_obj = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch",
    extra_headers={"custom-llm-provider": "vertex_ai"}
)

# 创建 Batch (注意需要 URL 编码 file_id)
import urllib.parse
encoded_file_id = urllib.parse.quote_plus(file_obj.id)

batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    extra_headers={"custom-llm-provider": "vertex_ai"}
)
```

---

### 3.5 vLLM (自托管)

**支持状态：完全支持 ✅ (v1.80+)**

LiteLLM v1.80 新增了对自托管 vLLM 服务的 Batch API 支持。

#### 配置示例

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-2-7b-chat-hf
      api_base: http://localhost:8000  # vLLM 服务地址
```

#### 使用方式

需要通过 `x-litellm-model` header 指定模型：

```bash
curl http://localhost:4000/v1/files \
  -H "Authorization: Bearer sk-1234" \
  -H "x-litellm-model: my-vllm-model" \
  -F purpose="batch" \
  -F file="@batch_requests.jsonl"
```

---

### 3.6 Anthropic (直连)

**支持状态：Passthrough 模式 ✅**

Anthropic 原生 Message Batches API 可通过 LiteLLM 的 **Passthrough 端点**访问，但这不是统一的 OpenAI 格式。

#### Passthrough 使用方式

```bash
curl --request POST \
  --url http://localhost:4000/anthropic/v1/messages/batches \
  --header "x-api-key: $LITELLM_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "anthropic-beta: message-batches-2024-09-24" \
  --header "content-type: application/json" \
  --data '{
    "requests": [
      {
        "custom_id": "my-first-request",
        "params": {
          "model": "claude-3-5-sonnet-20241022",
          "max_tokens": 1024,
          "messages": [
            {"role": "user", "content": "Hello, world"}
          ]
        }
      }
    ]
  }'
```

#### 注意事项

- 使用 Anthropic 原生 API 格式，非 OpenAI 兼容格式
- 需要添加 `anthropic-beta: message-batches-2024-09-24` header
- 不支持统一的 `/v1/batches` 端点

---

## 4. LiteLLM Managed Files (推荐方案)

### 4.1 什么是 Managed Files

从 v1.69.0 开始，LiteLLM 引入了 **Managed Files** 功能，这是处理跨 Provider Batch 任务的推荐方式。

### 4.2 主要优势

| 特性 | 说明 |
|------|------|
| **无需知道部署名** | 使用 LiteLLM 虚拟模型名而非 Provider 部署名 |
| **自动路由** | File ID 中嵌入路由信息，后续操作自动路由 |
| **无需数据库** | 状态信息编码在 ID 中，支持 Proxy 重启 |
| **访问控制** | Proxy Admin 可控制用户对 Batch 模型的访问 |

### 4.3 配置方式

```yaml
model_list:
  - model_name: "gpt-4o-batch"
    litellm_params:
      model: azure/gpt-4o-deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
    model_info:
      mode: batch  # 标记为 batch 模型
```

### 4.4 使用示例

```python
# 上传时指定 target_model_names
file = client.files.create(
    file=open("./request.jsonl", "rb"),
    purpose="batch",
    extra_body={"target_model_names": "gpt-4o-batch"}
)

# File ID 自动包含路由信息
# 例如: file-bGl0ZWxsbTpmaWxlLWFiYzEyMzttb2RlbCxncHQtNG8taWZvb2Q

# 后续操作无需再指定 provider
batch = client.batches.create(
    input_file_id=file.id,  # 自动路由
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

---

## 5. 请求文件格式 (JSONL)

### 5.1 标准格式

```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-batch", "messages": [{"role": "user", "content": "How are you?"}], "max_tokens": 1000}}
```

### 5.2 关键字段说明

| 字段 | 说明 |
|------|------|
| `custom_id` | 自定义请求 ID，用于关联结果 |
| `method` | HTTP 方法，通常为 `POST` |
| `url` | 端点路径，如 `/v1/chat/completions` |
| `body.model` | **使用 LiteLLM 虚拟模型名** (Managed Files 时) |

---

## 6. 常见问题与解决方案

### 6.1 Provider 选择问题

**问题**：不确定该使用哪个 `custom_llm_provider`

**解决方案**：使用 Managed Files，让 LiteLLM 自动处理

```python
# 通过 extra_body 指定模型，自动确定 provider
file = client.files.create(
    file=open("batch.jsonl", "rb"),
    purpose="batch",
    extra_body={"model": "my-batch-model"}  # 使用 model_list 中的名称
)
```

### 6.2 Vertex AI File ID 过长

**问题**：GCS 路径超过 64 字符限制

**解决方案**：URL 编码 File ID

```python
import urllib.parse
encoded_id = urllib.parse.quote_plus(file_obj.id)
```

### 6.3 Bedrock 结果获取失败

**问题**：`/files/{id}/content` 端点不返回结果

**解决方案**：直接从 S3 获取

```python
import litellm

content = await litellm.afile_content(
    file_id="s3://bucket-name/path/to/output.jsonl",
    custom_llm_provider="bedrock",
    aws_region_name="us-west-2"
)
```

### 6.4 模型名不匹配

**问题**：JSONL 中的模型名与 Provider 部署名不一致导致失败

**解决方案**：
1. 使用 Managed Files + `target_model_names`
2. LiteLLM 会自动将虚拟模型名替换为实际部署名

---

## 7. 版本演进时间线

| 版本 | 时间 | 重要更新 |
|------|------|----------|
| v1.69.0 | 2025年5月 | Managed Files 支持 Batches；Azure 负载均衡 |
| v1.77.2 | 2025年9月 | Bedrock Batches API 初步支持 |
| v1.79.1 | 2025年11月 | Batch API Rate Limiting |
| v1.80.0 | 2025年12月 | vLLM Batch + Files API；Batch API Spend Tracking |

---

## 8. 建议与最佳实践

### 8.1 Provider 选择建议

| 场景 | 推荐 Provider | 理由 |
|------|---------------|------|
| 生产环境首选 | **OpenAI / Azure** | 最稳定，功能最完整 |
| 需要 Claude 模型 | **Bedrock** | 通过 AWS 获得 50% 折扣 |
| Google Cloud 环境 | **Vertex AI** | 注意已知问题 |
| 自托管模型 | **vLLM** | 无额外成本 |

### 8.2 配置最佳实践

1. **始终设置 `mode: batch`** - 让 LiteLLM 正确识别 batch 模型
2. **使用 Managed Files** - 简化跨 Provider 操作
3. **启用负载均衡** - 对于 Azure 多部署场景
4. **注意版本兼容** - 确保使用 v1.69.0+ 以获得完整 Batch 支持

### 8.3 监控与成本追踪

从 v1.80.0 开始，LiteLLM 支持 **Batch API Spend Tracking**，可通过 metadata 进行精细成本追踪：

```python
batch = client.batches.create(
    input_file_id=file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "project": "research",
        "team": "ml-ops"
    }
)
```

---

## 9. 总结

LiteLLM 对 Batch API 的支持正在快速发展：

- **OpenAI/Azure** 已达到生产就绪状态
- **Bedrock/Vertex AI** 处于 Beta 阶段，存在一些限制
- **Anthropic 直连** 仅支持 Passthrough 模式
- **Managed Files** 是推荐的跨 Provider 解决方案

对于需要利用 Batch API 节省成本的场景，建议：
1. 优先考虑 OpenAI 或 Azure OpenAI
2. 使用 LiteLLM Managed Files 简化操作
3. 关注 GitHub Issues 获取最新问题修复情况

---

## 参考资料

- [LiteLLM Batch API 文档](https://docs.litellm.ai/docs/batches)
- [LiteLLM Managed Batches](https://docs.litellm.ai/docs/proxy/managed_batches)
- [Bedrock Batches](https://docs.litellm.ai/docs/providers/bedrock_batches)
- [Vertex Batch APIs](https://docs.litellm.ai/docs/providers/vertex_batch)
- [Azure OpenAI Batch](https://docs.litellm.ai/docs/providers/azure/)
- [vLLM Batch + Files API](https://docs.litellm.ai/docs/providers/vllm_batches)
- [GitHub Discussions #8958](https://github.com/BerriAI/litellm/discussions/8958)
- [GitHub Discussions #9632](https://github.com/BerriAI/litellm/discussions/9632)

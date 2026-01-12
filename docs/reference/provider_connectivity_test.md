# AI Provider API 连通性测试方法综合分析

## 概述

本报告分析了 OpenAI、Google Gemini、Anthropic、OpenRouter 和 Ollama 五个主要 AI Provider 是否提供类似 "echo" 的 API 连通性测试接口，以及其 Python SDK 的支持情况。

---

## 1. OpenAI

### 是否有专用的健康检查/Echo端点？
**❌ 没有** - OpenAI 不提供专门的 `/health`、`/ping` 或 `/echo` 端点。

### 可用的轻量级测试端点
| 端点 | 说明 | 是否需要认证 | 是否消耗Token |
|------|------|-------------|--------------|
| `GET /v1/models` | 列出可用模型 | ✅ 需要API Key | ❌ 不消耗 |

### Python SDK 支持
```python
from openai import OpenAI

client = OpenAI()

# 推荐的连通性测试方法
def test_openai_connectivity():
    try:
        models = client.models.list()
        return True
    except Exception as e:
        return False
```

### 推荐的最简单/稳健测试方法
```python
from openai import OpenAI

def test_openai_connection(api_key: str) -> dict:
    """测试 OpenAI API 连通性"""
    try:
        client = OpenAI(api_key=api_key, timeout=10.0)
        models = client.models.list()
        return {
            "status": "connected",
            "models_count": len(list(models)),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }
```

---

## 2. Anthropic (Claude)

### 是否有专用的健康检查/Echo端点？
**❌ 没有** - Anthropic 不提供专门的健康检查端点。

### 可用的轻量级测试端点
| 端点 | 说明 | 是否需要认证 | 是否消耗Token |
|------|------|-------------|--------------|
| `GET /v1/models` | 列出可用模型 | ✅ 需要API Key | ❌ 不消耗 |

### Python SDK 支持
```python
from anthropic import Anthropic

client = Anthropic()

# 推荐的连通性测试方法
def test_anthropic_connectivity():
    try:
        models = client.models.list()
        return True
    except Exception as e:
        return False
```

### 推荐的最简单/稳健测试方法
```python
from anthropic import Anthropic

def test_anthropic_connection(api_key: str) -> dict:
    """测试 Anthropic API 连通性"""
    try:
        client = Anthropic(api_key=api_key, timeout=10.0)
        models = client.models.list()
        return {
            "status": "connected",
            "models_count": len(list(models.data)),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }
```

---

## 3. Google Gemini

### 是否有专用的健康检查/Echo端点？
**❌ 没有** - Google Gemini 不提供专门的健康检查端点。

### 可用的轻量级测试端点
| 端点 | 说明 | 是否需要认证 | 是否消耗Token |
|------|------|-------------|--------------|
| `GET /v1beta/models` | 列出可用模型 | ✅ 需要API Key | ❌ 不消耗 |

### Python SDK 支持
```python
from google import genai

# 新版 SDK (google-genai)
client = genai.Client(api_key='YOUR_API_KEY')

# 推荐的连通性测试方法
def test_gemini_connectivity():
    try:
        for model in client.models.list():
            break  # 只需要确认能获取即可
        return True
    except Exception as e:
        return False
```

### 推荐的最简单/稳健测试方法
```python
from google import genai

def test_gemini_connection(api_key: str) -> dict:
    """测试 Google Gemini API 连通性"""
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        return {
            "status": "connected",
            "models_count": len(models),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }
```

---

## 4. OpenRouter

### 是否有专用的健康检查/Echo端点？
**❌ 没有** - OpenRouter 不提供专门的健康检查端点。

### 可用的轻量级测试端点
| 端点 | 说明 | 是否需要认证 | 是否消耗Token |
|------|------|-------------|--------------|
| `GET /api/v1/models` | 列出所有可用模型 | ⚠️ 可选（推荐带上） | ❌ 不消耗 |

### Python SDK 支持
**⚠️ OpenRouter 没有官方 Python SDK**，但提供 OpenAI 兼容的 API，可使用 OpenAI SDK。

```python
from openai import OpenAI

# 使用 OpenAI SDK 连接 OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY"
)
```

### 推荐的最简单/稳健测试方法
```python
import requests

def test_openrouter_connection(api_key: str = None) -> dict:
    """测试 OpenRouter API 连通性"""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return {
            "status": "connected",
            "models_count": len(data.get("data", [])),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }

# 或者使用 OpenAI SDK
from openai import OpenAI

def test_openrouter_via_openai_sdk(api_key: str) -> dict:
    """使用 OpenAI SDK 测试 OpenRouter 连通性"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=10.0
        )
        models = client.models.list()
        return {
            "status": "connected",
            "models_count": len(list(models)),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }
```

---

## 5. Ollama

### 是否有专用的健康检查/Echo端点？
**✅ 有！** - Ollama 是唯一提供健康检查端点的 Provider。

### 可用的轻量级测试端点
| 端点 | 说明 | 是否需要认证 | 是否消耗Token |
|------|------|-------------|--------------|
| `GET /` | 返回 "Ollama is running" | ❌ 不需要 | ❌ 不消耗 |
| `GET /api/tags` | 列出本地已下载的模型 | ❌ 不需要 | ❌ 不消耗 |

### Python SDK 支持
```python
import ollama

# 推荐的连通性测试方法
def test_ollama_connectivity():
    try:
        models = ollama.list()
        return True
    except Exception as e:
        return False
```

### 推荐的最简单/稳健测试方法
```python
import requests

def test_ollama_connection(base_url: str = "http://localhost:11434") -> dict:
    """测试 Ollama API 连通性（使用健康检查端点）"""
    try:
        # 方法1：使用根路径（最轻量）
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200 and "Ollama is running" in response.text:
            # 进一步获取模型列表
            tags_response = requests.get(f"{base_url}/api/tags", timeout=5)
            tags_data = tags_response.json()
            return {
                "status": "connected",
                "models_count": len(tags_data.get("models", [])),
                "error": None
            }
        return {
            "status": "unknown",
            "models_count": 0,
            "error": "Unexpected response"
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }

# 使用官方 SDK
import ollama

def test_ollama_via_sdk(host: str = "http://localhost:11434") -> dict:
    """使用官方 SDK 测试 Ollama 连通性"""
    try:
        client = ollama.Client(host=host)
        models = client.list()
        return {
            "status": "connected",
            "models_count": len(models.get("models", [])),
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "models_count": 0,
            "error": str(e)
        }
```

---

## 综合对比表

| Provider | 专用健康检查端点 | 推荐测试端点 | 需要认证 | SDK 支持 | 消耗 Token |
|----------|-----------------|-------------|---------|---------|-----------|
| OpenAI | ❌ | `/v1/models` | ✅ | ✅ `client.models.list()` | ❌ |
| Anthropic | ❌ | `/v1/models` | ✅ | ✅ `client.models.list()` | ❌ |
| Gemini | ❌ | `/v1beta/models` | ✅ | ✅ `client.models.list()` | ❌ |
| OpenRouter | ❌ | `/api/v1/models` | ⚠️ 可选 | ❌ 无官方SDK | ❌ |
| Ollama | ✅ `/` | `/` 或 `/api/tags` | ❌ | ✅ `ollama.list()` | ❌ |

---

## 统一测试工具类

```python
"""
统一的 AI Provider 连通性测试工具
"""
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseProviderHealthCheck(ABC):
    """Provider 健康检查基类"""
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        pass


class OpenAIHealthCheck(BaseProviderHealthCheck):
    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
    
    def test_connection(self) -> Dict[str, Any]:
        from openai import OpenAI
        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            models = list(client.models.list())
            return {"status": "connected", "models_count": len(models), "error": None}
        except Exception as e:
            return {"status": "failed", "models_count": 0, "error": str(e)}


class AnthropicHealthCheck(BaseProviderHealthCheck):
    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
    
    def test_connection(self) -> Dict[str, Any]:
        from anthropic import Anthropic
        try:
            client = Anthropic(api_key=self.api_key, timeout=self.timeout)
            models = client.models.list()
            return {"status": "connected", "models_count": len(list(models.data)), "error": None}
        except Exception as e:
            return {"status": "failed", "models_count": 0, "error": str(e)}


class GeminiHealthCheck(BaseProviderHealthCheck):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def test_connection(self) -> Dict[str, Any]:
        from google import genai
        try:
            client = genai.Client(api_key=self.api_key)
            models = list(client.models.list())
            return {"status": "connected", "models_count": len(models), "error": None}
        except Exception as e:
            return {"status": "failed", "models_count": 0, "error": str(e)}


class OpenRouterHealthCheck(BaseProviderHealthCheck):
    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
    
    def test_connection(self) -> Dict[str, Any]:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return {"status": "connected", "models_count": len(data.get("data", [])), "error": None}
        except Exception as e:
            return {"status": "failed", "models_count": 0, "error": str(e)}


class OllamaHealthCheck(BaseProviderHealthCheck):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout
    
    def test_connection(self) -> Dict[str, Any]:
        try:
            # 使用专用健康检查端点
            response = requests.get(self.base_url, timeout=self.timeout)
            if response.status_code == 200 and "Ollama is running" in response.text:
                # 获取模型列表
                tags = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout).json()
                return {"status": "connected", "models_count": len(tags.get("models", [])), "error": None}
            return {"status": "unknown", "models_count": 0, "error": "Unexpected response"}
        except Exception as e:
            return {"status": "failed", "models_count": 0, "error": str(e)}


# 使用示例
if __name__ == "__main__":
    # Ollama（本地，无需认证）
    ollama_check = OllamaHealthCheck()
    print("Ollama:", ollama_check.test_connection())
    
    # OpenRouter（可不带认证）
    openrouter_check = OpenRouterHealthCheck()
    print("OpenRouter:", openrouter_check.test_connection())
    
    # 以下需要 API Key
    # openai_check = OpenAIHealthCheck(api_key="sk-...")
    # anthropic_check = AnthropicHealthCheck(api_key="sk-ant-...")
    # gemini_check = GeminiHealthCheck(api_key="...")
```

---

## 最佳实践建议

### 1. 生产环境推荐做法

```python
import asyncio
from typing import Dict, Any

async def comprehensive_health_check(provider: str, config: dict) -> Dict[str, Any]:
    """
    生产级健康检查，包含：
    - 超时控制
    - 重试机制
    - 错误分类
    """
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(
                _do_health_check(provider, config),
                timeout=config.get("timeout", 10.0)
            )
            if result["status"] == "connected":
                return result
        except asyncio.TimeoutError:
            result = {"status": "timeout", "error": f"Timeout after {config.get('timeout', 10.0)}s"}
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))
    
    return result
```

### 2. 不同场景的推荐方案

| 场景 | 推荐方法 | 原因 |
|-----|---------|-----|
| 快速检查服务是否在线 | 调用 `/models` 端点 | 不消耗 token，响应快 |
| 验证 API Key 有效性 | 调用 `/models` 端点 | 会验证认证 |
| 验证模型是否可用 | 发送最小化请求（max_tokens=1） | 确保端到端可用 |
| 持续监控 | 定时调用 `/models` + 偶尔真实请求 | 平衡成本与覆盖度 |

---

## 结论

1. **只有 Ollama 提供专用的健康检查端点** (`/` 返回 "Ollama is running")
2. **所有云端 Provider (OpenAI/Anthropic/Gemini/OpenRouter) 都没有专用的 echo/ping 端点**
3. **最佳替代方案是使用 `/models` 端点**：
   - 不消耗 token
   - 能验证认证信息
   - 所有 Provider 都支持
   - Python SDK 都有对应方法

4. **SDK 支持情况**：
   - OpenAI: ✅ `client.models.list()`
   - Anthropic: ✅ `client.models.list()`
   - Gemini: ✅ `client.models.list()`
   - OpenRouter: ❌ 无官方SDK（可用OpenAI SDK兼容）
   - Ollama: ✅ `ollama.list()`
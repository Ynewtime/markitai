# Anthropic / Claude LLM 对 JSON 的支持

目前有几种方法可以让 Claude 返回结构化的 JSON 输出：

## 1. Tool Use（推荐方式）

通过 Anthropic Python SDK 的 **Tool Use** 功能，你可以定义一个带有 JSON Schema 的工具，Claude 会返回符合该 schema 的 JSON：

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "extract_person_info",
        "description": "提取人物信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "人名"},
                "age": {"type": "integer", "description": "年龄"},
                "occupation": {"type": "string", "description": "职业"}
            },
            "required": ["name", "age", "occupation"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_person_info"},  # 强制使用该工具
    messages=[
        {"role": "user", "content": "张三今年25岁，是一名软件工程师。"}
    ]
)

# 获取结构化输出
for block in response.content:
    if block.type == "tool_use":
        print(block.input)  # {'name': '张三', 'age': 25, 'occupation': '软件工程师'}
```

## 2. Prompt 指令方式

直接在 prompt 中要求返回 JSON（但不保证严格符合 schema）：

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {
            "role": "user", 
            "content": "请提取以下信息并以 JSON 格式返回：张三今年25岁，是一名软件工程师。\n返回格式：{\"name\": \"\", \"age\": 0, \"occupation\": \"\"}"
        }
    ]
)
```

---

## 推荐使用 Tool Use 的原因：

| 特性 | Tool Use | Prompt 指令 |
|------|----------|-------------|
| Schema 验证 | ✅ 严格遵循 | ❌ 不保证 |
| 类型安全 | ✅ 是 | ❌ 否 |
| 嵌套结构支持 | ✅ 好 | ⚠️ 不稳定 |
| 可靠性 | 高 | 中等 |

更多详情可参考 Anthropic 官方文档：https://docs.anthropic.com

---

# 主流 AI 平台 Python SDK JSON 输出格式配置调研报告

> 2025年1月

## 摘要

本报告对 OpenAI、Google Gemini、Anthropic、OpenRouter 和 Ollama 五大主流 AI 平台的 Python SDK 进行了深入调研，重点分析其 JSON 输出格式的配置能力。调研发现，各平台对结构化输出的支持程度存在显著差异：OpenAI 提供了最完善的原生 JSON Schema 验证机制，Google Gemini 支持丰富的类型定义，Anthropic 采用提示词引导方式，OpenRouter 作为聚合平台继承上游能力，Ollama 则提供了灵活的本地化配置选项。

---

## 功能支持总览

| 平台 | 原生JSON模式 | JSON Schema | 结构化输出 | Pydantic集成 | 推荐指数 |
|------|-------------|-------------|-----------|--------------|---------|
| OpenAI | ✅ 支持 | ✅ 完整支持 | ✅ 原生支持 | ✅ 支持 | ⭐⭐⭐⭐⭐ |
| Google Gemini | ✅ 支持 | ✅ 完整支持 | ✅ 原生支持 | ✅ 支持 | ⭐⭐⭐⭐⭐ |
| Anthropic | ❌ 不支持 | ❌ 不支持 | ⚠️ 提示词引导 | ⚠️ 第三方库 | ⭐⭐⭐ |
| OpenRouter | ✅ 透传支持 | ✅ 透传支持 | ✅ 依赖模型 | ✅ 兼容 | ⭐⭐⭐⭐ |
| Ollama | ✅ 支持 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ⭐⭐⭐⭐ |

---

## 1. OpenAI Python SDK

### 1.1 概述

OpenAI 的 Python SDK 提供了业界最完善的 JSON 输出支持，包括基础 JSON 模式和高级结构化输出（Structured Outputs）两种方案。

- **SDK 包名**: `openai`
- **安装命令**: `pip install openai`

### 1.2 方案一：JSON Mode（基础模式）

通过设置 `response_format` 参数为 `{"type": "json_object"}` 启用 JSON 模式。此模式保证输出为有效 JSON，但不验证具体结构。

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system", 
            "content": "你是一个助手，请以JSON格式返回结果。"
        },
        {
            "role": "user", 
            "content": "列出3种编程语言及其特点"
        }
    ]
)

print(response.choices[0].message.content)
```

### 1.3 方案二：Structured Outputs（结构化输出）

这是 OpenAI 2024 年推出的高级特性，通过 JSON Schema 精确定义输出结构，支持 `strict` 模式确保 100% 模式匹配。

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "programming_languages",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "paradigm": {"type": "string"},
                                "year_created": {"type": "integer"},
                                "popular_frameworks": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["name", "paradigm", "year_created"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["languages"],
                "additionalProperties": False
            }
        }
    },
    messages=[
        {"role": "user", "content": "列出3种流行的编程语言信息"}
    ]
)
```

### 1.4 方案三：Pydantic 集成

OpenAI SDK 原生支持 Pydantic 模型，使用 `client.beta.chat.completions.parse()` 方法自动处理模式转换和响应解析。

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

class Framework(BaseModel):
    name: str
    description: str

class ProgrammingLanguage(BaseModel):
    name: str
    paradigm: str
    year_created: int
    popular_frameworks: Optional[List[Framework]] = None

class LanguageList(BaseModel):
    languages: List[ProgrammingLanguage]

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "列出3种编程语言的详细信息"}
    ],
    response_format=LanguageList
)

# 直接获取解析后的 Pydantic 对象
result = completion.choices[0].message.parsed
print(result.languages[0].name)
```

### 1.5 注意事项

- 使用 JSON 模式时，**必须**在 system 或 user 消息中明确提示模型输出 JSON 格式
- `strict` 模式下，schema 必须满足特定约束：
  - 所有字段需显式列出 `required`
  - 必须设置 `additionalProperties: false`
  - 不支持某些高级 JSON Schema 特性如 `$ref`
- 支持结构化输出的模型：`gpt-4o-mini-2024-07-18` 及之后版本、`gpt-4o-2024-08-06` 及之后版本

---

## 2. Google Gemini Python SDK

### 2.1 概述

Google Gemini 的 Python SDK 提供了强大的结构化输出能力，支持通过 JSON Schema 或 Python 类型定义输出格式。

- **SDK 包名**: `google-generativeai`（旧版）或 `google-genai`（新版）
- **安装命令**: `pip install google-generativeai` 或 `pip install google-genai`

### 2.2 方案一：基础 JSON 模式

通过 `generation_config` 设置 `response_mime_type` 为 `application/json` 启用 JSON 输出。

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json"
    }
)

response = model.generate_content("列出3种水果及其营养价值，以JSON格式返回")
print(response.text)
```

### 2.3 方案二：JSON Schema 定义

使用 `response_schema` 参数配合 JSON Schema 精确控制输出结构。

```python
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

genai.configure(api_key="YOUR_API_KEY")

schema = {
    "type": "object",
    "properties": {
        "fruits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "color": {"type": "string"},
                    "calories_per_100g": {"type": "number"},
                    "vitamins": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "color", "calories_per_100g"]
            }
        }
    },
    "required": ["fruits"]
}

model = genai.GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        response_schema=schema
    )
)

response = model.generate_content("提供5种常见水果的营养信息")
print(response.text)
```

### 2.4 方案三：使用 typing 和 TypedDict

Gemini SDK 支持直接使用 Python 类型注解定义输出结构，更加 Pythonic。

```python
import google.generativeai as genai
from typing import TypedDict, List

class Fruit(TypedDict):
    name: str
    color: str
    calories_per_100g: int
    vitamins: List[str]

class FruitList(TypedDict):
    fruits: List[Fruit]

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": FruitList
    }
)

response = model.generate_content("列出3种热带水果")
print(response.text)
```

### 2.5 方案四：Pydantic 模型（新版 SDK）

新版 `google-genai` SDK 支持直接使用 Pydantic 模型。

```python
from google import genai
from pydantic import BaseModel
from typing import List

class Fruit(BaseModel):
    name: str
    color: str
    taste: str

class FruitBasket(BaseModel):
    fruits: List[Fruit]
    total_count: int

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="列出5种水果",
    config={
        "response_mime_type": "application/json",
        "response_schema": FruitBasket
    }
)

print(response.text)
```

### 2.6 枚举类型支持

Gemini 支持使用 Python Enum 类型约束输出值的范围。

```python
import google.generativeai as genai
from enum import Enum
from typing import TypedDict

class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentResult(TypedDict):
    text: str
    sentiment: Sentiment
    confidence: float

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": SentimentResult
    }
)

response = model.generate_content("分析这句话的情感：'今天天气真好！'")
print(response.text)
```

---

## 3. Anthropic Python SDK

### 3.1 概述

Anthropic 的 Claude API **目前不提供原生的 JSON 模式或结构化输出参数**。实现 JSON 输出需要通过提示词工程或使用第三方工具库。

- **SDK 包名**: `anthropic`
- **安装命令**: `pip install anthropic`

### 3.2 方案一：提示词引导

通过在系统提示或用户消息中明确要求 JSON 格式输出，并使用 XML 标签约束输出。

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="""你是一个数据处理助手。
请始终以 JSON 格式返回结果。
输出应该是有效的 JSON，不要包含任何其他文本。
将 JSON 放在 <json></json> 标签内。""",
    messages=[
        {
            "role": "user",
            "content": "列出3种编程语言，包含名称、创建年份和主要用途"
        }
    ]
)

# 提取 JSON 内容
import re
import json

text = response.content[0].text
json_match = re.search(r'<json>(.*?)</json>', text, re.DOTALL)
if json_match:
    data = json.loads(json_match.group(1))
    print(data)
```

### 3.3 方案二：使用 Tool Use 功能

Anthropic 的 Tool Use（函数调用）功能可以间接实现结构化输出，通过定义工具的输入 schema 来约束输出格式。

```python
from anthropic import Anthropic

client = Anthropic()

# 定义工具作为输出格式的约束
tools = [
    {
        "name": "output_languages",
        "description": "输出编程语言信息的结构化数据",
        "input_schema": {
            "type": "object",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "语言名称"
                            },
                            "year": {
                                "type": "integer",
                                "description": "创建年份"
                            },
                            "paradigm": {
                                "type": "string",
                                "description": "编程范式"
                            }
                        },
                        "required": ["name", "year", "paradigm"]
                    }
                }
            },
            "required": ["languages"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "output_languages"},
    messages=[
        {
            "role": "user",
            "content": "列出3种编程语言的信息"
        }
    ]
)

# 从工具调用中提取结构化数据
for block in response.content:
    if block.type == "tool_use":
        print(block.input)  # 这里是结构化的 JSON 数据
```

### 3.4 方案三：使用 Instructor 库（推荐）

Instructor 是一个流行的第三方库，为多个 LLM API 提供统一的 Pydantic 结构化输出支持。

**安装**: `pip install instructor`

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from typing import List

class Language(BaseModel):
    name: str
    year: int
    paradigm: str
    popular_for: List[str]

class LanguageList(BaseModel):
    languages: List[Language]

# 使用 instructor 包装 Anthropic 客户端
client = instructor.from_anthropic(Anthropic())

result = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "列出3种编程语言的详细信息"}
    ],
    response_model=LanguageList
)

# result 已经是 LanguageList 类型的 Pydantic 对象
for lang in result.languages:
    print(f"{lang.name} ({lang.year}): {lang.paradigm}")
```

### 3.5 注意事项

- Claude 模型对提示词的遵循度很高，通过清晰的指令通常能获得正确格式的 JSON
- Tool Use 方式更可靠，但会产生额外的 token 消耗
- 使用 Instructor 库是目前获得类似 OpenAI 结构化输出体验的最佳方式
- Anthropic 官方可能在未来版本中添加原生 JSON 模式支持

---

## 4. OpenRouter

### 4.1 概述

OpenRouter 是一个 AI 模型聚合平台，提供统一的 API 访问多种模型。它兼容 OpenAI SDK，JSON 输出能力取决于底层模型的支持情况。

- **SDK**: 使用 `openai` 包配合自定义 `base_url`
- **安装命令**: `pip install openai`

### 4.2 基础使用

OpenRouter 完全兼容 OpenAI 的 API 格式，包括 `response_format` 参数。

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY"
)

# JSON Mode（需模型支持）
response = client.chat.completions.create(
    model="openai/gpt-4o",  # OpenRouter 模型格式：provider/model
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "返回 JSON 格式的响应"
        },
        {
            "role": "user",
            "content": "列出3个城市及其人口"
        }
    ]
)

print(response.choices[0].message.content)
```

### 4.3 结构化输出

对于支持结构化输出的模型，可以使用 `json_schema` 格式。

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY"
)

response = client.chat.completions.create(
    model="openai/gpt-4o-2024-08-06",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "cities",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "country": {"type": "string"},
                                "population": {"type": "integer"}
                            },
                            "required": ["name", "country", "population"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["cities"],
                "additionalProperties": False
            }
        }
    },
    messages=[
        {"role": "user", "content": "列出世界上3个最大的城市"}
    ]
)

print(response.choices[0].message.content)
```

### 4.4 模型支持情况

| 模型 | JSON Mode | JSON Schema | Strict 模式 |
|------|-----------|-------------|-------------|
| openai/gpt-4o | ✅ | ✅ | ✅ |
| anthropic/claude-3.5-sonnet | ❌ | ❌ | ❌ |
| google/gemini-pro-1.5 | ✅ | ✅ | ⚠️ 部分 |
| meta-llama/llama-3.1-70b | ✅ | ⚠️ 有限 | ❌ |

---

## 5. Ollama Python SDK

### 5.1 概述

Ollama 是一个本地运行大语言模型的工具，其 Python SDK 提供了完善的 JSON 输出支持，包括原生 JSON 模式和结构化输出功能。

- **SDK 包名**: `ollama`
- **安装命令**: `pip install ollama`

### 5.2 方案一：JSON 模式

通过设置 `format` 参数为 `"json"` 启用 JSON 模式。

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "列出3种水果，以JSON格式返回，包含名称和颜色"
        }
    ],
    format="json"  # 启用 JSON 模式
)

print(response["message"]["content"])
```

### 5.3 方案二：结构化输出（JSON Schema）

Ollama 支持通过 `format` 参数传入 JSON Schema 定义精确的输出结构。

```python
import ollama

schema = {
    "type": "object",
    "properties": {
        "fruits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "color": {"type": "string"},
                    "taste": {"type": "string"}
                },
                "required": ["name", "color", "taste"]
            }
        }
    },
    "required": ["fruits"]
}

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "列出3种热带水果的信息"}
    ],
    format=schema  # 传入 JSON Schema
)

import json
data = json.loads(response["message"]["content"])
print(data)
```

### 5.4 方案三：Pydantic 集成

Ollama SDK 原生支持 Pydantic 模型，可以直接使用 `model_json_schema()` 方法生成 schema。

```python
import ollama
from pydantic import BaseModel
from typing import List

class Fruit(BaseModel):
    name: str
    color: str
    taste: str
    origin_country: str

class FruitBasket(BaseModel):
    fruits: List[Fruit]
    total_count: int

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "列出4种水果的详细信息"}
    ],
    format=FruitBasket.model_json_schema()  # 使用 Pydantic schema
)

import json
data = json.loads(response["message"]["content"])
result = FruitBasket.model_validate(data)
print(f"共 {result.total_count} 种水果")
for fruit in result.fruits:
    print(f"  - {fruit.name}: {fruit.color}, {fruit.taste}")
```

### 5.5 OpenAI 兼容模式

Ollama 提供 OpenAI 兼容的 API 端点，可以使用 OpenAI SDK 访问。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama 不需要真实 API key
)

response = client.chat.completions.create(
    model="llama3.2",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "你是一个助手，始终以 JSON 格式返回结果"
        },
        {
            "role": "user",
            "content": "列出3个国家及其首都"
        }
    ]
)

print(response.choices[0].message.content)
```

### 5.6 注意事项

- 不同模型对 JSON 格式的支持程度不同，推荐使用较新的模型如 `llama3.2`、`qwen2.5`、`mistral` 等
- 结构化输出的质量受模型能力影响，小参数模型可能无法完全遵循复杂 schema
- 本地运行时注意模型大小与系统资源的匹配

---

## 6. 总结与建议

### 6.1 选型建议

| 需求场景 | 推荐方案 |
|---------|---------|
| 需要最可靠的结构化输出 | **OpenAI** - Structured Outputs 在 strict 模式下保证 100% 模式匹配 |
| 使用 Google 生态 | **Gemini** - 优秀的 JSON 支持，与 Python 类型系统集成良好 |
| 使用 Claude | **Instructor 库** 或 **Tool Use** - 获得类似 OpenAI 的结构化输出体验 |
| 需要模型灵活性 | **OpenRouter** - 在不同模型间切换，保持相同的 API 调用方式 |
| 需要本地部署 | **Ollama** - 完善的 JSON 支持，可使用 OpenAI 兼容接口 |

### 6.2 最佳实践

1. **始终在提示词中明确说明**需要 JSON 输出
2. **使用 Pydantic 定义数据模型**，便于验证和类型检查
3. 对于关键业务场景，**添加输出验证逻辑**
4. 考虑使用 **Instructor** 等库统一不同平台的结构化输出体验
5. **测试不同模型**在特定 schema 下的表现，选择最适合的模型

### 6.3 未来展望

结构化输出正在成为 LLM API 的标准功能，预计 Anthropic 等目前不支持原生 JSON 模式的平台将在未来版本中添加此功能。各平台的 Pydantic 集成也在不断完善，使得类型安全的 LLM 应用开发变得更加便捷。

---

## 附录：SDK 安装命令汇总

| 平台 | 安装命令 |
|------|---------|
| OpenAI | `pip install openai` |
| Google Gemini | `pip install google-generativeai` 或 `pip install google-genai` |
| Anthropic | `pip install anthropic` |
| OpenRouter | `pip install openai`（使用 OpenAI SDK） |
| Ollama | `pip install ollama` |
| Instructor | `pip install instructor` |
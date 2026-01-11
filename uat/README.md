# UAT: Image Analysis JSON Mode Tests

用户验收测试脚本，用于测试各 LLM Provider 的图像分析和 JSON 输出模式。

## 快速开始

```bash
# 设置 API Keys（可选，未设置的将跳过）
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export OPENROUTER_API_KEY=your_key

# 运行所有测试
uv run python uat/run_all.py

# 或指定测试图片
uv run python uat/run_all.py path/to/image.png
```

## 单独运行

```bash
# Anthropic (Tool Use 实现结构化输出)
uv run python uat/test_anthropic_image.py

# OpenAI (支持原生 JSON mode)
uv run python uat/test_openai_image.py

# Google Gemini (支持原生 JSON mode)
uv run python uat/test_gemini_image.py

# Ollama 本地 (支持原生 JSON mode)
ollama pull llama3.2-vision  # 先拉取模型
uv run python uat/test_ollama_image.py

# OpenRouter (支持原生 JSON mode，取决于底层模型)
uv run python uat/test_openrouter_image.py
```

## JSON Mode 支持情况

| Provider | JSON Mode | 实现方式 |
|----------|-----------|----------|
| OpenAI | ✅ 原生支持 | `response_format={"type": "json_object"}` |
| Gemini | ✅ 原生支持 | `response_mime_type="application/json"` |
| Ollama | ✅ 原生支持 | `format="json"` |
| OpenRouter | ✅ 原生支持 | 取决于底层模型 |
| Anthropic | ✅ Tool Use | `tools` + `tool_choice` 强制结构化输出 |

## Anthropic Tool Use 说明

Anthropic 不支持原生 JSON Mode，但可以通过 **Tool Use** 实现可靠的结构化 JSON 输出：

```python
import anthropic

client = anthropic.Anthropic()

# 定义工具 schema
tool = {
    "name": "output_image_analysis",
    "description": "Output structured image analysis results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "alt_text": {"type": "string"},
            "image_type": {"type": "string"},
            # ...
        },
        "required": ["alt_text", "image_type"]
    }
}

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    tools=[tool],
    tool_choice={"type": "tool", "name": "output_image_analysis"},  # 强制使用工具
    messages=[...]
)

# 从 tool_use block 提取结构化数据
for block in response.content:
    if block.type == "tool_use":
        data = block.input  # 已经是 dict，符合 schema
```

### 优势

| 特性 | Tool Use | Prompt 指令 |
|------|----------|-------------|
| Schema 验证 | ✅ 严格遵循 | ❌ 不保证 |
| 类型安全 | ✅ 是 | ❌ 否 |
| 可靠性 | 高 | 中等 |
| 代码块包裹 | ❌ 不会 | ✅ 经常出现 |

## 预期结果

### 支持 JSON Mode 的 Provider（OpenAI/Gemini/Ollama/OpenRouter）
- 输出直接是有效 JSON，无需正则提取
- 不会出现 ` ```json ``` ` 代码块包裹
- `json.loads()` 直接解析成功

### Anthropic（Tool Use 模式）
- 通过 `tool_use` block 返回结构化数据
- `block.input` 已经是 Python dict
- 符合定义的 JSON Schema

### Anthropic（Prompt 模式 - 不推荐）
- 可能出现代码块包裹
- 需要正则提取 JSON
- 依赖 `_parse_response` 中的容错逻辑

## 故障排查

1. **API Key 未设置**: 检查环境变量
2. **Ollama 连接失败**: 确保 `ollama serve` 正在运行
3. **模型不存在**: 运行 `ollama pull <model>` 拉取模型
4. **JSON 解析失败**: 检查输出是否被代码块包裹
5. **Anthropic Tool Use 失败**: 确保 `tool_choice` 配置正确

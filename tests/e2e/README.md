# UAT: User Acceptance Tests

用户验收测试脚本，包含：
- **弹性特性测试**: AIMD 限速、DLQ、背压队列、混沌测试
- **图像分析 JSON Mode 测试**: 各 LLM Provider 的图像分析和 JSON 输出模式

---

## 弹性特性测试 (cc branch)

```bash
# 运行所有弹性测试
uv run python uat/run_resilience.py

# 单独运行各项测试
uv run python uat/test_aimd_limiter.py      # AIMD 自适应限速器
uv run python uat/test_dead_letter_queue.py # 死信队列
uv run python uat/test_bounded_queue.py     # 背压队列
uv run python uat/test_chaos_provider.py    # 混沌模拟提供商
```

### 测试内容

| 测试文件 | 功能 | 说明 |
|----------|------|------|
| `test_aimd_limiter.py` | AIMD 限速器 | 自适应并发控制，基于 429 响应调整 |
| `test_dead_letter_queue.py` | 死信队列 | 失败追踪、重试计数、永久失败隔离 |
| `test_bounded_queue.py` | 背压队列 | 防止 OOM，生产者阻塞机制 |
| `test_chaos_provider.py` | 混沌提供商 | 模拟延迟、429、500、超时等故障 |

### 前置条件

无需额外配置，仅需安装依赖：

```bash
uv sync --all-extras
```

---

### 真实场景测试

使用 CLI 命令验证实际转换功能。

```bash
# 方式 1: 通过 uv run 运行（推荐，无需安装）
uv run markit --help

# 方式 2: 安装到环境后直接运行
uv pip install -e .
markit --help
```

以下示例使用 `uv run markit`，如已安装可省略 `uv run` 前缀。

#### 1. Provider 连接测试

```bash
# 测试已配置的 LLM Provider 连接
uv run markit provider test

# 列出已配置的 Provider
uv run markit provider list

# 列出已配置的模型
uv run markit model list
```

#### 2. 单文件转换

```bash
# 基础转换（不使用 LLM）
uv run markit convert document.docx
uv run markit convert presentation.pptx
uv run markit convert spreadsheet.xlsx
uv run markit convert document.pdf

# 指定输出目录
uv run markit convert document.pdf -o ./output

# 使用 LLM 增强（清理格式、添加 frontmatter、生成摘要）
uv run markit convert document.pdf --llm

# 使用 LLM 分析图片（生成 alt text）
uv run markit convert document.pdf --analyze-image

# 同时生成图片描述 markdown 文件
uv run markit convert document.pdf --analyze-image-with-md
```

#### 3. 批量转换

```bash
# 批量转换目录
uv run markit batch ./docs -o ./output

# 递归处理子目录
uv run markit batch ./docs -o ./output -r

# 快速模式（跳过验证，最少重试）
uv run markit batch ./docs -o ./output --fast

# 干运行（显示计划但不执行）
uv run markit batch ./docs -o ./output --dry-run
```

#### 4. 断点续传测试

```bash
# 首次运行（中途 Ctrl+C 中断）
uv run markit batch ./docs -o ./output -r

# 检查状态文件
cat .markit-state.json

# 恢复运行（跳过已完成文件）
uv run markit batch ./docs -o ./output -r --resume
```

#### 5. 高负载测试

```bash
# 准备测试目录（100+ 文件）
mkdir -p test_load
for i in {1..100}; do echo "Document $i" > test_load/doc_$i.txt; done

# 运行批量转换
uv run markit batch ./test_load -o ./output_load

# 检查结果
ls -la ./output_load | wc -l
```

#### 6. LLM 增强测试

```bash
# 指定 Provider
uv run markit convert doc.pdf --llm --llm-provider openai

# 指定模型
uv run markit convert doc.pdf --llm --llm-model gpt-4o

# 详细日志
uv run markit convert doc.pdf --llm -v
```

### 测试矩阵

| 场景 | 命令 | 验证点 |
|------|------|--------|
| 基础转换 | `uv run markit convert doc.pdf` | 输出 .md 文件，图片提取到 assets/ |
| LLM 增强 | `uv run markit convert doc.pdf --llm` | Frontmatter、摘要、格式清理 |
| 图像分析 | `uv run markit convert doc.pdf --analyze-image` | 图片有 alt text |
| 批量处理 | `uv run markit batch ./docs -o ./out -r` | 所有文件转换，保持目录结构 |
| 断点续传 | `--resume` | 不重复处理已完成文件 |
| Provider 回退 | 主 Provider 不可用时 | 自动切换备用 Provider |

---

## 图像分析 JSON Mode 测试

### 快速开始

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

### 单独运行

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

### JSON Mode 支持情况

| Provider | JSON Mode | 实现方式 |
|----------|-----------|----------|
| OpenAI | ✅ 原生支持 | `response_format={"type": "json_object"}` |
| Gemini | ✅ 原生支持 | `response_mime_type="application/json"` |
| Ollama | ✅ 原生支持 | `format="json"` |
| OpenRouter | ✅ 原生支持 | 取决于底层模型 |
| Anthropic | ✅ Tool Use | `tools` + `tool_choice` 强制结构化输出 |

### Anthropic Tool Use 说明

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

#### 优势

| 特性 | Tool Use | Prompt 指令 |
|------|----------|-------------|
| Schema 验证 | ✅ 严格遵循 | ❌ 不保证 |
| 类型安全 | ✅ 是 | ❌ 否 |
| 可靠性 | 高 | 中等 |
| 代码块包裹 | ❌ 不会 | ✅ 经常出现 |

### 预期结果

#### 支持 JSON Mode 的 Provider（OpenAI/Gemini/Ollama/OpenRouter）
- 输出直接是有效 JSON，无需正则提取
- 不会出现 ` ```json ``` ` 代码块包裹
- `json.loads()` 直接解析成功

#### Anthropic（Tool Use 模式）
- 通过 `tool_use` block 返回结构化数据
- `block.input` 已经是 Python dict
- 符合定义的 JSON Schema

#### Anthropic（Prompt 模式 - 不推荐）
- 可能出现代码块包裹
- 需要正则提取 JSON
- 依赖 `_parse_response` 中的容错逻辑

### 故障排查

1. **API Key 未设置**: 检查环境变量
2. **Ollama 连接失败**: 确保 `ollama serve` 正在运行
3. **模型不存在**: 运行 `ollama pull <model>` 拉取模型
4. **JSON 解析失败**: 检查输出是否被代码块包裹
5. **Anthropic Tool Use 失败**: 确保 `tool_choice` 配置正确

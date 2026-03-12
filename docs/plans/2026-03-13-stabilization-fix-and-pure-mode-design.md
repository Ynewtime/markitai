# Stabilization Fix & Pure Mode Design

## Background

日志分析发现 `sample.pptx` 在 vision 增强处理后产生 3 条 WARNING：

1. `_stabilize_paged_markdown` 对 slide 8 被调用两次，产生重复告警（Problem A）
2. LLM (`gemini-3.1-flash-lite-preview`) 对 slide 8 产生结构漂移，占位符未被正确处理（Problem B）
3. 用户提出引入**纯净模式**，让 markitai 在 LLM 阶段变成透明管道（Problem C）

## Problem A: 重复 WARNING 去重

### 根因

`_stabilize_paged_markdown` 在 vision 流程中被调用两次：

1. **第一次**：`enhance_document_with_vision`（document.py:1092）— LLM 返回后立即稳定化
2. **第二次**：`stabilize_written_llm_output`（core.py:~527）— 写 `.llm.md` 后读回校验

两次使用相同的 baseline（原始 `.md`），第二次重复检测到 slide 8 异常并再次回退，产生重复 WARNING。

### 方案：A1 — ConversionContext 标记去重

在 `ConversionContext` 上新增字段 `paged_stabilized: bool = False`。第一次稳定化后设为 `True`，`stabilize_written_llm_output` 检查此标记，已稳定化则跳过。

### 改动点

| 文件 | 改动 |
|------|------|
| `workflow/core.py` ConversionContext | 新增 `paged_stabilized: bool = False` 字段 |
| `workflow/single.py` enhance_with_vision | 第一次稳定化后设 `ctx.paged_stabilized = True`（需透传 ctx） |
| `workflow/core.py` stabilize_written_llm_output 调用处 | 检查 `ctx.paged_stabilized`，为 True 则跳过 |

### 注意事项

- `SingleFileWorkflow.enhance_with_vision` 当前不持有 `ctx`，需要确认调用链能否透传，或者改为在 `core.py` 的调用处设置标记
- 标准 LLM 路径（`process_with_standard_llm`）中 `process_document_with_llm` 内部也有一次稳定化 + 外部 `stabilize_written_llm_output`，同样需要去重

## Problem B: Prompt 优化——减少 LLM 结构漂移

### 根因

8 张 slide 的截图 + 文本在一个 combined call 中发给轻量模型，模型在处理最后几张 slide 时质量下降——占位符未替换、内容膨胀。LLM 的注意力对长上下文尾部衰减。

### 方案：B2 — 在 prompt 尾部追加关键规则提醒

利用 LLM 的 primacy/recency effect，在 user prompt 末尾和图片序列末尾重申占位符保护规则。

### 改动点

| 文件 | 改动 |
|------|------|
| `prompts/document_vision_user.md` | 在 `{content}` 之后追加 REMINDER 段落 |
| `llm/document.py` enhance_document_with_vision | 在 content_parts 末尾（所有图片之后）追加 text REMINDER |

### Prompt 改动内容

**document_vision_user.md**（在末尾追加）:

```
---

REMINDER: All `__MARKITAI_*__` placeholders must appear in your output exactly as in the input. Do not remove, modify, or merge any placeholder. Do not wrap output in a code block.
```

**document.py content_parts 末尾**:

```python
content_parts.append({
    "type": "text",
    "text": "\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders exactly as-is. "
            "Do not remove or modify any placeholder. "
            "Output every page/slide — do not skip the last pages.",
})
```

## Problem C: 纯净模式（Pure Mode）

### 需求

引入 `--pure` 开关，让 markitai 在 LLM 阶段变成透明管道：

- 原始 MD 原封不动发给 LLM
- LLM 返回什么就写什么到 `.llm.md`
- markitai 自身不做任何内容清洗/加工/过滤/处理
- 默认关闭，用户通过 CLI `--pure` 或环境变量 `MARKITAI_PURE=1` 开启

### 纯净模式跳过清单

| 环节 | 正常模式 | 纯净模式 |
|------|----------|----------|
| 格式转换（pymupdf4llm、markitdown 等） | ✅ | ✅ |
| 写 `.md` 基础文件 | ✅ | ✅ |
| ContentProtection（占位符替换） | ✅ | ❌ |
| 智能截断 | ✅ | ❌ |
| Frontmatter 生成 | ✅ | ❌ |
| Vision 增强（截图+LLM） | ✅ | ❌ |
| 图片 alt text / desc 分析 | ✅ | ❌ |
| `_stabilize_paged_markdown` | ✅ | ❌ |
| `fix_malformed_image_refs` | ✅ | ❌ |
| `stabilize_written_llm_output` | ✅ | ❌ |
| LLM 清洗（原始 MD 直送） | ✅ | ✅ |

### 实现路径

#### 配置层

**config.py `LLMConfig`**:

```python
class LLMConfig(BaseModel):
    enabled: bool = False
    pure: bool = False  # Pure mode: raw MD → LLM → output
    model_list: list[ModelConfig] = Field(default_factory=list)
    ...
```

**CLI main.py**:

```python
@click.option("--pure", is_flag=True, help="Pure mode: raw MD → LLM → output, no markitai processing.")
```

**环境变量**: `MARKITAI_PURE=1`，在 CLI 参数解析中映射到 `config.llm.pure`

#### 流水线层

**新增顶层函数** `process_with_pure_llm`（core.py）:

```python
async def process_with_pure_llm(ctx: ConversionContext) -> ConversionStepResult:
    """Pure mode: send raw markdown to LLM, write response as-is."""
    # 调用 workflow.process_document_pure
    # 不做任何后处理
```

**新增方法** `process_document_pure`（single.py SingleFileWorkflow）:

```python
async def process_document_pure(
    self,
    markdown: str,
    source: str,
    output_file: Path,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Pure mode: send raw markdown to LLM, write response as-is."""
    cleaned = await self.processor.clean_document_pure(markdown, source)
    llm_output = output_file.with_suffix(".llm.md")
    atomic_write_text(llm_output, cleaned)
    # ...
```

**新增方法** `clean_document_pure`（document.py LLMProcessor / DocumentMixin）:

```python
async def clean_document_pure(self, markdown: str, source: str) -> str:
    """Pure cleaning: no protection, no stabilization, no truncation."""
    system_prompt = self._prompt_manager.get_prompt("cleaner_system")
    user_prompt = self._prompt_manager.get_prompt("cleaner_user", content=markdown)
    response = await self._call_llm(
        model="default",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        context=source,
    )
    return response.content
```

**core.py `convert_document_core` Step 7 分支**:

```python
if ctx.config.llm.enabled and ctx.conversion_result is not None:
    if ctx.config.llm.pure:
        result = await process_with_pure_llm(ctx)
        if not result.success:
            return result
    else:
        # 现有逻辑不变...
```

### 改动范围汇总（A + B + C）

| 文件 | 改动内容 |
|------|----------|
| `config.py` | `LLMConfig.pure: bool = False` |
| `constants.py` | `ENV_PURE = "MARKITAI_PURE"`（如需要） |
| `cli/main.py` | `--pure` CLI flag + env 映射到 config |
| `workflow/core.py` | A: `paged_stabilized` 字段 + 检查；C: pure 分支 + `process_with_pure_llm` |
| `workflow/single.py` | A: 设 `ctx.paged_stabilized`；C: `process_document_pure` 方法 |
| `llm/document.py` | B: content_parts 末尾 REMINDER；C: `clean_document_pure` 方法 |
| `prompts/document_vision_user.md` | B: 尾部 REMINDER |

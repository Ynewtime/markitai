# Stabilization Fix & Pure Mode Design

## Background

日志分析发现 `sample.pptx` 在 vision 增强处理后产生 3 条 WARNING：

1. `_stabilize_paged_markdown` 对 slide 8 被调用多次，产生重复告警（Problem A）
2. LLM (`gemini-3.1-flash-lite-preview`) 对 slide 8 产生结构漂移，占位符未被正确处理（Problem B）
3. 用户提出引入**纯净模式**，让 markitai 在 LLM 阶段变成透明管道（Problem C）

## Problem A: 重复 WARNING 去重

### 根因

Vision 路径中 `_stabilize_paged_markdown` 实际被调用**三次**：

1. **第一次**：`_enhance_with_frontmatter`（document.py:1449）或 `enhance_document_with_vision`（document.py:1092）— LLM 返回后立即稳定化
2. **第二次**：`enhance_with_vision`（single.py:401）— 调用 `maybe_stabilize_markdown`，对已稳定化的输出再次稳定化
3. **第三次**：`stabilize_written_llm_output`（core.py:940）— 写 `.llm.md` 后读回校验

三次都使用相同的 baseline（原始 `.md`），后续调用重复检测到 slide 8 异常并再次回退，产生重复 WARNING。

标准 LLM 路径同样存在类似问题：`process_document_with_llm`（single.py:153）内部一次稳定化 + 外部 `stabilize_written_llm_output`（core.py:761）再来一次。

### 方案：A1 — ConversionContext 标记去重

在 `ConversionContext` 上新增字段 `paged_stabilized: bool = False`。底层稳定化（document.py 内部）完成后，在 workflow 层设标记。后续 `maybe_stabilize_markdown` 和 `stabilize_written_llm_output` 检查此标记，为 True 则跳过。

### 改动点

| 文件 | 改动 |
|------|------|
| `workflow/core.py` ConversionContext | 新增 `paged_stabilized: bool = False` 字段 |
| `workflow/single.py` enhance_with_vision | 移除冗余的 `maybe_stabilize_markdown` 调用（single.py:401-402），因为 `enhance_document_complete` 内部已经做了稳定化。设 `ctx.paged_stabilized = True`（需透传或在 core.py 调用处设置） |
| `workflow/single.py` process_document_with_llm | 同理，`process_document`（document.py:1685）内部已经稳定化，移除 single.py:153 的冗余调用 |
| `workflow/core.py` stabilize_written_llm_output 调用处 | 检查 `ctx.paged_stabilized`，为 True 则跳过 |

### 实现策略

由于 `SingleFileWorkflow` 当前不持有 `ctx`，有两个选择：

- **选项 1**（推荐）：不透传 `ctx`，而是在 `core.py` 的调用处设标记。vision 路径和标准路径的底层方法（document.py）已经内部做了稳定化，只需在 core.py 调用完 `process_with_vision_llm` / `process_with_standard_llm` 后设 `ctx.paged_stabilized = True`，让后续 `stabilize_written_llm_output` 跳过。同时移除 single.py 中的冗余 `maybe_stabilize_markdown` 调用。
- **选项 2**：给 `SingleFileWorkflow` 方法添加 `ctx` 参数——改动面更大，不推荐。

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
| `llm/document.py` _enhance_with_frontmatter | 同样在 content_parts 末尾追加 text REMINDER（此方法是 single-batch ≤10 pages 的主路径） |

### Prompt 改动内容

**document_vision_user.md**（在末尾追加）:

```
---

REMINDER: All `__MARKITAI_*__` placeholders must appear in your output exactly as in the input. Do not remove, modify, or merge any placeholder. Do not wrap output in a code block.
```

**document.py content_parts 末尾**（`enhance_document_with_vision` 和 `_enhance_with_frontmatter` 两处）:

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
| `_remove_uncommented_screenshots` | ✅ | ❌ |
| `_strip_leaked_markdown_boundaries` | ✅ | ❌ |
| `_restore_images_or_fallback` | ✅ | ❌ |
| `format_llm_output`（frontmatter 拼装） | ✅ | ❌ |
| 注释页面图片链接追加 | ✅ | ❌ |
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

**隐式行为**：`--pure` 自动隐含 `--llm`（类似 `--screenshot-only` 隐含 `--screenshot`）。若用户同时指定 `--pure --alt` 或 `--pure --desc`，静默忽略（pure 模式跳过图片分析）。

#### 流水线层

**新增顶层函数** `process_with_pure_llm`（core.py）:

```python
async def process_with_pure_llm(ctx: ConversionContext) -> ConversionStepResult:
    """Pure mode: send raw markdown to LLM, write response as-is.

    No ContentProtection, stabilization, frontmatter, vision, or alt text.
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(ctx.config, processor=processor)

    ctx.conversion_result.markdown, doc_cost, doc_usage = (
        await workflow.process_document_pure(
            ctx.conversion_result.markdown,
            ctx.input_path.name,
            ctx.output_file,
        )
    )
    ctx.llm_cost += doc_cost
    merge_llm_usage(ctx.llm_usage, doc_usage)

    return ConversionStepResult(success=True)
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
    try:
        cleaned = await self.processor.clean_document_pure(markdown, source)
        llm_output = output_file.with_suffix(".llm.md")
        atomic_write_text(llm_output, cleaned)
        logger.info(f"Written LLM version (pure): {llm_output}")
        cost = self.processor.get_context_cost(source)
        usage = self.processor.get_context_usage(source)
        return markdown, cost, usage
    except Exception as e:
        logger.error(f"Pure LLM processing failed: {format_error_message(e)}")
        return markdown, 0.0, {}
```

**新增方法** `clean_document_pure`（document.py DocumentMixin）:

```python
async def clean_document_pure(self, markdown: str, source: str) -> str:
    """Pure cleaning: no protection, no stabilization, no truncation.

    Uses the standard cleaner prompt — the LLM decides what to clean.
    markitai does not process the input or output in any way.
    """
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

注意：使用现有 `cleaner_system` / `cleaner_user` prompt，因为用户期望的是 LLM 做清洗、markitai 不做加工。prompt 中关于占位符的规则在纯净模式下不会生效（因为没有占位符替换），这是符合预期的。

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

#### 缓存行为

纯净模式下 `clean_document_pure` 通过 `_call_llm` 调用，该方法内部已经有 LLM 级别的缓存支持（PersistentCache），无需额外处理。

### 与其他选项的交互

| 组合 | 行为 |
|------|------|
| `--pure`（无 `--llm`） | 自动隐含 `--llm` |
| `--pure --alt` / `--pure --desc` | 静默忽略 alt/desc，只做纯文本清洗 |
| `--pure --screenshot` | 截图仍然生成（格式转换阶段），但不用于 vision 增强 |
| `--pure --screenshot-only` | `--screenshot-only` 优先，`--pure` 被忽略（互斥语义，screenshot-only 有自己的 LLM 路径） |
| `--preset rich --pure` | preset 设置的 alt/desc/screenshot 被 pure 覆盖跳过 |

### 改动范围汇总（A + B + C）

| 文件 | 改动内容 |
|------|----------|
| `config.py` | `LLMConfig.pure: bool = False` |
| `cli/main.py` | `--pure` CLI flag + env `MARKITAI_PURE` 映射 + `--pure` 隐含 `--llm` |
| `workflow/core.py` | A: `paged_stabilized` 字段 + `stabilize_written_llm_output` 跳过检查；C: pure 分支 + `process_with_pure_llm` |
| `workflow/single.py` | A: 移除冗余 `maybe_stabilize_markdown` 调用；C: `process_document_pure` 方法 |
| `llm/document.py` | B: `enhance_document_with_vision` 和 `_enhance_with_frontmatter` 的 content_parts 末尾 REMINDER；C: `clean_document_pure` 方法 |
| `prompts/document_vision_user.md` | B: 尾部 REMINDER |

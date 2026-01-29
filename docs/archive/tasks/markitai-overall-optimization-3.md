# Markitai 6 项问题修复计划

> 修复 PPT 页码、大写扩展名、性能优化、中文列头、Vision 负载均衡、state/report 合并

---

## 问题 1: PPT 页码信息丢失

**根因**: `cleaner.md` 第14行删除了 `<!-- Slide number: X -->`

**修复**:
```diff
# prompts/cleaner.md 第14行
- - 删除 HTML 注释（如 `<!-- Slide number: X -->`）
+ - 保留幻灯片标记（如 `<!-- Slide number: X -->`），删除其他 HTML 注释
```

用户要求格式: `<!-- Slide 1 -->` (简化)

**额外改动**: office.py 中 MarkItDown 输出的 `<!-- Slide number: X -->` 需转换为 `<!-- Slide X -->`

---

## 问题 2: 大写 JPG 扩展名未匹配

**根因**: Linux 下 glob 大小写敏感，`candy.JPG` 不匹配 `*.jpg`

**修复位置**: `batch.py:293-307` discover_files 方法

```python
# 当前: input_path.glob(f"*{ext}")
# 修复: 同时搜索大小写变体
for ext in extensions:
    # 搜索原始扩展名 + 大写变体
    patterns = [f"*{ext}", f"*{ext.upper()}"]
    for pattern in patterns:
        candidates = list(input_path.glob(pattern))
```

或使用 `fnmatch` + `os.listdir` 实现大小写不敏感匹配。

---

## 问题 3: 深层性能优化

### 3.1 合并 caption + description 为单次调用

**当前**: `llm.py:779-835` 两次 vision API 调用
**修复**: 使用 Instructor 结构化输出，单次调用返回所有字段

```python
from instructor import from_litellm
from pydantic import BaseModel

class ImageAnalysisResult(BaseModel):
    alt: str           # 简短 alt 文本 (10-15词)
    description: str   # 详细 Markdown 描述
    extracted_text: str  # OCR 识别文本

# 修改 _analyze_with_two_calls → _analyze_with_combined_prompt
async def _analyze_with_combined_prompt(self, image_path: Path, model: str) -> ImageAnalysisResult:
    client = from_litellm(self.runtime.get_client())
    return await client.chat.completions.create(
        model=model,
        response_model=ImageAnalysisResult,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/..."}},
                {"type": "text", "text": image_analysis_prompt}
            ]
        }]
    )
```

**预期收益**: Vision API 调用减少 50%

### 3.2 合并 Vision增强 + Doc LLM

**当前流程** (OCR/PPTX模式):
```
1. enhance_document_with_vision (Vision模型) → enhanced_content
2. process_with_llm (Text模型) → cleaner + frontmatter
```

**冗余分析**:
- Vision增强后的内容再被 cleaner 处理一遍
- 两次 LLM 调用，第二次本质上是"清理Vision输出"

**修复**: 合并为单次 Vision 调用

```python
class EnhancedDocumentResult(BaseModel):
    markdown: str      # 已清理的 markdown
    title: str
    description: str
    tags: list[str]

# 修改 enhance_document_with_vision
# 直接输出: 清理后的 markdown + frontmatter
# 删除后续的 process_with_llm 调用 (对有图片的文档)
```

**文档分流**:
- 有图片 (OCR/PPTX): Vision增强 → 直接输出 (跳过 cleaner)
- 无图片: cleaner + frontmatter (不变)

**预期收益**: 有图片的文档 LLM 调用减少 50%

### 3.3 合并 cleaner + frontmatter

**当前**: `llm.py:1007-1044` 并行但发送两次完整内容
**修复**: 合并为单次调用 + Instructor 结构化输出

```python
class DocumentProcessResult(BaseModel):
    cleaned_markdown: str
    frontmatter: Frontmatter

# 单次调用，返回结构化结果
```

**预期收益**: Token 消耗减少 40%, 调用次数减少 50%

### 3.4 内容哈希缓存

**新增**: 基于 (prompt_hash, content_hash) 的请求缓存
- 避免重复处理相同内容
- TTL 5 分钟

---

## 问题 4: 中文列头问题

**根因**: `cleaner.md` 第45-46行硬编码了中文/英文:
```
- 若第一列是纯数字行号且无列头，补充列头为 "#" 或 "No."（英文表格）/ "序号"（中文表格）
```

**修复**: 优化规则，使用文档正文语言而非硬编码

```diff
【表格规范】
-- 若第一列是纯数字行号且无列头，补充列头为 "#" 或 "No."（英文表格）/ "序号"（中文表格）
++ 若第一列是纯数字行号且无列头，补充列头时使用与文档正文一致的语言
```

---

## 问题 5: Vision 模型负载均衡

**分析**: 配置正确，但 `simple-shuffle` 策略按权重随机选择
- gemini-2.5-flash-lite weight=7 (主导)
- 其他 vision 模型 weight=3-5

**验证方法**: 检查 `assets.desc.json` 中 model 字段分布

**可能原因**:
1. 批量小（只有 4 张图），随机采样偏差大
2. 其他模型失败后 fallback 到主模型

**修复**: 增加日志，验证 Router 行为

---

## 问题 6: state 和 report 合并

**当前**:
- `.markitai-state-{dir}-{hash}.json` - 用于 resume (根目录)
- `reports/{name}.report.json` - 最终报告

**字段重叠**: ~50%

**修复方案**: 保留 `reports/{name}.report.json`，将 state 合并进去，删除根目录 state 文件

### 6.1 统一文件结构

```json
{
  "version": "1.0",
  "started_at": "...",
  "updated_at": "...",   // 每次保存更新
  "generated_at": "...", // 完成时设置

  "input_dir": "...",
  "output_dir": "...",
  "options": {...},

  "files": {
    "path": {
      "status": "pending|completed|failed",
      "output": "...",
      "error": "...",
      "started_at": "...",
      "completed_at": "...",
      "duration_seconds": 0,
      "images_extracted": 0,
      "llm_cost_usd": 0,
      "llm_usage": {...}
    }
  },

  "summary": {...},      // 完成时计算
  "llm_usage": {...}     // 聚合统计
}
```

### 6.2 代码改动

| 文件 | 改动 |
|------|------|
| `batch.py` | 合并 save_state + generate_report；load_state 改为 load_report |
| `cli.py` | 删除单独的 report 保存调用 |
| `config.py` | 删除 `state_file` 配置项 |

### 6.3 文件位置

```
output/
├── reports/
│   └── {name}.report.json  # 合并后的唯一文件，支持 --resume
├── file.md
└── assets/
```

> 删除根目录的 `.markitai-state-*.json` 文件

---

## 实施顺序

1. **问题 2**: 大写扩展名 (简单)
2. **问题 4**: 中文列头 (简单)
3. **问题 1**: PPT 页码 (简单)
4. **问题 5**: Vision 负载均衡验证 (调查)
5. **问题 6**: state/report 合并 (中等)
6. **问题 3**: 性能优化 (复杂)
   - 3.1 合并 caption+description
   - 3.2 合并 Vision+Cleaner
   - 3.3 合并 cleaner+frontmatter
   - 3.4 内容缓存

---

## 验证

```bash
# 清理输出
rm -rf ./output

# 运行测试
uv run markitai packages/markitai/tests/fixtures --preset rich -o ./output --verbose

# 验证点:
# 1. candy.JPG 被处理
# 2. PPT 输出包含 <!-- Slide X -->
# 3. 表格无中文列头
# 4. assets.desc.json 有多个 model
# 5. 只生成 .report.json, 无 state 文件
# 6. --resume 正常工作

# 单元测试
uv run pytest tests/unit -n auto
```

---

## 关键文件

| 文件 | 问题 |
|------|------|
| `prompts/cleaner.md` | #1 页码, #4 列头 |
| `batch.py` | #2 扩展名, #6 合并 |
| `llm.py` | #3 性能, #5 负载均衡 |
| `cli.py` | #3 并行化, #6 report |
| `office.py` | #1 页码格式化 |

# Markitai 性能问题分析与优化方案

> 日期: 2026-01-16

## 问题总结

### 日志 1 (22-05-39) - 性能问题

**超时统计**：

| 文件 | 处理时间 | 超时次数 | 问题 |
|------|----------|----------|------|
| file-sample_100kB.doc | 410.19s | 3 次 408 | 等待 ~6 分钟 |
| file_example_PPT_250kB.ppt | 147.32s | 1 次 408 | - |
| file-example_PDF_500_kB.pdf | 128.28s | 0 | Vision 86s |
| Free_Test_Data_500KB_PPTX.pptx | 113.27s | 1 次 500 | Vision 54s |
| file_example_XLSX_100.xlsx | 91.54s | 0 | - |
| file_example_XLS_100.xls | 78.67s | 0 | - |

**根因**：
1. `timeout: 120s` + 指数退避 (1/2/4/8s...) + 最多 4 次重试 = 单文件最长 ~500s
2. 免费模型 (mimo-v2-flash, glm-4.7) 不稳定，频繁超时
3. Vision 处理一次性发送所有页面图片，token 量大

### 日志 2 (21-20-58) - Error 问题

1. **ContextWindowExceededError**：
   - DeepSeek 限制 131072 tokens
   - 请求 387347 tokens（PPTX 页面图片 base64 编码后很大）
   - 没有配置 `context_window_fallbacks`

2. **完全失败**：4 次超时后失败，无降级处理

### base64 图片覆盖 Bug

**Bug 位置**：`cli.py:1311-1314`

```python
# Step 1: Write .md file with original extracted text + basic frontmatter
base_md_content = _add_basic_frontmatter(extracted_text, file_path.name)  # BUG: 使用未替换的 extracted_text
base_md_content += commented_images_str
output_file.write_text(base_md_content, encoding="utf-8")  # 覆盖了之前替换过的内容！
```

**流程问题**：
1. Line 1270-1273: 替换 `result.markdown` 中的 base64 → 正确
2. Line 1277: 写入替换后的内容 → 正确
3. Line 1311-1315: **使用未替换的 `extracted_text` 重新写入，覆盖了正确内容** → Bug!

---

## 当前配置分析

```json
"router_settings": {
  "num_retries": 2,      // Router 层重试
  "timeout": 120         // 120 秒超时
}
```

代码中：`DEFAULT_MAX_RETRIES = 3`（额外的重试层）

**问题**：两层重试叠加，导致等待时间过长

---

## 优化方案

### 方案 A: 配置调优

1. **调整 timeout**: 90s（平衡响应速度和成功率）

2. **配置 fallbacks**：
   - `default` 失败时尝试 `vision`
   - `context_window_fallbacks`: gemini/* → gemini-3-flash-preview

3. **模型权重调整**：
   - 提高付费模型权重（稳定）
   - 降低免费模型权重（不稳定但保留）

### 方案 B: 代码优化

1. **Vision 分批处理**：
   - 当前：`max_pages_per_batch = 10`
   - 优化：`max_pages_per_batch = 3`
   - 位置：`llm.py:766`

2. **统一重试逻辑**：
   - `DEFAULT_MAX_RETRIES = 0`（完全依赖 Router 重试）
   - 位置：`llm.py:38`

3. **修复 base64 覆盖 bug**：
   - 在 PPTX+LLM 模式写入前对 `extracted_text` 执行 base64 替换
   - 位置：`cli.py:1299`

---

## 实施详情

### Step 1: markitai.json 配置修改

```json
{
  "llm": {
    "model_list": [
      // default 模型 - 调整权重
      { "model_name": "default", "litellm_params": { "model": "deepseek/deepseek-chat", "weight": 7 } },
      { "model_name": "default", "litellm_params": { "model": "openrouter/minimax/minimax-m2.1", "weight": 5 } },
      { "model_name": "default", "litellm_params": { "model": "openrouter/xiaomi/mimo-v2-flash:free", "weight": 2 } },
      { "model_name": "default", "litellm_params": { "model": "openrouter/z-ai/glm-4.7", "weight": 1 } },
      // vision 模型保持不变
    ],
    "router_settings": {
      "timeout": 90,
      "num_retries": 3,
      "fallbacks": [{ "default": ["vision"] }],
      "context_window_fallbacks": [{ "gemini/*": ["gemini/gemini-3-flash-preview"] }]
    }
  }
}
```

### Step 2: llm.py 修改

```python
# Line 38
DEFAULT_MAX_RETRIES = 0  # 完全依赖 Router 重试

# Line 766
max_pages_per_batch: int = 3  # 从 10 改为 3
```

### Step 3: cli.py 修复 (Line 1299)

```python
# 在 PPTX+LLM 模式使用 extracted_text 之前，先替换其中的 base64
if base64_images and image_result.saved_images:
    extracted_text = image_processor.replace_base64_with_paths(
        extracted_text,
        image_result.saved_images,
    )
```

---

## 关键文件

| 文件 | 修改内容 |
|------|----------|
| `markitai.json` | 权重、timeout、fallbacks、context_window_fallbacks |
| `llm.py:38` | DEFAULT_MAX_RETRIES = 0 |
| `llm.py:766` | max_pages_per_batch = 3 |
| `cli.py:1299` | extracted_text 也需要 base64 替换 |

---

## 验证方法

```bash
markitai packages/markitai/tests/fixtures --llm --ocr --alt --desc --screenshot -o ./output --verbose

# 观察指标：
# 1. 总处理时间（目标：< 3 分钟）
# 2. 超时次数（目标：减少 50%+）
# 3. 无 ContextWindowExceededError
# 4. .md 文件中无 data:image（base64 已替换）
```

---

## 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 单文件最长等待 | ~500s | ~300s |
| 免费模型超时 | 频繁 | 减少（降权后选中概率低） |
| Context 超限 | 无降级 | 自动切换 gemini-3-flash-preview |
| Vision 批次 | 10 页/批 | 3 页/批（token 更少） |
| base64 残留 | 有 | 无 |

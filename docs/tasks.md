# 任务清单

> 完成任务后，标记完成即可，不要改下面的任务描述

20260131-2（进行中）：

## 问题修复

1. ✅ **Frontmatter 换行问题** - 所有 frontmatter 字段（title、description、tags）现已规范化：
   - title: 单行，限制 200 字符，换行替换为空格
   - description: 单行，限制 150 字符，换行替换为空格
   - tags: 不含空格（用连字符替代），限制 30 字符/tag
2. ✅ **--screenshot-only 语义重定义**：
   - `--screenshot-only`: 仅截图，输出 screenshots/ 目录下的截图
   - `--llm --screenshot-only`: 截图 + LLM 提取，输出 .md + .llm.md + screenshots/

## 提示词精简计划

### 当前状态（8对提示词）

| 提示词 | 调用方法 | 有视觉参考 | 生成frontmatter |
|--------|----------|:----------:|:---------------:|
| `cleaner` | `clean_markdown()` | ❌ | ❌ |
| `document_process` | `_process_document_combined()` | ❌ | ✅ |
| `document_enhance` | `enhance_document_with_vision()` | ✅ | ❌ |
| `document_enhance_complete` | `_enhance_with_frontmatter()` | ✅ | ✅ |
| `image_analysis` | `analyze_image()` 主路径 | ✅ | N/A |
| `image_caption` | `_analyze_with_two_calls()` 回退 | ✅ | N/A |
| `image_description` | `_analyze_with_two_calls()` 回退 | ✅ | N/A |
| `page_content` | `extract_page_content()` | ✅ | N/A |

### 深度分析结论

经过对代码库的详细分析，原始精简方案需要修订：

**❌ 不可删除：image_caption / image_description**
- 原因：这两个提示词是 `_analyze_with_two_calls()` 的回退机制
- 触发条件：当 Instructor (MD_JSON) 和 JSON 模式都失败时
- 流程：先用 image_caption 获取 alt text，再用 image_description 获取描述
- 影响：删除会导致回退机制失效

**❌ 不可合并：cleaner + document_process**
- cleaner: 纯格式清洗，无结构化输出，无 frontmatter
- document_process: 格式清洗 + Pydantic 结构化输出 + frontmatter
- 原因：两者使用场景和输出格式完全不同

**✅ 可合并：document_enhance + document_enhance_complete**
- 共同点：都需要视觉参考（图片）
- 区别：前者不生成 frontmatter，后者生成
- 方案：合并为一个提示词，通过参数控制是否生成 frontmatter

### 修订后精简方案（8对 → 7对）

```
保持不变 (有独立职责):
  ├── cleaner             # 纯格式清洗
  ├── document_process    # 格式清洗 + 结构化输出
  ├── image_analysis      # 图片分析（主路径）
  ├── image_caption       # alt text 回退
  ├── image_description   # 描述回退
  ├── page_content        # PDF 页面内容提取
  ├── url_enhance         # URL 增强
  └── screenshot_extract  # 截图内容提取

合并 (功能高度重叠):
  └── document_enhance + document_enhance_complete → document_vision
      # 参数: include_frontmatter: bool = False
```

### 实施步骤

1. ✅ 创建 `document_vision_system.md` 提示词（合并两个 document_enhance）
   - 使用 `{metadata_section}` 变量控制是否生成元数据
   - 包含完整的格式清理规则和内容完整性检测
2. ✅ 修改 `DocumentProcessor` 使用新的合并提示词
   - `enhance_document_with_vision()` 使用 `metadata_section=""`
   - `_enhance_with_frontmatter()` 使用完整元数据生成指令
3. ⏳ 保留旧提示词（向后兼容）- 不删除，仅标记为废弃
   - `document_enhance_*` 和 `document_enhance_complete_*` 保留在 PROMPT_NAMES
   - 用户自定义提示词仍可使用旧名称
4. ✅ 更新配置和类型
   - 更新 config.py 添加 `document_vision_system/user`
   - 更新 config.schema.json
   - 更新 prompts/__init__.py 添加新提示词名称

---

20260131-1（进行中）：

1. ✅ 依赖精简 - 已完成 agent-browser → Playwright 迁移，移除了 Node.js 依赖
2. ✅ 补充 ffmpeg 依赖说明 - 已在 `check-deps` 命令中添加 FFmpeg 检测和安装提示
3. ✅ Twitter 页面抓取质量优化 - Jina API 使用 JSON 模式，修复标题提取 bug
4. ✅ 已完全移除 agent-browser，全量迁移到 Playwright
5. ⏳ **CLI 输出格式优化** - 需要系统性重构

---

## CLI 日志系统深度分析 (20260131-1-5)

### 问题根源

经过对代码库的全面调研，发现"日志杂乱"的根本原因：

```
┌─────────────────────────────────────────────────────┐
│  1. 11+ 个独立 Console 实例 (配置不一致)             │
│  2. loguru 拦截器深度计算错误 (源位置显示错误)       │
│  3. Rich Live 与 loguru 缺乏同步机制                │
│  4. 异步任务日志顺序无保证 (乱序输出)               │
│  5. 部分第三方库日志绕过拦截 (ONNX, markitdown)    │
│  6. 日志级别过滤逻辑复杂且易错                      │
└─────────────────────────────────────────────────────┘
```

### 影响范围

| 场景 | 严重程度 | 原因 |
|------|---------|------|
| 单文件模式 | ⚠️ 轻微 | quiet 模式默认禁用控制台日志 |
| 批处理模式 | ❌ 严重 | 多 Console + Progress bar + 并发竞争 |
| URL 批处理 | ⚠️ 中等 | 异步任务日志乱序 |

### 当前状态 (部分完成)

已完成的初步工作：
- ✅ 集中配置在 `cli/logging_config.py`
- ✅ 拦截 20+ 第三方库日志 (INTERCEPTED_LOGGERS)
- ✅ 警告过滤 (SUPPRESSED_WARNINGS)

待解决的问题：
- ⏳ Console 实例散乱（11+ 个独立实例）
- ⏳ InterceptHandler 深度追踪不准确
- ⏳ Rich Live 与 loguru 同步机制缺失
- ⏳ 异步任务日志上下文丢失
- ⏳ 部分库绕过拦截（ONNX Runtime, markitdown）

### 任务拆分

#### P0 - 立即修复（影响用户体验）

**5.1 中心化 Console 实例**
- 创建 `markitai/cli/console.py` 模块
- 提供 `get_console()` 和 `get_stderr_console()` 工厂函数
- 替换所有独立 Console() 创建
- 涉及文件：11 个 CLI 模块

**5.2 修复 InterceptHandler 深度追踪**
- 当前 `depth=2` 硬编码导致源位置显示错误
- 改用 loguru 内置的 frame 追踪机制
- 涉及文件：`cli/logging_config.py`

**5.3 协调 Rich Live 与 loguru**
- 在 Live display 期间统一控制所有输出源
- 不仅移除 console handler，还需管理其他 sink
- 涉及文件：`batch.py`, `cli/processors/batch.py`

#### P1 - 改进（提升稳定性）

**5.4 日志拦截级别调整**
- 将第三方日志器设为 DEBUG（由上层 filter 决定显示）
- 防止 DEBUG 日志绕过拦截
- 涉及文件：`cli/logging_config.py`

**5.5 改进日志过滤逻辑**
- 使用精确匹配替代子字符串匹配
- 处理日志名称层级（logger name hierarchy）
- 涉及文件：`cli/logging_config.py`

**5.6 拦截遗漏的第三方库**
- 检查 markitdown 是否有 print() 调用
- 捕获 ONNX Runtime 的直接 stderr 输出
- 涉及文件：`cli/logging_config.py`, 可能需要 subprocess 拦截

#### P2 - 长期改进

**5.7 异步日志上下文追踪**
- 使用 `contextvars` 绑定任务 ID
- 改进日志的可追踪性
- 涉及文件：`cli/processors/url.py`, `fetch_playwright.py`

**5.8 统一日志格式规范**
- 定义标准的日志前缀格式：`[STAGE]`, `[MODEL]`, `[URL]` 等
- 便于日志解析和过滤
- 涉及：所有使用 logger 的文件

**5.9 分离关注点架构**
- 日志系统 vs 进度显示 vs 用户交互的分离
- 每个层次有明确的输出渠道
- 涉及：架构级重构

### 第三方库日志状态表

| 库名 | 当前状态 | 问题 | 优先级 |
|-----|---------|------|-------|
| litellm | ✓ 已拦截 | - | - |
| httpx | ✓ 已拦截 | - | - |
| instructor | ✓ 已拦截 | - | - |
| playwright | ⚠️ 部分 | DEBUG 可能泄露 | P1 |
| rapidocr | ⚠️ 部分 | ONNX 绕过 | P1 |
| markitdown | ❌ 未检查 | 可能有 print() | P1 |
| pydub | ✓ 已拦截 | - | - |

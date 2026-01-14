# MarkIt v0.2.0 技术决策基础文档

> **性质**：完全重构，Breaking Change，不兼容 v0.1.x

## 一、项目定位

### 1.1 核心价值

**一句话定位**：多提供商 LLM 驱动的文档转换系统，实现"程序提取 + LLM 清理"的职责分离。

**解决的问题**：
- 文档格式碎片化（Word、PPT、Excel、PDF → 统一 Markdown）
- 机械转换质量低（页眉页脚混入、标题混乱、表格破损）
- 半自动化缺失（无工具整合程序能力和 LLM 理解能力）

### 1.2 差异化优势

| 维度 | MarkIt | 竞品（markitdown/pandoc/docling）|
|------|--------|--------------------------------|
| LLM 增强 | ✅ 完整 | ❌ 无 |
| 多提供商 | ✅ 5+ 个 | ❌ 无 |
| 批量处理 | ✅ 企业级 | ❌ 无 |
| 图片智能分析 | ✅ 去重+压缩+LLM | 基础/无 |

### 1.3 目标用户

1. 知识库迁移系统提供商
2. 内容运营团队（批量导入）
3. AI 训练数据准备
4. 企业文档治理

---

## 二、已验证的设计原则（必须保留）

### 2.1 核心哲学：程序提取，LLM 清理

| 职责 | 程序 | LLM |
|------|------|-----|
| 文本/图片提取 | ✅ | |
| 版式保留 | ✅ | |
| 识别无效信息 | | ✅ |
| 内容清洗 | | ✅ |
| 元数据抽取 | | ✅ |

**禁止**：用正则清理"无效内容"（v0.1.5 失败案例已验证）

### 2.2 多 Chunk 处理策略

```
首 Chunk → 完整 frontmatter + 内容
续 Chunk → 仅 partial_metadata (JSON) + 内容

合并阶段：
1. 解析首段 frontmatter
2. 合并续段元数据（去重）
3. 注入最终 frontmatter 到开头
```

### 2.3 分层并发控制

```
全局层：LLM 总并发数 (llm_workers)
   ↓
Credential 层：每凭证 AIMD 限流（自适应 rate limit）
   ↓
文件层：每文件 chunk semaphore
```

### 2.4 Prompt 集中管理

```
config/prompts/
├── enhancement_{zh,en}.md
├── enhancement_continuation_{zh,en}.md
├── summary_{zh,en}.md
└── image_analysis_{zh,en}.md
```

---

## 三、需要重新设计的部分

### 3.1 LLM Provider 层 → LiteLLM

**当前问题**：
- 5 个 Provider 实现（~1300 行代码）
- 每新增一个 Provider 需要写完整类
- SDK 依赖散乱（openai, anthropic, google-genai, ollama）

**v0.2.0 方案**：
- 统一使用 LiteLLM SDK
- 保留自定义 AIMD 限流（LiteLLM 不提供）
- 保留并发回退机制（LiteLLM 不提供）

### 3.2 CLI 交互设计 → 极简化

**最终设计**：
```bash
# 核心用法
markit document.docx              # 单文件 → ./output/document.md
markit ./docs                     # 目录 → ./output/...（保持结构）
markit ./docs -o ./my-output      # 指定输出目录

# 功能选项
markit path --llm                 # 启用 LLM 增强（默认关闭）
markit path --ocr                 # 启用 OCR（扫描件）
markit path --dry-run             # 预览计划

# 无子命令
# 用户直接编辑 markit.json 配置
```

**移除**：
- 所有子命令（config、provider、model）
- `convert` / `batch` 区分
- `--recursive` 参数

**默认行为**：
- LLM 增强：默认关闭（需 `--llm`）
- 输出目录：`./output/`

### 3.3 配置系统 → JSON 格式

**用户决策**：不兼容 v0.1.x，采用 JSON 格式

**设计方向**：
```json
{
  "llm": {
    "providers": [
      {
        "name": "deepseek",
        "model": "deepseek/deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY"
      }
    ],
    "default": "deepseek"
  },
  "output": {
    "format": "markdown",
    "language": "zh"
  }
}
```

**简化点**：
- 移除 credentials/models 分离（合并为 providers）
- 直接使用 LiteLLM 模型格式
- 配置文件名：`markit.json`

### 3.4 测试体系

**v0.2.0 目标**：
- 当前覆盖率 81% → 保持或提高
- 完善 E2E 测试（LiteLLM 各 Provider）
- 自动化回归测试

---

## 四、技术选型待决策

### 4.1 LLM SDK

| 选项 | 优点 | 缺点 | 推荐 |
|------|------|------|------|
| **LiteLLM** | 统一 API、100+ Provider、自动定价 | 依赖体积大 | ✅ |
| 保留当前 | 已验证稳定 | 维护成本高、新 Provider 慢 | ❌ |

### 4.2 OCR 引擎（v0.2.0 或 v0.2.1）

| 选项 | 中文准确度 | 依赖 | 推荐 |
|------|-----------|------|------|
| **RapidOCR** | ⭐⭐⭐⭐⭐ | ONNX 轻量 | ✅ 主引擎 |
| **PaddleOCR** | ⭐⭐⭐⭐⭐ | PaddlePaddle | ✅ 备引擎 |

### 4.3 配置格式

| 选项 | 说明 | 推荐 |
|------|------|------|
| JSON | 用户决策 | ✅ 采用 |
| YAML | 当前格式 | ❌ 弃用 |
| TOML | 已迁移过 | ❌ |

---

## 五、v0.2.0 范围界定

### 包含

1. **LLM 层重构**：切换至 LiteLLM
2. **CLI 重设计**：`markit <path>` 极简入口
3. **配置重设计**：JSON 格式，简化结构
4. **OCR 能力**：RapidOCR 引擎
5. **测试完善**：E2E 测试全部功能
6. **全中文化**：CLI 输出、文档全部中文

### 不包含

1. v0.1.x 兼容性
2. 新转换格式（延期）
3. GUI/Web 界面

---

## 六、已确认决策

| 决策项 | 结果 |
|--------|------|
| CLI 结构 | `markit <path>` 自动识别，无子命令 |
| 配置格式 | JSON（`markit.json`），用户手动编辑 |
| v0.1.x 兼容 | 不兼容 |
| LLM 默认 | 关闭（需 `--llm` 启用）|
| 输出目录 | `./output/` |
| OCR | 包含 RapidOCR |
| 语言 | 全部中文 |

---

## 七、实施路线图

### Phase 1: 核心框架

1. 创建新的 CLI 入口（`markit <path>`）
2. 实现 JSON 配置加载
3. 保留核心转换逻辑

### Phase 2: LiteLLM 集成

1. 创建 LiteLLMAdapter
2. 保留 AIMD 限流逻辑
3. 保留并发回退机制
4. 删除旧 Provider 代码

### Phase 3: OCR 集成

1. 集成 RapidOCR
2. 实现 PDF 扫描件识别
3. 图片 OCR 支持

### Phase 4: 测试与文档

1. E2E 测试
2. 全中文文档
3. 示例配置

---

## 八、关键文件变更

### 新建

| 文件 | 用途 |
|------|------|
| `src/markit/cli/main.py` | 新 CLI 入口（重写）|
| `src/markit/llm/litellm_adapter.py` | LiteLLM 适配器 |
| `src/markit/ocr/rapidocr.py` | RapidOCR 集成 |
| `markit.example.json` | 配置示例 |

### 删除

| 文件 | 原因 |
|------|------|
| `src/markit/llm/openai.py` | LiteLLM 替代 |
| `src/markit/llm/anthropic.py` | LiteLLM 替代 |
| `src/markit/llm/gemini.py` | LiteLLM 替代 |
| `src/markit/llm/ollama.py` | LiteLLM 替代 |
| `src/markit/llm/openrouter.py` | LiteLLM 替代 |
| `src/markit/cli/commands/*.py` | 子命令移除 |

### 重写

| 文件 | 变更 |
|------|------|
| `src/markit/config/settings.py` | JSON 格式 |
| `src/markit/llm/manager.py` | 简化，使用 LiteLLM |
| `pyproject.toml` | 依赖更新 |

# Markitai 文档/测试优化分析报告

> 生成时间: 2026-01-17
> 分析范围: docs/, packages/markitai/src/, packages/markitai/tests/
> 版本: 0.2.0

---

## 执行摘要

本次分析发现 **8 类文档/测试问题**，涉及规格文档与实现不一致、配置 Schema 缺失字段、测试文件缺失等。建议优先处理高优先级问题以保持文档与代码同步。

| 优先级 | 问题数 | 分类 |
|--------|--------|------|
| 🔴 高  | 3 | CLI 参数、Schema 缺失字段、SKILL.md 缺失 |
| 🟡 中  | 3 | RouterSettings 默认值、Prompts 配置、Preset 系统 |
| 🟢 低  | 2 | 依赖列表、LogConfig 默认值 |

---

## 1. 高优先级问题

### 1.1 🔴 CLI 参数与文档不一致

**问题描述**：spec.md 和 requirement.md 描述的 `--image` 参数在实际代码中已被拆分为多个独立参数。

| 文档 | 实际代码 |
|------|----------|
| `--image` (spec.md 3.2节) | `--alt/--no-alt` |
| `--image` (requirement.md 第21行) | `--desc/--no-desc` |
| - | `--screenshot/--no-screenshot` |

**代码位置**：
- spec.md:238 `--image FLAG`
- requirement.md:21 `markitai document.pdf --image`
- cli.py:216-234 实际参数定义

**建议修复**：
1. 更新 spec.md 3.2节，替换 `--image` 为 `--alt`, `--desc`, `--screenshot`
2. 更新 requirement.md 接口示例
3. 添加 `--preset` 参数说明

---

### 1.2 🔴 config.schema.json 缺失字段

**问题描述**：config.py 中的多个配置字段未同步到 JSON Schema，导致 IDE 校验不完整。

**缺失字段列表**：

| 配置块 | 缺失字段 | config.py 位置 |
|--------|----------|----------------|
| ImageConfig | `alt_enabled: bool = False` | config.py:91 |
| ImageConfig | `desc_enabled: bool = False` | config.py:92 |
| OCRConfig | `enable_screenshot: bool = False` | config.py:106 |
| PromptsConfig | `image_analysis: str \| None = None` | config.py:117 |
| MarkitaiConfig | `presets: dict[str, PresetConfig]` | config.py:164 |
| (新定义) | `PresetConfig` 类型定义 | config.py:136-144 |

**建议修复**：
将以上字段添加到 `config.schema.json`，保持与 config.py 同步。

---

### 1.3 🔴 tests/SKILL.md 文件缺失

**问题描述**：requirement.md 第9行明确要求维护 `tests/SKILL.md` 用于开发者和大模型进行测试，但该文件不存在。

**requirement.md 原文**：
> 所有特性都需要测试覆盖，除了支持程序自动执行的单元测试外，需要维护一个用于开发者和大模型进行测试的 tests/SKILL.md 文件

**spec.md 14.3节已有模板**，但文件未创建。

**建议修复**：
根据 spec.md 14.3节模板创建 `tests/SKILL.md`。

---

## 2. 中优先级问题

### 2.1 🟡 RouterSettings 默认值不一致

**问题描述**：spec.md 与 config.py 中的默认值不同。

| 字段 | spec.md 4.3节 | config.py | config.schema.json |
|------|---------------|-----------|-------------------|
| `num_retries` | 3 | **2** | 2 |
| `timeout` | 60 | **120** | 120 |

**代码位置**：
- spec.md:347-348
- config.py:64-65
- config.schema.json:362-370

**建议修复**：
统一 spec.md 与代码一致：`num_retries=2`, `timeout=120`

---

### 2.2 🟡 Prompts 配置文档不完整

**问题描述**：实际支持的 prompts 数量与文档不符。

| 来源 | prompts 数量 | 列表 |
|------|--------------|------|
| spec.md 10.1节 | 4 | cleaner, frontmatter, image_caption, image_description |
| config.py PromptsConfig | 5 | 上述 + image_analysis |
| prompts/\_\_init\_\_.py PROMPT_NAMES | **7** | 上述 + page_content, document_enhance |
| prompts/*.md 文件 | **7** | 全部 |

**缺失文档的 prompts**：
- `image_analysis` - 合并的图片分析提示词
- `page_content` - 页面内容提取提示词
- `document_enhance` - 文档增强提示词

**建议修复**：
1. 在 spec.md 10.1节补充 `image_analysis`, `page_content`, `document_enhance`
2. 在 config.py PromptsConfig 添加 `page_content` 和 `document_enhance` 字段

---

### 2.3 🟡 Preset 系统未文档化

**问题描述**：v0.2.0 新增的 preset 功能（rich/standard/minimal）在 spec.md 中完全缺失。

**实现位置**：
- config.py:136-151 `PresetConfig` 和 `BUILTIN_PRESETS`
- cli.py:205-209 `--preset` 参数
- cli.py:341-354 preset 应用逻辑

**建议修复**：
在 spec.md 中新增 "Presets 系统" 章节，说明：
- 内置预设定义（rich/standard/minimal）
- CLI 使用方式（`--preset rich`）
- 自定义预设配置方法

---

## 3. 低优先级问题

### 3.1 🟢 依赖列表不一致

**问题描述**：spec.md 依赖列表与 pyproject.toml 不同步。

| 差异 | spec.md 1.3节/主包配置 | pyproject.toml |
|------|------------------------|----------------|
| 移除 | `click-default-group>=1.2.4` | ❌ 已移除 |
| 新增 | - | `instructor>=1.14.0` |

**建议修复**：
1. 从 spec.md 移除 `click-default-group` 相关描述
2. 添加 `instructor` 依赖说明（用于 LLM structured output）

---

### 3.2 🟢 LogConfig.level 默认值不一致

**问题描述**：config.schema.json 的默认值与 config.py 不一致。

| 来源 | 默认值 |
|------|--------|
| config.py:130 | `"DEBUG"` |
| spec.md 4.3节 | `"DEBUG"` |
| config.schema.json:168 | `"INFO"` ❌ |

**建议修复**：
将 config.schema.json 第168行的 `"default": "INFO"` 改为 `"default": "DEBUG"`

---

## 4. 测试覆盖分析

### 4.1 现有测试文件

```
tests/
├── unit/
│   ├── test_batch.py
│   ├── test_cli_helpers.py
│   ├── test_config.py
│   ├── test_converter.py
│   ├── test_image.py
│   ├── test_image_converter.py
│   ├── test_llm.py
│   ├── test_ocr.py
│   ├── test_prompts.py
│   └── test_security.py
├── integration/
│   └── test_cli.py
├── fixtures/
│   └── (测试文件)
└── conftest.py
```

### 4.2 测试覆盖建议

| 模块 | 现有测试 | 建议补充 |
|------|----------|----------|
| Preset 系统 | ❌ 无 | 添加 preset 加载/应用测试 |
| config.schema.json 验证 | ❌ 无 | 添加 schema 与 config.py 同步验证 |
| 新增 prompts | ❌ 无 | page_content, document_enhance 测试 |

---

## 5. 文档结构优化建议

### 5.1 spec.md 章节补充

建议在 spec.md 添加以下内容：

1. **3.2节** - 补充 `--preset`, `--alt`, `--desc`, `--screenshot` 参数
2. **新增 4.5节** - "Presets 配置" 章节
3. **10.1节** - 补充 image_analysis, page_content, document_enhance prompts

### 5.2 requirement.md 更新

更新接口示例以反映当前 CLI 设计：

```bash
# 原：
markitai document.pdf --image

# 改为：
markitai document.pdf --preset rich          # 使用 rich 预设
markitai document.pdf --alt --desc           # 手动启用图片分析
```

---

## 6. 总结

### 6.1 必须修复（阻断性问题）

1. ✅ 创建 `tests/SKILL.md`
2. ✅ 同步 `config.schema.json` 字段

### 6.2 应该修复（文档准确性）

1. ✅ 更新 spec.md CLI 参数章节
2. ✅ 更新 spec.md RouterSettings 默认值
3. ✅ 补充 spec.md Prompts 和 Preset 章节
4. ✅ 更新 requirement.md 接口示例

### 6.3 可选修复（低优先级）

1. ✅ 同步依赖列表
2. ✅ 修复 schema LogConfig 默认值

---

## 附录：关键文件引用

| 文件 | 用途 |
|------|------|
| docs/requirement.md | 需求文档 |
| docs/spec.md | 技术规格文档 |
| packages/markitai/src/markitai/cli.py | CLI 实现 |
| packages/markitai/src/markitai/config.py | 配置模型 |
| packages/markitai/src/markitai/config.schema.json | JSON Schema |
| packages/markitai/src/markitai/prompts/__init__.py | Prompt 管理 |
| packages/markitai/pyproject.toml | 依赖配置 |

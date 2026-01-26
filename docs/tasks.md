# Markitai 任务清单

## 官网建设任务

来源: `20260121-官网规划`
创建: 2026-01-21

### 技术选型

| 项目 | 选择 |
|------|------|
| 框架 | VitePress 2.x alpha (`vitepress@next`) |
| 目录 | `website/` (独立于现有 `docs/` 技术文档) |
| 语言 | 中英双语 (英文为默认) |
| 功能 | 本地搜索、深色模式、GitHub Actions 自动部署 |

### 目录结构

```
website/
├── .vitepress/
│   └── config.ts              # VitePress 配置（含 i18n）
├── public/
│   └── logo.svg               # Logo (可选)
├── index.md                   # 英文首页
├── guide/
│   ├── getting-started.md     # 英文快速开始
│   ├── configuration.md       # 英文配置说明
│   └── cli.md                 # 英文 CLI 命令
├── zh/                        # 中文目录
│   ├── index.md               # 中文首页
│   └── guide/
│       ├── getting-started.md # 中文快速开始
│       ├── configuration.md   # 中文配置说明
│       └── cli.md             # 中文 CLI 命令
```

### 任务清单

- [x] **任务 1: 初始化 VitePress 项目** (2026-01-21)
  - 创建 `website/` 目录
  - 创建 `website/package.json`
  - 安装 `vitepress@next` 依赖
  - 创建 `.vitepress/config.ts` 配置文件
  - 配置中英双语 (locales)
  - 配置本地搜索 (search: { provider: 'local' })
  - 配置 socialLinks (GitHub)
  - 配置 footer

- [x] **任务 2: 创建英文首页 (index.md)** (2026-01-21)
  - Hero Section: name, text, tagline, actions
  - Features Section: 4 个核心特性
    - Multi-format Support (📄)
    - LLM Enhancement (🤖)
    - Batch Processing (⚡)
    - OCR Recognition (🔍)

- [x] **任务 3: 创建英文文档** (2026-01-21)
  - `guide/getting-started.md`: 安装、快速开始、预设
  - `guide/configuration.md`: 配置文件、环境变量、优先级
  - `guide/cli.md`: CLI 命令参考

- [x] **任务 4: 创建中文首页 (zh/index.md)** (2026-01-21)
  - 复用英文首页结构，翻译为中文
  - 从现有 README.md 迁移内容

- [x] **任务 5: 创建中文文档** (2026-01-21)
  - `zh/guide/getting-started.md`: 从 README.md 迁移
  - `zh/guide/configuration.md`: 从 README.md 和 spec.md 提取
  - `zh/guide/cli.md`: 从 spec.md 提取 CLI 文档

- [x] **任务 6: 配置 GitHub Actions 自动部署** (2026-01-21)
  - 创建 `.github/workflows/deploy-website.yml`
  - 触发条件: push 到 main 分支且 website/ 有变更
  - 部署目标: GitHub Pages
  - 配置 pnpm 缓存

- [x] **任务 7: 更新 .gitignore** (2026-01-21)
  - 添加 `website/.vitepress/cache`
  - 添加 `website/.vitepress/dist`
  - 添加 `website/node_modules`

- [x] **任务 8: 添加 npm scripts** (2026-01-21)
  - `docs:dev`: 启动开发服务器
  - `docs:build`: 构建生产版本
  - `docs:preview`: 预览生产版本

- [x] **任务 9: 主题定制 (CSS-only)** (2026-01-21)
  - 创建 `.vitepress/theme/index.ts` 主题入口
  - 创建 `.vitepress/theme/custom.css` 自定义样式
  - 品牌颜色配合 logo (#18181b)
  - Hero 标题渐变效果
  - 深色模式适配
  - 构建测试通过

### 页面内容规划

#### 首页 (index.md / zh/index.md)

```yaml
layout: home

hero:
  name: Markitai
  text: Document to Markdown Converter  # 英文
  # text: 开箱即用的 Markdown 转换器           # 中文
  tagline: With native LLM enhancement support
  # tagline: 原生支持 LLM 增强
  actions:
    - theme: brand
      text: Get Started / 快速开始
      link: /guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/Ynewtime/markitai

features:
  - icon: 📄
    title: Multi-format Support / 多格式支持
    details: DOCX, PPTX, XLSX, PDF, TXT, MD, JPG/PNG/WebP, URLs
  - icon: 🤖
    title: LLM Enhancement / LLM 增强
    details: Format cleaning, metadata generation, image analysis
  - icon: ⚡
    title: Batch Processing / 批量处理
    details: Concurrent conversion with resume capability
  - icon: 🔍
    title: OCR Recognition / OCR 识别
    details: Text extraction from scanned PDFs and images
```

#### 快速开始 (guide/getting-started.md)

1. 安装要求 (Python 3.11+)
2. 安装命令 (`uv add markitai`)
3. 基础用法
4. LLM 增强
5. 预设系统 (rich/standard/minimal)
6. 批量处理

#### 配置说明 (guide/configuration.md)

1. 配置优先级
2. 配置文件格式 (markitai.json)
3. 环境变量
4. LLM 配置
5. 缓存配置

#### CLI 命令 (guide/cli.md)

1. 基础命令 (`markitai <input>`)
2. 转换选项 (`--llm`, `--preset`, `--alt`, `--desc`, `--screenshot`, `--ocr`)
3. 输出选项 (`-o`, `--resume`)
4. 配置命令 (`markitai config`)
5. 缓存命令 (`markitai cache`)

### VitePress 配置参考

```ts
// .vitepress/config.ts
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Markitai',
  description: 'Document to Markdown converter with native LLM support',

  locales: {
    root: {
      label: 'English',
      lang: 'en',
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      description: '开箱即用的 Markdown 转换器，原生支持 LLM 增强',
      themeConfig: {
        nav: [
          { text: '指南', link: '/zh/guide/getting-started' }
        ],
        sidebar: {
          '/zh/guide/': [
            { text: '快速开始', link: '/zh/guide/getting-started' },
            { text: '配置说明', link: '/zh/guide/configuration' },
            { text: 'CLI 命令', link: '/zh/guide/cli' }
          ]
        }
      }
    }
  },

  themeConfig: {
    search: { provider: 'local' },
    nav: [
      { text: 'Guide', link: '/guide/getting-started' }
    ],
    sidebar: {
      '/guide/': [
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Configuration', link: '/guide/configuration' },
        { text: 'CLI Reference', link: '/guide/cli' }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Ynewtime/markitai' }
    ],
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present'
    }
  }
})
```

### GitHub Actions 部署配置参考

```yaml
# .github/workflows/deploy-website.yml
name: Deploy Website

on:
  push:
    branches: [main]
    paths:
      - 'website/**'
      - '.github/workflows/deploy-website.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
          cache-dependency-path: website/pnpm-lock.yaml
      - name: Install dependencies
        run: pnpm install
        working-directory: website
      - name: Build
        run: pnpm docs:build
        working-directory: website
      - uses: actions/upload-pages-artifact@v3
        with:
          path: website/.vitepress/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

---

## 代码重构任务

来源: `20260121-需求规划-1` 需求点2
更新: 2026-01-21 (基于最新代码深入分析)
实施: 2026-01-21

### 已完成 (2026-01-21)

- [x] **统一 `detect_language()` 函数实现**
  - 删除 `llm.py` 中的 `_detect_language` 方法
  - 统一使用 `workflow/helpers.py` 中的 `detect_language()` 返回 `"zh"/"en"`
  - 新增 `get_language_name()` 函数将代码转换为完整名称 `"Chinese"/"English"`
  - `llm.py` 通过 `get_language_name(detect_language(content))` 获取语言名称

- [x] **提取目录创建工具函数**
  - 新建 `utils/paths.py` 模块
  - 实现 `ensure_dir()`, `ensure_subdir()`, `ensure_assets_dir()`, `ensure_screenshots_dir()`
  - 更新 `image.py` (4处), `pdf.py` (3处), `office.py` (3处), `cli.py` (5处), `helpers.py` (1处)

- [x] **统一 MIME 类型映射**
  - 在 `constants.py` 添加 `EXTENSION_TO_MIME`, `MIME_TO_EXTENSION`, `IMAGE_EXTENSIONS`
  - 新建 `utils/mime.py` 实现 `get_mime_type()`, `get_extension_from_mime()`
  - 更新 `llm.py` (5处), `pdf.py` (1处), `image.py` (1处)

- [x] **提取图片扩展名常量**
  - 在 `constants.py` 添加 `IMAGE_EXTENSIONS` 元组
  - 更新 `cli.py` 中 6 处重复的图片扩展名检查

- [x] **创建 LLM 用量累加器类**
  - 在 `workflow/helpers.py` 添加 `LLMUsageAccumulator` 类
  - 提供 `add(cost, usage)` 和 `reset()` 方法

- [x] **统一 LLM Processor 实例化模式**
  - 在 `workflow/helpers.py` 添加 `create_llm_processor()` 工厂函数
  - 更新 `cli.py` (5处) 和 `workflow/single.py` (1处)

- [x] **提取 `normalize_markdown_whitespace()` 到 utils** (2026-01-21)
  - 新建 `utils/text.py` 模块
  - 将 `LLMProcessor._normalize_whitespace` 移动为独立函数
  - 更新 `llm.py` 和 `cli.py` 调用新函数
  - 保留 `LLMProcessor._normalize_whitespace` 作为兼容性包装器

- [x] **重构 PowerShell COM 转换脚本** (2026-01-21)
  - 文件: `converter/legacy.py`
  - 新增 `COMAppConfig` 数据类封装 Office 应用配置
  - 提取 `_build_single_file_script()` 和 `_build_batch_script()` 模板函数
  - 统一 `_convert_with_com()` 和 `_batch_convert_with_com()` 通用函数
  - 保留原有函数别名确保向后兼容
  - **代码减少**: ~220 行 → ~180 行 (~18% 减少)

- [x] **创建 workflow/core.py 核心转换模块** (2026-01-21)
  - 新建 `workflow/core.py` 实现统一转换流程
  - 定义 `ConversionContext` 和 `ConversionStepResult` 数据类
  - 实现步骤函数: `validate_and_detect_format()`, `prepare_output_directory()`, `convert_document()`, `resolve_output_file()`, `process_embedded_images()`, `write_base_markdown()`, `process_with_vision_llm()`, `process_with_standard_llm()`, `analyze_embedded_images()`
  - 实现 `convert_document_core()` 管道函数
  - **状态**: 代码完成，但 `cli.py` 中的 `process_single_file()` 和 `process_file()` 尚未迁移使用
  - **原因**: 测试覆盖率不足 (cli.py 52%)，直接替换风险较高
  - **计划**: 提升测试覆盖率后再迁移

### 已完成：迁移至 `convert_document_core()` (2026-01-21)

<details>
<summary>详细迁移方案 (点击展开)</summary>

#### 代码结构对比

| 方面 | `process_single_file()` | `process_file()` | `workflow/core.py` |
|------|------------------------|------------------|-------------------|
| 行数 | ~400 行 | ~380 行 | ~650 行 |
| 错误处理 | `SystemExit(1)` | 返回 `ProcessResult` | 返回 `ConversionStepResult` |
| 进度显示 | `ProgressReporter` | 无 (批处理用 Live) | 无 |
| dry-run | 支持 | 由外层处理 | 不支持 |
| 预转换文件 | 不支持 | 支持 `preconverted_map` | 支持 `actual_file` |
| 多进程图片 | 不使用 | 使用 (大批量) | 支持 |
| 共享 Processor | 每次新建 | 使用 `shared_processor` | 支持 |
| 目录结构保持 | 不需要 | 需要 `relative_to()` | 需外部传入 |
| 报告生成 | 内部生成 | 返回结果供聚合 | 不处理 |
| stdout 输出 | 输出 markdown | 不输出 | 不处理 |

#### `workflow/core.py` 缺失功能

1. **dry-run 支持** - 需在调用层处理
2. **进度显示** - 需通过回调或调用层处理
3. **报告生成** - 需在调用层处理
4. **stdout 输出** - 需在调用层处理
5. **cache_hit 检测** - 需在 context 中添加

#### 循环依赖问题

`workflow/core.py` 从 `cli.py` 导入 `resolve_output_path()`，需移动到共享模块。

---

#### 阶段 0: 前置准备

**任务 0.1**: 提取 `resolve_output_path()` 到 `utils/output.py`

```python
# utils/output.py
def resolve_output_path(base_path: Path, on_conflict: str) -> Path | None:
    """Resolve output file path with conflict handling."""
    if not base_path.exists():
        return base_path
    if on_conflict == "skip":
        return None
    elif on_conflict == "overwrite":
        return base_path
    elif on_conflict == "rename":
        stem, suffix, parent = base_path.stem, base_path.suffix, base_path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
    return base_path
```

**任务 0.2**: 扩展 `ConversionContext` 数据类

```python
@dataclass
class ConversionContext:
    # ... existing fields ...
    duration_seconds: float = 0.0
    cache_hit: bool = False
    input_base_dir: Path | None = None  # For batch relative path
    on_stage_complete: Callable[[str, float], None] | None = None
```

**任务 0.3**: 添加 `workflow/core.py` 单元测试 (目标覆盖率 > 80%)

---

#### 阶段 1: 迁移 `process_single_file()`

**任务 1.1**: 创建 `_process_single_file_v2()` 包装函数

**任务 1.2**: 添加特性开关 `MARKITAI_USE_WORKFLOW_CORE`

```python
USE_WORKFLOW_CORE = os.environ.get("MARKITAI_USE_WORKFLOW_CORE", "0") == "1"

async def process_single_file(...):
    if USE_WORKFLOW_CORE:
        return await _process_single_file_v2(...)
    # 现有逻辑...
```

**任务 1.3**: 集成测试 `tests/integration/test_workflow_core_cli.py`

---

#### 阶段 2: 迁移 `process_file()`

**任务 2.1**: 创建 `_create_batch_process_file_v2()` 工厂函数

**任务 2.2**: 添加特性开关和集成测试

---

#### 阶段 3: 清理

**任务 3.1**: 将特性开关默认设为 `True`

**任务 3.2**: 清理旧代码 (保留 `_legacy` 后缀一个版本)

**任务 3.3**: 更新文档

---

#### 依赖前提

| 依赖项 | 当前状态 | 目标状态 |
|--------|---------|---------|
| `cli.py` 覆盖率 | 52% | > 70% |
| `workflow/core.py` 覆盖率 | 23% | > 80% |
| `workflow/single.py` 覆盖率 | 20% | > 60% |

#### 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| 回归 bug | 高 | 特性开关、充分测试、保留回退 |
| 行为差异 | 中 | 对比测试、详细日志 |
| 性能下降 | 低 | 基准测试 |

</details>

- [x] **阶段 0: 前置准备** (2026-01-21)
  - [x] 0.1 提取 `resolve_output_path()` 到 `utils/output.py`
  - [x] 0.2 扩展 `ConversionContext` 数据类 (添加 `duration_seconds`, `cache_hit`, `input_base_dir`, `on_stage_complete`)
  - [x] 0.3 添加 `workflow/core.py` 单元测试 (43 个测试, 782 行)

- [x] **阶段 1: 迁移 `process_single_file()`** (2026-01-21)
  - [x] 1.1 创建 `_process_single_file_v2()` 包装函数 (`cli.py:1370-1530`)
  - [x] 1.2 添加特性开关 `MARKITAI_USE_LEGACY_CLI` (反向逻辑: v2 为默认)
  - [x] 1.3 集成测试 (`TestWorkflowCoreV2`, `TestLegacyFallback`)

- [x] **阶段 2: 迁移 `process_file()`** (2026-01-21)
  - [x] 2.1 创建 `_create_process_file_v2()` 工厂函数 (`cli.py:2907`)
  - [x] 2.2 添加特性开关和集成测试

- [x] **阶段 3: 清理和文档更新** (2026-01-21)
  - [x] 3.1 v2 已为默认实现，`MARKITAI_USE_LEGACY_CLI=1` 可回退到旧版
  - [x] 3.2 保留旧代码供回退使用 (标记为 legacy)
  - [x] 3.3 CLI 帮助和集成测试已包含 v2 说明

---

## 测试代码优化任务

来源: 代码重构分析 (2026-01-21)
实施: 2026-01-21

### 已完成 (2026-01-21)

- [x] **扩展 `conftest.py` 公共 Fixtures**
  - 文件: `packages/markitai/tests/conftest.py`
  - **新增 Fixtures**:
    - `cli_runner` - Click CLI 测试运行器
    - `llm_config` - 测试用 LLM 配置
    - `prompts_config` - 测试用 Prompts 配置
    - `sample_txt_file(tmp_path)` - 创建测试文本文件
    - `sample_md_file(tmp_path)` - 创建测试 Markdown 文件
    - `create_test_image` - 工厂 fixture，创建测试图片
    - `sample_png_bytes` - 最小 PNG 字节数据
    - `mock_llm_response` - 工厂 fixture，创建 mock LLM 响应

- [x] **重复 Fixture 清理** (2026-01-21)
  - 清理并统一使用 conftest.py 中的公共 fixtures
  - **已清理位置**:
    - `test_llm.py`: 移除 `llm_config`/`prompts_config`，使用 conftest.py 版本
    - `test_cli.py`: 创建 `runner` 别名指向 `cli_runner`
    - `test_cache.py`: 创建 `runner` 别名指向 `cli_runner`
    - `test_url.py`: 移除类级别 `runner` fixtures，使用 `cli_runner`
    - `test_image_converter.py`: 使用 `sample_png_bytes` 简化 `sample_image`

### 延后任务

- [x] **分层 conftest.py 结构** (2026-01-21, 关闭)
  - **决定**: 当前 `conftest.py` 仅 285 行，对项目规模合理，暂不分层
  - 未来如测试数量显著增长可重新评估

---

## 缓存增强任务

来源: `20260121-缓存优化` 需求

### 需求背景

1. **精细化缓存控制**：批处理时需要支持对单个文件或子目录禁用缓存，而非全局禁用
2. **缓存命中准确性**：当前 hash 计算只用前 50000 字符，大文档末尾改变不触发缓存失效

### 任务清单

- [x] **任务 1: 新增 `--no-cache-for` CLI 参数** (2026-01-21)
  - 文件: `cli.py`
  - 参数: `--no-cache-for <pattern>` 支持文件路径和 glob 模式（逗号分隔）
  - 示例:
    - `--no-cache-for file1.pdf` 匹配 `输入目录/file1.pdf`
    - `--no-cache-for "*.pdf"` 匹配输入目录下的 PDF
    - `--no-cache-for "**/file1.pdf"` 匹配所有子目录下的 file1.pdf
    - `--no-cache-for "*.pdf,reports/**"` 混合模式
  - 注意: `--no-cache` 保持全局禁用语义不变

- [x] **任务 2: 更新配置模型** (2026-01-21)
  - 文件: `config.py`
  - 在 `CacheConfig` 中添加 `no_cache_patterns: list[str]`

- [x] **任务 3: 实现缓存跳过逻辑** (2026-01-21)
  - 文件: `llm.py`
  - 修改 `LLMProcessor.__init__()` 接受 `no_cache_patterns` 参数
  - 修改 `PersistentCache` 实现 `_should_skip_cache(context)` 方法
  - 使用 `fnmatch` 进行 glob 匹配
  - 匹配基于相对路径（相对于输入目录）

- [x] **任务 4: 修复 hash 计算（首+尾+长度）** (2026-01-21)
  - 文件: `llm.py`
  - 修改 `SQLiteCache._compute_hash()`
  - 新算法: `hash(prompt + length + head[:25000] + tail[-25000:])`
  - 确保首尾改变和长度变化都触发缓存失效

- [x] **任务 5: 更新 LLMProcessor 调用点** (2026-01-21)
  - 文件: `cli.py`, `workflow/single.py`
  - 更新所有 `LLMProcessor()` 构造调用，传入 `no_cache_patterns`

- [x] **任务 6: 添加测试** (2026-01-21)
  - 文件: `tests/integration/test_cache.py`
  - 新增 `TestNoCachePatterns` 测试类（6 个测试）
  - 新增 `TestCacheHashComputation` 测试类（6 个测试）
  - 测试 `--no-cache-for` 单文件、glob 模式、混合模式
  - 测试新 hash 计算逻辑

- [x] **任务 7: 更新 CLI 帮助文档** (2026-01-21)
  - `--no-cache`: "Disable LLM result caching (force fresh API calls)."
  - `--no-cache-for`: "Disable cache for specific files/patterns (comma-separated, supports glob)."

---

## 缓存查看增强任务

来源: `20260121-缓存调试` 需求

### 需求背景

1. 现有 `markitai cache stats` 只显示基础统计，无法查看具体缓存条目
2. 需要按模型、key 等维度分析缓存使用情况
3. 检查缓存命中：使用 `markitai cache stats -v` 查看缓存条目
   - 注：`--dry-run` 不检查具体缓存命中（与业界实践一致，dry-run 显示「会执行什么」而非「缓存是否命中」）
   - 缓存命中是运行时行为，精确预测需要完整转换文档并计算 hash，开销较大

### 任务清单

- [x] **任务 1: 添加 `-v/--verbose` 参数** (2026-01-21)
  - 文件: `cli.py`
  - 默认行为（无 `-v`）: 保持现有输出（基础统计）
  - `-v` 模式: 显示按模型分组统计 + 最近 N 条缓存条目

- [x] **任务 2: 添加 `--limit` 参数** (2026-01-21)
  - 文件: `cli.py`
  - 参数: `--limit N`（默认 20）
  - 控制 `-v` 模式下显示的条目数量

- [x] **任务 3: 添加 `--scope` 参数** (2026-01-21)
  - 文件: `cli.py`
  - 参数: `--scope project|global|all`（默认 all）
  - 控制显示哪个缓存的详细信息

- [x] **任务 4: SQLiteCache 新增方法** (2026-01-21)
  - 文件: `llm.py`
  - 新增 `list_entries(limit: int) -> list[dict]`: 列出缓存条目
  - 新增 `stats_by_model() -> dict[str, dict]`: 按模型分组统计
  - 新增 `_parse_value_preview(value: str) -> str`: 解析值预览

- [x] **任务 5: 更新输出格式** (2026-01-21)
  - 普通模式: 保持现有输出
  - Verbose 模式: 使用 rich Table 显示条目列表
  - JSON 模式: 包含完整的 by_model 和 entries 数据

- [x] **任务 6: 添加测试** (2026-01-21)
  - 文件: `tests/integration/test_cache.py`
  - 新增 `TestSQLiteCacheVerboseMethods` 测试类（9 个测试）
  - 在 `TestCacheCLICommands` 中新增 5 个测试
  - 测试 `cache stats -v` 输出格式
  - 测试 `--scope` 和 `--limit` 参数

### 详细设计

#### CLI 参数

```bash
markitai cache stats                          # 基础统计（现有行为）
markitai cache stats -v                       # 详细模式：by-model + 最近条目
markitai cache stats -v --limit 50            # 显示最近 50 条
markitai cache stats -v --scope project       # 只看 project cache
markitai cache stats -v --scope global        # 只看 global cache
markitai cache stats --json                   # JSON 输出（现有）
markitai cache stats -v --json                # JSON 详细输出
```

#### 输出格式

**`markitai cache stats -v`** (详细模式):

```
Cache Statistics
Enabled: True

Global Cache
  Path: /home/user/.markitai/cache.db
  Entries: 42
  Size: 1.5 MB / 1024.0 MB

  By Model:
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
  ┃ Model                          ┃ Entries ┃ Size     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
  │ gemini/gemini-2.5-flash-lite   │ 20      │ 0.8 MB   │
  │ openai/gpt-5.2                 │ 15      │ 0.5 MB   │
  │ deepseek/deepseek-chat         │ 7       │ 0.2 MB   │
  └────────────────────────────────┴─────────┴──────────┘

  Recent Entries:
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Key                      ┃ Model                      ┃ Size   ┃ Preview                           ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ a1b2c3d4e5f6...          │ gemini/gemini-2.5-flash    │ 2.1 KB │ image: Colorful bar chart...      │
  │ b2c3d4e5f6a7...          │ openai/gpt-5.2             │ 4.5 KB │ frontmatter: Lorem ipsum doc...   │
  │ c3d4e5f6a7b8...          │ deepseek/deepseek-chat     │ 1.8 KB │ text: # Document Title\n\nThis... │
  └──────────────────────────┴────────────────────────────┴────────┴───────────────────────────────────┘
```

#### JSON 输出格式

**`markitai cache stats -v --json`**:

```json
{
  "enabled": true,
  "project": null,
  "global": {
    "db_path": "/home/user/.markitai/cache.db",
    "count": 42,
    "size_bytes": 1572864,
    "size_mb": 1.5,
    "max_size_mb": 1024.0,
    "by_model": {
      "gemini/gemini-2.5-flash-lite": {
        "count": 20,
        "size_bytes": 838860,
        "size_mb": 0.8
      },
      "openai/gpt-5.2": {
        "count": 15,
        "size_bytes": 524288,
        "size_mb": 0.5
      }
    },
    "entries": [
      {
        "key": "a1b2c3d4e5f6...",
        "model": "gemini/gemini-2.5-flash-lite",
        "size_bytes": 2150,
        "created_at": "2026-01-21T11:30:00+08:00",
        "accessed_at": "2026-01-21T13:45:00+08:00",
        "preview": "image: Colorful bar chart..."
      }
    ]
  }
}
```

### 实现要点

1. **Value Preview 解析**:
   - `caption` 存在 → `image: {caption[:40]}...`
   - `title` 存在 → `frontmatter: {title[:40]}...`
   - 其他 → `text: {value[:40]}...`

2. **性能考虑**:
   - `list_entries()` 使用 `LIMIT` 避免大量数据
   - 不加载完整 value，只取 `substr(value, 1, 200)`

---

## 已完成

### URL 增强功能 (2026-01-21)

- [x] **URL 图片下载与 `--alt`/`--desc` 支持**
  - 新增 `download_url_images()` 函数 (`image.py`)
  - 支持并发下载（默认 5 个并发）
  - 自动解析相对 URL
  - 失败时跳过并警告，保留原始 URL
  - 复用现有图片处理流程（质量、格式转换）

- [x] **URL 批量处理支持**
  - 新增 `urls.py` 模块：URL 列表解析
  - 支持 `.urls` 文件扩展名自动识别（无需显式参数）
  - 支持纯文本格式（一行一个 URL，`#` 注释）
  - 支持 JSON 格式（数组或对象数组）
  - 新增 `process_url_batch()` 函数
  - 批处理目录自动检测 `*.urls` 文件
  - 新增 19 个测试用例 (`tests/integration/test_url.py`)

### 20260121-需求规划-1

- [x] **需求点1: 单文件终端输出优化** (2026-01-21)
  - 单文件模式默认不打印日志，直接输出转换结果到 stdout
  - `--verbose` 时先打印日志再输出结果
  - 批处理行为保持不变
  - 涉及: `cli.py` (`setup_logging()`, `process_single_file()`)

- [x] **需求点3: URL 转换支持** (2026-01-21)
  - 新增 `markitai <url>` 命令支持
  - 利用 markitdown 原生 URL 转换能力
  - 支持 http/https 协议
  - 支持 `--llm` 参数进行 LLM 增强
  - 不支持 `--alt`/`--desc`/`--screenshot`/`--ocr` (markitdown 不下载图片)
  - 涉及: `cli.py` (`is_url()`, `url_to_filename()`, `process_url()`)

- [x] **需求点4: 版本号更新** (2026-01-21 ~ 2026-01-22)
  - 0.2.0 → 0.2.5 (2026-01-21): URL 转换、单文件输出优化
  - 0.2.5 → 0.3.0 (2026-01-22): 性能优化、缓存增强、workflow/core 重构

### 代码质量改进 (2026-01-21)

- [x] **URL 文件名跨平台兼容**
  - 新增 `_sanitize_filename()` 处理 Windows 非法字符
  - 移除 `< > : " / \ | ? *` 等字符
  - 限制文件名长度

- [x] **URL 错误处理优化**
  - 友好的错误提示信息 (SSL/连接/代理错误)
  - 区分不同类型的网络错误

- [x] **新增测试用例**
  - `TestUrlHelpers`: URL 检测和文件名生成测试
  - `TestSingleFileOutput`: 单文件输出行为测试
  - `TestUrlConversion`: URL 转换测试
  - 共新增 12 个测试用例

---

## 性能优化任务

来源: `20260122-性能分析`
创建: 2026-01-22

### 背景

通过对代码库的深度分析，发现了多个性能瓶颈和优化机会。本章节记录完整的分析结果和实施方案。

### 当前架构概述

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           markitai CLI 批处理流程                       │
├─────────────────────────────────────────────────────────────────────────┤
│  app() [cli.py:704]                                                     │
│    │                                                                    │
│    ├─► discover_files() [batch.py:672]    ─► 文件列表                   │
│    ├─► find_url_list_files() [cli.py:3424] ─► URL 列表                  │
│    │                                                                    │
│    ├─► create shared LLMProcessor [cli.py:3516-3528]                    │
│    ├─► create unified Semaphore [cli.py:3972]                           │
│    │                                                                    │
│    └─► asyncio.gather(*all_tasks) [cli.py:4104]                         │
│          │                                                              │
│          ├─► process_file_with_state()                                  │
│          │     └─► async with semaphore                                 │
│          │           └─► convert_document_core() [core.py:575]          │
│          │                 └─► run_in_converter_thread()                │
│          │                 └─► process_with_llm()                       │
│          │                                                              │
│          └─► process_url_with_state()                                   │
│                └─► async with semaphore                                 │
│                      └─► fetch_url() [fetch.py:700]                     │
│                      └─► process_with_llm()                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 发现的性能问题

#### 高优先级 (P1-P5)

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| **P1** | URL 和文件共享同一个 Semaphore | `cli.py:3972` | 高延迟 URL (如 x.com ~60s) 阻塞文件处理槽位 |
| **P2** | `workflow/core.py` 每次转换创建新 ThreadPoolExecutor | `core.py:114-124` | 线程创建销毁开销，未复用全局 executor |
| **P3** | Browser 抓取每个 URL 启动 5 个子进程 | `fetch.py:442-530` | 巨大的进程创建开销 |
| **P4** | FetchCache 每次操作创建新 SQLite 连接 | `fetch.py:120-126` | 连接创建开销 |
| **P5** | EMF/WMF 转 PNG 缺少 LibreOffice 隔离配置 | `image.py:172-182` | 并发执行时可能冲突 |

#### 中优先级 (M1-M5)

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| **M1** | PDF 页面渲染顺序执行 | `pdf.py:593-630` | 大 PDF 处理时间长 |
| **M2** | Vision LLM 和嵌入图片分析顺序执行 | `core.py:640-647` | 可并行节省一次 LLM 往返 |
| **M3** | Jina 每次创建新 httpx 客户端 | `fetch.py:665` | 无连接复用 |
| **M4** | MarkItDown 每次创建新实例 | `fetch.py:386` | 对象创建开销 |
| **M5** | 图片批量分析缓存检查顺序执行 | `llm.py:2509-2524` | I/O 等待 |

#### 低优先级 (L1-L3)

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| **L1** | 图片读取使用同步 `read_bytes()` | `llm.py:1637` | 阻塞事件循环 |
| **L2** | FetchCache LRU 逐个驱逐 | `fetch.py:212-222` | 效率低 |
| **L3** | 无 URL 抓取重试机制 | `fetch.py:788-855` | 临时失败无法恢复 |

### 问题详细分析

#### P1: 共享 Semaphore 问题

```python
# cli.py:3972 - 当前实现
semaphore = asyncio.Semaphore(cfg.batch.concurrency)  # 默认 15

# 文件和 URL 共享同一个信号量
async def process_file_with_state(file_path):
    async with semaphore:  # cli.py:4044-4046
        ...

async def process_url_with_state(url, source_file, custom_name):
    async with semaphore:  # cli.py:3993-3995
        ...
```

**问题**: 当 x.com 这样的 SPA 网站需要 60+ 秒时，会长时间占用信号量槽位，阻塞本地文件处理。

#### P2: ThreadPoolExecutor 重复创建

```python
# workflow/core.py:114-124 - 问题代码
async def run_in_converter_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:  # 每次创建新的！
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

# cli.py:208-222 - 已有全局 executor 但未被 core.py 使用
_CONVERTER_EXECUTOR: ThreadPoolExecutor | None = None
_CONVERTER_MAX_WORKERS = min(os.cpu_count() or 4, 8)
```

#### P3: Browser 抓取的 5 步顺序调用

```python
# fetch.py:442-530 - 每个 URL 需要 5 个子进程调用
async def fetch_with_browser(url, ...):
    # 1. agent-browser open <url>
    await _run_browser_command(["open", url], ...)  # L442-456

    # 2. agent-browser wait --load domcontentloaded
    await _run_browser_command(["wait", "--load", wait_for], ...)  # L459-467

    # 3. agent-browser wait 2000 (额外等待 JS)
    await _run_browser_command(["wait", str(extra_wait_ms)], ...)  # L470-478

    # 4. agent-browser snapshot -c --json
    await _run_browser_command(["snapshot", "-c", "--json"], ...)  # L482-506

    # 5. agent-browser get title + get url (两次调用)
    await _run_browser_command(["get", "title"], ...)  # L510-530
```

#### M1: PDF 页面渲染顺序执行

```python
# pdf.py:593-630 - 顺序渲染
def _render_pages_for_llm(self, doc, output_dir, dpi=150):
    page_images = []
    for page_num in range(len(doc)):  # 顺序处理每一页
        page = doc[page_num]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        # ... 保存图像
    return page_images

# 对比: _convert_with_ocr() 已实现并行处理 (pdf.py:443-462)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_page_with_screenshot, i): i ...}
```

#### M2: Vision LLM 和嵌入图片分析顺序执行

```python
# workflow/core.py:640-647 - 顺序执行
if has_page_images:
    result = await process_with_vision_llm(ctx)
    if not result.success:
        return result
    result = await analyze_embedded_images(ctx)  # 这里可以并行化
```

### 实施方案

#### 任务 1: 复用全局 ThreadPoolExecutor (P2) - 低风险

**目标**: 修改 `workflow/core.py` 复用 `cli.py` 中的全局 converter executor

**方案**: 在 `utils/` 创建共享的 executor 管理模块

```python
# utils/executor.py (新文件)
from concurrent.futures import ThreadPoolExecutor
import os

_CONVERTER_EXECUTOR: ThreadPoolExecutor | None = None
_CONVERTER_MAX_WORKERS = min(os.cpu_count() or 4, 8)

def get_converter_executor() -> ThreadPoolExecutor:
    """Get or create the shared converter thread pool executor."""
    global _CONVERTER_EXECUTOR
    if _CONVERTER_EXECUTOR is None:
        _CONVERTER_EXECUTOR = ThreadPoolExecutor(
            max_workers=_CONVERTER_MAX_WORKERS,
            thread_name_prefix="markitai-converter",
        )
    return _CONVERTER_EXECUTOR

async def run_in_converter_thread(func, *args, **kwargs):
    """Run a function in the shared converter thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    executor = get_converter_executor()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
```

**修改文件**:
- 新建 `utils/executor.py`
- 修改 `workflow/core.py`: 导入并使用新的 `run_in_converter_thread`
- 修改 `cli.py`: 导入并使用新模块，删除重复代码

**预期收益**: 减少线程创建销毁开销

---

#### 任务 2: LibreOffice 隔离配置修复 (P5) - 低风险

**目标**: 为 `image.py` 中的 EMF/WMF 转换添加 LibreOffice 隔离配置

**当前问题代码**:
```python
# image.py:172-182 - 缺少隔离配置！
cmd = [
    soffice,
    "--headless",
    "--convert-to",
    "png",
    "--outdir",
    str(temp_path),
    str(temp_in),
]
subprocess.run(cmd, capture_output=True, timeout=30)
```

**修复方案**:
```python
# image.py - 添加隔离配置
with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile_dir:
    profile_url = Path(profile_dir).as_uri()
    cmd = [
        soffice,
        "--headless",
        f"-env:UserInstallation={profile_url}",  # 添加隔离配置
        "--convert-to",
        "png",
        "--outdir",
        str(temp_path),
        str(temp_in),
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
```

**修改文件**:
- `converter/image.py`: 添加 `-env:UserInstallation` 参数

**预期收益**: 避免并发 LibreOffice 调用冲突

---

#### 任务 3: 分离 URL 和文件 Semaphore (P1) - 中风险

**目标**: URL 处理和文件处理使用独立的信号量，避免高延迟 URL 阻塞文件处理

**配置扩展**:
```python
# config.py - BatchConfig 扩展
class BatchConfig(BaseModel):
    concurrency: int = 15              # 文件处理并发数 (保持兼容)
    url_concurrency: int | None = None # URL 处理并发数，None 表示使用独立默认值 3
```

**CLI 实现**:
```python
# cli.py - process_batch() 修改
# 当前 (cli.py:3972)
semaphore = asyncio.Semaphore(cfg.batch.concurrency)

# 修改后
file_semaphore = asyncio.Semaphore(cfg.batch.concurrency)
url_concurrency = cfg.batch.url_concurrency if cfg.batch.url_concurrency else 3
url_semaphore = asyncio.Semaphore(url_concurrency)

# 文件处理使用 file_semaphore
async def process_file_with_state(file_path):
    async with file_semaphore:
        ...

# URL 处理使用 url_semaphore
async def process_url_with_state(url, source_file, custom_name):
    async with url_semaphore:
        ...
```

**新增 CLI 参数**:
```python
@click.option(
    "--url-concurrency",
    type=int,
    default=None,
    help="URL processing concurrency (default: 3). Separate from file concurrency.",
)
```

**修改文件**:
- `config.py`: 扩展 `BatchConfig`
- `cli.py`: 分离信号量逻辑，添加 `--url-concurrency` 参数

**预期收益**: 高延迟 URL 不再阻塞本地文件处理

---

#### 任务 4: 并行 Vision LLM 和嵌入图片分析 (M2) - 低风险

**目标**: Vision LLM 处理和嵌入图片分析可以并行执行

**当前代码**:
```python
# workflow/core.py:640-647
if has_page_images:
    result = await process_with_vision_llm(ctx)
    if not result.success:
        return result
    result = await analyze_embedded_images(ctx)
```

**修改方案**:
```python
# workflow/core.py - 并行执行
if has_page_images:
    # 并行执行 Vision LLM 和嵌入图片分析
    vision_task = asyncio.create_task(process_with_vision_llm(ctx))
    embed_task = asyncio.create_task(analyze_embedded_images(ctx))

    vision_result, embed_result = await asyncio.gather(
        vision_task, embed_task, return_exceptions=True
    )

    # 检查结果
    if isinstance(vision_result, Exception):
        return ConversionStepResult(success=False, error=str(vision_result))
    if not vision_result.success:
        return vision_result
    if isinstance(embed_result, Exception):
        logger.warning(f"Embedded image analysis failed: {embed_result}")
    elif not embed_result.success:
        logger.warning(f"Embedded image analysis failed: {embed_result.error}")
```

**修改文件**:
- `workflow/core.py`: 修改 `convert_document_core()` 中的执行逻辑

**预期收益**: Vision 模式下节省一次完整 LLM 处理时间 (2-5 秒)

---

#### 任务 5: Browser 抓取优化 (P3) - 中风险

**目标**: 减少 Browser 抓取的子进程调用次数

**方案 A: 合并 get title/url 调用**

分析 `agent-browser snapshot --json` 输出，如果已包含 title/url 信息，则省略后续调用。

```python
# fetch.py - fetch_with_browser() 优化
async def fetch_with_browser(url, ...):
    # 1. open
    await _run_browser_command(["open", url], ...)

    # 2. wait --load
    await _run_browser_command(["wait", "--load", wait_for], ...)

    # 3. wait extra
    if extra_wait_ms > 0:
        await _run_browser_command(["wait", str(extra_wait_ms)], ...)

    # 4. snapshot (获取 markdown + 元数据)
    snapshot_result = await _run_browser_command(["snapshot", "-c", "--json"], ...)
    snapshot_data = json.loads(snapshot_result.stdout)

    # 从 snapshot 提取 title 和 url，避免额外调用
    title = snapshot_data.get("title", "")
    final_url = snapshot_data.get("url", url)
    markdown = snapshot_data.get("markdown", "")

    # 只在 snapshot 没有提供时才调用 get
    if not title:
        title_result = await _run_browser_command(["get", "title"], ...)
        title = title_result.stdout.strip()
```

**方案 B: 批量 URL 复用浏览器会话** (未来优化)

```python
# 新增 fetch_urls_batch_with_browser() 函数
async def fetch_urls_batch_with_browser(urls: list[str], ...):
    session = f"markitai-batch-{uuid.uuid4().hex[:8]}"
    results = []
    try:
        for url in urls:
            result = await fetch_with_browser(url, session=session, ...)
            results.append(result)
    finally:
        await _run_browser_command(["close"], session=session)
    return results
```

**修改文件**:
- `fetch.py`: 优化 `fetch_with_browser()` 实现

**预期收益**: 减少 20-40% 的子进程调用

---

#### 任务 6: 并行 PDF 页面渲染 (M1) - 中风险

**目标**: 将 `_render_pages_for_llm()` 改为并行渲染

**当前代码**:
```python
# pdf.py:593-630
def _render_pages_for_llm(self, doc, output_dir, dpi=150):
    page_images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        # ...
```

**修改方案**:
```python
# pdf.py - 并行渲染
def _render_pages_for_llm(self, doc, output_dir, dpi=150, max_workers=None):
    import pymupdf
    from concurrent.futures import ThreadPoolExecutor

    total_pages = len(doc)
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, total_pages, 4)

    def render_page(page_num):
        # 每个线程打开自己的文档副本 (pymupdf 线程安全要求)
        thread_doc = pymupdf.open(doc.name)
        page = thread_doc[page_num]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        # ... 保存图像
        thread_doc.close()
        return (page_num, image_path)

    page_images = [None] * total_pages
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(render_page, i): i for i in range(total_pages)}
        for future in as_completed(futures):
            page_num, image_path = future.result()
            page_images[page_num] = image_path

    return [p for p in page_images if p]
```

**修改文件**:
- `converter/pdf.py`: 修改 `_render_pages_for_llm()` 方法

**预期收益**: 大 PDF 处理时间显著减少

---

#### 任务 7: SQLite 连接复用 (P4) - 低风险

**目标**: FetchCache 复用 SQLite 连接而非每次创建

**当前代码**:
```python
# fetch.py:120-126
def _get_connection(self) -> Any:
    conn = sqlite3.connect(str(self._db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
```

**修改方案**:
```python
# fetch.py - 连接复用
class FetchCache:
    def __init__(self, db_path, max_size_bytes):
        self._db_path = db_path
        self._max_size_bytes = max_size_bytes
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            with self._lock:
                if self._connection is None:  # Double-check
                    self._connection = sqlite3.connect(
                        str(self._db_path),
                        timeout=30.0,
                        check_same_thread=False  # 允许跨线程使用
                    )
                    self._connection.execute("PRAGMA journal_mode=WAL")
                    self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
```

**修改文件**:
- `fetch.py`: 修改 `FetchCache` 类

**预期收益**: 减少数据库连接创建开销

---

#### 任务 8: Jina/MarkItDown 实例复用 (M3, M4) - 低风险

**目标**: 复用 httpx 客户端和 MarkItDown 实例

**Jina 优化**:
```python
# fetch.py - 模块级共享客户端
_jina_client: httpx.AsyncClient | None = None

def get_jina_client(timeout: int = 30) -> httpx.AsyncClient:
    global _jina_client
    if _jina_client is None:
        _jina_client = httpx.AsyncClient(timeout=timeout)
    return _jina_client

async def fetch_with_jina(url, api_key, timeout):
    client = get_jina_client(timeout)
    # ... 使用共享客户端
```

**MarkItDown 优化**:
```python
# fetch.py - 模块级共享实例
_markitdown_instance: MarkItDown | None = None

def get_markitdown() -> MarkItDown:
    global _markitdown_instance
    if _markitdown_instance is None:
        _markitdown_instance = MarkItDown()
    return _markitdown_instance
```

**修改文件**:
- `fetch.py`: 添加实例复用逻辑

**预期收益**: 减少对象创建开销

---

### 实施顺序

| 阶段 | 任务 | 复杂度 | 风险 | 预期收益 |
|------|------|--------|------|----------|
| **阶段 1** | 任务 1: ThreadPoolExecutor 复用 | 低 | 低 | 中 |
| **阶段 1** | 任务 2: LibreOffice 隔离修复 | 低 | 低 | 低(稳定性) |
| **阶段 2** | 任务 4: 并行 Vision+嵌入图片 | 低 | 低 | 中 |
| **阶段 2** | 任务 7: SQLite 连接复用 | 低 | 低 | 中 |
| **阶段 2** | 任务 8: Jina/MarkItDown 复用 | 低 | 低 | 低 |
| **阶段 3** | 任务 3: 分离 Semaphore | 中 | 中 | 高 |
| **阶段 3** | 任务 5: Browser 抓取优化 | 中 | 中 | 中 |
| **阶段 4** | 任务 6: 并行 PDF 渲染 | 中 | 中 | 高(大PDF) |

### 测试验证

#### 自动化测试

```bash
# 代码质量检查
cd packages/markitai
uv run ruff check src tests
uv run pyright src tests

# 单元测试和集成测试
uv run pytest -v
```

#### 真实场景测试

```bash
# 单 URL 测试 (含高延迟 x.com)
uv run markitai https://x.com/Gorden_Sun/status/2013925532925317459 \
    --preset standard --no-cache -o output-single-perf-1 --verbose

# 批处理测试 (混合文件和 URL)
uv run markitai packages/markitai/tests/fixtures \
    --preset rich -o ./output-batch-perf-1 --verbose
```

#### 性能对比指标

| 指标 | 优化前基准 | 优化后目标 |
|------|-----------|-----------|
| 单 URL (x.com) | ~3s (缓存) | 无显著变化 |
| 批处理 (7 文件 + 3 URL) | ~105s | < 90s |
| 大 PDF (100+ 页) 渲染 | 待测量 | -30% |
| 内存峰值 | 待测量 | 无显著增加 |

### 实施状态 (2026-01-22)

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| **阶段 1** | 任务 1: ThreadPoolExecutor 复用 (P2) | ✅ 完成 | 新建 `utils/executor.py` |
| **阶段 1** | 任务 2: LibreOffice 隔离修复 (P5) | ✅ 完成 | `image.py` 添加 `-env:UserInstallation` |
| **阶段 2** | 任务 4: 并行 Vision+嵌入图片 (M2) | ✅ 完成 | `workflow/core.py` 使用 `asyncio.gather()` |
| **阶段 2** | 任务 7: SQLite 连接复用 (P4) | ✅ 完成 | `fetch.py` FetchCache 连接复用 |
| **阶段 2** | 任务 8: Jina/MarkItDown 复用 (M3, M4) | ✅ 完成 | `fetch.py` 添加共享实例 |
| **阶段 3** | 任务 3: 分离 URL/文件 Semaphore (P1) | ✅ 完成 | `config.py` 新增 `url_concurrency` |
| **阶段 3** | 任务 5: Browser 抓取优化 (P3) | ✅ 完成 | 并行获取 title 和 URL |
| **阶段 4** | 任务 6: 并行 PDF 渲染 (M1) | ✅ 完成 | `pdf.py` 使用 ThreadPoolExecutor |

#### 修改文件汇总

| 文件 | 变更类型 | 描述 |
|------|----------|------|
| `utils/executor.py` | 新建 | 共享 ThreadPoolExecutor 模块 |
| `utils/__init__.py` | 修改 | 导出 executor 函数 |
| `workflow/core.py` | 修改 | 导入共享 executor，并行 Vision 处理 |
| `cli.py` | 修改 | 导入共享 executor，分离 semaphore，新增 `--url-concurrency` |
| `fetch.py` | 修改 | SQLite 连接复用，Jina/MarkItDown 实例复用，并行获取 title/URL |
| `image.py` | 修改 | LibreOffice 隔离配置 |
| `config.py` | 修改 | 新增 `url_concurrency` 配置 |
| `constants.py` | 修改 | 新增 `DEFAULT_URL_CONCURRENCY` |
| `converter/pdf.py` | 修改 | 并行页面渲染 |
| `pyproject.toml` | 修改 | 添加 UP047 到 ruff ignore 列表 |

#### 测试验证

```bash
# 代码质量检查 - 通过
cd packages/markitai
uv run ruff check src tests      # All checks passed!
uv run pyright src tests         # 0 errors, 0 warnings

# 单元测试 - 全部通过
uv run pytest -v                 # 541 tests passed
```

### 回滚方案

所有优化都保持向后兼容:

1. **配置兼容**: 新配置项有默认值，不影响现有配置
2. **环境变量回退**: 可通过环境变量禁用新优化
   - `MARKITAI_DISABLE_EXECUTOR_SHARING=1` - 禁用 executor 共享
   - `MARKITAI_DISABLE_PARALLEL_VISION=1` - 禁用并行 Vision 处理
3. **代码保留**: 旧实现标记为 `_legacy` 后缀保留一个版本

---

## 待办任务

来源: 2026-01-22 代码审查
创建: 2026-01-22
更新: 2026-01-22

### 已完成 (2026-01-22)

- [x] **任务 1: 更新 CLI 文档 - 新增参数**
  - 文件: `website/guide/cli.md`, `website/zh/guide/cli.md`
  - 在 "URL Options" 部分添加以下参数文档:
    - `--url-concurrency <n>`: URL 并发数量控制（独立于 `--batch-concurrency`，默认 3）
    - `--agent-browser`: 强制使用浏览器渲染 URL（适用于 SPA 网站）
    - `--jina`: 强制使用 Jina Reader API
  - 说明 `--agent-browser` 和 `--jina` 互斥
  - 更新 `--batch-concurrency` 说明，添加与 `--url-concurrency` 的关系提示

- [x] **任务 2: 添加 `utils/executor.py` 单元测试**
  - 新建: `tests/unit/test_executor.py` (15 个测试用例)
  - 测试覆盖:
    - `get_converter_executor()` 线程安全初始化（双重检查锁定）
    - `run_in_converter_thread()` 异步执行、参数传递、异常传播
    - `shutdown_converter_executor()` 清理逻辑
    - 多线程并发调用 `get_converter_executor()` 返回同一实例
    - 并发执行性能验证

- [x] **任务 3: 更新 tasks.md 版本号记录**
  - 更新版本历史: 0.2.0 → 0.2.5 → 0.3.0
  - 添加 0.3.0 版本变更说明

- [x] **任务 4: 优化 report.json 字段命名**
  - **问题**: `summary.total` 歧义 - 不包含 `.urls` 文件，但名称暗示"总数"
  - **解决方案**: 统一使用更精确的命名
  - **修改内容**:
    - `summary.total` → `summary.total_documents` (待转换的文档数)
    - `summary.completed` → `summary.completed_documents`
    - `summary.failed` → `summary.failed_documents`
    - `summary.pending` → `summary.pending_documents`
    - `local_files` → `documents` (报告顶层)
    - `url_files` → `url_sources` (报告顶层)
  - **兼容性**: `json_order.py` 自动转换旧字段名到新字段名
  - **文件修改**:
    - `batch.py`: `_compute_summary()` 字段名
    - `json_order.py`: 字段排序定义、转换逻辑
    - `cli.py`: 单文件/URL 报告生成
    - 测试文件: 更新断言

### 已完成 (2026-01-22)

- [x] **任务 5: 清理旧版 CLI 实现** (2026-01-22)
  - 移除 `process_single_file()` 旧版实现 (~390 行)
  - 移除 `process_file()` 内部的 v1 逻辑 (~370 行)
  - 移除 `MARKITAI_USE_LEGACY_CLI` 环境变量支持
  - 更新相关测试 (移除 `TestLegacyFallback` 类)
  - 清理未使用的导入
  - **实际收益**: cli.py 从 ~4100 行减少到 ~3333 行 (~767 行, ~18.7%)

- [x] **任务 6: 任务文档结构优化** (2026-01-22)
  - 将已完成任务统一移动到 "已完成" 章节
  - 为待办任务添加优先级标注 (P1/P2/P3)
  - 明确任务状态标记

---

## 字段命名重构任务

来源: 代码审查 (2026-01-22)
实施: 2026-01-22

### 已完成 (2026-01-22)

- [x] **统一内部命名: url_source_files → url_sources**
  - `batch.py`: `BatchState.url_source_files` → `BatchState.url_sources`
  - `cli.py`: 变量 `url_source_file_set` → `url_sources_set`
  - `json_order.py`: `SUMMARY_FIELD_ORDER` 中 `url_source_files` → `url_sources`
  - 保留向后兼容: `from_dict()` 仍接受旧字段名

- [x] **report.json 字段重命名** (之前已完成)
  - `summary.total` → `summary.total_documents`
  - `local_files` → `documents`
  - `url_files` → `url_sources`

- [x] **state.json 字段重命名** (之前已完成)
  - `files` → `documents`
  - `url_source_files` → `url_sources`

- [x] **assets.json → images.json 重命名** (之前已完成)
  - 文件名: `assets.json` → `images.json`
  - `assets` 数组 → `images` 数组
  - `asset` 字段 → `path` 字段

- [x] **website 项目依赖文档补充** (2026-01-22)
  - 更新 `website/guide/getting-started.md`
  - 更新 `website/zh/guide/getting-started.md`
  - 添加可选依赖表格 (Node.js, agent-browser, Jina, LLM API)

---

## URL 截图功能

来源: 功能规划 (2026-01-22)
实施: 2026-01-22
状态: ✅ 已完成

### 需求背景

当前 `--screenshot` 选项仅支持 PDF/PPTX 本地文档，将页面/幻灯片渲染为 JPEG 图片。对于 URL 转换，该选项被标记为"不支持"并显示警告。

通过集成 agent-browser 的截图功能，可以为 URL 转换添加网页截图支持，捕获完整网页的视觉布局，便于 LLM 分析和存档。

### agent-browser 截图能力

```bash
# 基础命令
agent-browser screenshot [path]        # 截取当前视口
agent-browser screenshot --full [path] # 截取完整页面（含滚动区域）
agent-browser set viewport 1920 1080   # 设置视口大小

# 输出选项
# - 指定路径：直接保存为 PNG/JPG
# - 无路径：输出 base64 编码数据
# - --json：JSON 格式输出

# 多 session 支持（并行处理）
agent-browser --session url1 open https://example1.com
agent-browser --session url2 open https://example2.com
```

### 当前实现分析

#### PDF/PPTX 截图流程

```
converter/pdf.py:187-231
1. 渲染每页为 JPEG (DPI=150)
2. 保存到 output/screenshots/
3. 在 markdown 末尾以注释引用：<!-- ![Page 1](screenshots/doc.page0001.jpg) -->
4. 返回 metadata["page_images"] 供 LLM 处理
```

#### URL 处理流程 (fetch.py)

```
fetch_url()
  ├─ AUTO: 检测是否需要 JS → 选择策略
  ├─ STATIC: markitdown 直接抓取
  ├─ BROWSER: agent-browser open → wait → snapshot
  └─ JINA: Jina Reader API

返回 FetchResult:
  - content: markdown 内容
  - strategy_used: 实际使用的策略
  - title: 页面标题
  - url/final_url: URL 信息
  - metadata: 附加数据
  - cache_hit: 缓存命中标识
```

### 设计决策

| 决策点 | 结论 | 说明 |
|--------|------|------|
| 策略升级 | ✅ 自动升级 | `--screenshot` 启用时自动升级到 `browser` 策略 |
| 并发策略 | 内联截图 | 在 `fetch_with_browser()` 内顺序执行，利用多 URL 并行 |
| 视口配置 | 仅配置文件 | 不添加 CLI 参数，通过 `screenshot.*` 配置 |
| 截图格式 | JPEG + 压缩 | 使用 ImageProcessor 压缩，控制文件大小 |

### 并发策略详解

#### 方案分析

| 方案 | 描述 | 可行性 |
|------|------|--------|
| A: 内联截图 | 在同一 session 中顺序执行 open → snapshot → screenshot | ✅ 简单 |
| B: 分离截图任务 | 抓取完成后，将截图任务放入独立队列 | ⚠️ 需要保持 session 打开 |
| C: 并行抓取+截图 | 每个 URL 独立 session，多 URL 并行 | ✅ 当前已支持 |

#### 选择方案 A 的原因

1. **已有并行机制**: 多个 URL 通过 `url_semaphore` 并行，每个使用独立 session
2. **无需额外复杂性**: 截图在 `fetch_with_browser()` 内顺序执行
3. **资源可控**: `url_concurrency` (默认 3) 控制并行浏览器实例数
4. **性能足够**: 3 个 URL 并行时，单个增加 2s，总时间增加很小

```
时间轴示例:
URL1: [open→wait→snapshot→screenshot] ────────────────→ done
URL2:    [open→wait→snapshot→screenshot] ─────────────→ done
URL3:       [open→wait→snapshot→screenshot] ──────────→ done
            ↑ 并行执行，总时间 ≈ max(单个URL时间)
```

### 策略限制

| 策略 | 截图支持 | 处理方式 |
|------|----------|----------|
| `browser` | ✅ 完全支持 | agent-browser 原生支持 |
| `auto` | ✅ 自动升级 | 检测到 `--screenshot` 时升级到 browser |
| `static` | ⚠️ 自动升级 | 记录日志并升级到 browser |
| `jina` | ⚠️ 自动升级 | 记录日志并升级到 browser |

### 缓存策略

| 内容类型 | 缓存方式 | 说明 |
|----------|----------|------|
| 页面内容 | FetchCache | 已有实现，SQLite 存储 |
| 截图文件 | 文件存在性检查 | 简单缓存，截图存在则跳过 |

### 文件命名

```python
def _url_to_screenshot_filename(url: str) -> str:
    """Generate safe filename for URL screenshot.
    
    Examples:
        https://example.com/path → example.com_path.full.jpg
        https://x.com/user/status/123 → x.com_user_status_123.full.jpg
    """
```

### 性能影响

| 操作 | 耗时估计 | 备注 |
|------|----------|------|
| 设置视口 | ~100ms | 仅首次 |
| 全页面截图 | 500ms-3s | 取决于页面长度 |
| 图片压缩 | ~200ms | 使用 ImageProcessor |
| **总额外开销** | **1-4s/URL** | 在已有处理时间基础上 |

### 配置扩展

```python
# constants.py - 新增常量
DEFAULT_SCREENSHOT_VIEWPORT_WIDTH = 1920
DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT = 1080
DEFAULT_SCREENSHOT_QUALITY = 85
DEFAULT_SCREENSHOT_MAX_HEIGHT = 10000

# config.py - 扩展 ScreenshotConfig
class ScreenshotConfig(BaseModel):
    enabled: bool = False
    viewport_width: int = DEFAULT_SCREENSHOT_VIEWPORT_WIDTH   # 视口宽度
    viewport_height: int = DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT # 视口高度
    quality: int = DEFAULT_SCREENSHOT_QUALITY                 # JPEG 质量 (1-100)
    max_height: int = DEFAULT_SCREENSHOT_MAX_HEIGHT           # URL 截图最大高度
```

### 核心代码变更

#### 1. 扩展 FetchResult (fetch.py)

```python
@dataclass
class FetchResult:
    content: str
    strategy_used: str
    title: str | None = None
    url: str = ""
    final_url: str | None = None
    metadata: dict = field(default_factory=dict)
    cache_hit: bool = False
    screenshot_path: Path | None = None  # 新增
```

#### 2. 扩展 fetch_with_browser() (fetch.py)

```python
async def fetch_with_browser(
    url: str,
    command: str = "agent-browser",
    timeout: int = 30000,
    wait_for: str = "domcontentloaded",
    extra_wait_ms: int = 2000,
    session: str | None = None,
    # 新增参数
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
) -> FetchResult:
    """..."""
    
    # ... 现有逻辑 (open, wait, snapshot) ...
    
    # 新增：截图步骤
    screenshot_path = None
    if screenshot and screenshot_dir:
        try:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置视口（如果配置了）
            if screenshot_config:
                viewport_args = [
                    *base_args, "set", "viewport",
                    str(screenshot_config.viewport_width),
                    str(screenshot_config.viewport_height)
                ]
                await _run_browser_command(viewport_args)
            
            # 检查截图是否已存在（简单缓存）
            safe_filename = _url_to_screenshot_filename(url)
            screenshot_path = screenshot_dir / safe_filename
            
            if not screenshot_path.exists():
                # 捕获全页面截图
                screenshot_args = [*base_args, "screenshot", "--full", str(screenshot_path)]
                await _run_browser_command(screenshot_args)
                
                # 压缩截图（如果需要）
                if screenshot_config and screenshot_path.exists():
                    _compress_screenshot(screenshot_path, screenshot_config)
                
                logger.debug(f"Screenshot saved: {screenshot_path}")
            else:
                logger.debug(f"Screenshot exists, skipping: {screenshot_path}")
                
        except Exception as e:
            # 截图失败不阻断主流程
            logger.warning(f"Screenshot failed for {url}: {e}")
            screenshot_path = None
    
    return FetchResult(
        content=markdown_content,
        strategy_used="browser",
        title=title,
        url=url,
        final_url=final_url,
        metadata={"renderer": "agent-browser", "wait_for": wait_for},
        screenshot_path=screenshot_path,
    )
```

#### 3. 扩展 fetch_url() (fetch.py)

```python
async def fetch_url(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool = False,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    # 新增参数
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
) -> FetchResult:
    """..."""
    
    # 截图需要 browser 策略，自动升级
    if screenshot and strategy not in (FetchStrategy.BROWSER, FetchStrategy.AUTO):
        logger.info(f"Screenshot requires browser strategy, upgrading from {strategy.value}")
        strategy = FetchStrategy.BROWSER
    
    # ... 传递 screenshot 参数给 fetch_with_browser() ...
```

#### 4. 修改 CLI process_url() (cli.py)

```python
async def process_url(url, output_dir, cfg, ...):
    # 移除 --screenshot 警告，仅保留 --ocr
    unsupported_opts = []
    if cfg.ocr.enabled:
        unsupported_opts.append("--ocr")
    
    # 策略升级逻辑移到 fetch_url() 内部
    
    # 调用 fetch_url
    fetch_result = await fetch_url(
        url,
        fetch_strategy,
        cfg.fetch,
        screenshot=cfg.screenshot.enabled,
        screenshot_dir=ensure_screenshots_dir(output_dir) if cfg.screenshot.enabled else None,
        screenshot_config=cfg.screenshot if cfg.screenshot.enabled else None,
    )
    
    # 处理截图结果
    screenshots_count = 0
    if fetch_result.screenshot_path and fetch_result.screenshot_path.exists():
        rel_path = fetch_result.screenshot_path.relative_to(output_dir)
        markdown_for_llm += f"\n\n<!-- ![Full Page]({rel_path}) -->"
        screenshots_count = 1
```

### 实施任务清单 (2026-01-22 已完成)

#### 阶段 1: 核心实现 ✅

- [x] **任务 1.1: 扩展配置**
  - 文件: `constants.py`, `config.py`
  - 添加截图相关常量和 `ScreenshotConfig` 字段

- [x] **任务 1.2: 扩展 FetchResult**
  - 文件: `fetch.py`
  - 添加 `screenshot_path` 字段

- [x] **任务 1.3: 添加辅助函数**
  - 文件: `fetch.py`
  - 实现 `_url_to_screenshot_filename()`
  - 实现 `_compress_screenshot()`

- [x] **任务 1.4: 修改 fetch_with_browser()**
  - 文件: `fetch.py`
  - 添加截图参数和逻辑
  - 实现截图存在性检查（简单缓存）
  - 截图失败不阻断主流程

- [x] **任务 1.5: 修改 fetch_url()**
  - 文件: `fetch.py`
  - 添加截图参数
  - 实现策略自动升级

- [x] **任务 1.6: 修改 CLI process_url()**
  - 文件: `cli.py`
  - 移除 `--screenshot` 警告
  - 调用 `fetch_url()` 时传递截图参数
  - 处理截图结果，更新 markdown

- [x] **任务 1.7: 修改批处理 process_url()**
  - 文件: `cli.py`
  - 同样的截图处理逻辑
  - 更新 `ProcessResult.screenshots` 计数

#### 阶段 2: 测试 ✅

- [x] **任务 2.1: 单元测试**
  - 文件: `tests/unit/test_fetch.py`
  - 测试 `FetchResult.screenshot_path`
  - 测试 `_url_to_screenshot_filename()`
  - 新增 11 个测试用例

- [x] **任务 2.2: 集成测试**
  - 文件: `tests/unit/test_schema_sync.py` 更新
  - 文件: `config.schema.json` 更新

#### 阶段 3: 文档 ✅

- [x] **任务 3.1: 更新 CLI 文档**
  - 文件: `website/guide/cli.md`, `website/zh/guide/cli.md`
  - 移除 `--screenshot` "不支持 URL" 的说明
  - 添加 URL 截图行为说明

- [x] **任务 3.2: 更新配置文档**
  - 文件: `website/guide/configuration.md`, `website/zh/guide/configuration.md`
  - 添加 `screenshot.viewport_width/height` 说明
  - 添加 `screenshot.quality` 说明
  - 添加 `screenshot.max_height` 说明

### 实际工作量

| 阶段 | 任务 | 时间 |
|------|------|------|
| 核心实现 | 任务 1.1-1.7 | ~1.5 小时 |
| 测试 | 任务 2.1-2.2 | ~0.5 小时 |
| 文档 | 任务 3.1-3.2 | ~0.5 小时 |
| **总计** | | **~2.5 小时** |

### 未来优化方向

如果实际使用中发现性能瓶颈，可考虑：

| 优化 | 描述 | 复杂度 |
|------|------|--------|
| 增加 `url_concurrency` | 提高并行 URL 数 | 配置调整 |
| 分离内容/截图 semaphore | 截图使用独立并发控制 | 中等 |
| 超长页面分段截图 | 对超过 `max_height` 的页面分段 | 高 |

---

## Markitai → Markitai 品牌重命名

来源: PyPI 原包名已被占用，重命名为 `markitai`
实施: 2026-01-22
状态: ✅ 已完成

### 重命名决策

| 决策点 | 选择 | 说明 |
|--------|------|------|
| 新品牌名 | `markitai` | 强调 AI/LLM 特性 |
| Python 包名 | 全部改 | `from markitai import ...` |
| 目录结构 | 全部改 | `packages/markitai/src/markitai/` |
| 配置文件 | 直接改 | `markitai.json`, `~/.markitai/` |
| 环境变量 | 改前缀 | `MARKITAI_CONFIG` 等 |
| 内部占位符 | 全部改 | `__MARKITAI_*__` |
| GitHub 仓库 | 改名 | `Ynewtime/markitai` |
| 版本号 | 保持 | 0.3.0 |

### 改动范围统计

| 类别 | 估计文件数 | 估计改动点 |
|------|-----------|-----------|
| Python 包名/目录 | 2 目录 | 重命名 |
| Import 语句 | ~50 文件 | ~200 处 |
| CLI 命令/入口 | 1 文件 | 1 处 |
| PyPI 包名 | 1 文件 | 1 处 |
| 配置文件名 | 5+ 文件 | ~20 处 |
| 环境变量 | 5+ 文件 | ~15 处 |
| 内部占位符 | 3+ 文件 | ~25 处 |
| 文档内容 | 15+ 文件 | ~150 处 |
| 测试文件 | 10+ 文件 | ~60 处 |
| GitHub URL | 5+ 文件 | ~15 处 |
| **总计** | **~70 文件** | **~500+ 处** |

### 实施计划

#### 阶段 1: 目录结构重命名 ✅

- [x] **任务 1.1: 重命名包目录**
  ```
  packages/markit/ → packages/markitai/
  packages/markit/src/markit/ → packages/markitai/src/markitai/
  ```

#### 阶段 2: pyproject.toml 更新 ✅

- [x] **任务 2.1: 更新包配置**
  - `name = "markit"` → `name = "markitai"`
  - `markit = "markit.cli:app"` → `markitai = "markitai.cli:app"`
  - `packages = ["src/markit"]` → `packages = ["src/markitai"]`
  - `known-first-party = ["markit"]` → `known-first-party = ["markitai"]`

#### 阶段 3: Python 代码修改 ✅

- [x] **任务 3.1: Import 语句全局替换**
  - `from markit.` → `from markitai.`
  - `import markit` → `import markitai`

- [x] **任务 3.2: 配置文件名常量** (`constants.py`)
  - `DEFAULT_GLOBAL_CACHE_DIR = "~/.markit"` → `"~/.markitai"`
  - `DEFAULT_PROJECT_CACHE_DIR = ".markit"` → `".markitai"`
  - `DEFAULT_PROMPTS_DIR = "~/.markit/prompts"` → `"~/.markitai/prompts"`
  - `DEFAULT_LOG_DIR = "~/.markit/logs"` → `"~/.markitai/logs"`
  - `CONFIG_FILENAME = "markitai.json"` → `"markitai.json"`

- [x] **任务 3.3: CLI 帮助文本和示例** (`cli.py`)
  - 所有 `markit` 命令示例 → `markitai`

- [x] **任务 3.4: 环境变量**
  - `MARKIT_CONFIG` → `MARKITAI_CONFIG`
  - `MARKIT_LOG_DIR` → `MARKITAI_LOG_DIR`
  - `MARKIT_PROMPT_DIR` → `MARKITAI_PROMPT_DIR`

- [x] **任务 3.5: 内部占位符** (`llm.py`, `prompts/*.md`)
  - `__MARKIT_*__` → `__MARKITAI_*__`

- [x] **任务 3.6: YAML Frontmatter 字段**
  - `markit_processed` → `markitai_processed`

- [x] **任务 3.7: 报告/日志文件名**
  - `markit.*.report.json` → `markitai.*.report.json`
  - `markit_*.log` → `markitai_*.log`
  - `markit_preconv_` → `markitai_preconv_`

- [x] **任务 3.8: User-Agent** (`image.py`)
  - 更新 User-Agent 字符串中的品牌名和 URL

- [x] **任务 3.9: JSON Schema** (`config.schema.json`)
  - 更新描述和默认值

- [x] **任务 3.10: config.py**
  - 更新配置路径和注释

#### 阶段 4: 测试文件更新 ✅

- [x] **任务 4.1: 测试 Import 语句**
  - 所有 `tests/**/*.py` 文件

- [x] **任务 4.2: 测试断言和 Fixture**
  - 配置文件名、环境变量等断言

- [x] **任务 4.3: SKILL.md**
  - 命令示例更新

#### 阶段 5: 文档更新 ✅

- [x] **任务 5.1: README.md**
  - 品牌名、安装命令、示例

- [x] **任务 5.2: 网站文档** (`website/`)
  - `index.md`, `zh/index.md`
  - `guide/*.md`, `zh/guide/*.md`
  - `.vitepress/config.ts`

- [x] **任务 5.3: 技术文档** (`docs/`)
  - `spec.md`, `requirement.md`
  - `tasks.md` 标题

- [x] **任务 5.4: CHANGELOG.md**
  - 更新 GitHub URL（新版本）
  - 保留历史版本链接

#### 阶段 6: 验证 ✅

- [x] **任务 6.1: 依赖同步**
  - `uv sync`

- [x] **任务 6.2: 代码检查**
  - `ruff check src tests`
  - `pyright src`

- [x] **任务 6.3: 测试**
  - `pytest tests/ --tb=short -q`

- [x] **任务 6.4: 遗漏检查**
  - `grep -r "markitai" --include="*.py" --include="*.md" --include="*.json"`

#### 阶段 7: GitHub 操作（手动）

- [ ] **任务 7.1: 仓库重命名**
  - GitHub Settings → Repository name → `markitai`

### 注意事项

1. **向后兼容性**: 无。用户需要迁移配置文件
2. **历史记录**: CHANGELOG 中的旧版本链接保持指向原仓库
3. **迁移指南**: 在 CHANGELOG 0.3.0 中说明迁移步骤

---

## 代码质量与稳定性修复任务

来源: `markitai-overall-review-030.md` 深度分析
创建: 2026-01-23
状态: 待评审

### 问题概览

本任务基于 `docs/reference/markitai-overall-review-030.md` 的问题分析，分为三大类：

| 类别 | 问题数 | 风险等级 |
|------|--------|----------|
| 基础设施与工程问题 | 7 | 高-中 |
| 核心逻辑问题（影响 .llm.md 输出） | 5 | 高-中 |
| 性能瓶颈问题 | 5 | 中 |

### 优先级定义

- **P0（必须修复）**: 直接影响输出正确性的 Bug
- **P1（应该修复）**: 影响项目可维护性和合规性
- **P2（可以后续）**: 性能优化和改进项

---

### P0 - 核心逻辑问题（直接影响 .llm.md 输出）

#### P0-1: 图片引用错位 Bug（严重）

**问题位置**:
- `image.py:543-547` (`process_and_save`)
- `image.py:249-259` (`replace_base64_with_paths`)
- `workflow/core.py:233-287`

**问题描述**:
```python
# process_and_save() 处理图片时
for idx, image in enumerate(images, start=1):
    if self.is_duplicate(image_data):
        continue  # 跳过，但 markdown 中对应位置的 base64 仍存在
    if self.should_filter(width, height):
        continue  # 同上

# replace_base64_with_paths() 替换时
image_iter = iter(saved_images)  # saved_images 数量少于原始 base64 图片数
def replace_match(match):
    img = next(image_iter)  # 按顺序取下一张 —— 错位发生！
```

**影响**: 当图片被去重/过滤时，后续图片路径会前移，导致图片引用错位。

**修复方案**:

- [ ] **任务 P0-1.1: 重构图片处理返回结构**
  - 文件: `image.py`
  - `process_and_save()` 返回包含原始索引映射的结果
  - 新增 `ImageProcessResult` 数据类:
    ```python
    @dataclass
    class ProcessedImage:
        original_index: int
        saved_path: Path | None  # None 表示被过滤/去重
        skip_reason: str | None  # "duplicate" | "filtered" | None
    
    @dataclass
    class ImageBatchResult:
        processed: list[ProcessedImage]
        saved_count: int
        filtered_count: int
        deduplicated_count: int
    ```

- [ ] **任务 P0-1.2: 重构 replace_base64_with_paths()**
  - 文件: `image.py`
  - 基于原始索引匹配替换，而非顺序迭代
  - 被过滤的图片保留原 base64 或替换为空 alt 文本

- [ ] **任务 P0-1.3: 更新 workflow/core.py 调用点**
  - 文件: `workflow/core.py`
  - 适配新的返回结构

- [ ] **任务 P0-1.4: 添加测试用例**
  - 文件: `tests/unit/test_image.py`
  - 测试场景: 有去重、有过滤、混合场景

---

#### P0-2: 文档截断丢失内容

**问题位置**: `llm.py:3859-3864`

**问题描述**:
```python
# _process_document_combined() 中
markdown = self._smart_truncate(markdown, 8000)  # 硬截断！
```

超过 8000 字符的文档后半部分内容会静默丢失。

**修复方案**:

- [ ] **任务 P0-2.1: 添加截断警告**
  - 文件: `llm.py`
  - 当内容被截断时记录 warning 日志
  - 在 ProcessResult 中标记 `content_truncated: bool`

- [ ] **任务 P0-2.2: 增大默认截断阈值**
  - 文件: `constants.py`, `llm.py`
  - 将默认值从 8000 提高到 32000（考虑模型 context 限制）
  - 添加配置项 `llm.max_content_chars`

- [ ] **任务 P0-2.3: 实现分段处理策略（可选 - 复杂度高）**
  - 将长文档分段处理后合并
  - 预估工作量较大，可作为后续优化

---

#### P0-3: 短 slide 被图片替换（启发式逻辑风险）

**问题位置**: `llm.py:1950-1996`

**问题描述**:
```python
# _unprotect_content() 中的启发式逻辑
# 当 slide 段落很短（<10 字符）且无标题/图片时，会从 protected["images"] 抽图替换
```

可能将 "Agenda"、"Thanks"、"Q&A" 等合法短文本页错误替换为图片。

**修复方案**:

- [ ] **任务 P0-3.1: 移除短 slide 自动塞图逻辑**
  - 文件: `llm.py`
  - 完全移除 1950-1996 行的启发式逻辑
  - 添加日志记录移除前的行为，便于追踪问题

- [ ] **任务 P0-3.2: 排查其他启发式逻辑**
  - 搜索代码中类似的启发式处理
  - 记录并评估风险
  - 建议检查的关键词: `if len(`, `< 10`, `heuristic`, `auto`, `guess`

---

#### P0-4: Alt 文本回填竞态/超时

**问题位置**: `workflow/single.py:282-305`

**问题描述**:
```python
# 图片分析并行等待 .llm.md 出现，最多 120s 超时后放弃
# 用户不知道 alt 更新失败
```

**修复方案**:

- [ ] **任务 P0-4.1: 改为确定性串行流程**
  - 文件: `workflow/single.py`
  - 流程改为: 先完成 LLM 清理生成 .llm.md → 再执行图片分析回填
  - 移除轮询等待逻辑

- [ ] **任务 P0-4.2: 添加失败状态反馈**
  - 在 ProcessResult 中添加 `alt_update_status: str`
  - 可选值: "success" | "skipped" | "timeout" | "error"

---

#### P0-5: 截图清理规则过宽

**问题位置**: `llm.py:3970-3979`, `llm.py:3981-4124`

**问题描述**:
`_remove_uncommented_screenshots()` 可能误删用户原文中的合法 `screenshots/` 引用。

**修复方案**:

- [ ] **任务 P0-5.1: 使用更精确的匹配模式**
  - 文件: `llm.py`
  - 只匹配 markitai 生成的特定命名格式
  - 格式: `screenshots/{filename}.page{NNNN}.jpg`

- [ ] **任务 P0-5.2: 添加测试用例**
  - 测试用户原文包含 `screenshots/` 引用的场景

---

### P1 - 基础设施与工程问题

#### P1-1: 依赖不可复现

**问题位置**: `.gitignore:32`, `packages/markitai/pyproject.toml`

**问题描述**:
- `uv.lock` 被 .gitignore 忽略
- 依赖使用 `>=` 形式，无版本锁定

**修复方案**:

- [ ] **任务 P1-1.1: 将 uv.lock 纳入版本控制**
  - 从 `.gitignore` 移除 `uv.lock`
  - 运行 `uv lock` 生成锁文件
  - 提交 `uv.lock`

---

#### P1-2: 质量门禁缺失

**问题位置**: `.github/workflows/` (缺少 Python CI)

**修复方案**:

- [ ] **任务 P1-2.1: 创建 Python CI 工作流**
  - 新建: `.github/workflows/ci.yml`
  - 内容:
    - 触发: push/PR 到 main
    - 作业: ruff check, ruff format --check, pyright, pytest
    - Python 版本: 3.11, 3.12, 3.13
    - 缓存: uv cache

---

#### P1-3: 工具配置不一致

**问题位置**: 
- `packages/markitai/pyproject.toml` (ruff/pyright 配置)
- `.pre-commit-config.yaml` (从根运行)
- `pyproject.toml` (无 tool 配置)

**修复方案**:

- [ ] **任务 P1-3.1: 统一 ruff/pyright 配置到根 pyproject.toml**
  - 将 `[tool.ruff]` 和 `[tool.pyright]` 从 `packages/markitai/pyproject.toml` 移到根 `pyproject.toml`
  - 或在 pre-commit 中显式指定配置路径

---

#### P1-4: LICENSE 文件缺失

**问题位置**: 根目录

**修复方案**:

- [ ] **任务 P1-4.1: 创建 LICENSE 文件**
  - 新建: `LICENSE` (MIT 许可证)
  - 更新 `packages/markitai/pyproject.toml` 添加 license 字段

---

#### P1-5: 文档中的示例 API Key

**问题位置**: `docs/reference/litellm*.md`, `docs/spec.md`

**修复方案**:

- [ ] **任务 P1-5.1: 替换示例 API Key**
  - 将 `sk-xxxx`, `sk-1234` 等替换为 `YOUR_API_KEY_HERE`
  - 涉及文件:
    - `docs/spec.md`
    - `docs/reference/litellm.md`
    - `docs/reference/litellm_batches.md`

---

#### P1-6: 跨平台与 CI 稳定性

**问题位置**: `website/package.json`, `.github/workflows/deploy-website.yml`

**修复方案**:

- [ ] **任务 P1-6.1: 修复跨平台 cp 命令**
  - 文件: `website/package.json`
  - 方案 A: 使用 `shx cp`（需要添加 shx 依赖）
  - 方案 B: 使用 Node.js 脚本替代

- [ ] **任务 P1-6.2: 升级 VitePress 到稳定版本**
  - 文件: `website/package.json`
  - 从 `^2.0.0-alpha.15` 升级到最新稳定版

- [ ] **任务 P1-6.3: CI 使用冻结安装**
  - 文件: `.github/workflows/deploy-website.yml`
  - 将 `pnpm install` 改为 `pnpm install --frozen-lockfile`

---

#### P1-7: Python 版本限制过严

**问题位置**: `pyproject.toml`, `packages/markitai/pyproject.toml`

**修复方案**:

- [ ] **任务 P1-7.1: 放宽 Python 版本要求**
  - 将 `requires-python = "==3.13.*"` 改为 `requires-python = ">=3.11"`
  - 需要验证依赖兼容性
  - 更新 CI 测试矩阵覆盖 3.11, 3.12, 3.13

---

### P2 - 性能问题（已有部分在性能优化任务中）

以下问题在 `docs/reference/markitai-overall-review-030.md` 中提到，但与现有"性能优化任务"可能有重叠：

#### P2-1: async 流程中的同步阻塞

**问题位置**: `workflow/core.py:264`

**状态**: 检查是否已在性能优化中解决

- [ ] **任务 P2-1.1: 确认同步阻塞是否已优化**
  - 检查 `process_embedded_images()` 是否使用 `asyncio.to_thread()`

---

#### P2-2: SQLite Cache 性能

**问题位置**: `llm.py:327-335`, `llm.py:414-431`

**状态**: 检查是否已在性能优化中解决

- [ ] **任务 P2-2.1: 确认连接复用是否已实现**
  - 对比 `fetch.py` 中 FetchCache 的优化

---

#### P2-3: io_semaphore 形同虚设

**问题位置**: `llm.py:1107-1115`

**问题描述**:
```python
@property
def io_semaphore(self) -> asyncio.Semaphore:
    if self._runtime:
        return self._runtime.io_semaphore
    return asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)  # 每次创建新的！
```

**修复方案**:

- [ ] **任务 P2-3.1: 添加实例级缓存**
  - 文件: `llm.py`
  - 参考 `semaphore` 属性的实现（`llm.py:1094-1104`）
  - 添加 `_local_io_semaphore` 缓存

---

### 实施顺序建议

| 阶段 | 任务 | 优先级 | 预估工时 |
|------|------|--------|----------|
| **阶段 1** | P0-1 图片引用错位 | P0 | 4h |
| **阶段 1** | P0-3 移除短 slide 启发式逻辑 | P0 | 1h |
| **阶段 2** | P0-2 文档截断警告 | P0 | 2h |
| **阶段 2** | P0-4 Alt 文本回填串行化 | P0 | 2h |
| **阶段 2** | P0-5 截图清理规则精确化 | P0 | 1h |
| **阶段 3** | P1-1 uv.lock 纳入版本控制 | P1 | 0.5h |
| **阶段 3** | P1-2 Python CI 工作流 | P1 | 2h |
| **阶段 3** | P1-4 LICENSE 文件 | P1 | 0.5h |
| **阶段 4** | P1-3 工具配置统一 | P1 | 1h |
| **阶段 4** | P1-5 示例 API Key 替换 | P1 | 0.5h |
| **阶段 4** | P1-6 跨平台/CI 稳定性 | P1 | 1h |
| **阶段 4** | P1-7 Python 版本放宽 | P1 | 2h |
| **阶段 5** | P2-* 性能问题确认 | P2 | 1h |

**总预估工时**: ~18h

### 测试验证

完成修复后需要验证：

```bash
# 代码质量
cd packages/markitai
uv run ruff check src tests
uv run pyright src tests

# 单元测试
uv run pytest tests/ -v

# 集成测试 - 图片处理
uv run markitai test-fixtures/with-images.pptx -o output-test --verbose

# 集成测试 - 长文档
uv run markitai test-fixtures/long-document.pdf --llm -o output-test --verbose
```

### 回滚方案

1. **Git 分支策略**: 在 `fix/overall-review-030` 分支开发
2. **增量提交**: 每个任务独立提交，便于 cherry-pick 或 revert
3. **特性开关**: P0 级别修复不需要开关，直接修复 Bug

---

## 真实场景测试修复任务

来源: 2026-01-24 真实场景测试 (`output-030`) 发现的问题
创建: 2026-01-24

### 问题总览

| 序号 | 问题 | 优先级 | 根本原因 |
|------|------|--------|----------|
| R1 | X tweet 没有使用 agent-browser | P0 | Playwright 浏览器未安装，但错误处理不完善 |
| R2 | candy.JPG 图片描述完全错误 | P0 | 视觉模型不可用时仍"瞎猜"描述 |
| R3 | concise.llm.md 被翻译成中文 | P0 | LLM 不遵守"禁止翻译"指令 |
| R4 | PDF page 5 内容丢失 | ❓ | 需确认（初步分析内容完整） |
| R5 | PPTX 页眉页脚未清理 | P1 | LLM 未执行清理指令 |
| R6 | 没有网页截图 | P0 | 同 R1，Playwright 未安装 |
| R7 | DOC 图片 alt 丢失 | P0 | 同 R2，视觉模型不可用 |
| R8 | sub_dir/assets 缺少 images.json | P1 | 子目录图片未写入 images.json |

### R1/R6: Playwright 浏览器未安装（agent-browser 不可用）

**问题描述**:

当 `agent-browser` 命令存在但 Playwright 浏览器未安装时：
1. `is_agent_browser_available()` 返回 True（因为命令存在）
2. 实际执行时 Playwright 报错：`browserType.launch: Executable doesn't exist`
3. 错误被捕获后静默回退到 static 策略，用户无感知

**日志证据**:
```
[URL] Browser fetch failed: agent-browser open failed: browserType.launch: Executable doesn't exist at /home/tseng/.cache/ms-playwright/chromium_headless_shell-1208
```

**问题位置**: `fetch.py:468-477`

```python
def is_agent_browser_available(command: str = "agent-browser") -> bool:
    """Check if agent-browser CLI is installed and available."""
    return shutil.which(command) is not None  # 只检查命令存在，不检查浏览器
```

**修复方案**:

- [ ] **任务 R1.1: 增强 agent-browser 可用性检测**
  - 文件: `fetch.py`
  - 新增 `verify_agent_browser_ready()` 函数
  - 调用 `agent-browser --version` 或 `agent-browser status` 验证完整性
  - 缓存检测结果（避免重复检测）

- [ ] **任务 R1.2: 首次运行时友好提示**
  - 文件: `cli.py`
  - 当检测到 agent-browser 未完全安装时，输出安装指引
  - 提示: `agent-browser install` 或 `npx playwright install`

- [ ] **任务 R1.3: 添加 --check-deps 命令**
  - 文件: `cli.py`
  - 新增 `markitai --check-deps` 检查所有可选依赖状态
  - 输出: agent-browser/playwright, OCR, LLM 配置状态

---

### R2/R7: 视觉模型不可用时错误生成描述

**问题描述**:

candy.JPG 实际是一只猫的照片，但输出描述为：
> "A screenshot showing a user interface with text input and output areas, likely from an AI assistant application."

这是因为：
1. Gemini API（视觉模型）因区域限制不可用
2. 系统回退到非视觉模型（DeepSeek）
3. 非视觉模型无法看到图片内容，在"瞎猜"

**日志证据**:
```
[Router] No vision-capable models configured, using main router
```

**问题位置 1**: `llm.py:1248-1256` - Router 初始化回退

```python
vision_models = [
    m for m in self.config.model_list
    if m.model_info and m.model_info.supports_vision
]
if not vision_models:
    logger.warning("[Router] No vision-capable models configured, using main router")
    self._vision_router = self.router  # 回退到非视觉模型！
```

**问题位置 2（核心 Bug）**: `llm.py:_analyze_with_json_mode()` - 使用错误的 router

```python
# 错误代码 - 使用 self.router 而非 self.vision_router
response = await self.router.acompletion(...)  # BUG!

# 正确代码 - 应该使用 self.vision_router
response = await self.vision_router.acompletion(...)
```

**根因分析（更新于 2026-01-24）**:
1. 用户已正确配置 `supports_vision: true` 模型
2. `vision_router` 正确初始化包含 9 个视觉模型
3. **核心 Bug**: `_analyze_with_json_mode` 方法使用 `self.router`（主 router）而非 `self.vision_router`
4. 导致即使有视觉模型配置，JSON mode 回退时仍使用非视觉模型

**修复方案**:

- [x] **任务 R2.1: 修复 _analyze_with_json_mode 使用错误的 router** ✅ (2026-01-24)
  - 文件: `llm.py`
  - 将 `_analyze_with_json_mode` 中的 `self.router.acompletion()` 改为 `self.vision_router.acompletion()`
  - 确保所有图片分析回退路径都使用 `vision_router`

- [ ] **任务 R2.2: 添加视觉模型配置验证**
  - 文件: `config.py`, `cli.py`
  - 启动时检查：如果启用了 `--alt` 或 `--desc`，但没有配置视觉模型
  - 输出警告并列出推荐的视觉模型配置

- [ ] **任务 R2.3: 模型能力声明标准化**
  - 文件: `config.schema.json`, `config.py`
  - 确保 `model_info.supports_vision` 在常用模型配置中正确设置
  - 文档中添加视觉模型配置示例（Gemini, GPT-4o, Claude 3）

---

### R3: LLM 不遵守"禁止翻译"指令

**问题描述**:

`concise.llm.md` 原文为英文，但输出被翻译成中文：
```
如果你想更快地取得进展，就写简洁的解释。用简单、强烈且清晰的语言解释思想...
```

**Prompt 已明确要求**（cleaner.md 第 8 行）:
```
**禁止翻译**：原文是什么语言就保留什么语言，禁止将中文翻译成英文或反过来
```

**问题分析**:

1. DeepSeek 模型对中文指令响应好，但可能"过度服务"
2. 当系统语言/文档语言不一致时，模型可能误判目标语言
3. Prompt 中的"禁止翻译"可能被忽视

**修复方案**:

- [x] **任务 R3.1: 加强语言保持指令** ✅ (2026-01-24)
  - 文件: `prompts/cleaner.md`, `prompts/document_enhance.md`, `prompts/url_enhance.md`
  - 在所有 prompt 文件开头添加 **核心原则** 部分，使用中英双语强调
  - 添加明确的语言保持规则：
    ```markdown
    - **禁止翻译（CRITICAL - DO NOT TRANSLATE）**：
      - 英文输入 → 英文输出（English in → English out）
      - 中文输入 → 中文输出（中文输入 → 中文输出）
      - 绝对禁止将任何语言翻译成另一种语言
      - 违反此规则将导致输出无效
    ```

- [ ] **任务 R3.2: 输出语言验证**
  - 文件: `llm.py`
  - 对比输入输出的主要语言
  - 如果语言发生变化，记录警告并可选拒绝/重试

- [ ] **任务 R3.3: 添加 --preserve-language 测试用例**
  - 文件: `tests/unit/test_llm.py`
  - 测试：英文输入应保持英文输出
  - 测试：中文输入应保持中文输出
  - 测试：混合语言文档应保持原有比例

---

### R5: PPTX 页眉页脚未清理 ✅ 已修复

**问题描述**:

`Free_Test_Data_500KB_PPTX.pptx.llm.md` 中仍保留页眉页脚：
```
FTD
FREE TEST DATA
2
```

**修复方案**:

- [x] **任务 R5.1: 代码层面后处理清理** ✅ (2026-01-24)
  - 文件: `utils/text.py`, `llm.py`
  - 新增 `clean_ppt_headers_footers()` 函数
  - 在 `format_llm_output()` 中调用后处理
  - 自动检测并清理重复出现的页眉页脚模式

- [x] **任务 R5.2: 加强 prompt 清理指令** ✅ (2026-01-24)
  - 文件: `prompts/cleaner.md`, `prompts/document_enhance.md`
  - 添加更具体的页眉页脚示例和特征描述
  - 明确删除条件：相同模式在 ≥3 页重复出现

---

### R8: 子目录 images.json 未生成

**问题描述**:

- `output-030/assets/images.json` 存在 ✓
- `output-030/sub_dir/assets/images.json` 不存在 ✗

**根因分析（更新于 2026-01-24）**:

原问题位置分析不准确。真正的原因是：
1. `workflow/single.py:analyze_images()` 中，当图片分析失败（`analysis is None`）时
2. 代码直接跳过该图片（`continue`），不记录到 `asset_descriptions`
3. 如果所有图片都分析失败，`asset_descriptions` 为空，images.json 不生成

**问题位置**: `workflow/single.py:229-241`

```python
# 错误代码 - 分析失败时跳过
if analysis is None:
    continue  # 跳过失败的图片！

# 正确代码 - 使用默认值
if analysis is None:
    analysis_caption = "Image"
    analysis_desc = "Image analysis failed"
    analysis_text = ""
    analysis_usage = {}
else:
    analysis_caption = analysis.caption
    ...
```

**修复方案**:

- [x] **任务 R8.1: 修复图片分析失败时的处理逻辑** ✅ (2026-01-24)
  - 文件: `workflow/single.py`
  - 当 `analysis is None` 时，使用默认值而非跳过
  - 默认值: `alt="Image"`, `desc="Image analysis failed"`, `text=""`
  - 确保图片分析失败时仍记录到 `images.json`

---

### 测试覆盖任务

为防止回归，需要建立完整的输出格式测试覆盖：

- [ ] **任务 T1: 创建 fixture-based 集成测试**
  - 文件: `tests/integration/test_output_format.py`
  - 使用 `tests/fixtures/` 作为输入
  - 验证输出 markdown 格式符合预期

- [ ] **任务 T2: 语言保持测试**
  - 输入英文文档 → 输出应为英文
  - 输入中文文档 → 输出应为中文
  - 检测翻译行为并失败

- [ ] **任务 T3: 图片 alt 测试**
  - 使用 mock vision model
  - 验证：有视觉模型 → 生成描述
  - 验证：无视觉模型 → 使用默认 alt，不瞎猜

- [ ] **任务 T4: PPTX 页眉页脚清理测试**
  - 使用带固定页眉页脚的 PPTX fixture
  - 验证输出中不包含重复的页眉页脚文本

- [ ] **任务 T5: 子目录 images.json 测试**
  - 使用嵌套目录结构的 fixtures
  - 验证每个子目录正确生成 images.json

---

### 实施顺序

| 阶段 | 任务 | 优先级 | 状态 | 说明 |
|------|------|--------|------|------|
| **阶段 1** | R2.1 修复 vision_router 使用 | P0 | ✅ 完成 | 修复 _analyze_with_json_mode 使用错误的 router |
| **阶段 1** | R3.1 语言保持指令 | P0 | ✅ 完成 | 加强所有 prompt 的语言保持规则 |
| **阶段 1** | R8.1 图片分析失败处理 | P0 | ✅ 完成 | 分析失败时使用默认值而非跳过 |
| **阶段 2** | R5.1 页眉页脚后处理 | P0 | ✅ 完成 | clean_ppt_headers_footers() 后处理 |
| **阶段 2** | R5.2 prompt 指令加强 | P0 | ✅ 完成 | 加强 cleaner.md/document_enhance.md |
| **阶段 2** | 链接格式修复 | P0 | ✅ 完成 | fix_broken_markdown_links() 后处理 |
| **阶段 2** | 残留占位符清理 | P0 | ✅ 完成 | clean_residual_placeholders() 后处理 |
| **阶段 2** | PDF 内容保留加强 | P0 | ✅ 完成 | 加强 document_enhance.md 禁止删除指令 |
| **阶段 3** | R1.1-R1.2 agent-browser 检测 | P1 | 待办 | 改善用户体验 |
| **阶段 3** | T1-T5 测试覆盖 | P1 | 待办 | 防止回归 |
| **阶段 4** | R2.2, R3.2 验证逻辑 | P2 | 待办 | 增强健壮性 |

### 前置条件

在开始修复前，用户需要：

1. **安装 Playwright 浏览器**（如果需要 URL 截图功能）:
   ```bash
   agent-browser install
   # 或
   npx playwright install chromium
   ```

2. **配置视觉模型**（如果需要图片描述功能）:
   ```yaml
   # markitai.yaml
   llm:
     enabled: true
     model_list:
       - model_name: "vision"
         litellm_params:
           model: "gemini/gemini-2.0-flash"
           api_key: "${GEMINI_API_KEY}"
         model_info:
           supports_vision: true
   ```

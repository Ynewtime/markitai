# Markitai 技术规格文档

> 版本: 0.4.2
> 最后更新: 2026-02-03

---

## 1. 概述

### 1.1 项目定位

Markitai 是一个开箱即用的 Markdown 转换器，原生支持 LLM 增强。核心设计理念：

- **程序转换 + LLM 优化**：基础转换只做格式转换，数据清洗、格式优化、排版等交由大模型处理
- **不造轮子**：基于社区优秀依赖实现，避免重复造轮子
- **测试驱动**：所有特性都需要测试覆盖

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| 最小依赖 | 基础转换不引入或仅引入最少的数据清洗规则 |
| 可选 LLM | LLM 功能为可选增强，不强制依赖 |
| 配置优先 | 所有行为可通过配置自定义 |
| 优雅降级 | LLM 失败时保留基础转换结果 |

### 1.3 技术栈

| 组件 | 库 | 用途 |
|------|------|------|
| 包管理 | uv | Monorepo workspace 管理 |
| PDF 转换 | pymupdf4llm | PDF → Markdown + 图片提取 |
| Office 转换 | markitdown[all] | Word/PPT/Excel 基础转换 |
| LLM 网关 | litellm | 统一 LLM 调用、成本追踪、负载均衡 |
| OCR | rapidocr | 扫描版 PDF 和图片文字识别 |
| CLI | click | 命令行接口 |
| 日志 | loguru | 日志记录 |

**Python 版本要求**: >=3.11

---

## 2. 系统架构

> 详细架构说明见 [architecture.md](./architecture.md)

### 2.1 核心模块

| 模块 | 职责 |
|------|------|
| `ConfigManager` | 配置加载与合并 (config.py) |
| `Converter` | 文档基础转换 (converter/) |
| `LLMProcessor` | LLM 调用与处理 (llm/) |
| `ImageProcessor` | 图片提取、压缩、分析 (image.py) |
| `BatchProcessor` | 批量处理与断点恢复 (batch.py) |
| `FetchModule` | URL 抓取 - static/browser/jina 策略 (fetch.py) |

### 2.2 数据流

**本地文件**:
```
输入文件 → 格式检测 → 基础转换 → 图片提取压缩 → [LLM 增强] → 输出 Markdown
```

**URL 抓取**:
```
.urls 文件 → 解析 URL → 选择策略 (static/playwright/jina) → 抓取 → [LLM 增强] → 输出
```

---

## 3. 接口设计

> 详细 CLI 参数见 [CLI 参考](https://markitai.ynewtime.com/guide/cli)

### 3.1 主命令

```bash
markitai [OPTIONS] INPUT
```

### 3.2 预设配置

| 预设 | 等效参数 | 适用场景 |
|------|----------|----------|
| `rich` | `--llm --alt --desc --screenshot` | 复杂文档，需要完整分析 |
| `standard` | `--llm --alt --desc` | 普通文档，不需要截图 |
| `minimal` | 无增强 | 仅基础转换 |

### 3.3 子命令

- `markitai config list|init|validate` - 配置管理
- `markitai cache stats|clear` - 缓存管理
- `markitai check-deps` - 依赖检查

---

## 4. 配置管理

> 详细配置说明见 [配置指南](https://markitai.ynewtime.com/guide/configuration)

### 4.1 优先级

```
CLI 参数 > 环境变量 > 配置文件 > 默认值
```

### 4.2 配置文件查找

1. `--config` 指定路径
2. `MARKITAI_CONFIG` 环境变量
3. `./markitai.json`
4. `~/.markitai/config.json`

### 4.3 环境变量引用

配置值支持 `env:VAR_NAME` 语法引用环境变量：

```json
{
  "llm": {
    "model_list": [{
      "litellm_params": {
        "api_key": "env:OPENAI_API_KEY"
      }
    }]
  }
}
```

---

## 5. 转换引擎

### 5.1 支持格式

| 格式 | 转换器 | 依赖 |
|------|--------|------|
| PDF | PDFConverter | pymupdf4llm |
| DOCX/PPTX/XLSX | OfficeConverter | markitdown |
| DOC/XLS/PPT | LegacyConverter | LibreOffice / pywin32 |
| PNG/JPG/WebP | ImageConverter | rapidocr |
| TXT/MD | TextConverter | - |

### 5.2 转换器模式

```python
@register_converter(FileFormat.PDF)
class PDFConverter(BaseConverter):
    def convert(self, input_path: Path, output_dir: Path | None = None) -> ConvertResult:
        ...
```

---

## 6. LLM 集成

### 6.1 LiteLLM Router

使用 LiteLLM Router 实现多模型负载均衡：

- 支持 100+ LLM providers
- 权重路由（`simple-shuffle`）
- 自动重试和 fallback
- 成本跟踪

### 6.2 模型组

| 模型组 | 用途 |
|--------|------|
| `default` | 文本处理（清洗、frontmatter） |
| `vision` | 图片分析、页面内容提取 |

### 6.3 本地提供商

支持通过 CLI 工具调用（订阅制，无额外 API 费用）：

- `claude-agent/*` - Claude Code CLI
- `copilot/*` - GitHub Copilot CLI

---

## 7. 图片处理

### 7.1 处理流程

```
提取图片 → 过滤（尺寸/去重）→ 压缩 → 保存到 assets/ → [LLM 分析]
```

### 7.2 分析模式

| 模式 | 输出 | 说明 |
|------|------|------|
| `--alt` | Markdown alt 文本 | 简洁的图片描述 |
| `--desc` | images.json | 详细的图片描述文件 |

---

## 8. 提示词管理

### 8.1 内置提示词

提示词位于 `prompts/` 目录，采用系统/用户分离：

- `cleaner_system.md` / `cleaner_user.md`
- `frontmatter_system.md` / `frontmatter_user.md`
- `image_analysis_system.md` / `image_analysis_user.md`
- ...

### 8.2 自定义覆盖

通过配置文件指定自定义提示词路径：

```json
{
  "prompts": {
    "cleaner_system": "~/.markitai/prompts/my_cleaner_system.md"
  }
}
```

---

## 9. 批量处理

### 9.1 功能

- 可配置并发数（文件/URL/LLM）
- 断点续传（`--resume`）
- 实时进度显示（Rich Live）
- 错误隔离（单文件失败不影响整体）

### 9.2 状态管理

```
output/.markitai_state.json  # 批处理状态
output/.markitai_report.json # 处理报告
```

---

## 10. URL 抓取

### 10.1 策略选择

| 策略 | 触发条件 | 说明 |
|------|----------|------|
| `static` | 静态页面 | requests + BeautifulSoup |
| `playwright` | SPA/动态页面 | Playwright 渲染 |
| `jina` | 配置指定 | Jina Reader API |

### 10.2 SPA 检测

自动检测 React/Vue/Angular 等框架，学习并缓存需要浏览器渲染的域名。

---

## 11. 缓存系统

### 11.1 缓存类型

| 缓存 | 存储 | 用途 |
|------|------|------|
| LLM Cache | SQLite | LLM 响应缓存（基于内容哈希）|
| Fetch Cache | SQLite | URL 抓取结果缓存 |
| SPA Domain Cache | JSON | SPA 域名学习 |

### 11.2 缓存控制

- `--no-cache` - 全局禁用
- `--no-cache-for "*.pdf"` - 模式匹配禁用

---

## 附录

### A. 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 配置错误 |
| 5 | LLM 调用失败（已降级） |
| 10 | 部分文件处理失败 |

### B. 输出结构

```
output/
├── document.docx.md        # 基础 Markdown
├── document.docx.llm.md    # LLM 增强版本
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json
└── screenshots/            # --screenshot
    └── example_com.full.jpg
```

---

*文档结束*

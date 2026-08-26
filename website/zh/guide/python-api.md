# Python API

Markitai 也可以作为库使用，不经过 CLI。`markitai.convert()` 及其异步版本 `markitai.aconvert()` 运行与 CLI 完全相同的转换管线——包括 LLM 增强——并返回带类型的结果。

::: warning 暂定接口
markitai 处于 0.x 阶段，Python API 为**暂定（provisional）**接口：签名与结果字段可能在次版本中调整。
:::

```bash
uv add markitai   # 或：pip install markitai
```

## 基础转换

```python
import markitai

# 内存模式：转换并直接拿到 Markdown 字符串
out = markitai.convert("report.pdf")
print(out.markdown)

# 写入目录（同时保留抽取的图片资源）
out = markitai.convert("report.pdf", output_dir="out/")
print(out.output_path)  # out/report.pdf.md
print(out.assets)  # 图片位于 out/.markitai/assets/

# URL 用法相同
out = markitai.convert("https://example.com/article")
```

不传 `output_dir` 时，转换在一个私有临时目录中进行、结束后即删除：你拿到内存中的 Markdown，路径字段为 `None`，图片链接保持相对的 `.markitai/assets/...` 形式。需要图片文件时请传入 `output_dir`。

库调用保证 `stdout` 干净——PyMuPDF/ONNX 的解析器噪声在 `convert()` 入口即被抑制，无需任何 CLI 初始化。诊断信息经 [loguru](https://github.com/Delgan/loguru) 输出到 stderr，可用 `logger.disable("markitai")` 静默。

## LLM 增强转换

`llm=True` 运行完整的增强管线（清理 + frontmatter），并同时返回两个版本：

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-5.4-nano"  # 或配置 llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)  # 增强后的正文
print(out.frontmatter["title"])  # 解析后的 YAML frontmatter
print(out.usage.cost_usd)  # 本次转换的 LLM 花费
```

模型解析规则与 CLI 完全一致：优先使用[配置文件](/zh/guide/configuration)中的 `llm.model_list`，其次是 `MODEL` 环境变量。`config=None`（默认）时加载与 CLI 相同的配置文件层级；传入 `markitai.MarkitaiConfig` 实例可完全编程控制：

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()  # 纯默认值，忽略配置文件
cfg.llm.pure = True  # 原样 LLM 清理，不加 frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

关键字开关（`llm`、`ocr`、`screenshot`、`alt`、`desc`）覆盖配置值，与 CLI 开关一一对应；`None` 表示沿用配置。带截图和多源内容的 URL 转换会自动启用 vision 增强（截图作为 LLM 的视觉参照），`screenshot.screenshot_only` 则直接让模型读页面截图——与 CLI 行为一致。`profile="rag" | "obsidian" | "okf"` 对应 `--profile`，为下游消费者塑形输出——LlamaIndex 摄取示例见[输出 Profile](./output-profiles.md)。

## 异步用法

在事件循环内请使用 `aconvert`——CPU 密集的转换工作（PyMuPDF、ONNX、Office 抽取）在共享线程池中执行，事件循环始终保持响应：

```python
import asyncio
import markitai


async def main() -> None:
    results = await asyncio.gather(
        markitai.aconvert("a.pdf", output_dir="out/"),
        markitai.aconvert("https://example.com/b", output_dir="out/"),
    )
    for out in results:
        print(out.source, "->", out.output_path or len(out.markdown))


asyncio.run(main())
```

在运行中的事件循环里调用同步的 `convert()` 会抛出 `RuntimeError`。长驻应用退出时可调用 `await markitai.fetch.close_shared_clients()` 释放共享 HTTP 客户端（同步 `convert()` 会自动完成这一步）。

## ConversionOutput

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` | 传入的路径或 URL |
| `markdown` | `str` | 基础转换的 Markdown 正文（已去除 frontmatter） |
| `llm_markdown` | `str \| None` | LLM 增强正文，未启用 LLM 时为 `None` |
| `frontmatter` | `dict` | 最完整输出的 YAML frontmatter（已解析） |
| `output_path` | `Path \| None` | 写入的基础 `.md` 文件 |
| `llm_output_path` | `Path \| None` | 写入的 `.llm.md` 文件 |
| `assets` | `list[Path]` | 抽取的图片文件 |
| `screenshots` | `list[Path]` | 渲染的页面/截图文件 |
| `images` | `list[dict]` | 逐图 LLM 分析条目（alt/描述） |
| `usage` | `ConversionUsage` | `cost_usd`、token 总量与按模型明细 |
| `skip_reason` | `str \| None` | 因冲突策略跳过时为 `"exists"` |
| `duration` | `float` | 耗时（秒） |

失败会直接抛异常而非返回残缺结果：管线失败抛 `ConversionError`，URL 不可达抛 `FetchError`，启用 LLM 但无法解析模型抛 `ValueError`。每次调用只转换一个文件或 URL；目录批处理仍由 CLI 负责。

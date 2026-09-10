# Python API

把 markitai 当库用。`markitai.convert()` 和它的异步版本 `markitai.aconvert()` 跑的是和 CLI 一样的管线，包括 LLM 增强，并返回类型化的结果。

::: warning API 稳定性
Python API 目前是临时接口：函数签名和结果字段在小版本里仍可能变化。CLI 才是稳定接口。依赖某个库细节时，请固定精确版本。
:::

```bash
uv add markitai   # 或：pip install markitai
```

## 基础转换

```python
import markitai

# 内存模式：直接拿到 Markdown 字符串
out = markitai.convert("report.pdf")
print(out.markdown)

# 写入目录，同时保留抽取的图片
out = markitai.convert("report.pdf", output_dir="out/")
print(out.output_path)  # out/report.pdf.md
print(out.assets)       # 图片在 out/.markitai/assets/ 下

# URL 用法相同
out = markitai.convert("https://example.com/article")
```

不传 `output_dir` 时，转换在一个临时目录里进行，结束后即删除。你在内存里拿到 Markdown，路径字段为 `None`，图片链接保持相对的 `.markitai/assets/...` 形式。需要图片文件时传 `output_dir`。

库调用不会污染 `stdout`。诊断信息通过 [loguru](https://github.com/Delgan/loguru) 输出到 stderr，用 `logger.disable("markitai")` 可以静音。

## LLM 增强转换

`llm=True` 执行清洗和 frontmatter 生成，并同时返回两个版本：

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-5.6-luna"  # 或配置 llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)          # 增强后的正文
print(out.frontmatter["title"])  # 解析后的 YAML frontmatter
print(out.usage.cost_usd)        # 这次转换的 LLM 花费
```

模型的解析方式和 CLI 一样：先看[配置](/zh/guide/configuration)里的 `llm.model_list`，再看环境变量 `MODEL`。默认加载和 CLI 相同的配置文件。要完全自己控制，传一个 `markitai.MarkitaiConfig`：

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()   # 纯默认值，忽略配置文件
cfg.llm.pure = True      # 只做 LLM 清洗，不生成 frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

关键字开关 `llm`、`ocr`、`screenshot`、`alt`、`desc` 对应 CLI 参数，会覆盖配置；传 `None` 沿用配置值。`profile="rag" | "obsidian" | "okf"` 对应 `--profile`，见[输出 Profile](./output-profiles.md)。

## 异步用法

在事件循环里用 `aconvert`。CPU 密集的工作在线程池里跑，循环不会被卡住：

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

在运行中的事件循环里调用同步的 `convert()` 会抛出 `RuntimeError`。长期运行的应用应在退出时调用 `await markitai.fetch.close_shared_clients()` 释放共享的 HTTP 客户端。

一次性脚本偶尔会在转换成功后以退出码 134 结束，这是 onnxruntime 卸载时的一个小问题。在脚本末尾加上 `from markitai.utils.shutdown import finalize_process; finalize_process(0)` 可以避免。长期运行的宿主不要用它，它会直接结束整个进程。

## ConversionOutput

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` | 传入的路径或 URL |
| `markdown` | `str` | 基础 Markdown 正文（已去掉 frontmatter） |
| `llm_markdown` | `str \| None` | LLM 增强后的正文，未开 LLM 时为 `None` |
| `frontmatter` | `dict` | 最完整那份输出的 YAML frontmatter，已解析 |
| `output_path` | `Path \| None` | 写出的基础 `.md` 文件 |
| `llm_output_path` | `Path \| None` | 写出的 `.llm.md` 文件 |
| `assets` | `list[Path]` | 抽取出的图片文件 |
| `screenshots` | `list[Path]` | 渲染的页面或截图文件 |
| `images` | `list[dict]` | 每张图片的 LLM 分析条目（alt、描述） |
| `usage` | `ConversionUsage` | `cost_usd`、token 总量、按模型的明细 |
| `skip_reason` | `str \| None` | 因冲突策略跳过时为 `"exists"` |
| `duration` | `float` | 耗时（秒） |

失败时直接抛异常，不返回半成品：管线失败抛 `ConversionError`，URL 不可达抛 `FetchError`，开了 LLM 却没有模型抛 `ValueError`。一次调用转一个文件或 URL，目录批量仍由 CLI 负责。

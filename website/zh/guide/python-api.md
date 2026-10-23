# Python API

`markitai.convert()` 和它的异步版本 `markitai.aconvert()` 跑 CLI 的管线（包括 LLM 增强），返回类型化的结果。这个包就是 Rust 引擎本身，加载在你的 Python 进程里：没有其他依赖，也不另起进程。

::: warning API 稳定性
Python API 尚未定型：签名和字段在小版本里仍可能变化。请固定精确版本；CLI 才是稳定接口。
:::

把它加进你的项目（需要 Python 3.10 或更高版本；提供 macOS 11+、glibc 2.28+ 的 Linux 和 Windows 的 wheel，x86-64 和 ARM64 都有）。[安装脚本](/zh/guide/getting-started)装的是命令，不是可导入的库：

```bash
uv add markitai   # 或：pip install markitai
```

这个包同时会安装 `markitai`、`mkai` 和 `markitai-mcp` 三个命令。

## 基础转换

```python
import markitai

# 内存模式：直接拿到 Markdown 字符串
out = markitai.convert("report.pdf")
print(out.markdown)

# 把输出和抽取的图片写入目录
out = markitai.convert("report.pdf", output_dir="out/")
print(out.output_path)  # out/report.pdf.md
print(out.assets)  # 图片在 out/.markitai/assets/ 下

# URL 用法相同
out = markitai.convert("https://example.com/article")
```

- 不传 `output_dir` 时什么都不写：Markdown 在内存里返回，`output_path` 为 `None`，`assets` 为空，图片链接保持 `.markitai/assets/...`。
- 一次调用转一个文件或 URL；传目录会抛出 `IsADirectoryError`（批量仍由 CLI 负责）。
- `stdout` 保持干净；提示通过 `out.warnings` 返回。

## LLM 增强转换

`llm=True` 增加清洗和 frontmatter，并同时返回两个版本：

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-6-luna"  # 或配置 llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)  # 增强后的正文
print(out.frontmatter["title"])  # 解析后的 YAML frontmatter
print(out.usage.cost_usd)  # 这次转换的 LLM 花费
```

```mermaid
flowchart LR
  L[llm.model_list] -->|为空| M[MODEL]
  M -->|未设置| D[检测到的 key 和订阅]
  D -->|都没有| E[NoModelConfiguredError]
```

模型的选择和 CLI 一样（[配置](/zh/guide/configuration)，然后是[检测到的](/zh/guide/configuration#defaults-markitai-picks-for-you) API key 和已登录的订阅）；`./.env` 和 `~/.markitai/.env` 的读取方式与 CLI 相同。默认加载 CLI 的配置文件，除非传入配置：

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()  # 纯默认值，忽略配置文件（config={} 也一样）
cfg.llm.pure = True  # 只做 LLM 清洗，不生成 frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

关键字 `llm`、`ocr`、`screenshot`、`alt`、`desc` 和 `profile`（[输出 Profile](./output-profiles.md)）对应 CLI 参数；`None` 沿用配置值。`config` 也接受与 `markitai.json` 结构相同的普通 `dict`。`MarkitaiConfig` 在转换时检查取值，值不对时抛出 `ValueError`。

## 异步用法

在事件循环里用 `aconvert`；转换在线程里运行，不占 GIL：

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

- 在运行中的事件循环里调用同步的 `convert()` 会抛出 `RuntimeError`。
- 取消 `aconvert` 只是停止等待；已经在运行的转换仍会完成。

## ConversionOutput {#conversionoutput}

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` | 传入的路径或 URL |
| `markdown` | `str` | 基础正文，不含 frontmatter |
| `llm_markdown` | `str \| None` | 增强后的正文，或 `None` |
| `frontmatter` | `dict` | 解析后的 frontmatter |
| `output_path` | `Path \| None` | 写出的基础 `.md` 文件 |
| `llm_output_path` | `Path \| None` | 写出的 `.llm.md` 文件 |
| `assets` | `list[Path]` | 抽取出的图片 |
| `screenshots` | `list[Path]` | 截图文件 |
| `images` | `list[dict]` | 每张图片的 LLM 分析 |
| `usage` | `ConversionUsage` | `cost_usd`、`requests`、token、`by_model` |
| `skip_reason` | `str \| None` | 被跳过时为 `"exists"` |
| `duration` | `float` | 耗时（秒） |
| `warnings` | `list[str]` | 提示（见下） |
| `page_sources` | `list[dict]` | PDF 每页文字来源（见下） |

`warnings`：疑似扫描页（加 `ocr=True` 重试）、PDF 隐藏文字（可能是提示注入）、OCR 没识别出文字、幻灯片无法渲染、URL 截图没拍到、图片分析失败，或 LLM 增强失败（`llm_markdown` 为 `None`；设 `llm.on_failure = "fail"` 则抛出 `ConversionError`）。

`page_sources`：每页一条 `{"page", "source", "reason"}`；`source` 为 `"text"`、`"ocr"` 或 `"mixed"`，`reason` 是简短说明，例如 `"no text layer"`。

### 异常

失败时抛异常；markitai 自己的异常从 `markitai` 导入。每个异常都带 `.usage`（失败的调用已经花掉的部分，或 `None`）：

| 异常 | 何时 |
|------|------|
| `ConversionError` | 转换失败、格式不支持（`.code` 说明是哪种） |
| `FetchError` | URL 不可达 |
| `NoModelConfiguredError` | 开了 LLM 却没有可用模型（是一种 `ValueError`；不发请求） |
| `ConfigFileError` | 配置无效（是一种 `ValueError`） |
| `FileNotFoundError`、`IsADirectoryError` | 路径不对 |

## 其他语言

同一个引擎也提供给 Node.js 和 Go，以及任何能调用 C 的语言。这些绑定需要从源码检出构建（Rust 1.92+ 和 C 工具链），暂未发布到 npm 或 Go module proxy。

| 语言 | 需要 | 构建 | 调用 |
|------|------|------|------|
| Node.js | 18+ | `npm --prefix bindings/node run build`，然后 `npm pack` | `const { convert } = require('markitai'); const out = await convert('report.pdf', { llm: false })` |
| Go | 1.23+、cgo；macOS 和 Linux | `cargo build --release --locked -p markitai-ffi`，然后用 `replace` 指向 `bindings/go` | `out, err := markitai.Convert("report.pdf", &markitai.Options{LLM: markitai.Bool(false)})` |
| C | C 编译器 | `cargo build --release --locked -p markitai-ffi` | 用 JSON 请求调用 `markitai_convert_json()`，再调用 `markitai_buffer_free()`（`bindings/c/markitai.h`） |

结果带上面那些字段（Node.js 里是 snake_case，Go 里是类型化的结构体）；失败时是带 `code` 的 `ConversionError`。详见 [`docs/bindings.md`](https://github.com/Ynewtime/markitai/blob/main/docs/bindings.md)。

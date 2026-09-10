# MCP 服务器

`markitai-mcp` 服务器随 markitai 主包发布（安装 `mcp` extra 即得；`markitai mcp` 子命令启动的是同一个服务，官方 MCP Registry 条目用的就是 `uvx --from "markitai[mcp]" markitai mcp`），通过 [Model Context Protocol](https://modelcontextprotocol.io)（stdio 传输）把 markitai 的转换能力提供给 AI Agent。Agent 可用四个工具完成本地文档与 URL 的单个及批量转换，运行的是与 CLI、Python API 完全相同的管线。

无需安装步骤：`uvx` 按需拉取并运行。

## 接入

**Claude Code** —— 一条命令：

```bash
claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp
```

**Claude Desktop** —— 在 `claude_desktop_config.json` 中加一个条目：

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["--from", "markitai[mcp]", "markitai-mcp"] }
  }
}
```

其他 MCP 客户端同理：command 填 `uvx`，参数填 `["--from", "markitai[mcp]", "markitai-mcp"]`。没有 uv 时，`pip install "markitai[mcp]"` 提供同名的 `markitai-mcp` 命令。

## 工具

| 工具 | 用途 |
|------|------|
| `convert_document` | 单个本地文件（绝对路径）→ Markdown |
| `convert_url` | 单个网页 → 提取正文的干净 Markdown |
| `batch_convert` | 多个路径/URL 后台转换 → 返回 `job_id` |
| `job_status` | 查询批量任务的进度与逐项结果 |

每次转换都写出真实文件——写入 Agent 传入的 `output_dir`，或一个新建的临时目录（路径随结果返回）。结果默认内联 Markdown 文本，超过约 40 KB 时改为截断预览加 `markdown_file`（完整输出的文件路径），大文档不会灌爆模型上下文。三个转换工具都支持 `profile`（`rag`、`obsidian`、`okf`）为下游消费者塑形输出，省略时跟随服务器配置。

批量任务在服务器进程内执行，任务表保存在内存中：轮询 `job_status` 直到 `status` 为 `"completed"`（或 `"cancelled"`），再读取各项的 `markdown_file`。任务为 `"completed"` 时，`results[i]` 对应你传入的 `sources[i]`；运行中或取消的任务省略未完成项，此时通过 `source` 识别部分结果。每项写入 `output_dir/batch-<job_id>/<item_number>/`（编号从 `0001` 开始），同名文件及其资源相互隔离，同时运行多个批次也不会覆盖。`batch_convert` 接受 `concurrency`（默认 10）限制同时转换的数量。任务会随着新任务完成以及服务器重启被遗忘——`job_status` 会说明属于哪种情况；已写出的文件仍在。

## LLM 增强

LLM 增强按调用以 `llm: true` 开启：省略 `llm` 时跟随服务器自己的 markitai 配置（`llm.enabled`，默认关闭），传 `llm: false` 则强制关闭。`alt`/`desc` 控制图片分析，`ocr` 与 `screenshot` 对应各自能力；开启 LLM 可能产生提供商费用。开启前需要配置模型——写在 `~/.markitai/config.json`（[与 CLI 共用](/zh/guide/configuration)），或通过服务器条目的环境变量：

```json
{
  "mcpServers": {
    "markitai": {
      "command": "uvx",
      "args": ["--from", "markitai[mcp]", "markitai-mcp"],
      "env": {
        "MODEL": "openai/gpt-5.6-luna",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

在未配置模型的情况下以 `llm: true` 调用会得到一条可读的错误，其中原样给出上述配置指引，Agent 可以直接转述修复方法。

可选能力沿用 markitai 的 extras：扫描件 OCR 需要 `markitai[ocr]`，URL 截图需要 `markitai[browser]`——uvx 场景写作 `"args": ["--from", "markitai[mcp]", "--with", "markitai[ocr]", "markitai-mcp"]`。

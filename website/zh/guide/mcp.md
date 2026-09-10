# MCP 服务器

`markitai-mcp` 随 markitai 主包发布，通过 [Model Context Protocol](https://modelcontextprotocol.io)（stdio）把转换能力提供给 AI Agent。Agent 拿到四个工具，可以单个或批量转换本地文档和 URL，走的是和 CLI 相同的管线。

不用先安装：`uvx` 按需拉取并运行。

## 接入

**Claude Code**，一条命令：

```bash
claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp
```

**Claude Desktop**，在 `claude_desktop_config.json` 里加一条：

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["--from", "markitai[mcp]", "markitai-mcp"] }
  }
}
```

其他 MCP 客户端同理：command 填 `uvx`，参数填 `["--from", "markitai[mcp]", "markitai-mcp"]`。没有 uv 时，`pip install "markitai[mcp]"` 提供同名的 `markitai-mcp` 命令。`markitai mcp` 从 CLI 启动的是同一个服务，[MCP Registry](https://registry.modelcontextprotocol.io) 条目用的就是这种写法。

## 工具

| 工具 | 用途 |
|------|------|
| `convert_document` | 单个本地文件（绝对路径）转 Markdown |
| `convert_url` | 单个网页转成只含正文的干净 Markdown |
| `batch_convert` | 多个路径或 URL 后台转换，返回 `job_id` |
| `job_status` | 批量任务的进度和逐项结果 |

每次转换都写出真实文件，写进 Agent 传入的 `output_dir`，或一个新建的临时目录（路径随结果返回）。结果默认内联 Markdown，超过约 40 KB 时改为预览加 `markdown_file`（完整输出的路径），大文档不会灌爆模型上下文。转换工具还接受 `profile`（`rag`、`obsidian` 或 `okf`）为下游消费者塑形输出。

批量任务在服务进程内执行。轮询 `job_status` 直到 `status` 为 `"completed"`，再读取各项的 `markdown_file`。每项写入 `output_dir/batch-<job_id>/<item_number>/`，同名文件不会互相覆盖。`concurrency`（默认 10）限制同时转换的数量。服务重启后任务记录会丢失，已写出的文件仍在。

## LLM 增强

传 `llm: true` 开启增强（可能产生提供商费用）。在 `~/.markitai/config.json` 里配置模型（[与 CLI 共用](/zh/guide/configuration)），或在服务条目里设置环境变量：

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

省略 `llm` 时跟随服务自己的配置（`llm.enabled`，默认关闭），传 `llm: false` 则强制关闭。`alt` 和 `desc` 控制图片分析，`ocr` 和 `screenshot` 对应各自能力。没配模型却传了 `llm: true`，会得到一条把上面配置方法原样带上的错误，Agent 可以直接转述。

可选能力沿用 markitai 的 extras：OCR 需要 `markitai[ocr]`，URL 截图需要 `markitai[browser]`。uvx 场景这样加：`"args": ["--from", "markitai[mcp]", "--with", "markitai[ocr]", "markitai-mcp"]`。

# markitai-mcp

MCP (Model Context Protocol) server for [markitai](https://pypi.org/project/markitai/): document and URL → Markdown conversion as agent tools, over stdio.

Tools: `convert_document`, `convert_url`, `batch_convert`, `job_status`. LLM enhancement is off by default and opt-in per call. Large results are written to disk and returned as a path plus preview instead of flooding the model context.

Run it with zero setup:

```bash
uvx markitai-mcp
```

Claude Code:

```bash
claude mcp add markitai -- uvx markitai-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["markitai-mcp"] }
  }
}
```

Full guide: <https://markitai.dev/guide/mcp>

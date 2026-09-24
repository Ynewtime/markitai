# MCP Server

`markitai-mcp` ships with markitai and exposes its conversions to AI agents over the [Model Context Protocol](https://modelcontextprotocol.io) (stdio). Agents get four tools for converting local documents and URLs, one at a time or in batches, through the same pipeline as the CLI.

Nothing to install first: `uvx` fetches and runs it on demand.

## Setup

**Claude Code**, one command:

```bash
claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp
```

**Claude Desktop**, one entry in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["--from", "markitai[mcp]", "markitai-mcp"] }
  }
}
```

Any other MCP client works the same way: command `uvx`, arguments `["--from", "markitai[mcp]", "markitai-mcp"]`. Without uv, `pip install "markitai[mcp]"` provides the same `markitai-mcp` command. `markitai mcp` starts the same server from the CLI, which is the form the [MCP Registry](https://registry.modelcontextprotocol.io) entry uses. Both forms load `./.env` (the server's working directory) and then `~/.markitai/.env`, as the CLI does; variables already set, such as the `env` block below, take precedence.

## Tools

| Tool | Purpose |
|------|---------|
| `convert_document` | One local file (absolute path) to Markdown |
| `convert_url` | One web page to clean main-content Markdown |
| `batch_convert` | Many paths or URLs in the background; returns a `job_id` |
| `job_status` | Progress and per-item results of a batch job |

Every conversion writes real files, into the `output_dir` the agent passes or into a fresh temporary directory, and the result carries that path. Results inline the Markdown; past about 40 KB they switch to a preview plus `markdown_file`, the path to the full output, so a huge document never floods the model context. The conversion tools also accept `profile` (`rag`, `obsidian` or `okf`) to shape the output for a downstream consumer.

Results also carry `warnings`: notices that didn't fail the conversion but are worth acting on, such as pages that look scanned (retry with `ocr: true`), hidden PDF text that may be a prompt injection, OCR that found no text, or a URL screenshot that wasn't captured. Each successful `job_status` result has its own `warnings` too.

Batch jobs run inside the server process. Poll `job_status` until `status` is `"completed"`, then read the `markdown_file` paths. Each item lands in `output_dir/batch-<job_id>/<item_number>/`, so same-named files never collide. `concurrency` (default 10) bounds how many conversions run at once. A server restart forgets the jobs; the written files remain.

## LLM Enhancement

Pass `llm: true` to enable enhancement (provider charges may apply). Models resolve as in the CLI: `llm.model_list` in `~/.markitai/config.json` ([shared with the CLI](/guide/configuration)), then `MODEL`, then automatic detection of provider API keys (for example the ones `markitai init` writes to `~/.markitai/.env`) and signed-in subscription providers. Several detected providers share one pool and the server logs a warning on stderr; set `MODEL` to pin one. You can also set environment variables in the server entry:

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

Omitting `llm` follows the server's own config (`llm.enabled`, off by default); `llm: false` forces it off. `alt` and `desc` control image analysis, `ocr` and `screenshot` their capabilities. Calling with `llm: true` and no model configured fails with an error that repeats this setup guidance, so agents can relay the fix. Other failures (a missing file, a directory, a relative path, an unreachable URL, a failed conversion) come back as tool errors that state the reason. `batch_convert` rejects relative paths up front, as `convert_document` does.

Optional capabilities follow the markitai extras: OCR needs `markitai[ocr]`, URL screenshots need `markitai[browser]`. With uvx, add them as `"args": ["--from", "markitai[mcp]", "--with", "markitai[ocr]", "markitai-mcp"]`.

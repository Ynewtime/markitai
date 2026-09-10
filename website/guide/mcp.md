# MCP Server

The `markitai-mcp` server (bundled with markitai — install the `mcp` extra) exposes markitai conversions to AI agents over the [Model Context Protocol](https://modelcontextprotocol.io) (stdio transport). Agents get four tools — single and batch conversion of local documents and URLs — running the same pipeline as the CLI and the Python API.

No installation step is required: `uvx` fetches and runs it on demand.

## Setup

**Claude Code** — one command:

```bash
claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp
```

**Claude Desktop** — one entry in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "markitai": { "command": "uvx", "args": ["--from", "markitai[mcp]", "markitai-mcp"] }
  }
}
```

Any other MCP client works the same way: command `uvx`, arguments `["--from", "markitai[mcp]", "markitai-mcp"]`. Without uv, `pip install "markitai[mcp]"` provides the same `markitai-mcp` command. `markitai mcp` is the same server as a subcommand of the CLI — the form the [MCP Registry](https://registry.modelcontextprotocol.io) entry uses (`uvx --from "markitai[mcp]" markitai mcp`).

## Tools

| Tool | Purpose |
|------|---------|
| `convert_document` | One local file (absolute path) → Markdown |
| `convert_url` | One web page → clean main-content Markdown |
| `batch_convert` | Many paths/URLs in the background → returns a `job_id` |
| `job_status` | Progress and per-item results for a batch job |

Every conversion writes real files — into the `output_dir` the agent passes, or into a fresh temporary directory whose path is returned. Results inline the markdown text, but past ~40 KB they switch to a truncated preview plus `markdown_file`, the path to the complete output, so huge documents never flood the model context. All three conversion tools also take `profile` (`rag`, `obsidian` or `okf`) to shape the written output for a downstream consumer, and follow the server's config when it is omitted.

Batch jobs run inside the server process with an in-memory job table: poll `job_status` until `status` is `"completed"` (or `"cancelled"`), then read the `markdown_file` paths. `results[i]` answers the `sources[i]` you passed once the job is `"completed"`; running or cancelled jobs omit unfinished items, so identify partial results by `source`. Each item writes into `output_dir/batch-<job_id>/<item_number>/` (numbers start at `0001`), keeping same-named files and their assets separate even across simultaneous batches. `batch_convert` accepts `concurrency` (default 10) to bound how many conversions run at once. Jobs are forgotten as newer ones finish and on server restart — `job_status` says which case it is; the written files remain.

## LLM Enhancement

LLM enhancement is enabled per call with `llm: true`: omitting `llm` follows the server's own markitai config (`llm.enabled`, off by default), and `llm: false` forces it off. Pass `alt`/`desc` for image analysis and `ocr`/`screenshot` for those capabilities; enabling LLM can incur provider charges. It needs a configured model — either in `~/.markitai/config.json` ([shared with the CLI](/guide/configuration)), or via environment variables in the server entry:

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

Calling a tool with `llm: true` and no resolvable model fails with a readable error that repeats exactly this setup guidance, so agents can relay the fix.

Optional capabilities follow the markitai extras: OCR for scanned documents needs `markitai[ocr]`, URL screenshots need `markitai[browser]` — with uvx, add e.g. `"args": ["--from", "markitai[mcp]", "--with", "markitai[ocr]", "markitai-mcp"]`.

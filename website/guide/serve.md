# Web Workspace

`markitai serve` opens a local web page for converting files and URLs. Drop in files, paste links, watch progress, preview and download the results. It works in English and Chinese, on desktop and on a phone.

Everything runs on your machine. Jobs and history live on disk, and nothing leaves the host except what you send to the fetch strategies and LLM providers you configure.

![The markitai web workspace: a composer card with a URL field, a Convert button, and Options, CLI, and Upload toggles.](/workbench.png){.light-only}
![The markitai web workspace: a composer card with a URL field, a Convert button, and Options, CLI, and Upload toggles.](/workbench.dark.png){.dark-only}

## Starting the Server

Install the `serve` extra, then start it:

```bash
uv tool install "markitai[serve]" --force
markitai serve
```

The workspace opens in your browser at `http://127.0.0.1:3600`.

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to reach it from other devices |
| `--port` | `3600` | Port to listen on |
| `--no-open` | off | Do not open the browser |
| `--no-auth` | off | Disable the access token |
| `--allowed-host <hostname>` | — | Extra hostname to accept when you browse to a DNS name instead of an IP (repeatable) |

## Access Token

At startup the server prints a URL that ends in `#token=…`. Requests from this machine need no token. Every other device must open that URL; scripts send the token as `Authorization: Bearer <token>`.

Set `MARKITAI_SERVE_TOKEN` to keep the same token across restarts. `--no-auth` drops the token entirely: other devices can still upload, download and use history, but their URL targets are limited to public addresses and LLM settings are locked.

## Accessing from Other Devices

Bind all interfaces, then open the printed token URL on the phone or laptop:

```bash
markitai serve --host 0.0.0.0
```

If you reach the server through a DNS name rather than an IP, add `--allowed-host my-box.lan`. Unknown hostnames are rejected, which blocks DNS-rebinding attacks from malicious web pages.

::: warning
The token URL is a credential. Anyone holding it can run conversions with your LLM providers, and read, download or delete history. Share it only with devices you trust.
:::

## The Workspace

- **Composer**: drag in files or folders, or paste URLs. The default screen asks nothing. Every option sits behind **Options**, grouped as Preset, Enhance (LLM, OCR, image analysis), Output (profile) and Advanced (fetch strategy, cache, compression).
- **CLI preview**: the equivalent command line appears when Options is open or you press **CLI**, ready to copy.
- **Live progress**: each item streams its status, and a notification fires when a job finishes in a background tab.
- **Per-item actions**: retry a failed item, or LLM-enhance a finished one without reconverting the rest.
- **Preview**: rendered Markdown with a base vs enhanced comparison, plus a print-to-PDF menu with optional header and footer.
- **Downloads**: single files, a per-job zip, or the whole history as one archive.
- **Limits**: 50 items per job and 100 MB per uploaded file.

### Presets, Overrides and Commands

Presets match the CLI:

| Preset | LLM | alt | desc | screenshot | OCR |
|--------|:---:|:---:|:----:|:----------:|:---:|
| `minimal` | – | – | – | – | – |
| `standard` | ✓ | ✓ | ✓ | – | – |
| `rich` | ✓ | ✓ | ✓ | ✓ | – |

No preset turns on OCR; it is always a separate choice. Picking a preset resets those five switches. Changing one switch keeps the rest and shows **Custom**.

Every option has hover or tap help that says what it depends on, where the data goes and what it may cost.

## LLM Settings

The settings dialog edits the same configuration as `markitai config`: discover providers, browse model lists, set weights and test connections without exposing stored keys. Changes apply to web jobs and to later CLI runs alike.

## History

Finished jobs stay under `~/.markitai/serve/jobs/` for 7 days. From the history page you can reopen a job, download its outputs again, delete it, or download everything as one zip.

CLI runs started with [`--record-history`](/guide/cli#record-history) show up here too, marked with a CLI badge, and behave like any other job.

## API Overview

The UI runs on a small REST + SSE API that scripts can call directly:

| Endpoint | Description |
|----------|-------------|
| `GET /api/capabilities` | Server version, presets, LLM and extras status |
| `POST /api/jobs` | Create a job (multipart form: `files`, `urls` JSON array, `options` JSON) |
| `GET /api/jobs/{job_id}` | Job status and items |
| `GET /api/jobs/{job_id}/events` | Live progress stream (SSE) |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | Retry an item, or LLM-enhance it with `operation: "enhance"` |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | Remove an item from a job |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | Item result; sibling assets via `GET /api/jobs/{job_id}/files/{path}` |
| `GET /api/jobs/{job_id}/archive` | Download the job as a zip |
| `GET /api/history` | List history entries |
| `GET /api/history/archive` | Download all of history as one zip |
| `DELETE /api/history/{job_id}` | Delete one history entry |
| `/api/settings/llm*` | LLM provider, model and deployment management |

State-changing requests from another web origin are rejected, so a random web page cannot drive the server from your browser.

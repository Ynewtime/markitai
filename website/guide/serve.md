# Web Workspace

`markitai serve` starts a local web UI on top of the conversion core: upload files or folders, submit URLs, watch live progress, preview and download results, configure LLM providers, and revisit seven days of conversion history — all from a bilingual (EN/中文), accessible interface that also works on phones and narrow windows. Use it when you prefer a browser over the command line, want to compare base and LLM-enhanced output side by side, or prefer to manage LLM providers visually.

Everything runs on your machine: jobs and history live on disk, and nothing leaves the host except the fetch strategies and LLM providers you configure.

## Starting the Server

The server requires the `serve` extra (FastAPI + uvicorn):

```bash
uv tool install "markitai[serve]" --force
markitai serve
```

It listens on `http://127.0.0.1:3600` and opens your browser automatically once startup completes.

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `3600` | Port to listen on |
| `--no-open` | off | Do not open the browser after startup |
| `--no-auth` | off | Disable the access token (restores the pre-token remote policy) |
| `--allowed-host <hostname>` | — | Additional hostname to accept in the `Host`/`Origin` headers (repeatable) |

## Access Token

At startup the server generates a random access token and prints a ready-to-open URL like `http://192.168.1.50:3600/?token=mk_…`. Requests from this machine never need the token; requests from any other machine must carry it — the web UI picks it up from that URL automatically (and scrubs it from the address bar), scripts send `Authorization: Bearer <token>` (or `?token=` on download/SSE URLs). Set `MARKITAI_SERVE_TOKEN` to pin a fixed token across restarts; pass `--no-auth` to disable the token entirely, which limits other machines to public-URL conversions and blocks their access to LLM settings.

## Accessing from Other Devices

To use the workspace from another machine or a phone on the same network, bind all interfaces and open the printed token URL on that device:

```bash
markitai serve --host 0.0.0.0
```

Add `--allowed-host my-box.lan` if you browse to a DNS name instead of an IP: the server only accepts requests whose `Host`/`Origin` is localhost, an IP literal, or an allow-listed hostname. Any other DNS name is rejected, which blocks DNS-rebinding attacks where a malicious web page tries to reach the API from your browser.

::: warning
The token URL is a credential — anyone holding it gets full access: conversions with your configured LLM providers (including intranet URLs), history, downloads, deletion, and LLM settings. Share it only with devices you trust, and prefer the default loopback bind when you don't need LAN access.
:::

## The Workspace

- **Composer**: drag in files or folders, or paste URLs. Options mirror the CLI presets — `minimal`, `standard`, `rich` — plus individual LLM and OCR toggles and the output profile (`rag`, `obsidian`, `okf`). The profile is available with or without LLM.
- **Live progress**: each item streams its status as it converts; a notification fires when a job finishes in a background tab.
- **Per-item actions**: retry a failed item in place, or LLM-enhance a finished one without reconverting its siblings; filter large result sets to find an item quickly.
- **Preview**: rendered Markdown preview with a base vs LLM-enhanced comparison, and a PDF settings menu that prints the preview to a clean A4 document (optional custom header/footer).
- **Downloads**: individual output files, a per-job zip archive, or a whole-history archive — and a one-click copy of the equivalent CLI command for any job.
- **Limits**: up to 50 items per job and 100 MB per uploaded file.

## LLM Settings

The settings dialog manages the same configuration as `markitai config`: discover local and API-backed providers, browse live model lists, configure weighted deployments, and test connections without exposing stored credentials. Changes apply to both web jobs and subsequent CLI runs.

## History

Every finished job is kept on disk under `~/.markitai/serve/jobs/` for **7 days**, then cleaned up automatically. From the history page you can reopen a job to preview and download its outputs again, delete a single entry, or download everything as one zip.

History entries carry an origin. CLI runs recorded with [`--record-history`](/guide/cli#record-history) (or `MARKITAI_RECORD_HISTORY` / `history.record` in the config) appear alongside browser-created jobs with a small "CLI" badge and behave exactly like them — same TTL, deletion, and archive download — showing up live without restarting the server.

## API Overview

The UI is built on a small REST + SSE API that you can also drive from scripts:

| Endpoint | Description |
|----------|-------------|
| `GET /api/capabilities` | Server version, available presets, LLM and extras status |
| `POST /api/jobs` | Create a job (multipart form: `files`, `urls` JSON array, `options` JSON) |
| `GET /api/jobs/{job_id}` | Job status and items |
| `GET /api/jobs/{job_id}/events` | Live progress stream (SSE) |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | Retry an item, or LLM-enhance it with `operation: "enhance"` |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | Remove an item from a job |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | Item result; sibling assets via `GET /api/jobs/{job_id}/files/{path}` |
| `GET /api/jobs/{job_id}/archive` | Download the whole job as a zip |
| `GET /api/history` | List history entries |
| `GET /api/history/archive` | Download all of history as one zip |
| `DELETE /api/history/{job_id}` | Delete one history entry |
| `/api/settings/llm*` | LLM provider, model, and deployment management |

::: tip
The same `Host`/`Origin` allow-list guards the API: state-changing requests with a cross-site origin are rejected, so an arbitrary web page cannot drive it from your browser.
:::

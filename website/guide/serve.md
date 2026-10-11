# Web Workspace

`markitai serve` opens a local web workbench for converting files and URLs. Drop in files or a folder, paste links, watch progress, inspect and preview each result next to its original, and download everything. It works in English, Chinese and Japanese, on desktop and on a phone.

Everything runs on your machine. Jobs and history live on disk, and nothing leaves the host except what you send to the fetch strategies and LLM providers you configure.

![The markitai web workbench: a sidebar with Convert, Conversions and Settings, the list of converted files grouped by submission, and an inspector showing the selected file's status, sizes, cost and options.](/workbench.png){.light-only}
![The markitai web workbench: a sidebar with Convert, Conversions and Settings, the list of converted files grouped by submission, and an inspector showing the selected file's status, sizes, cost and options.](/workbench.dark.png){.dark-only}

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

At startup the server prints a URL that ends in `#token=…`, and the browser it opens uses that URL. Requests from this machine need no token, except for the LLM settings (`/api/settings/llm…`), which hold your provider keys: any local program or a page on another `localhost` port also connects from this machine, so those routes always need the token. Every other device must open that URL; scripts send the token as `Authorization: Bearer <token>`. A request whose client address the server can't see counts as coming from another device.

A tab opened without the token asks for it where it is needed (**Settings → Access token** holds it for that tab). Downloads opened by the browser use a single-use link that expires after a minute, so the token itself never appears in a URL, the browser history or the download list.

Set `MARKITAI_SERVE_TOKEN` to keep the same token across restarts. `--no-auth` drops the token entirely: other devices can still upload, download and use history, but they cannot submit URLs (public or private) and LLM settings are locked. This machine then reaches the LLM settings without a token.

## Accessing from Other Devices

Bind all interfaces, then open the printed token URL on the phone or laptop:

```bash
markitai serve --host 0.0.0.0
```

If you reach the server through a DNS name rather than an IP, add `--allowed-host my-box.lan`. The server rejects unknown hostnames, which blocks DNS-rebinding attacks from malicious web pages.

::: warning
The token URL is a credential. Anyone holding it can run conversions with your LLM providers, and read, download or delete history. Share it only with devices you trust.
:::

## The Workspace

The sidebar has three places: **Convert**, **Conversions** and **Settings** (Models, Appearance and language, Access token). On a phone they move to a bar at the bottom, and details open as sheets.

- **Composer**: paste URLs (one per line), drop files or a folder, or use **Upload** (its menu also picks a folder). Files are staged until you press **Convert**, with upload progress, and can be combined with URLs in one submission. A `.urls` file is read as a list of URLs, as on the command line. URLs are checked before anything is sent.
- **Options**: nothing is asked by default. **Options** holds the preset, Enhance (LLM, OCR, image analysis), Output (profile) and Advanced (fetch strategy, files backend, cache, compression), with the equivalent command line ready to copy.
- **Conversions**: one list of every item, grouped by submission (named after the folder or `.urls` list it came from). It updates live; filter it by name or by All, Failed and Notices. **Stop remaining** stops the items of a running submission that have not started; the ones already converting finish, and stopped items can be retried later. A notification fires when a submission finishes in a background tab.
- **Inspector**: select an item to see its status, the sizes of its source and output, the models used and the cost, its options, how long it is kept and its notices, with the start of the result. From there: **Preview**, **Compare**, **Download .md**, **Run again** or **Retry**, **Enhance with LLM**, or delete it. Retry is not offered for a file type markitai cannot convert. When an action on the item fails (a Retry the server refuses, say), the inspector says so at its top.
- **Cost**: a cost is shown only when it is known. When the provider has no price for a model, the item says the cost is unknown instead of `$0`, and when only some requests were priced it shows "at least" that amount.
- **Preview**: the result opens beside the list (**F** for full width), with Rendered, Source, Files and, for LLM-enhanced items, Diff against the unenhanced Markdown, plus Export PDF with an optional header and footer. Large tables show their first 500 rows at once (**Show all rows** adds the rest); code blocks are highlighted.
- **Compare**: shows the original next to the Markdown: the pages of a PDF (hover a heading or table in the Markdown to outline it on the page), an image, an HTML or text document, or the page screenshots a conversion saved (Office files, web pages). Formats a browser cannot show offer **Download the original** instead; URL items keep no copy of the page.
- **Notices**: things that don't fail an item but are worth acting on (pages that look scanned, hidden PDF text that may be a prompt injection, OCR that found no text, slides that couldn't be rendered, a URL screenshot that wasn't captured, images whose analysis failed and kept their original alt text, an LLM enhancement that failed and left the unenhanced result) are counted on the item and listed in its inspector; repeats of one notice are grouped. Each item gets only its own notices, and they stay with the job in history.
- **Downloads**: single files, a submission's zip, or everything with **Download all**.
- **Limits**: 1000 items per job, 100 MB per uploaded file and 5 GB per upload.

### Presets, Overrides and Commands

Presets match the CLI:

| Preset | LLM | alt | desc | screenshot | OCR |
|--------|:---:|:---:|:----:|:----------:|:---:|
| `minimal` | – | – | – | – | – |
| `standard` | ✓ | ✓ | ✓ | – | – |
| `rich` | ✓ | ✓ | ✓ | ✓ | – |

No preset turns on OCR; you always pick that separately. Picking a preset resets those five switches. Changing one switch keeps the rest and shows **Custom**.

Every option has hover or tap help that says what it depends on, where the data goes and what it may cost.

## LLM Settings

**Settings → Models** edits the same configuration as `markitai config`: discover providers, browse model lists, set weights and test connections without exposing stored keys. **Add models** opens a provider picker: search by name or model prefix, the common providers as tiles, the others under filters (Clouds, Inference, Gateways, Local, Subscriptions, Model makers). It offers every provider markitai can route through LiteLLM, about 90, each with the address it uses and the key variable it reads; enter a key (and an address, for servers of your own), then pick from the provider's model list, or type model IDs where the provider publishes none. Changes apply to web jobs and to later CLI runs alike.

The provider editor shows a saved key only in masked form (`sk…7890`; an `env:` reference is shown as is). Leave the field as it is to keep the saved key, type a new key to replace it, or clear it to remove the key. A saved API base address is shown without its credentials (`https://…@proxy.example/v1?api_key=…`): left as shown it keeps the saved address; to change it, type the whole address.

A key is tied to the address it was saved with, so the settings page never sends a key you didn't type to a server you picked. It refuses (HTTP 422) to pair an `env:` reference with a custom address, to use an `env:` variable other than the provider key variables (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, ...), and to change the address of a connection without re-entering its key. To use such a pairing, either enter the key itself together with the new address, or write the pair in the config file (`markitai config edit`, or edit it by hand; see [Custom API Endpoint](/guide/configuration#custom-api-endpoint)). Pairs already in the config file keep working from the settings page.

A connection saved here without a key while it names an address (a local server, a gateway) stores the placeholder key `markitai-no-key`, which no server accepts as a credential: without it, LiteLLM would read the server's own key variable (`OPENAI_API_KEY`, `GROQ_API_KEY`, ...) and send it to that address. A server that checks no key (LM Studio, vLLM, llama.cpp) works with it; the settings page shows such a connection as having no key. A connection with a key cannot be turned into one without a key at another address (HTTP 422): add a new connection for that server. Entries you write in the config file yourself are unchanged: one without `api_key` still uses the provider's key variable from the environment.

## History

Finished jobs stay under `~/.markitai/serve/jobs/` for 7 days after they finish (the inspector shows the date); expired jobs disappear from the list at once and their files are deleted within the hour. They stay in **Conversions** after a reload or a restart: reopen them, download their outputs again, retry or enhance their items, delete them, or download everything as one zip.

CLI runs started with [`--record-history`](/guide/cli#record-history) show up here too, marked with a CLI badge. You can open, download and delete them like any other job, and retry or LLM-enhance their URL items. File items from a CLI run can't be retried or enhanced here, because the run keeps only the outputs and not the original file; run the CLI on the file again instead.

An `--llm-batch` run that stopped waiting while its provider batch was still running records those items as skipped with the reason *LLM batch still running*. Only the base Markdown is in history so far. Finish them with the `markitai --llm-batch-collect` command the run printed, instead of enhancing them here, which would pay for the same enhancement twice.

## API Overview

The workbench runs on a small REST + SSE API that scripts can call directly. Errors are JSON `{"detail", "code", "reason"}`: `detail` is an English sentence, `reason` a stable cause such as `llm_unavailable` or `file_too_large`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/capabilities` | Server version, presets, LLM and extras status, limits and supported features |
| `POST /api/jobs` | Create a job (multipart form: `files`, `urls` JSON array, `options` JSON, optional `label`). A `.urls` file becomes URL items. Asking for LLM processing without a usable model is refused (`llm_unavailable`) |
| `GET /api/jobs/{job_id}` | Job status and items (each item carries its `warnings`, `error_code` and, when a model ran, `pricing`) |
| `GET /api/jobs/{job_id}/events` | Live progress stream (SSE) |
| `POST /api/jobs/{job_id}/cancel` | Stop the items still waiting to start |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | Retry an item, or LLM-enhance it with `operation: "enhance"` |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | Permanently remove an item with its upload, outputs, images and screenshots |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | Item result (`?max_bytes=` for an excerpt); sibling assets via `GET /api/jobs/{job_id}/files/{path}` |
| `GET /api/jobs/{job_id}/items/{item_id}/assets/{path}` | One of the item's own extracted images (PNG, JPEG, GIF, WebP, AVIF), inline, for display |
| `GET /api/jobs/{job_id}/items/{item_id}/original` | What can be shown of the item's original; `/original/file`, `/original/pages/{n}?width=` and `/original/pages/{n}/text` serve it |
| `GET /api/jobs/{job_id}/archive` | Download the job as a zip |
| `POST /api/download-tickets` | A single-use, one-minute `?ticket=` link for one download |
| `GET /api/history` | List history entries (`?include=items` adds every item) |
| `GET /api/history/archive` | Download all of history as one zip |
| `DELETE /api/history/{job_id}` | Delete one history entry |
| `/api/settings/llm*` | LLM provider, model and deployment management |

The server rejects state-changing requests from another web origin, so a random web page cannot drive it from your browser. Cloudflare processing (`strategy` or `backend` `"cloudflare"`) needs `"remote_processing": "cloudflare"` in the same request's options, as the workbench sends after you confirm it, and is refused for callers without the token.

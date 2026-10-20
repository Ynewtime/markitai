# Web Workspace

`markitai serve` opens a local web workbench for converting files and URLs. Drop in files or a folder, paste links, watch progress, inspect and preview each result next to its original, and download everything. It works in English, Chinese and Japanese, on desktop and on a phone.

Everything runs on your machine. Jobs and history live on disk, and nothing leaves the host except what you send to the fetch strategies and LLM providers you configure.

![The markitai web workbench: a converted PDF open in a window centred over the dimmed list of conversions, on its Preview tab, showing the rendered Markdown of a U.S. petroleum balance sheet table.](/workbench.png){.light-only}
![The markitai web workbench: a converted PDF open in a window centred over the dimmed list of conversions, on its Preview tab, showing the rendered Markdown of a U.S. petroleum balance sheet table.](/workbench.dark.png){.dark-only}

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

**Settings → Access token → Regenerate token…** replaces the token: every other tab and device (on this machine too) must sign in again, and unused download links stop working. The new token is shown once, with its sign-in link and a QR code; like the startup token it lasts until the server stops. **Sign out of this tab** only forgets the token in that tab.

**Settings → Access token → User tokens** gives other people or devices a token of their own: name it and **Create token**; it is shown once, with Copy, its sign-in link and a QR code, then listed by its last characters with when it was last used. A user token converts files and URLs (public addresses only: its URLs, and the pictures its documents link to, are not fetched from the server's own network) and sees, downloads, stops, retries and deletes only its own conversions; Models, Access token and Edit config need the server's token (the admin token, or this machine without a token), and a tab signed in with a user token says so there and shows Appearance and its own Models under Settings (see [Users' Models and Shared Models](#users-models-and-shared-models)). The admin sees everyone's conversions. **Revoke** signs a user token out at once, with its download links and live progress, also on this machine. Only a SHA-256 digest of each user token is kept, in `~/.markitai/serve/access.json` (readable by your user only, never the config file), so user tokens survive restarts and a regenerated admin token; the server's log names a token by its label, never the token.

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

The sidebar has three places: **Convert**, **Conversions** and **Settings** (Appearance and language, Models, Access token, Edit config). On a phone they move to a bar at the bottom, and a conversion opens full screen. Under them a line says which build of the server answers: its version, the commit it was built from (`-dirty` when the build had uncommitted changes) and when it was built, in your time zone (`1.2.1 1a2b3c4 10-15 09:30`; hover it for the full time). A server run from a source checkout has no build time.

- **Composer**: paste URLs (one per line), drop files or a folder, or use **Upload** (its menu also picks a folder). Files are staged until you press **Convert**, with upload progress, and can be combined with URLs in one submission. A `.urls` file is read as a list of URLs, as on the command line. URLs are checked before anything is sent.
- **Options**: nothing is asked by default. **Options** holds the preset, Enhance (LLM, OCR, image analysis), Output (profile) and Advanced (fetch strategy, files backend, cache, compression), with the equivalent command line ready to copy. The options start from **the server's defaults** (`config.json` over the built-in defaults: the keys `llm.enabled`, `ocr.enabled`, `image.alt_enabled`, `image.desc_enabled`, `screenshot.enabled`, `screenshot.screenshot_only`, `llm.pure`, `cache.no_cache`, `image.compress`, `output.profile` and `fetch.strategy`, which the command line reads too). A row above the command line says whether they are the server's, yours or changed, and has **Reset to defaults** and **Save as default**: the admin's choices are written to `config.json` and apply to the next conversion at once, a user token's are kept in its own file (**Use the server's defaults** undoes them). Cloudflare is never a default.
- **Conversions**: one list of every item, grouped by submission (named after the folder or `.urls` list it came from). It updates live; filter it by name or by All, Failed and Notices. **New conversion** leads back to Convert, and "⋯" holds **Retry failed** and **Download all**. To delete several at once, check them with the box at the left of a row or a submission's header (Space checks the focused row, Shift extends the selection, Ctrl/⌘+A checks every listed row, Escape clears; on a phone, tap **Select** first), then **Delete** in the bar that appears and confirm; items still converting cannot be checked. **Stop remaining** stops the items of a running submission that have not started; the ones already converting finish, and stopped items can be retried later. On a phone it stays in the header of the full-screen window that covers the list. A notification fires when a submission finishes in a background tab.
- **The conversion window**: click an item (or press Enter on it) to open it in a window over the list, with three tabs: **Details**, **Preview** and **Compare**. ↑/↓ or the arrows in its header move to the previous or next item, **C** switches to Compare and Escape closes it. Its expand button (or **F**) makes it fill the browser and back (each browser keeps the choice); on a phone it fills the screen. **Details** shows the item's status, the sizes of its source and output, the models used and the cost, its options, how long it is kept and its notices, with the start of the result. The header has **Download**, which opens Preview's **Files**, and, when it is the fix, **Retry**; "⋯" holds **Enhance with LLM**, **Run again** and Delete. Retry is not offered for a file type markitai cannot convert. When an action on the item fails (a Retry the server refuses, say), Details says so at its top.
- **Cost**: a cost is shown only when it is known. When the provider has no price for a model, the item says the cost is unknown instead of `$0`, and when only some requests were priced it shows "at least" that amount.
- **Preview**: the window's second tab, with Rendered, Source, Files and, for LLM-enhanced items, Diff against the unenhanced Markdown; its "⋯" adds Copy and Export PDF with an optional header and footer. **Files** lists the original first, then every result file, each with its own download, and **Download all (.zip)** of the result with its original. Pictures on the web (absolute `https://` addresses in the Markdown) are drawn, without sending a referrer; other addresses show a labelled placeholder. A video written as a linked poster (`[![Video](poster)](https://…/clip.mp4)`; `.mp4`, `.m4v`, `.webm`, `.mov`, `.ogv`) plays in place: nothing is loaded until you play it, and one that cannot play turns back into its link. A table row broken by a line break left in a cell is put back together; Source keeps the text as written. Large tables show their first 500 rows at once (**Show all rows** adds the rest); code blocks are highlighted.
- **Compare**: shows the original next to the Markdown: the pages of a PDF (hover a heading or table in the Markdown to outline it on the page), an image, an HTML or text document, or the page screenshots a conversion saved (Office files, web pages). Formats a browser cannot show offer **Download the original** instead. A web page is shown as the server fetched it, in a sandboxed frame: with the site's own pictures, style sheets and fonts (your browser loads them from the site over HTTPS), none of its scripts, links opening in a new tab (with the screenshot beside it when one was saved); hovering a heading, paragraph, list item, table or picture on either side outlines its counterpart on the other, on the page and on its screenshot alike. A page the browser loaded keeps a screenshot even when none was asked for (a compare copy, never part of the output and never sent to a model). Compare shows only what shows the page: when the site answered with a verification check or a blank page, it says so and offers to open the page instead. Pages fetched through Jina, Defuddle or the fetch cache are not kept; their screenshot stands in.
- **Notices**: things that don't fail an item but are worth acting on (pages that look scanned, hidden PDF text that may be a prompt injection, OCR that found no text, slides that couldn't be rendered, a URL screenshot that wasn't captured, images whose analysis failed and kept their original alt text, an LLM enhancement that failed and left the unenhanced result) are counted on the item and listed in its Details; repeats of one notice are grouped, and page lists read as ranges (`pages 3–5, 9`). A PDF whose pages are at least nine in ten without readable text says "This PDF is scanned: turn on OCR and convert again"; otherwise the pages without text are named ("Pages 3–5, 9 (4 of 120) have no readable text…"). A model answer that did not match the document page by page is explained in those words, and the converted Markdown is kept. Each item gets only its own notices, and they stay with the job in history.
- **Downloads**: single files, a submission's zip, or everything with **Download all** (in the Conversions "⋯").
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

When the server has a provider's key variable set (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `MOONSHOT_API_KEY`, ... any provider the picker offers), the picker lists it with the saved connections ("Using OPENAI_API_KEY"); a connection that already uses that variable at the provider's own endpoint (a model written into the config without a key, for example) stands for it, so each provider is one saved row, and the common tiles mark it **Added**. Its models are listed with that key from the provider's own endpoint, and a model added from it is saved with the reference `env:OPENAI_API_KEY`, never the key itself; that saved connection then takes the card's place. The key only goes to its provider's endpoint: to use another address, connect the provider with a key you enter.

A model test says why it failed: no API key (the key variable is not set on the server; then nothing is sent: set it where `markitai serve` runs and restart it, or add a key to the connection), the key was rejected (HTTP 401/403), the provider does not offer the model, is limiting requests or out of quota, could not be reached, did not answer in time, or returned another error. The provider's own words are under "Provider's message". The API answers with the same category as `reason` (`no_key`, `key_rejected`, `model_not_found`, `rate_limited`, `network`, `timeout`, `provider_error`) and, for a missing key, the variable's name as `key_variable`.

The provider editor shows a saved key only in masked form (`sk…7890`; an `env:` reference is shown as is). Leave the field as it is to keep the saved key, type a new key to replace it, or clear it to remove the key. A saved API base address is shown without its credentials (`https://…@proxy.example/v1?api_key=…`): left as shown it keeps the saved address; to change it, type the whole address.

A key is tied to the address it was saved with, so the settings page never sends a key you didn't type to a server you picked. It refuses (HTTP 422) to pair an `env:` reference with a custom address, to use an `env:` variable other than the provider key variables (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, ...), and to change the address of a connection without re-entering its key. To use such a pairing, either enter the key itself together with the new address, or write the pair in the config file (`markitai config edit`, or edit it by hand; see [Custom API Endpoint](/guide/configuration#custom-api-endpoint)). Pairs already in the config file keep working from the settings page.

A connection saved here without a key while it names an address (a local server, a gateway) stores the placeholder key `markitai-no-key`, which no server accepts as a credential: without it, LiteLLM would read the server's own key variable (`OPENAI_API_KEY`, `GROQ_API_KEY`, ...) and send it to that address. A server that checks no key (LM Studio, vLLM, llama.cpp) works with it; the settings page shows such a connection as having no key. A connection with a key cannot be turned into one without a key at another address (HTTP 422): add a new connection for that server. Entries you write in the config file yourself are unchanged: one without `api_key` still uses the provider's key variable from the environment.

Vertex AI (and watsonx and GigaChat) sign in with the server's own login, not a key: Google's application default credentials (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`), `WATSONX_TOKEN`, `GIGACHAT_ACCESS_TOKEN`. LiteLLM sends that login to whatever address the entry names, so the settings page saves and tests these models only without an address, at the provider's own endpoint (HTTP 422 otherwise): **Add models** shows Vertex AI without an address field, with a note that it uses the server's Google login; a private endpoint goes in the config file. Amazon Bedrock may take an address: without a key it sends the placeholder there, never the server's AWS credentials.

### Users' Models and Shared Models

A user token manages models of its own under **Settings → Models** (the same page, without the config file line and detected models): Add models, Test, Edit, Delete and weights. On every saved row the weight is edited in place: click it, type a whole number from 0 to 100, press Enter or leave the field to save, Escape to cancel (the admin's rows work the same way). They are kept in `~/.markitai/serve/user-models/<token id>.json` (readable by your user only, never the config file, so Edit config and the CLI never see them), are used only by that user's conversions, and their keys are shown masked only. A user may add only a model with its own API key, or the address of a server that takes none (stored with `markitai-no-key`); the server refuses a provider that would sign in with its own account or cloud login (Claude Code, Copilot and ChatGPT subscriptions, Vertex AI, watsonx, GigaChat, SageMaker, Bedrock without a key), an `env:` reference, and an address on the server's own machine (`localhost`, `127.0.0.1`, `::1`, `0.0.0.0`; so Ollama or LM Studio need an address of the user's own); 50 models at most. Beyond that check, every request made for a user's own model (its conversions, model discovery and tests) reaches public addresses only: the name is resolved when the connection is made and an address outside the public internet (loopback, private, link-local such as cloud metadata, unique-local) is refused, so a name that points at the server's machine or network fails too; no redirect is followed, and a request an environment proxy would carry is refused, because the proxy would choose the address. Models the admin shares are not restricted.

The admin's saved models are **Private** by default; **Shared** lets every user's conversions use one (saved as `"model_info": {"share": "shared"}` in the config file). Users see a shared model under **Shared by the admin** as its model ID, provider and weight only: its key and address stay on the server, and a failure a user sees never names where it is served. A user's conversions route over one pool, its own models and the shared ones by weight, a failed request moving on to another model of the pool; prompts, budgets and the cache stay the admin's, and the job shows which model ran and what it cost. A user with neither has no LLM options, and a request for one is refused (`llm_unavailable`): the server's environment keys and detected models never serve a user. Turning a model private applies to the next request.

Each shared row has **Test** and a weight of the user's own. The weight applies to that user's conversions only: 0 turns the shared model off for that user, other users and the admin are not affected, and **Use the admin's N** follows the admin's weight again (it is kept in the user's own file as `shared_weights`). Test runs on the server with the admin's key and address and says only whether the model passed and the reason when it did not ("ask the admin to check it"), never the provider's words, the address or the key; because the admin's key pays, a user runs one such test at a time and at most six a minute.

A user token's **Settings → Edit config** edits its own models as text: one JSON document with `models` and `providers` (its entries, as `llm.model_list` and `llm.providers` would hold them) and `shared` (each shared model with the user's weight for it), keys masked as on the admin's page, with the same check, Review and conflict flow. The server's configuration file is never shown to a user or changed by one; the rules above apply to what a user saves. **Settings → Access token** shows how many models each user has; revoking a token deletes them with their keys (a server stopped during that deletes them at its next start).

## Edit Config

**Settings → Edit config** shows the configuration file the server reads (the one `markitai config` edits) as text, with its secrets masked as `markitai config show` redacts them: API keys, tokens, passwords, cookies and other values under a secret-sounding key, and every HTTP header value, read `sk…7890`; an `env:` reference is shown as it is; a saved address shows without its credentials. The JSON is checked as you type (with its line and column), **Review changes** shows a line diff, and **Save config** writes exactly the reviewed text, keeping its layout, key order and line endings (a file written on Windows stays CRLF; the page says so). A masked value left where it was keeps the stored secret; to change one, type the whole new value or an `env:` reference. If the file changed since the page read it, the save is refused and **Load current file** reads it again. Models and connections apply at once; the page says when other changes need a restart.

## History

Finished jobs stay under `~/.markitai/serve/jobs/` for 7 days after they finish (its Details tab shows the date); expired jobs disappear from the list at once and their files are deleted within the hour. They stay in **Conversions** after a reload or a restart: reopen them, download their outputs again, retry or enhance their items, delete them, or download everything as one zip.

CLI runs started with [`--record-history`](/guide/cli#record-history) show up here too, marked with a CLI badge. You can open, download and delete them like any other job, and retry or LLM-enhance their URL items. File items from a CLI run can't be retried or enhanced here, because the run keeps only the outputs and not the original file; run the CLI on the file again instead.

An `--llm-batch` run that stopped waiting while its provider batch was still running records those items as skipped with the reason *LLM batch still running*. Only the base Markdown is in history so far. Finish them with the `markitai --llm-batch-collect` command the run printed, instead of enhancing them here, which would pay for the same enhancement twice.

## API Overview

The workbench runs on a small REST + SSE API that scripts can call directly. Errors are JSON `{"detail", "code", "reason"}`: `detail` is an English sentence, `reason` a stable cause such as `llm_unavailable` or `file_too_large`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/capabilities` | Server version and build (`build: {version, commit, built_at, dirty}`), presets, LLM and extras status, limits and supported features |
| `POST /api/jobs` | Create a job (multipart form: `files`, `urls` JSON array, `options` JSON, optional `label`). A `.urls` file becomes URL items. Asking for LLM processing without a usable model is refused (`llm_unavailable`) |
| `GET /api/jobs/{job_id}` | Job status and items (each item carries its `warnings`, `error_code`, `page_sources` (where each PDF page's text came from; empty for other items) and, when a model ran, `pricing`) |
| `GET /api/jobs/{job_id}/events` | Live progress stream (SSE) |
| `POST /api/jobs/{job_id}/cancel` | Stop the items still waiting to start |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | Retry an item, or LLM-enhance it with `operation: "enhance"` |
| `DELETE /api/jobs/{job_id}/items/{item_id}` | Permanently remove an item with its upload, outputs, images and screenshots |
| `GET /api/jobs/{job_id}/items/{item_id}/result` | Item result (`?max_bytes=` for an excerpt); sibling assets via `GET /api/jobs/{job_id}/files/{path}` |
| `GET /api/jobs/{job_id}/items/{item_id}/assets/{path}` | One of the item's own extracted images (PNG, JPEG, GIF, WebP, AVIF), inline, for display |
| `GET /api/jobs/{job_id}/items/{item_id}/original` | What can be shown of the item's original (for a web page also how usable its copy and screenshot were, `capture`); `/original/file`, `/original/view`, `/original/pages/{n}?width=`, `/original/pages/{n}/text`, `/original/outline` (a PDF's bookmarks) and `/original/blocks` (where a web page's blocks are on its screenshots) serve it |
| `GET /api/jobs/{job_id}/archive` | Download the job as a zip |
| `POST /api/download-tickets` | A single-use, one-minute `?ticket=` link for one download |
| `GET /api/history` | List history entries (`?include=items` adds every item) |
| `GET /api/history/archive` | Download all of history as one zip |
| `DELETE /api/history/{job_id}` | Delete one history entry |
| `POST /api/history/delete` | Delete several jobs and items in one request (`{"jobs": [...], "items": [{"job_id", "item_id"}]}`, up to 1000), each as its single route would; every outcome in the answer |
| `/api/settings/llm*` | LLM provider, model and deployment management |
| `GET`/`PUT /api/convert-defaults` | The options the composer starts from (every caller; `PUT` `{"expected_revision", "defaults"}`: the admin's go to `config.json`, a user token's to its own file, `null` removes them) |
| `GET`/`PUT /api/settings/config` | The config file as text, secrets masked; saved as sent against its `revision` |
| `GET`/`POST /api/access-tokens`, `DELETE /api/access-tokens/{id}` | User tokens (admin only; the list says each one's `model_count`) |
| `/api/me/models*` | A user token's own models, with the suffixes and rules of `/api/settings/llm*` (403 `user_token_required` for anyone else); `PATCH /api/me/models/shared/{id}` sets its weight for a shared model, `POST /api/me/models/test` with `{"shared_id"}` tests one |
| `GET`/`PUT /api/me/config` | A user token's own models as one masked JSON document (`models`, `providers`, `shared`), saved against the Models page's `revision` |

The server rejects state-changing requests from another web origin, so a random web page cannot drive it from your browser. Cloudflare processing (`strategy` or `backend` `"cloudflare"`) needs `"remote_processing": "cloudflare"` in the same request's options, as the workbench sends after you confirm it, and, since it bills the server's account, is refused (403 `remote_processing_forbidden`) for anyone but the admin: callers without the token and user tokens. A user token gets 403 `settings_forbidden` on the settings and token routes and 404 `job_not_found` for any conversion it did not create; `GET /api/capabilities` says who is asking (`access.role`: `admin` or `user`).

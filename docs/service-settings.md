# Service settings

The native service manages saved LLM deployments and provider connections through
`/api/settings/llm`. Changes apply to future jobs and retries; running conversions
retain the configuration snapshot they started with. Saving models does not turn
on `llm.enabled` or rewrite unrelated configuration fields.

The selected file is fixed at startup: explicit `--config`, `MARKITAI_CONFIG`,
project `markitai.json`, then the user configuration under `MARKITAI_HOME` (or the
normal user state directory). A missing default file is created on the first
save. Responses identify `config_path` and `config_origin`.

## Endpoints

| Method | Path after `/api/settings/llm` | Purpose |
|---|---|---|
| GET | empty | Revision and secret-free saved/detected deployments |
| GET | `/providers` | Common, environment and saved connection cards |
| GET | `/providers/{id}/credentials` | Explicitly requested editable saved values |
| POST | `/models` | Add one deployment; legacy compatible |
| PUT / DELETE | `/models/{model_name}` | Change one unambiguous routing group |
| POST | `/deployments/batch` | Atomically add up to 50 deployments and remove saved ones (`remove`), all or nothing |
| PATCH / DELETE | `/deployments/{id}` | Change a stable deployment identity |
| PATCH / DELETE | `/providers/{id}` | Change a connection or remove it and its deployments |
| POST | `/config/open` | Ask the server host's system opener to open the selected file |

Provider detection, live discovery and the transient connection test use
`/detected`, `/model-discovery` and `/test`; those operations are separate from
saving settings. They do not persist a connection draft. A failed `/test` answers
`{ok: false, detail, reason}` with the category of the failure (see
[LLM](llm.md)).

A routing weight is a whole number from 0 to 100 (0 = off; the model stays
saved): every write of one outside that range, Edit config included, is 422
`invalid_weight` (a weight the file already held for the same entry may stay).
`POST /deployments/batch` takes `{expected_revision, deployments?, remove?}`:
the ids in `remove` and every addition's credential reference name deployments of
the list `expected_revision` identifies, before the change (so the model a
connection card stands for can be removed while another is added through that
card); removals apply first, and an unknown id (404 `settings_entry_missing`), a
repeated id (422) or a stale revision (409) saves nothing. Views report
`batch_remove: true`.

The deployment view contains `deployment_id`, `routing_group`, `model`, `weight`,
credential-presence flags, a sanitized API origin, `persisted` and, on saved rows,
`share` (`private` or `shared`). `PATCH /deployments/{id}` accepts `share` alone
(with `expected_revision`) or with the other fields: `shared` saves
`model_info.share: "shared"` in the entry and lets user tokens' conversions use
the model, `private` removes it (see
[users' own models](serve.md#users-own-models-and-shared-models)). Users' own
models live outside this file under `/api/me/models`, with the same shapes; a
user's own weight for a shared model, its test of one and its Edit config
(`/api/me/config`) change only the user's file, never this one or the admin's
routing. A routing group
may contain multiple deployments. Legacy name-based changes return 409 when the
name is ambiguous. New deployments receive UUID identities; the first v2 mutation
also assigns persistent identities to legacy entries.

An OpenAI-compatible endpoint is saved with `provider: "custom"`: its models
keep the `openai/` prefix (`openai/<id>`), and it needs an `api_base`, sent or
inherited from the saved connection or deployment it reuses. The connection is
saved with provider `custom`, and its models route to that address. A further
model of the endpoint may name only `credential_provider_id`; it stays with the
endpoint. Every other provider must be the model's own prefix, otherwise 422
`invalid_settings_request`. The provider list keeps offering the OpenAI and
OpenAI-compatible endpoint cards after such a connection is saved, so another
endpoint, or OpenAI itself, can still be added; a saved connection of any other
provider takes the place of that provider's card.

For example, a batch request contains `expected_revision` and `deployments`:

```json
{
  "expected_revision": "revision from GET /api/settings/llm",
  "deployments": [
    {
      "model_name": "default",
      "model": "openai/my-model",
      "credential_provider_id": "env:openai"
    }
  ]
}
```

V2 deployment/provider changes require the revision. Two clients submitting the
same revision cannot both overwrite the configuration: a stale request receives
409 with `detail.code = "stale_revision"` and `detail.current_revision`. Reload the
view and reconcile the user's draft before retrying. Revisions hash the raw model
and provider arrays, including order, defaults explicitly present and unknown
fields; secret values contribute to the hash but are never returned in the view.

## Credentials and partial updates

HTTP `api_key` and `api_base` fields accept literal values only: client `env:`
references are rejected before any environment lookup. Server configuration files
may still use them. Discovery can select a saved `provider_id`/`deployment_id`, or
a known environment card such as `provider_id: "env:openai"`; adding a deployment
uses `credential_provider_id`/`credential_deployment_id`. Environment card IDs are
catalog provider names, never arbitrary environment variable names.

A server credential is bound to its provider and full configured endpoint.
Changing the provider or endpoint requires an explicit replacement key or
`api_key: null`; changing the endpoint without either returns 422. Default ports
and trailing slashes may normalize, but paths and query strings still matter.
An implicit server endpoint cannot be replaced with a client-supplied URL.
Custom endpoints without credential references use only the caller's literal key;
they do not borrow the server's default key or a different saved model's key.
Literal/cleared connections persist `use_environment_credentials: false` so
later conversions and restarts preserve that boundary. This internal setting is
not accepted in HTTP requests; old server configuration retains its environment
fallback unless the flag is explicitly false.

A deployment or connection saved here without a key while it names an address
(a local server, a gateway, an OpenAI-compatible endpoint) stores
`"api_key": "markitai-no-key"`, the placeholder Python markitai 1.2.1 stores too,
so neither backend sends the server's own key variable (`OPENAI_API_KEY`,
`LM_STUDIO_API_KEY`, ...) to an address a settings caller chose. It is no
credential: conversions and connection tests send it as written, which a server
that checks no key (LM Studio, vLLM, llama.cpp, Ollama) accepts, model discovery
sends no key, and an entry holding it never uses the environment's credentials
(Bedrock's and Vertex AI's included). Views show it as no key (`api_key_masked`
null, `api_key_configured` false), and such an endpoint counts as routable. A
draft connection test at an address the caller names uses it too. Changing a
connection or deployment that has a key into one without a key at another host
in one request is 422 `invalid_settings_request` with the detail `a connection
with a key cannot become one without a key at another address; add a new
connection for that server instead`; clearing the key in place, or together with
the address (back to the provider's own endpoint), is allowed. Entries written by
hand in the configuration file are never given the placeholder: without
`api_key` they use the provider's key variable wherever their `api_base` points.

A Bedrock or Vertex AI entry saved here with neither a key nor an address uses
the server's own cloud credentials (the AWS chain, the Google login) at the
cloud's own endpoint. Vertex AI signs every request with the server's Google
login whatever key an entry carries, so a `vertex_ai/` or `vertex_ai_beta/`
model is refused an address on every write and on a draft connection test (422
`invalid_settings_request`, detail `api_base: Vertex AI models send the server's
own Google credentials to any address, so here they can only use Vertex AI's own
endpoint; set another address in the config file`, the same words as Python
markitai 1.2.1's; `watsonx/` and `gigachat/`, which only Python routes, name
their token). Another address for Vertex AI is set in the configuration file.
Bedrock keeps its address: there it sends the placeholder as its Bedrock API key
and never signs with the server's AWS credentials.

The workbench leaves the key input blank when editing a saved connection. Blank
means keep; the row's Remove action deletes it. Stored values and
references are never copied into editable key inputs or echoed in updates.

A replacement key is written over the stored one in the selected configuration
file (file `0600`, parent `0700`) and into every deployment linked to that
provider, so the previous value is gone: one key per provider, with no second
slot, backup or history. The page only ever reads back whether a key is saved and
its last four characters, or the variable an environment reference names.

Omitting a patch field keeps it. Explicit `null` clears `api_key` or `api_base`;
an empty provider credential field is invalid. Model names cannot be null. Weight
is a nonnegative integer; zero disables a deployment. Unknown request fields and
masked values containing `…` are rejected.

A deployment linked to a saved provider normally inherits missing credentials.
When a deployment explicitly clears one field, native settings detach that
connection link and copy the other effective credential field if it was omitted.
This makes clear effective without altering sibling deployments or the saved
connection. Normal provider patching updates all linked or matching legacy
connection deployments together.

Deleting the last deployment keeps its connection available for later reuse.
Deleting a provider removes its linked/matching deployments as well. Batch
creation accepts `credential_provider_id` or `credential_deployment_id` to reuse
an existing connection without sending its secret through a list response.

Collection responses show only credential-presence flags and HTTP(S) origin; they
omit userinfo, URL paths, query and fragments. The explicit credentials endpoint
returns the raw stored fields, including an `env:NAME` reference rather than its
resolved value. `api_base_placeholder` is separate, so editing a key does not
accidentally persist a default endpoint as an override.

All settings responses, including errors, use `Cache-Control: no-store`. Every
client, including loopback and same-host reverse proxies, needs the server token.
With explicit `--no-auth`, only a direct loopback peer without forwarding headers
may access settings; remote settings access remains blocked. Host and Origin checks still apply. Config open affects the
server host, not a remote client's machine; a missing file returns 404.

## Edit config

The admin can edit the selected file as text (the workbench's Settings → Edit
config). `GET /api/settings/config` answers `{"path", "origin", "exists",
"revision", "text", "masked", "problem", "read_only"}`: `revision` is `sha256:`
and the hex SHA-256 of the file's bytes (`absent` without a file), and `text` is
the file's own text, layout and key order unchanged (`{}` without a file; null,
with `problem` such as `line 3, column 9: expected …`, when the file is not UTF-8
text or not a JSON object), with only each secret's string literal replaced by
its masked form: a string under a key `markitai config
show` redacts (API keys, tokens, secrets, passwords, credentials, cookies,
authorization, and every string beneath such a key) or a value of an
`extra_http_headers` object becomes `sk…7890`, unless it is an
`env:` reference, and an `api_base` loses its userinfo and secret query values.
`masked` counts them.

`PUT /api/settings/config` takes exactly `{"expected_revision", "text"}` (else
422 `invalid_settings_request`; `text` over 1 MiB or a body over 1.5 MiB 413
`config_too_large`). It is refused with 409 `settings_read_only` under session
model/provider overrides, and with 409 `detail.code = "stale_revision"` (and
`detail.current_revision`) when the file changed since it was read. `text` must
be a JSON object (422 `invalid_config`, a parse error as `line L, column C: …`).
A masked value sent back unchanged at the position GET sent it is put back in
place as the stored literal; a masked value anywhere else is 422
`invalid_config` naming its JSON pointer (type the full value or an `env:`
reference). The result must pass the same validation as at startup (422
`invalid_config` with its message), and is saved like the other settings
changes (below), exactly as written: the sent text, byte for byte, with only the
kept masked values put back. Line endings are part of the text: GET sends a
CRLF file with CRLF, and PUT writes whatever endings it is sent (the workbench
edits with LF, as a browser text box does, and sends each line back with the
file's own ending, so an unchanged line stays byte for byte). A file that
starts with a byte order mark is not readable configuration (null `text`, with
`problem`). The models and providers apply at once; the answer is the GET
shape for the new file plus `restart_required`, true when anything else changed,
which the service reads at its next start.

## Publication and limits

The service rereads the selected raw JSON inside its settings mutation lock,
checks the revision, validates the candidate, and writes an owner-only temporary
file before an atomic replacement. Unknown configuration/model fields survive.
The runtime snapshot changes only after publication. File/configuration errors
use fixed diagnostics without echoing credential values.

A failure before replacement preserves the old file and runtime snapshot. If the
replacement succeeds but syncing its parent directory fails, the new runtime is
activated to match the published file and the API returns an explicit 500 saying
that the write occurred but its durability could not be confirmed. Reload settings
before another change; this case is not reported as a successful durable save.

Configuration reads and writes are limited to 8 MiB; settings request bodies to
1 MiB. A symlinked configuration file, or one in a symlinked folder, is read and
saved through its link, which is kept; nonregular configuration files are rejected, and reads never block on a
FIFO. New configuration files are 0600
on Unix; new directories are private. Existing project directories are not
chmodded. External editors that do not cooperate with the service can race the
last read/rename interval; no cross-process transactional guarantee is claimed.

`--config-json` remains a session overlay. If it explicitly supplies a model or
provider list, the runtime uses that overlay and setting mutations return a
conflict instead of silently persisting session credentials or reporting a save
that an overlay would mask. Persistent views still describe the selected saved
file. Non-model overrides do not prevent settings writes.

Environment detection is session-only and does not inspect external OAuth or CLI
authentication stores. A configured pool takes precedence at startup. When models
were detected for an initially empty pool, saving one leaves other session
candidates available without persisting them. Explicit `MODEL` selects the
process's automatic model. Common provider cards do not perform network requests;
`refresh` recomputes these inexpensive cards immediately.

Subscription models (`copilot/`, `claude-agent/` and `chatgpt/gpt-5.5`) run
through their installed official runtimes as described in
[subscriptions](subscriptions.md); they take no API key or base URL, and their
provider cards stay offered after models are added, listing models through the
installed runtime when their page opens. Other
OAuth or local-CLI providers from the reference are not implemented. The
settings schema can retain such configuration for compatibility, but listing or
storing a model is not a claim that the converter can execute it.

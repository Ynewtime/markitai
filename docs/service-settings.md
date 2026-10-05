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
| POST | `/deployments/batch` | Atomically add 1–50 deployments |
| PATCH / DELETE | `/deployments/{id}` | Change a stable deployment identity |
| PATCH / DELETE | `/providers/{id}` | Change a connection or remove it and its deployments |
| POST | `/config/open` | Ask the server host's system opener to open the selected file |

Provider detection, live discovery and the transient connection test use
`/detected`, `/model-discovery` and `/test`; those operations are separate from
saving settings. They do not persist a connection draft.

The deployment view contains `deployment_id`, `routing_group`, `model`, `weight`,
credential-presence flags, a sanitized API origin and `persisted`. A routing group
may contain multiple deployments. Legacy name-based changes return 409 when the
name is ambiguous. New deployments receive UUID identities; the first v2 mutation
also assigns persistent identities to legacy entries.

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
1 MiB. Nonregular and symlink configuration leaves are rejected; Unix reads use
nonblocking/no-follow flags to avoid FIFO hangs. New configuration files are 0600
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
[subscriptions](subscriptions.md); they take no API key or base URL. Other
OAuth or local-CLI providers from the reference are not implemented. The
settings schema can retain such configuration for compatibility, but listing or
storing a model is not a claim that the converter can execute it.

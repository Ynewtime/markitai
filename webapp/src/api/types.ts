/** Mirror of the `markitai serve` API contract (scratchpad/API_CONTRACT.md). */

/** Server-side per-job item cap (folder drops truncate against it). */
export const MAX_JOB_ITEMS = 50;

export type ItemKind = "file" | "url";
export type ItemStatus = "queued" | "running" | "done" | "error";
export type JobStatus = "running" | "done";
export type Preset = "minimal" | "standard" | "rich";

export interface JobOptions {
  preset: Preset | null;
  llm: boolean | null;
  ocr: boolean | null;
}

export interface CreatedItem {
  item_id: string;
  name: string;
  kind: ItemKind;
}

export interface CreateJobResponse {
  job_id: string;
  items: CreatedItem[];
}

/** Payload of `event: item` (and of items inside the snapshot). */
export interface ItemPayload {
  item_id: string;
  name: string;
  kind: ItemKind;
  status: ItemStatus;
  error: string | null;
  output: string | null;
  duration_ms: number | null;
  finished_at: string | null;
  cost_usd: number | null;
  /** The selected output is an LLM-enhanced Markdown variant. */
  llm_enhanced: boolean;
  /** Most recent operation that produced this state. */
  operation: "convert" | "retry" | "enhance";
  /** Completed as a skip — status stays "done" but there is no new result. */
  skipped: boolean;
  /** Skip reason, e.g. "exists" | "image_only"; null unless skipped. */
  skip_reason: string | null;
}

/** Payload of `event: job`. */
export interface JobPayload {
  status: JobStatus;
  done: number;
  failed: number;
  total: number;
}

/** Payload of `event: snapshot` and `GET /api/jobs/{id}`. */
export interface JobSnapshot extends JobPayload {
  job_id: string;
  created_at: string;
  finished_at: string | null;
  options: JobOptions;
  items: ItemPayload[];
}

export interface ItemResult {
  name: string;
  variant: "llm" | "base";
  markdown: string;
  artifacts: { relpath: string; size: number }[];
}

export interface Capabilities {
  version: string;
  llm: {
    configured: boolean;
    routable: boolean;
    effective: boolean;
    models: string[];
  };
  presets: string[];
  extras: { browser: boolean; svg: boolean; kreuzberg: boolean };
}

/** `GET /api/settings/llm` — secret-free deployment and session detection view. */
export type LLMSource = "config" | "detected" | "none";

export interface LLMDeployment {
  deployment_id: string;
  routing_group: string;
  model: string;
  weight: number;
  api_key_configured: boolean;
  api_base_configured: boolean;
  /** Sanitized scheme + host + port only. */
  api_base: string | null;
  persisted: boolean;
}

export interface LLMSettingsPayload {
  configured: boolean;
  routable: boolean;
  source: LLMSource;
  config_path: string;
  config_origin: "explicit" | "environment" | "project" | "user" | "default";
  revision: string;
  deployments: LLMDeployment[];
  detected: LLMDeployment[];
}

export type LLMSettingsUpdate =
  | { model: string; api_key?: string; api_base?: string }
  | { deployment_id: string };

export interface LLMProviderCredentials {
  api_key: string | null;
  // the RAW saved base (null when the connection uses the provider default)
  api_base: string | null;
  // the provider default, offered for the editor's placeholder only
  api_base_placeholder?: string | null;
}

export interface ProviderConnection {
  id: string;
  provider_id?: string;
  deployment_id?: string;
  provider: string;
  label: string;
  kind: "local_cli" | "oauth" | "environment" | "configured" | "common";
  status: string;
  source: string;
  default_model?: string;
  credential?: string;
  api_key_configured?: boolean;
  api_base_configured?: boolean;
  api_base?: string | null;
  model_count?: number;
  supports_discovery: boolean;
}

export interface ModelCandidate {
  model: string;
  label: string;
  supports_vision: boolean;
}

export interface ModelDiscoveryResult {
  provider: string;
  status: "ok" | "partial" | "unavailable";
  source: string;
  authoritative: boolean;
  cached: boolean;
  stale: boolean;
  models: ModelCandidate[];
  detail?: string;
}

/** `POST /api/settings/llm/test` — always 200, `ok` flags the result. */
export interface LLMTestResult {
  ok: boolean;
  detail: string;
}

/** Body of `POST /api/settings/llm/models` (R3 per-entry CRUD). */
export interface LLMModelCreate {
  model_name: string;
  model: string;
  provider?: string;
  api_key?: string;
  api_base?: string;
  weight?: number;
  credential_provider_id?: string;
  credential_deployment_id?: string;
}

export interface LLMModelUpdate {
  model_name?: string;
  model?: string;
  api_key?: string | null;
  api_base?: string | null;
  weight?: number;
  expected_revision: string;
}

export interface LLMDeploymentBatch {
  expected_revision: string;
  deployments: LLMModelCreate[];
}

export interface LLMProviderUpdate {
  api_key?: string | null;
  api_base?: string | null;
  expected_revision: string;
}

/** One entry of `GET /api/history` (time-descending). */
export interface HistoryEntry {
  job_id: string;
  created_at: string;
  finished_at: string | null;
  status: JobStatus;
  total: number;
  done: number;
  failed: number;
  skipped: number;
  llm_enhanced: number;
  cost_usd: number | null;
  names_preview: string[];
  kinds_preview: ItemKind[];
  duration_ms: number | null;
  size_bytes: number;
}

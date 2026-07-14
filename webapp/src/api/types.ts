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
  cost_usd: number | null;
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

export interface ProviderConnection {
  id: string;
  deployment_id?: string;
  provider: string;
  label: string;
  kind: "local_cli" | "oauth" | "environment" | "configured" | "common";
  status: string;
  source: string;
  default_model?: string;
  credential?: string;
  api_base_configured?: boolean;
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
  api_key?: string;
  api_base?: string;
  weight?: number;
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
  names_preview: string[];
  size_bytes: number;
}

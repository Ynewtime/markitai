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
  llm: { configured: boolean; models: string[] };
  presets: string[];
  extras: { browser: boolean; svg: boolean; kreuzberg: boolean };
}

/** `GET /api/settings/llm` — masked view of the LLM configuration. */
export type LLMSource = "config" | "detected" | "none";

export interface LLMSettingsModel {
  model_name: string;
  model: string;
  /** null | verbatim "env:VAR" | masked plaintext like "sk…f3ab". */
  api_key_masked: string | null;
  api_base: string | null;
}

export interface LLMSettingsPayload {
  configured: boolean;
  source: LLMSource;
  config_path: string;
  models: LLMSettingsModel[];
}

/** Body of `POST /api/settings/llm/test` — either ad-hoc values (unsaved
 * form values) or a `model_name` reference to a stored entry, which the
 * backend probes with its stored credentials (masked keys are never sent). */
export type LLMSettingsUpdate =
  | { model: string; api_key?: string; api_base?: string }
  | { model_name: string };

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
}

/** Body of `PUT /api/settings/llm/models/{model_name}` — omitted field keeps
 * the stored value, explicit null clears it, a new string replaces it. */
export interface LLMModelUpdate {
  model?: string;
  api_key?: string | null;
  api_base?: string | null;
}

/** `GET /api/settings/llm/detected` — local CLI provider candidates. */
export interface DetectedProvider {
  provider: string;
  model: string;
  label: string;
  requires_api_key: boolean;
}

/** One entry of `GET /api/history` (time-descending). */
export interface HistoryEntry {
  job_id: string;
  created_at: string;
  finished_at: string;
  status: string;
  total: number;
  done: number;
  failed: number;
  skipped: number;
  names_preview: string[];
  size_bytes: number;
}

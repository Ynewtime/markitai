// Shapes of the `markitai serve` HTTP API as this workbench reads them (docs/serve.md).
// Field names are checked against the Rust serializers by tests/serve_web.rs.

export type ItemKind = "file" | "url";
export type ItemStatus = "queued" | "running" | "done" | "error";
/** `error` is the service's explicit persistence failure (`persistence_error`). */
export type JobStatus = "running" | "done" | "error";
export type Preset = "minimal" | "standard" | "rich";
export type OutputProfile = "rag" | "obsidian" | "okf";
export type FetchStrategy = "auto" | "static" | "playwright" | "defuddle" | "jina" | "cloudflare";
export type ConversionBackend = "native" | "cloudflare";

/** The request options; null leaves the server configuration in charge. */
export interface JobOptions {
  /** Authorization is for this request only; never restore it from history. */
  remote_processing?: "cloudflare" | null;
  preset: string | null;
  llm: boolean | null;
  ocr: boolean | null;
  profile: OutputProfile | null;
  alt: boolean | null;
  desc: boolean | null;
  screenshot: boolean | null;
  screenshot_only: boolean | null;
  pure: boolean | null;
  no_cache: boolean | null;
  no_compress: boolean | null;
  strategy: FetchStrategy | null;
  backend: ConversionBackend | null;
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

/** Request coverage of a model cost subtotal. */
export interface Pricing {
  priced_requests: number;
  unpriced_requests: number;
  cost_status: "complete" | "partial" | "unknown";
  incomplete_request_observations?: number;
  pricing_snapshots?: string[];
}

export interface ModelUsageRow {
  requests?: number;
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
  priced_requests?: number;
  unpriced_requests?: number;
  cost_status?: string;
  incomplete_request_observations?: number;
}

export interface AttemptUsage {
  cost_usd?: number;
  requests?: number;
  input_tokens?: number;
  output_tokens?: number;
  by_model?: Record<string, ModelUsageRow>;
}

export interface AttemptDiagnostics {
  last_attempt?: {
    operation: "convert" | "retry" | "enhance";
    status: "done" | "error";
    error: string | null;
    usage?: AttemptUsage;
  };
}

/** One item, as in `event: item` and inside snapshots. */
export interface RerunFailure {
  operation: "retry" | "enhance";
  error_code: string;
  error: string;
  failed_at: string;
}

export interface ItemPayload {
  /** This item's saved selections, without permission for another request. Older servers omit it. */
  options?: JobOptions;
  remote_processing?: RemoteProcessing | null;
  item_id: string;
  name: string;
  kind: ItemKind;
  status: ItemStatus;
  error: string | null;
  /** Stable cause of a failed item (core conversion code or service cause). */
  error_code?: string;
  output: string | null;
  output_name: string | null;
  duration_ms: number | null;
  finished_at: string | null;
  cost_usd: number | null;
  pricing?: Pricing;
  diagnostics?: AttemptDiagnostics;
  /** Latest failed rerun; the successful output represented by this row was kept. */
  rerun_failure?: RerunFailure;
  llm_enhanced: boolean;
  operation: "convert" | "retry" | "enhance";
  skipped: boolean;
  skip_reason: string | null;
  retryable: boolean;
  warnings: string[];
}

/** `event: job`. */
export interface JobProgress {
  status: JobStatus;
  done: number;
  failed: number;
  total: number;
}

/** `event: snapshot` and `GET /api/jobs/{id}`. */
export interface JobSnapshot extends JobProgress {
  job_id: string;
  created_at: string;
  finished_at: string | null;
  options: Record<string, unknown>;
  items: ItemPayload[];
  persistence_error?: string;
}

export interface Artifact {
  relpath: string;
  size: number;
}

export interface ItemResult {
  name: string;
  variant: "llm" | "base";
  markdown: string;
  artifacts: Artifact[];
}

export interface PresetFeatures {
  llm: boolean;
  ocr: boolean;
  alt: boolean;
  desc: boolean;
  screenshot: boolean;
}

export interface CloudflareCapability {
  configured: boolean;
  available: boolean;
  reason: null | "not_configured" | "invalid_configuration" | "disabled_by_policy" | "client_not_trusted";
  browser_rendering: boolean;
  file_conversion: boolean;
  file_extensions: string[];
}

/** Any accepted attempt requested Cloudflare; this does not prove execution. */
export interface RemoteProcessing {
  provider: "cloudflare";
  requested: true;
  execution: "unknown";
  external_charges: "not_included";
  notice?: string;
}

export interface Capabilities {
  remote_services?: { cloudflare: CloudflareCapability };
  version: string;
  llm: {
    configured?: boolean;
    routable: boolean;
    effective?: boolean;
    models?: string[];
  };
  presets: string[];
  preset_options: Record<string, PresetFeatures>;
  extras: { browser: boolean; svg: boolean };
  limits: { max_job_items: number };
}

/** One entry of `GET /api/history`, newest first. */
export interface HistoryEntry {
  remote_processing?: RemoteProcessing | null;
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
  pricing?: Pricing;
  names_preview: string[];
  kinds_preview: ItemKind[];
  duration_ms: number | null;
  size_bytes: number;
  origin: "web" | "cli";
  retryable: boolean;
}

export interface Deployment {
  deployment_id: string;
  routing_group: string;
  model: string;
  weight: number;
  api_key_configured: boolean;
  api_base_configured: boolean;
  api_base: string | null;
  persisted: boolean;
}

export interface SettingsView {
  configured: boolean;
  routable: boolean;
  source: "config" | "detected" | "none";
  config_path: string;
  config_origin: string;
  revision: string;
  deployments: Deployment[];
  detected: Deployment[];
}

export interface ProviderCard {
  id: string;
  provider_id?: string;
  deployment_id?: string;
  provider: string;
  label: string;
  kind: string;
  status: string;
  source: string;
  /** Display metadata only; never echo this server reference as an API key. */
  credential?: string;
  api_key_configured?: boolean;
  api_base_configured?: boolean;
  api_base?: string | null;
  model_count?: number;
  supports_discovery: boolean;
  /** The provider's documented endpoint; null when every server has its own. */
  default_base?: string | null;
  /** The environment variable read for the key. */
  key_variable?: string | null;
  /** A local server that works without a key. */
  key_optional?: boolean;
}

export interface ProviderCredentials {
  /** Server-held value/reference: keep out of editable inputs and outgoing writes. */
  api_key: string | null;
  /** "environment" when the key was found in the environment rather than the configuration. */
  api_key_source?: string | null;
  api_base: string | null;
  api_base_placeholder: string | null;
}

export interface ModelCandidate {
  model: string;
  label: string;
  supports_vision: boolean;
}

export interface DiscoveryResult {
  provider: string;
  status: "ok" | "partial" | "unavailable";
  source?: string;
  authoritative?: boolean;
  cached?: boolean;
  stale?: boolean;
  models: ModelCandidate[];
  detail?: string;
}

export interface ProbeResult {
  ok: boolean;
  detail: string;
}

export interface NewDeployment {
  model_name: string;
  model: string;
  weight: number;
  provider?: string;
  api_key?: string;
  api_base?: string;
  /** Saved connection ID, or an environment card reference such as env:openai. */
  credential_provider_id?: string;
  credential_deployment_id?: string;
}

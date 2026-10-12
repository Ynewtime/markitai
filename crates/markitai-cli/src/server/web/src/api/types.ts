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
  /** Model names that produced the retained result. */
  models?: string[];
  /** Bytes of the uploaded original; null for URLs or when unknown. */
  source_size?: number | null;
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
  /** What the submission is called in the list, if it was named. */
  label?: string | null;
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
  /** `markdown` was cut to the requested `max_bytes`. */
  truncated?: boolean;
  /** Size of the whole selected Markdown. */
  markdown_bytes?: number;
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
  limits: { max_job_items: number; history_retention_days?: number | null };
  implementation?: "rust" | "python";
  docs_url?: string;
  /** This backend's source repository, for the sidebar's link. */
  source_url?: string;
  features?: Features;
}

/** What a backend implements beyond the base job API; absent or false hides it. */
export interface Features {
  cancel: boolean;
  download_tickets: boolean;
  url_lists: boolean;
  history_items: boolean;
  original: boolean;
  result_excerpt: boolean;
  job_labels: boolean;
  /** v1.6: GET /api/jobs/{id}/items/{item}/assets/{relpath} serves the item's own images inline. */
  item_assets?: boolean;
  /** v1.11: GET /api/jobs/{id}/items/{item}/archive zips one item's files and its original. */
  item_archive?: boolean;
  /** v1.11: POST /api/access-token/regenerate replaces the server's access token. */
  token_regenerate?: boolean;
}

/** v1.11: `POST /api/access-token/regenerate`. */
export interface RegeneratedToken {
  token: string;
  /** The address other devices on the network open, with the new token; null when the server is not on the network. */
  network_url: string | null;
  /** The new token survives a restart of the server. */
  persistent: boolean;
  /** The variable that sets the token when the server starts (it comes back then), or null. */
  startup_variable: string | null;
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
  label?: string | null;
  /** With `?include=items`. */
  options?: JobOptions;
  /** With `?include=items`. */
  items?: ItemPayload[];
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
  /** No key needs to be entered: a local server, a subscription runtime, or a
   * cloud that reads its own credentials. */
  key_optional?: boolean;
  /** The picker group (`common`, `subscriptions`, `makers`, `clouds`,
   * `inference`, `gateways`, `local`); a server without one (Python 1.2.1)
   * leaves it out and the picker infers Common from the provider names. */
  group?: string;
  /** Key of a vendored brand mark; null for a monogram. */
  icon?: string | null;
  /** Further model prefixes routed the same way and other searched names. */
  aliases?: string[];
}

export interface ProviderCredentials {
  /** The saved key masked (`sk…7890`), or an `env:NAME` reference; never the key. Display only. */
  api_key_masked?: string | null;
  /** The same display value under its older name. */
  api_key?: string | null;
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
  /** v1.11: why an unavailable list failed, with the test's categories (v1.10). */
  reason?: string;
  key_variable?: string;
}

export interface ProbeResult {
  ok: boolean;
  detail: string;
  /** Why a test failed (v1.10): no_key, key_rejected, model_not_found, rate_limited, network, timeout, provider_error. */
  reason?: string;
  /** The key variable not set on the server (with `no_key`). */
  key_variable?: string;
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

/** `GET /api/jobs/{job_id}/items/{item_id}/original`. */
export interface OriginalDescriptor {
  /** What the UI can show:
   *  "pdf"         uploaded PDF; page images and a text layer
   *  "image"       uploaded raster image; one page image
   *  "html"        an HTML document (uploaded .html/.htm/.xhtml, or a URL item's fetched page snapshot)
   *  "text"        plain text (uploaded .md/.markdown/.txt/.csv/.tsv/.json/.xml/.yaml/.yml/.log)
   *  "screenshots" no renderable original, but the conversion saved page screenshots (Office files, web pages)
   *  "none"        nothing to show; see `reason` */
  kind: "pdf" | "image" | "html" | "text" | "screenshots" | "none";
  name: string;                 // upload name or URL
  media_type: string | null;    // of the stored original bytes
  size: number | null;          // bytes of the stored original
  /** API path for downloading the stored original bytes (also allowed as a ticket path); null when none is kept. */
  download: string | null;
  /** v1.11, html only: API path of the kept page for a sandboxed frame (opened through a ticket). */
  view?: string | null;
  /** Page boxes, in order. pdf: points (1/72 in) of the visible page box after rotation; image, screenshots:
   *  pixels. At most 2000 entries (the first pages); `page_count` is the total. Empty for text/none, and for
   *  html unless a URL item also saved page screenshots (v1.11: then those, served as pages). */
  pages: { width: number; height: number }[];
  page_count: number;           // pdf: total pages; image: 1; screenshots: number of images; else 0
  /** pdf only: GET .../pages/{n}/text answers. */
  text_layer: boolean;
  /** Set when kind is "none" (and MAY explain a fallback):
   *  "not_kept"          no original stored: a CLI-recorded job, a URL page that was not kept as HTML
   *                      (for example Jina/Cloudflare/Defuddle results or a page-cache hit), or an upload removed
   *  "unsupported_format" the format has no browser view (docx, pptx, xlsx, epub, eml, msg, ipynb, …) and no
   *                      screenshots were saved; the UI offers "Download original"
   *  "too_large"         the stored original exceeds the backend's preview limit (UI still offers the download)
   *  "render_unavailable" this backend build cannot render the format */
  reason: string | null;
}

/** `GET /api/jobs/{job_id}/items/{item_id}/original/pages/{n}/text`. */
export interface PageText {
  page: number;
  width: number;   // points, same box as the descriptor's pages[n-1]
  height: number;
  /** Reading-order lines; bbox = [x0, y0, x1, y1] in points from the page's TOP-left corner, y growing down. */
  lines: { text: string; bbox: [number, number, number, number] }[];
}

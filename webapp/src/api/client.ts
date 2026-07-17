import type {
  Capabilities,
  CreateJobResponse,
  HistoryEntry,
  LLMDeploymentBatch,
  ItemResult,
  JobOptions,
  JobSnapshot,
  LLMModelUpdate,
  LLMProviderCredentials,
  LLMProviderUpdate,
  LLMSettingsPayload,
  ModelDiscoveryResult,
  ProviderConnection,
  LLMSettingsUpdate,
  LLMTestResult,
} from "./types";

/** Structured API failure. `code` mirrors the server's machine-readable
 * `detail.code` (e.g. "stale_revision") so callers can branch without parsing
 * the message; `currentRevision` accompanies stale_revision 409s so callers
 * can rebase instead of blindly retrying. */
export class ApiError extends Error {
  status: number;
  code: string | null;
  currentRevision: string | null;

  constructor(
    message: string,
    status: number,
    code: string | null = null,
    currentRevision: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.currentRevision = currentRevision;
  }
}

/** FastAPI 422 bodies carry `detail` as an array of {msg, loc, …}. */
function flattenDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) =>
      typeof d === "object" && d !== null && "msg" in d
        ? String((d as { msg: unknown }).msg)
        : JSON.stringify(d),
    );
    return msgs.length > 0 ? msgs.join("; ") : null;
  }
  return null;
}

/** English fallback per code; the UI localizes by matching on `error.code`. */
function messageForCode(code: string): string {
  if (code === "stale_revision") return "Settings were changed in another window";
  return code.replace(/_/g, " ");
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  try {
    const body: unknown = await res.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      const msg = flattenDetail(detail);
      if (msg !== null) return new ApiError(msg, res.status);
      if (typeof detail === "object" && detail !== null && "code" in detail) {
        const code = String((detail as { code: unknown }).code);
        const revision = (detail as { current_revision?: unknown }).current_revision;
        return new ApiError(
          messageForCode(code),
          res.status,
          code,
          typeof revision === "string" ? revision : null,
        );
      }
    }
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(`${res.status} ${res.statusText}`, res.status);
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as T;
}

export function fetchCapabilities(): Promise<Capabilities> {
  return getJson<Capabilities>("/api/capabilities");
}

/** Job snapshot for session restore. Resolves null on 404 (server was
 * restarted and forgot the job) so callers can silently drop it; any other
 * failure throws. */
export async function fetchJobSnapshot(jobId: string): Promise<JobSnapshot | null> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as JobSnapshot;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as T;
}

export function fetchLLMSettings(): Promise<LLMSettingsPayload> {
  return getJson<LLMSettingsPayload>("/api/settings/llm");
}

export async function fetchProviderConnections(
  refresh = false,
): Promise<ProviderConnection[]> {
  const payload = await getJson<{ providers: ProviderConnection[] }>(
    `/api/settings/llm/providers?refresh=${refresh ? "true" : "false"}`,
  );
  return payload.providers;
}

export function discoverLLMModels(body: {
  provider: string;
  provider_id?: string;
  deployment_id?: string;
  api_key?: string;
  api_base?: string;
  refresh?: boolean;
}): Promise<ModelDiscoveryResult> {
  return postJson<ModelDiscoveryResult>("/api/settings/llm/model-discovery", body);
}

export function addLLMDeployments(
  body: LLMDeploymentBatch,
): Promise<LLMSettingsPayload> {
  return postJson<LLMSettingsPayload>("/api/settings/llm/deployments/batch", body);
}

export async function updateLLMDeployment(
  deploymentId: string,
  body: LLMModelUpdate,
): Promise<LLMSettingsPayload> {
  const res = await fetch(
    `/api/settings/llm/deployments/${encodeURIComponent(deploymentId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as LLMSettingsPayload;
}

export function fetchLLMProviderCredentials(
  providerId: string,
): Promise<LLMProviderCredentials> {
  return getJson<LLMProviderCredentials>(
    `/api/settings/llm/providers/${encodeURIComponent(providerId)}/credentials`,
  );
}

export async function updateLLMProvider(
  providerId: string,
  body: LLMProviderUpdate,
): Promise<LLMSettingsPayload> {
  const res = await fetch(
    `/api/settings/llm/providers/${encodeURIComponent(providerId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as LLMSettingsPayload;
}

export async function deleteLLMProvider(
  providerId: string,
  expectedRevision: string,
): Promise<LLMSettingsPayload> {
  const query = new URLSearchParams({ expected_revision: expectedRevision });
  const res = await fetch(
    `/api/settings/llm/providers/${encodeURIComponent(providerId)}?${query}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as LLMSettingsPayload;
}

export async function deleteLLMDeployment(
  deploymentId: string,
  expectedRevision: string,
): Promise<LLMSettingsPayload> {
  const query = new URLSearchParams({ expected_revision: expectedRevision });
  const res = await fetch(
    `/api/settings/llm/deployments/${encodeURIComponent(deploymentId)}?${query}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as LLMSettingsPayload;
}

/** Always resolves 200 {ok, detail}; may take up to ~15s. */
export function testLLMSettings(body: LLMSettingsUpdate): Promise<LLMTestResult> {
  return postJson<LLMTestResult>("/api/settings/llm/test", body);
}

export function fetchHistory(): Promise<HistoryEntry[]> {
  return getJson<HistoryEntry[]>("/api/history");
}

export async function deleteJobItem(jobId: string, itemId: string): Promise<void> {
  const res = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw await errorFromResponse(res);
}

export async function deleteHistoryJob(jobId: string): Promise<boolean> {
  const res = await fetch(`/api/history/${encodeURIComponent(jobId)}`, { method: "DELETE" });
  if (res.status === 404) return false;
  if (!res.ok) throw await errorFromResponse(res);
  return true;
}

export async function createJob(
  files: File[],
  urls: string[],
  options: JobOptions,
): Promise<CreateJobResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  form.append("urls", JSON.stringify(urls));
  form.append("options", JSON.stringify(options));
  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as CreateJobResponse;
}

export function fetchItemResult(
  jobId: string,
  itemId: string,
): Promise<ItemResult> {
  return getJson<ItemResult>(
    `/api/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}/result`,
  );
}

/** Queue a terminal item in place. The response keeps the original job and
 * item IDs; SSE resumes on that job and updates the existing ledger row. */
export async function retryJobItem(
  jobId: string,
  itemId: string,
  options?: JobOptions,
): Promise<CreateJobResponse> {
  const res = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}/retry`,
    {
      method: "POST",
      ...(options === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ options }),
          }),
    },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as CreateJobResponse;
}

/** Re-run one terminal item with LLM required, preserving its row identity. */
export async function enhanceJobItem(
  jobId: string,
  itemId: string,
  options: JobOptions,
): Promise<CreateJobResponse> {
  const res = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}/retry`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation: "enhance", options }),
    },
  );
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as CreateJobResponse;
}

/** Fetch one artifact as text (diff view pulls .md and .llm.md). */
export async function fetchJobFileText(
  jobId: string,
  relpath: string,
): Promise<string> {
  const res = await fetch(jobFileUrl(jobId, relpath));
  if (!res.ok) throw await errorFromResponse(res);
  return await res.text();
}

export function jobEventsUrl(jobId: string): string {
  return `/api/jobs/${encodeURIComponent(jobId)}/events`;
}

/** Encode each path segment (keeps "/" separators, escapes "#", "?", spaces). */
export function encodeRelPath(relpath: string): string {
  return relpath.split("/").map(encodeURIComponent).join("/");
}

export function jobFileUrl(jobId: string, relpath: string): string {
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${encodeRelPath(relpath)}`;
}

export function jobArchiveUrl(jobId: string): string {
  return `/api/jobs/${encodeURIComponent(jobId)}/archive`;
}

export function historyArchiveUrl(): string {
  return "/api/history/archive";
}

import type {
  Capabilities,
  CreateJobResponse,
  HistoryEntry,
  LLMDeploymentBatch,
  ItemResult,
  JobOptions,
  JobSnapshot,
  LLMModelUpdate,
  LLMSettingsPayload,
  ModelDiscoveryResult,
  ProviderConnection,
  LLMSettingsUpdate,
  LLMTestResult,
} from "./types";

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

async function errorDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      const msg = flattenDetail(detail);
      if (msg !== null) return msg;
      if (typeof detail === "object" && detail !== null && "code" in detail) {
        return String((detail as { code: unknown }).code);
      }
    }
  } catch {
    /* non-JSON error body */
  }
  return `${res.status} ${res.statusText}`;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errorDetail(res));
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
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as JobSnapshot;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
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
  if (!res.ok) throw new Error(await errorDetail(res));
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
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as LLMSettingsPayload;
}

/** Always resolves 200 {ok, detail}; may take up to ~15s. */
export function testLLMSettings(body: LLMSettingsUpdate): Promise<LLMTestResult> {
  return postJson<LLMTestResult>("/api/settings/llm/test", body);
}

export function fetchHistory(): Promise<HistoryEntry[]> {
  return getJson<HistoryEntry[]>("/api/history");
}

export async function deleteHistoryJob(jobId: string): Promise<boolean> {
  const res = await fetch(`/api/history/${encodeURIComponent(jobId)}`, { method: "DELETE" });
  if (res.status === 404) return false;
  if (!res.ok) throw new Error(await errorDetail(res));
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
  if (!res.ok) throw new Error(await errorDetail(res));
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

/** R4: retry a terminal item as a new single-item job. No body — the new
 * job inherits the original job's options server-side. 409 = item not
 * terminal; 404 = job/item gone or the upload was cleaned up. */
export async function retryJobItem(
  jobId: string,
  itemId: string,
): Promise<CreateJobResponse> {
  const res = await fetch(
    `/api/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}/retry`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as CreateJobResponse;
}

/** Fetch one artifact as text (diff view pulls .md and .llm.md). */
export async function fetchJobFileText(
  jobId: string,
  relpath: string,
): Promise<string> {
  const res = await fetch(jobFileUrl(jobId, relpath));
  if (!res.ok) throw new Error(await errorDetail(res));
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

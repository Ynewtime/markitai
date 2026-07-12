import type {
  Capabilities,
  CreateJobResponse,
  DetectedProvider,
  HistoryEntry,
  ItemResult,
  JobOptions,
  JobSnapshot,
  LLMModelCreate,
  LLMModelUpdate,
  LLMSettingsPayload,
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
      const msg = flattenDetail((body as { detail: unknown }).detail);
      if (msg !== null) return msg;
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

/** Write call where the response body is not part of the contract — callers
 * re-GET the affected resource after success. */
async function mutate(url: string, method: string, body?: unknown): Promise<void> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
}

export function fetchLLMSettings(): Promise<LLMSettingsPayload> {
  return getJson<LLMSettingsPayload>("/api/settings/llm");
}

export function fetchDetectedProviders(): Promise<DetectedProvider[]> {
  return getJson<DetectedProvider[]>("/api/settings/llm/detected");
}

export function addLLMModel(body: LLMModelCreate): Promise<void> {
  return mutate("/api/settings/llm/models", "POST", body);
}

export function updateLLMModel(modelName: string, body: LLMModelUpdate): Promise<void> {
  return mutate(`/api/settings/llm/models/${encodeURIComponent(modelName)}`, "PUT", body);
}

export function deleteLLMModel(modelName: string): Promise<void> {
  return mutate(`/api/settings/llm/models/${encodeURIComponent(modelName)}`, "DELETE");
}

/** Always resolves 200 {ok, detail}; may take up to ~15s. */
export function testLLMSettings(body: LLMSettingsUpdate): Promise<LLMTestResult> {
  return postJson<LLMTestResult>("/api/settings/llm/test", body);
}

export function fetchHistory(): Promise<HistoryEntry[]> {
  return getJson<HistoryEntry[]>("/api/history");
}

export function deleteHistoryJob(jobId: string): Promise<void> {
  return mutate(`/api/history/${encodeURIComponent(jobId)}`, "DELETE");
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

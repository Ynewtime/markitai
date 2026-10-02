// Every request to the service goes through here: same origin only, redirects
// refused, Bearer token when one is held, and connectivity reported as window
// events (`markitai:offline`, `markitai:online`, `markitai:unauthorized`).
import { bearer, serviceURL } from "./token.ts";
import type {
  Capabilities,
  CreateJobResponse,
  Deployment,
  DiscoveryResult,
  HistoryEntry,
  ItemResult,
  JobOptions,
  JobSnapshot,
  NewDeployment,
  ProbeResult,
  ProviderCard,
  ProviderCredentials,
  SettingsView,
} from "./types.ts";

/** A refused request. `reason` (or a settings conflict's `detail.code`) is the
 * stable cause the interface localizes; `detail` keeps the service's wording. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  readonly reason: string | null;
  readonly code: string | null;
  readonly detail: string;
  readonly currentRevision: string | null;
  constructor(status: number, body: unknown) {
    const value = (body ?? {}) as { detail?: unknown; reason?: unknown; code?: unknown };
    const structured =
      value.detail !== null && typeof value.detail === "object"
        ? (value.detail as { code?: unknown; current_revision?: unknown })
        : null;
    const detail =
      typeof value.detail === "string"
        ? value.detail
        : typeof structured?.code === "string"
          ? structured.code.replaceAll("_", " ")
          : "";
    super(detail || `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.reason =
      typeof value.reason === "string" ? value.reason : typeof structured?.code === "string" ? structured.code : null;
    this.code = typeof value.code === "string" ? value.code : null;
    this.detail = detail;
    this.currentRevision = typeof structured?.current_revision === "string" ? structured.current_revision : null;
  }
}

/** The service could not be reached (stopped server, dropped network, refused redirect). */
export class NetworkError extends Error {
  constructor() {
    super("network");
    this.name = "NetworkError";
  }
}

export const isAbort = (error: unknown): boolean =>
  error instanceof DOMException ? error.name === "AbortError" : (error as { name?: string })?.name === "AbortError";

function signal(name: string): void {
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent(name));
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
}

async function send(path: string, options: RequestOptions = {}): Promise<Response> {
  const url = serviceURL(path);
  const headers = new Headers(bearer());
  let body: BodyInit | undefined;
  if (options.body instanceof FormData) body = options.body;
  else if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }
  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      body,
      headers,
      signal: options.signal,
      credentials: "same-origin",
      redirect: "error",
      cache: "no-store",
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    signal("markitai:offline");
    throw new NetworkError();
  }
  signal("markitai:online");
  if (!response.ok) {
    let value: unknown = null;
    try {
      value = await response.json();
    } catch {
      value = null;
    }
    if (response.status === 401) signal("markitai:unauthorized");
    throw new ApiError(response.status, value);
  }
  return response;
}

async function json<T>(path: string, options?: RequestOptions): Promise<T> {
  const response = await send(path, options);
  if (response.status === 204) return null as T;
  return (await response.json()) as T;
}

/** A text body read up to `limit` bytes; larger files fail instead of filling memory. */
export async function fetchText(path: string, limit: number, abort?: AbortSignal): Promise<string> {
  const response = await send(path, { signal: abort });
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) {
    await response.body?.cancel();
    throw new RangeError("too-large");
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let total = 0;
  let text = "";
  try {
    for (;;) {
      const part = await reader.read();
      if (part.done) break;
      total += part.value.byteLength;
      if (total > limit) {
        await reader.cancel();
        throw new RangeError("too-large");
      }
      text += decoder.decode(part.value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

/** A download as a Blob, sent with the header token (no token in any URL). */
export async function fetchBlob(path: string): Promise<{ blob: Blob; name: string | null }> {
  const response = await send(path);
  const disposition = response.headers.get("content-disposition") ?? "";
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition)?.[1];
  let name = /filename="?([^";]+)"?/i.exec(disposition)?.[1] ?? null;
  if (encoded !== undefined) {
    try {
      name = decodeURIComponent(encoded);
    } catch {
      name = encoded;
    }
  }
  return { blob: await response.blob(), name };
}

export interface UploadHooks {
  signal?: AbortSignal;
  onProgress?: (loaded: number, total: number) => void;
  Request?: { new (): XMLHttpRequest };
}

/** The multipart job submission. fetch cannot report upload progress, so this
 * uses XMLHttpRequest with the same token, offline and error rules. XHR follows
 * redirects silently; an answer from another URL is rejected. */
export function upload<T>(path: string, body: FormData, hooks: UploadHooks = {}): Promise<T> {
  const url = serviceURL(path);
  const Request = hooks.Request ?? globalThis.XMLHttpRequest;
  return new Promise((resolve, reject) => {
    const aborted = () => new DOMException("The upload was aborted.", "AbortError");
    if (hooks.signal?.aborted) {
      reject(aborted());
      return;
    }
    const request = new Request();
    const cancel = () => request.abort();
    const settle = (finish: () => void) => {
      hooks.signal?.removeEventListener("abort", cancel);
      finish();
    };
    request.open("POST", url.href);
    for (const [name, value] of Object.entries(bearer())) request.setRequestHeader(name, value);
    request.upload.onprogress = (event) => hooks.onProgress?.(event.loaded, event.lengthComputable ? event.total : 0);
    request.upload.onload = (event) => {
      if (event?.lengthComputable) hooks.onProgress?.(event.total, event.total);
    };
    request.onabort = () => settle(() => reject(aborted()));
    request.onerror = () => {
      signal("markitai:offline");
      settle(() => reject(new NetworkError()));
    };
    request.onload = () => {
      if (request.responseURL && request.responseURL !== url.href) {
        settle(() => reject(new NetworkError()));
        return;
      }
      signal("markitai:online");
      let value: unknown = null;
      try {
        value = request.responseText ? JSON.parse(request.responseText) : null;
      } catch {
        value = null;
      }
      if (request.status >= 200 && request.status < 300) {
        settle(() => resolve(value as T));
        return;
      }
      if (request.status === 401) signal("markitai:unauthorized");
      settle(() => reject(new ApiError(request.status, value)));
    };
    hooks.signal?.addEventListener("abort", cancel, { once: true });
    request.send(body);
  });
}

/** Redraw at most once per interval, keep the latest values, never drop completion. */
export function throttle(
  update: (loaded: number, total: number) => void,
  interval = 100,
  now: () => number = () => Date.now(),
): ((loaded: number, total: number) => void) & { cancel(): void } {
  let last = -Infinity;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let latest: [number, number] = [0, 0];
  const flush = () => {
    timer = null;
    last = now();
    update(latest[0], latest[1]);
  };
  const report = (loaded: number, total: number) => {
    latest = [loaded, total];
    const finished = total > 0 && loaded >= total;
    if (finished || now() - last >= interval) {
      if (timer !== null) clearTimeout(timer);
      flush();
      return;
    }
    timer ??= setTimeout(flush, Math.max(0, interval - (now() - last)));
  };
  return Object.assign(report, {
    cancel() {
      if (timer !== null) clearTimeout(timer);
      timer = null;
    },
  });
}

const enc = encodeURIComponent;
export const encodePath = (relpath: string): string => relpath.split("/").map(enc).join("/");
export const filePath = (job: string, relpath: string): string => `/api/jobs/${enc(job)}/files/${encodePath(relpath)}`;
export const eventsPath = (job: string): string => `/api/jobs/${enc(job)}/events`;

/** A single-use URL, valid for one minute, for a download the browser opens
 * itself: the ticket stands in for the token, which never enters a URL. */
export async function downloadTicket(path: string): Promise<string> {
  const { url } = await json<{ url: string }>("/api/download-tickets", { method: "POST", body: { path } });
  const parsed = serviceURL(url);
  if (parsed.pathname !== path || [...parsed.searchParams.keys()].join() !== "ticket") throw new Error("Unexpected ticket URL");
  return parsed.pathname + parsed.search;
}

export const fetchCapabilities = () => json<Capabilities>("/api/capabilities");

/** null when the service no longer knows the job. */
export async function fetchSnapshot(job: string): Promise<JobSnapshot | null> {
  try {
    return await json<JobSnapshot>(`/api/jobs/${enc(job)}`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export function createJob(files: File[], urls: string[], options: JobOptions, hooks: UploadHooks = {}) {
  const body = new FormData();
  for (const file of files) body.append("files", file, file.name);
  body.append("urls", JSON.stringify(urls));
  body.append("options", JSON.stringify(options));
  return upload<CreateJobResponse>("/api/jobs", body, hooks);
}

export const stopJob = (job: string) => json<{ job_id: string; stopping: number }>(`/api/jobs/${enc(job)}/cancel`, { method: "POST" });

export function retryItem(job: string, item: string, options?: JobOptions, operation: "retry" | "enhance" = "retry") {
  const body = options === undefined ? { operation } : { operation, options };
  return json<CreateJobResponse>(`/api/jobs/${enc(job)}/items/${enc(item)}/retry`, { method: "POST", body });
}

export const deleteItem = (job: string, item: string) =>
  json<null>(`/api/jobs/${enc(job)}/items/${enc(item)}`, { method: "DELETE" });

export const fetchResult = (job: string, item: string) =>
  json<ItemResult>(`/api/jobs/${enc(job)}/items/${enc(item)}/result`);

export const fetchHistory = () => json<HistoryEntry[]>("/api/history");

/** false when it was already gone. */
export async function deleteHistoryJob(job: string): Promise<boolean> {
  try {
    await json<null>(`/api/history/${enc(job)}`, { method: "DELETE" });
    return true;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return false;
    throw error;
  }
}

const settings = "/api/settings/llm";
export const fetchSettings = () => json<SettingsView>(settings);
export const fetchProviders = (refresh = false) =>
  json<{ providers: ProviderCard[] }>(`${settings}/providers?refresh=${refresh}`).then((value) => value.providers ?? []);
export const fetchCredentials = (id: string) => json<ProviderCredentials>(`${settings}/providers/${enc(id)}/credentials`);
export const discoverModels = (body: Record<string, unknown>) =>
  json<DiscoveryResult>(`${settings}/model-discovery`, { method: "POST", body });
export const probeModel = (body: Record<string, unknown>) => json<ProbeResult>(`${settings}/test`, { method: "POST", body });
export const addDeployments = (revision: string, deployments: NewDeployment[]) =>
  json<SettingsView>(`${settings}/deployments/batch`, { method: "POST", body: { expected_revision: revision, deployments } });
export const updateDeployment = (id: string, body: Partial<Deployment> & Record<string, unknown>) =>
  json<SettingsView>(`${settings}/deployments/${enc(id)}`, { method: "PATCH", body });
export const deleteDeployment = (id: string, revision: string) =>
  json<SettingsView>(`${settings}/deployments/${enc(id)}?expected_revision=${enc(revision)}`, { method: "DELETE" });
export const updateProvider = (id: string, body: Record<string, unknown>) =>
  json<SettingsView>(`${settings}/providers/${enc(id)}`, { method: "PATCH", body });
export const deleteProvider = (id: string, revision: string) =>
  json<SettingsView>(`${settings}/providers/${enc(id)}?expected_revision=${enc(revision)}`, { method: "DELETE" });
export const openConfig = () => json<null>(`${settings}/config/open`, { method: "POST" });

/** Only a revision conflict keeps a settings draft for review; other 409s are plain errors. */
export const isRevisionConflict = (error: unknown): boolean =>
  error instanceof ApiError &&
  error.status === 409 &&
  (error.reason === null || error.reason === "stale_revision" || error.reason === "config_changed");

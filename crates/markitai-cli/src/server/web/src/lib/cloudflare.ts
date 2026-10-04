import type { CloudflareCapability, ItemKind, JobOptions, RemoteProcessing } from "../api/types.ts";
import { publicOptions } from "./options.ts";

export interface RemoteSource { name: string; kind: ItemKind }
export interface CloudflareScope { selected: boolean; urls: number; files: number; candidates: number; native: number }
export type CloudflareReason = NonNullable<CloudflareCapability["reason"]> | "unavailable" | "incompatible_strategy";

/** A disclosure estimate only. The server verifies the input format and policy. */
export function cloudflareScope(options: JobOptions, sources: readonly RemoteSource[], capability?: CloudflareCapability): CloudflareScope {
  const extensions = new Set((capability?.file_extensions ?? []).map((ext) => ext.toLowerCase().replace(/^\./, "")));
  const fileRoute = options.backend === "cloudflare" && options.ocr !== true && options.screenshot !== true && options.screenshot_only !== true;
  const files = sources.filter((source) => source.kind === "file");
  const eligible = files.filter((source) => fileRoute && extensions.has(source.name.split(".").at(-1)?.toLowerCase() ?? "")).length;
  return {
    selected: options.strategy === "cloudflare" || options.backend === "cloudflare",
    urls: options.strategy === "cloudflare" ? sources.filter((source) => source.kind === "url").length : 0,
    files: eligible,
    // Content sniffing may recognize a supported format despite its filename.
    candidates: fileRoute ? files.length : 0,
    native: fileRoute ? 0 : files.length,
  };
}

export function cloudflareReason(capability?: CloudflareCapability, options?: JobOptions): CloudflareReason | null {
  // Consent for one service cannot authorize a second remote URL service.
  // Keep the user's choices intact and ask them to resolve the conflict.
  if (options?.backend === "cloudflare" && (options.strategy === "defuddle" || options.strategy === "jina")) return "incompatible_strategy";
  if (!capability) return "unavailable";
  if (!capability.available) return capability.reason ?? "unavailable";
  if (options?.strategy === "cloudflare" && !capability.browser_rendering) return "unavailable";
  if (options?.backend === "cloudflare" && !capability.file_conversion) return "unavailable";
  return null;
}

/** Saved options may describe a route; they can never convey fresh permission. */
export function requestOptions(options: JobOptions, confirmed: boolean, capability?: CloudflareCapability): JobOptions {
  const clean = publicOptions(options);
  if ((clean.strategy === "cloudflare" || clean.backend === "cloudflare") && confirmed && cloudflareReason(capability, clean) === null) {
    return { ...clean, remote_processing: "cloudflare" };
  }
  return clean;
}

export function hasCloudflareRequest(value?: RemoteProcessing | null): boolean {
  return value?.provider === "cloudflare" && value.requested === true && value.external_charges === "not_included";
}

export interface CloudflareRequest {
  options: JobOptions;
  sources: readonly RemoteSource[];
}

/** Capture this click's scope. Later list/option mutations cannot join its consent. */
export function freezeCloudflareRequests(requests: readonly CloudflareRequest[]): CloudflareRequest[] {
  return requests.map((request) => ({
    options: requestOptions(request.options, false),
    sources: request.sources.map((source) => ({ name: source.name, kind: source.kind })),
  }));
}

export function cloudflareBatchScope(requests: readonly CloudflareRequest[], capability?: CloudflareCapability): CloudflareScope {
  return requests.reduce<CloudflareScope>((total, request) => {
    const scope = cloudflareScope(request.options, request.sources, capability);
    return { selected: total.selected || scope.selected, urls: total.urls + scope.urls,
      files: total.files + scope.files, candidates: total.candidates + scope.candidates, native: total.native + scope.native };
  }, { selected: false, urls: 0, files: 0, candidates: 0, native: 0 });
}

export function cloudflareBatchReason(requests: readonly CloudflareRequest[], capability?: CloudflareCapability): CloudflareReason | null {
  for (const request of requests) {
    if (!cloudflareScope(request.options, request.sources, capability).selected) continue;
    const reason = cloudflareReason(capability, request.options);
    if (reason) return reason;
  }
  return null;
}

/** A batch approval authorizes only the captured requests, each with its own explicit marker. */
export function confirmedCloudflareRequests(requests: readonly CloudflareRequest[], confirmed: boolean, capability?: CloudflareCapability): JobOptions[] | null {
  if (cloudflareBatchScope(requests, capability).selected && (!confirmed || cloudflareBatchReason(requests, capability) !== null)) return null;
  return requests.map((request) => requestOptions(request.options, confirmed, capability));
}

import type { Deployment, NewDeployment, ProviderCard, ProviderCredentials } from "../api/types.ts";

export type CredentialInputReason = "environment_reference" | "endpoint_key_required" | "key_required";

export class CredentialInputError extends Error {
  readonly reason: CredentialInputReason;
  constructor(reason: CredentialInputReason) {
    super(reason);
    this.reason = reason;
  }
}

function literal(value: string): string {
  const result = value.trim();
  if (result.startsWith("env:")) throw new CredentialInputError("environment_reference");
  return result;
}

/** A server-selected connection travels by identifier, never by echoed credentials. */
export function providerReference(provider: ProviderCard): { provider_id?: string; deployment_id?: string } {
  if (provider.kind === "environment") return { provider_id: `env:${provider.provider}` };
  if (provider.provider_id && !provider.provider_id.startsWith("legacy:")) return { provider_id: provider.provider_id };
  if (provider.deployment_id) return { deployment_id: provider.deployment_id };
  return {};
}

export function discoveryRequest(provider: ProviderCard, key: string, base: string, refresh: boolean): Record<string, unknown> {
  const reference = providerReference(provider);
  if (reference.provider_id || reference.deployment_id) return { provider: provider.provider, ...reference, refresh };
  const apiKey = literal(key);
  const apiBase = literal(base);
  return { provider: provider.provider, ...(apiKey ? { api_key: apiKey } : {}), ...(apiBase ? { api_base: apiBase } : {}), refresh };
}

export function deploymentCredentials(provider: ProviderCard, key: string, base: string): Partial<NewDeployment> {
  const reference = providerReference(provider);
  if (reference.provider_id) return { provider: provider.provider, credential_provider_id: reference.provider_id };
  if (reference.deployment_id) return { provider: provider.provider, credential_deployment_id: reference.deployment_id };
  const apiKey = literal(key);
  const apiBase = literal(base);
  return { provider: provider.provider, ...(apiKey ? { api_key: apiKey } : {}), ...(apiBase ? { api_base: apiBase } : {}) };
}

/** What the page may say about a saved key: never the key itself. */
export type SavedKey =
  | { state: "none" }
  | { state: "saved"; ending: string | null }
  /** `stored`: the configuration names the variable; otherwise the server found it in the environment. */
  | { state: "environment"; variable: string; stored: boolean };

export interface ProviderDraft {
  configured: boolean;
  key: SavedKey;
  /** The saved literal address; empty when it is the default or an environment reference. */
  base: string;
  baseRetained: boolean;
  /** The environment variable a retained address is read from. */
  baseVariable: string | null;
  placeholder: string;
}

/** Long literal keys show their last four characters; short ones show nothing. */
const KEY_ENDING_MIN = 12;
const variableOf = (value: string) => value.trim().slice("env:".length).trim();

function savedKey(value: string | null, detected: boolean): SavedKey {
  if (!value) return { state: "none" };
  if (value.trim().startsWith("env:")) return { state: "environment", variable: variableOf(value), stored: !detected };
  const key = value.trim();
  return { state: "saved", ending: key.length >= KEY_ENDING_MIN ? key.slice(-4) : null };
}

/** Neither saved literal keys nor environment references become editable input. */
export function providerDraft(credentials: ProviderCredentials): ProviderDraft {
  const baseRetained = credentials.api_base?.trim().startsWith("env:") === true;
  return {
    configured: Boolean(credentials.api_key),
    key: savedKey(credentials.api_key, credentials.api_key_source === "environment"),
    base: baseRetained ? "" : credentials.api_base ?? "",
    baseRetained,
    baseVariable: baseRetained ? variableOf(credentials.api_base ?? "") : null,
    placeholder: credentials.api_base_placeholder ?? "",
  };
}

function sameEndpoint(left: string, right: string): boolean {
  if (left === right) return true;
  try {
    const normalize = (value: string) => {
      const url = new URL(value);
      url.pathname = url.pathname.replace(/\/+$/, "");
      return url.href;
    };
    return normalize(left) === normalize(right);
  } catch { return false; }
}

/** One independent change to a saved provider: its key, or its address. */
export type ProviderChange =
  | { kind: "key"; key: string }
  | { kind: "removeKey" }
  /** A new address (blank: the provider default) and the key to use there. */
  | { kind: "base"; base: string; key: string };

/**
 * The PATCH body for one change, or null when nothing changes. A saved key is
 * never carried to a new address: moving the address needs a new key, and
 * without a saved key the move explicitly clears it, so no environment key
 * follows either.
 */
export function providerUpdate(loaded: ProviderDraft, change: ProviderChange, revision: string): Record<string, unknown> | null {
  if (change.kind === "removeKey") return { expected_revision: revision, api_key: null };
  const apiKey = literal(change.key);
  if (change.kind === "key") {
    if (!apiKey) throw new CredentialInputError("key_required");
    return { expected_revision: revision, api_key: apiKey };
  }
  const apiBase = literal(change.base);
  const moved = loaded.baseRetained || !sameEndpoint(apiBase, loaded.base);
  if (!moved) return apiKey ? { expected_revision: revision, api_key: apiKey } : null;
  if (loaded.configured && !apiKey) throw new CredentialInputError("endpoint_key_required");
  return { expected_revision: revision, api_key: apiKey || null, api_base: apiBase || null };
}

/** Persist a detected model without converting its server credential into a literal.
 * The server validates env provider IDs against its provider catalog. Runtime
 * subscriptions authenticate through their official CLI and need no env key. */
export function detectedDeployment(deployment: Deployment): NewDeployment {
  const provider = deployment.model.includes("/") ? deployment.model.split("/", 1)[0]!.toLowerCase() : "openai";
  const runtime = ["claude-agent", "copilot", "chatgpt"].includes(provider);
  return {
    model_name: deployment.routing_group,
    model: deployment.model,
    weight: deployment.weight,
    ...(runtime ? {} : { credential_provider_id: `env:${provider}` }),
  };
}

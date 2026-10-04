import type { Deployment, NewDeployment, ProviderCard, ProviderCredentials } from "../api/types.ts";

export class CredentialInputError extends Error {
  readonly reason: "environment_reference" | "endpoint_key_required";
  constructor(reason: "environment_reference" | "endpoint_key_required") {
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

export interface ProviderDraft {
  configured: boolean;
  base: string;
  baseRetained: boolean;
  placeholder: string;
}

/** Neither saved literal keys nor environment references become editable input. */
export function providerDraft(credentials: ProviderCredentials): ProviderDraft {
  const baseRetained = credentials.api_base?.startsWith("env:") === true;
  return {
    configured: Boolean(credentials.api_key),
    base: baseRetained ? "" : credentials.api_base ?? "",
    baseRetained,
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

export function providerUpdate(loaded: ProviderDraft, key: string, base: string, clearKey: boolean, revision: string): Record<string, unknown> {
  const apiKey = literal(key);
  const apiBase = literal(base);
  const changedBase = !sameEndpoint(apiBase, loaded.base);
  if (changedBase && !apiKey && !clearKey) throw new CredentialInputError("endpoint_key_required");
  return {
    expected_revision: revision,
    ...(clearKey ? { api_key: null } : apiKey ? { api_key: apiKey } : {}),
    ...(changedBase ? { api_base: apiBase || null } : {}),
  };
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

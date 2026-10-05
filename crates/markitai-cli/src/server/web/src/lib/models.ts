// Model identifiers entered by hand on a provider's page, and the endpoint a
// catalogue page compares its deployments at.

/** The routed model ID for `input` typed on `provider`'s page: the provider's
 * prefix is added unless the input already carries it, so IDs that contain
 * slashes of their own (`meta-llama/…`, `google/…`) still route to the page's
 * provider. An OpenAI-compatible endpoint (`custom`) routes as `openai/`. */
export function manualModelId(provider: string, input: string): string {
  const model = input.trim();
  if (!model) return "";
  const prefix = provider === "custom" ? "openai" : provider;
  return model.startsWith(`${prefix}/`) ? model : `${prefix}/${model}`;
}

/** The address a provider page works with: what the reader typed, or the saved
 * connection's own address. */
export const endpointFor = (draft: string, providerBase: string | null | undefined): string => draft.trim() || providerBase || "";

const normalizeEndpoint = (value: string | null | undefined) => (value ?? "").trim().replace(/\/+$/, "").toLowerCase();

/** Whether a deployment is routed at this endpoint. A deployment the provider
 * reaches at its own default address stores none, so an absent stored address
 * matches; the model id already names the provider. */
export function sameEndpoint(stored: string | null | undefined, effective: string): boolean {
  const value = normalizeEndpoint(stored);
  return value === "" || value === normalizeEndpoint(effective);
}

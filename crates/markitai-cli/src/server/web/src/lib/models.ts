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

/** A model ID without its provider prefix (`openai/gpt-5-mini` → `gpt-5-mini`). */
export function modelName(model: string): string {
  const slash = model.indexOf("/");
  return slash < 0 ? model : model.slice(slash + 1);
}

/** v1.16: the weight a field's text means, or null when it is not a whole number from 0 to 999 999. */
export function parseWeight(text: string): number | null {
  const value = text.trim();
  if (!/^\d{1,6}$/.test(value)) return null;
  return Number(value);
}

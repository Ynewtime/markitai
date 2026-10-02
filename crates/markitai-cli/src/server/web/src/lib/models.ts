// Model identifiers entered by hand on a provider's page.

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

// The addresses the API address field offers: a provider card's documented endpoints (contract
// v1.27 `endpoints`, labelled by area and plan), else its regions (v1.25), else its default
// address. The Kimi card offers both services' endpoints, the open platform's first.
import type { ProviderCard, ProviderEndpoint } from "../api/types.ts";

/** One address the field's list offers. `region` is an area code or a site shown as written; null
 * names the provider's default (a single address, or an older server's first region). */
export interface EndpointChoice {
  base: string;
  region: string | null;
  plan: string | null;
  /** The Kimi service the address belongs to, on the Kimi card. */
  service?: "moonshot" | "kimi_code";
}

type Card = Pick<ProviderCard, "default_base" | "regions" | "endpoints">;

/** A card's own addresses, in the server's order. */
export function cardEndpoints(card: Card | null | undefined): EndpointChoice[] {
  if (!card) return [];
  const listed = (card.endpoints ?? []).filter((endpoint: ProviderEndpoint) => typeof endpoint?.base === "string" && endpoint.base !== "");
  if (listed.length > 0) return listed.map((endpoint) => ({ base: endpoint.base, region: endpoint.region || null, plan: endpoint.plan || null }));
  const regions = (card.regions ?? []).filter((base) => typeof base === "string" && base !== "");
  // An older server names regions without areas: the default first, the others by their host.
  if (regions.length > 0) return regions.map((base, index) => ({ base, region: index === 0 ? null : hostOf(base), plan: null }));
  return card.default_base ? [{ base: card.default_base, region: null, plan: null }] : [];
}

/** The Kimi card's addresses: the open platform's, then Kimi Code's. */
export function kimiEndpoints(moonshot: Card, code: Card): EndpointChoice[] {
  return [
    ...cardEndpoints(moonshot).map((choice) => ({ ...choice, service: "moonshot" as const })),
    ...cardEndpoints(code).map((choice) => ({ ...choice, service: "kimi_code" as const })),
  ];
}

/** Whether a typed address is one of the choices (trailing slashes aside). */
export function matchingChoice(choices: readonly EndpointChoice[], typed: string): EndpointChoice | undefined {
  const value = typed.trim().replace(/\/+$/, "");
  return value ? choices.find((choice) => choice.base.replace(/\/+$/, "") === value) : undefined;
}

function hostOf(base: string): string | null {
  try {
    return new URL(base).host;
  } catch {
    return null;
  }
}

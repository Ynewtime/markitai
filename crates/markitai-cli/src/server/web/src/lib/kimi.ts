// Kimi's two services as one provider of Add models. Moonshot's open platform
// (`moonshot/`, keys from platform.kimi.com or platform.kimi.ai) and the Kimi
// Code membership (`kimi_code/`) issue keys that look alike, so the page asks
// for a key once and finds which service takes it: Load models tries the
// services in turn, each at its own documented regions only (the server's
// region retry), and the models are saved under the one that listed them. An
// address typed by hand is the one place a key goes. The configuration and
// the command line keep the two provider ids.
import type { ProviderCard } from "../api/types.ts";

export type KimiService = "moonshot" | "kimi_code";

/** The id of the one Kimi card the picker shows instead of the two services' cards. */
export const KIMI_CARD_ID = "workbench:kimi";

/** Whether a card is the merged Kimi card. */
export const isKimiCard = (card: { id?: string } | null | undefined): boolean => card?.id === KIMI_CARD_ID;

const builtIn = (card: ProviderCard): boolean => card.kind !== "configured" && card.kind !== "environment";

/** A service's card as a provider to connect with a pasted key. The server lists a provider
 * that has a saved connection by that connection only, so its built-in card is rebuilt from it:
 * the same provider, address default and variable, without the connection's key or address. */
function offered(card: ProviderCard): ProviderCard {
  if (builtIn(card)) return card;
  const { provider_id: _id, deployment_id: _deployment, api_base: _base, api_base_configured: _baseSet, api_key_configured: _keySet, credential: _credential, model_count: _count, ...rest } = card;
  return { ...rest, id: `common:${card.provider}`, kind: "common", status: "needs_credentials", source: "built_in" };
}

/** Each service as a provider to connect, when the server knows both (an older backend has no Kimi Code):
 * its built-in card, else one rebuilt from its saved or environment connection. */
export function kimiServices(cards: readonly ProviderCard[]): Record<KimiService, ProviderCard> | null {
  const find = (provider: KimiService) => cards.find((card) => card.provider === provider && builtIn(card)) ?? cards.find((card) => card.provider === provider);
  const moonshot = find("moonshot");
  const code = find("kimi_code");
  return moonshot && code ? { moonshot: offered(moonshot), kimi_code: offered(code) } : null;
}

/** The server's cards with the two services' built-in cards as one Kimi card, where the first of them was
 * (else last). Saved and environment cards stay as they are. */
export function mergeKimi(cards: readonly ProviderCard[]): ProviderCard[] {
  const services = kimiServices(cards);
  if (!services) return [...cards];
  const { moonshot, kimi_code: code } = services;
  const kimi: ProviderCard = {
    ...moonshot,
    id: KIMI_CARD_ID,
    label: "Kimi",
    icon: code.icon ?? "kimi",
    aliases: [...new Set(["moonshot ai", "kimi code", ...(moonshot.aliases ?? []), ...(code.aliases ?? [])])],
  };
  const service = (card: ProviderCard) => builtIn(card) && (card.provider === "moonshot" || card.provider === "kimi_code");
  const at = cards.findIndex(service);
  const rest = cards.filter((card) => !service(card));
  return at < 0 ? [...rest, kimi] : [...rest.slice(0, at), kimi, ...rest.slice(at)];
}

/** Kimi Code's endpoint hosts; any other address is the open platform's (or a gateway in front of it). */
const KIMI_CODE_HOSTS = new Set(["api.kimi.com", "api.kimi.ai"]);

/** The service an address typed by hand belongs to. */
export function kimiServiceAt(base: string): KimiService {
  try {
    const url = new URL(base.trim());
    return KIMI_CODE_HOSTS.has(url.hostname.toLowerCase()) && url.pathname.startsWith("/coding") ? "kimi_code" : "moonshot";
  } catch {
    return "moonshot";
  }
}

/** The order Load models tries a pasted key in: the open platform first, but a
 * key in the form Kimi Code's docs show (`sk-kimi-…`) at Kimi Code first, so it
 * usually reaches only the service that issued it. Either way the other one is
 * tried when the first refuses the key. */
export function kimiOrder(key: string): KimiService[] {
  return key.trim().startsWith("sk-kimi-") ? ["kimi_code", "moonshot"] : ["moonshot", "kimi_code"];
}

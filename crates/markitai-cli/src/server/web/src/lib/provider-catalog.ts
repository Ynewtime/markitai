// How the provider picker arranges the server's provider cards: saved
// connections, the common eight, then every other provider by name under a
// group filter. A server that sends no `group` (Python 1.2.1) still works:
// Common is inferred from the eight names and everything else is listed
// under All providers without a filter.
import type { ProviderCard } from "../api/types.ts";

/** The common providers, in the order the picker shows them. */
export const COMMON_PROVIDERS = ["openai", "anthropic", "gemini", "deepseek", "openrouter", "azure", "ollama", "mistral"];

/** The filter over All providers, in its order; `all` shows every group. */
export const PROVIDER_FILTERS = ["all", "clouds", "inference", "gateways", "local", "subscriptions", "makers"] as const;
export type ProviderFilter = (typeof PROVIDER_FILTERS)[number];

const SUBSCRIPTIONS = new Set(["claude-agent", "copilot", "chatgpt"]);
/** Clouds that sign requests with the server's own cloud credentials. */
const OWN_CREDENTIALS = new Set(["bedrock", "vertex_ai"]);

/** A subscription runtime signs in by itself: no key, no address. */
export const isSubscription = (card: Pick<ProviderCard, "provider" | "group">): boolean =>
  card.group === "subscriptions" || SUBSCRIPTIONS.has(card.provider);

/** Bedrock and Vertex AI read the server's cloud credentials and need no address. */
export const usesCloudCredentials = (card: Pick<ProviderCard, "provider">): boolean => OWN_CREDENTIALS.has(card.provider);

/** Providers that sign in with the server's own login whatever key an entry
 * carries, so the settings API refuses them an address (contract v1.9): they
 * take none here, and the page names the login instead. `google` is the
 * Google login; the others are the token variable. Bedrock is not here: with an
 * address it sends no server credential, so its optional address stays. */
const OWN_ENDPOINT_LOGIN: Record<string, string> = {
  vertex_ai: "google",
  vertex_ai_beta: "google",
  watsonx: "WATSONX_TOKEN",
  watsonx_text: "WATSONX_TOKEN",
  gigachat: "GIGACHAT_ACCESS_TOKEN",
};

/** The server login a provider always uses at its own endpoint, or null when it may take an address. */
export const ownEndpointLogin = (card: Pick<ProviderCard, "provider">): string | null => OWN_ENDPOINT_LOGIN[card.provider] ?? null;

/** The brand mark a card names; a server that names none (Python 1.2.1) gets
 * the mark whose key is the provider's prefix, when there is one. */
export const iconKey = (card: Pick<ProviderCard, "provider" | "icon">): string | null =>
  card.icon === undefined ? card.provider : card.icon;

/** A card that offers a provider to connect, rather than a connection. */
const builtIn = (card: ProviderCard): boolean => card.kind !== "configured" && card.kind !== "environment";

/** One provider in the Common grid or the All list: the card selecting it opens. */
export interface ProviderChoice {
  provider: string;
  label: string;
  group: string | null;
  icon: string | null;
  card: ProviderCard;
}

export interface ProviderArrangement {
  /** Saved connections and server environment credentials, as the server lists them. */
  saved: ProviderCard[];
  common: ProviderChoice[];
  /** Every other provider, by label, within the filter. */
  all: ProviderChoice[];
  /** The filters with at least one provider; only `all` when the server sends no groups. */
  filters: ProviderFilter[];
  /** Providers offered in Common and All before searching or filtering. */
  total: number;
}

const fold = (text: string) => text.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();

/** Whether a card matches a search: by name, model prefix or alias. */
export function matches(card: Pick<ProviderCard, "provider" | "label" | "aliases">, label: string, query: string): boolean {
  const words = fold(query).split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const haystack = [label, card.provider, ...(card.aliases ?? [])].map(fold).join(" \u0000 ");
  // Prefixes are written with underscores; a search may use spaces or hyphens.
  const loose = haystack.replace(/[_-]/g, " ");
  return words.every((word) => haystack.includes(word) || loose.includes(word.replace(/[_-]/g, " ")));
}

export function arrangeProviders(
  cards: ProviderCard[],
  options: { query: string; filter: ProviderFilter; label: (card: ProviderCard) => string },
): ProviderArrangement {
  const saved = cards.filter((card) => !builtIn(card));
  const grouped = cards.some((card) => typeof card.group === "string");
  // One choice per provider: its built-in card, else the first saved or
  // environment connection (whose provider the server no longer offers).
  const choices = new Map<string, ProviderChoice>();
  for (const card of [...cards.filter(builtIn), ...saved]) {
    if (choices.has(card.provider)) continue;
    if (!builtIn(card) && !(grouped ? card.group : COMMON_PROVIDERS.includes(card.provider))) continue;
    const common = grouped ? card.group === "common" : COMMON_PROVIDERS.includes(card.provider);
    choices.set(card.provider, {
      provider: card.provider,
      label: options.label(card),
      group: common ? "common" : grouped ? (card.group ?? null) : null,
      icon: iconKey(card),
      card,
    });
  }
  const offered = [...choices.values()];
  const rank = (choice: ProviderChoice) => {
    const index = COMMON_PROVIDERS.indexOf(choice.provider);
    return index < 0 ? COMMON_PROVIDERS.length : index;
  };
  const common = offered.filter((choice) => choice.group === "common").sort((a, b) => rank(a) - rank(b));
  const rest = offered
    .filter((choice) => choice.group !== "common")
    .sort((a, b) => a.label.localeCompare(b.label, "en", { sensitivity: "base" }) || a.provider.localeCompare(b.provider));
  const filters = PROVIDER_FILTERS.filter((filter) => filter === "all" || rest.some((choice) => choice.group === filter));
  const filter = filters.includes(options.filter) ? options.filter : "all";
  const found = (choice: ProviderChoice) => matches(choice.card, choice.label, options.query);
  return {
    saved: saved.filter((card) => matches(card, options.label(card), options.query)),
    common: common.filter(found),
    all: rest.filter((choice) => (filter === "all" || choice.group === filter) && found(choice)),
    filters,
    total: offered.length,
  };
}

/** One or two letters for a provider without a brand mark: the initials of
 * its first two words, without generic ones ("AI", "Labs", …). */
export function monogram(label: string): string {
  const words = label
    .split(/[\s/._-]+/)
    .filter((word) => word && !/^(ai|labs?|api|inc|cloud|inference)$/i.test(word));
  const letters = (words.length ? words : [label]).slice(0, 2).map((word) => [...word][0] ?? "");
  return letters.join("").toUpperCase();
}

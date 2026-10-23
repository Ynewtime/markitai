// How the provider picker arranges the server's provider cards: saved
// connections, the common eight, then every other provider by name under a
// group filter. A provider appears once among the saved rows: the server's
// environment key folds into the provider's connection at its own endpoint,
// and a provider with a saved row is marked as added in Common and All, where
// choosing it opens that connection (contract v1.11). A server that sends no `group` (Python 1.3.0) still works:
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

/** Providers that publish no model list, by both backends' catalogues. A
 * saved connection's card may still claim a list (Python 1.3.0 sends
 * `supports_discovery: true` for every saved connection), so the page knows
 * these by name and offers model IDs by hand instead of a failed load. */
const NO_MODEL_LIST = new Set(["bedrock", "vertex_ai", "vertex_ai_beta", "perplexity", "zai", "fireworks_ai"]);

/** Whether the page can load a provider's model list. */
export const listsModels = (card: Pick<ProviderCard, "provider" | "supports_discovery">): boolean =>
  card.supports_discovery !== false && !NO_MODEL_LIST.has(card.provider);

/** The brand mark a card names; a server that names none (Python 1.3.0) gets
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
  /** The provider has a saved row; choosing it opens that connection. */
  added: boolean;
}

/** How a saved row authenticates: the server's key variable, a key saved in
 * the connection, none, or nothing to say (subscriptions, local servers, clouds). */
export type SavedAuth = { kind: "environment"; variable: string } | { kind: "key" } | { kind: "none" } | null;

const variableOf = (credential: string | undefined): string | null => (credential?.trim().startsWith("env:") ? credential.trim().slice(4).trim() || null : null);

const originOf = (value: string | null | undefined): string | null => {
  if (!value) return null;
  try {
    return new URL(value).origin;
  } catch {
    return value;
  }
};

/** A connection at its provider's own endpoint (no address, or the default one). */
export const atOwnEndpoint = (card: Pick<ProviderCard, "api_base" | "default_base">): boolean =>
  !card.api_base || (card.default_base != null && originOf(card.api_base) === originOf(card.default_base));

/** What a saved row says about its key; `environment` is the server's key card folded into it. */
export function savedAuth(card: ProviderCard, environment: ProviderCard | null = null, served: string | null = null): SavedAuth {
  if (card.kind === "environment") return { kind: "environment", variable: variableOf(card.credential) ?? card.source };
  const variable = variableOf(card.credential);
  if (variable) return { kind: "environment", variable };
  if (card.api_key_configured) return { kind: "key" };
  if (environment) return { kind: "environment", variable: variableOf(environment.credential) ?? environment.source };
  // A model written by hand without a key (a `legacy:` connection) uses the
  // server's key variable wherever it points, as conversions do.
  if (served && card.provider_id?.startsWith("legacy:")) return { kind: "environment", variable: served };
  if (card.key_optional || isSubscription(card) || card.provider === "custom") return null;
  return card.kind === "configured" ? { kind: "none" } : null;
}

export interface ProviderArrangement {
  /** Saved connections and server environment credentials, as the server lists them,
   * an environment card folded into its provider's connection at its own endpoint. */
  saved: ProviderCard[];
  /** The environment card folded into a saved row, by the row's card id. */
  folded: Map<string, ProviderCard>;
  /** The server's key variable for each provider that has one. */
  variables: Map<string, string>;
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
  const listed = cards.filter((card) => !builtIn(card));
  // The server's key for a provider and the provider's connection at its own
  // endpoint are one row: the connection, which says it uses the key.
  const folded = new Map<string, ProviderCard>();
  const absorbed = new Set<ProviderCard>();
  for (const environment of listed.filter((card) => card.kind === "environment")) {
    const variable = variableOf(environment.credential) ?? environment.source;
    const own = listed.filter((card) => card.kind === "configured" && card.provider === environment.provider && atOwnEndpoint(card) && !folded.has(card.id));
    const host =
      own.find((card) => variableOf(card.credential) === variable) ?? own.find((card) => !card.api_key_configured && !variableOf(card.credential)) ?? own[0];
    if (!host) continue;
    folded.set(host.id, environment);
    absorbed.add(environment);
  }
  const saved = listed.filter((card) => !absorbed.has(card));
  const variables = new Map<string, string>();
  for (const card of listed) if (card.kind === "environment") variables.set(card.provider, variableOf(card.credential) ?? card.source);
  // Every provider but the OpenAI-compatible endpoint (each one is another server) is added once saved.
  const savedRow = new Map<string, ProviderCard>();
  for (const card of saved) if (card.provider !== "custom" && !savedRow.has(card.provider)) savedRow.set(card.provider, card);
  const grouped = cards.some((card) => typeof card.group === "string");
  // One choice per provider: its built-in card, else the first saved or
  // environment connection (whose provider the server no longer offers).
  const choices = new Map<string, ProviderChoice>();
  for (const card of [...cards.filter(builtIn), ...saved]) {
    if (choices.has(card.provider)) continue;
    if (!builtIn(card) && !(grouped ? card.group : COMMON_PROVIDERS.includes(card.provider))) continue;
    const common = grouped ? card.group === "common" : COMMON_PROVIDERS.includes(card.provider);
    const row = savedRow.get(card.provider);
    choices.set(card.provider, {
      provider: card.provider,
      label: options.label(card),
      group: common ? "common" : grouped ? (card.group ?? null) : null,
      icon: iconKey(card),
      card: row ?? card,
      added: row !== undefined,
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
    folded,
    variables,
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

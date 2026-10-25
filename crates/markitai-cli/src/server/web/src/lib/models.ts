import type { Deployment, ModelCandidate } from "../api/types.ts";

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

/** The hosts that refused a key, for the failure hint: the endpoints the
 * server names (v1.21), else the one the page asked. */
export function refusedHosts(endpoints: readonly string[] | undefined, asked: string): string[] {
  const host = (value: string) => {
    try {
      return new URL(value).host;
    } catch {
      return "";
    }
  };
  const hosts = (endpoints?.length ? endpoints : [asked]).map(host).filter(Boolean);
  return [...new Set(hosts)];
}

const normalizeEndpoint = (value: string | null | undefined) => (value ?? "").trim().replace(/\/+$/, "").toLowerCase();

const originOf = (value: string | null | undefined): string => {
  const text = (value ?? "").trim();
  if (!text) return "";
  try {
    return new URL(text).origin.toLowerCase();
  } catch {
    return normalizeEndpoint(text);
  }
};

/** Whether a deployment is routed at this endpoint. The server tells a saved address by its origin
 * only (`https://api.moonshot.cn`, never the path), so origins are compared. A deployment the
 * provider reaches at its own default address stores none: it is at the page of that default
 * (`defaultBase`), not at another region's or a gateway's; without a known default it matches. */
export function sameEndpoint(stored: string | null | undefined, effective: string, defaultBase?: string | null): boolean {
  const own = originOf(stored);
  const page = originOf(effective);
  if (!own) return !page || !defaultBase || page === originOf(defaultBase);
  return own === (page || originOf(defaultBase));
}

/** A model ID without its provider prefix (`openai/gpt-5-mini` → `gpt-5-mini`). */
export function modelName(model: string): string {
  const slash = model.indexOf("/");
  return slash < 0 ? model : model.slice(slash + 1);
}

/** Round 18: a routing weight is a whole number from 0 to 100, a share among the
 * models of one routing group; 0 turns a model off (it stays saved). Both
 * backends refuse any other write with 422 `invalid_weight`. */
export const WEIGHT_MAX = 100;
/** The placeholder of an empty weight field, the same digits in every language. */
export const WEIGHT_PLACEHOLDER = "0\u2013100";

/** The weight a field's text means, or null when it is not a whole number from 0 to 100. */
export function parseWeight(text: string): number | null {
  const value = text.trim();
  if (!/^\d{1,3}$/.test(value)) return null;
  const weight = Number(value);
  return weight <= WEIGHT_MAX ? weight : null;
}

/** ArrowUp/ArrowDown on a weight field: one step from what it holds (or `fallback`), kept within 0–100. */
export function stepWeight(text: string, fallback: number, step: 1 | -1): number {
  const typed = /^\s*\d+\s*$/.test(text) ? Number(text) : fallback;
  return Math.max(0, Math.min(WEIGHT_MAX, typed + step));
}

/** The model families a Claude alias names (Claude Code's documented aliases;
 * `inherit` follows Sonnet, as Python markitai resolves it). */
const CLAUDE_ALIASES: Record<string, string> = {
  sonnet: "sonnet",
  "sonnet[1m]": "sonnet",
  opus: "opus",
  "opus[1m]": "opus",
  opusplan: "opus",
  haiku: "haiku",
  inherit: "sonnet",
};

/** A Claude model's family and version (`claude-sonnet-5-5`, `claude-3-5-haiku-20241022`, `claude-opus-4`). */
function claudeVersion(id: string): { family: string; version: number[]; dated: boolean } | null {
  const dated = /-\d{8}$/.test(id);
  const bare = id.replace(/-\d{8}$/, "");
  const named = /^claude-([a-z]+)-(\d+)(?:-(\d+))?$/.exec(bare);
  if (named) return { family: named[1]!, version: [Number(named[2]), Number(named[3] ?? 0)], dated };
  const numbered = /^claude-(\d+)(?:-(\d+))?-([a-z]+)$/.exec(bare);
  if (numbered) return { family: numbered[3]!, version: [Number(numbered[1]), Number(numbered[2] ?? 0)], dated };
  return null;
}

/**
 * The listed model a saved model ID stands for when the provider names it
 * another way, or null. Claude's subscription runtime (`claude-agent`) takes
 * the aliases `sonnet`, `opus`, `haiku` (and `inherit`) for the newest model
 * of that family; Anthropic and the Claude runtime also take an undated ID for
 * its dated snapshot (`claude-sonnet-4-5` → `claude-sonnet-4-5-20250929`).
 * The provider lists carry no alias data, so this table follows the
 * providers' documentation; every other provider matches by exact ID only.
 */
export function aliasTarget(saved: string, listed: readonly string[]): string | null {
  const slash = saved.indexOf("/");
  if (slash < 0) return null;
  const provider = saved.slice(0, slash).toLowerCase();
  const id = saved.slice(slash + 1);
  if (provider !== "claude-agent" && provider !== "anthropic") return null;
  const prefix = `${provider}/`;
  const ids = listed.filter((model) => model.startsWith(prefix)).map((model) => model.slice(prefix.length));
  const family = provider === "claude-agent" ? CLAUDE_ALIASES[id.toLowerCase()] : undefined;
  const pick = (matches: string[]) => {
    const ranked = matches
      .map((candidate) => ({ candidate, info: claudeVersion(candidate)! }))
      .sort(
        (a, b) =>
          b.info.version[0]! - a.info.version[0]! ||
          b.info.version[1]! - a.info.version[1]! ||
          Number(a.info.dated) - Number(b.info.dated) ||
          b.candidate.localeCompare(a.candidate),
      );
    return ranked.length ? `${prefix}${ranked[0]!.candidate}` : null;
  };
  if (family) return pick(ids.filter((candidate) => claudeVersion(candidate)?.family === family));
  // An undated ID and its dated snapshots: the newest snapshot.
  if (!claudeVersion(id) || claudeVersion(id)!.dated) return null;
  const snapshots = ids.filter((candidate) => /-\d{8}$/.test(candidate) && candidate.replace(/-\d{8}$/, "") === id);
  snapshots.sort((a, b) => b.localeCompare(a));
  return snapshots.length ? `${prefix}${snapshots[0]}` : null;
}

/** One row of a provider page's model list. */
export interface PickerRow {
  /** The row's key and the model a tick adds: the listed ID, or a saved one the list lacks. */
  model: string;
  label: string;
  supports_vision: boolean;
  /** The saved models this row stands for (exact or alias); a tick off removes them all. */
  saved: Deployment[];
  /** Saved IDs that differ from `model` (an alias the provider resolves to it). */
  aliases: string[];
  /** Saved here, but the provider's list does not name it. */
  unlisted: boolean;
}


/**
 * The rows of a provider page's model list: every listed model, each with the
 * saved models of this group and endpoint it stands for (by ID, else by an
 * alias the provider defines), then every saved model of this provider, group
 * and endpoint the list does not show, as a row of its own. Configured rows
 * come first. `provider` is the page's provider (`custom` routes `openai/`),
 * `defaultBase` its documented address (a saved model without an address of
 * its own is at that one).
 */
export function pickerRows(
  provider: string,
  candidates: readonly ModelCandidate[],
  deployments: readonly Deployment[],
  group: string,
  endpoint: string,
  defaultBase?: string | null,
): PickerRow[] {
  const prefix = `${provider === "custom" ? "openai" : provider}/`;
  const here = deployments.filter((deployment) => deployment.routing_group === group.trim() && sameEndpoint(deployment.api_base, endpoint, defaultBase));
  const listed = candidates.map((candidate) => candidate.model);
  const rows: PickerRow[] = candidates.map((candidate) => ({
    model: candidate.model,
    label: candidate.label,
    supports_vision: candidate.supports_vision,
    saved: here.filter((deployment) => deployment.model === candidate.model),
    aliases: [],
    unlisted: false,
  }));
  const byModel = new Map(rows.map((row) => [row.model, row]));
  const extra = new Map<string, PickerRow>();
  // A saved model without an address of its own matches any page of its
  // provider; one the list lacks only shows on the page of its own endpoint.
  const ownEndpoint = (deployment: Deployment) =>
    deployment.api_base ? originOf(deployment.api_base) === originOf(endpoint) : !endpoint || originOf(endpoint) === originOf(defaultBase);
  for (const deployment of here) {
    if (byModel.has(deployment.model) || !deployment.model.startsWith(prefix)) continue;
    const target = aliasTarget(deployment.model, listed);
    const row = target === null ? undefined : byModel.get(target);
    if (row) {
      row.saved.push(deployment);
      if (!row.aliases.includes(deployment.model)) row.aliases.push(deployment.model);
      continue;
    }
    if (!ownEndpoint(deployment)) continue;
    const own = extra.get(deployment.model);
    if (own) own.saved.push(deployment);
    else
      extra.set(deployment.model, {
        model: deployment.model,
        label: modelName(deployment.model),
        supports_vision: false,
        saved: [deployment],
        aliases: [],
        unlisted: true,
      });
  }
  const all = [...rows, ...extra.values()];
  return [...all.filter((row) => row.saved.length > 0), ...all.filter((row) => row.saved.length === 0)];
}

/** The saved deployments a Save removes: those of every row the reader unticked. */
export function removals(rows: readonly PickerRow[], dropped: ReadonlySet<string>): string[] {
  const ids = new Set<string>();
  for (const row of rows) if (dropped.has(row.model)) for (const deployment of row.saved) ids.add(deployment.deployment_id);
  return [...ids];
}

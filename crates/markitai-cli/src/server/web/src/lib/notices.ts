// Conversion warnings, read as notices: one line per kind of problem with a
// count, instead of one line per occurrence. Backends word warnings in free
// English text, so kinds are recognized by shape; anything else is grouped by
// its template (names, numbers and quoted values blanked out) and shown with
// its first wording.

export type NoticeKind = "alt_kept" | "cost_incomplete" | "json_resent" | "image_requests_split" | "resent" | "deployment_skipped" | "other";

export interface NoticeEntry {
  itemKey: string;
  itemName: string;
  text: string;
}

export interface NoticeGroup {
  key: string;
  kind: NoticeKind;
  /** Occurrences, over every covered item. */
  count: number;
  /** Item keys the group speaks for, in first-seen order. */
  items: string[];
  names: string[];
  /** The first wording, and up to 50 for the details. */
  sample: string;
  samples: string[];
  /** Model names a cost or deployment notice is about. */
  models: string[];
}

const SHAPES: [NoticeKind, RegExp][] = [
  ["alt_kept", /^Image analysis failed for [\s\S]+?existing alt text was retained/],
  ["cost_incomplete", /^Cost is incomplete: /],
  ["json_resent", /^LLM JSON-text answer was rejected\b/],
  ["image_requests_split", /^Structured image analysis failed\b[\s\S]*separate caption and description requests/],
  ["resent", /^LLM request to \S+ failed \([\s\S]*\) and was sent again$/],
  ["deployment_skipped", /^LLM deployment \S+ [\s\S]* is skipped for this run$/],
];

/** The text with its variable parts blanked, so repeats of one message group together. */
export function template(text: string): string {
  return text
    .replace(/https?:\/\/\S+/g, "<url>")
    .replace(/(['"`‘“])[^'"`’”]{0,200}(['"`’”])/g, "<q>")
    .replace(/[\w.-]*[\w-]\.[A-Za-z][A-Za-z0-9]{1,5}\b/g, "<file>")
    .replace(/\d+(?:[.,]\d+)*/g, "<n>")
    .replace(/\s+/g, " ")
    .trim();
}

function kindOf(text: string): NoticeKind {
  for (const [kind, pattern] of SHAPES) if (pattern.test(text)) return kind;
  return "other";
}

/** Models named by a cost or deployment warning. */
export function noticeModels(kind: NoticeKind, text: string): string[] {
  if (kind === "cost_incomplete") return [...text.matchAll(/\d+ requests? to (\S+) (?:has|have|reported)/g)].map((match) => match[1] ?? "").filter(Boolean);
  if (kind === "deployment_skipped") return [/^LLM deployment (\S+)/.exec(text)?.[1] ?? ""].filter(Boolean);
  if (kind === "resent") return [/^LLM request to (\S+)/.exec(text)?.[1] ?? ""].filter(Boolean);
  return [];
}

/** The cost warning is the Cost fact in other words: rows do not mark it again. */
export const isCostWarning = (text: string): boolean => /^Cost is incomplete: /.test(text);

/** Warnings a row marks and the Notices filter counts. */
export const rowWarnings = (warnings: string[]): string[] => warnings.filter((text) => !isCostWarning(text));

export function groupNotices(entries: NoticeEntry[]): NoticeGroup[] {
  const groups = new Map<string, NoticeGroup>();
  for (const entry of entries) {
    const text = entry.text.trim();
    if (!text) continue;
    const kind = kindOf(text);
    // A recognized kind is one group whatever its details; others group by template.
    const key = kind === "other" ? `other:${template(text)}` : kind;
    let group = groups.get(key);
    if (!group) {
      group = { key, kind, count: 0, items: [], names: [], sample: text, samples: [], models: [] };
      groups.set(key, group);
    }
    group.count++;
    if (!group.items.includes(entry.itemKey)) {
      group.items.push(entry.itemKey);
      group.names.push(entry.itemName);
    }
    if (group.samples.length < 50 && !group.samples.includes(text)) group.samples.push(text);
    for (const model of noticeModels(kind, text)) if (!group.models.includes(model)) group.models.push(model);
  }
  // Recognized kinds first, then by how much they cover.
  return [...groups.values()].sort((left, right) => Number(left.kind === "other") - Number(right.kind === "other") || right.count - left.count);
}

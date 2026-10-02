// Model cost labels. A subtotal is only as complete as its request coverage:
// a zero or rounded cost establishes nothing without `cost_status: complete`.
import type { AttemptUsage, Pricing } from "../api/types.ts";

export interface PriceWords {
  complete: string;
  partialIncomplete: string;
  partial: string;
  unknown: string;
  recorded: string;
  lastFailed: string;
  lastFailedCost: string;
  last: string;
}

export const PRICE_WORDS: Record<"en" | "zh", PriceWords> = {
  en: {
    complete: "{amount} · all recorded requests priced",
    partialIncomplete: "{amount} known subtotal · complete request count unavailable",
    partial: "{amount} known subtotal · {count} unpriced request(s)",
    unknown: "Price unknown · {amount} known subtotal",
    recorded: "{amount} recorded subtotal · pricing completeness unavailable",
    lastFailed: "Last attempt failed",
    lastFailedCost: "Last attempt failed: {cost}",
    last: "Last attempt: {cost}",
  },
  zh: {
    complete: "{amount} · 所有记录的请求均已计价",
    partialIncomplete: "已知小计 {amount} · 无法获得完整请求数",
    partial: "已知小计 {amount} · {count} 个请求未计价",
    unknown: "价格未知 · 已知小计 {amount}",
    recorded: "已记录小计 {amount} · 无法确认计价是否完整",
    lastFailed: "上次尝试失败",
    lastFailedCost: "上次尝试失败：{cost}",
    last: "上次尝试：{cost}",
  },
};

const fill = (text: string, values: Record<string, string | number>) =>
  text.replace(/\{(\w+)\}/g, (match, name: string) => (name in values ? String(values[name]) : match));

const count = (value: unknown): number => (Number.isSafeInteger(value) && (value as number) > 0 ? (value as number) : 0);

/** Coverage of one attempt's usage; each model row must account for its requests. */
export function attemptPricing(usage: AttemptUsage | undefined | null): Pricing | null {
  if (!usage) return null;
  let priced = 0;
  let unpriced = 0;
  let incomplete = 0;
  for (const row of Object.values(usage.by_model ?? {})) {
    const requests = count(row.requests);
    incomplete += count(row.incomplete_request_observations);
    if (!requests) continue;
    const known = row.priced_requests;
    const missing = row.unpriced_requests;
    const expected = known === 0 ? "unknown" : (missing ?? 0) > 0 || count(row.incomplete_request_observations) > 0 ? "partial" : "complete";
    const consistent =
      Number.isSafeInteger(known) &&
      (known as number) >= 0 &&
      Number.isSafeInteger(missing) &&
      (missing as number) >= 0 &&
      (known as number) + (missing as number) === requests &&
      row.cost_status === expected;
    if (consistent) {
      priced += known as number;
      unpriced += missing as number;
    } else unpriced += requests;
  }
  if (Number.isSafeInteger(usage.requests)) unpriced += Math.max(0, (usage.requests as number) - priced - unpriced);
  if (priced + unpriced === 0 && incomplete === 0) return null;
  const pricing: Pricing = {
    priced_requests: priced,
    unpriced_requests: unpriced,
    cost_status: priced === 0 ? "unknown" : unpriced > 0 || incomplete > 0 ? "partial" : "complete",
  };
  if (incomplete > 0) pricing.incomplete_request_observations = incomplete;
  return pricing;
}

/** The full cost statement, for a title; empty when there is nothing to say. */
export function priceText(cost: number | null | undefined, pricing: Pricing | null | undefined, words: PriceWords = PRICE_WORDS.en): string {
  if (typeof cost !== "number" || !Number.isFinite(cost) || cost < 0) return "";
  const amount = `$${cost.toFixed(6)}`;
  if (pricing?.cost_status === "complete") return fill(words.complete, { amount });
  if (pricing?.cost_status === "partial") {
    return (pricing.incomplete_request_observations ?? 0) > 0
      ? fill(words.partialIncomplete, { amount })
      : fill(words.partial, { amount, count: pricing.unpriced_requests });
  }
  if (pricing?.cost_status === "unknown") return fill(words.unknown, { amount });
  return cost > 0 ? fill(words.recorded, { amount }) : "";
}

interface AttemptItem {
  cost_usd?: number | null;
  error?: string | null;
  diagnostics?: { last_attempt?: { status: string; error: string | null; usage?: AttemptUsage } };
}

export interface AttemptNotice<T> {
  label: string;
  error: T | null;
}

/** What the last attempt left behind when it differs from the retained result:
 * a failed retry/enhance, or usage recorded without a priced output. */
export function attemptNotice<T = { text: string; detail: string }>(
  item: AttemptItem,
  words: PriceWords = PRICE_WORDS.en,
  describe: (error: string) => T = (error) => ({ text: error, detail: "" }) as T,
): AttemptNotice<T> | null {
  const attempt = item.diagnostics?.last_attempt;
  if (!attempt || (attempt.status !== "error" && item.cost_usd != null)) return null;
  const cost = priceText(attempt.usage?.cost_usd, attemptPricing(attempt.usage), words);
  const failed = attempt.status === "error";
  const error = failed && typeof attempt.error === "string" && attempt.error !== item.error ? describe(attempt.error) : null;
  const label = failed ? (cost ? fill(words.lastFailedCost, { cost }) : words.lastFailed) : cost ? fill(words.last, { cost }) : "";
  return { label, error };
}

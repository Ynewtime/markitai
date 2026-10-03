// Model cost labels. A subtotal is only as complete as its request coverage:
// a zero or rounded cost establishes nothing without `cost_status: complete`.
import type { AttemptUsage, Pricing } from "../api/types.ts";
import type { NotificationModel } from "../components/notification.tsx";
import type { Dict, Locale } from "../i18n/index.ts";
import { itemErrorText } from "../i18n/errors.ts";
import { displayName } from "./format.ts";
import { settledIdentity, type SessionItem } from "./session.ts";

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

export interface ActionProblem {
  operation: "retry" | "enhance" | "delete" | "open";
  text: string;
  detail: string;
}

export interface ItemRequestFailure {
  operation: "retry" | "enhance" | "delete";
  error: unknown;
  identity: string | null;
}

export function actionNotification(name: string, t: Dict, action: ActionProblem): NotificationModel {
  const heading = action.operation === "retry" ? t.retryFailed : action.operation === "enhance" ? t.llmEnhanceFailed : action.operation === "delete" ? t.deleteFailed : t.jobLoadFailed;
  return { tone: "error", title: `${displayName(name)} · ${heading}`, message: action.text,
    ...(action.detail && action.detail.trim() !== action.text.trim() ? { detail: action.detail } : {}) };
}

/** A row's complete notice, apart from the optional OCR/plain-retry action.
 * A retained output is still successful; its later attempt owns its failure and cost. */
export function itemNotification(item: SessionItem, t: Dict, locale: Locale, action?: ActionProblem): NotificationModel | null {
  const terminal = item.status === "done" || item.status === "error";
  const retained = item.rerunFailure;
  const failure = retained ?? (item.status === "error" ? { error: item.error, error_code: item.errorCode } : null);
  const problem = failure ? itemErrorText(locale, { ...failure, kind: item.kind }) : null;
  const last = item.diagnostics?.last_attempt;
  const attempt = attemptNotice({ cost_usd: item.costUsd, error: item.error, diagnostics: item.diagnostics ?? undefined }, PRICE_WORDS[locale]);
  const warnings = terminal ? [...item.warnings] : [];
  const imageSkip = item.status === "done" && item.skipped && item.skipReason === "image_only";
  const noModel = item.status === "error" && item.errorCode === "no_model_configured";
  if (!action && !problem && !attempt?.label && !warnings.length && !imageSkip) return null;

  const name = displayName(item.name);
  const failedAttempt = last?.status === "error";
  const actionTitle = action?.operation === "retry" ? t.retryFailed : action?.operation === "enhance" ? t.llmEnhanceFailed : action?.operation === "delete" ? t.deleteFailed : t.jobLoadFailed;
  const heading = action ? actionTitle : retained ? t.rerunRetained(retained.operation) : noModel ? t.noModelTitle : imageSkip ? t.imageSkippedTitle : problem ? (problem.hint ? t.statusSkipped : t.statusFailed) : failedAttempt ? PRICE_WORDS[locale].lastFailed : `${t.statusDone} · ${t.itemWarningsTitle}`;
  const message = action?.text || (noModel ? t.noModelMessage(name) : imageSkip ? t.imageSkipped(name) : problem?.text || (failedAttempt ? itemErrorText(locale, { error: last.error, kind: item.kind }).text : "")) || (warnings.length ? t.itemWarnings(warnings.length) : attempt?.label ? t.attemptUsageNotice : "") || t.statusFailed;
  // Raw wording lives in the disclosure, once. Unknown errors already appear in full.
  const raw = [action?.detail, failure?.error, item.error, failedAttempt ? last.error : null]
    .filter((value): value is string => typeof value === "string" && value.trim() !== "" && value.trim() !== message.trim());
  const detail = [...new Set(raw)].join("\n\n");
  const lastCost = attempt?.label ? priceText(last?.usage?.cost_usd, attemptPricing(last?.usage), PRICE_WORDS[locale]) : "";
  const previousWarnings = !!retained || item.status === "done" && (failedAttempt || !!action);
  const warningsContext = previousWarnings && item.pricing?.cost_status !== "complete" ? priceText(item.costUsd, item.pricing, PRICE_WORDS[locale]) : "";
  return {
    tone: action || retained || (problem && !problem.hint && !noModel) || failedAttempt ? "error" : "warning",
    title: `${name} · ${heading}`,
    message,
    ...(detail ? { detail } : {}),
    ...(warnings.length ? { warnings, warningsTitle: previousWarnings ? t.previousResultWarnings : t.itemWarningsTitle, ...(warningsContext ? { warningsContext } : {}) } : {}),
    ...(lastCost ? { cost: fill(PRICE_WORDS[locale].last, { cost: lastCost }) } : {}),
  };
}

/** Only an observed row's transition is a new outcome. First loaded history is quiet. */
export function terminalNotices(previous: ReadonlyMap<string, string | null>, items: SessionItem[]): { next: Map<string, string | null>; changed: SessionItem[] } {
  const next = new Map<string, string | null>();
  const changed: SessionItem[] = [];
  for (const item of items) {
    const identity = settledIdentity(item);
    next.set(item.key, identity);
    if (identity !== null && previous.has(item.key) && previous.get(item.key) !== identity) changed.push(item);
  }
  return { next, changed };
}

export interface NotificationState {
  sequence: number;
  note: NotificationModel | null;
}

/** Closing does not consume the sequence; every explicit replay remounts its live region. */
export function publishNotice(previous: NotificationState, note: NotificationModel | null): NotificationState {
  return { sequence: previous.sequence + (note === null ? 0 : 1), note };
}

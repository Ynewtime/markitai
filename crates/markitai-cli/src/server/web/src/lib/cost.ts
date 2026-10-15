// What a conversion cost, in words a reader can trust. A subtotal is only as
// complete as its request coverage: a model that ran without a priced answer
// is "unknown", never $0, and a conversion that used no model says so.
import type { Pricing } from "../api/types.ts";
import type { SessionItem } from "./session.ts";

export type CostState =
  | { kind: "none" }
  | { kind: "complete"; usd: number }
  | { kind: "partial"; usd: number; unpriced: number }
  | { kind: "unknown"; usd: number | null };

const finite = (value: number | null | undefined): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;

/** One item or job: its known cost, its coverage, and whether a model ran at all. */
export function costState(cost: number | null | undefined, pricing: Pricing | null | undefined, modelRan: boolean): CostState {
  if (pricing) {
    const known = finite(cost) ? cost : null;
    if (pricing.cost_status === "complete" && known !== null) return { kind: "complete", usd: known };
    if (pricing.cost_status === "partial" && known !== null) return { kind: "partial", usd: known, unpriced: pricing.unpriced_requests };
    return { kind: "unknown", usd: known };
  }
  // Without coverage a positive amount is still only a recorded subtotal.
  if (modelRan || (finite(cost) && cost > 0)) return { kind: "unknown", usd: finite(cost) && cost > 0 ? cost : null };
  return { kind: "none" };
}

export const itemModelRan = (item: Pick<SessionItem, "llmEnhanced" | "models" | "pricing">): boolean =>
  item.llmEnhanced || item.models.length > 0 || (item.pricing !== null && item.pricing.priced_requests + item.pricing.unpriced_requests > 0);

export const itemCost = (item: SessionItem): CostState => costState(item.costUsd, item.pricing, itemModelRan(item));

/** The sum over items: complete only when every item that used a model is. */
export function totalCost(items: SessionItem[]): CostState & { unknownItems: number } {
  let usd = 0;
  let unpriced = 0;
  let unknownItems = 0;
  let partial = false;
  let any = false;
  for (const item of items) {
    const state = itemCost(item);
    if (state.kind === "none") continue;
    any = true;
    if (state.kind === "complete") usd += state.usd;
    else if (state.kind === "partial") {
      usd += state.usd;
      unpriced += state.unpriced;
      partial = true;
    } else {
      unknownItems++;
      if (state.usd !== null) usd += state.usd;
    }
  }
  if (!any) return { kind: "none", unknownItems };
  if (unknownItems > 0) return { kind: "unknown", usd: usd > 0 ? usd : null, unknownItems };
  if (partial) return { kind: "partial", usd, unpriced, unknownItems };
  return { kind: "complete", usd, unknownItems };
}

/** "$0.0123"; tiny amounts keep two significant digits instead of reading "$0". */
export function fmtUsd(usd: number): string {
  if (usd === 0) return "$0";
  if (usd < 0.0001) {
    const digits = Math.min(12, 1 - Math.floor(Math.log10(usd)));
    return `$${usd.toFixed(digits).replace(/0+$/, "")}`;
  }
  const text = usd.toFixed(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return `$${text}`;
}

/** Whether a conversion's options ask for a model: `llm` when the request says, else its preset's. */
export function llmAsked(options: { llm?: boolean | null; preset?: string | null } | null | undefined): boolean {
  if (typeof options?.llm === "boolean") return options.llm;
  return options?.preset === "standard" || options?.preset === "rich";
}

/** What the cost line says before (or instead of) an amount. `amount` = show the cost state. */
export type CostPhase = "queued" | "counting" | "waiting_model" | "no_model_asked" | "amount" | "no_model_answered" | "failed_none" | "stopped_none";

/** One item's cost line by its state: a conversion still running has no cost yet, whatever it asked for. */
export function itemCostPhase(item: Pick<SessionItem, "status" | "operation" | "options" | "skipReason" | "llmEnhanced" | "models" | "pricing" | "costUsd">): CostPhase {
  const asked = llmAsked(item.options) || item.operation === "enhance";
  if (item.status === "queued") return asked ? "queued" : "no_model_asked";
  if (item.status === "running") return item.operation === "enhance" ? "waiting_model" : asked ? "counting" : "no_model_asked";
  if (costState(item.costUsd, item.pricing, itemModelRan(item)).kind !== "none") return "amount";
  if (item.skipReason === "user_stopped") return "stopped_none";
  if (item.status === "error") return "failed_none";
  return asked ? "no_model_answered" : "amount";
}

/** A job's or the list's cost line: while any item that asks for a model is still converting,
 * the amount so far is not the cost. */
export function totalCostPhase(items: Parameters<typeof itemCostPhase>[0][]): "counting" | "amount" {
  return items.some((item) => {
    const phase = itemCostPhase(item);
    return phase === "queued" || phase === "counting" || phase === "waiting_model";
  })
    ? "counting"
    : "amount";
}

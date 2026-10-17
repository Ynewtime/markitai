// Why a model test failed, by category. The server says it in `reason` (API
// contract v1.10); a server that does not yet is read from its text, so the
// page names the same categories for both backends. The provider's own words
// stay available as the detail.
import type { DiscoveryResult, ProbeResult } from "../api/types.ts";

export const TEST_REASONS = ["no_key", "key_rejected", "model_not_found", "rate_limited", "network", "timeout", "provider_error"] as const;
export type TestReason = (typeof TEST_REASONS)[number];

export interface TestFailure {
  reason: TestReason;
  /** The key variable that is not set on the server, when known. */
  variable: string | null;
  /** The server's text, as it came. */
  detail: string;
}

const VARIABLE = /\b([A-Z][A-Z0-9_]*_(?:API_KEY|ACCESS_TOKEN|TOKEN|KEY))\b/;

/** The category of a failure text (LiteLLM's exception names, HTTP statuses, transport errors). */
export function classifyTestText(text: string): TestReason {
  if (/\b[A-Z][A-Z0-9_]*_(?:API_KEY|ACCESS_TOKEN|TOKEN|KEY)\b[^.]*\b(?:is not set|not set|missing|unset|empty)\b|\bno api key\b|missing (?:an? )?api[_ ]?key|api[_ ]?key\b[^.]{0,40}\b(?:must be set|is required|missing|not (?:set|provided|found))/i.test(text)) return "no_key";
  if (/\b(?:Timeout|timed? ?out|Deadline)/i.test(text)) return "timeout";
  if (/RateLimit|\b429\b|rate.?limit|too many requests|quota|insufficient[_ ](?:balance|quota|funds)|\b402\b|billing/i.test(text)) return "rate_limited";
  if (/Authentication|PermissionDenied|Unauthori[sz]ed|Forbidden|\b40[13]\b|invalid[_ ]?(?:api[_ ]?)?key|incorrect api key|api key not valid/i.test(text)) return "key_rejected";
  if (/NotFound|\b404\b|model[_ ]not[_ ]found|does not exist|unknown model|no such model|not a valid model/i.test(text)) return "model_not_found";
  if (/APIConnection|Connect(?:ion)?Error|connection (?:refused|reset|error)|could not (?:connect|resolve)|name or service not known|getaddrinfo|ECONN|ENOTFOUND|dns|ssl|certificate|network|unreachable|redirect/i.test(text)) return "network";
  return "provider_error";
}

export function testFailure(result: Pick<ProbeResult, "detail" | "reason" | "key_variable">): TestFailure {
  const detail = typeof result.detail === "string" ? result.detail : "";
  const told = TEST_REASONS.find((reason) => reason === result.reason);
  const reason = told ?? classifyTestText(detail);
  const variable = typeof result.key_variable === "string" && result.key_variable ? result.key_variable : reason === "no_key" ? (VARIABLE.exec(detail)?.[1] ?? null) : null;
  return { reason, variable, detail };
}

export const RUNTIME_REASONS = ["runtime_missing", "runtime_unsupported", "not_signed_in", "runtime_failed"] as const;
export type RuntimeReason = (typeof RUNTIME_REASONS)[number];

/** A subscription runtime that gave no list: the cause, and what to run about it (Rust, contract v1.18). */
export interface RuntimeFailure {
  reason: RuntimeReason;
  name: string;
  required: string;
  installed: string | null;
  step: { kind: "install" | "sign_in"; command: string } | null;
  /** The server's English sentence, kept as the detail. */
  detail: string;
}

export function runtimeFailure(result: Pick<DiscoveryResult, "status" | "detail" | "reason" | "runtime" | "next_step">): RuntimeFailure | null {
  if (result.status !== "unavailable") return null;
  const reason = RUNTIME_REASONS.find((value) => value === result.reason);
  if (!reason || !result.runtime || typeof result.runtime.name !== "string") return null;
  const step = result.next_step && (result.next_step.kind === "install" || result.next_step.kind === "sign_in") && typeof result.next_step.command === "string" ? result.next_step : null;
  return {
    reason,
    name: result.runtime.name,
    required: String(result.runtime.required ?? ""),
    installed: typeof result.runtime.installed === "string" ? result.runtime.installed : null,
    step,
    detail: result.detail ?? "",
  };
}

/** Whether adding a model by hand can work at all: not while the runtime is missing, unsupported or signed out.
 * A runtime that merely failed may do better next time, so entry stays open with a warning. */
export const addingIsBlocked = (failure: RuntimeFailure | null): boolean => failure !== null && failure.reason !== "runtime_failed";

/** Why a model list could not be loaded, when the server says so (v1.11
 * `reason`) or its text is a failed request (Python 1.2.1's "model discovery
 * failed (…)"); null for a list that is only partial or that the provider does
 * not publish, which keep their own note. */
export function listFailure(result: Pick<DiscoveryResult, "status" | "detail" | "reason" | "key_variable">): TestFailure | null {
  if (result.status !== "unavailable") return null;
  const told = TEST_REASONS.some((reason) => reason === result.reason);
  if (!told && !/discovery (?:failed|returned|timed out|request failed)/i.test(result.detail ?? "")) return null;
  return testFailure({ detail: result.detail ?? "", reason: result.reason, key_variable: result.key_variable });
}

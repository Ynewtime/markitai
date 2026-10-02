// Cost labels; scripts/test_ui_pricing.cjs runs this file for the repository gate.
import assert from "node:assert/strict";
import test from "node:test";
import { attemptNotice, attemptPricing, PRICE_WORDS, priceText } from "./pricing.ts";

const describe = (text: string) => ({ text, detail: "", formats: "" });
const old = { cost_usd: 0.5, error: null, pricing: { cost_status: "complete" as const, priced_requests: 1, unpriced_requests: 0 } };
const usage = { requests: 1, cost_usd: 0, by_model: { legacy: { requests: 1, cost_usd: 0 } } };
const failed = { ...old, diagnostics: { last_attempt: { status: "error", error: "Provider refused this attempt", usage } } };

test("a failed attempt states what it cost and why it failed, once", () => {
  assert.deepEqual(attemptNotice(failed, PRICE_WORDS.en, describe), {
    label: "Last attempt failed: Price unknown · $0.000000 known subtotal",
    error: describe("Provider refused this attempt"),
  });
  assert.equal(priceText(failed.cost_usd, failed.pricing), "$0.500000 · all recorded requests priced");
  assert.equal(attemptNotice({ ...failed, error: "Provider refused this attempt" }, PRICE_WORDS.en, describe)?.error, null);
  assert.deepEqual(attemptNotice({ ...failed, error: "Prior unrelated failure" }, PRICE_WORDS.en, describe)?.error, describe("Provider refused this attempt"));
  assert.equal(attemptNotice({ ...old, diagnostics: { last_attempt: { status: "done", error: null, usage } } }), null);
  assert.equal(attemptNotice(old), null);
  assert.deepEqual(attemptNotice({ cost_usd: null, error: null, diagnostics: { last_attempt: { status: "done", error: null, usage } } }), {
    label: "Last attempt: Price unknown · $0.000000 known subtotal",
    error: null,
  });
  assert.deepEqual(
    attemptNotice({ ...old, diagnostics: { last_attempt: { status: "error", error: "<script>literal</script>", usage: { requests: 1, by_model: {} } } } }, PRICE_WORDS.en, describe),
    { label: "Last attempt failed", error: describe("<script>literal</script>") },
  );
});

test("coverage counts requests, not a zero or rounded subtotal", () => {
  const aggregate = {
    requests: 0,
    cost_usd: 0,
    by_model: { sub: { requests: 0, input_tokens: 30, output_tokens: 7, priced_requests: 0, unpriced_requests: 0, cost_status: "unknown", incomplete_request_observations: 1 } },
  };
  assert.deepEqual(attemptPricing(aggregate), { priced_requests: 0, unpriced_requests: 0, cost_status: "unknown", incomplete_request_observations: 1 });
  assert.equal(priceText(0, attemptPricing(aggregate)), "Price unknown · $0.000000 known subtotal");
  const mixed = { ...aggregate, requests: 1, by_model: { ...aggregate.by_model, api: { requests: 1, priced_requests: 1, unpriced_requests: 0, cost_status: "complete" } } };
  assert.deepEqual(attemptPricing(mixed), { priced_requests: 1, unpriced_requests: 0, cost_status: "partial", incomplete_request_observations: 1 });
  assert.equal(priceText(0.5, attemptPricing(mixed)), "$0.500000 known subtotal · complete request count unavailable");
  assert.equal(priceText(0.25, { priced_requests: 1, unpriced_requests: 2, cost_status: "partial" }), "$0.250000 known subtotal · 2 unpriced request(s)");
  assert.equal(priceText(0.25, undefined), "$0.250000 recorded subtotal · pricing completeness unavailable");
  assert.equal(priceText(0, undefined), "");
  assert.equal(priceText(Number.NaN, undefined), "");
  assert.equal(priceText(0.5, { priced_requests: 1, unpriced_requests: 0, cost_status: "complete" }, PRICE_WORDS.zh), "$0.500000 · 所有记录的请求均已计价");
  assert.equal(attemptPricing(null), null);
  assert.equal(attemptPricing({ requests: 0, by_model: {} }), null);
});

// Cost labels; scripts/test_ui_pricing.cjs runs this file for the repository gate.
import assert from "node:assert/strict";
import test from "node:test";
import { actionNotification, attemptNotice, attemptPricing, itemNotification, publishNotice, quietRestored, terminalNotices, widenNotice, PRICE_WORDS, priceText } from "./pricing.ts";
import { en } from "../i18n/en.ts";
import { zh } from "../i18n/zh.ts";
import { seedItem, type SessionItem } from "./session.ts";

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

const result: SessionItem = { ...seedItem("private-job", { itemId: "1", name: "报告.pdf", kind: "file", sizeBytes: 500 }), status: "done", output: "old.md", finishedAt: "2026-10-03T00:00:00Z", costUsd: 0.5, pricing: { cost_status: "complete", priced_requests: 1, unpriced_requests: 0 } };
const retained: SessionItem = { ...result, rerunFailure: { operation: "enhance", error_code: "enhancement_failed", error: "LLM enhancement did not produce a result: original raw reason", failed_at: "2026-10-03T01:00:00Z" } };

test("a retained output without attempt usage never invents a free or priced attempt", () => {
  const before = structuredClone(retained);
  for (const [locale, words] of [["en", en], ["zh", zh]] as const) {
    const notice = itemNotification(retained, words, locale);
    assert.equal(notice?.tone, "error");
    assert.ok(notice?.title.includes(words.rerunRetained("enhance")));
    assert.ok(notice?.title.includes("报告.pdf"));
    assert.equal(notice?.detail, retained.rerunFailure?.error);
    assert.equal(notice?.cost, undefined);
    assert.ok(!JSON.stringify(notice).includes("$0.000000"));
    assert.ok(!JSON.stringify(notice).includes("$0.500000"));
  }
  assert.deepEqual(retained, before);
  assert.equal(retained.status, "done");
  assert.equal(retained.output, "old.md");
});

test("a paid failed attempt reports its own partial coverage and complete raw errors", () => {
  const item: SessionItem = { ...retained, diagnostics: { last_attempt: { operation: "enhance", status: "error", error: "Different provider wording\nwith all details", usage: { requests: 2, cost_usd: 0.25, by_model: { local: { requests: 2, cost_usd: 0.25, priced_requests: 1, unpriced_requests: 1, cost_status: "partial" } } } } } };
  const before = structuredClone(item);
  const notice = itemNotification(item, en, "en");
  assert.equal(notice?.cost, "Last attempt: $0.250000 known subtotal · 1 unpriced request(s)");
  assert.equal(notice?.detail, `${retained.rerunFailure?.error}\n\nDifferent provider wording\nwith all details`);
  assert.ok(notice?.cost);
  assert.ok(!notice.cost.includes("$0.500000"));
  assert.deepEqual(item, before);
  const complete = { ...item, diagnostics: { last_attempt: { ...item.diagnostics!.last_attempt!, usage: { requests: 1, cost_usd: 0.1, by_model: { local: { requests: 1, cost_usd: 0.1, priced_requests: 1, unpriced_requests: 0, cost_status: "complete" } } } } } };
  assert.equal(itemNotification(complete, zh, "zh")?.cost, "上次尝试：$0.100000 · 所有记录的请求均已计价");
});

test("successful warnings preserve every full value and never call conversion failed", () => {
  const item = { ...result, warnings: ["first warning", "long detail\n" + "字".repeat(5000), "<script>literal warning</script>", "first warning"] };
  for (const [locale, words] of [["en", en], ["zh", zh]] as const) {
    const notice = itemNotification(item, words, locale);
    assert.equal(notice?.tone, "warning");
    // The tone icon and the warnings section carry the state; no text suffix
    // or count line repeats them.
    assert.equal(notice?.title, `报告.pdf · ${words.statusDone}`);
    assert.equal(notice?.message, "");
    assert.deepEqual(notice?.warnings, item.warnings);
    assert.ok(!notice?.title.includes(words.statusFailed));
    notice!.warnings![0] = "only the notification copy changed";
    assert.equal(item.warnings[0], "first warning");
  }
  assert.equal(itemNotification(result, en, "en"), null);
  assert.equal(itemNotification({ ...result, status: "running", warnings: ["old warning"] }, en, "en"), null);
});

test("initial errors and action refusals preserve raw details without duplicate sentences", () => {
  const original = "HTTP 503: exact service response\n" + "token".repeat(100);
  const item: SessionItem = { ...result, status: "error", error: original, errorCode: "fetch_error", output: null };
  const notice = itemNotification(item, en, "en");
  assert.equal(notice?.tone, "error");
  assert.equal(notice?.detail, original);
  const literal = itemNotification({ ...item, error: "Unrecognized full message", errorCode: null }, en, "en");
  assert.equal(literal?.message, "Unrecognized full message");
  assert.equal(literal?.detail, undefined);
  const problem = { operation: "delete" as const, text: "Refused", detail: "Full response\nunchanged" };
  const refused = itemNotification(result, en, "en", problem);
  assert.equal(refused?.title, "报告.pdf · Deletion failed");
  assert.equal(refused?.detail, problem.detail);
  assert.equal(refused?.cost, undefined);
  assert.equal(result.status, "done");
  assert.equal(result.output, "old.md");
  assert.deepEqual(actionNotification("report.pdf", en, { ...problem, detail: "Refused" }), { tone: "error", title: "report.pdf · Deletion failed", message: "Refused" });
});

test("no-model and image-skip notices remain actionable descriptions", () => {
  const missing: SessionItem = { ...result, status: "error", output: null, errorCode: "no_model_configured", error: "No model configured" };
  assert.equal(itemNotification(missing, en, "en")?.title, `报告.pdf · ${en.noModelTitle}`);
  assert.equal(itemNotification(missing, en, "en")?.tone, "warning");
  const image = { ...result, skipped: true, skipReason: "image_only", output: null };
  assert.equal(itemNotification(image, zh, "zh")?.message, zh.imageSkipped("报告.pdf"));
  assert.equal(itemNotification({ ...image, skipReason: "exists" }, en, "en"), null);
});

test("retained warnings belong to the prior result, independently of a new unpriced or paid refusal", () => {
  const raw = "Some observed LLM requests have unknown prices; known subtotal is not the complete charge.";
  const item: SessionItem = { ...retained, costUsd: 0, pricing: { cost_status: "unknown", priced_requests: 0, unpriced_requests: 1 }, warnings: [raw] };
  const before = structuredClone(item);
  const notice = itemNotification(item, zh, "zh");
  assert.equal(notice?.warningsTitle, "旧结果提示");
  assert.deepEqual(notice?.warnings, [raw]);
  assert.equal(notice?.warningsContext, "价格未知 · 已知小计 $0.000000");
  assert.equal(notice?.cost, undefined);
  assert.deepEqual(item, before);
  const paid: SessionItem = { ...item, diagnostics: { last_attempt: { operation: "enhance", status: "error", error: "HTTP 503 current attempt", usage: { cost_usd: 0.1, requests: 1, by_model: { local: { requests: 1, priced_requests: 1, unpriced_requests: 0, cost_status: "complete" } } } } } };
  const paidNote = itemNotification(paid, en, "en");
  assert.equal(paidNote?.warningsTitle, "Previous result warnings");
  assert.deepEqual(paidNote?.warnings, [raw]);
  assert.equal(paidNote?.warningsContext, "Price unknown · $0.000000 known subtotal");
  assert.equal(paidNote?.cost, "Last attempt: $0.100000 · all recorded requests priced");
});

test("terminal identities are quiet for identical snapshots, locale changes and first history adoption", () => {
  const seed = { ...result, status: "queued" as const, finishedAt: null };
  const first = terminalNotices(new Map(), [result]);
  assert.deepEqual(first.changed, []);
  assert.deepEqual(terminalNotices(first.next, [{ ...result }]).changed, []);
  // The dictionary is absent from the identity: changing the UI language cannot replay history.
  itemNotification({ ...result, warnings: ["warning"] }, zh, "zh");
  assert.deepEqual(terminalNotices(first.next, [result]).changed, []);
  const queued = terminalNotices(new Map(), [seed]);
  assert.deepEqual(terminalNotices(queued.next, [result]).changed, [result]);
  const afterResult = terminalNotices(queued.next, [result]);
  assert.deepEqual(terminalNotices(afterResult.next, [retained]).changed, [retained]);
  const afterFailure = terminalNotices(afterResult.next, [retained]);
  assert.deepEqual(terminalNotices(afterFailure.next, [{ ...retained }]).changed, []);
  const secondFailure = { ...retained, rerunFailure: { ...retained.rerunFailure!, failed_at: "2026-10-03T02:00:00Z" } };
  assert.deepEqual(terminalNotices(afterFailure.next, [secondFailure]).changed, [secondFailure]);
});

test("a job restored from this tab's seeds settles without a notice until the reader acts on it", () => {
  const quiet = new Set([result.jobId]);
  assert.deepEqual(quietRestored([result], quiet), []);
  assert.deepEqual(quietRestored([{ ...result, jobId: "fresh-job" }], quiet), [{ ...result, jobId: "fresh-job" }]);
  // Retry or enhance clears the mark: that run is one the reader asked for.
  quiet.delete(result.jobId);
  assert.deepEqual(quietRestored([result], quiet), [result]);
});

test("replaying the same notice while open or after close publishes a new live-region key", () => {
  const notice = itemNotification(retained, en, "en")!;
  const first = publishNotice({ sequence: 0, note: null }, notice);
  const again = publishNotice(first, notice);
  assert.equal(again.sequence, first.sequence + 1);
  assert.equal(again.note, notice);
  const closed = publishNotice(again, null);
  assert.equal(closed.sequence, again.sequence);
  assert.equal(closed.note, null);
  const reopened = publishNotice(closed, notice);
  assert.equal(reopened.sequence, closed.sequence + 1);
  assert.equal(first.note, notice);
  assert.equal(first.sequence, 1);
});

test("a shared conversion warning widens one card instead of repeating per row", () => {
  const warning = "Cost is incomplete: 2 requests to openai/gpt-4.1-mini have no reviewed price.";
  const first = itemNotification({ ...result, warnings: [warning] }, en, "en")!;
  const second = itemNotification({ ...result, key: "private-job/2", name: "b.pdf", warnings: [warning] }, en, "en")!;
  assert.deepEqual(first.covers, ["报告.pdf"]);
  const widened = widenNotice(first, second);
  assert.deepEqual(widened?.covers, ["报告.pdf", "b.pdf"]);
  assert.deepEqual(widened?.warnings, [warning]);
  assert.equal(widened?.detail, "报告.pdf\nb.pdf");
  assert.equal(widened?.warningsContext, undefined, "a per-row subtotal would be wrong for two rows");
  // A different warning, an action card or a failure is about one row and replaces the card.
  const other = "Image analysis failed; base Markdown and assets retained: timeout";
  assert.equal(widenNotice(first, itemNotification({ ...result, warnings: [other] }, en, "en")!), null);
  assert.equal(widenNotice(first, itemNotification({ ...result, warnings: [warning] }, en, "en", { operation: "retry", text: "refused", detail: "" })!), null);
  assert.equal(widenNotice(first, itemNotification({ ...result, status: "error", error: "boom", warnings: [warning] }, en, "en")!), null);
  assert.equal(widenNotice(null, second), null);
  // The same row twice does not inflate the count.
  assert.equal(widenNotice(first, itemNotification({ ...result, warnings: [warning] }, en, "en")!), null);
});

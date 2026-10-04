import assert from "node:assert/strict";
import test from "node:test";
import type { HistoryEntry, ItemPayload, JobSnapshot } from "../api/types.ts";
import { emptyOptions } from "./options.ts";
import {
  canRetry,
  settleCurrentJobSnapshot,
  failedToRetry,
  mergeItem,
  mergeLedger,
  readSeeds,
  reconcile,
  requeued,
  rowMatches,
  seedItem,
  sessionStats,
  settledIdentity,
  waitingJobs,
  writeSeeds,
  type SessionItem,
  type SessionJob,
} from "./session.ts";

function payload(id: string, patch: Partial<ItemPayload> = {}): ItemPayload {
  return {
    item_id: id,
    name: `${id}.pdf`,
    kind: "file",
    status: "done",
    error: null,
    output: `${id}.pdf.md`,
    output_name: null,
    duration_ms: 100,
    finished_at: "2026-10-02T08:00:00Z",
    cost_usd: null,
    llm_enhanced: false,
    operation: "convert",
    skipped: false,
    skip_reason: null,
    retryable: true,
    warnings: [],
    ...patch,
  };
}

const row = (job: string, id: string, patch: Partial<ItemPayload> = {}): SessionItem =>
  mergeItem(seedItem(job, { itemId: id, name: `${id}.pdf`, kind: "file", sizeBytes: 10 }), payload(id, patch), 1000);

const job = (jobId: string, status: SessionJob["status"], createdAt: string | null = null): SessionJob => ({
  jobId,
  status,
  createdAt,
  options: emptyOptions(),
  persistenceError: null,
});

test("delayed final snapshots cannot overwrite a later retry or repopulate cleared jobs", async () => {
  const snapshot = (status: JobSnapshot["status"]): JobSnapshot => ({
    job_id: "j", status, created_at: "2026-10-02T08:00:00Z", finished_at: null,
    options: {}, items: [payload("1")], done: 1, failed: 0, total: 1,
  });
  const generations = new Map<string, symbol>();
  let received: JobSnapshot[] = [];
  let notifications = 0;
  const receive = (value: JobSnapshot) => { received.push(value); };
  const complete = () => { notifications++; };
  for (const invalidate of ["retry", "clear"] as const) {
    const old = Symbol("old");
    generations.set("j", old);
    let resolve!: (value: JobSnapshot) => void;
    const pending = new Promise<JobSnapshot>((done) => { resolve = done; });
    const finishing = settleCurrentJobSnapshot(pending, () => generations.get("j") === old, receive, complete);
    if (invalidate === "retry") generations.set("j", Symbol("next"));
    else generations.clear();
    resolve(snapshot("done"));
    await finishing;
    assert.deepEqual(received, []);
    assert.equal(notifications, 0);
  }
  const latest = Symbol("latest");
  generations.set("j", latest);
  const current = () => generations.get("j") === latest;
  await settleCurrentJobSnapshot(Promise.resolve(snapshot("running")), current, receive, complete);
  assert.deepEqual(received.map((value) => value.status), ["running"]);
  assert.equal(notifications, 0);
  received = [];
  await settleCurrentJobSnapshot(Promise.resolve(snapshot("done")), current, receive, complete);
  assert.deepEqual(received.map((value) => value.status), ["done"]);
  assert.equal(notifications, 1);
  await settleCurrentJobSnapshot(Promise.resolve(null), current, receive, complete);
  assert.equal(notifications, 1);
});

test("events merge into rows, keep the known upload size and time a running row", () => {
  const seeded = seedItem("j", { itemId: "i1", name: "a.pdf", kind: "file", sizeBytes: 42 });
  const running = mergeItem(seeded, payload("i1", { status: "running", output: null }), 5000);
  assert.deepEqual([running.status, running.startedAt, running.sizeBytes], ["running", 5000, 42]);
  const still = mergeItem(running, payload("i1", { status: "running", output: null }), 9000);
  assert.equal(still.startedAt, 5000);
  const failed = mergeItem(still, payload("i1", { status: "error", error: "x", error_code: "conversion_error" }), 9500);
  assert.deepEqual([failed.errorCode, failed.startedAt], ["conversion_error", null]);
  const again = requeued(failed, "retry");
  assert.deepEqual([again.status, again.error, again.errorCode, again.operation], ["queued", null, null, "retry"]);
});

test("a snapshot owns membership but keeps each job's place in the list", () => {
  const before = [row("a", "1"), row("b", "1"), row("b", "2"), row("c", "1")];
  const after = reconcile(before, "b", [payload("2"), payload("3")], 0);
  assert.deepEqual(
    after.map((item) => item.key),
    ["a/1", "b/2", "b/3", "c/1"],
  );
  assert.equal(after[1]?.sizeBytes, 10);
  assert.equal(after[2]?.sizeBytes, null);
});

test("retained rerun failures survive snapshots without invented usage and clear on a new attempt", () => {
  const original = row("j", "1", { cost_usd: 0.25, llm_enhanced: true, operation: "enhance" });
  const outcome = { operation: "enhance" as const, error_code: "enhancement_failed", error: "No enhanced result", failed_at: "2026-10-03T00:00:00Z" };
  const incoming = payload("1", { cost_usd: 0.25, llm_enhanced: true, operation: "enhance", rerun_failure: outcome });
  const restored = reconcile([original], "j", [incoming], 2000)[0]!;
  assert.deepEqual(restored.rerunFailure, outcome);
  assert.equal(restored.output, original.output);
  assert.equal(restored.finishedAt, original.finishedAt);
  assert.equal(restored.diagnostics, null);
  assert.equal(canRetry(restored), true);
  assert.notEqual(settledIdentity(restored), settledIdentity(original));
  const again = mergeItem(restored, { ...incoming, rerun_failure: { ...outcome, failed_at: "2026-10-03T00:01:00Z" } }, 3000);
  assert.notEqual(settledIdentity(again), settledIdentity(restored));
  assert.deepEqual(sessionStats([restored]), sessionStats([original]));
  const queued = requeued(restored, "retry");
  assert.equal(queued.rerunFailure, null);
  assert.equal(settledIdentity(queued), null);
  const success = mergeItem(queued, payload("1", { finished_at: "2026-10-03T00:02:00Z" }), 4000);
  assert.equal(success.rerunFailure, null);
  assert.equal(canRetry(success), false);
  assert.notEqual(settledIdentity(success), settledIdentity(restored));
});

test("counters keep skipped and failed apart from converted", () => {
  const stats = sessionStats([
    row("j", "1", { duration_ms: 100 }),
    row("j", "2", { skipped: true, skip_reason: "image_only" }),
    row("j", "3", { status: "error", error: "x" }),
    row("j", "4", { cost_usd: 0.25, llm_enhanced: true, duration_ms: 50 }),
  ]);
  assert.deepEqual(stats, { done: 2, skipped: 1, failed: 1, total: 4, costTotal: 0.25, hasCost: true, doneDurationMs: 150 });
});

test("retry is offered only where converting again can help", () => {
  assert.equal(canRetry(row("j", "1")), false);
  assert.equal(canRetry(row("j", "1", { status: "error", error: "x" })), true);
  assert.equal(canRetry(row("j", "1", { skipped: true, skip_reason: "image_only" })), true);
  assert.equal(canRetry(row("j", "1", { skipped: true, skip_reason: "pending_batch" })), false);
  assert.equal(canRetry(row("j", "1", { status: "error", error: "x", retryable: false })), false);
  assert.equal(canRetry(row("j", "1", { status: "error", error_code: "unsupported", error: "Unsupported file format: '.x'." })), false);
  const items = [row("j", "1", { status: "error", error: "x" }), row("j", "2", { skipped: true, skip_reason: "image_only" })];
  assert.deepEqual(failedToRetry(items).map((item) => item.itemId), ["1"]);
});

test("only running jobs with original items still waiting can be stopped", () => {
  const items = [row("a", "1", { status: "queued" }), row("b", "1", { status: "queued", operation: "retry" }), row("c", "1", { status: "queued" })];
  assert.deepEqual(waitingJobs(items, { a: job("a", "running"), b: job("b", "running"), c: job("c", "done") }), ["a"]);
});

function entry(id: string, finished: string, patch: Partial<HistoryEntry> = {}): HistoryEntry {
  return {
    job_id: id,
    created_at: finished,
    finished_at: finished,
    status: "done",
    total: 2,
    done: 2,
    failed: 0,
    skipped: 0,
    llm_enhanced: 0,
    cost_usd: null,
    names_preview: ["a.pdf", "b.docx"],
    kinds_preview: ["file", "file"],
    duration_ms: 10,
    size_bytes: 100,
    origin: "web",
    retryable: true,
    ...patch,
  };
}

test("live and saved jobs share one ledger, newest activity first, running on top", () => {
  const items = [row("old", "1", { finished_at: "2026-10-01T00:00:00Z" }), row("live", "1", { status: "running" }), row("live", "2")];
  const rows = mergeLedger(items, { old: job("old", "done"), live: job("live", "running") }, [
    entry("saved", "2026-10-01T12:00:00Z"),
    entry("live", "2026-10-02T00:00:00Z"),
  ]);
  assert.deepEqual(
    rows.map((value) => value.key),
    ["live/1", "live/2", "archive:saved", "old/1"],
  );
  const [first, , saved] = rows;
  assert.ok(first && saved);
  assert.equal(rowMatches(first, "all", "1.PDF"), true);
  assert.equal(rowMatches(saved, "failed", ""), false);
  assert.equal(rowMatches(saved, "done", "b.doc"), true);
  assert.equal(rowMatches({ kind: "session", key: "k", item: row("x", "9", { skipped: true }) }, "skipped", ""), true);
});

test("session seeds survive a reload and a damaged store reads as empty", () => {
  const values = new Map<string, string>();
  const store = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    removeItem: (key: string) => void values.delete(key),
  };
  writeSeeds([{ jobId: "j", items: [{ itemId: "i1", name: "a", kind: "file", sizeBytes: 1 }] }], store);
  assert.equal(readSeeds(store)[0]?.jobId, "j");
  values.set("markitai.session", '[{"jobId":1},null,"x",{"jobId":"ok","items":[]}]');
  assert.deepEqual(readSeeds(store).map((value) => value.jobId), ["ok"]);
  values.set("markitai.session", "{");
  assert.deepEqual(readSeeds(store), []);
  writeSeeds([], store);
  assert.equal(values.has("markitai.session"), false);
});

test("snapshot and SSE item options keep per-item selections but never saved consent", () => {
  const initial = seedItem("job", { itemId: "one", name: "one.pdf", kind: "file", sizeBytes: 10 });
  const selected = { ...emptyOptions(), backend: "cloudflare" as const, strategy: "auto" as const, remote_processing: "cloudflare" as const };
  const merged = mergeItem(initial, payload("one", { options: selected }), 1000);
  assert.equal(merged.options?.backend, "cloudflare");
  assert.equal(merged.options?.strategy, "auto");
  assert.equal(merged.options?.remote_processing, undefined);
  assert.equal(selected.remote_processing, "cloudflare");
  const changed = mergeItem(merged, payload("one", { options: { ...emptyOptions(), backend: "native" } }), 1001);
  assert.equal(changed.options?.backend, "native");
  const oldServer = mergeItem(changed, payload("one"), 1002);
  assert.equal(oldServer.options, undefined);
  const rows = reconcile([], "job", [payload("one", { options: selected }), payload("two", { options: { ...emptyOptions(), backend: "native" } })], 1000);
  assert.deepEqual(rows.map((item) => item.options?.backend), ["cloudflare", "native"]);
});

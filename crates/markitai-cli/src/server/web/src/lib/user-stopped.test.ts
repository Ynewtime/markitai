import assert from "node:assert/strict";
import test from "node:test";
import type { ItemPayload, JobSnapshot } from "../api/types.ts";
import { emptyOptions } from "./options.ts";
import { en } from "../i18n/en.ts";
import { zh } from "../i18n/zh.ts";
import { counts, itemMatches } from "./conversions.ts";
import { prepareRetryBatch, retryBatchAnnouncement } from "./retry-batch.ts";
import { canRetry, itemFromPayload, requeued, sessionStats, snapshotStats } from "./session.ts";

function payload(id: string, patch: Partial<ItemPayload> = {}): ItemPayload {
  return { item_id: id, name: `${id}.txt`, kind: "file", status: "error", error: "cancelled (stopped by request)", error_code: "cancelled", output: null, output_name: null,
    duration_ms: null, finished_at: "2026-10-04T10:00:00Z", cost_usd: null, llm_enhanced: false, operation: "convert", skipped: false, skip_reason: null, retryable: true,
    warnings: [], options: { ...emptyOptions(), ocr: false, backend: "native" }, ...patch };
}
function snapshot(items: ItemPayload[], id = "j"): JobSnapshot {
  return { job_id: id, status: "done", created_at: "2026-10-04T09:00:00Z", finished_at: "2026-10-04T10:00:00Z", options: {}, items, done: 0, failed: items.length, total: items.length };
}

test("requested stops use skipped presentation without changing wire status, cost, options or output", () => {
  const raw = payload("stop", { cost_usd: 0.02 });
  const item = itemFromPayload("j", raw, 0);
  assert.equal(raw.status, "error"); assert.equal(raw.skipped, false);
  assert.equal(item.status, "done"); assert.equal(item.skipped, true); assert.equal(item.skipReason, "user_stopped");
  assert.equal(item.output, null); assert.equal(item.costUsd, 0.02); assert.equal(item.errorCode, "cancelled"); assert.equal(item.error, raw.error);
  assert.deepEqual(item.options, raw.options); assert.equal(canRetry(item), true);
  assert.deepEqual(sessionStats([item]), { done: 0, failed: 0, skipped: 1, total: 1, costTotal: 0.02, hasCost: true, doneDurationMs: 0 });
  assert.equal(itemMatches(item, "all", ""), true);
  assert.equal(itemMatches(item, "failed", ""), false);
  assert.deepEqual([counts([item]).stopped, counts([item]).failed], [1, 0]);
  const queued = requeued(item,"retry"); assert.equal(queued.status,"queued"); assert.equal(queued.skipped,false); assert.equal(queued.skipReason,null); assert.equal(queued.error,null); assert.deepEqual(queued.options,item.options);
});

test("shutdown, internal failures, ambiguous codes, running items and retained output are never mapped to requested stops", () => {
  for (const patch of [
    {error_code:"shutdown",error:"cancelled (server shutdown)"}, {error_code:"internal_error"}, {error_code:"future_code"},
    {error_code:undefined,error:"cancelled"}, {status:"running" as const}, {output:"retained.md"},
  ]) {
    const raw=payload("other",patch); const item=itemFromPayload("j",raw,0);
    assert.equal(item.status,raw.status); assert.equal(item.skipped,false); assert.equal(item.skipReason,null);
  }
  assert.equal(itemFromPayload("j",payload("legacy",{error_code:undefined}),0).skipReason,"user_stopped");
});

test("only failed and explicitly stopped items join batch retries; per-item options survive", () => {
  const stopped=itemFromPayload("j",payload("stop"),0);
  const failed=itemFromPayload("j",payload("failed",{error_code:"shutdown",error:"cancelled (server shutdown)"}),0);
  const image=itemFromPayload("j",payload("image",{status:"done",error:null,error_code:undefined,skipped:true,skip_reason:"image_only"}),0);
  const running=itemFromPayload("j",payload("running",{status:"running",error:null,error_code:undefined}),0);
  const batch=prepareRetryBatch([stopped,failed,image,running],{}, {...emptyOptions(),ocr:true});
  assert.deepEqual(batch.map(x=>x.item.itemId),["stop","failed"]); assert.equal(batch[0].options.ocr,false); assert.equal(batch[0].options.backend,"native");
});

test("English and Chinese stop wording and completion totals agree with stopped rows", () => {
  const item=itemFromPayload("j",payload("stop"),0);
  for (const [lang,t] of [["en",en],["zh",zh]] as const) {
    assert.ok(t.statusStopped.length > 0 && t.stoppedLine.length > 8 && lang.length === 2);
    const shutdown = itemFromPayload("j",payload("shutdown",{error_code:"shutdown",error:"cancelled (server shutdown)"}),0);
    assert.deepEqual([counts([item, shutdown]).stopped, counts([item, shutdown]).failed], [1, 1]);
  }
  assert.equal(en.notifyBody(6,0,8),"6 done · 8 skipped"); assert.equal(zh.notifyBody(6,0,8),"完成 6 · 跳过 8");
  const stats=snapshotStats(snapshot([payload("stop"),payload("success",{status:"done",error:null,error_code:undefined,output:"success.md"}),payload("bad",{error_code:"internal_error"})]));
  assert.deepEqual([stats.done,stats.failed,stats.skipped],[1,1,1]);
});

test("batch retry announcements match captured failed/stopped selection in both languages", () => {
  const stopped=itemFromPayload("j",payload("stop"),0);
  const failed=itemFromPayload("j",payload("failed",{error_code:"internal_error"}),0);
  for (const t of [en,zh]) {
    const mixed=prepareRetryBatch([failed,stopped],{},emptyOptions());
    assert.equal(retryBatchAnnouncement(mixed,{attempted:2,failed:0},t),t.announceRetryStopped(2));
    assert.equal(retryBatchAnnouncement(mixed,{attempted:2,failed:1},t),t.announceRetryStoppedFailed(2,1));
    const onlyFailed=prepareRetryBatch([failed],{},emptyOptions());
    assert.equal(retryBatchAnnouncement(onlyFailed,{attempted:1,failed:0},t),t.announceRetryAll(1));
    assert.equal(retryBatchAnnouncement(onlyFailed,{attempted:1,failed:1},t),t.announceRetryAllFailed(1,1));
  }
});


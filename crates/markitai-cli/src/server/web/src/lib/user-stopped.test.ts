import assert from "node:assert/strict";
import test from "node:test";
import type { HistoryEntry, ItemPayload, JobSnapshot } from "../api/types.ts";
import { emptyOptions } from "./options.ts";
import { en } from "../i18n/en.ts";
import { zh } from "../i18n/zh.ts";
import { resolveHistoryPresentation, HistorySummaryError, historySummaryProblem, mergeHistoryErrors } from "./history-presentation.ts";
import { itemNotification } from "./pricing.ts";
import { prepareRetryBatch, retryBatchAnnouncement } from "./retry-batch.ts";
import { canRetry, itemFromPayload, requeued, rowMatches, sessionStats, snapshotStats } from "./session.ts";

function payload(id: string, patch: Partial<ItemPayload> = {}): ItemPayload {
  return { item_id: id, name: `${id}.txt`, kind: "file", status: "error", error: "cancelled (stopped by request)", error_code: "cancelled", output: null, output_name: null,
    duration_ms: null, finished_at: "2026-10-04T10:00:00Z", cost_usd: null, llm_enhanced: false, operation: "convert", skipped: false, skip_reason: null, retryable: true,
    warnings: [], options: { ...emptyOptions(), ocr: false, backend: "native" }, ...patch };
}
function snapshot(items: ItemPayload[], id = "j"): JobSnapshot {
  return { job_id: id, status: "done", created_at: "2026-10-04T09:00:00Z", finished_at: "2026-10-04T10:00:00Z", options: {}, items, done: 0, failed: items.length, total: items.length };
}
function history(id = "j", patch: Partial<HistoryEntry> = {}): HistoryEntry {
  return { job_id: id, created_at: "2026-10-04T09:00:00Z", finished_at: "2026-10-04T10:00:00Z", status: "done", total: 1, done: 0, failed: 1, skipped: 0, llm_enhanced: 0, cost_usd: null, names_preview: ["a.txt"], kinds_preview: ["file"], duration_ms: null, size_bytes: 0, origin: "web", retryable: true, ...patch };
}

test("requested stops use skipped presentation without changing wire status, cost, options or output", () => {
  const raw = payload("stop", { cost_usd: 0.02 });
  const item = itemFromPayload("j", raw, 0);
  assert.equal(raw.status, "error"); assert.equal(raw.skipped, false);
  assert.equal(item.status, "done"); assert.equal(item.skipped, true); assert.equal(item.skipReason, "user_stopped");
  assert.equal(item.output, null); assert.equal(item.costUsd, 0.02); assert.equal(item.errorCode, "cancelled"); assert.equal(item.error, raw.error);
  assert.deepEqual(item.options, raw.options); assert.equal(canRetry(item), true);
  assert.deepEqual(sessionStats([item]), { done: 0, failed: 0, skipped: 1, total: 1, costTotal: 0.02, hasCost: true, doneDurationMs: 0 });
  for (const filter of ["all", "skipped"] as const) assert.equal(rowMatches({kind:"session",key:item.key,item},filter,""),true);
  for (const filter of ["done", "failed"] as const) assert.equal(rowMatches({kind:"session",key:item.key,item},filter,""),false);
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

test("English and Chinese stop notices and completion totals agree with skipped rows", () => {
  const item=itemFromPayload("j",payload("stop"),0);
  for (const [lang,t] of [["en",en],["zh",zh]] as const) {
    const notice=itemNotification(item,t,lang); assert.ok(notice); assert.equal(notice.tone,"warning"); assert.ok(notice.title.includes(t.statusStopped));
    assert.ok(notice.message.length>15); assert.ok(notice.detail?.includes("stopped by request"));
    assert.equal(notice.cost,undefined);
    const shutdown = itemNotification(itemFromPayload("j",payload("shutdown",{error_code:"shutdown",error:"cancelled (server shutdown)"}),0),t,lang);
    assert.equal(shutdown?.tone,"error"); assert.ok(shutdown?.title.includes(t.statusFailed));
  }
  assert.equal(en.notifyBody(6,0,8),"6 done · 8 skipped"); assert.equal(zh.notifyBody(6,0,8),"完成 6 · 跳过 8");
  const stats=snapshotStats(snapshot([payload("stop"),payload("success",{status:"done",error:null,error_code:undefined,output:"success.md"}),payload("bad",{error_code:"internal_error"})]));
  assert.deepEqual([stats.done,stats.failed,stats.skipped],[1,1,1]);
});

test("fresh-tab history derives stop totals from persisted items, keeping real failures and filters consistent", async () => {
  const input=history("j",{total:3,failed:2,done:1});
  const snap=snapshot([payload("stop"),payload("success",{status:"done",error:null,error_code:undefined,output:"success.md"}),payload("bad",{error_code:"shutdown",error:"cancelled (server shutdown)"})]);
  const result=await resolveHistoryPresentation([input],async()=>snap);
  assert.equal(input.failed,2); assert.deepEqual(result.errors,{});
  const entry=result.entries[0]; assert.deepEqual([entry.done,entry.failed,entry.skipped],[2,1,1]);
  for (const filter of ["done","failed","skipped"] as const) assert.equal(rowMatches({kind:"archive",key:"j",entry},filter,""),true);
});

test("history detail failure or stale running snapshot stays explicit instead of guessing skipped", async () => {
  for (const response of [null,{...snapshot([payload("stop")]),status:"running" as const},snapshot([payload("stop")],"wrong")]) {
    const entry=history(); const result=await resolveHistoryPresentation([entry],async()=>response);
    assert.equal(result.entries[0],entry); assert.ok(result.errors.j); assert.equal(result.entries[0].skipped,0);
  }
  const result=await resolveHistoryPresentation([history()],async()=>{throw new Error("offline")});
  assert.ok(result.errors.j instanceof HistorySummaryError);
  assert.equal((result.errors.j as HistorySummaryError).detail,"offline");
});

test("history resolution is limited to loaded failed web entries with at most four concurrent reads", async () => {
  let active=0,max=0;const ids:string[]=[];
  const entries=[...Array.from({length:9},(_,i)=>history(String(i))),history("cli",{origin:"cli"}),history("ok",{failed:0,done:1})];
  const result=await resolveHistoryPresentation(entries,async id=>{ids.push(id);max=Math.max(max,++active);await new Promise(resolve=>setTimeout(resolve,1));active--;return snapshot([payload("stop")],id)});
  assert.equal(ids.length,9);assert.equal(max,4);assert.equal(result.entries.length,11);assert.equal(result.entries[9],entries[9]);
});

test("unmount or newer history refresh discards old reads and stops scheduling", async () => {
  let current=true;let calls=0;const complete:Array<()=>void>=[];
  const entries=Array.from({length:9},(_,i)=>history(String(i)));
  const pending=resolveHistoryPresentation(entries,async id=>{calls++;await new Promise<void>(r=>complete.push(r));return snapshot([payload("stop")],id)},()=>current);
  assert.equal(calls,4);current=false;complete.forEach(r=>r());const result=await pending;
  assert.equal(calls,4);assert.deepEqual(result.entries,entries);assert.deepEqual(result.errors,{});
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

test("history errors are localized while raw detail stays separate; cleared actions do not mask summary errors", () => {
  const summary=new HistorySummaryError(new Error("Saved item details are unavailable"));
  assert.deepEqual(historySummaryProblem(summary,zh),{text:"无法核对已保存的任务汇总 · 请刷新重试",detail:"Saved item details are unavailable"});
  assert.equal(historySummaryProblem(summary,en)?.text,en.historySummaryUnavailable);
  assert.equal(historySummaryProblem(new Error("other"),zh),null);
  for (const actions of [{},{j:undefined},{j:null}]) assert.equal(mergeHistoryErrors({j:summary},actions).j,summary);
  const action=new Error("delete refused");assert.equal(mergeHistoryErrors({j:summary},{j:action}).j,action);
  assert.deepEqual(mergeHistoryErrors({},{}),{});
});

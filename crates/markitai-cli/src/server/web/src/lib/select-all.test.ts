// The list's select-all box (2.0.1): three states over what the filters list, a click that checks all of it
// or clears, and submission headers that agree with it.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ItemPayload } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { checkable, checkState, isGroup, placeholderJob, submissions, toggleAll, type ConvJob } from "./conversions.ts";
import { itemFromPayload, type SessionItem } from "./session.ts";

function payload(id: string, patch: Partial<ItemPayload> = {}): ItemPayload {
  return {
    item_id: id, name: `${id}.pdf`, kind: "file", status: "done", error: null, output: `${id}.pdf.md`, output_name: null,
    duration_ms: 1200, finished_at: "2026-10-11T10:00:00Z", cost_usd: null, llm_enhanced: false, operation: "convert",
    skipped: false, skip_reason: null, retryable: true, warnings: [], ...patch,
  };
}
const item = (jobId: string, id: string, patch: Partial<ItemPayload> = {}): SessionItem => itemFromPayload(jobId, payload(id, patch), 0);
const job = (jobId: string): ConvJob => ({ ...placeholderJob(jobId), status: "done", createdAt: "2026-10-11T09:00:00Z", finishedAt: "2026-10-11T10:00:00Z" });

// A pair (sample.pdf, xlsx-complex.xlsx), a failed single and a single with a notice.
const items = [
  item("pair", "i1", { name: "sample.pdf" }),
  item("pair", "i2", { name: "xlsx-complex.xlsx" }),
  item("bad", "i1", { name: "broken.docx", status: "error", error: "unreadable", output: null }),
  item("note", "i1", { name: "chart.png", warnings: ["image has no description"] }),
];
const jobs = { pair: job("pair"), bad: job("bad"), note: job("note") };
const keys = (list: SessionItem[]) => list.map((entry) => entry.key).sort();

test("the box is off with nothing checked, mixed with part of the list, on with all of it", () => {
  const list = submissions(jobs, items);
  const listed = checkable(list);
  assert.equal(listed.length, 4);
  assert.equal(checkState(listed, new Set()), "off");
  assert.equal(checkState(listed, new Set([items[0]!.key])), "mixed");
  assert.equal(checkState(listed, new Set(keys(items))), "on");
  // a click from off or mixed checks everything listed; from on it clears
  assert.deepEqual([...toggleAll(list, new Set())].sort(), keys(items));
  assert.deepEqual([...toggleAll(list, new Set([items[0]!.key]))].sort(), keys(items));
  assert.equal(toggleAll(list, new Set(keys(items))).size, 0);
});

test("the box covers only what the name filter and the tab list", () => {
  const failed = submissions(jobs, items, "failed");
  assert.deepEqual(keys(checkable(failed)), [items[2]!.key]);
  assert.deepEqual([...toggleAll(failed, new Set())], [items[2]!.key]);
  // an item checked under another tab neither fills nor half-fills the box
  assert.equal(checkState(checkable(failed), new Set([items[0]!.key])), "off");
  assert.equal(checkState(checkable(failed), new Set([items[0]!.key, items[2]!.key])), "on");
  const named = submissions(jobs, items, "all", "XLSX");
  assert.deepEqual([...toggleAll(named, new Set())], [items[1]!.key]);
  const notices = submissions(jobs, items, "notices", "chart");
  assert.deepEqual([...toggleAll(notices, new Set())], [items[3]!.key]);
  // a list narrowed by Show items
  const only = submissions(jobs, items, "all", "", new Set([items[0]!.key, items[3]!.key]));
  assert.deepEqual([...toggleAll(only, new Set())].sort(), keys([items[0]!, items[3]!]));
});

test("submission headers agree with the box, folded or filtered, and converting items stay out", () => {
  const list = submissions(jobs, items);
  const all = toggleAll(list, new Set());
  for (const submission of list.filter(isGroup)) assert.equal(checkState(submission.shown, all), "on");
  // the box takes a group's items whether the group is open or folded: checkable ignores folding
  const half = new Set([items[0]!.key]);
  const pair = list.find((submission) => submission.job.jobId === "pair")!;
  assert.equal(checkState(pair.shown, half), "mixed");
  assert.equal(checkState(checkable(list), half), "mixed");
  // filtered to one of the pair, the header and the box count only that one
  const named = submissions(jobs, items, "all", "sample");
  assert.equal(checkState(named[0]!.shown, half), "on");
  assert.equal(checkState(checkable(named), half), "on");
  // an item still converting cannot be checked; a list of only those locks the box
  const running = [...items, item("live", "i1", { name: "slides.pptx", status: "running", output: null, finished_at: null })];
  const withLive = submissions({ ...jobs, live: { ...job("live"), status: "running" } }, running);
  assert.ok(!toggleAll(withLive, new Set()).has(running[4]!.key));
  assert.equal(checkState(checkable(withLive), new Set(keys(items))), "on");
  assert.equal(checkState([running[4]!], new Set()), "locked");
  assert.equal(checkState(checkable(submissions(jobs, items, "all", "nothing matches")), new Set()), "locked");
});

test("the box is a labelled checkbox at the head of the filter row, over the rows' boxes", () => {
  const list = readFileSync(new URL("../components/conversion-list.tsx", import.meta.url), "utf8");
  const page = readFileSync(new URL("../components/conversions-page.tsx", import.meta.url), "utf8");
  const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
  const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.]/g, "\\.")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";
  assert.match(list, /role="checkbox"/);
  assert.match(list, /aria-checked=\{state === "mixed" \? "mixed" : state === "on"\}/);
  assert.match(list, /aria-label=\{t\.selectAllListed\}/);
  // first in the toolbar, before the name filter, and only while rows show boxes
  const toolbar = page.slice(page.indexOf('"toolbar'));
  assert.ok(toolbar.indexOf("<SelectAll") < toolbar.indexOf('class="search"'));
  assert.match(page, /\{selecting && <SelectAll /);
  // the same 8px inset as the rows' boxes
  assert.match(block(".row"), /padding: 0 8px;/);
  assert.match(block(".toolbar .select-all"), /margin-left: 8px;/);
  for (const dict of [en, zh, ja]) assert.ok(dict.selectAllListed.length > 0);
  assert.equal(zh.selectAllListed, "全选列表中的任务");
});

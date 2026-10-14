// The shared workbench's list, notice, cost, table, front-matter, Compare and
// routing rules; scripts/test_web_dist.py runs every src/**/*.test.ts suite.
import assert from "node:assert/strict";
import test from "node:test";
import type { ItemPayload } from "../api/types.ts";
import { family, tokenize, LANGS } from "../highlight/main.ts";
import { headerActs } from "./acts.ts";
import { locate, norm, type PageLines } from "./compare.ts";
import { counts, hasNotices, isGroup, itemOrder, jobKey, navigationOrder, placeholderJob, submissionLabel, submissions, type ConvJob } from "./conversions.ts";
import { costState, fmtUsd, itemCost, totalCost } from "./cost.ts";
import { parseFrontMatter } from "./frontmatter.ts";
import { groupNotices, template } from "./notices.ts";
import { en } from "../i18n/en.ts";
import { listQuery, parseListState, parseRoute, routePath } from "./route.ts";
import { itemFromPayload, type SessionItem } from "./session.ts";
import { capTables, MARKER, NUMERIC, tableSlice } from "./tables.ts";

function payload(id: string, patch: Partial<ItemPayload> = {}): ItemPayload {
  return {
    item_id: id, name: `${id}.pdf`, kind: "file", status: "done", error: null, output: `${id}.pdf.md`, output_name: null,
    duration_ms: 1200, finished_at: "2026-10-11T10:00:00Z", cost_usd: null, llm_enhanced: false, operation: "convert",
    skipped: false, skip_reason: null, retryable: true, warnings: [], ...patch,
  };
}
const item = (jobId: string, id: string, patch: Partial<ItemPayload> = {}): SessionItem => itemFromPayload(jobId, payload(id, patch), 0);
const job = (jobId: string, patch: Partial<ConvJob> = {}): ConvJob => ({ ...placeholderJob(jobId), status: "done", createdAt: "2026-10-11T09:00:00Z", finishedAt: "2026-10-11T10:00:00Z", ...patch });

test("submissions list newest activity first, running on top, and a filter keeps only matching items", () => {
  const jobs = {
    old: job("old", { finishedAt: "2026-10-10T10:00:00Z" }),
    new: job("new", { finishedAt: "2026-10-11T12:00:00Z" }),
    live: job("live", { status: "running" }),
  };
  const items = [
    item("old", "i1", { finished_at: "2026-10-10T10:00:00Z" }),
    item("new", "i1", { finished_at: "2026-10-11T12:00:00Z" }),
    item("new", "i2", { status: "error", error: "Conversion failed", output: null, error_code: "conversion_error", finished_at: "2026-10-11T12:00:00Z" }),
    item("live", "i1", { status: "running", output: null, finished_at: null }),
  ];
  assert.deepEqual(submissions(jobs, items).map((entry) => entry.job.jobId), ["live", "new", "old"]);
  const failed = submissions(jobs, items, "failed");
  assert.deepEqual(failed.map((entry) => [entry.job.jobId, entry.shown.map((row) => row.itemId)]), [["new", ["i2"]]]);
  assert.deepEqual(submissions(jobs, items, "all", "OLD/").length, 0);
  assert.deepEqual(submissions(jobs, items, "all", "i2.PDF").map((entry) => entry.shown.length), [1]);
  // A submission of one item is a row; of several, a group with a header.
  const list = submissions(jobs, items);
  assert.deepEqual(list.map(isGroup), [false, true, false]);
  assert.deepEqual(navigationOrder(list, new Set()), ["live/i1", jobKey("new"), "new/i1", "new/i2", "old/i1"]);
  assert.deepEqual(navigationOrder(list, new Set(["new"])), ["live/i1", jobKey("new"), "old/i1"]);
  assert.deepEqual(itemOrder(list).map((row) => row.key), ["live/i1", "new/i1", "new/i2", "old/i1"]);
});

test("counts keep converted, failed, stopped and skipped apart; notices include warnings and kept reruns", () => {
  const rows = [
    item("j", "a"),
    item("j", "b", { status: "error", output: null, error: "x", error_code: "conversion_error" }),
    item("j", "c", { status: "error", output: null, error: "cancelled (stopped by request)", error_code: "cancelled" }),
    item("j", "d", { skipped: true, skip_reason: "image_only", output: null }),
    item("j", "e", { warnings: ["something"] }),
    item("j", "f", { rerun_failure: { operation: "enhance", error_code: "enhancement_failed", error: "no", failed_at: "2026-10-11T10:00:00Z" } }),
  ];
  assert.deepEqual(counts(rows), { total: 6, done: 3, failed: 1, stopped: 1, skipped: 1, running: 0, queued: 0, notices: 3 });
  assert.equal(hasNotices(rows[2] as SessionItem), false, "a stop is not a notice");
});

test("a submission is called what it was submitted as, never by its job id", () => {
  const words = { files: (n: number) => `${n} files`, pages: (n: number) => `${n} web pages`, filesAndPages: (f: number, p: number) => `${f} files and ${p} web pages` };
  const files = [item("j", "a"), item("j", "b")];
  const urls = [item("j", "u", { kind: "url", name: "https://example.com/a" })];
  assert.equal(submissionLabel(job("j"), files, words), "2 files");
  assert.equal(submissionLabel(job("j"), urls, words), "1 web pages");
  assert.equal(submissionLabel(job("j"), [...files, ...urls], words), "2 files and 1 web pages");
  assert.equal(submissionLabel(job("j", { label: " sample-folder " }), files, words), "sample-folder");
});

test("an unknown cost is unknown, never $0, and no model reads as no model", () => {
  assert.deepEqual(costState(0, { priced_requests: 0, unpriced_requests: 3, cost_status: "unknown" }, true), { kind: "unknown", usd: 0 });
  assert.deepEqual(costState(null, null, true), { kind: "unknown", usd: null });
  assert.deepEqual(costState(0, null, false), { kind: "none" });
  assert.deepEqual(costState(0.25, null, false), { kind: "unknown", usd: 0.25 });
  assert.deepEqual(costState(0.5, { priced_requests: 2, unpriced_requests: 0, cost_status: "complete" }, true), { kind: "complete", usd: 0.5 });
  assert.deepEqual(costState(0.5, { priced_requests: 1, unpriced_requests: 2, cost_status: "partial" }, true), { kind: "partial", usd: 0.5, unpriced: 2 });
  // Python 1.2 sends no pricing: an enhanced item without it is unknown.
  assert.equal(itemCost(item("j", "a", { llm_enhanced: true, cost_usd: 0 })).kind, "unknown");
  assert.equal(itemCost(item("j", "b", { cost_usd: 0 })).kind, "none");
  const total = totalCost([
    item("j", "a", { llm_enhanced: true, cost_usd: 0.5, pricing: { priced_requests: 1, unpriced_requests: 0, cost_status: "complete" } }),
    item("j", "b", { llm_enhanced: true, cost_usd: 0, pricing: { priced_requests: 0, unpriced_requests: 1, cost_status: "unknown" } }),
    item("j", "c"),
  ]);
  assert.deepEqual(total, { kind: "unknown", usd: 0.5, unknownItems: 1 });
  assert.deepEqual([fmtUsd(0.0123), fmtUsd(0.00004), fmtUsd(1.5)], ["$0.0123", "$0.00004", "$1.5"]);
});

test("warnings group by kind and template, counted, with their names", () => {
  const entries = [
    ...Array.from({ length: 31 }, (_, n) => ({ itemKey: "j/a", itemName: "gen.docx", text: `Image analysis failed for img-${n}.png: timeout; existing alt text was retained and no analysis entry was produced.` })),
    { itemKey: "j/b", itemName: "b.pdf", text: "Cost is incomplete: 3 requests to fake-model have no reviewed price; 1 request to fake-vision has no reviewed price. cost_usd is the known priced subtotal; the complete cost is unknown." },
    { itemKey: "j/c", itemName: "c.pdf", text: "Cost is incomplete: 1 request to fake-model has no reviewed price. cost_usd is the known priced subtotal; the complete cost is unknown." },
    { itemKey: "j/d", itemName: "d.html", text: "Remote image https://a.example/x.png was not downloaded (HTTP 404)" },
    { itemKey: "j/e", itemName: "e.html", text: "Remote image https://b.example/y.jpg was not downloaded (HTTP 403)" },
  ];
  const groups = groupNotices(entries);
  assert.deepEqual(groups.map((group) => [group.kind, group.count, group.items.length]), [["alt_kept", 31, 1], ["cost_incomplete", 2, 2], ["other", 2, 2]]);
  assert.deepEqual(groups[1]?.models, ["fake-model", "fake-vision"]);
  assert.equal(groups[0]?.samples.length, 31);
  assert.equal(template("File 'a b.png' has 12 pages"), "File <q> has <n> pages");
});

test("a document no model answered reads as one notice beside its failure", () => {
  const groups = groupNotices([
    { itemKey: "j/a", itemName: "a.epub", text: "LLM enhancement failed: LLM request failed: connection refused (deployment openai/x at 127.0.0.1:9)" },
    { itemKey: "j/a", itemName: "a.epub", text: "No model succeeded for this document; only the base Markdown was kept." },
    { itemKey: "j/b", itemName: "b.pptx", text: "No model succeeded for this document; only the base Markdown was kept." },
  ]);
  assert.deepEqual(groups.map((group) => [group.kind, group.count, group.items.length]), [["no_model", 2, 2], ["other", 1, 1]]);
  assert.equal(en.noticeTitle("no_model", 1, [], ""), "No model answered, so only the base Markdown was kept");
});

test("long tables are cut before parsing, with a marker and every row kept for Show all", () => {
  const rows = Array.from({ length: 1200 }, (_, n) => `| row ${n} | ${n} |`);
  const markdown = ["# Data", "", "| Name | Value |", "| --- | ---: |", ...rows, "", "after", "```", "| not | a |", "| --- | --- |", "```"].join("\n");
  const cut = capTables(markdown, 500);
  assert.equal(cut.tables.length, 1);
  assert.equal(cut.tables[0]?.total, 1200);
  assert.ok(cut.markdown.includes(`${MARKER}0`));
  assert.ok(cut.markdown.includes("| row 499 |") && !cut.markdown.includes("| row 500 |"));
  assert.ok(cut.markdown.endsWith("```\n| not | a |\n| --- | --- |\n```"));
  assert.equal(tableSlice(cut.tables[0]!, 500, 502), "| Name | Value |\n| --- | ---: |\n| row 500 | 500 |\n| row 501 | 501 |");
  assert.equal(capTables("| a |\n| - |\n| 1 |").tables.length, 0);
  for (const value of ["12", "-3.5", "1,234,567", "45%", "$12.50", "7k"]) assert.ok(NUMERIC.test(value), value);
  for (const value of ["v1.2.3x", "row 1", "", "abc"]) assert.ok(!NUMERIC.test(value), value);
});

test("front matter reads title, description, tags and plain fields", () => {
  const block = ["---", "title: Markitai Fixture Deck", 'description: "A synthetic deck: slide 2 holds a table."', "tags:", "  - markdown", "  - sample", "source: sample.pptx", "keywords: [a, b]", "summary: >", "  folded", "  text", "---"].join("\n");
  const front = parseFrontMatter(block);
  assert.equal(front?.title, "Markitai Fixture Deck");
  assert.equal(front?.description, "folded text");
  assert.deepEqual(front?.tags, ["markdown", "sample", "a", "b"]);
  assert.deepEqual(front?.fields, [["source", "sample.pptx"]]);
  assert.equal(parseFrontMatter(null), null);
});

test("Compare locates headings and tables in reading order on the pages' text lines", () => {
  const pages: PageLines[] = [
    {
      page: 1,
      lines: [
        { text: "Language comparison", bbox: [50, 40, 300, 60] },
        { text: "A table in the browser's default style.", bbox: [50, 70, 400, 82] },
        { text: "Language", bbox: [50, 100, 110, 112] },
        { text: "Year", bbox: [120, 100, 150, 112] },
        { text: "Typing", bbox: [160, 100, 200, 112] },
        { text: "Python", bbox: [50, 116, 100, 128] },
        { text: "TypeScript", bbox: [50, 132, 120, 144] },
        { text: "2012", bbox: [120, 132, 150, 144] },
        { text: "Each language has its trade-offs.", bbox: [50, 160, 300, 172] },
      ],
    },
    { page: 2, lines: [{ text: "2. Language comparison", bbox: [50, 40, 300, 60] }] },
  ];
  const found = locate(pages, [
    { id: "h1", kind: "heading", text: ["Language comparison"] },
    { id: "t1", kind: "table", text: ["Language", "Year", "Typing"], last: ["TypeScript", "2012", "Static"] },
    { id: "h2", kind: "heading", text: ["Language comparison"] },
    { id: "missing", kind: "heading", text: ["Nowhere to be found"] },
  ]);
  assert.deepEqual(found.get("h1"), { page: 1, box: [50, 40, 300, 60] });
  assert.deepEqual(found.get("t1"), { page: 1, box: [50, 100, 200, 144] });
  assert.equal(found.get("h2")?.page, 2, "the second occurrence finds its own page");
  assert.equal(found.has("missing"), false);
  assert.equal(norm("Ｌａｎｇｕａｇｅ  Comparison!"), "languagecomparison");
});

test("code is highlighted by language family without reparsing it as HTML", () => {
  const code = (className: string, text: string) => ({ className, textContent: text }) as unknown as Element;
  assert.equal(family(code("language-ts", "let a = 1")), "clike");
  assert.equal(family(code("language-plaintext", "x")), null);
  assert.equal(family(code("", '{"a": 1}')), "json");
  assert.equal(family(code("", "def f():\n  pass")), "python");
  const runs = tokenize('const x = "<b>" // note', LANGS.clike ?? []);
  assert.deepEqual(runs.filter(([cls]) => cls).map(([cls, text]) => [cls, text]), [["k", "const"], ["s", '"<b>"'], ["c", "// note"]]);
  assert.equal(runs.map(([, text]) => text).join(""), 'const x = "<b>" // note');
  // A keyword inside an identifier stays plain.
  assert.deepEqual(tokenize("ifx", LANGS.clike ?? []), [["", "ifx"]]);
});

test("views and the list selection are real addresses", () => {
  assert.deepEqual(parseRoute("/jobs/"), { view: "jobs" });
  assert.deepEqual(parseRoute("/settings/appearance"), { view: "settings", section: "appearance" });
  assert.deepEqual(parseRoute("/settings"), { view: "settings", section: null });
  assert.deepEqual(parseRoute("/"), { view: "convert" });
  assert.deepEqual(parseRoute("/index.html"), { view: "convert" });
  // Any other address is no page: the workbench says so instead of showing Convert.
  assert.deepEqual(parseRoute("/anything"), { view: "missing", path: "/anything" });
  assert.deepEqual(parseRoute("/settings/nope"), { view: "missing", path: "/settings/nope" });
  assert.equal(routePath({ view: "missing", path: "/anything" }), "/anything");
  assert.equal(routePath({ view: "settings", section: "access" }), "/settings/access");
  assert.deepEqual(parseListState("?sel=0123456789ab/i3&view=compare"), { selected: "0123456789ab/i3", mode: "compare" });
  // A conversion opens on its Preview; Details has its own address.
  assert.deepEqual(parseListState("?sel=job:0123456789ab"), { selected: "job:0123456789ab", mode: "preview" });
  assert.deepEqual(parseListState("?sel=0123456789ab/i3&view=details"), { selected: "0123456789ab/i3", mode: "inspector" });
  assert.deepEqual(parseListState("?sel=../../x&view=other"), { selected: null, mode: "preview" });
  assert.equal(listQuery({ selected: "a/i1", mode: "preview" }), "?sel=a%2Fi1");
  assert.equal(listQuery({ selected: "a/i1", mode: "inspector" }), "?sel=a%2Fi1&view=details");
  assert.equal(listQuery({ selected: null, mode: "preview" }), "");
});

test("a phone window keeps Stop as a button while the other secondary actions fold into the menu", () => {
  // The Python e2e bug: right after Convert the full-screen summary covered the
  // list's Stop, and its own Stop was folded into "⋯".
  const shown = [
    { key: "retry-failed", primary: true },
    { key: "zip" },
    { key: "stop", keep: true },
  ];
  assert.deepEqual(
    headerActs(shown, true).inline.map((act) => act.key),
    ["retry-failed", "stop"],
  );
  assert.deepEqual(
    headerActs(shown, true).folded.map((act) => act.key),
    ["zip"],
  );
  assert.deepEqual(headerActs(shown, false), { inline: shown, folded: [] });
});

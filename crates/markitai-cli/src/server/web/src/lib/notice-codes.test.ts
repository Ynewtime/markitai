import assert from "node:assert/strict";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { groupNotices, noticeCodes, noticeEntries, rowEntries } from "./notices.ts";

const layout = "PDF layout refinement was skipped (timeout); the original page reader's output is retained.";
const merged = "Merged table cells are represented by their origin cell with empty covered cells in Markdown.";
const unknown = "Something new happened in a later release.";
const noModel = "No model succeeded for this document; the base Markdown was kept.";
const retried = "A model answer could not be used (invalid JSON); a later answer was used.";

const item = (warnings: string[], codes: string[] | null) => ({
  key: "i1",
  name: "notes.pdf",
  warnings,
  noticeCodes: codes ? noticeCodes(codes.map((code) => ({ code })), warnings) : [],
});

test("a warning no wording pattern knows is titled by its code in every language", () => {
  const groups = groupNotices(noticeEntries(item([layout, merged, layout], ["pdf.layout", "table.merged_cells", "pdf.layout"])));
  assert.deepEqual(groups.map((group) => [group.kind, group.count]), [["code:pdf.layout", 2], ["code:table.merged_cells", 1]]);
  const [first] = groups;
  assert.equal(en.noticeTitle(first!.kind, first!.count, first!.models, first!.sample, first!.pages), "PDF layout refinement was skipped; the page reader's output is kept");
  assert.match(zh.noticeTitle(first!.kind, first!.count, first!.models, first!.sample, first!.pages), /PDF 版面优化/);
  assert.match(ja.noticeTitle(first!.kind, first!.count, first!.models, first!.sample, first!.pages), /レイアウト/);
  // The server's own words stay behind Details.
  assert.deepEqual(first!.samples, [layout]);
});

test("without codes, or with an unknown code, the server's wording is shown", () => {
  for (const codes of [null, ["other"], ["brand.new_code"]]) {
    const [group] = groupNotices(noticeEntries(item([unknown], codes)));
    assert.equal(group!.kind, "other");
    assert.equal(zh.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), unknown);
  }
});

test("a code reaches a known kind when its wording changed, and the wording pattern still comes first", () => {
  const [changed] = groupNotices(noticeEntries(item(["Every model failed for this file."], ["llm.no_model_succeeded"])));
  assert.equal(changed!.kind, "no_model");
  const [known] = groupNotices(noticeEntries(item([noModel], ["other"])));
  assert.equal(known!.kind, "no_model");
  assert.deepEqual(groupNotices(noticeEntries(item(["Retried with another answer."], ["llm.answer_retried"]))), []);
  assert.deepEqual(groupNotices(noticeEntries(item([retried], null))), []);
});

test("codes are kept only when they pair one to one with the warnings", () => {
  assert.deepEqual(noticeCodes([{ code: "pdf.layout" }], [layout, merged]), []);
  assert.deepEqual(noticeCodes(undefined, [layout]), []);
  assert.deepEqual(noticeCodes([{ code: "office.page", items: ["3", 4 as unknown as string] }], ["Office page 3: x"]), [{ code: "office.page", items: ["3"] }]);
  // A row keeps the code of each warning it marks.
  const cost = "Cost is incomplete: 1 request to some-model has no reviewed price.";
  assert.deepEqual(rowEntries(item([cost, layout], ["llm.cost_incomplete", "pdf.layout"])).map((entry) => entry.code), ["pdf.layout"]);
});

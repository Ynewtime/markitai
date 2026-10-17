import assert from "node:assert/strict";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { zh } from "../i18n/zh.ts";
import { groupNotices, rowWarnings } from "./notices.ts";

const placed = "Embedded PDF images are placed after their page's text; their exact position is not reconstructed. Add --screenshot to keep each page's appearance.";
const scanned = "PDF is scanned: 3 of 3 pages have no readable text (pages 1–3); run again with --ocr to read them.";

test("the workbench says Page Screenshots, not the CLI flag", () => {
  const [text] = rowWarnings([placed]);
  assert.ok(text && !text.includes("--screenshot") && text.includes("Turn on Page Screenshots"));
});

test("a document told it is scanned is not also told where its images go", () => {
  assert.deepEqual(rowWarnings([placed, scanned]), [scanned]);
  assert.equal(rowWarnings([placed]).length, 1);
  const groups = groupNotices([
    { itemKey: "a", itemName: "a.pdf", text: placed },
    { itemKey: "a", itemName: "a.pdf", text: scanned },
    { itemKey: "b", itemName: "b.pdf", text: placed },
  ]);
  const kinds = groups.map((group) => [group.kind, group.items.join()]);
  assert.deepEqual(kinds.sort(), [["pdf_images_placed", "b"], ["pdf_scanned", "a"]]);
});

test("the common office and PDF notices have a title in every language", () => {
  const texts = [placed, "The picture bank's declared slot count differs from its complete records; referenced figures were recovered by their actual record ordinals.", "Word picture metadata had an inconsistent count; images were recovered from complete records.", 'Hidden worksheet "Hidden notes" is omitted by the native spreadsheet reader.'];
  for (const text of texts) {
    const [group] = groupNotices([{ itemKey: "a", itemName: "a", text }]);
    assert.ok(group && group.kind !== "other", text);
    const zhTitle = zh.noticeTitle(group.kind, group.count, group.models, group.sample, group.pages);
    assert.ok(/[一-鿿]/.test(zhTitle) && !/--screenshot/.test(zhTitle), zhTitle);
    assert.notEqual(en.noticeTitle(group.kind, group.count, group.models, group.sample, group.pages), group.sample);
  }
});

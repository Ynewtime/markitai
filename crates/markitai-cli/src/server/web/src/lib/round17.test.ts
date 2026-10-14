import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { marked } from "../../../../../../../vendor/web/marked.js";
import { fetchConfigDocument, modelsApi, saveConfigDocument } from "../api/client.ts";
import { setToken } from "../api/token.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { buildTime, versionLine } from "./build.ts";
import { remoteVideo } from "./markdown.ts";
import { parseWeight } from "./models.ts";
import { compressPageLists, groupNotices, pageRanges, parsePages, pdfPageNotice } from "./notices.ts";
import { repairTables, rowCells } from "./tables.ts";

const html = (markdown: string) => marked.parse(markdown, { async: false, gfm: true }) as string;
const rows = (markup: string) => (markup.match(/<tr>/g) ?? []).length;

test("a shared model's weight and a user's own models document use their own routes", async () => {
  setToken("t-user", null);
  const original = globalThis.fetch;
  const sent: { url: string; method: string; body: unknown }[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: new URL(String(url)).pathname, method: init?.method ?? "GET", body: init?.body ? JSON.parse(String(init.body)) : null });
    return new Response(JSON.stringify({ revision: "r2", shared: [], deployments: [] }), { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    const api = modelsApi("user");
    await api.updateShared("s 1", "r1", 0);
    await api.updateShared("s1", "r2", null);
    await api.probeModel({ shared_id: "s1" });
    await fetchConfigDocument("user");
    await saveConfigDocument("r2", "{}\n", "user");
    await fetchConfigDocument();
    assert.deepEqual(
      sent.map((request) => `${request.method} ${request.url}`),
      ["PATCH /api/me/models/shared/s%201", "PATCH /api/me/models/shared/s1", "POST /api/me/models/test", "GET /api/me/config", "PUT /api/me/config", "GET /api/settings/config"],
    );
    assert.deepEqual(sent[0]?.body, { expected_revision: "r1", weight: 0 });
    assert.deepEqual(sent[1]?.body, { expected_revision: "r2", weight: null });
    assert.deepEqual(sent[2]?.body, { shared_id: "s1" });
    assert.deepEqual(sent[4]?.body, { expected_revision: "r2", text: "{}\n" });
  } finally {
    globalThis.fetch = original;
    setToken("", null);
  }
});

test("a weight typed in place is a whole number from 0", () => {
  assert.equal(parseWeight("0"), 0);
  assert.equal(parseWeight(" 12 "), 12);
  assert.equal(parseWeight("999999"), 999999);
  for (const bad of ["", "-1", "1.5", "1e3", "abc", "1000000", "２"]) assert.equal(parseWeight(bad), null, bad);
});

test("the version line is the version, the commit and the build time, without the product name", () => {
  const now = new Date(2026, 9, 14, 23, 0);
  const built = new Date(2026, 9, 14, 22, 48, 5).toISOString();
  assert.equal(buildTime(built, now), "10-14 22:48");
  assert.equal(buildTime(new Date(2025, 0, 2, 3, 4).toISOString(), now), "2025-01-02");
  assert.equal(buildTime("not a time", now), null);
  const line = versionLine({ version: "1.2.1", build: { version: "1.2.1", commit: "36583fe", built_at: built, dirty: false } }, now);
  assert.equal(line?.text, "1.2.1 36583fe 10-14 22:48");
  assert.equal(line?.short, "1.2.1 36583fe");
  assert.ok(!/markitai/i.test(line?.text ?? ""));
  const dirty = versionLine({ version: "1.3.0-dev", build: { version: "1.3.0-dev", commit: "abc1234", built_at: built, dirty: true } }, now);
  assert.equal(dirty?.text, "1.3.0-dev abc1234-dirty 10-14 22:48");
  // An older backend: the version alone.
  assert.equal(versionLine({ version: "1.2.1" }, now)?.text, "1.2.1");
  assert.equal(versionLine({ version: "1.2.1", build: { version: "1.2.1", commit: null, built_at: null } }, now)?.text, "1.2.1");
  for (const dict of [en, zh, ja]) {
    const title = dict.buildTitle({ version: "1.3.0-dev", commit: "abc1234", dirty: true, built: "2026-10-14 22:48:05 UTC+08:00" });
    assert.match(title, /1\.3\.0-dev/);
    assert.match(title, /abc1234/);
    assert.match(title, /2026-10-14 22:48:05/);
  }
});

test("page lists read as ranges", () => {
  assert.equal(pageRanges([3, 1, 2, 5, 7, 8, 9, 9]), "1–3, 5, 7–9");
  assert.equal(pageRanges([4, 5]), "4, 5");
  assert.equal(pageRanges([]), "");
  assert.deepEqual(parsePages("1–3, 5, 8-9"), [1, 2, 3, 5, 8, 9]);
  const all = Array.from({ length: 491 }, (_, index) => index + 1).join(", ");
  assert.equal(compressPageLists(`491 pages look scanned (pages ${all}); consider re-running`), "491 pages look scanned (pages 1–491); consider re-running");
  assert.equal(compressPageLists("pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 21, 22"), "pages 1–12, 15, 20–22");
  assert.equal(compressPageLists("page 4 and 7 items"), "page 4 and 7 items");
});

const item = (text: string, key = "j/a", name = "atlas.pdf") => ({ itemKey: key, itemName: name, text });

test("a scanned PDF is said plainly, with the one action, from either backend's wording", () => {
  const list = Array.from({ length: 491 }, (_, index) => index + 1).join(", ");
  // Python 1.2.1 today (no total), Python with the v1.16 total, Rust's v1.16 wording.
  const python = `[PDF] atlas.pdf: 491 pages look scanned/garbled (pages ${list}); consider re-running with --ocr`;
  assert.deepEqual(pdfPageNotice(python)?.pages.length, 491);
  assert.equal(pdfPageNotice(python)?.total, null);
  const pythonTotal = "[PDF] atlas.pdf: 491 of 491 pages look scanned/garbled (pages 1–491); consider re-running with --ocr";
  const rust = "PDF is scanned: 488 of 491 pages have no readable text (pages 1–480, 483–490); run again with --ocr to read them.";
  for (const text of [pythonTotal, rust]) {
    const [group] = groupNotices([item(text)]);
    assert.equal(group?.kind, "pdf_scanned", text);
    assert.equal(en.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), "This PDF is scanned: turn on OCR and convert again");
    assert.match(zh.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), /扫描件.*OCR/);
    assert.match(ja.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), /スキャン.*OCR/);
  }
  // Without the document's page count, an unbroken run from page 1 reads as a scan too (Python 1.2.1's wording)…
  const [old] = groupNotices([item(python)]);
  assert.equal(old?.kind, "pdf_scanned");
  assert.ok(old!.samples.every((sample) => !sample.includes("1, 2, 3") && sample.includes("pages 1–491")));
  // …but a short run, or one with gaps, names its pages, as ranges, with the same action.
  const [gaps] = groupNotices([item("[PDF] atlas.pdf: 4 pages look scanned/garbled (pages 1, 2, 3, 7); consider re-running with --ocr")]);
  assert.equal(gaps?.kind, "pdf_ocr_pages");
  assert.equal(en.noticeTitle(gaps!.kind, gaps!.count, gaps!.models, gaps!.sample, gaps!.pages), "Pages 1–3, 7 have no readable text: turn on OCR and convert again to read them");
});

test("some pages without text are named as ranges, also from one warning per page", () => {
  const rust = "PDF pages 3–5, 9 (4 of 120): native text was not recovered (scanned); OCR is required for these pages.";
  const [group] = groupNotices([item(rust)]);
  assert.equal(group?.kind, "pdf_ocr_pages");
  assert.deepEqual(group?.pages, { ranges: "3–5, 9", count: 4, total: 120 });
  assert.equal(en.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), "Pages 3–5, 9 (4 of 120) have no readable text: turn on OCR and convert again to read them");
  assert.match(zh.noticeTitle(group!.kind, group!.count, group!.models, group!.sample, group!.pages), /^第 3–5, 9 页（共 120 页中的 4 页）没有可读文字/);
  // Older Rust: a line per page.
  const older = [1, 2, 3, 7].map((page) => item(`PDF page ${page}: native text was not recovered (scanned); OCR is required for this page.`));
  const [joined] = groupNotices(older);
  assert.equal(joined?.kind, "pdf_ocr_pages");
  assert.equal(joined?.pages?.ranges, "1–3, 7");
  assert.deepEqual(joined?.samples, ["PDF pages 1–3, 7: native text was not recovered (scanned); OCR is required for these pages."]);
  // Two documents: counted, not merged.
  const two = groupNotices([item(rust), item("PDF page 2 (1 of 9): native text was not recovered (garbled); OCR is required for this page.", "j/b", "b.pdf")]);
  assert.equal(two[0]?.count, 2);
  assert.equal(two[0]?.pages, null);
  assert.equal(en.noticeTitle(two[0]!.kind, 2, [], "", null), "2 PDFs have pages without readable text: turn on OCR and convert them again");
  // Any other warning repeated page by page reads as one, with its pages.
  const charts = groupNotices([4, 5, 6].map((page) => item(`PDF page ${page}: chart placement could not be verified; original text is retained.`)));
  assert.equal(charts.length, 1);
  assert.equal(charts[0]?.sample, "PDF pages 4–6: chart placement could not be verified; original text is retained.");
  assert.deepEqual(charts[0]?.pages, { ranges: "4–6", count: 3, total: null });
});

test("a model answer a guard refused is explained in plain words", () => {
  for (const text of [
    "atlas.pdf: LLM enhancement failed (LLM answer text was discarded by the page drift guard); kept the unenhanced output",
    "LLM enhancement failed; base Markdown retained: LLM answer discarded or replaced too much source content to be a document cleanup",
  ]) {
    const [group] = groupNotices([item(text)]);
    assert.equal(group?.kind, "answer_rejected", text);
    const said = en.noticeTitle(group!.kind, 1, [], group!.sample, null);
    // Python's guard compares page by page; Rust's checks how much of the document is left.
    assert.match(said, text.includes("drift guard") ? /did not match the document page by page/ : /left out or changed too much/);
    assert.match(said, /not used/);
    assert.match(said, /kept as it was/);
    assert.ok(!/drift|guard/.test(said));
    assert.ok(!/drift|guard/.test(zh.noticeTitle(group!.kind, 1, [], group!.sample, null)));
    assert.ok(!/drift|guard/.test(ja.noticeTitle(group!.kind, 1, [], group!.sample, null)));
  }
});

test("only an https video file plays in place", () => {
  assert.equal(remoteVideo("https://video.example/amplify/1280x720/clip.mp4?tag=1"), "https://video.example/amplify/1280x720/clip.mp4?tag=1");
  assert.equal(remoteVideo("https://media.example/a.WEBM"), "https://media.example/a.WEBM");
  for (const no of ["http://video.example/clip.mp4", "https://user:pw@video.example/clip.mp4", "https://x.example/status/1/video/1", "movie.mp4", "data:video/mp4;base64,AAAA", "https://video.example/clip.mp4.html"])
    assert.equal(remoteVideo(no), null, no);
  const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
  assert.match(css, /\.doc-body \.doc-video \{/);
});

test("a table that a line break in a cell broke renders as one table", () => {
  const head = "Plan\n\n| Task | Time | Note |\n| --- | --- | --- |\n| Plant bulbs | 08:50 | Bring gloves |\n";
  const tail = "| Weed paths | 09:40 | Kneeling pads |\n| Rake leaves | 10:00 | |\n";
  for (const breakText of [" ", " ", "\u000b", "\n", "\n\n"]) {
    const markdown = `${head}| Water beds | 09:10 | Bring the hose${breakText}and a bucket |\n${tail}`;
    const repaired = repairTables(markdown);
    const markup = html(repaired);
    assert.equal(rows(markup), 5, JSON.stringify(breakText));
    assert.ok(!/<p>\|/.test(markup), JSON.stringify(breakText));
    assert.match(markup, /Bring the hose<br>and a bucket/, JSON.stringify(breakText));
  }
  // The parser alone loses the rest of the table at a line separator (the owner's report).
  assert.equal(rows(html(`${head}| Water beds | 09:10 | Bring the hose and a bucket |\n${tail}`)), 2);
  // A cell broken over several lines, list-like ones too, and a well-formed table are left as they read.
  assert.match(html(repairTables("| Task | Note |\n| --- | --- |\n| Water | Steps:\n1. fill\n2. pour |\n| Weed | Pads |\n")), /Steps:<br>1\. fill<br>2\. pour/);
  const fine = "| a | b |\n| - | - |\n| x | y\n\nParagraph after.\n";
  assert.equal(repairTables(fine), fine);
  const fenced = "```\n| a | b |\n| - | - |\n| x y |\n```\n";
  assert.equal(repairTables(fenced), fenced);
  assert.equal(rowCells("| a | b \\| c | `x|y` |"), 3);
  assert.equal(rowCells("a | b"), 2);
});

test("the new words exist in every language", () => {
  for (const dict of [en, zh, ja]) {
    assert.ok(dict.weightEditAria(dict.modelWeight(2), "gpt-4o-mini").startsWith(dict.modelWeight(2)));
    assert.ok(dict.sharedWeightResetAria(dict.sharedWeightReset(2), "m").startsWith(dict.sharedWeightReset(2)));
    assert.ok(dict.sharedWeightReset(3).includes("3"));
    assert.ok(dict.weightSaved("m", 0) !== dict.weightSaved("m", 2));
    assert.ok(dict.userConfigMasked(2).includes("2"));
    assert.ok(dict.sharedTestFailHint("rate_limited") !== dict.sharedTestFailHint("key_rejected"));
    for (const key of ["sharedByAdminWeights", "sharedOffForYou", "userConfigSubtitle", "userConfigNote", "userConfigSaved", "userConfigStale", "userConfigHint"] as const) assert.ok(dict[key].length > 4, key);
  }
});

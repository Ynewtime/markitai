// Document actions, option A (owner, 2026-10-26): one list everywhere in four groups, Download
// (Markdown .md · PDF… · JSON .json · All files .zip · View file list), Copy Markdown, Re-convert…,
// Delete; one zip wording; Re-convert… starts from the document's own options.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { BUILTIN_PRESETS, composerFromOptions, emptyOptions, resolveOptions } from "./options.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");
const inspector = read("../components/inspector.tsx");
const peek = read("../components/peek.tsx");
const page = read("../components/conversions-page.tsx");
const menu = read("../components/action-menu.tsx");

const order = (source: string, marks: string[]) => {
  const at = marks.map((mark) => source.indexOf(mark));
  assert.ok(at.every((index) => index >= 0), `missing one of ${marks.join(", ")}`);
  assert.deepEqual([...at].sort((a, b) => a - b), at, `out of order: ${marks.join(", ")}`);
};

test("Download lists Markdown, PDF…, JSON, the zip and the file list, in that order", () => {
  order(inspector, ['key: "dl-md"', 'key: "dl-json"', 'key: "dl-zip"']);
  // PDF… goes in after the first entry (Markdown), the file list last
  order(peek, ["...(first ? [first] : [])", 'key: "dl-pdf"', "...rest,", 'key: "dl-files"']);
  // the split caret and the phone's "⋯" share one list
  assert.match(peek, /entries=\{formats\(act, "split"\)\}/);
  assert.match(peek, /formats\(act, "menu"\)/);
});

test("\"⋯\" holds Copy Markdown, the page, Re-convert… and Delete, in groups; JSON, Enhance and Run again are gone", () => {
  order(peek, ['group: "download"', 'key: "copy"', 'key: "page"', 'group: "again"']);
  for (const gone of ['key: "json"', 'key: "enhance"', 'key: "again"', "t.runAgain", "t.enhance", "downloadAs"]) {
    assert.ok(!inspector.includes(gone) && !peek.includes(gone), gone);
  }
  assert.match(inspector, /more\.push\(\{ key: "reconvert"/);
  // Delete is its own group, after every other
  assert.match(menu, /\{danger &&\s*\(grouped \?/);
});

test("the list header downloads first, and the selection bar ends with Delete", () => {
  order(page, ['key: "zip", label: t.downloadAllZip', 'key: "retry", label: t.retryFailedN']);
  order(page, ['class="btn select-reconvert"', 'class="btn btn-danger select-delete"']);
});

test("one zip wording, ellipses only before a dialog, short labels in every language", () => {
  assert.equal(en.downloadAllZip, "Download all .zip");
  assert.equal(zh.downloadAllZip, "下载全部 .zip");
  assert.equal(ja.downloadAllZip, "すべてダウンロード .zip");
  assert.match(inspector, /label: t\.downloadAllZip, icon: "DownloadSimple", run: \(\) => actions\.zip/);
  for (const dict of [en, zh, ja]) {
    assert.ok(dict.reconvert.endsWith("…"));
    assert.ok(dict.downloadPdf.endsWith("…"));
    assert.ok(!dict.viewFileList.endsWith("…"));
    assert.ok(dict.copyMarkdown.includes("Markdown"));
  }
  assert.equal(zh.viewFileList, "查看文件列表");
  assert.equal(en.viewFileList, "View file list");
  assert.equal(ja.viewFileList, "ファイル一覧を表示");
});

test("Re-convert… starts from the document's own options", () => {
  const app = read("../app.tsx");
  assert.match(app, /setComposer\(composerFromOptions\(first, presets\)\)/);
  const cases = [
    { ...emptyOptions(), preset: "minimal", llm: false, ocr: true, alt: false, desc: false, screenshot: false, strategy: "static" as const, profile: "rag" as const },
    { ...emptyOptions(), preset: "standard", llm: true, ocr: false, alt: true, desc: false, screenshot: true, format: "json" as const },
    { ...emptyOptions(), preset: "rich", llm: true, ocr: false, alt: true, desc: true, screenshot: true, pure: false, no_cache: true },
  ];
  for (const options of cases) {
    const state = composerFromOptions(options, BUILTIN_PRESETS);
    const again = resolveOptions(state, BUILTIN_PRESETS);
    assert.equal(state.preset, options.preset);
    for (const key of ["llm", "ocr", "alt", "desc", "screenshot", "profile", "format"] as const) assert.equal(again[key] ?? null, options[key] ?? null, key);
    assert.equal(again.strategy, options.strategy ?? "auto");
    assert.equal(again.no_cache, options.no_cache ?? false);
  }
  // A preset this composer does not know: the closest one with the same switches.
  const named = composerFromOptions({ ...emptyOptions(), preset: "house", llm: true, alt: true, desc: true, screenshot: false }, BUILTIN_PRESETS);
  assert.equal(named.preset, "standard");
  assert.equal(resolveOptions(named).llm, true);
});

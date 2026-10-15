import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fetchConvertDefaults, saveConvertDefaults } from "../api/client.ts";
import { setToken } from "../api/token.ts";
import type { ConvertDefaults } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { itemCostPhase, llmAsked, totalCostPhase } from "./cost.ts";
import { locate } from "./compare.ts";
import { frontLine, frontRows } from "./front-rows.ts";
import { cardFacts, leadsWithHeading, parseFrontMatter } from "./frontmatter.ts";
import { jsonTokens } from "./json-tokens.ts";
import { groupNotices, ocrBlankNotice } from "./notices.ts";
import { applyPreset, composerFromDefaults, defaultsOf, initialComposer, readRemembered, resolveOptions, sameAsDefaults } from "./options.ts";
import { seedItem, type SessionItem } from "./session.ts";

const item = (patch: Partial<SessionItem>): SessionItem => ({ ...seedItem("j1", { itemId: "i1", name: "notes.md", kind: "file", sizeBytes: 10 }), ...patch });

test("a conversion's cost line says what will happen until there is an amount", () => {
  const standard = { preset: "standard", llm: true } as SessionItem["options"];
  const plain = { preset: "minimal", llm: false } as SessionItem["options"];
  assert.equal(itemCostPhase(item({ status: "queued", options: standard })), "queued");
  assert.equal(itemCostPhase(item({ status: "queued", options: plain })), "no_model_asked");
  assert.equal(itemCostPhase(item({ status: "running", options: standard })), "counting");
  assert.equal(itemCostPhase(item({ status: "running", options: plain })), "no_model_asked");
  assert.equal(itemCostPhase(item({ status: "running", operation: "enhance", options: plain })), "waiting_model");
  assert.equal(itemCostPhase(item({ status: "done", options: standard, models: ["m"], costUsd: 0.01, pricing: null })), "amount");
  assert.equal(itemCostPhase(item({ status: "done", options: standard })), "no_model_answered");
  assert.equal(itemCostPhase(item({ status: "done", options: plain })), "amount");
  assert.equal(itemCostPhase(item({ status: "error", options: standard })), "failed_none");
  assert.equal(itemCostPhase(item({ status: "error", skipReason: "user_stopped", options: standard })), "stopped_none");
  // A preset alone asks for a model when it is standard or rich.
  assert.equal(llmAsked({ preset: "rich" }), true);
  assert.equal(llmAsked({ preset: "rich", llm: false }), false);
  assert.equal(totalCostPhase([item({ status: "done", options: plain }), item({ status: "running", options: standard })]), "counting");
  assert.equal(totalCostPhase([item({ status: "done", options: plain })]), "amount");
  for (const dict of [en, zh, ja]) for (const key of ["costQueued", "costCounting", "costWaitingModel", "costNoModelAsked", "costNoModelAnswered", "costFailedNone", "costStoppedNone"] as const) assert.ok(dict[key].length > 0);
  assert.equal(zh.costCounting, "计算中…");
});

test("front matter: no title of its own over a heading, tags inline, every key a row", () => {
  const front = parseFrontMatter("---\ntitle: Allotment rota\nsource: rota.xlsx\ndescription: Who does what\ntags:\n- gardening\n- volunteers\nmarkitai_processed: '2026-10-15T09:43:12+08:00'\n---");
  assert.ok(front);
  // The body's own heading is the title, whatever its level or words.
  assert.equal(cardFacts(front, "## Sheet one\n\ntext")?.title, null);
  assert.equal(leadsWithHeading("\n### Notes\n"), true);
  assert.equal(leadsWithHeading("Text\n# Later"), false);
  const shown = cardFacts(front, "# Allotment rota\n")!;
  assert.deepEqual(frontLine(shown), { kind: "tags", tags: ["gardening", "volunteers"] });
  assert.deepEqual(
    frontRows(shown).map((row) => [row.key, row.kind]),
    [["description", "text"], ["tags", "tags"], ["source", "text"], ["markitai_processed", "time"]],
  );
  // Only other keys: still shown (the folded line says the first value).
  const only = cardFacts(parseFrontMatter("---\nsource: https://example.org/a\n---"), "Body")!;
  assert.deepEqual(frontLine(only), { kind: "text", text: "https://example.org/a" });
  assert.equal(frontRows(only)[0]?.kind, "link");
  assert.equal(en.frontKey("markitai_processed"), "Processed");
  assert.equal(zh.frontKey("description"), "简介");
  assert.equal(ja.frontKey("custom_key"), "custom_key");
  assert.equal(zh.frontProperties(4), "属性 · 4");
});

test("JSON pieces cover the text exactly, keys apart from strings", () => {
  const text = '{\n  "llm": {"enabled": true, "weight": -2.5e3},\n  "x": [null, "a\\"b"], bad\n  "open';
  const pieces = jsonTokens(text);
  assert.equal(pieces.map(([, piece]) => piece).join(""), text);
  const kinds = (kind: string) => pieces.filter(([k]) => k === kind).map(([, piece]) => piece);
  assert.deepEqual(kinds("key"), ['"llm"', '"enabled"', '"weight"', '"x"']);
  assert.deepEqual(kinds("string"), ['"a\\"b"', '"open']);
  assert.deepEqual(kinds("number"), ["-2.5e3"]);
  assert.deepEqual(kinds("literal"), ["true", "null"]);
  assert.equal(jsonTokens("").length, 0);
});

const server: ConvertDefaults = { llm: false, ocr: true, alt: true, desc: false, screenshot: false, screenshot_only: false, pure: false, no_cache: false, no_compress: false, profile: null, strategy: "auto" };

test("the composer starts from the server's defaults and saves its choices back", () => {
  const composer = composerFromDefaults(server);
  assert.equal(composer.preset, "minimal");
  assert.equal(composer.ocr, true);
  assert.equal(sameAsDefaults(composer, server), true);
  // The picture switches wait for a model: on with LLM, the server's alt text applies.
  assert.equal(resolveOptions({ ...composer, llm: true }).alt, true);
  // A preset that agrees on the model wins.
  const enhanced = composerFromDefaults({ ...server, llm: true, alt: false, desc: false });
  assert.equal(enhanced.preset, "standard");
  assert.equal(resolveOptions(enhanced).alt, false);
  const changed = applyPreset(composer, "rich");
  assert.equal(sameAsDefaults(changed, server), false);
  const saved = defaultsOf(changed)!;
  assert.deepEqual([saved.llm, saved.screenshot, saved.ocr], [true, true, false]);
  assert.equal(sameAsDefaults(changed, saved), true);
  assert.equal(sameAsDefaults(composerFromDefaults(saved), saved), true);
  // Cloudflare asks each time: never saved.
  assert.equal(defaultsOf({ ...composer, advanced: { ...composer.advanced, strategy: "cloudflare" } }), null);
  assert.equal(defaultsOf({ ...composer, advanced: { ...composer.advanced, backend: "cloudflare" } }), null);
  // Older backends: the last choice in this browser still applies.
  assert.equal(initialComposer(readRemembered(null)).preset, "minimal");
});

test("defaults are read and saved with the revision", async () => {
  setToken("t-admin", null);
  const original = globalThis.fetch;
  const sent: { url: string; method: string; body: unknown }[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: new URL(String(url)).pathname, method: init?.method ?? "GET", body: init?.body ? JSON.parse(String(init.body)) : null });
    return new Response(JSON.stringify({ defaults: server, server, own: null, origin: "server", scope: "server", can_save: true, revision: "sha256:2" }), { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    await fetchConvertDefaults();
    await saveConvertDefaults("sha256:1", server);
    await saveConvertDefaults("sha256:2", null);
  } finally {
    globalThis.fetch = original;
  }
  assert.deepEqual(sent.map((request) => [request.url, request.method]), [["/api/convert-defaults", "GET"], ["/api/convert-defaults", "PUT"], ["/api/convert-defaults", "PUT"]]);
  assert.deepEqual(sent[1]?.body, { expected_revision: "sha256:1", defaults: server });
  assert.deepEqual(sent[2]?.body, { expected_revision: "sha256:2", defaults: null });
});

test("pages OCR found no text on read as one notice with ranges, in either backend's words", () => {
  assert.deepEqual(ocrBlankNotice("[OCR] No text found on 2 pages of book.pdf (pages 166, 360); they are blank in the output"), { pages: [166, 360], total: null, office: false });
  assert.deepEqual(ocrBlankNotice("[OCR] No text found on 1 page of a.pdf (page 7); it is blank in the output"), { pages: [7], total: null, office: false });
  assert.deepEqual(ocrBlankNotice("[OCR] book.pdf: 2 of 491 pages have no text (pages 166, 360); they are left blank in the output"), { pages: [166, 360], total: 491, office: false });
  assert.deepEqual(ocrBlankNotice("[OCR] a: b.pdf: 1 of 9 pages has no text (page 4); it is left blank in the output"), { pages: [4], total: 9, office: false });
  assert.deepEqual(ocrBlankNotice("PDF pages 2, 245–246 (3 of 490): OCR found no text; they are blank in the output."), { pages: [2, 245, 246], total: 490, office: false });
  assert.deepEqual(ocrBlankNotice("PDF page 1 (1 of 1): OCR found no text; it is blank in the output."), { pages: [1], total: 1, office: false });
  assert.deepEqual(ocrBlankNotice("Office pages 2, 5 (2 of 9): OCR found no text; the OCR supplement leaves them out."), { pages: [2, 5], total: 9, office: true });
  assert.equal(ocrBlankNotice("PDF pages 3–5 (3 of 120): native text was not recovered (scanned); OCR is required for these pages."), null);
  const rust = "PDF pages 3–5, 9 (4 of 120): OCR found no text; they are blank in the output.";
  const [group] = groupNotices([{ itemKey: "a", itemName: "a.pdf", text: rust }]);
  assert.equal(group?.kind, "ocr_blank");
  assert.equal(en.noticeTitle("ocr_blank", 1, [], rust, group?.pages ?? null), "OCR found no text on pages 3–5, 9 (4 of 120): they are blank in the output");
  assert.equal(en.noticeTitle("ocr_blank", 1, [], "Office pages 2 (1 of 9): OCR found no text; the OCR supplement leaves them out.", { ranges: "2", count: 1, total: 9 }), "OCR found no text on page 2 (1 of 9): nothing was added from them");
  assert.equal(zh.noticeTitle("ocr_blank", 1, [], "", { ranges: "7", count: 1, total: null }), "OCR 在第 7 页没有识别到文字：这些页在输出中是空白的");
  const two = groupNotices([
    { itemKey: "a", itemName: "a.pdf", text: "[OCR] No text found on 1 page of a.pdf (page 7); it is blank in the output" },
    { itemKey: "b", itemName: "b.pdf", text: "PDF page 2 (1 of 30): OCR found no text; it is blank in the output." },
  ]);
  assert.deepEqual([two.length, two[0]?.count, two[0]?.pages], [1, 2, null]);
});

test("styles: tables keep to the column, the front line, the status dot, the config layer", () => {
  const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
  assert.ok(!/--bleed/.test(css), "no table reaches past the reading column any more");
  assert.match(css, /\.doc-body \.table-scroll > table \{\s*width: 100%;\s*margin-inline: 0;/);
  assert.match(css, /\.live-dot > span \{[^}]*background: var\(--text-3\);[^}]*animation: pulse/);
  assert.match(css, /\.live-dot\.is-done > span \{\s*background: var\(--success\);\s*animation: none;/);
  assert.match(css, /\.config-editor textarea \{[^}]*color: transparent;/);
  assert.match(css, /@media \(forced-colors: active\) \{\s*\.config-highlight \{\s*display: none;/);
});

test("Compare never pairs a block with an earlier page when it is not found ahead", () => {
  const line = (text: string, y: number) => ({ text, bbox: [10, y, 200, y + 12] as [number, number, number, number] });
  const pages = [
    { page: 1, lines: [line("Appendix", 20), line("Introduction", 40)] },
    { page: 2, lines: [line("Introduction", 20), line("The garden opens at eight.", 40)] },
  ];
  const found = locate(pages, [
    { id: "a", kind: "heading", text: ["Introduction"] },
    { id: "b", kind: "block", text: ["The garden opens at eight."] },
    { id: "c", kind: "heading", text: ["Appendix"] },
  ]);
  assert.equal(found.get("a")?.page, 1);
  assert.equal(found.get("b")?.page, 2);
  // Only the contents on page 1 say "Appendix": before the cursor, so it stays unlinked (its page is not read yet).
  assert.equal(found.has("c"), false);
});

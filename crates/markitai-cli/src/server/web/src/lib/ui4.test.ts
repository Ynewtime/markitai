// 2.0.1 owner items (2026-10-24, design G): Details open with a quiet "request options" row above
// the status line, the command line folded under it (one choice per browser), and the start of the
// result as Rendered or Source with its real line count.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { readRequestOpen, storeRequestOpen } from "./peek-pref.ts";
import { lineCount, sourceStart, START_LINES } from "./results.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");

test("a line count matches the server's markdown_lines", () => {
  // the same cases as files.rs `a_line_count_counts_a_last_line_without_its_break`
  for (const [text, lines] of [["", 0], ["a", 1], ["a\n", 1], ["a\n\nb", 3], ["a\r\nb\r\n", 2]] as const) assert.equal(lineCount(text), lines, JSON.stringify(text));
});

test("Source shows the first 12 lines, the front matter's muted, and the whole result's count", () => {
  const markdown = ["---", "title: Harbour tide survey", "tags:", "- sample", "---", "", "# Harbour tide survey", "", ...Array.from({ length: 30 }, (_, n) => `Reading ${n + 1}`)].join("\n") + "\n";
  const start = sourceStart({ markdown });
  assert.equal(START_LINES, 12);
  assert.equal(start.lines.length, 12);
  assert.equal(start.total, 38);
  assert.deepEqual(start.lines.slice(0, 6).map((line) => line.front), [true, true, true, true, true, false]);
  assert.equal(start.lines[6]?.text, "# Harbour tide survey");
  // an excerpt names the whole count the server sent, or none
  assert.equal(sourceStart({ markdown, truncated: true, markdown_lines: 4210 }).total, 4210);
  assert.equal(sourceStart({ markdown, truncated: true }).total, null);
  assert.equal(sourceStart({ markdown, truncated: true }).lines.length, 12);
  // a short result shows all of its lines; no front matter, nothing muted; one still open covers every shown line
  assert.deepEqual(sourceStart({ markdown: "# One\n\nTwo\n" }), { lines: [{ text: "# One", front: false }, { text: "", front: false }, { text: "Two", front: false }], total: 3 });
  assert.ok(sourceStart({ markdown: ["---", ...Array.from({ length: 20 }, (_, n) => `key${n}: v`), "---", "body"].join("\n") }).lines.every((line) => line.front));
});

test("the line count reads as the owner wrote it, in en, zh and ja", () => {
  assert.equal(zh.startLines(12, 4210), "前 12 行 / 共 4,210 行");
  assert.equal(zh.startLines(33, 33), "共 33 行");
  assert.equal(zh.startLines(12, null), "前 12 行");
  assert.equal(en.startLines(12, 4210), "First 12 of 4,210 lines");
  assert.equal(en.startLines(1, 1), "1 line");
  assert.equal(en.startLines(33, 33), "33 lines");
  assert.equal(ja.startLines(12, 40), "先頭 12 行 / 全 40 行");
  assert.equal(zh.requestShow, "查看请求参数");
  assert.equal(zh.requestHide, "收起请求参数");
  for (const dict of [en, zh, ja]) assert.ok(dict.requestShow && dict.requestHide && dict.startViews);
});

test("the request options stay open or folded for the browser, and storage that throws changes nothing", () => {
  const store = new Map<string, string>();
  const storage = { getItem: (key: string) => store.get(key) ?? null, setItem: (key: string, value: string) => void store.set(key, value) };
  assert.equal(readRequestOpen(storage), false);
  storeRequestOpen(true, storage);
  assert.equal(store.get("markitai.request"), "open");
  assert.equal(readRequestOpen(storage), true);
  storeRequestOpen(false, storage);
  assert.equal(readRequestOpen(storage), false);
  const broken = {
    getItem: (): string | null => {
      throw new Error("denied");
    },
    setItem: (): void => {
      throw new Error("denied");
    },
  };
  assert.equal(readRequestOpen(broken), false);
  storeRequestOpen(true, broken); // does not throw
});

test("Details: the request row comes first, its mark in the status mark's column; the strip opens under the status line", () => {
  const inspector = read("../components/inspector.tsx");
  const item = inspector.slice(inspector.indexOf("export function ItemInspector"), inspector.indexOf("export function fmtWhen"));
  const toggle = item.indexOf("<RequestToggle");
  const line = item.indexOf('<p class={failed ? "insp-line is-error" : "insp-line"}>');
  const strip = item.indexOf("<RequestStrip");
  assert.ok(toggle > 0 && toggle < line && line < strip, "toggle, status line, then the strip");
  assert.ok(strip < item.indexOf('<dl class="facts">'), "the strip comes before the facts");
  assert.ok(!/<dt>\{t\.factOptions\}<\/dt>\s*<dd class="fact-request">/.test(inspector), "no command line row in the facts any more");
  assert.match(inspector, /aria-expanded=\{open\}/);
  assert.match(inspector, /<Icon name="Terminal" size=\{14\} \/>/);
  const css = read("../styles/app.css");
  const block = (selector: string) => css.slice(css.indexOf(`\n${selector} {`) + 1, css.indexOf("}", css.indexOf(`\n${selector} {`)));
  // one column: the mark's box and gap are the status glyph's
  assert.match(block(".glyph"), /width: 18px;/);
  assert.match(block(".req-mark"), /width: 18px;/);
  assert.match(block(".insp.is-peek .insp-line"), /gap: 8px;/);
  assert.match(block(".req-toggle"), /gap: 8px;[\s\S]*padding: 0;/);
});

test("the start of a result switches Rendered and Source, Rendered first; only a Markdown output has the switch", () => {
  const inspector = read("../components/inspector.tsx");
  assert.match(inspector, /let startView: StartView = "rendered";/);
  assert.match(inspector, /const markdown = \/\\\.md\$\/i\.test\(item\.output \?\? ""\);/);
  assert.match(inspector, /<span class="seg start-views" role="group" aria-label=\{t\.startViews\}>/);
  assert.match(inspector, /aria-pressed=\{view === value\}/);
});

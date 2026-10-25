// 2.1.0 owner items (2026-10-24, design G): Details open with a quiet "request options" row above
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

test("Details (owner 2026-10-25): the status line first, the request toggle under it, the command block straight under the toggle", () => {
  const inspector = read("../components/inspector.tsx");
  for (const [from, to] of [["export function ItemInspector", "export function fmtWhen"], ["export function JobInspector", "export function stopAct"]]) {
    const body = inspector.slice(inspector.indexOf(from), inspector.indexOf(to));
    const line = body.indexOf('class={failed ? "insp-line is-error" : "insp-line"}') >= 0 ? body.indexOf('class={failed ? "insp-line is-error" : "insp-line"}') : body.indexOf('<p class="insp-line">');
    const toggle = body.indexOf("<RequestToggle");
    const strip = body.indexOf("<RequestStrip");
    assert.ok(line > 0 && line < toggle && toggle < strip, `${from}: status line, toggle, then the strip`);
    // nothing between the toggle and its block
    assert.match(body.slice(toggle), /^<RequestToggle [^\n]*\/>\}\n\s*\{request && requestOpen && <RequestStrip /);
    assert.ok(strip < body.indexOf("<Facts>"), "the strip comes before the facts");
  }
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
  assert.match(block(".insp-request"), /margin-top: 4px;/);
});

test("Details facts: the label column takes its labels' width up to 120 px (96 px on a phone), a longer label scrolls on one line", () => {
  const css = read("../styles/app.css");
  assert.match(css, /\n\.facts \{\n {2}display: grid;\n {2}\/\*[^*]*\*\/\n {2}grid-template-columns: fit-content\(120px\) minmax\(0, 1fr\);/);
  assert.match(css, /\n {2}\.facts \{\n {4}grid-template-columns: fit-content\(96px\) minmax\(0, 1fr\);/);
  assert.match(css, /\.front-key,\n\.front-cell,\n\.facts > dt \{\n {2}overflow-x: auto;\n {2}overflow-y: hidden;\n {2}overscroll-behavior-x: contain;\n {2}scrollbar-width: none;\n {2}white-space: nowrap;/);
  const inspector = read("../components/inspector.tsx");
  assert.match(inspector, /watchScrollCells\(list\.current, "\.facts > dt"\)/);
  assert.equal((inspector.match(/<dl class="facts">/g) ?? []).length, 0, "every facts list watches its labels");
});

test("the start of a result switches Rendered and Source, Rendered first; only a Markdown output has the switch", () => {
  const inspector = read("../components/inspector.tsx");
  assert.match(inspector, /let startView: StartView = "rendered";/);
  assert.match(inspector, /const markdown = \/\\\.md\$\/i\.test\(item\.output \?\? ""\);/);
  assert.match(inspector, /<span class="seg start-views" role="group" aria-label=\{t\.startViews\}>/);
  assert.match(inspector, /aria-pressed=\{view === value\}/);
});

test("properties: one fixed key column (120 px, 92 px on a phone); keys and values on one line that scrolls in its cell, no ellipsis", () => {
  const css = read("../styles/app.css");
  const rule = (selector: string, from = 0) => {
    const at = css.indexOf(`\n${selector} {`, from);
    assert.ok(at >= 0, selector);
    return css.slice(at + 1, css.indexOf("}", at));
  };
  assert.match(rule(".front-table"), /grid-template-columns: 120px minmax\(0, 1fr\);/);
  const phone = css.indexOf("@media (max-width: 760px) {\n  .front-table {");
  assert.match(css.slice(phone, css.indexOf("}\n}", phone)), /grid-template-columns: 92px minmax\(0, 1fr\);/);
  assert.ok(!/text-overflow: ellipsis/.test(rule(".front-table dt")), "a key is not cut with an ellipsis");
  assert.match(rule(".front-tags"), /width: max-content;/);
  assert.ok(!/flex-wrap: wrap/.test(rule(".front-tags")), "the tags stay on one line until the row is opened");
  assert.match(rule(".front-cell.is-open"), /overflow: visible;\n {2}white-space: normal;/);
  assert.match(rule(".front-cell.is-open .front-tags"), /flex-wrap: wrap;/);
  assert.match(rule('.front-more[aria-expanded="true"] svg'), /transform: rotate\(180deg\);/);
  // paper cannot scroll: keys, values and fact labels wrap again in print, without chevrons
  assert.match(css, /@media print \{\n {2}\.front-key,\n {2}\.front-cell,\n {2}\.facts > dt \{\n {4}overflow: visible;\n {4}white-space: normal;/);
  const card = read("../components/front-matter.tsx");
  assert.match(card, /<dt>\n\s+<span class="front-key" title=\{[^\n]*\}>\n\s+\{label\}\n\s+<\/span>\n\s+<\/dt>/);
  assert.ok(!/\.title = /.test(card), "no tooltip in place of the key");
  assert.match(card, /watchScrollCells\(target, "\.front-key, \.front-cell"/);
});

test("properties: a row whose value overflows gets a chevron that opens it (and folds it again), named for its key", () => {
  const card = read("../components/front-matter.tsx");
  assert.match(card, /const more = opened \|\| longKeys\.has\(row\.key\);/);
  assert.match(card, /\{more && \(\n\s+<button type="button" class="front-more" aria-expanded=\{opened\} aria-label=\{opened \? t\.frontCollapseRow\(label\) : t\.frontExpandRow\(label\)\}/);
  // a tap on the value opens it too, but not on its link or while selecting text
  assert.match(card, /if \(!more \|\| \(event\.target as Element\)\.closest\("a"\) \|\| getSelection\(\)\?\.toString\(\)\) return;/);
  assert.equal(zh.frontExpandRow("简介"), "展开简介");
  assert.equal(zh.frontCollapseRow("简介"), "收起简介");
  assert.equal(en.frontExpandRow("Summary"), "Expand Summary");
  assert.equal(ja.frontCollapseRow("概要"), "概要を折りたたむ");
  const cells = read("./scroll-cells.ts");
  assert.match(cells, /new ResizeObserver\(later\)/);
  assert.match(cells, /new MutationObserver\(later\)/);
});

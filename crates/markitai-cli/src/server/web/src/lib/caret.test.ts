// Edit config's caret and coloured layer line up with the textarea's text (owner report, 2026-10-25).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { caretHeight, caretTop, lineColumn } from "./caret.ts";

const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.[\]()"=:>,\n]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";

test("the coloured layer and the textarea share every metric that places a line's text", () => {
  // one rule for both, so neither can drift
  const shared = block(".config-highlight,\n.config-editor textarea");
  for (const rule of ["margin: 0;", "padding: 14px 16px;", "border: 0;", "font-family: var(--font-mono);", "font-size: var(--text-sm);", "font-variant-ligatures: none;", "line-height: 1.65;", "letter-spacing: 0;", "tab-size: 2;", "white-space: pre;"]) {
    assert.ok(shared.includes(rule), rule);
  }
  // and no rule of either's own sets them
  const own = (selector: string) => new RegExp(`\\}\\n${selector.replace(/[.[\]()"=:>]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";
  for (const selector of [".config-highlight", ".config-editor textarea"]) {
    assert.ok(own(selector).length > 0, selector);
    assert.ok(!/font|line-height|padding|letter-spacing|tab-size/.test(own(selector)), selector);
  }
});

test("the page's caret: one font-high bar centred on its line, the textarea's own hidden only meanwhile", () => {
  assert.deepEqual(lineColumn('{\n  "llm": [\n', 11), { line: 1, column: 9, lineStart: 2 });
  assert.deepEqual(lineColumn("ab", 0), { line: 0, column: 0, lineStart: 0 });
  // 12 px text on a 19.8 px line: a 14 px caret, 2.9 px from the line's top and bottom
  assert.equal(caretHeight(12, 19.8), 14);
  assert.equal(caretHeight(20, 18), 18);
  const metrics = { paddingTop: 14, lineHeight: 19.8, fontSize: 12 };
  const top = caretTop(3, metrics);
  const lineTop = 14 + 3 * 19.8;
  assert.ok(Math.abs(top - lineTop - (19.8 - 14) / 2) < 1e-9);
  assert.ok(Math.abs(lineTop + 19.8 - (top + 14) - (top - lineTop)) < 1e-9);
  assert.match(block(".config-caret"), /position: absolute;/);
  assert.match(block(".config-code.has-caret textarea"), /caret-color: transparent;/);
  // without the page's caret (no script, forced colours) the textarea's shows
  const forced = css.slice(css.indexOf("@media (forced-colors: active) {\n  .config-highlight,"));
  assert.match(forced, /\.config-highlight,\n {2}\.config-caret \{\n {4}display: none;/);
  assert.match(forced, /caret-color: CanvasText;/);
  const layer = readFileSync(new URL("../components/json-highlight.tsx", import.meta.url), "utf8");
  assert.match(layer, /host\?\.classList\.add\("has-caret"\)/);
  assert.match(layer, /if \(document\.activeElement !== area \|\| start !== area\.selectionEnd\) \{\s*bar\.hidden = true;/);
});

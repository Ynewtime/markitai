import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { buildDate, versionLine } from "./build.ts";
import { mathTitle } from "./math-syntax.ts";
import { entryOnly, listEntry } from "./route.ts";

const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
const page = readFileSync(new URL("../components/conversions-page.tsx", import.meta.url), "utf8");

test("a history entry narrows the list only with a list of item keys", () => {
  assert.equal(listEntry(null), null);
  assert.equal(listEntry("dialog"), null);
  assert.deepEqual(entryOnly(listEntry({ wb: "only", only: ["a/i1", "b/i2"] })), ["a/i1", "b/i2"]);
  assert.equal(entryOnly(listEntry({ wb: "summary" })), null);
  assert.equal(entryOnly(listEntry({ only: ["a/i1", 3] })), null);
  assert.equal(entryOnly(null), null);
});

test("Show items narrows the list with a bar over it, not a chip in the filter row", () => {
  assert.ok(!css.includes(".subset-chip"));
  assert.match(css, /\n\.subset-bar \{/);
  // the bar counts the narrowed items still listed, not the keys it was given
  assert.match(page, /showingItems\(onlyLeft\)/);
  assert.ok(!page.includes("showingItems(props.only.length)"));
});

test("a row's notice mark stands on its name's line", () => {
  // a monospace figure rode ~3 px above the name; the Inter figure moves down the 1 px its smaller size leaves
  const mark = /\n\.warn-mark \{\n([^}]*)\}/.exec(css)?.[1] ?? "";
  assert.match(mark, /font-family: var\(--font-sans\);/);
  assert.match(/\n\.warn-wrap \{\n([^}]*)\}/.exec(css)?.[1] ?? "", /top: 1px;/);
});

test("contents entries keep to one line, the whole heading in their tooltip", () => {
  const text = /\n\.toc-text \{\n([^}]*)\}/.exec(css)?.[1] ?? "";
  assert.match(text, /white-space: nowrap;/);
  assert.match(text, /text-overflow: ellipsis;/);
  assert.ok(!text.includes("line-clamp"));
  assert.equal(mathTitle("The probability of getting $k$ heads when flipping $n$ coins"), "The probability of getting k heads when flipping n coins");
  assert.equal(mathTitle("Plain heading"), "Plain heading");
  assert.equal(mathTitle("Area $$\\pi r^2$$"), "Area \\pi r^2");
});

test("a submission's header takes a single item's columns: box, then chevron over the status marks", () => {
  const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.]/g, "\\.")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";
  const columns = (selector: string) => /grid-template-columns: ([^;]+);/.exec(block(selector))?.[1];
  assert.equal(columns(".group-head"), columns(".row"));
  assert.equal(columns(".group-head.has-check"), columns(".row.has-check"));
  assert.match(block(".group-caret"), /place-items: center start;/);
  assert.match(block(".group-caret"), /padding-left: 2px;/);
});

test("the sidebar and the phone Settings list show the build's date and time in local time", () => {
  const built = new Date(2026, 9, 17, 19, 54, 7).toISOString();
  assert.equal(buildDate(built), "20261017 19:54");
  assert.equal(buildDate(new Date(2025, 0, 2, 3, 4).toISOString()), "20250102 03:04");
  assert.equal(buildDate("not a time"), null);
  const line = versionLine({ version: "1.3.0-dev", build: { version: "1.3.0-dev", commit: "3efaa92", built_at: built, dirty: false } });
  assert.equal(line?.parts.date, "20261017 19:54");
  assert.equal(versionLine({ version: "1.3.0-dev" })?.parts.date, null);
  const shell = readFileSync(new URL("../components/shell.tsx", import.meta.url), "utf8");
  const settings = readFileSync(new URL("../components/settings-page.tsx", import.meta.url), "utf8");
  // the row reads "Build: 20261017 19:54" in every language: the label is not translated
  for (const source of [shell, settings]) {
    assert.match(source, /\{version\.parts\.date\}/);
    assert.match(source, />Build:</);
  }
});

test("folded front matter is one button across the row, its property count before the tags", () => {
  const card = readFileSync(new URL("../components/front-matter.tsx", import.meta.url), "utf8");
  const folded = card.slice(card.lastIndexOf("return ("));
  assert.match(folded, /<button type="button" class="front-fold" aria-expanded="false"/);
  // the count comes first, then the tags, then the Summary mark (a span inside the button, not a second button)
  assert.ok(folded.indexOf("front-count") < folded.indexOf("front-line"));
  assert.ok(folded.indexOf("front-line") < folded.indexOf("front-toggle"));
  assert.equal((folded.match(/<button/g) ?? []).length, 1);
  assert.match(/\n\.front-fold \{\n([^}]*)\}/.exec(css)?.[1] ?? "", /flex: 1;/);
});

// 2.1.0 owner items on the release candidate (2026-10-25): the options dialog's command scrolls
// instead of being cut, the properties line stays put and folded says only "Properties · N", and
// Details' result start in Source opens Preview's Source.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { openPreviewAs, takePreviewView } from "./peek-pref.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");
const css = read("../styles/app.css");
const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.[\]()"=:>]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";

test("the dialog's command is one line that scrolls in its box, fading while there is more", () => {
  const line = block(".sheet-cli .cli-text");
  assert.match(line, /overflow-x: auto;/);
  assert.match(line, /scrollbar-width: none;/);
  assert.match(line, /white-space: nowrap;/);
  assert.ok(!/ellipsis|overflow: hidden/.test(line));
  assert.match(block(".sheet-cli .cli-text.is-more"), /mask-image: linear-gradient\(to right/);
  // Copy keeps its place at the box's right end
  assert.match(block(".sheet-cli .icon-btn"), /flex: none;/);
  const dialog = read("../components/options-dialog.tsx");
  assert.match(dialog, /<code ref=\{cli\} class="cli-text"[^>]*onScroll=\{\(event\) => markMore\(event\.currentTarget\)\}/);
});

test("the properties line is the same button folded and open, with no tags or Summary on it", () => {
  const card = read("../components/front-matter.tsx");
  // one return, so the button is the same element in both states and keeps its place and focus
  assert.equal((card.match(/\n  return \(/g) ?? []).length, 1);
  const head = card.slice(card.indexOf('<button type="button" class="front-head"'), card.indexOf("</button>", card.indexOf('class="front-head"')));
  assert.match(head, /aria-expanded=\{open\}/);
  assert.deepEqual([...head.matchAll(/<Icon name="(\w+)"/g)].map((m) => m[1]), ["Tag", "CaretDown"]);
  assert.match(head, /t\.frontProperties\(rows\.length\)/);
  assert.ok(!/front-line|front-toggle|frontSummary|front-fold/.test(card));
  for (const dict of [en, zh, ja]) assert.equal("frontSummary" in dict, false);
  // the open state only turns the chevron; the line's own box is the same
  assert.match(block('.front-head[aria-expanded="true"] > svg:last-child'), /transform: rotate\(180deg\);/);
  assert.ok(!/display|margin|gap|padding-top/.test(block(".front-meta.is-open")));
});

test("Details' start in Source opens Preview in Source, once; Rendered stays the default", () => {
  assert.equal(takePreviewView(), "rendered");
  openPreviewAs("source");
  assert.equal(takePreviewView(), "source");
  assert.equal(takePreviewView(), "rendered");
  const inspector = read("../components/inspector.tsx");
  assert.match(inspector, /openPreviewAs\(shownSource \? "source" : "rendered"\);\s*onOpen\(\);/);
  const peek = read("../components/peek.tsx");
  assert.match(peek, /useState<Tab>\(takePreviewView\)/);
  // another item opens in Rendered again
  assert.match(peek, /if \(shownKey\.current !== item\.key\) setTab\("rendered"\);/);
});

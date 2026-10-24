// 2.0.1 pre-release review of the rc (2026-10-25): the findings fixed in the workbench.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");
const css = read("../styles/app.css");
const block = (selector: string, within = css) => new RegExp(`\\n {0,2}${selector.replace(/[.[\]()"=:>*]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(within)?.[1] ?? "";

test("M4: the (i) beside a label is a small round target again, 24 px on a phone", () => {
  const help = block(".opt-help");
  assert.match(help, /width: 18px;/);
  assert.match(help, /border-radius: 50%;/);
  assert.match(help, /color: var\(--text-3\);/);
  assert.match(css, /\.opt-help:focus-visible,\n\.weight-pill:focus-visible/);
  const hover = css.slice(css.indexOf("@media (hover: hover) and (pointer: fine)"));
  assert.match(hover, /\.opt-help:hover,/);
  const phone = css.slice(css.indexOf("@media screen and (max-width: 760px) {\n  /* an (i)"));
  assert.match(block(".opt-help::after", phone), /inset: -3px;/);
});

test("M3: Escape on an open (i) bubble closes the bubble only", () => {
  const tip = read("../components/help-tooltip.tsx");
  // caught first, on the window in the capture phase, and not passed on to the dialog or page under it
  assert.match(tip, /window\.addEventListener\("keydown", escape, true\);/);
  assert.match(tip, /window\.removeEventListener\("keydown", escape, true\);/);
  const escape = tip.slice(tip.indexOf("const escape = (event: KeyboardEvent)"), tip.indexOf("const other ="));
  assert.match(escape, /event\.preventDefault\(\);\s*event\.stopImmediatePropagation\(\);\s*dismiss\(\);/);
});

test("L8: a bubble opened from the keyboard stays while focus is on its control", () => {
  const tip = read("../components/help-tooltip.tsx");
  const timer = tip.slice(tip.indexOf("const timer = setTimeout"), tip.indexOf("AUTO_HIDE_MS + extraMs"));
  assert.match(timer, /!hovered\.current && !anchor\.current\?\.querySelector\(":focus-visible"\) && !anchor\.current\?\.matches\(":focus-visible"\)/);
});

test("M2: the options dialog has a history entry, so Back closes it and stays on Convert", () => {
  const panel = read("../components/options-panel.tsx");
  assert.match(panel, /history\.pushState\(\{ wb: OPTIONS_ENTRY \}, "", location\.href\);/);
  // Back (popstate off the entry) closes the dialog; closing by hand goes back over the entry
  assert.match(panel, /const onPop = \(\) => \{\s*if \(listEntry\(\)\?\.wb === OPTIONS_ENTRY\) return;/);
  assert.match(panel, /if \(pushed\.current && listEntry\(\)\?\.wb === OPTIONS_ENTRY\) \{\s*backing\.current = true;\s*history\.back\(\);/);
  // an entry left from before a reload is dropped
  assert.match(panel, /if \(listEntry\(\)\?\.wb === OPTIONS_ENTRY\) history\.replaceState\(null, "", location\.href\);/);
  // "Add a model" leaves only after that Back
  assert.match(panel, /if \(backing\.current\) afterClose\.current = onAddModel;\s*else onAddModel\(\);/);
});

test("M5: a choice menu opens upward when there is more room there, capped and scrolling", () => {
  const pick = read("../components/pick-menu.tsx");
  const menu = pick.slice(pick.indexOf("export function ChoiceMenu"), pick.indexOf("export interface DownloadEntry"));
  // measured against the phone's tab bar (a fixed element: getClientRects, not offsetParent)
  assert.match(menu, /bar && bar\.getClientRects\(\)\.length > 0 \? bar\.getBoundingClientRect\(\)\.top : window\.innerHeight/);
  assert.match(menu, /const flip = box\.scrollHeight > below && above > below;/);
  assert.match(menu, /box\.style\.maxHeight = `\$\{Math\.max\(120, Math\.floor\(flip \? above : below\)\)\}px`;/);
  assert.match(menu, /class=\{up \? "menu is-start is-up opt-choice-menu" : "menu is-start opt-choice-menu"\}/);
  assert.match(block(".opt-choice-menu"), /overflow-y: auto;/);
});

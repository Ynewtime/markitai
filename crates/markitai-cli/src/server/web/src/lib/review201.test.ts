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

test("L7: focus stays on a button that turns unavailable, and goes back to a menu's trigger", () => {
  const dialog = read("../components/options-dialog.tsx");
  const bar = dialog.slice(dialog.indexOf("function DefaultsBar"));
  // Reset and Save as default are aria-disabled (focusable), never disabled under the pressed key
  assert.match(bar, /aria-disabled=\{!changed \|\| undefined\} onClick=\{\(\) => changed && control\.reset\(\)\}/);
  assert.match(bar, /aria-disabled=\{same \|\| remote \|\| saving \|\| undefined\}/);
  assert.match(bar, /onClick=\{\(\) => !\(same \|\| remote \|\| saving\) && control\.save\(\)\}/);
  assert.ok(!/disabled=\{!changed\}|disabled=\{same \|\| remote \|\| saving\}/.test(bar));
  assert.match(css, /\.btn:disabled,\n\.btn\[aria-disabled="true"\] \{\n {2}opacity: 0\.55;/);
  assert.match(css, /\.btn:not\(:disabled\):not\(\[aria-disabled="true"\]\):active/);
  // a Download entry and a "⋯" entry give focus back to their trigger
  const pick = read("../components/pick-menu.tsx");
  assert.match(pick, /menu\.close\(\);\s*entry\.run\(\);/);
  const more = read("../components/action-menu.tsx");
  assert.match(more, /const choose = \(run: \(\) => void\) => \{\s*setOpen\(false\);\s*trigger\.current\?\.focus\(\);\s*run\(\);/);
});

test("L9: every label a screen reader is given is one it reads", () => {
  // the dialog's command strip is a named group, as Details' strip is
  assert.match(read("../components/options-dialog.tsx"), /<code ref=\{cli\} class="cli-text" role="group" tabIndex=\{0\} aria-label=\{t\.cliAria\}/);
  // a duration's spoken form is sr-only text, not a label on a plain span
  const bits = read("../components/item-bits.tsx");
  assert.match(bits, /<span aria-hidden="true">\{fmtDur\(ms\)\}<\/span>\s*<span class="sr-only">\{t\.ariaDuration\(minutes, seconds\)\}<\/span>/);
  // with the phone's view switch there are no tabs: a pane is a region named by its view
  const view = read("../components/markdown-view.tsx");
  assert.match(view, /barHost && compactBar \? \{ role: "region" as const, "aria-label": tabLabels\[tab\] \} : \{ role: "tabpanel" as const, "aria-labelledby": `\$\{prefix\}-tab-\$\{tab\}` \}/);
  assert.equal((view.match(/\{\.\.\.panel\("(rendered|source|diff|files)"\)\}/g) ?? []).length, 4);
  assert.ok(!/role="tabpanel" aria-labelledby/.test(view));
});

test("L10: the weight tooltip joins its sentences in the page's language", async () => {
  assert.match(read("../components/weight-pill.tsx"), /title=\{`\$\{title\}\$\{t\.toastLead\}\$\{t\.weightEditHint\}`\}/);
  const { zh } = await import("../i18n/zh.ts");
  const { ja } = await import("../i18n/ja.ts");
  for (const dict of [zh, ja]) assert.equal(`${dict.weightHint}${dict.toastLead}${dict.weightEditHint}`.includes(". "), false);
});

test("L11: the properties line and its chevrons highlight on hover only where a pointer hovers", () => {
  const hover = css.slice(css.indexOf("@media (hover: hover) and (pointer: fine) {"));
  const outside = css.slice(0, css.indexOf("@media (hover: hover) and (pointer: fine) {"));
  for (const selector of [".front-head:hover", ".front-more:hover"]) {
    assert.ok(hover.includes(`  ${selector} {`), selector);
    assert.ok(!outside.includes(`\n${selector} {`), selector);
  }
});

test("L12: a visually hidden box takes no room after the page's last line", () => {
  const hidden = block(".sr-only");
  for (const rule of ["margin: -1px;", "padding: 0;", "border: 0;", "clip-path: inset(50%);", "width: 1px;", "height: 1px;"]) assert.ok(hidden.includes(rule), rule);
});

test("L13: a cut value can be read: two lines and a tooltip, key tooltips, a fade on both scrolled edges", () => {
  const dialog = read("../components/options-dialog.tsx");
  assert.match(dialog, /"sheet-row-value is-changed" : "sheet-row-value"\} title=\{group\.value\}>/);
  const value = block(".sheet-row-value");
  assert.match(value, /-webkit-line-clamp: 2;/);
  assert.ok(!/white-space: nowrap|text-overflow: ellipsis/.test(value));
  assert.match(read("../components/front-matter.tsx"), /<span class="front-key" title=\{label === row\.key \? label : `\$\{label\} \(\$\{row\.key\}\)`\}>/);
  assert.match(read("./scroll-cells.ts"), /cell\.classList\.toggle\("is-less", cell\.scrollLeft > 1\);/);
  assert.match(block(".sheet-cli .cli-text.is-less.is-more"), /mask-image: linear-gradient\(to right, transparent, #000 20px, #000 calc\(100% - 20px\), transparent\);/);
});

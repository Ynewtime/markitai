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

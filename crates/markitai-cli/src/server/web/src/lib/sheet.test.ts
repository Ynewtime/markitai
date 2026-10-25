// 2.1.0 owner decision (2026-10-24, option B): the conversion options leave the page for a dialog, a
// sheet from the bottom on a phone and a centred dialog near the top on a wider screen. Its first page
// lists the groups with their values; a group opens in place; the foot keeps the command and the defaults.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ConvertDefaults } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import {
  applyPreset,
  BUILTIN_DEFAULTS,
  changeAdvanced,
  changedOptions,
  composerFromDefaults,
  groupValue,
  initialComposer,
  OPTION_GROUPS,
  readRemembered,
  sameAsDefaults,
} from "./options.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");

test("a group's value is its choice, the switches that are on, or Off", () => {
  assert.equal(groupValue([], zh.optionOff), "关");
  assert.equal(groupValue(["OCR"], zh.optionOff), "OCR");
  assert.equal(groupValue([zh.profileNone, zh.advJson], zh.optionOff), "默认 · JSON");
  for (const dict of [en, zh, ja]) assert.match(dict.optionRow(dict.advEnhance, dict.ocr), /OCR$/);
  assert.equal(zh.optionRow("增强", "OCR"), "增强：OCR");
  assert.equal(en.optionRow("Enhance", "OCR"), "Enhance: OCR");
});

test("what differs from the defaults is named per option, the way Save as default counts it", () => {
  const start = initialComposer(readRemembered(null));
  assert.equal(changedOptions(start, BUILTIN_DEFAULTS).size, 0);
  assert.ok(sameAsDefaults(start, BUILTIN_DEFAULTS));
  const ocr = { ...start, ocr: true, format: "json" as const };
  assert.deepEqual([...changedOptions(ocr, BUILTIN_DEFAULTS)].sort(), ["format", "ocr"]);
  // Alt text and descriptions count only while a model analyses pictures.
  const alt = { ...start, advanced: changeAdvanced(start.advanced, "alt", true) };
  assert.equal(changedOptions(alt, BUILTIN_DEFAULTS).size, 0);
  const standard = applyPreset(start, "standard");
  assert.deepEqual([...changedOptions(standard, BUILTIN_DEFAULTS)].sort(), ["alt", "desc", "llm"]);
  // Cloudflare is never a default: choosing it is a difference, whatever else matches.
  const remote = { ...start, advanced: { ...start.advanced, strategy: "cloudflare" as const, backend: "cloudflare" as const } };
  assert.deepEqual([...changedOptions(remote, BUILTIN_DEFAULTS)].sort(), ["backend", "strategy"]);
  assert.equal(sameAsDefaults(remote, BUILTIN_DEFAULTS), false);
  // Saved defaults are compared as saved: a v1.20 key the composer has not set still holds.
  const saved: ConvertDefaults = { ...BUILTIN_DEFAULTS, ocr: true, profile: "rag", strategy: "static", ocr_lang: "ko", page_markers: true };
  const composer = composerFromDefaults(saved);
  assert.equal(changedOptions(composer, saved).size, 0);
  assert.deepEqual([...changedOptions({ ...composer, ocrLang: "ja" }, saved)], ["ocr_lang"]);
  // Every option key belongs to one group, so a difference always shows on a row.
  const keys = Object.values(OPTION_GROUPS).flat();
  assert.equal(new Set(keys).size, keys.length);
  for (const key of ["llm", "ocr", "alt", "desc", "screenshot", "screenshot_only", "pure", "no_cache", "no_compress", "profile", "strategy", "format", "ocr_lang", "ocr_all_pages", "page_markers", "backend", "text_only"])
    assert.ok(keys.includes(key as never), key);
});

test("the composer has no inline panel: Options opens the dialog and carries a dot while the choices differ", () => {
  const panel = read("../components/options-panel.tsx");
  assert.ok(!/class="opts"|opts-grid|cli-bar/.test(panel), "the drawer is gone");
  assert.match(panel, /aria-haspopup="dialog"/);
  assert.match(panel, /\{changed\.size > 0 && <span class="tool-dot" aria-hidden="true" \/>\}/);
  assert.match(panel, /aria-label=\{changed\.size > 0 \? t\.optionRow\(t\.options, t\.defaultsChanged\) : t\.options\}/);
  // Closing gives focus back to Options.
  assert.match(panel, /setOpen\(false\);\n\s+requestAnimationFrame\(\(\) => trigger\.current\?\.focus\(\)\);/);
  const app = read("../app.tsx");
  assert.match(app, /onAddModel=\{role === "admin" \|\| userModels \? \(\) => navigate\(\{ view: "settings", section: "models" \}\) : undefined\}/);
});

test("the dialog: modal, rows that name their value, a group in place with Back, choices that apply at once", () => {
  const dialog = read("../components/options-dialog.tsx");
  assert.match(dialog, /role="dialog"\n\s+aria-modal="true"\n\s+aria-labelledby="opts-page"/);
  assert.match(dialog, /useModal\(dialog, onClose\);/);
  assert.match(dialog, /aria-label=\{t\.optionRow\(group\.name, group\.value\)\}/);
  assert.match(dialog, /class=\{differs\(group\.page\) \? "sheet-row-value is-changed" : "sheet-row-value"\}/);
  assert.match(dialog, /class="icon-btn sheet-back" aria-label=\{t\.back\}/);
  // Back focuses the row the group was opened from.
  assert.match(dialog, /querySelector<HTMLElement>\(`\[data-page="\$\{left\.current\}"\]`\)\?\.focus\(\)/);
  // A single choice is a radio list (arrow keys move and choose), several are switches.
  assert.match(dialog, /role="radiogroup"/);
  assert.match(dialog, /role="radio"[\s\S]*aria-checked=\{on\}[\s\S]*tabIndex=\{choice === focusable \? 0 : -1\}/);
  assert.match(dialog, /role="switch"/);
  // Cloudflare is a dimmed choice with its reason; there is no separate Cloudflare group.
  assert.match(dialog, /value === "cloudflare" && reason\n\s+\? short\(t\.cloudflareReason\(reason\)\)/);
  assert.ok(!/text="Cloudflare"/.test(dialog));
  // The foot: the command on one line with a copy button, then Reset and Save as default.
  assert.match(dialog, /class="icon-btn sheet-copy"/);
  assert.match(dialog, /\{t\.defaultsReset\}[\s\S]*\{t\.defaultsSave\}/);
  // The handle pulls the sheet down (on a phone only); far enough closes it.
  assert.match(dialog, /const SHEET = "\(max-width: 760px\)";/);
  assert.match(dialog, /!window\.matchMedia\(SHEET\)\.matches\) return;/);
  assert.match(dialog, /if \(distance > SWIPE_CLOSE\) onClose\(\);/);
});

test("a sheet on a phone, a dialog near the top on a wider screen", () => {
  const css = read("../styles/app.css");
  assert.match(css, /\n\.veil\.opts-veil \{\n {2}align-items: flex-start;\n {2}padding-top: 10vh;\n\}/);
  assert.match(css, /\n\.dialog\.opts-dialog \{\n {2}width: min\(480px, 100%\);/);
  assert.match(css, /\n\.sheet-handle \{\n {2}display: none;\n\}/);
  const phone = css.slice(css.indexOf("@media screen and (max-width: 760px)"));
  assert.match(phone, /\.veil\.opts-veil \{\n {4}align-items: flex-end;\n {4}padding: 0;\n {2}\}/);
  assert.match(phone, /\.dialog\.opts-dialog \{\n {4}width: 100%;\n {4}max-height: 88dvh;/);
  assert.match(phone, /\.sheet-handle \{\n {4}display: block;/);
  assert.match(css, /\.sheet-row \{[^}]*min-height: 44px;/);
  assert.match(css, /\.sheet-cli \.cli-text \{[^}]*overflow-x: auto;[^}]*white-space: nowrap;/);
});

test("the dialog's words exist in every language, short and without the drawer's paragraphs", () => {
  for (const dict of [en, zh, ja]) {
    for (const key of ["optionsTitle", "back", "optionOff", "noModel", "addModelLink"] as const) assert.ok(dict[key].length > 0, key);
    // The second lines are one short line each.
    for (const key of ["helpLlm", "helpOcr", "helpAlt", "helpDesc", "helpScreenshot", "helpRag", "helpJson", "helpAuto", "helpDefuddle", "helpNative", "helpPure", "helpCache", "helpCompress"] as const)
      assert.ok(dict[key].length <= 60, `${key}: ${dict[key]}`);
  }
  assert.equal(zh.optionsTitle, "转换选项");
  assert.equal(zh.noModel, "未配置模型");
  assert.equal(zh.addModelLink, "去添加");
  for (const gone of ["presetHint", "helpEnhance", "helpAdvanced", "advSourceExclusive", "noticeDefuddle"]) assert.ok(!(gone in en), gone);
});

// Contract v1.20: JSON output, the local OCR language, every-page OCR and page markers
// in the composer, the request, the command line and the saved defaults.
import assert from "node:assert/strict";
import test from "node:test";
import type { ConvertDefaults } from "../api/types.ts";
import { cliCommand } from "./cli.ts";
import { groupNotices } from "./notices.ts";
import {
  ADVANCED_DEFAULTS,
  composerFromDefaults,
  defaultsOf,
  hasAdvancedChoices,
  publicOptions,
  resolveOptions,
  sameAsDefaults,
  type Composer,
} from "./options.ts";

const base: Composer = { preset: "minimal", llm: false, ocr: false, profile: null, advanced: { ...ADVANCED_DEFAULTS }, format: null, ocrLang: null };

test("an older backend's request never carries the v1.20 keys", () => {
  const options = resolveOptions({ ...base, ocr: true });
  for (const key of ["format", "ocr_lang", "ocr_all_pages", "page_markers"]) assert.equal(key in options, false, key);
});

test("JSON output, the OCR language, every-page OCR and page markers reach the request and the command line", () => {
  const chosen: Composer = { ...base, ocr: true, format: "json", ocrLang: "ko", advanced: { ...ADVANCED_DEFAULTS, ocrAllPages: true, pageMarkers: true } };
  const options = resolveOptions(chosen);
  assert.equal(options.format, "json");
  assert.equal(options.ocr_lang, "ko");
  assert.equal(options.ocr_all_pages, true);
  assert.equal(options.page_markers, true);
  assert.equal(
    cliCommand(["./scan.pdf"], options),
    `markitai ./scan.pdf -o output/ --ocr -f json --page-markers --config-json '{"ocr":{"lang":"ko","per_page_routing":false}}'`,
  );
  // The language is local OCR's: with a model OCR reads through it, so the key stays out.
  const withModel = resolveOptions({ ...chosen, llm: true });
  assert.equal("ocr_lang" in withModel, false);
  assert.equal(withModel.ocr_all_pages, true);
  // Without OCR neither OCR setting is sent.
  const noOcr = resolveOptions({ ...chosen, ocr: false });
  assert.equal("ocr_lang" in noOcr, false);
  assert.equal("ocr_all_pages" in noOcr, false);
  // The automatic language and Markdown output are the command line's defaults.
  assert.equal(cliCommand([], resolveOptions({ ...base, ocr: true, format: "markdown", ocrLang: "en" })), "markitai <files-or-urls> -o output/ --ocr");
});

test("saved options keep the v1.20 keys with valid values only", () => {
  const options = publicOptions({ format: "json", ocr_lang: "zh_tw", ocr_all_pages: false, page_markers: true, origin: "cli" });
  assert.equal(options.format, "json");
  assert.equal(options.ocr_lang, "zh_tw");
  assert.equal(options.ocr_all_pages, false);
  assert.equal(options.page_markers, true);
  const bad = publicOptions({ format: "yaml", ocr_lang: "../en", ocr_all_pages: "yes", page_markers: 1 });
  for (const key of ["format", "ocr_lang", "ocr_all_pages", "page_markers"]) assert.equal(key in bad, false, key);
});

test("the v1.20 defaults start the composer, are saved back and count as met", () => {
  const defaults: ConvertDefaults = {
    llm: false,
    ocr: true,
    alt: false,
    desc: false,
    screenshot: false,
    screenshot_only: false,
    pure: false,
    no_cache: false,
    no_compress: false,
    profile: null,
    strategy: "auto",
    format: "json",
    ocr_lang: "ko",
    ocr_all_pages: true,
    page_markers: false,
  };
  const composer = composerFromDefaults(defaults);
  assert.equal(composer.format, "json");
  assert.equal(composer.ocrLang, "ko");
  assert.equal(composer.advanced.ocrAllPages, true);
  assert.equal(composer.advanced.pageMarkers, false);
  assert.ok(hasAdvancedChoices(composer.advanced), "every-page OCR opens the advanced fold");
  assert.deepEqual(defaultsOf(composer), defaults);
  assert.ok(sameAsDefaults(composer, defaults));
  assert.equal(sameAsDefaults({ ...composer, format: "markdown" }, defaults), false);
  assert.equal(sameAsDefaults({ ...composer, ocrLang: "en" }, defaults), false);
  // An older backend's defaults leave the keys unset, and saving sends none of them.
  const { format: _f, ocr_lang: _l, ocr_all_pages: _a, page_markers: _p, ...older } = defaults;
  const plain = composerFromDefaults(older);
  assert.equal(plain.format, null);
  assert.equal(plain.ocrLang, null);
  assert.deepEqual(defaultsOf(plain), older);
  assert.ok(sameAsDefaults(plain, older));
});

test("Python's first-run warnings are notices of their own", () => {
  const groups = groupNotices([
    { itemKey: "a", itemName: "a.pdf", text: "First conversion since installing: the libraries conversions use load once (several seconds on macOS); later conversions start right away." },
    { itemKey: "b", itemName: "b.png", text: "First OCR use since installing: the OCR libraries and models load once (up to a minute on macOS); later OCR conversions start right away." },
  ]);
  assert.deepEqual(
    groups.map((group) => group.kind),
    ["first_conversion", "first_ocr"],
  );
});

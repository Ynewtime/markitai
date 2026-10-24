import test from "node:test";
import assert from "node:assert/strict";
import { groupNotices, isUsedRetry, rowWarnings } from "./notices.ts";
import { emptyOptions } from "./options.ts";
import { publicSettings, requestView, wrapUnits } from "./request.ts";

const rich = { ...emptyOptions(), preset: "rich", llm: true, alt: true, desc: true, screenshot: true, ocr: false, no_cache: false };

test("a job's settings read as its command line, the rest as chips", () => {
  const settings = publicSettings({
    "llm.enabled": true,
    "image.alt_enabled": true,
    "image.desc_enabled": true,
    "screenshot.enabled": true,
    "llm.keep_base": true,
    "cache.no_cache": true,
    "output.format": "json",
    "image.format": "webp",
    "image.quality": 85,
    "fetch.strategy": "static",
  });
  const view = requestView(["field notes.pdf"], rich, settings, false);
  assert.equal(view?.command, "markitai 'field notes.pdf' -o output/ --preset rich --keep-base -f json --no-cache");
  assert.deepEqual(view?.chips, [
    { key: "image.format", value: "webp" },
    { key: "image.quality", value: "85" },
  ]);
  // A switch the preset does not have spells the switches out instead of naming it.
  const ocr = requestView(["scan.png"], { ...rich, preset: "standard" }, { "llm.enabled": true, "ocr.enabled": true, "ocr.lang": "ja" }, false);
  assert.equal(ocr?.command, "markitai scan.png -o output/ --llm --ocr");
  assert.deepEqual(ocr?.chips, [{ key: "ocr.lang", value: "ja" }]);
  // The URL strategy is told for a URL only.
  assert.equal(requestView(["https://example.org/a"], null, { "fetch.strategy": "static" }, true)?.command, "markitai https://example.org/a -o output/ --strategy static");
  // Everything at its default: the bare command, which is what ran.
  assert.equal(requestView(["notes.txt"], emptyOptions(), {}, false)?.command, "markitai notes.txt -o output/");
});

test("without settings the request's own options are all that is shown", () => {
  const options = { ...emptyOptions(), preset: "minimal", llm: false, ocr: true, alt: false, desc: false, screenshot: false, no_cache: true, ocr_lang: "zh", ocr_all_pages: true };
  const view = requestView(["scan.pdf"], options, null, false);
  assert.equal(view?.command, "markitai scan.pdf -o output/ --ocr --no-cache");
  assert.deepEqual(view?.chips, [
    { key: "ocr.lang", value: "zh" },
    { key: "ocr.per_page_routing", value: "false" },
  ]);
  assert.equal(requestView(["scan.pdf"], null, null, false), null);
});

test("settings keep only dotted keys with plain values", () => {
  assert.deepEqual(publicSettings({ "image.format": "webp", "cache.no_cache": true, api_key: "x", "llm.model_list": [{}], "x.y": { a: 1 }, "bad key.z": 1 }), {
    "image.format": "webp",
    "cache.no_cache": true,
  });
  assert.equal(publicSettings(null), null);
  assert.equal(publicSettings([1]), null);
});

test("an answer requested again and used is no notice, in either wording", () => {
  const used = [
    "A model answer could not be used, so the request was sent again; a later answer was used",
    "The provider withheld a model answer as a recitation of published text, so the request was sent again; a later answer was used",
    "The model's JSON-text answer was not used (the answer dropped 3 of the 16 markers that keep the document's pages, pictures, links and code exactly as they are (“<u>” ⟦MKTI:0123456789abcdef:00000002⟧)); the request was sent again in JSON-text mode and a later answer was used",
    "The provider withheld the tool-call answer as a recitation of published text; the request was sent again in JSON-schema mode and a later answer was used",
  ];
  for (const text of used) assert.equal(isUsedRetry(text), true, text);
  // One that ended without an answer is still a warning.
  const failed = "LLM JSON-text answer was rejected (the answer dropped a marker); the request was sent again in JSON-text mode";
  assert.equal(isUsedRetry(failed), false);
  assert.deepEqual(rowWarnings([...used, failed]), [failed]);
  const groups = groupNotices([...used, failed].map((text) => ({ itemKey: "j/i1", itemName: "notes.pdf", text })));
  assert.deepEqual(groups.map((group) => group.kind), ["json_resent"]);
});

test("the command wraps between words, each flag kept with its value", () => {
  const view = requestView(["field notes.pdf"], null, { "output.profile": "obsidian", "cache.no_cache": true }, false);
  assert.deepEqual(view?.words, ["markitai", "'field notes.pdf'", "-o", "output/", "--profile", "obsidian", "--no-cache"]);
  assert.deepEqual(wrapUnits(view?.words ?? []), ["markitai", "'field notes.pdf'", "-o output/", "--profile obsidian", "--no-cache"]);
});

test("a request naming a preset this page does not know spells its switches out instead of crashing", () => {
  // An API job submitted with a configured preset the server's table does not list.
  const team = { ...emptyOptions(), preset: "team", llm: true, ocr: true, alt: false, desc: false, screenshot: false };
  assert.equal(requestView(["notes.pdf"], team, null, false)?.command, "markitai notes.pdf -o output/ --llm --ocr");
  assert.equal(
    requestView(["notes.pdf"], team, { "llm.enabled": true, "ocr.enabled": true, "image.format": "webp" }, false)?.command,
    "markitai notes.pdf -o output/ --llm --ocr",
  );
  // A CLI history entry (`markitai notes.pdf --preset team`): its options carry the preset, its settings what ran.
  const cli = { ...emptyOptions(), preset: "team" };
  const view = requestView(["notes.pdf"], cli, { "ocr.enabled": true, "ocr.lang": "ja" }, false);
  assert.equal(view?.command, "markitai notes.pdf -o output/ --ocr");
  assert.deepEqual(view?.chips, [{ key: "ocr.lang", value: "ja" }]);
  // A Python 1.2.1 job: a preset name and its switches, no settings.
  const python = { ...emptyOptions(), preset: "Team Notes", llm: false, ocr: false, alt: false, desc: false, screenshot: false };
  assert.equal(requestView(["notes.pdf"], python, null, false)?.command, "markitai notes.pdf -o output/");
  // A preset the server lists is named while the request equals it.
  const table = { team: { llm: true, ocr: true, alt: false, desc: false, screenshot: false } };
  assert.equal(requestView(["notes.pdf"], team, null, false, table)?.command, "markitai notes.pdf -o output/ --preset team");
});

test("the copied command quotes every value", () => {
  // A configured preset with a space, or one a shell would expand, stays one quoted word.
  const table = { "team notes": { llm: true, ocr: false, alt: false, desc: false, screenshot: false }, "$(x)": { llm: false, ocr: true, alt: false, desc: false, screenshot: false } };
  const team = { ...emptyOptions(), preset: "team notes", llm: true, ocr: false, alt: false, desc: false, screenshot: false };
  assert.equal(requestView(["a.pdf"], team, null, false, table)?.command, "markitai a.pdf -o output/ --preset 'team notes'");
  assert.equal(requestView(["a.pdf"], team, { "llm.enabled": true }, false, table)?.command, "markitai a.pdf -o output/ --preset 'team notes'");
  const odd = { ...emptyOptions(), preset: "$(x)", llm: false, ocr: true, alt: false, desc: false, screenshot: false };
  assert.equal(requestView(["a.pdf"], odd, null, false, table)?.command, "markitai a.pdf -o output/ --preset '$(x)'");
  // The flag keeps its quoted value when the line wraps.
  assert.deepEqual(wrapUnits(requestView(["a.pdf"], team, null, false, table)?.words ?? []).slice(-1), ["--preset 'team notes'"]);
});

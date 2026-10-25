import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ProviderCard } from "../api/types.ts";
import { BRAND_MARKS } from "../components/brand-marks.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { isKimiCard, KIMI_CARD_ID, kimiOrder, kimiServiceAt, kimiServices, mergeKimi } from "./kimi.ts";
import { arrangeProviders, providerLabel } from "./provider-catalog.ts";

const card = (provider: string, label: string, extra: Partial<ProviderCard> = {}): ProviderCard => ({
  id: `common:${provider}`,
  provider,
  label,
  kind: "common",
  status: "needs_credentials",
  source: "built_in",
  supports_discovery: true,
  ...extra,
});
// As the Rust server sends them (provider_management.rs).
const moonshot = card("moonshot", "Moonshot AI", {
  group: "makers",
  icon: "moonshot",
  aliases: ["kimi", "kimi platform", "kimi open platform"],
  default_base: "https://api.moonshot.ai/v1",
  key_variable: "MOONSHOT_API_KEY",
});
const kimiCode = card("kimi_code", "Kimi Code", {
  group: "makers",
  icon: "kimi",
  aliases: ["kimi for coding", "moonshot"],
  default_base: "https://api.kimi.com/coding/v1",
  key_variable: "KIMI_API_KEY",
});
const xai = card("xai", "xAI", { group: "makers", icon: "xai", aliases: ["grok"] });
const saved = (provider: string, id: string): ProviderCard => ({
  ...(provider === "moonshot" ? moonshot : kimiCode),
  id: `provider:${id}`,
  kind: "configured",
  source: "config",
  status: "ready",
  provider_id: id,
  api_key_configured: true,
  model_count: 1,
});
const arrange = (cards: ProviderCard[], query = "") => arrangeProviders(mergeKimi(cards), { query, filter: "all", label: (item) => providerLabel(en, item) });

test("Add models offers one Kimi card for the open platform and Kimi Code", () => {
  const view = arrange([moonshot, kimiCode, xai]);
  assert.deepEqual(
    view.all.map((choice) => choice.label),
    ["Kimi", "xAI"],
  );
  const kimi = view.all[0]!;
  assert.equal(kimi.card.id, KIMI_CARD_ID);
  assert.ok(isKimiCard(kimi.card));
  // Kimi's own mark; the open platform's address and variable stay the card's defaults.
  assert.equal(kimi.icon, "kimi");
  assert.equal(kimi.card.provider, "moonshot");
  // Every name either service was known by finds it.
  for (const query of ["kimi", "moonshot", "kimi code", "kimi platform", "kimi for coding"]) {
    assert.deepEqual(
      arrange([moonshot, kimiCode, xai], query).all.map((choice) => choice.label),
      ["Kimi"],
      query,
    );
  }
  // A server without Kimi Code (an older backend) keeps its Moonshot card, named as Kimi's platform.
  assert.deepEqual(
    arrange([moonshot, xai]).all.map((choice) => choice.label),
    ["Kimi Platform", "xAI"],
  );
  assert.equal(kimiServices([moonshot, xai]), null);
});

test("the Kimi card opens a saved connection only when both services have one", () => {
  // The server lists a provider with a saved connection by that connection only, without its built-in card.
  const one = arrange([moonshot, saved("kimi_code", "kc")]);
  // Saved rows stay one per connection, named by service.
  assert.deepEqual(
    one.saved.map((item) => providerLabel(zh, item)),
    ["Kimi Code"],
  );
  const kimi = one.all.find((choice) => choice.label === "Kimi")!;
  assert.equal(kimi.added, false);
  assert.equal(kimi.card.id, KIMI_CARD_ID);
  // No separate Kimi Code choice comes back from the saved row.
  assert.equal(one.all.filter((choice) => choice.provider === "kimi_code").length, 0);
  // A pasted key goes to Kimi Code as a new connection, never with the saved one's id.
  const services = kimiServices([moonshot, saved("kimi_code", "kc")])!;
  assert.equal(services.kimi_code.kind, "common");
  assert.equal(services.kimi_code.provider_id, undefined);
  assert.equal(services.kimi_code.api_key_configured, undefined);
  assert.equal(services.kimi_code.default_base, "https://api.kimi.com/coding/v1");
  const platform = arrange([kimiCode, saved("moonshot", "ms")]);
  assert.equal(platform.all.find((choice) => choice.label === "Kimi")?.added, false);
  const both = arrange([saved("kimi_code", "kc"), saved("moonshot", "ms"), xai]);
  const added = both.all.find((choice) => choice.label === "Kimi")!;
  assert.equal(added.added, true);
  assert.equal(added.card.provider_id, "ms");
  assert.deepEqual(
    both.saved.map((item) => providerLabel(zh, item)).sort(),
    ["Kimi Code", "Kimi 开放平台"],
  );
});

test("a pasted key goes first to the service its form names; a typed address picks its service", () => {
  assert.deepEqual(kimiOrder("sk-0123456789abcdef"), ["moonshot", "kimi_code"]);
  assert.deepEqual(kimiOrder("  sk-kimi-0123456789"), ["kimi_code", "moonshot"]);
  assert.equal(kimiServiceAt("https://api.kimi.com/coding/v1"), "kimi_code");
  assert.equal(kimiServiceAt("https://api.kimi.ai/coding/v1/"), "kimi_code");
  assert.equal(kimiServiceAt("https://api.moonshot.cn/v1"), "moonshot");
  // A gateway or another path is the open platform's protocol.
  assert.equal(kimiServiceAt("https://gateway.example.com/kimi/v1"), "moonshot");
  assert.equal(kimiServiceAt("https://api.kimi.com/v1"), "moonshot");
  assert.equal(kimiServiceAt("not a url"), "moonshot");
});

test("the Kimi page tries the services in turn and saves under the one that listed the models", () => {
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  // Each service is asked in the key's order, with the address only when one was typed.
  assert.match(page, /const order = typed \? \[kimiServiceAt\(typed\)\] : kimiOrder\(draftKey\);/);
  assert.match(page, /discoverModels\(discoveryRequest\(services\[next\], draftKey, typed, refresh\)\)/);
  // The next one only after a refusal.
  assert.match(page, /if \(listFailure\(result\)\?\.reason !== "key_rejected"\) break;/);
  // Saving, the picker and its rows read the service found.
  assert.match(page, /deploymentCredentials\(target, draftKey/);
  assert.match(page, /pickerRows\(target\.provider/);
  // The terms show for a membership key only.
  assert.match(page, /kimiFound\.service === "kimi_code" && <span>\{t\.kimiMemberTerms\}<\/span>/);
});

test("refusals and hints are short, and the Kimi texts exist in every language", () => {
  const both = ["api.moonshot.ai", "api.moonshot.cn"];
  assert.equal(en.discoveryFailHint("key_rejected", null, both, "moonshot"), "api.moonshot.ai and api.moonshot.cn refused the key. Check that it is correct and still active.");
  assert.equal(zh.discoveryFailHint("key_rejected", null, both, "moonshot"), "api.moonshot.ai 和 api.moonshot.cn 拒绝了这个密钥。请确认密钥正确且仍然有效。");
  for (const dict of [en, zh, ja]) {
    // A caller that names no host (an older page) reads as before.
    assert.equal(dict.discoveryFailHint("key_rejected", null), dict.discoveryFailHint("key_rejected", null, [], ""));
    // A Moonshot model's test names both regions' addresses; other providers' does not.
    const tested = dict.testFailHint("key_rejected", null, "moonshot/kimi-k3");
    assert.ok(tested.includes("https://api.moonshot.cn/v1") && tested.includes("https://api.moonshot.ai/v1"), tested);
    assert.ok(!dict.testFailHint("key_rejected", null, "openai/gpt-x").includes("moonshot"));
    assert.ok(dict.regionElsewhere("https://api.moonshot.cn/v1").includes("https://api.moonshot.cn/v1"));
    assert.notEqual(dict.kimiFound("kimi_code"), dict.kimiFound("moonshot"));
    assert.ok(dict.kimiMemberTerms.length < 90, dict.kimiMemberTerms);
    assert.ok(dict.kimiRefused.includes("Kimi Code"));
    assert.ok(dict.keyVariableHint(["MOONSHOT_API_KEY", "KIMI_API_KEY"]).includes("KIMI_API_KEY"));
  }
});

test("Kimi's brand mark is its own beside Moonshot's", () => {
  assert.ok(BRAND_MARKS.kimi && BRAND_MARKS.moonshot);
  assert.ok(BRAND_MARKS.kimi.paths[0] !== BRAND_MARKS.moonshot.paths[0]);
  assert.equal(BRAND_MARKS.kimi.viewBox, "0 0 24 24");
});

test("Refresh on the Kimi card keeps sending the key to the address typed by hand", async () => {
  const { kimiTypedBase } = await import("./kimi.ts");
  // first Load models with a typed gateway: that address
  assert.equal(kimiTypedBase("https://gateway.example.test/v1", null), "https://gateway.example.test/v1");
  // it listed the models and fills the field: Refresh uses it again, never the built-in hosts
  assert.equal(kimiTypedBase("https://gateway.example.test/v1", { base: "https://gateway.example.test/v1", typed: true }), "https://gateway.example.test/v1");
  // an address the page filled in after finding the service: Refresh searches again
  assert.equal(kimiTypedBase("https://api.moonshot.cn/v1", { base: "https://api.moonshot.cn/v1", typed: false }), "");
  // the reader changed the found address: that address
  assert.equal(kimiTypedBase(" https://other.example.test/v1 ", { base: "https://api.moonshot.cn/v1", typed: false }), "https://other.example.test/v1");
  assert.equal(kimiTypedBase("  ", null), "");
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  assert.match(page, /const typed = kimiTypedBase\(draftBase, kimiFound\);/);
  assert.match(page, /setKimiFound\(\{ service, base, typed: typed !== "" \}\);/);
});

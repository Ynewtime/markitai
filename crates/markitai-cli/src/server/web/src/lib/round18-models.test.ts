// Round 18, model settings: the routing weight range (0–100), one atomic save
// for additions and removals on a provider page, and saved models a provider's
// list names another way (an alias) or not at all. Invented models only.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { modelsApi } from "../api/client.ts";
import { setToken } from "../api/token.ts";
import type { Deployment, ModelCandidate } from "../api/types.ts";
import { en } from "../i18n/en.ts";
import { apiErrorText } from "../i18n/errors.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { aliasTarget, parseWeight, pickerRows, removals, stepWeight, WEIGHT_MAX, WEIGHT_PLACEHOLDER } from "./models.ts";

const deployment = (model: string, extra: Partial<Deployment> = {}): Deployment => ({
  deployment_id: `id-${model}`,
  routing_group: "default",
  model,
  weight: 1,
  api_key_configured: true,
  api_base_configured: false,
  api_base: null,
  persisted: true,
  ...extra,
});
const candidate = (model: string): ModelCandidate => ({ model, label: model.split("/").slice(1).join("/"), supports_vision: false });

test("a routing weight is a whole number from 0 to 100", () => {
  assert.equal(WEIGHT_MAX, 100);
  for (const [text, value] of [["0", 0], [" 7 ", 7], ["100", 100], ["050", 50]] as const) assert.equal(parseWeight(text), value, text);
  for (const bad of ["", " ", "-1", "1.5", "1e2", "101", "999", "1000", "abc", "２", "+5"]) assert.equal(parseWeight(bad), null, bad);
  // ArrowUp/ArrowDown step from the field's number (or the saved value) and stay within 0–100.
  assert.equal(stepWeight("100", 3, 1), 100);
  assert.equal(stepWeight("0", 3, -1), 0);
  assert.equal(stepWeight("250", 3, -1), 100);
  assert.equal(stepWeight("abc", 3, 1), 4);
  assert.equal(stepWeight("", 3, -1), 2);
  // The empty field's placeholder: the same digits and en dash in every language.
  assert.equal(WEIGHT_PLACEHOLDER, "0–100");
});

test("the weight messages and the saved-model notes read in every language", () => {
  for (const dict of [en, zh, ja]) {
    assert.match(dict.weightInvalid, /100/);
    assert.match(dict.weightKeysHint, /100/);
    assert.match(dict.weightRangeHint, /100/);
    assert.ok(dict.savedNotListed.length > 4);
    assert.match(dict.savedAsAlias("claude-agent/sonnet"), /claude-agent\/sonnet/);
  }
  assert.equal(en.savedNotListed, "Saved, not in the provider's current list");
  for (const locale of ["en", "zh", "ja"] as const) {
    const text = apiErrorText(locale, { detail: "Routing weight must be a whole number from 0 to 100", code: "invalid_request", reason: "invalid_weight" });
    assert.ok(text && text.text !== "Routing weight must be a whole number from 0 to 100" && /100/.test(text.text), locale);
  }
});

test("the form's weight field and the pill take 0–100 only", () => {
  const read = (name: string) => readFileSync(new URL(`../components/${name}`, import.meta.url), "utf8");
  const picker = read("model-picker.tsx");
  assert.match(picker, /max=\{WEIGHT_MAX\}/);
  assert.match(picker, /step=\{1\}/);
  assert.match(picker, /placeholder=\{WEIGHT_PLACEHOLDER\}/);
  assert.match(read("weight-pill.tsx"), /maxLength=\{3\}/);
  // Save waits for a valid weight.
  assert.match(read("settings-models.tsx"), /disabled=\{addBusy \|\| !weightValid/);
});

test("Claude's aliases and undated IDs stand for the listed model they resolve to", () => {
  const listed = [
    "claude-agent/claude-haiku-5-5",
    "claude-agent/claude-sonnet-5-5",
    "claude-agent/claude-sonnet-4-6",
    "claude-agent/claude-opus-5-1",
    "claude-agent/claude-3-5-haiku-20241022",
  ];
  assert.equal(aliasTarget("claude-agent/sonnet", listed), "claude-agent/claude-sonnet-5-5");
  assert.equal(aliasTarget("claude-agent/inherit", listed), "claude-agent/claude-sonnet-5-5");
  assert.equal(aliasTarget("claude-agent/opus", listed), "claude-agent/claude-opus-5-1");
  assert.equal(aliasTarget("claude-agent/haiku", listed), "claude-agent/claude-haiku-5-5");
  assert.equal(aliasTarget("claude-agent/sonnet[1m]", listed), "claude-agent/claude-sonnet-5-5");
  // No alias for another provider, an unknown alias, or a family the list lacks.
  assert.equal(aliasTarget("claude-agent/default", listed), null);
  assert.equal(aliasTarget("copilot/sonnet", ["copilot/claude-sonnet-5-5"]), null);
  assert.equal(aliasTarget("claude-agent/opus", ["claude-agent/claude-haiku-5-5"]), null);
  // An undated Anthropic ID and its newest dated snapshot.
  assert.equal(
    aliasTarget("anthropic/claude-sonnet-4-5", ["anthropic/claude-sonnet-4-5-20250101", "anthropic/claude-sonnet-4-5-20250929", "anthropic/claude-opus-4-5-20251101"]),
    "anthropic/claude-sonnet-4-5-20250929",
  );
  assert.equal(aliasTarget("anthropic/claude-sonnet-4-5-20250929", ["anthropic/claude-sonnet-4-5"]), null);
});

test("every saved model appears in its provider's list, ticked: by ID, by alias, or as a row of its own", () => {
  const saved = [
    deployment("claude-agent/sonnet"),
    deployment("claude-agent/claude-haiku-5-5"),
    deployment("claude-agent/claude-retired-1"),
    // Another group, and another provider: not this page's.
    deployment("claude-agent/opus", { routing_group: "fast" }),
    deployment("openai/garden-model"),
  ];
  const rows = pickerRows(
    "claude-agent",
    [candidate("claude-agent/claude-haiku-5-5"), candidate("claude-agent/claude-sonnet-5-5"), candidate("claude-agent/claude-opus-5-1")],
    saved,
    "default",
    "",
    null,
  );
  assert.deepEqual(
    rows.map((row) => [row.model, row.saved.map((item) => item.model), row.aliases, row.unlisted]),
    [
      ["claude-agent/claude-haiku-5-5", ["claude-agent/claude-haiku-5-5"], [], false],
      ["claude-agent/claude-sonnet-5-5", ["claude-agent/sonnet"], ["claude-agent/sonnet"], false],
      ["claude-agent/claude-retired-1", ["claude-agent/claude-retired-1"], [], true],
      ["claude-agent/claude-opus-5-1", [], [], false],
    ],
  );
  // Unticking a row removes what it stands for, the alias included.
  assert.deepEqual(removals(rows, new Set(["claude-agent/claude-sonnet-5-5", "claude-agent/claude-retired-1", "claude-agent/claude-opus-5-1"])), [
    "id-claude-agent/sonnet",
    "id-claude-agent/claude-retired-1",
  ]);
});

test("a saved model the list lacks shows only on the page of its own endpoint", () => {
  const saved = [
    deployment("openai/at-default"),
    deployment("openai/at-gateway", { api_base: "https://gateway.example" }),
  ];
  // The OpenAI page at its documented address: the one without an address of its own.
  const openai = pickerRows("openai", [], saved, "default", "", "https://api.openai.com/v1");
  assert.deepEqual(openai.map((row) => [row.model, row.unlisted]), [["openai/at-default", true]]);
  // An OpenAI-compatible endpoint's page (a saved connection reads its origin): its own model only.
  const gateway = pickerRows("custom", [candidate("openai/listed")], saved, "default", "https://gateway.example", null);
  assert.deepEqual(gateway.map((row) => row.model), ["openai/at-gateway", "openai/listed"]);
});

test("Save sends additions and removals as one batch under the page's revision", async () => {
  setToken("t-admin", null);
  const original = globalThis.fetch;
  const sent: { url: string; method: string; body: unknown }[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: new URL(String(url)).pathname, method: init?.method ?? "GET", body: init?.body ? JSON.parse(String(init.body)) : null });
    return new Response(JSON.stringify({ revision: "r2", deployments: [], detected: [], batch_remove: true }), { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    const add = { model_name: "default", model: "openai/garden-new", weight: 1, credential_deployment_id: "legacy-anchor" };
    await modelsApi("admin").addDeployments("r1", [add], ["legacy-anchor"]);
    await modelsApi("user").addDeployments("r1", [], ["d2"]);
    await modelsApi("user").addDeployments("r1", [add]);
    assert.deepEqual(
      sent.map((request) => [request.method, request.url, request.body]),
      [
        ["POST", "/api/settings/llm/deployments/batch", { expected_revision: "r1", deployments: [add], remove: ["legacy-anchor"] }],
        ["POST", "/api/me/models/deployments/batch", { expected_revision: "r1", remove: ["d2"] }],
        ["POST", "/api/me/models/deployments/batch", { expected_revision: "r1", deployments: [add] }],
      ],
    );
  } finally {
    globalThis.fetch = original;
    setToken("", null);
  }
  // The page uses the single request whenever the backend says it takes `remove`.
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  assert.match(page, /settings\?\.batch_remove === true[^\n]*\n[^\n]*\n\s*accept\(await addDeployments\(revision, deployments, remove\)\)/);
});

test("Kimi #6: a Moonshot model saved at one region is not this other region's, and an explicit one matches its own", () => {
  const saved = [
    deployment("moonshot/kimi-k3"),
    // the server reports a saved address by its origin only
    deployment("moonshot/kimi-k3-cn", { api_base: "https://api.moonshot.cn" }),
  ];
  const listed = [candidate("moonshot/kimi-k3"), candidate("moonshot/kimi-k3-cn")];
  const china = pickerRows("moonshot", listed, saved, "default", "https://api.moonshot.cn/v1", "https://api.moonshot.ai/v1");
  // on the China page: the .cn model is configured, the default-region one is not (unticking it must not delete it)
  assert.deepEqual(china.map((row) => [row.model, row.saved.map((item) => item.model)]), [
    ["moonshot/kimi-k3-cn", ["moonshot/kimi-k3-cn"]],
    ["moonshot/kimi-k3", []],
  ]);
  assert.deepEqual(removals(china, new Set(["moonshot/kimi-k3"])), []);
  // on the default page: the other way round
  const global = pickerRows("moonshot", listed, saved, "default", "https://api.moonshot.ai/v1", "https://api.moonshot.ai/v1");
  assert.deepEqual(global.map((row) => [row.model, row.saved.map((item) => item.model)]), [
    ["moonshot/kimi-k3", ["moonshot/kimi-k3"]],
    ["moonshot/kimi-k3-cn", []],
  ]);
});

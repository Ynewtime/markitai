import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { zh } from "../i18n/zh.ts";
import { cardEndpoints, kimiEndpoints, matchingChoice } from "./endpoints.ts";

const zai = {
  default_base: "https://api.z.ai/api/paas/v4",
  regions: ["https://api.z.ai/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4"],
  endpoints: [
    { base: "https://api.z.ai/api/paas/v4", region: "global" },
    { base: "https://open.bigmodel.cn/api/paas/v4", region: "china" },
    { base: "https://api.z.ai/api/coding/paas/v4", region: "global", plan: "coding" },
    { base: "https://open.bigmodel.cn/api/coding/paas/v4", region: "china", plan: "coding" },
  ],
};

test("the address list is the card's endpoints, else its regions, else its default address", () => {
  const choices = cardEndpoints(zai);
  assert.equal(choices.length, 4);
  assert.deepEqual(choices[2], { base: "https://api.z.ai/api/coding/paas/v4", region: "global", plan: "coding" });
  assert.deepEqual(
    choices.map((choice) => zh.endpointLabel(choice.region, choice.plan)),
    ["国际", "中国大陆", "国际 · Coding Plan", "中国大陆 · Coding Plan"],
  );
  // An older server (v1.25) names regions only: the default, then each by its host.
  const older = cardEndpoints({ default_base: zai.default_base, regions: zai.regions });
  assert.deepEqual(
    older.map((choice) => en.endpointLabel(choice.region, choice.plan)),
    ["Default", "open.bigmodel.cn"],
  );
  // One address (Python, or a provider with one endpoint): that one, as the default.
  assert.deepEqual(cardEndpoints({ default_base: "https://api.openai.com/v1" }), [{ base: "https://api.openai.com/v1", region: null, plan: null }]);
  // No address (a custom endpoint, vLLM): nothing to list.
  assert.deepEqual(cardEndpoints({ default_base: null }), []);
  assert.deepEqual(cardEndpoints(null), []);
  // Model Studio's regions are named by place; a Kimi site as written.
  assert.equal(en.endpointLabel("us_virginia", null), "US (Virginia)");
  assert.equal(zh.endpointLabel("hong_kong", null), "中国香港");
  assert.equal(zh.endpointLabel("kimi.com", null, "kimi_code"), "Kimi Code · kimi.com");
});

test("the Kimi card lists the open platform's addresses, then Kimi Code's", () => {
  const moonshot = { default_base: "https://api.moonshot.ai/v1", endpoints: [{ base: "https://api.moonshot.ai/v1", region: "global" }, { base: "https://api.moonshot.cn/v1", region: "china" }] };
  const code = { default_base: "https://api.kimi.com/coding/v1", endpoints: [{ base: "https://api.kimi.com/coding/v1", region: "kimi.com" }, { base: "https://api.kimi.ai/coding/v1", region: "kimi.ai" }] };
  const choices = kimiEndpoints(moonshot, code);
  assert.deepEqual(
    choices.map((choice) => zh.endpointLabel(choice.region, choice.plan, choice.service)),
    ["Kimi 开放平台 · 国际", "Kimi 开放平台 · 中国大陆", "Kimi Code · kimi.com", "Kimi Code · kimi.ai"],
  );
});

test("a typed address matches a choice whatever its trailing slash, and nothing else", () => {
  const choices = cardEndpoints(zai);
  assert.equal(matchingChoice(choices, " https://api.z.ai/api/coding/paas/v4/ ")?.plan, "coding");
  assert.equal(matchingChoice(choices, "https://api.z.ai/api/paas"), undefined);
  assert.equal(matchingChoice(choices, ""), undefined);
});

test("the address field is a combobox of its own, not a datalist, in both forms", () => {
  const field = readFileSync(new URL("../components/address-field.tsx", import.meta.url), "utf8");
  assert.ok(!/<datalist\s/.test(field));
  assert.match(field, /role="combobox"/);
  assert.match(field, /role="listbox"/);
  assert.match(field, /event\.key === "Escape" && shown[\s\S]*?stopPropagation/);
  assert.match(field, /onMouseDown=\{\(event\) => event\.preventDefault\(\)\}/);
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  assert.equal(page.match(/<AddressField/g)?.length, 2);
  // A Coding Plan address says its terms under the field.
  assert.match(page, /matchingChoice\(endpointChoices, draftBase\)\?\.plan && \(/);
});

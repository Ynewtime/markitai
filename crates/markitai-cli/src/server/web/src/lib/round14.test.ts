import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { MODELS_BASE, modelsApi } from "../api/client.ts";
import { setToken } from "../api/token.ts";
import { en } from "../i18n/en.ts";
import { apiErrorText } from "../i18n/errors.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { modelName } from "./models.ts";
import { ADMIN_SECTIONS, parseRoute } from "./route.ts";

test("a user token's Models page calls the same routes under /api/me/models", async () => {
  assert.deepEqual(MODELS_BASE, { admin: "/api/settings/llm", user: "/api/me/models" });
  setToken("t-user", null);
  const original = globalThis.fetch;
  const sent: { url: string; method: string; body: unknown }[] = [];
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: new URL(String(url)).pathname + new URL(String(url)).search, method: init?.method ?? "GET", body: init?.body ? JSON.parse(String(init.body)) : null });
    return new Response(JSON.stringify({ providers: [], revision: "r2" }), { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    for (const scope of ["admin", "user"] as const) {
      sent.length = 0;
      const api = modelsApi(scope);
      const base = MODELS_BASE[scope];
      await api.fetchSettings();
      await api.fetchProviders(true);
      await api.fetchCredentials("p 1");
      await api.discoverModels({ provider: "openai" });
      await api.probeModel({ deployment_id: "d1" });
      await api.addDeployments("r1", [{ model_name: "default", model: "openai/gpt-x", weight: 1 }]);
      await api.updateDeployment("d1", { expected_revision: "r1", share: "shared" });
      await api.deleteDeployment("d1", "r1");
      await api.updateProvider("p1", { expected_revision: "r1", api_key: "k" });
      await api.deleteProvider("p1", "r1");
      assert.deepEqual(
        sent.map((request) => `${request.method} ${request.url}`),
        [
          `GET ${base}`,
          `GET ${base}/providers?refresh=true`,
          `GET ${base}/providers/p%201/credentials`,
          `POST ${base}/model-discovery`,
          `POST ${base}/test`,
          `POST ${base}/deployments/batch`,
          `PATCH ${base}/deployments/d1`,
          `DELETE ${base}/deployments/d1?expected_revision=r1`,
          `PATCH ${base}/providers/p1`,
          `DELETE ${base}/providers/p1?expected_revision=r1`,
        ],
        scope,
      );
      assert.deepEqual(sent[6]?.body, { expected_revision: "r1", share: "shared" });
    }
  } finally {
    globalThis.fetch = original;
    setToken("", null);
  }
});

test("shared models show their name without the provider prefix", () => {
  assert.equal(modelName("openai/gpt-5-mini"), "gpt-5-mini");
  assert.equal(modelName("openrouter/meta/llama-4"), "meta/llama-4");
  assert.equal(modelName("plain-model"), "plain-model");
});

test("the user-model refusals read in every language", () => {
  for (const reason of ["user_token_required", "server_credentials", "key_or_address_required", "server_address", "model_limit"]) {
    for (const locale of ["en", "zh", "ja"] as const) {
      const text = apiErrorText(locale, { detail: "service words", code: "invalid_request", reason });
      assert.ok(text && text.text && text.text !== "service words", `${locale} ${reason}`);
    }
  }
});

test("revoking says what happens to the token's models, in every language", () => {
  for (const dict of [en, zh, ja]) {
    assert.match(dict.userTokenRevokeText(3), /3/);
    assert.notEqual(dict.userTokenRevokeText(0), dict.userTokenRevokeText(3));
    assert.match(dict.userTokenModels(2), /2/);
  }
  assert.match(en.userTokenRevokeText(1), /its 1 model is deleted with their keys|1 model is deleted/);
  // The Convert page says where models are added.
  assert.match(en.advNeedsModel, /Settings → Models/);
  assert.match(zh.advNeedsModel, /设置 → 模型/);
  // A user token on a backend without its own models is told to ask the admin.
  for (const dict of [en, zh, ja]) assert.ok(dict.advNeedsModelAdmin && dict.advNeedsModelAdmin !== dict.advNeedsModel);
  // The options dialog dims LLM and says so under it, with the way to Settings → Models when there is one.
  const dialog = readFileSync(new URL("../components/options-dialog.tsx", import.meta.url), "utf8");
  assert.match(dialog, /sub=\{llmReady \? t\.helpLlm : noModel\}/);
  assert.match(dialog, /\{onAddModel \? t\.noModel : short\(needsModel\)\}/);
});

test("Models stays an admin-guarded address while the caller is unknown, and a user has its own", () => {
  assert.deepEqual(parseRoute("/settings/models"), { view: "settings", section: "models" });
  assert.ok(ADMIN_SECTIONS.has("models"));
  const app = readFileSync(new URL("../app.tsx", import.meta.url), "utf8");
  // v1.16: a user's Edit config (its own models) joins them when the backend has it.
  assert.match(app, /userModels\s+\? \["appearance", "models", \.\.\.\(features\?\.user_config === true \? \(\["config"\] as const\) : \[\]\)\]\s+: \["appearance"\]/);
  assert.match(app, /scope=\{role === "user" \? "user" : "admin"\}/);
  assert.match(app, /canShare=\{role === "admin" && userModels\}/);
});

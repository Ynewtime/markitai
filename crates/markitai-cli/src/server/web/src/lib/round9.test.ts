import assert from "node:assert/strict";
import test from "node:test";
import type { ProviderCard } from "../api/types.ts";
import { listFailure } from "./model-test.ts";
import { readFull, storeFull } from "./peek-pref.ts";
import { arrangeProviders, atOwnEndpoint, savedAuth } from "./provider-catalog.ts";
import { providerDraft } from "./provider-credentials.ts";
import { qrModules, qrPath } from "./qr.ts";

const memory = (initial: Record<string, string> = {}) => {
  const values = new Map(Object.entries(initial));
  return { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => void values.set(key, value), values };
};

test("full screen is remembered per browser and the removed docked layout opens as the window", () => {
  const docked = memory({ "markitai.peek": "docked" });
  assert.equal(readFull(docked), false);
  assert.equal(docked.values.get("markitai.peek"), "window");
  const store = memory();
  assert.equal(readFull(store), false);
  storeFull(true, store);
  assert.equal(readFull(store), true);
  storeFull(false, store);
  assert.equal(store.values.get("markitai.peek"), "window");
  assert.equal(readFull(null), false);
});

test("a QR code has the version's size, its three finder patterns and timing lines", () => {
  const link = `http://192.168.1.81:8765/#token=${"a".repeat(64)}`;
  const modules = qrModules(link)!;
  // 96 bytes at level M need version 6: 41 modules.
  assert.equal(modules.length, 41);
  for (const [x, y] of [
    [0, 0],
    [34, 0],
    [0, 34],
  ] as const) {
    for (let i = 0; i < 7; i++) {
      assert.ok(modules[y]![x + i] && modules[y + 6]![x + i] && modules[y + i]![x] && modules[y + i]![x + 6], "finder ring");
      assert.equal(modules[y + 1]![x + 1], false);
    }
    assert.ok(modules[y + 3]![x + 3], "finder centre");
  }
  for (let i = 8; i < 33; i++) assert.equal(modules[6]![i], i % 2 === 0);
  assert.equal(qrModules("hi")!.length, 21);
  assert.equal(qrModules("x".repeat(3000)), null);
  assert.match(qrPath([[true, true, false, true]], 0), /^M0 0h2v1h-2zM3 0h1v1h-1z$/);
});

const card = (extra: Partial<ProviderCard> & Pick<ProviderCard, "id" | "provider" | "kind">): ProviderCard => ({
  label: extra.provider,
  status: "ready",
  source: "config",
  supports_discovery: true,
  ...extra,
});

test("a provider is one saved row: the server's key folds into its connection at its own endpoint", () => {
  // As Python 1.3.0 lists the owner's OpenAI: the environment card, the connection made from it, the Common tile.
  const environment = card({ id: "env:openai", provider: "openai", kind: "environment", source: "OPENAI_API_KEY", credential: "env:OPENAI_API_KEY" });
  const connection = card({ id: "provider:p1", provider_id: "p1", provider: "openai", kind: "configured", api_base: "https://api.openai.com", default_base: "https://api.openai.com/v1", model_count: 1 });
  const proxy = card({ id: "provider:p2", provider_id: "p2", provider: "openai", kind: "configured", api_base: "https://proxy.example", api_key_configured: true });
  const common = card({ id: "common:openai", provider: "openai", kind: "common", source: "built_in" });
  const deepseek = card({ id: "env:deepseek", provider: "deepseek", kind: "environment", source: "DEEPSEEK_API_KEY" });
  const view = arrangeProviders([environment, connection, proxy, common, deepseek], { query: "", filter: "all", label: (item) => item.label });
  assert.deepEqual(
    view.saved.map((item) => item.id),
    ["provider:p1", "provider:p2", "env:deepseek"],
  );
  assert.equal(view.folded.get("provider:p1"), environment);
  assert.deepEqual(savedAuth(connection, environment), { kind: "environment", variable: "OPENAI_API_KEY" });
  assert.deepEqual(savedAuth(proxy), { kind: "key" });
  // The Common tile is marked and opens the saved connection.
  const tile = view.common.find((choice) => choice.provider === "openai")!;
  assert.equal(tile.added, true);
  assert.equal(tile.card, connection);
  assert.equal(view.common.find((choice) => choice.provider === "deepseek")?.added, true);
  assert.ok(atOwnEndpoint(connection));
  assert.ok(!atOwnEndpoint(proxy));
  // A connection the card names as served by a key variable says so on its own.
  assert.deepEqual(savedAuth(card({ id: "provider:legacy:d1", provider: "deepseek", kind: "configured", credential: "env:DEEPSEEK_API_KEY" })), { kind: "environment", variable: "DEEPSEEK_API_KEY" });
  // A model written by hand without a key at another address still uses the server's key variable.
  const handWritten = card({ id: "provider:legacy:d2", provider_id: "legacy:d2", provider: "deepseek", kind: "configured", api_base: "http://127.0.0.1:9990" });
  const both = arrangeProviders([deepseek, handWritten], { query: "", filter: "all", label: (item) => item.label });
  assert.deepEqual(both.saved.map((item) => item.id), ["env:deepseek", "provider:legacy:d2"]);
  assert.deepEqual(savedAuth(handWritten, null, both.variables.get("deepseek")), { kind: "environment", variable: "DEEPSEEK_API_KEY" });
  // A saved keyless connection (the placeholder) has no key.
  assert.deepEqual(savedAuth(card({ id: "provider:k1", provider_id: "k1", provider: "deepseek", kind: "configured" }), null, "DEEPSEEK_API_KEY"), { kind: "none" });
  // The OpenAI-compatible endpoint is never "added": each one is another server.
  const custom = card({ id: "provider:c1", provider_id: "c1", provider: "custom", kind: "configured", api_base: "http://127.0.0.1:8000" });
  const tiles = arrangeProviders([custom, card({ id: "common:custom", provider: "custom", kind: "common" })], { query: "", filter: "all", label: (item) => item.label });
  assert.equal(tiles.all.find((choice) => choice.provider === "custom")?.added, false);
});

test("a failed model list says why, from the server's reason or Python's text", () => {
  assert.deepEqual(listFailure({ status: "unavailable", detail: "Model discovery returned HTTP 401: Authentication Fails", reason: "key_rejected" }), {
    reason: "key_rejected",
    variable: null,
    detail: "Model discovery returned HTTP 401: Authentication Fails",
  });
  assert.equal(listFailure({ status: "unavailable", detail: "DEEPSEEK_API_KEY is not set on the server, so nothing was sent", reason: "no_key", key_variable: "DEEPSEEK_API_KEY" })?.variable, "DEEPSEEK_API_KEY");
  assert.equal(listFailure({ status: "unavailable", detail: "model discovery failed (HTTPStatusError)" })?.reason, "provider_error");
  assert.equal(listFailure({ status: "unavailable", detail: "This provider does not publish a model list; enter model IDs manually" }), null);
  assert.equal(listFailure({ status: "partial", detail: "x", reason: "timeout" }), null);
});

test("a hand-written connection served by the server's key shows that key, not Not set", () => {
  const credentials = { api_key: null, api_key_masked: null, api_base: null, api_base_placeholder: "https://api.deepseek.com" };
  assert.deepEqual(providerDraft(credentials).key, { state: "none" });
  assert.deepEqual(providerDraft(credentials, "env:DEEPSEEK_API_KEY").key, { state: "environment", variable: "DEEPSEEK_API_KEY", stored: false });
  // A stored key wins over the card's note.
  assert.equal(providerDraft({ ...credentials, api_key_masked: "sk…7890" }, "env:DEEPSEEK_API_KEY").key.state, "saved");
});

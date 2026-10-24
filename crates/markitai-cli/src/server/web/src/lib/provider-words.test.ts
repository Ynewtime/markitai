import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { providerWords, regionBlocked } from "./model-test.ts";

test("the provider's message opens only with the provider's or the network's own words", () => {
  // The provider's HTTP answer: its status with its message or error code.
  assert.deepEqual(providerWords("Model discovery returned HTTP 401: Invalid Authentication", "key_rejected", "Key rejected"), {
    kind: "provider",
    text: "HTTP 401: Invalid Authentication",
  });
  assert.deepEqual(providerWords("Model connection returned HTTP 401 (invalid_request_error/invalid_api_key)", "key_rejected"), {
    kind: "provider",
    text: "HTTP 401 (invalid_request_error/invalid_api_key)",
  });
  // A bare status adds something only where the heading names none.
  assert.deepEqual(providerWords("Model discovery returned HTTP 404", "provider_error"), { kind: "provider", text: "HTTP 404" });
  assert.equal(providerWords("Model discovery returned HTTP 401", "key_rejected"), null);
  assert.equal(providerWords("Model discovery returned HTTP 429", "rate_limited"), null);
  // The network stack's account of a failed connection, under its own label.
  assert.deepEqual(providerWords("Model discovery request failed: connection refused", "network"), { kind: "connection", text: "connection refused" });
  assert.deepEqual(providerWords("Model connection request failed: TLS handshake failed (invalid peer certificate: UnknownIssuer)", "network"), {
    kind: "connection",
    text: "TLS handshake failed (invalid peer certificate: UnknownIssuer)",
  });
  // An older backend's text in no known shape stays the provider's.
  assert.deepEqual(providerWords("model discovery failed (AuthenticationError: bad key)", "key_rejected"), {
    kind: "provider",
    text: "model discovery failed (AuthenticationError: bad key)",
  });
});

test("markitai's own sentences and a repeated heading open nothing", () => {
  for (const [detail, reason] of [
    ["Claude CLI is not signed in", "not_signed_in"],
    ["Claude Code CLI is not signed in", "not_signed_in"],
    ["Codex CLI is not signed in with a ChatGPT subscription", "not_signed_in"],
    ["Claude CLI is authenticated through a non-subscription or unrecognized method", "not_signed_in"],
    ["GitHub Copilot CLI was not found on the server's PATH", "runtime_missing"],
    ["Claude Code CLI 2.0.1 is installed, but the adapter needs 2.1.284", "runtime_unsupported"],
    ["Claude returned an invalid or conflicting protocol response", "runtime_failed"],
    ["Model discovery timed out", "timeout"],
    ["Model connection test timed out", "timeout"],
    ["Model discovery could not complete", "provider_error"],
    ["Model discovery request failed", "network"],
    ["Model discovery request failed: the address redirected (HTTP 302), which discovery does not follow", "network"],
    ["DEEPSEEK_API_KEY is not set on the server, so nothing was sent", "no_key"],
    ["MOONSHOT_API_KEY is not set on the server", "no_key"],
    ["Model discovery needs an API key, and none is set, so nothing was sent", "no_key"],
    ["This model needs an API key, and none is set", "no_key"],
    ["Model discovery response is not JSON", "provider_error"],
    ["Model discovery response has no model list", "provider_error"],
    ["Model discovery exceeds 1000 records", "provider_error"],
    ["Too many model discovery requests are active", "provider_error"],
    ["Model credentials or endpoint configuration are invalid or unavailable", "provider_error"],
    ["This model provider is not supported by the native runtime", "provider_error"],
    ["Model connection returned an invalid completion", "provider_error"],
    ["Model connection returned HTTP 429: the account's quota or billing does not allow this request", "rate_limited"],
    ["Model connection returned HTTP 404: the model is unavailable", "model_not_found"],
    ["Model connection returned HTTP 403: the model is not available in this region", "provider_error"],
  ] as const)
    assert.equal(providerWords(detail, reason), null, detail);
  for (const dict of [en, zh, ja]) assert.equal(providerWords(dict.testFailTitle("timeout"), "timeout", dict.testFailTitle("timeout")), null);
  assert.equal(providerWords("", "provider_error"), null);
  // The region refusal is said in the test's own text instead.
  assert.ok(regionBlocked("Model connection returned HTTP 403: the model is not available in this region"));
  assert.ok(!regionBlocked("Model connection returned HTTP 403: Forbidden"));
  for (const dict of [en, zh, ja]) {
    assert.ok(!dict.testFailHint("provider_error", null, "openai/x", false).includes(dict === en ? "below" : dict === zh ? "下方" : "下に"));
    assert.ok(dict.testFailHint("provider_error", null, "openai/x").length > dict.testFailHint("provider_error", null, "openai/x", false).length);
  }
});

test("a runtime notice shows no provider's message and the list card shows only the provider's words", () => {
  const notice = readFileSync(new URL("../components/runtime-notice.tsx", import.meta.url), "utf8");
  assert.ok(!notice.includes("testFailDetail") && !notice.includes("<details"));
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  assert.ok(!page.includes("<p>{discovery.detail}</p>"));
  assert.ok(page.includes("providerWords(discovery.detail") && page.includes("providerWords(failure.detail"));
});

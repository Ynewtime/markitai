import assert from "node:assert/strict";
import test from "node:test";
import type { Deployment, ProviderCard } from "../api/types.ts";
import { CredentialInputError, deploymentCredentials, detectedDeployment, discoveryRequest, providerDraft, providerUpdate } from "./provider-credentials.ts";

const common: ProviderCard = {
  id: "common:openai", provider: "openai", label: "OpenAI", kind: "common",
  status: "available", source: "catalogue", supports_discovery: true,
};

const rejected = (reason: string) => (error: unknown) => error instanceof CredentialInputError && error.reason === reason;

test("selecting an environment card then loading and adding uses only its approved identifier", () => {
  const card = { ...common, kind: "environment", credential: "env:OPENAI_API_KEY" };
  // Even a stale draft cannot redirect a server-held credential or echo its reference.
  assert.deepEqual(discoveryRequest(card, "env:AWS_SECRET_ACCESS_KEY", "https://other.invalid/v1", false), {
    provider: "openai", provider_id: "env:openai", refresh: false,
  });
  assert.deepEqual(deploymentCredentials(card, card.credential, "https://other.invalid/v1"), {
    provider: "openai", credential_provider_id: "env:openai",
  });
});

test("saved and legacy deployment cards retain their connection identifiers without draft credentials", () => {
  const saved = { ...common, kind: "configured", provider_id: "p123" };
  const legacy = { ...common, kind: "configured", provider_id: "legacy:d456", deployment_id: "d456" };
  assert.deepEqual(discoveryRequest(saved, "stale-key", "https://other.invalid", true), {
    provider: "openai", provider_id: "p123", refresh: true,
  });
  assert.deepEqual(deploymentCredentials(legacy, "stale-key", "https://other.invalid"), {
    provider: "openai", credential_deployment_id: "d456",
  });
  assert.deepEqual(discoveryRequest(legacy, "", "", false), {
    provider: "openai", deployment_id: "d456", refresh: false,
  });
});

test("fresh credentials accept literal keys and reject environment expressions in either field", () => {
  assert.deepEqual(deploymentCredentials(common, " new-literal-key ", "https://api.example.invalid/v1"), {
    provider: "openai", api_key: "new-literal-key", api_base: "https://api.example.invalid/v1",
  });
  for (const [key, base] of [[" env:OPENAI_API_KEY ", ""], ["literal", "env:OPENAI_API_BASE"]]) {
    assert.throws(() => discoveryRequest(common, key!, base!, false), rejected("environment_reference"));
    assert.throws(() => deploymentCredentials(common, key!, base!), rejected("environment_reference"));
  }
});

test("opening stored credentials describes the key without exposing it or an environment reference", () => {
  const literal = providerDraft({ api_key: "sk-fixture-0000-abcd", api_base: "env:OPENAI_API_BASE", api_base_placeholder: "https://api.openai.com/v1" });
  assert.deepEqual(literal.key, { state: "saved", ending: "abcd" });
  assert.equal(literal.configured, true);
  assert.equal(literal.baseRetained, true);
  assert.equal(literal.baseVariable, "OPENAI_API_BASE");
  assert.equal(literal.base, "");
  assert.ok(!JSON.stringify(literal).includes("sk-fixture-0000-abcd"));
  // A short key is too revealing to quote even in part.
  assert.deepEqual(providerDraft({ api_key: "short-key", api_base: null, api_base_placeholder: null }).key, { state: "saved", ending: null });
  const reference = providerDraft({ api_key: " env:OPENAI_API_KEY ", api_base: null, api_base_placeholder: null });
  assert.deepEqual(reference.key, { state: "environment", variable: "OPENAI_API_KEY", stored: true });
  assert.ok(!JSON.stringify(reference).includes("env:"));
  // A key the server found in the environment is not a stored reference.
  const detected = providerDraft({ api_key: "env:OPENAI_API_KEY", api_key_source: "environment", api_base: null, api_base_placeholder: null });
  assert.deepEqual(detected.key, { state: "environment", variable: "OPENAI_API_KEY", stored: false });
  assert.equal(detected.configured, true);
  const empty = providerDraft({ api_key: null, api_base: "https://api.example.invalid/v1", api_base_placeholder: null });
  assert.deepEqual(empty.key, { state: "none" });
  assert.equal(empty.configured, false);
  assert.equal(empty.base, "https://api.example.invalid/v1");
});

test("the key is replaced or removed on its own and never left blank", () => {
  const draft = providerDraft({ api_key: "stored-secret", api_base: "https://api.example.invalid/v1", api_base_placeholder: null });
  assert.deepEqual(providerUpdate(draft, { kind: "key", key: " replacement " }, "rev1"), { expected_revision: "rev1", api_key: "replacement" });
  assert.deepEqual(providerUpdate(draft, { kind: "removeKey" }, "rev1"), { expected_revision: "rev1", api_key: null });
  assert.throws(() => providerUpdate(draft, { kind: "key", key: "  " }, "rev1"), rejected("key_required"));
  assert.throws(() => providerUpdate(draft, { kind: "key", key: "env:OTHER_KEY" }, "rev1"), rejected("environment_reference"));
});

test("moving the address never carries the saved key, while equivalent URLs change nothing", () => {
  const draft = providerDraft({ api_key: "env:OPENAI_API_KEY", api_base: "https://api.example.invalid/v1/", api_base_placeholder: null });
  assert.equal(providerUpdate(draft, { kind: "base", base: "https://api.example.invalid:443/v1", key: "" }, "rev1"), null);
  assert.deepEqual(providerUpdate(draft, { kind: "base", base: "https://api.example.invalid/v1", key: "replacement" }, "rev1"), {
    expected_revision: "rev1", api_key: "replacement",
  });
  assert.throws(() => providerUpdate(draft, { kind: "base", base: "https://other.invalid/v1", key: "" }, "rev1"), rejected("endpoint_key_required"));
  assert.throws(() => providerUpdate(draft, { kind: "base", base: "https://other.invalid/v1", key: "env:OTHER_KEY" }, "rev1"), rejected("environment_reference"));
  assert.throws(() => providerUpdate(draft, { kind: "base", base: "env:OTHER_BASE", key: "replacement" }, "rev1"), rejected("environment_reference"));
  assert.deepEqual(providerUpdate(draft, { kind: "base", base: "https://other.invalid/v1", key: "replacement" }, "rev1"), {
    expected_revision: "rev1", api_key: "replacement", api_base: "https://other.invalid/v1",
  });
  // A blank address returns to the provider default, still with a new key.
  assert.deepEqual(providerUpdate(draft, { kind: "base", base: " ", key: "replacement" }, "rev1"), {
    expected_revision: "rev1", api_key: "replacement", api_base: null,
  });
});

test("without a saved key, moving the address explicitly connects without one", () => {
  const draft = providerDraft({ api_key: null, api_base: null, api_base_placeholder: "https://api.openai.com/v1" });
  assert.deepEqual(providerUpdate(draft, { kind: "base", base: "https://other.invalid/v1", key: "" }, "rev1"), {
    expected_revision: "rev1", api_key: null, api_base: "https://other.invalid/v1",
  });
  assert.deepEqual(providerUpdate(draft, { kind: "base", base: "https://other.invalid/v1", key: "fresh" }, "rev1"), {
    expected_revision: "rev1", api_key: "fresh", api_base: "https://other.invalid/v1",
  });
  assert.equal(providerUpdate(draft, { kind: "base", base: "", key: "" }, "rev1"), null);
  // An address read from the environment always counts as a move, back to the default when blank.
  const retained = providerDraft({ api_key: null, api_base: "env:OPENAI_API_BASE", api_base_placeholder: "https://api.openai.com/v1" });
  assert.deepEqual(providerUpdate(retained, { kind: "base", base: "", key: "" }, "rev1"), {
    expected_revision: "rev1", api_key: null, api_base: null,
  });
});

test("saving detected API models preserves environment binding without returning a key or endpoint", () => {
  const deployment: Deployment = {
    deployment_id: "detected-id", routing_group: "default", model: "openai/sample", weight: 2,
    api_key_configured: false, api_base_configured: false, api_base: null, persisted: false,
  };
  for (const [model, provider] of [
    ["openai/sample", "openai"], ["anthropic/sample", "anthropic"], ["gemini/sample", "gemini"],
    ["deepseek/sample", "deepseek"], ["openrouter/vendor/model", "openrouter"], ["unprefixed-model", "openai"],
  ]) {
    assert.deepEqual(detectedDeployment({ ...deployment, model: model! }), {
      model_name: "default", model, weight: 2, credential_provider_id: `env:${provider}`,
    });
  }
});

test("saving detected subscription models keeps official CLI authentication instead of inventing an environment credential", () => {
  for (const provider of ["claude-agent", "copilot", "chatgpt"]) {
    const deployment: Deployment = {
      deployment_id: "detected-id", routing_group: "default", model: `${provider}/sample`, weight: 1,
      api_key_configured: false, api_base_configured: false, api_base: null, persisted: false,
    };
    assert.deepEqual(detectedDeployment(deployment), { model_name: "default", model: deployment.model, weight: 1 });
  }
});

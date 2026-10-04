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

test("opening and saving stored credentials does not expose or echo literal keys or environment references", () => {
  for (const key of ["stored-secret", "env:OPENAI_API_KEY"]) {
    const draft = providerDraft({ api_key: key, api_base: "env:OPENAI_API_BASE", api_base_placeholder: "https://api.openai.com/v1" });
    assert.equal(draft.configured, true);
    assert.equal(draft.baseRetained, true);
    assert.equal(draft.base, "");
    assert.ok(!JSON.stringify(draft).includes(key));
    assert.deepEqual(providerUpdate(draft, "", "", false, "rev1"), { expected_revision: "rev1" });
  }
});

test("editing an endpoint requires explicit replacement or removal, while equivalent URLs keep the key", () => {
  const draft = providerDraft({ api_key: "env:OPENAI_API_KEY", api_base: "https://api.example.invalid/v1/", api_base_placeholder: null });
  assert.deepEqual(providerUpdate(draft, "", "https://api.example.invalid:443/v1", false, "rev1"), { expected_revision: "rev1" });
  assert.throws(() => providerUpdate(draft, "", "https://other.invalid/v1", false, "rev1"), rejected("endpoint_key_required"));
  assert.throws(() => providerUpdate(draft, "env:OTHER_KEY", "https://other.invalid/v1", false, "rev1"), rejected("environment_reference"));
  assert.deepEqual(providerUpdate(draft, "replacement", "https://other.invalid/v1", false, "rev1"), {
    expected_revision: "rev1", api_key: "replacement", api_base: "https://other.invalid/v1",
  });
  assert.deepEqual(providerUpdate(draft, "", "https://other.invalid/v1", true, "rev1"), {
    expected_revision: "rev1", api_key: null, api_base: "https://other.invalid/v1",
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

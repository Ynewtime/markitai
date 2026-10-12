import assert from "node:assert/strict";
import test from "node:test";
import type { ProviderCard } from "../api/types.ts";
import { arrangeProviders, iconKey, isSubscription, matches, monogram, usesCloudCredentials } from "./provider-catalog.ts";

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
const label = (item: ProviderCard) => item.label;
const names = (choices: { provider: string }[]) => choices.map((choice) => choice.provider);

// As the Rust server sends them: a group, an icon key (null for a monogram) and aliases.
const rust: ProviderCard[] = [
  card("openai", "OpenAI", { group: "common", icon: "openai", aliases: [] }),
  card("anthropic", "Anthropic", { group: "common", icon: "anthropic", aliases: ["claude"] }),
  card("mistral", "Mistral AI", { group: "common", icon: "mistral", kind: "compatible", aliases: [] }),
  card("ollama", "Ollama", { group: "common", icon: "ollama", aliases: ["ollama_chat"] }),
  card("groq", "Groq", { group: "inference", icon: "groq", kind: "compatible", aliases: [] }),
  card("bedrock", "Amazon Bedrock", { group: "clouds", icon: "bedrock", aliases: ["aws"] }),
  card("databricks", "Databricks", { group: "clouds", icon: null, kind: "compatible", aliases: [] }),
  card("copilot", "GitHub Copilot", { group: "subscriptions", icon: "githubcopilot", aliases: ["github"] }),
  card("custom", "OpenAI-compatible endpoint", { group: "local", icon: null, aliases: ["openai_like"] }),
  card("hosted_vllm", "vLLM", { group: "local", icon: "vllm", kind: "compatible", aliases: [] }),
];

test("the common eight come first in their order; the rest by name under a group filter", () => {
  const view = arrangeProviders(rust, { query: "", filter: "all", label });
  assert.deepEqual(names(view.common), ["openai", "anthropic", "ollama", "mistral"]);
  assert.deepEqual(names(view.all), ["bedrock", "databricks", "copilot", "groq", "custom", "hosted_vllm"]);
  assert.deepEqual(view.filters, ["all", "clouds", "inference", "local", "subscriptions"]);
  assert.equal(view.total, rust.length);
  const clouds = arrangeProviders(rust, { query: "", filter: "clouds", label });
  assert.deepEqual(names(clouds.all), ["bedrock", "databricks"]);
  // A filter that names no group of this server shows everything.
  assert.deepEqual(names(arrangeProviders(rust, { query: "", filter: "makers", label }).all), names(view.all));
});

test("search reads names, prefixes and aliases, and narrows every section", () => {
  const search = (query: string, filter: "all" | "local" = "all") => arrangeProviders(rust, { query, filter, label });
  assert.deepEqual(names(search("claude").common), ["anthropic"]);
  assert.deepEqual(names(search("OLLAMA chat").common), ["ollama"]);
  assert.deepEqual(names(search("hosted vllm").all), ["hosted_vllm"]);
  assert.deepEqual(names(search("openai_like").all), ["custom"]);
  assert.deepEqual(names(search("aws").all), ["bedrock"]);
  assert.deepEqual(names(search("v", "local").all), ["hosted_vllm"]);
  const none = search("watsonx");
  assert.equal(none.common.length + none.all.length + none.saved.length, 0);
  assert.ok(matches(card("veniceai", "Venice AI"), "Venice AI", "venice ai"));
});

test("a saved connection is a row, and its provider stays offered through it", () => {
  const saved = card("anthropic", "Anthropic", { id: "provider:p1", provider_id: "p1", kind: "configured", group: "common", icon: "anthropic", model_count: 2 });
  const environment = card("groq", "Groq", { id: "env:groq", kind: "environment", source: "GROQ_API_KEY", group: "inference", icon: "groq" });
  // The server hides a provider's built-in card once a connection exists.
  const cards = [...rust.filter((item) => item.provider !== "anthropic" && item.provider !== "groq"), saved, environment];
  const view = arrangeProviders(cards, { query: "", filter: "all", label });
  assert.deepEqual(view.saved.map((item) => item.id), ["provider:p1", "env:groq"]);
  assert.equal(view.common.find((choice) => choice.provider === "anthropic")?.card, saved);
  assert.equal(view.all.find((choice) => choice.provider === "groq")?.card, environment);
  assert.deepEqual(names(arrangeProviders(cards, { query: "groq", filter: "all", label }).all), ["groq"]);
});

test("without groups (Python 1.2.1) Common is inferred from the eight names and the rest has no filter", () => {
  const python = [
    card("openai", "OpenAI"),
    card("anthropic", "Anthropic"),
    card("gemini", "Gemini"),
    card("ollama", "Ollama"),
    card("deepseek", "DeepSeek"),
    card("openrouter", "OpenRouter"),
    card("azure", "Azure OpenAI"),
    card("custom", "OpenAI-compatible endpoint"),
    card("xai", "xAI", { kind: "compatible" }),
    card("openai", "OpenAI", { id: "provider:p2", provider_id: "p2", kind: "configured" }),
  ];
  const view = arrangeProviders(python, { query: "", filter: "clouds", label });
  assert.deepEqual(names(view.common), ["openai", "anthropic", "gemini", "deepseek", "openrouter", "azure", "ollama"]);
  assert.deepEqual(names(view.all), ["custom", "xai"]);
  assert.deepEqual(view.filters, ["all"]);
  assert.deepEqual(view.saved.map((item) => item.id), ["provider:p2"]);
  // Without an icon key, a mark named like the prefix is tried.
  assert.equal(iconKey(python[0]!), "openai");
  assert.equal(iconKey(rust[6]!), null);
});

test("subscriptions and the clouds with their own credentials are recognised with or without groups", () => {
  assert.ok(isSubscription(card("chatgpt", "ChatGPT")));
  assert.ok(isSubscription(card("copilot", "GitHub Copilot", { group: "subscriptions" })));
  assert.ok(!isSubscription(card("openai", "OpenAI", { group: "common" })));
  assert.ok(usesCloudCredentials(card("vertex_ai", "Vertex AI")) && !usesCloudCredentials(card("azure_ai", "Azure AI Foundry")));
});

test("a monogram is one or two initials without generic words", () => {
  assert.equal(monogram("Databricks"), "D");
  assert.equal(monogram("Sarvam AI"), "S");
  assert.equal(monogram("Docker Model Runner"), "DM");
  assert.equal(monogram("text-generation-webui"), "TG");
  assert.equal(monogram("AI/ML API"), "M");
  assert.equal(monogram("SCX.ai"), "S");
  assert.equal(monogram("W&B Inference"), "W");
});

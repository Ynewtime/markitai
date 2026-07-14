import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LLMSettingsPayload } from "../api/types";
import { dicts } from "../i18n";
import { SettingsModal } from "./SettingsModal";

const api = vi.hoisted(() => ({
  fetchSettings: vi.fn(),
  fetchProviders: vi.fn(),
  discover: vi.fn(),
  add: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
  test: vi.fn(),
}));

vi.mock("../api/client", () => ({
  fetchLLMSettings: api.fetchSettings,
  fetchProviderConnections: api.fetchProviders,
  discoverLLMModels: api.discover,
  addLLMDeployments: api.add,
  updateLLMDeployment: api.update,
  deleteLLMDeployment: api.remove,
  testLLMSettings: api.test,
}));

const emptySettings: LLMSettingsPayload = {
  configured: false,
  routable: false,
  source: "none",
  config_path: "~/.markitai/config.json",
  config_origin: "user",
  revision: "revision-1",
  deployments: [],
  detected: [],
};

describe("SettingsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads models after choosing a provider and adds the checked set atomically", async () => {
    const user = userEvent.setup();
    api.fetchSettings.mockResolvedValue(emptySettings);
    api.fetchProviders.mockResolvedValue([
      {
        id: "env:openai",
        provider: "openai",
        label: "OpenAI",
        kind: "environment",
        status: "ready",
        source: "OPENAI_API_KEY",
        credential: "env:OPENAI_API_KEY",
        supports_discovery: true,
      },
    ]);
    api.discover.mockResolvedValue({
      provider: "openai",
      status: "ok",
      source: "live_api",
      authoritative: true,
      cached: false,
      stale: false,
      models: [
        { model: "openai/gpt-test", label: "GPT Test", supports_vision: true },
      ],
    });
    api.add.mockResolvedValue({
      ...emptySettings,
      configured: true,
      revision: "revision-2",
      deployments: [
        {
          deployment_id: "deployment-1",
          routing_group: "default",
          model: "openai/gpt-test",
          weight: 1,
          api_key_configured: true,
          api_base_configured: false,
          api_base: null,
          persisted: true,
        },
      ],
    });
    const onSaved = vi.fn();

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={onSaved}
        announce={() => undefined}
      />,
    );

    await user.click(await screen.findByRole("button", { name: dicts.en.addModels }));
    await user.click(screen.getByRole("button", { name: /OpenAI/ }));
    expect(document.querySelector(".provider-picker")).not.toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: dicts.en.breadcrumbAria })).toHaveTextContent(
      "llm settings/add models/OpenAI",
    );
    expect(screen.getByRole("heading", { name: "OpenAI" })).toBeVisible();
    expect(screen.getByText("Using credentials from OPENAI_API_KEY.")).toBeVisible();
    expect(screen.queryByText(/ready ·/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(dicts.en.setApiKey)).not.toBeInTheDocument();
    await waitFor(() => {
      expect(api.discover).toHaveBeenCalledWith(
        expect.objectContaining({ provider: "openai", refresh: false }),
      );
    });
    await user.click(await screen.findByRole("checkbox", { name: /GPT Test/ }));
    await user.click(screen.getByRole("button", { name: dicts.en.refreshModels }));
    await waitFor(() => {
      expect(api.discover).toHaveBeenLastCalledWith(
        expect.objectContaining({ provider: "openai", refresh: true }),
      );
    });
    await user.click(await screen.findByRole("checkbox", { name: /GPT Test/ }));
    await user.click(screen.getByRole("button", { name: dicts.en.addModelsCount(1) }));

    await waitFor(() => {
      expect(api.add).toHaveBeenCalledWith({
        expected_revision: "revision-1",
        deployments: [
          {
            model_name: "default",
            model: "openai/gpt-test",
            weight: 1,
            api_key: "env:OPENAI_API_KEY",
          },
        ],
      });
      expect(onSaved).toHaveBeenCalledOnce();
      expect(api.fetchSettings).toHaveBeenCalledOnce();
    });
  });

  it("keeps the stored API base when editing unrelated deployment fields", async () => {
    const user = userEvent.setup();
    const deployment = {
      deployment_id: "deployment-1",
      routing_group: "default",
      model: "openai/gpt-test",
      weight: 1,
      api_key_configured: true,
      api_base_configured: true,
      api_base: "https://proxy.example",
      persisted: true,
    };
    const configuredSettings = {
      ...emptySettings,
      configured: true,
      deployments: [deployment],
    };
    api.fetchSettings.mockResolvedValue(configuredSettings);
    api.fetchProviders.mockResolvedValue([]);
    api.update.mockResolvedValue({
      ...configuredSettings,
      revision: "revision-2",
    });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    await user.click(await screen.findByRole("button", { name: dicts.en.edit }));
    expect(screen.getByLabelText(dicts.en.setApiBase)).toHaveValue("");
    expect(screen.getByLabelText(dicts.en.setApiBase)).toHaveAttribute(
      "placeholder",
      dicts.en.keepBaseHint,
    );
    await user.click(screen.getByRole("button", { name: dicts.en.save }));

    await waitFor(() => {
      expect(api.update).toHaveBeenCalledWith("deployment-1", {
        model_name: "default",
        model: "openai/gpt-test",
        weight: 1,
        expected_revision: "revision-1",
      });
    });
  });

  it("automatically loads a saved provider connection without raw status labels", async () => {
    const user = userEvent.setup();
    api.fetchSettings.mockResolvedValue(emptySettings);
    api.fetchProviders.mockResolvedValue([
      {
        id: "deployment:chatgpt-1",
        deployment_id: "chatgpt-1",
        provider: "chatgpt",
        label: "ChatGPT",
        kind: "configured",
        status: "ready",
        source: "config",
        supports_discovery: true,
      },
    ]);
    api.discover.mockResolvedValue({
      provider: "chatgpt",
      status: "ok",
      source: "live_api",
      authoritative: true,
      cached: false,
      stale: false,
      models: [
        { model: "chatgpt/gpt-test", label: "GPT Test", supports_vision: false },
      ],
    });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    await user.click(await screen.findByRole("button", { name: dicts.en.addModels }));
    await user.click(screen.getByRole("button", { name: /ChatGPT/ }));

    expect(await screen.findByRole("checkbox", { name: /GPT Test/ })).toBeVisible();
    expect(api.discover).toHaveBeenCalledWith({
      provider: "chatgpt",
      deployment_id: "chatgpt-1",
      refresh: false,
    });
    expect(screen.getByText("Available models load automatically from this saved connection.")).toBeVisible();
    expect(screen.queryByText(/ready · config|ok · live_api/)).not.toBeInTheDocument();
  });

  it("returns one level and hides model controls when discovery is unavailable", async () => {
    const user = userEvent.setup();
    api.fetchSettings.mockResolvedValue(emptySettings);
    api.fetchProviders.mockResolvedValue([
      {
        id: "common:openai",
        provider: "openai",
        label: "OpenAI",
        kind: "common",
        status: "needs_credentials",
        source: "built_in",
        supports_discovery: true,
      },
    ]);
    api.discover.mockResolvedValue({
      provider: "openai",
      status: "unavailable",
      source: "live_api",
      authoritative: false,
      cached: false,
      stale: false,
      models: [],
      detail: "model discovery failed (HTTPStatusError)",
    });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    await user.click(await screen.findByRole("button", { name: dicts.en.addModels }));
    await user.click(screen.getByRole("button", { name: /OpenAI/ }));
    await user.type(screen.getByLabelText(/api key/i), "sk-invalid");
    await user.click(screen.getByRole("button", { name: dicts.en.loadModels }));
    expect(await screen.findByText(dicts.en.modelsUnavailable)).toBeVisible();
    expect(screen.getByText("model discovery failed (HTTPStatusError)")).toBeVisible();
    expect(screen.queryByRole("searchbox", { name: dicts.en.searchModels })).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText(dicts.en.manualModelPh)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: dicts.en.cancel }));
    expect(document.querySelector(".provider-picker")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: dicts.en.addModels })).toBeVisible();
  });
});

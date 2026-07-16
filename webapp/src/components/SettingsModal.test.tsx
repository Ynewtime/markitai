import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LLMSettingsPayload } from "../api/types";
import { dicts } from "../i18n";
import { SettingsModal } from "./SettingsModal";

const api = vi.hoisted(() => ({
  fetchSettings: vi.fn(),
  fetchProviders: vi.fn(),
  fetchCredentials: vi.fn(),
  discover: vi.fn(),
  add: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
  updateProvider: vi.fn(),
  removeProvider: vi.fn(),
  test: vi.fn(),
}));

vi.mock("../api/client", () => ({
  fetchLLMSettings: api.fetchSettings,
  fetchProviderConnections: api.fetchProviders,
  fetchLLMProviderCredentials: api.fetchCredentials,
  discoverLLMModels: api.discover,
  addLLMDeployments: api.add,
  updateLLMDeployment: api.update,
  deleteLLMDeployment: api.remove,
  updateLLMProvider: api.updateProvider,
  deleteLLMProvider: api.removeProvider,
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
    await user.click(
      screen.getByRole("button", { name: dicts.en.selectProvider("OpenAI") }),
    );
    expect(document.querySelector(".provider-picker")).not.toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: dicts.en.breadcrumbAria })).toHaveTextContent(
      `${dicts.en.settingsTitle}/${dicts.en.addModels}/OpenAI`,
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
    expect(await screen.findByRole("checkbox", { name: /GPT Test/ })).toBeChecked();
    await user.click(screen.getByRole("button", { name: dicts.en.addModelsCount(1) }));

    await waitFor(() => {
      expect(api.add).toHaveBeenCalledWith({
        expected_revision: "revision-1",
        deployments: [
          {
            model_name: "default",
            model: "openai/gpt-test",
            provider: "openai",
            weight: 1,
            api_key: "env:OPENAI_API_KEY",
          },
        ],
      });
      expect(onSaved).toHaveBeenCalledOnce();
      expect(api.fetchSettings).toHaveBeenCalledOnce();
    });
  });

  it("keeps provider credentials out of model editing", async () => {
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
    api.test.mockResolvedValue({
      ok: true,
      detail: "openai/gpt-test responded",
    });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    expect(await screen.findByText(dicts.en.modelWeight(1))).toHaveAttribute(
      "title",
      dicts.en.weightHint,
    );
    expect(screen.queryByText("w1")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: dicts.en.test }));
    await waitFor(() => {
      expect(document.querySelector(".model-test.ok svg")).not.toBeNull();
    });
    expect(screen.queryByText(/gpt-test responded/i)).not.toBeInTheDocument();
    expect(await screen.findByRole("status")).toHaveTextContent(
      `${dicts.en.modelTestPassed}${dicts.en.modelTestReady("openai/gpt-test")}`,
    );
    await user.click(screen.getByRole("button", { name: dicts.en.edit }));
    expect(screen.queryByLabelText(dicts.en.setApiKey)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(dicts.en.setApiBase)).not.toBeInTheDocument();
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

  it("moves model-test failures into a notification", async () => {
    const user = userEvent.setup();
    api.fetchSettings.mockResolvedValue({
      ...emptySettings,
      configured: true,
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
    api.fetchProviders.mockResolvedValue([]);
    api.test.mockResolvedValue({
      ok: false,
      detail: "Authentication failed",
    });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    await user.click(await screen.findByRole("button", { name: dicts.en.test }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      `${dicts.en.modelTestFailed}Authentication failed`,
    );
    expect(document.querySelector(".model-test.fail svg")).not.toBeNull();
    expect(document.querySelector(".prov-detail")).toBeNull();
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
    await user.click(
      screen.getByRole("button", { name: dicts.en.selectProvider("ChatGPT") }),
    );

    expect(await screen.findByRole("checkbox", { name: /GPT Test/ })).toBeVisible();
    expect(api.discover).toHaveBeenCalledWith({
      provider: "chatgpt",
      deployment_id: "chatgpt-1",
      refresh: false,
    });
    expect(screen.getByText("Available models load automatically from this saved connection.")).toBeVisible();
    expect(screen.queryByText(/ready · config|ok · live_api/)).not.toBeInTheDocument();
  });

  it("edits and explicitly deletes a saved provider connection", async () => {
    const user = userEvent.setup();
    const provider = {
      id: "provider:deepseek-1",
      provider_id: "deepseek-1",
      provider: "deepseek",
      label: "DeepSeek",
      kind: "configured" as const,
      status: "ready",
      source: "config",
      api_key_configured: true,
      api_base_configured: false,
      api_base: null,
      model_count: 0,
      supports_discovery: true,
    };
    api.fetchSettings.mockResolvedValue(emptySettings);
    api.fetchProviders.mockResolvedValue([provider]);
    api.fetchCredentials.mockResolvedValue({
      api_key: "sk-current-secret",
      api_base: "https://api.deepseek.com/v1",
    });
    api.updateProvider.mockResolvedValue({ ...emptySettings, revision: "revision-2" });
    api.removeProvider.mockResolvedValue({ ...emptySettings, revision: "revision-3" });

    render(
      <SettingsModal
        t={dicts.en}
        onClose={() => undefined}
        onSaved={() => undefined}
        announce={() => undefined}
      />,
    );

    expect(screen.queryByText(dicts.en.providersTitle)).not.toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: dicts.en.addModels }),
    );
    expect(await screen.findByText(dicts.en.providersTitle)).toBeVisible();
    expect(
      screen.getByRole("navigation", { name: dicts.en.breadcrumbAria }),
    ).toHaveTextContent(`${dicts.en.settingsTitle}/${dicts.en.addModels}`);
    await user.click(
      screen.getByRole("button", { name: dicts.en.editProvider("DeepSeek") }),
    );
    const keyInput = await screen.findByLabelText(dicts.en.setApiKey);
    const baseInput = screen.getByLabelText(dicts.en.setApiBase);
    expect(api.fetchCredentials).toHaveBeenCalledWith("deepseek-1");
    expect(keyInput).toHaveAttribute("type", "password");
    expect(keyInput).toHaveValue("sk-current-secret");
    expect(baseInput).toHaveAttribute("type", "url");
    expect(baseInput).toHaveValue("https://api.deepseek.com/v1");
    expect(
      screen.queryByRole("button", {
        name: dicts.en.revealField(dicts.en.setApiBase),
      }),
    ).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", {
        name: dicts.en.revealField(dicts.en.setApiKey),
      }),
    );
    expect(keyInput).toHaveAttribute("type", "text");
    await user.clear(keyInput);
    await user.type(keyInput, "sk-new-key");
    await user.click(screen.getByRole("button", { name: dicts.en.save }));
    await waitFor(() => {
      expect(api.updateProvider).toHaveBeenCalledWith("deepseek-1", {
        api_key: "sk-new-key",
        api_base: "https://api.deepseek.com/v1",
        expected_revision: "revision-1",
      });
    });

    api.fetchProviders.mockResolvedValue([provider]);
    await waitFor(() =>
      expect(screen.queryByLabelText(dicts.en.setApiKey)).not.toBeInTheDocument(),
    );
    await user.click(
      screen.getByRole("button", { name: dicts.en.deleteProvider("DeepSeek") }),
    );
    expect(screen.getByRole("alertdialog")).toHaveTextContent(
      dicts.en.deleteProviderDescription(0),
    );
    await user.click(
      screen.getByRole("button", { name: dicts.en.deletePermanently }),
    );
    await waitFor(() => {
      expect(api.removeProvider).toHaveBeenCalledWith("deepseek-1", "revision-2");
    });
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
    await user.click(
      screen.getByRole("button", { name: dicts.en.selectProvider("OpenAI") }),
    );
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

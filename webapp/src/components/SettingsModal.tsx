import { useCallback, useEffect, useRef, useState } from "react";
import {
  addLLMDeployments,
  deleteLLMDeployment,
  discoverLLMModels,
  fetchLLMSettings,
  fetchProviderConnections,
  testLLMSettings,
  updateLLMDeployment,
} from "../api/client";
import type {
  LLMDeployment,
  LLMModelCreate,
  LLMSettingsPayload,
  ModelDiscoveryResult,
  ProviderConnection,
} from "../api/types";
import type { Dict } from "../i18n";
import { ModelPicker } from "./settings/ModelPicker";
import { ProviderPicker } from "./settings/ProviderPicker";
import { XIcon } from "./icons";

const EXIT_MS = 120;
const CONFIRM_RESET_MS = 4000;
type RowTest = { state: "busy" } | { state: "ok" | "fail"; detail: string };

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isLocalProvider(provider: string): boolean {
  return provider === "claude-agent" || provider === "copilot" || provider === "chatgpt";
}

export function SettingsModal({
  t,
  onClose,
  onSaved,
  announce,
}: {
  t: Dict;
  onClose: () => void;
  onSaved: () => void;
  announce: (message: string) => void;
}) {
  const [settings, setSettings] = useState<LLMSettingsPayload | null>(null);
  const [providers, setProviders] = useState<ProviderConnection[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  const [closing, setClosing] = useState(false);
  const closingRef = useRef(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);
  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setClosing(true);
    closeTimerRef.current = window.setTimeout(onClose, EXIT_MS);
  }, [onClose]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    cardRef.current?.focus();
    return () => {
      document.body.style.overflow = previous;
      if (closeTimerRef.current !== null) window.clearTimeout(closeTimerRef.current);
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        requestClose();
        return;
      }
      if (event.key !== "Tab") return;
      const root = cardRef.current;
      if (root === null) return;
      const focusable = Array.from(
        root.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("disabled") && element.offsetParent !== null);
      if (focusable.length === 0) {
        event.preventDefault();
        root.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (!(active instanceof HTMLElement) || !root.contains(active)) {
        event.preventDefault();
        first?.focus();
      } else if (event.shiftKey && (active === first || active === root)) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [requestClose]);

  useEffect(() => {
    let stale = false;
    fetchLLMSettings().then(
      (nextSettings) => {
        if (!stale) setSettings(nextSettings);
      },
      (error: unknown) => {
        if (!stale) setLoadError(errorText(error));
      },
    );
    fetchProviderConnections().then(
      (nextProviders) => {
        if (!stale) setProviders(nextProviders);
      },
      () => undefined,
    );
    return () => {
      stale = true;
    };
  }, []);

  const [rowTests, setRowTests] = useState<Record<string, RowTest>>({});
  const runTest = async (deployment: LLMDeployment) => {
    const id = deployment.deployment_id;
    setRowTests((previous) => ({ ...previous, [id]: { state: "busy" } }));
    try {
      const result = await testLLMSettings({ deployment_id: id });
      setRowTests((previous) => ({
        ...previous,
        [id]: { state: result.ok ? "ok" : "fail", detail: result.detail },
      }));
    } catch (error) {
      setRowTests((previous) => ({
        ...previous,
        [id]: { state: "fail", detail: errorText(error) },
      }));
    }
  };

  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const confirmTimerRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    },
    [],
  );

  const removeDeployment = async (deployment: LLMDeployment) => {
    if (settings === null) return;
    const id = deployment.deployment_id;
    if (confirmDelete !== id) {
      setConfirmDelete(id);
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
      confirmTimerRef.current = window.setTimeout(() => setConfirmDelete(null), CONFIRM_RESET_MS);
      return;
    }
    setConfirmDelete(null);
    setListError(null);
    try {
      const next = await deleteLLMDeployment(id, settings.revision);
      setSettings(next);
      onSaved();
    } catch (error) {
      setListError(errorText(error));
    }
  };

  const [editing, setEditing] = useState<LLMDeployment | null>(null);
  const [editGroup, setEditGroup] = useState("");
  const [editModel, setEditModel] = useState("");
  const [editKey, setEditKey] = useState("");
  const [editBase, setEditBase] = useState("");
  const [editWeight, setEditWeight] = useState(1);
  const [editBusy, setEditBusy] = useState(false);
  const openEdit = (deployment: LLMDeployment) => {
    setEditing(deployment);
    setEditGroup(deployment.routing_group);
    setEditModel(deployment.model);
    setEditKey("");
    // The API exposes only a secret-free origin, not the full stored path.
    // Keep this blank so saving another field cannot truncate the real base URL.
    setEditBase("");
    setEditWeight(deployment.weight);
    setListError(null);
  };
  const saveEdit = async () => {
    if (settings === null || editing === null || editBusy) return;
    setEditBusy(true);
    setListError(null);
    try {
      const next = await updateLLMDeployment(editing.deployment_id, {
        model_name: editGroup.trim(),
        model: editModel.trim(),
        ...(editKey.trim() === "" ? {} : { api_key: editKey.trim() }),
        ...(editBase.trim() === "" ? {} : { api_base: editBase.trim() }),
        weight: editWeight,
        expected_revision: settings.revision,
      });
      setSettings(next);
      setEditing(null);
      onSaved();
      announce(t.saved);
    } catch (error) {
      setListError(errorText(error));
    } finally {
      setEditBusy(false);
    }
  };

  const [adding, setAdding] = useState(false);
  const [provider, setProvider] = useState<ProviderConnection | null>(null);
  const [draftKey, setDraftKey] = useState("");
  const [draftBase, setDraftBase] = useState("");
  const [discovery, setDiscovery] = useState<ModelDiscoveryResult | null>(null);
  const [discovering, setDiscovering] = useState(false);
  const [selectedModels, setSelectedModels] = useState(new Set<string>());
  const [routingGroup, setRoutingGroup] = useState("default");
  const [weight, setWeight] = useState(1);
  const [addBusy, setAddBusy] = useState(false);
  const autoDiscoveryRef = useRef<string | null>(null);
  const usesExistingConnection =
    provider !== null &&
    (isLocalProvider(provider.provider) ||
      provider.deployment_id !== undefined ||
      provider.kind === "environment");
  const usesDefaultOllama = provider?.provider === "ollama";
  const autoDiscoversModels = usesExistingConnection || usesDefaultOllama;
  const requiresPrimaryBase =
    provider?.provider === "azure" || provider?.provider === "custom";
  const showsApiKey =
    provider !== null && !usesExistingConnection && provider.provider !== "ollama";
  const canLoadModels =
    provider !== null &&
    (autoDiscoversModels ||
      (provider.provider === "custom" && draftBase.trim() !== "") ||
      (provider.provider === "azure" &&
        draftKey.trim() !== "" &&
        draftBase.trim() !== "") ||
      (provider.provider !== "custom" &&
        provider.provider !== "azure" &&
        draftKey.trim() !== ""));

  const resetAddFlow = () => {
    setAdding(false);
    setProvider(null);
    setDiscovery(null);
    setSelectedModels(new Set());
    setListError(null);
    autoDiscoveryRef.current = null;
  };

  const backToProviders = () => {
    setProvider(null);
    setDiscovery(null);
    setSelectedModels(new Set());
    setListError(null);
    autoDiscoveryRef.current = null;
  };

  const chooseProvider = (next: ProviderConnection) => {
    setProvider(next);
    setDraftKey(next.credential ?? "");
    setDraftBase("");
    setDiscovery(null);
    setSelectedModels(new Set());
    setListError(null);
    autoDiscoveryRef.current = null;
  };

  const loadModels = async (refresh = false) => {
    if (provider === null || discovering) return;
    setDiscovering(true);
    setListError(null);
    try {
      const result = await discoverLLMModels({
        provider: provider.provider,
        ...(provider.deployment_id === undefined
          ? {}
          : { deployment_id: provider.deployment_id }),
        ...(draftKey.trim() === "" ? {} : { api_key: draftKey.trim() }),
        ...(draftBase.trim() === "" ? {} : { api_base: draftBase.trim() }),
        refresh,
      });
      setDiscovery(result);
      setSelectedModels(new Set());
    } catch (error) {
      setListError(errorText(error));
    } finally {
      setDiscovering(false);
    }
  };

  useEffect(() => {
    if (
      provider === null ||
      !autoDiscoversModels ||
      discovery !== null ||
      discovering ||
      autoDiscoveryRef.current === provider.id
    ) {
      return;
    }
    autoDiscoveryRef.current = provider.id;
    void loadModels(false);
  }, [provider, autoDiscoversModels, discovery, discovering]);

  const addSelected = async () => {
    if (settings === null || provider === null || selectedModels.size === 0 || addBusy) return;
    setAddBusy(true);
    setListError(null);
    const local = isLocalProvider(provider.provider);
    const deployments: LLMModelCreate[] = [...selectedModels].map((model) => ({
      model_name: routingGroup.trim() || "default",
      model,
      weight,
      ...(provider.deployment_id === undefined
        ? {}
        : { credential_deployment_id: provider.deployment_id }),
      ...(!local && draftKey.trim() !== "" ? { api_key: draftKey.trim() } : {}),
      ...(!local && draftBase.trim() !== "" ? { api_base: draftBase.trim() } : {}),
    }));
    try {
      const next = await addLLMDeployments({
        expected_revision: settings.revision,
        deployments,
      });
      setSettings(next);
      setAdding(false);
      setProvider(null);
      setDiscovery(null);
      setSelectedModels(new Set());
      onSaved();
      announce(t.modelsAdded(deployments.length));
    } catch (error) {
      setListError(errorText(error));
    } finally {
      setAddBusy(false);
    }
  };

  const saveDetected = async (deployment: LLMDeployment) => {
    if (settings === null) return;
    setListError(null);
    try {
      const next = await addLLMDeployments({
        expected_revision: settings.revision,
        deployments: [
          {
            model_name: deployment.routing_group,
            model: deployment.model,
            weight: deployment.weight,
          },
        ],
      });
      setSettings(next);
      onSaved();
      announce(t.saved);
    } catch (error) {
      setListError(errorText(error));
    }
  };

  const renderDeployment = (deployment: LLMDeployment, detected = false) => {
    const test = rowTests[deployment.deployment_id];
    const confirming = confirmDelete === deployment.deployment_id;
    return (
      <div className="prov" key={`${detected ? "detected" : "configured"}:${deployment.deployment_id}`}>
        <div className="prov-line">
          <span className="prov-name mono">{deployment.routing_group}</span>
          <span className="prov-model mono" title={deployment.model}>{deployment.model}</span>
          <span className="minibadge">{detected ? "session" : `w${deployment.weight}`}</span>
          <span className="prov-acts">
            <button type="button" className="rowact" disabled={test?.state === "busy"} onClick={() => void runTest(deployment)}>
              {test?.state === "busy" ? t.testing : t.test}
            </button>
            {detected ? (
              <button type="button" className="rowact" onClick={() => void saveDetected(deployment)}>{t.saveToConfig}</button>
            ) : (
              <>
                <button type="button" className="rowact" onClick={() => openEdit(deployment)}>{t.edit}</button>
                <button
                  type="button"
                  className={confirming ? "rowact warn" : "rowact"}
                  aria-label={confirming ? t.confirmDeleteModel(deployment.model) : t.deleteModel(deployment.model)}
                  onClick={() => void removeDeployment(deployment)}
                >
                  {confirming ? t.histConfirm : t.histDelete}
                </button>
              </>
            )}
          </span>
        </div>
        {test !== undefined && test.state !== "busy" && (
          <p className={test.state === "fail" ? "prov-detail mono fail" : "prov-detail mono"} role="status">
            {test.state} · {test.detail}
          </p>
        )}
      </div>
    );
  };

  return (
    <div
      className={closing ? "mdl-veil out" : "mdl-veil"}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      <div ref={cardRef} className="mdl settings-modal" role="dialog" aria-modal="true" aria-labelledby="mdl-title" tabIndex={-1}>
        <div className="mdl-head">
          <nav className="settings-crumbs mono" aria-label={t.breadcrumbAria}>
            {adding ? (
              <>
                <button type="button" onClick={resetAddFlow}>
                  {t.settingsTitle}
                </button>
                <span aria-hidden="true">/</span>
                {provider === null ? (
                  <h2 id="mdl-title">{t.addModels}</h2>
                ) : (
                  <>
                    <button type="button" onClick={backToProviders}>
                      {t.addModels}
                    </button>
                    <span aria-hidden="true">/</span>
                    <h2 id="mdl-title">{provider.label}</h2>
                  </>
                )}
              </>
            ) : (
              <h2 id="mdl-title">{t.settingsTitle}</h2>
            )}
          </nav>
          <button type="button" className="gearbtn" aria-label={t.close} title={t.close} onClick={requestClose}>
            <XIcon size={16} />
          </button>
        </div>

        <div className="mdl-body">
          {settings === null ? (
            <p className={loadError === null ? "mdl-dim mono" : "errline"}>{loadError ?? t.loading}</p>
          ) : adding ? (
            provider === null ? (
              <>
                {providers.length === 0 ? (
                  <p className="mdl-dim mono">{t.loading}</p>
                ) : (
                  <ProviderPicker
                    t={t}
                    providers={providers}
                    selectedId={null}
                    onSelect={chooseProvider}
                  />
                )}
                <button type="button" className="btn ghost" onClick={resetAddFlow}>
                  {t.cancel}
                </button>
              </>
            ) : (
              <>
                <section className="provider-detail">
                  <div className="provider-detail-head">
                    <div className="provider-detail-copy">
                      <h3>
                        {autoDiscoversModels
                          ? t.modelCatalogTitle
                          : t.connectProviderTitle(provider.label)}
                      </h3>
                      <p>
                        {t.providerDetailHint(
                          provider.kind,
                          provider.label,
                          provider.source,
                        )}
                      </p>
                    </div>
                    <button
                      type="button"
                      className={
                        discovery === null && !autoDiscoversModels
                          ? "btn primary"
                          : "btn ghost"
                      }
                      disabled={discovering || !canLoadModels}
                      onClick={() => void loadModels(discovery !== null)}
                    >
                      {discovering
                        ? t.loading
                        : discovery === null
                          ? t.loadModels
                          : t.refreshModels}
                    </button>
                  </div>

                  {showsApiKey && (
                    <div
                      className={
                        requiresPrimaryBase
                          ? "connection-fields"
                          : "connection-fields single"
                      }
                    >
                      <label className="fld">
                        <span className="lbl">
                          {t.setApiKey}
                          {provider.provider !== "custom" && (
                            <span className="required-field">{t.requiredField}</span>
                          )}
                        </span>
                        <input
                          type="password"
                          value={draftKey}
                          placeholder={t.providerKeyPh(provider.provider)}
                          autoComplete="off"
                          onChange={(event) => setDraftKey(event.target.value)}
                        />
                      </label>
                      {requiresPrimaryBase && (
                        <label className="fld">
                          <span className="lbl">
                            {t.setApiBase}
                            <span className="required-field">{t.requiredField}</span>
                          </span>
                          <input
                            type="url"
                            value={draftBase}
                            placeholder="https://example.com/v1"
                            onChange={(event) => setDraftBase(event.target.value)}
                          />
                        </label>
                      )}
                    </div>
                  )}

                  {!requiresPrimaryBase &&
                    !usesExistingConnection &&
                    provider.provider !== "ollama" && (
                      <details className="connection-options">
                        <summary className="mono">{t.customApiBase}</summary>
                        <label className="fld">
                          <span className="lbl">{t.setApiBase}</span>
                          <input
                            type="url"
                            value={draftBase}
                            placeholder="https://example.com/v1"
                            onChange={(event) => setDraftBase(event.target.value)}
                          />
                        </label>
                      </details>
                    )}

                  {provider.provider === "ollama" && (
                    <details className="connection-options">
                      <summary className="mono">{t.customApiBase}</summary>
                      <label className="fld">
                        <span className="lbl">{t.setApiBase}</span>
                        <input
                          type="url"
                          value={draftBase}
                          placeholder="http://127.0.0.1:11434"
                          onChange={(event) => setDraftBase(event.target.value)}
                        />
                      </label>
                    </details>
                  )}
                </section>

                {discovering && discovery === null && (
                  <div className="model-skeleton" role="status">
                    <span className="sr-only">{t.modelCatalogLoading}</span>
                    {[0, 1, 2, 3].map((index) => (
                      <span className="model-skeleton-row" key={index} aria-hidden="true">
                        <span />
                        <span />
                      </span>
                    ))}
                  </div>
                )}

                {discovery?.status === "unavailable" && (
                  <div className="discovery-state error" role="alert">
                    <strong>{t.modelsUnavailable}</strong>
                    {discovery.detail !== undefined && <span>{discovery.detail}</span>}
                  </div>
                )}
                {discovery !== null &&
                  discovery.status === "partial" && (
                    <div className="discovery-state" role="status">
                      <strong>{t.modelsPartial}</strong>
                      {discovery.detail !== undefined && <span>{discovery.detail}</span>}
                    </div>
                  )}

                {discovery !== null && discovery.status !== "unavailable" && (
                  <ModelPicker
                    t={t}
                    provider={provider.provider}
                    candidates={discovery.models}
                    deployments={settings.deployments}
                    apiBase={draftBase}
                    routingGroup={routingGroup}
                    weight={weight}
                    selected={selectedModels}
                    onRoutingGroup={setRoutingGroup}
                    onWeight={setWeight}
                    onSelected={setSelectedModels}
                  />
                )}
                <div className="set-actions detail-actions">
                  {discovery !== null && discovery.status !== "unavailable" && (
                    <button type="button" className="btn primary" disabled={addBusy || selectedModels.size === 0} onClick={() => void addSelected()}>
                      {addBusy ? t.saving : t.addModelsCount(selectedModels.size)}
                    </button>
                  )}
                  <button type="button" className="btn ghost" onClick={backToProviders}>
                    {t.cancel}
                  </button>
                </div>
              </>
            )
          ) : (
            <>
              <div className="settings-summary">
                <span className="mono">{t.deploymentsConfigured(settings.deployments.length)}</span>
                <button type="button" className="btn primary" onClick={() => setAdding(true)}>{t.addModels}</button>
              </div>
              {settings.deployments.length === 0 ? (
                <p className="mdl-dim mono">{t.setStatusNone}</p>
              ) : (
                <div className="provlist">{settings.deployments.map((deployment) => renderDeployment(deployment))}</div>
              )}
              {settings.detected.length > 0 && (
                <section className="detected-section">
                  <h3 className="picker-group mono">{t.detectedSession}</h3>
                  <div className="provlist">{settings.detected.map((deployment) => renderDeployment(deployment, true))}</div>
                </section>
              )}
              {editing !== null && (
                <form className="mdl-form" onSubmit={(event) => { event.preventDefault(); void saveEdit(); }}>
                  <div className="connection-fields">
                    <label className="fld"><span className="lbl">{t.routingGroup}</span><input value={editGroup} onChange={(event) => setEditGroup(event.target.value)} /></label>
                    <label className="fld"><span className="lbl">{t.setModel}</span><input value={editModel} onChange={(event) => setEditModel(event.target.value)} /></label>
                    <label className="fld"><span className="lbl">{t.setApiKey}</span><input type="password" value={editKey} placeholder={t.keepKeyHint} onChange={(event) => setEditKey(event.target.value)} /></label>
                    <label className="fld"><span className="lbl">{t.setApiBase}</span><input value={editBase} placeholder={t.keepBaseHint} onChange={(event) => setEditBase(event.target.value)} /></label>
                    <label className="fld"><span className="lbl">{t.weight}</span><input type="number" min={0} value={editWeight} onChange={(event) => setEditWeight(Math.max(0, Number(event.target.value) || 0))} /></label>
                  </div>
                  <div className="set-actions">
                    <button type="submit" className="btn primary" disabled={editBusy || editGroup.trim() === "" || editModel.trim() === ""}>{editBusy ? t.saving : t.save}</button>
                    <button type="button" className="btn ghost" onClick={() => setEditing(null)}>{t.cancel}</button>
                  </div>
                </form>
              )}
              <p className="mdl-src mono">{t.setSourceLbl} {settings.config_origin} · {settings.config_path}</p>
            </>
          )}
          {listError !== null && <p className="errline mdl-err" role="alert">{listError}</p>}
        </div>
      </div>
    </div>
  );
}

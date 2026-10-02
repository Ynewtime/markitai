// Model settings, as a 760px dialog with a breadcrumb: the configured models,
// then Add models → provider → model catalogue. Escape steps back one level.
// Every write carries the revision this draft started from; a conflict keeps
// the draft and reloads the lists, and the person decides to use the new revision.
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import {
  addDeployments,
  deleteDeployment,
  deleteProvider,
  discoverModels,
  fetchCredentials,
  fetchProviders,
  fetchSettings,
  isRevisionConflict,
  openConfig,
  probeModel,
  updateDeployment,
  updateProvider,
} from "../api/client.ts";
import type { Deployment, DiscoveryResult, NewDeployment, ProviderCard, SettingsView } from "../api/types.ts";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { serviceNote } from "../i18n/errors.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";
import { ModelPicker } from "./model-picker.tsx";
import { Notification, type NotificationModel } from "./notification.tsx";
import { providerLabel, ProviderPicker } from "./provider-picker.tsx";

const EXIT_MS = 120;
type Described = { text: string; detail: string };

function Secret({
  t,
  id,
  label,
  value,
  placeholder,
  disabled,
  onInput,
}: {
  t: Dict;
  id: string;
  label: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  onInput: (value: string) => void;
}) {
  const [shown, setShown] = useState(false);
  const toggle = shown ? t.concealField(label) : t.revealField(label);
  return (
    <div class="field">
      <label class="field-label" for={id}>
        {label}
      </label>
      <div class="secret">
        <input
          id={id}
          type={shown ? "text" : "password"}
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          spellcheck={false}
          onInput={(event) => onInput(event.currentTarget.value)}
        />
        <button type="button" class="secret-toggle" aria-label={toggle} title={toggle} disabled={disabled} onClick={() => setShown((value) => !value)}>
          <Icon name={shown ? "EyeSlash" : "Eye"} size={16} />
        </button>
      </div>
    </div>
  );
}

export function SettingsModal({
  t,
  locale,
  onClose,
  onSaved,
  announce,
  describe,
}: {
  t: Dict;
  locale: Locale;
  onClose: () => void;
  onSaved: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => Described;
}) {
  const card = useRef<HTMLDivElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const narrow = useMedia(NARROW);
  const [closing, setClosing] = useState(false);
  const [settings, setSettings] = useState<SettingsView | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [providers, setProviders] = useState<ProviderCard[]>([]);
  const [providersFailed, setProvidersFailed] = useState(false);
  // The revision every write of this draft is checked against.
  const [revision, setRevision] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [listError, setListError] = useState<Described | null>(null);
  const [listNote, setListNote] = useState<string | null>(null);

  const requestClose = useCallback(() => {
    if (closing) return;
    setClosing(true);
    setTimeout(onClose, EXIT_MS);
  }, [closing, onClose]);

  const refreshProviders = useCallback(async (refresh = false) => {
    try {
      setProviders(await fetchProviders(refresh));
      setProvidersFailed(false);
    } catch {
      setProvidersFailed(true);
    }
  }, []);

  useEffect(() => {
    let stale = false;
    fetchSettings().then(
      (view) => {
        if (stale) return;
        setSettings(view);
        setRevision(view.revision);
      },
      (error: unknown) => !stale && setLoadError(error),
    );
    void refreshProviders();
    return () => {
      stale = true;
    };
  }, [refreshProviders]);

  /** After a successful write: the new view and revision; the conflict is over. */
  const accept = (view: SettingsView) => {
    setSettings(view);
    setRevision(view.revision);
    setConflict(false);
    setListError(null);
    setListNote(null);
    onSaved();
  };

  /** A failed write. A revision conflict reloads the lists but keeps the draft and its revision. */
  const fail = async (error: unknown) => {
    setListNote(null);
    if (isRevisionConflict(error)) {
      try {
        setSettings(await fetchSettings());
        void refreshProviders(true);
      } catch {
        /* The conflict message still stands. */
      }
      setConflict(true);
      setListError({ text: t.settingsConflict, detail: describe(error).detail || describe(error).text });
      return;
    }
    setListError(describe(error));
  };

  // ---- model rows: test, edit, delete, save detected
  const [tests, setTests] = useState<Record<string, "busy" | "ok" | "fail">>({});
  const [testNote, setTestNote] = useState<NotificationModel | null>(null);
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), []);
  const settle = (id: string, state: "ok" | "fail") => {
    clearTimeout(timers.current[id]);
    setTests((previous) => ({ ...previous, [id]: state }));
    timers.current[id] = setTimeout(
      () =>
        setTests((previous) => {
          const next = { ...previous };
          delete next[id];
          return next;
        }),
      state === "ok" ? 1400 : 1800,
    );
  };
  const runTest = async (deployment: Deployment) => {
    const id = deployment.deployment_id;
    clearTimeout(timers.current.note);
    setTestNote(null);
    setTests((previous) => ({ ...previous, [id]: "busy" }));
    try {
      const result = await probeModel({ deployment_id: id });
      settle(id, result.ok ? "ok" : "fail");
      if (result.ok) {
        setTestNote({ tone: "success", title: t.modelTestPassed, message: t.modelTestReady(deployment.model) });
        timers.current.note = setTimeout(() => setTestNote(null), 3000);
      } else {
        const note = serviceNote(locale, result.detail);
        setTestNote({ tone: "error", title: t.modelTestFailed, message: note.text || t.modelTestFailed, detail: note.detail });
      }
    } catch (error) {
      settle(id, "fail");
      const text = describe(error);
      setTestNote({ tone: "error", title: t.modelTestFailed, message: text.text, detail: text.detail });
    }
  };

  const removeDeployment = async (deployment: Deployment) => {
    if (!settings || revision === null) return false;
    const index = settings.deployments.findIndex((item) => item.deployment_id === deployment.deployment_id);
    const rest = settings.deployments.filter((item) => item.deployment_id !== deployment.deployment_id);
    const successor = index < 0 ? null : ((rest[index] ?? rest[index - 1])?.deployment_id ?? null);
    try {
      accept(await deleteDeployment(deployment.deployment_id, revision));
      void refreshProviders(true);
      requestAnimationFrame(() => {
        const row = successor === null ? null : card.current?.querySelector<HTMLElement>(`[data-deployment="${CSS.escape(successor)}"] button`);
        (row ?? addButton.current)?.focus();
      });
      return true;
    } catch (error) {
      await fail(error);
      return false;
    }
  };

  const [editing, setEditing] = useState<Deployment | null>(null);
  const [editGroup, setEditGroup] = useState("");
  const [editModel, setEditModel] = useState("");
  const [editWeight, setEditWeight] = useState(1);
  const [editWeightDraft, setEditWeightDraft] = useState<string | null>(null);
  const [editBusy, setEditBusy] = useState(false);
  const openEdit = (deployment: Deployment) => {
    setEditing(deployment);
    setEditGroup(deployment.routing_group);
    setEditModel(deployment.model);
    setEditWeight(deployment.weight);
    setEditWeightDraft(null);
    setListError(null);
  };
  const saveEdit = async () => {
    if (!editing || revision === null || editBusy) return;
    setEditBusy(true);
    try {
      accept(
        await updateDeployment(editing.deployment_id, {
          model_name: editGroup.trim(),
          model: editModel.trim(),
          weight: editWeight,
          expected_revision: revision,
        }),
      );
      setEditing(null);
      announce(t.saved);
    } catch (error) {
      await fail(error);
    } finally {
      setEditBusy(false);
    }
  };

  const saveDetected = async (deployment: Deployment) => {
    if (revision === null) return;
    try {
      accept(await addDeployments(revision, [{ model_name: deployment.routing_group, model: deployment.model, weight: deployment.weight }]));
      announce(t.saved);
    } catch (error) {
      await fail(error);
    }
  };

  // ---- a saved provider's credentials. Unchanged fields are not sent (kept);
  // a field emptied on purpose is sent as null (cleared).
  const [provEdit, setProvEdit] = useState<ProviderCard | null>(null);
  const [provKey, setProvKey] = useState("");
  const [provBase, setProvBase] = useState("");
  const [provLoaded, setProvLoaded] = useState<{ key: string; base: string; placeholder: string } | null>(null);
  const [provBusy, setProvBusy] = useState(false);
  const provRequest = useRef(0);
  const openProvider = async (connection: ProviderCard) => {
    if (connection.provider_id === undefined) return;
    const request = ++provRequest.current;
    setProvEdit(connection);
    setProvKey("");
    setProvBase("");
    setProvLoaded(null);
    setEditing(null);
    setListError(null);
    try {
      const credentials = await fetchCredentials(connection.provider_id);
      if (request !== provRequest.current) return;
      setProvKey(credentials.api_key ?? "");
      setProvBase(credentials.api_base ?? "");
      setProvLoaded({ key: credentials.api_key ?? "", base: credentials.api_base ?? "", placeholder: credentials.api_base_placeholder ?? "" });
    } catch (error) {
      if (request !== provRequest.current) return;
      setProvEdit(null);
      await fail(error);
    }
  };
  const closeProvider = () => {
    provRequest.current++;
    setProvEdit(null);
    setProvLoaded(null);
  };
  const saveProvider = async () => {
    if (!provEdit?.provider_id || !provLoaded || revision === null || provBusy) return;
    const body: Record<string, unknown> = { expected_revision: revision };
    if (provKey.trim() !== provLoaded.key) body.api_key = provKey.trim() === "" ? null : provKey.trim();
    if (provBase.trim() !== provLoaded.base) body.api_base = provBase.trim() === "" ? null : provBase.trim();
    if (Object.keys(body).length === 1) {
      closeProvider();
      return;
    }
    setProvBusy(true);
    try {
      accept(await updateProvider(provEdit.provider_id, body));
      closeProvider();
      void refreshProviders(true);
      announce(t.providerSaved);
    } catch (error) {
      await fail(error);
    } finally {
      setProvBusy(false);
    }
  };
  const removeProvider = async (connection: ProviderCard) => {
    if (!connection.provider_id || revision === null) return false;
    try {
      accept(await deleteProvider(connection.provider_id, revision));
      if (provEdit?.provider_id === connection.provider_id) closeProvider();
      void refreshProviders(true);
      return true;
    } catch (error) {
      await fail(error);
      return false;
    }
  };

  // ---- add models: provider, then its catalogue
  const [adding, setAdding] = useState(false);
  const [provider, setProvider] = useState<ProviderCard | null>(null);
  const [draftKey, setDraftKey] = useState("");
  const [draftBase, setDraftBase] = useState("");
  const [discovery, setDiscovery] = useState<DiscoveryResult | null>(null);
  const [discovering, setDiscovering] = useState(false);
  const [selected, setSelected] = useState(new Set<string>());
  const [group, setGroup] = useState("default");
  const [weight, setWeight] = useState(1);
  const [addBusy, setAddBusy] = useState(false);
  const autoLoaded = useRef<string | null>(null);
  const existing =
    provider !== null && (provider.provider_id !== undefined || provider.deployment_id !== undefined || provider.kind === "environment");
  const ollama = provider?.provider === "ollama";
  // A local server with a known address (Ollama, LM Studio) needs no key.
  const local = provider?.key_optional === true;
  // The provider lists no models: IDs are entered by hand.
  const manualOnly = provider !== null && provider.supports_discovery === false;
  const autoLoads = !manualOnly && (existing || (local && Boolean(provider?.default_base)));
  // Azure, a custom endpoint and servers without a default address (vLLM) need one.
  const needsBase =
    provider !== null && (provider.provider === "azure" || provider.provider === "custom" || (provider.default_base === null && !existing));
  const showsKey = provider !== null && !existing && !ollama;
  const keyRequired = provider !== null && provider.provider !== "custom" && !local;
  const canLoad =
    provider !== null &&
    !manualOnly &&
    (autoLoads ||
      (needsBase ? draftBase.trim() !== "" && (!keyRequired || draftKey.trim() !== "") : !keyRequired || draftKey.trim() !== ""));

  const linked = (connection: ProviderCard) => ({
    ...(connection.provider_id === undefined || connection.provider_id.startsWith("legacy:") ? {} : { provider_id: connection.provider_id }),
    ...(connection.deployment_id === undefined ? {} : { deployment_id: connection.deployment_id }),
  });

  const resetAdd = () => {
    setAdding(false);
    setProvider(null);
    closeProvider();
    setDiscovery(null);
    setSelected(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const backToProviders = () => {
    setProvider(null);
    setDiscovery(null);
    setSelected(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const focusHeading = () => requestAnimationFrame(() => card.current?.querySelector<HTMLElement>("#settings-title")?.focus());

  useModal(card, () => {
    if (adding && provider !== null) {
      backToProviders();
      focusHeading();
    } else if (adding) {
      resetAdd();
      focusHeading();
    } else requestClose();
  });

  const choose = (next: ProviderCard) => {
    setProvider(next);
    closeProvider();
    setDraftKey(next.credential ?? "");
    setDraftBase("");
    setDiscovery(null);
    setSelected(new Set());
    setListError(null);
    autoLoaded.current = null;
  };

  const load = async (refresh = false) => {
    if (!provider || discovering) return;
    setDiscovering(true);
    setListError(null);
    try {
      const result = await discoverModels({
        provider: provider.provider,
        ...linked(provider),
        ...(draftKey.trim() ? { api_key: draftKey.trim() } : {}),
        ...(draftBase.trim() ? { api_base: draftBase.trim() } : {}),
        refresh,
      });
      setDiscovery(result);
      const available = new Set(result.models.map((candidate) => candidate.model));
      setSelected((previous) => new Set([...previous].filter((model) => available.has(model))));
    } catch (error) {
      await fail(error);
    } finally {
      setDiscovering(false);
    }
  };

  useEffect(() => {
    if (!provider || !autoLoads || discovery !== null || discovering || autoLoaded.current === provider.id) return;
    autoLoaded.current = provider.id;
    void load(false);
    // `load` reads the current draft; the guard above keys the request by provider.
  }, [provider, autoLoads, discovery, discovering]);

  const addSelected = async () => {
    if (!provider || !selected.size || revision === null || addBusy) return;
    setAddBusy(true);
    const deployments: NewDeployment[] = [...selected].map((model) => ({
      model_name: group.trim() || "default",
      model,
      weight,
      provider: provider.provider,
      ...(provider.provider_id === undefined || provider.provider_id.startsWith("legacy:") ? {} : { credential_provider_id: provider.provider_id }),
      ...(provider.deployment_id === undefined ? {} : { credential_deployment_id: provider.deployment_id }),
      ...(!existing && draftKey.trim() ? { api_key: draftKey.trim() } : {}),
      ...(!existing && draftBase.trim() ? { api_base: draftBase.trim() } : {}),
      ...(existing && provider.kind === "environment" && draftKey.trim() ? { api_key: draftKey.trim() } : {}),
    }));
    try {
      accept(await addDeployments(revision, deployments));
      setAdding(false);
      setProvider(null);
      setDiscovery(null);
      setSelected(new Set());
      void refreshProviders(true);
      announce(t.modelsAdded(deployments.length));
    } catch (error) {
      await fail(error);
    } finally {
      setAddBusy(false);
    }
  };

  const source = settings && (
    <p class="dialog-source">
      {t.setSourceLbl} {t.originLabel(settings.config_origin)} ·{" "}
      <button
        type="button"
        class="dotted-link"
        title={t.openConfigFile}
        onClick={() =>
          void openConfig().catch((error: unknown) => {
            setListError(describe(error));
          })
        }
      >
        {settings.config_path}
      </button>
    </p>
  );

  const modelRow = (deployment: Deployment, detected = false) => {
    const test = tests[deployment.deployment_id];
    return (
      <div class="model-row" data-deployment={deployment.deployment_id} key={`${detected ? "d" : "c"}:${deployment.deployment_id}`}>
        <div class="model-line">
          <span class="model-group">{deployment.routing_group}</span>
          <span class="model-id" title={deployment.model}>
            {deployment.model}
          </span>
          <span class="mini-pill" title={t.weightHint}>
            {detected ? t.sessionBadge : narrow ? t.modelWeightShort(deployment.weight) : t.modelWeight(deployment.weight)}
          </span>
          <span class="text-actions">
            <button
              type="button"
              class={`text-btn test-btn${test ? ` is-${test}` : ""}`}
              disabled={test === "busy"}
              aria-label={test === "busy" ? t.testing : test === "ok" ? t.modelTestPassed : test === "fail" ? t.modelTestFailed : t.test}
              onClick={() => void runTest(deployment)}
            >
              <span class="test-swap" key={test ?? "idle"}>
                {test === "busy" ? (
                  <span class="spinner" aria-hidden="true" />
                ) : test === "ok" ? (
                  <Icon name="CheckBold" size={16} />
                ) : test === "fail" ? (
                  <Icon name="WarningFill" size={15} />
                ) : (
                  t.test
                )}
              </span>
            </button>
            {detected ? (
              <button type="button" class="text-btn" onClick={() => void saveDetected(deployment)}>
                {t.saveToConfig}
              </button>
            ) : (
              <>
                <button type="button" class="text-btn" onClick={() => openEdit(deployment)}>
                  {t.edit}
                </button>
                <ConfirmPopover
                  triggerLabel={t.deleteModel(deployment.model)}
                  title={t.deleteModelTitle(deployment.model)}
                  description={t.deleteModelDescription}
                  confirmLabel={t.deletePermanently}
                  cancelLabel={t.cancel}
                  busyLabel={t.deleting}
                  onConfirm={() => removeDeployment(deployment)}
                />
              </>
            )}
          </span>
        </div>
      </div>
    );
  };

  const providerEditor = provEdit && (
    <form
      key={provEdit.provider_id}
      class="dialog-form"
      onSubmit={(event) => {
        event.preventDefault();
        void saveProvider();
      }}
    >
      <h3 class="group-title">{t.editProvider(providerLabel(t, provEdit))}</h3>
      <div class="field-grid">
        <Secret
          t={t}
          id="provider-key"
          label={t.setApiKey}
          value={provKey}
          placeholder={provLoaded ? t.setKeyPh : t.loading}
          disabled={!provLoaded}
          onInput={setProvKey}
        />
        <div class="field">
          <label class="field-label" for="provider-base">
            {t.setApiBase}
          </label>
          <input
            id="provider-base"
            type="url"
            inputMode="url"
            value={provBase}
            placeholder={provLoaded ? provLoaded.placeholder || t.setBasePh : t.loading}
            disabled={!provLoaded}
            autoComplete="off"
            spellcheck={false}
            onInput={(event) => setProvBase(event.currentTarget.value)}
          />
        </div>
      </div>
      <div class="form-actions">
        <button type="submit" class="btn btn-primary" disabled={provBusy || !provLoaded}>
          {provBusy ? t.saving : t.save}
        </button>
        <button type="button" class="btn btn-ghost" onClick={closeProvider}>
          {t.cancel}
        </button>
      </div>
    </form>
  );

  const label = provider ? providerLabel(t, provider) : "";

  return (
    <div
      class={closing ? "veil is-leaving" : "veil"}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      {testNote && (
        <Notification
          note={testNote}
          closeLabel={t.close}
          onClose={() => {
            clearTimeout(timers.current.note);
            setTestNote(null);
          }}
        />
      )}
      <div ref={card} class="dialog dialog-settings" role="dialog" aria-modal="true" aria-labelledby="settings-title" tabIndex={-1}>
        <div class="dialog-head">
          <nav class="crumbs" aria-label={t.breadcrumbAria}>
            {adding ? (
              <>
                <button type="button" onClick={resetAdd}>
                  {t.settingsTitle}
                </button>
                <span aria-hidden="true">/</span>
                {provider === null ? (
                  <h2 id="settings-title" tabIndex={-1}>
                    {t.addModels}
                  </h2>
                ) : (
                  <>
                    <button type="button" onClick={backToProviders}>
                      {t.addModels}
                    </button>
                    <span aria-hidden="true">/</span>
                    <h2 id="settings-title" tabIndex={-1}>
                      {label}
                    </h2>
                  </>
                )}
              </>
            ) : (
              <h2 id="settings-title" tabIndex={-1}>
                {t.settingsTitle}
              </h2>
            )}
          </nav>
          <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={requestClose}>
            <Icon name="X" size={16} />
          </button>
        </div>

        <div class="dialog-body">
          {settings === null ? (
            <p class={loadError ? "line-error" : "dialog-dim"}>{loadError ? describe(loadError).text : t.loading}</p>
          ) : adding ? (
            provider === null ? (
              <>
                {providers.length === 0 ? (
                  providersFailed ? (
                    <>
                      <p class="line-error" role="alert">
                        {t.providersLoadFailed}
                      </p>
                      <button type="button" class="btn btn-ghost" onClick={() => void refreshProviders()}>
                        {t.retryLoad}
                      </button>
                    </>
                  ) : (
                    <p class="dialog-dim">{t.loading}</p>
                  )
                ) : (
                  <ProviderPicker t={t} providers={providers} onSelect={choose} onEdit={(connection) => void openProvider(connection)} onDelete={removeProvider} />
                )}
                {providerEditor}
                {!provEdit && (
                  <button type="button" class="btn btn-ghost dialog-end" onClick={resetAdd}>
                    {t.cancel}
                  </button>
                )}
              </>
            ) : (
              <>
                <section class="provider-detail">
                  <div class="provider-detail-head">
                    <div class="provider-detail-copy">
                      <h3>{autoLoads ? t.modelCatalogTitle : t.connectProviderTitle(label)}</h3>
                      <p>{t.providerDetailHint(provider.kind, provider.provider, provider.source, { local, manualOnly, needsBase: needsBase && provider.provider !== "azure" && provider.provider !== "custom" })}</p>
                      {(provider.kind === "common" || provider.kind === "compatible") && (provider.default_base || provider.key_variable) && (
                        <p class="provider-facts">
                          {provider.default_base && <span class="provider-fact">{provider.default_base}</span>}
                          {provider.key_variable && <span class="provider-fact">{local ? t.optionalKeyVariable(provider.key_variable) : provider.key_variable}</span>}
                        </p>
                      )}
                    </div>
                    {!manualOnly && (
                      <button
                        type="button"
                        class={discovery === null && !autoLoads ? "btn btn-primary" : "btn btn-ghost"}
                        disabled={discovering || !canLoad}
                        onClick={() => void load(discovery !== null)}
                      >
                        {discovering ? t.loading : discovery === null ? t.loadModels : t.refreshModels}
                      </button>
                    )}
                  </div>
                  {showsKey && (
                    <div class={needsBase ? "field-grid" : "field-grid is-single"}>
                      <label class="field">
                        <span class="field-label">
                          {t.setApiKey}
                          {keyRequired && <span class="field-required">{t.requiredField}</span>}
                        </span>
                        <input type="password" value={draftKey} placeholder={t.providerKeyPh(provider.key_variable ?? null, provider.provider)} autoComplete="off" onInput={(event) => setDraftKey(event.currentTarget.value)} />
                      </label>
                      {needsBase && (
                        <label class="field">
                          <span class="field-label">
                            {t.setApiBase}
                            <span class="field-required">{t.requiredField}</span>
                          </span>
                          <input
                            type="url"
                            value={draftBase}
                            placeholder={provider.provider === "hosted_vllm" ? "http://127.0.0.1:8000/v1" : "https://example.com/v1"}
                            onInput={(event) => setDraftBase(event.currentTarget.value)}
                          />
                        </label>
                      )}
                    </div>
                  )}
                  {!needsBase && !existing && (
                    <details class="disclosure">
                      <summary>{t.customApiBase}</summary>
                      <label class="field disclosure-field">
                        <span class="field-label">{t.setApiBase}</span>
                        <input
                          type="url"
                          value={draftBase}
                          placeholder={ollama ? "http://127.0.0.1:11434" : (provider.default_base ?? "https://example.com/v1")}
                          onInput={(event) => setDraftBase(event.currentTarget.value)}
                        />
                      </label>
                    </details>
                  )}
                </section>
                {discovering && discovery === null && (
                  <div class="skeleton" role="status">
                    <span class="sr-only">{t.modelCatalogLoading}</span>
                    {[0, 1, 2, 3].map((index) => (
                      <span class="skeleton-row" key={index} aria-hidden="true">
                        <span />
                        <span />
                      </span>
                    ))}
                  </div>
                )}
                {discovery && discovery.status !== "ok" && (
                  <div class={discovery.status === "unavailable" ? "discovery-note is-error" : "discovery-note"} role={discovery.status === "unavailable" ? "alert" : "status"}>
                    <strong>{discovery.status === "unavailable" ? t.modelsUnavailable : t.modelsPartial}</strong>
                    {discovery.detail && <span title={serviceNote(locale, discovery.detail).detail || undefined}>{serviceNote(locale, discovery.detail).text}</span>}
                  </div>
                )}
                {/* An unavailable catalogue still leaves manual model entry. */}
                {(discovery || manualOnly) && (
                  <ModelPicker
                    t={t}
                    manualOnly={manualOnly}
                    provider={provider.provider}
                    candidates={discovery?.models ?? []}
                    deployments={settings.deployments}
                    apiBase={draftBase}
                    group={group}
                    weight={weight}
                    selected={selected}
                    onGroup={setGroup}
                    onWeight={setWeight}
                    onSelected={setSelected}
                  />
                )}
                <div class="form-actions form-actions-ruled">
                  {(discovery || manualOnly) && (
                    <button type="button" class="btn btn-primary" disabled={addBusy || !selected.size} onClick={() => void addSelected()}>
                      {addBusy ? t.saving : t.addModelsCount(selected.size)}
                    </button>
                  )}
                  <button type="button" class="btn btn-ghost" onClick={backToProviders}>
                    {t.cancel}
                  </button>
                </div>
              </>
            )
          ) : (
            <>
              <div class="settings-summary">
                <span>{t.modelsConfigured(settings.deployments.length)}</span>
                <button ref={addButton} type="button" class="btn btn-primary" onClick={() => setAdding(true)}>
                  {t.addModels}
                </button>
              </div>
              {settings.deployments.length === 0 ? (
                <p class="dialog-dim">{t.setStatusNone}</p>
              ) : (
                <div class="model-rows">{settings.deployments.map((deployment) => modelRow(deployment))}</div>
              )}
              {settings.detected.length > 0 && (
                <section class="detected">
                  <h3 class="group-title">{t.detectedSession}</h3>
                  <div class="model-rows">{settings.detected.map((deployment) => modelRow(deployment, true))}</div>
                </section>
              )}
              {editing && (
                <form
                  class="dialog-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void saveEdit();
                  }}
                >
                  <div class="field-grid">
                    <label class="field">
                      <span class="field-label">{t.routingGroup}</span>
                      <input value={editGroup} onInput={(event) => setEditGroup(event.currentTarget.value)} />
                    </label>
                    <label class="field">
                      <span class="field-label">{t.setModel}</span>
                      <input value={editModel} onInput={(event) => setEditModel(event.currentTarget.value)} />
                    </label>
                    <label class="field">
                      <span class="field-label">{t.weight}</span>
                      <input
                        type="number"
                        min={0}
                        value={editWeightDraft ?? editWeight}
                        onInput={(event) => {
                          setEditWeightDraft(event.currentTarget.value);
                          setEditWeight(Math.max(0, Math.floor(Number(event.currentTarget.value)) || 0));
                        }}
                        onBlur={() => setEditWeightDraft(null)}
                      />
                    </label>
                  </div>
                  <div class="form-actions">
                    {source}
                    <button type="submit" class="btn btn-primary" disabled={editBusy || !editGroup.trim() || !editModel.trim()}>
                      {editBusy ? t.saving : t.save}
                    </button>
                    <button type="button" class="btn btn-ghost" onClick={() => setEditing(null)}>
                      {t.cancel}
                    </button>
                  </div>
                </form>
              )}
              {!editing && source}
            </>
          )}
          {listError && (
            <p class="line-error dialog-error" role="alert" title={listError.detail || undefined}>
              {listError.text}
              {conflict && (
                <>
                  {" · "}
                  <button
                    type="button"
                    class="text-link"
                    onClick={() => {
                      if (settings) setRevision(settings.revision);
                      setConflict(false);
                      setListError(null);
                      setListNote(t.draftKept);
                    }}
                  >
                    {t.useCurrentRevision}
                  </button>
                </>
              )}
            </p>
          )}
          {listNote && (
            <p class="line-note" role="status">
              {listNote}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

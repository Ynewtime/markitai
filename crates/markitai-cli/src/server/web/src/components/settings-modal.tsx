// Model settings, as a 760px dialog with a breadcrumb: the configured models,
// then Add models → provider → model catalogue. A model row's Edit opens the
// provider's page the same way. Escape steps back one level.
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
  updateProvider,
} from "../api/client.ts";
import type { Deployment, DiscoveryResult, NewDeployment, ProviderCard, SettingsView } from "../api/types.ts";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { serviceNote } from "../i18n/errors.ts";
import { endpointFor, sameEndpoint } from "../lib/models.ts";
import {
  CredentialInputError,
  deploymentCredentials,
  detectedDeployment,
  discoveryRequest,
  providerDraft,
  providerUpdate,
  type ProviderChange,
  type ProviderDraft,
} from "../lib/provider-credentials.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";
import { ModelPicker } from "./model-picker.tsx";
import { Notification, type NotificationModel } from "./notification.tsx";
import { manageable, providerLabel, ProviderPicker } from "./provider-picker.tsx";

const EXIT_MS = 120;
type Described = { text: string; detail: string };

function Secret({
  t,
  id,
  label,
  value,
  placeholder,
  disabled,
  required = false,
  onInput,
}: {
  t: Dict;
  id: string;
  label: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
  onInput: (value: string) => void;
}) {
  const [shown, setShown] = useState(false);
  const toggle = shown ? t.concealField(label) : t.revealField(label);
  return (
    <div class="field">
      <label class="field-label" for={id}>
        {label}
        {required && <span class="field-required">{t.requiredField}</span>}
      </label>
      <div class="secret">
        <input
          id={id}
          type={shown ? "text" : "password"}
          required={required}
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

  const refreshProviders = useCallback(async (refresh = false): Promise<ProviderCard[] | null> => {
    try {
      const list = await fetchProviders(refresh);
      setProviders(list);
      setProvidersFailed(false);
      return list;
    } catch {
      setProvidersFailed(true);
      return null;
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

  const credentialText = (error: CredentialInputError) =>
    ({ environment_reference: t.literalCredentialsOnly, endpoint_key_required: t.endpointKeyRequired, key_required: t.keyRequired })[error.reason];

  /** A failed write. A revision conflict reloads the lists but keeps the draft and its revision. */
  const fail = async (error: unknown) => {
    setListNote(null);
    if (error instanceof CredentialInputError) {
      setListError({ text: credentialText(error), detail: "" });
      return;
    }
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

  // Editing opens the provider's own page in the add flow: its connection and
  // catalogue, where more of its models can be added.
  const openEdit = (deployment: Deployment) => {
    const name = deployment.model.includes("/") ? deployment.model.split("/", 1)[0]!.toLowerCase() : "openai";
    const cards = providers.filter((card) => card.provider === name);
    const card =
      cards.find((card) => manageable(card) && card.api_base != null && card.api_base === deployment.api_base) ??
      cards.find(manageable) ??
      cards[0];
    if (!card) return;
    setAdding(true);
    choose(card);
  };

  const saveDetected = async (deployment: Deployment) => {
    if (revision === null) return;
    try {
      accept(await addDeployments(revision, [detectedDeployment(deployment)]));
      announce(t.saved);
    } catch (error) {
      await fail(error);
    }
  };

  const removeProvider = async (connection: ProviderCard) => {
    if (!connection.provider_id || revision === null) return false;
    try {
      accept(await deleteProvider(connection.provider_id, revision));
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
  /** Models this group and endpoint already route that the reader unticked. */
  const [dropped, setDropped] = useState(new Set<string>());
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

  // ---- a saved provider's connection, at the top of its page. The key and the
  // address change independently; a write sends only the field that changes,
  // and moving the address asks for the key to use there.
  const [conn, setConn] = useState<ProviderDraft | null>(null);
  const [connFailed, setConnFailed] = useState(false);
  const [connMode, setConnMode] = useState<"key" | "base" | null>(null);
  const [connKey, setConnKey] = useState("");
  const [connBase, setConnBase] = useState("");
  const [connBusy, setConnBusy] = useState(false);
  const [connError, setConnError] = useState<string | null>(null);
  const connRequest = useRef(0);
  const saved = provider !== null && manageable(provider);

  const closeConnEdit = () => {
    setConnMode(null);
    setConnKey("");
    setConnBase("");
    setConnError(null);
  };
  const resetConnection = () => {
    connRequest.current++;
    setConn(null);
    setConnFailed(false);
    closeConnEdit();
  };
  const loadConnection = async (connection: ProviderCard) => {
    if (connection.provider_id === undefined) return;
    const request = ++connRequest.current;
    setConn(null);
    setConnFailed(false);
    try {
      const credentials = await fetchCredentials(connection.provider_id);
      if (request === connRequest.current) setConn(providerDraft(credentials));
    } catch (error) {
      if (request !== connRequest.current) return;
      setConnFailed(true);
      await fail(error);
    }
  };
  const openConnEdit = (mode: "key" | "base", base = "") => {
    setConnMode(mode);
    setConnKey("");
    setConnBase(base);
    setConnError(null);
    requestAnimationFrame(() => card.current?.querySelector<HTMLElement>(mode === "key" ? "#conn-key" : "#conn-base")?.focus());
  };
  const saveConnection = async (change: ProviderChange, done: string): Promise<boolean> => {
    const current = provider;
    if (!current?.provider_id || !conn || revision === null || connBusy) return false;
    let body: Record<string, unknown> | null;
    try {
      body = providerUpdate(conn, change, revision);
    } catch (error) {
      if (!(error instanceof CredentialInputError)) throw error;
      setConnError(credentialText(error));
      return false;
    }
    if (body === null) {
      closeConnEdit();
      return true;
    }
    setConnBusy(true);
    setConnError(null);
    try {
      accept(await updateProvider(current.provider_id, body));
      closeConnEdit();
      announce(done);
      const list = await refreshProviders(true);
      // A legacy connection is saved under a new identity: pick it again from the list.
      if (current.provider_id.startsWith("legacy:")) {
        backToProviders();
        focusHeading();
        return true;
      }
      const next = list?.find((item) => item.id === current.id) ?? current;
      setProvider(next);
      void loadConnection(next);
      if (next.supports_discovery !== false) void load(discovery !== null, next);
      return true;
    } catch (error) {
      if (isRevisionConflict(error)) await fail(error);
      else setConnError(describe(error).text);
      return false;
    } finally {
      setConnBusy(false);
    }
  };

  const resetAdd = () => {
    setAdding(false);
    setProvider(null);
    resetConnection();
    setDiscovery(null);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const backToProviders = () => {
    setProvider(null);
    resetConnection();
    setDiscovery(null);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const focusHeading = () => requestAnimationFrame(() => card.current?.querySelector<HTMLElement>("#settings-title")?.focus());

  useModal(card, () => {
    if (connMode !== null) {
      // An open key or address form closes first; its toggle takes the focus back.
      const toggle = card.current?.querySelector<HTMLElement>(`[aria-controls="conn-${connMode}-form"]`);
      closeConnEdit();
      requestAnimationFrame(() => toggle?.focus());
    } else if (adding && provider !== null) {
      backToProviders();
      focusHeading();
    } else if (adding) {
      resetAdd();
      focusHeading();
    } else requestClose();
  });

  const choose = (next: ProviderCard, focus = "#settings-title") => {
    setProvider(next);
    resetConnection();
    if (manageable(next)) void loadConnection(next);
    requestAnimationFrame(() => card.current?.querySelector<HTMLElement>(focus)?.focus());
    setDraftKey("");
    setDraftBase("");
    setDiscovery(null);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };

  const load = async (refresh = false, from = provider) => {
    if (!from || discovering) return;
    setDiscovering(true);
    setListError(null);
    try {
      const result = await discoverModels(discoveryRequest(from, draftKey, draftBase, refresh));
      setDiscovery(result);
      const available = new Set(result.models.map((candidate) => candidate.model));
      setSelected((previous) => new Set([...previous].filter((model) => available.has(model))));
      setDropped((previous) => new Set([...previous].filter((model) => available.has(model))));
    } catch (error) {
      await fail(error);
    } finally {
      setDiscovering(false);
    }
  };

  useEffect(() => {
    if (!adding || !provider || !autoLoads || discovery !== null || discovering || autoLoaded.current === provider.id) return;
    autoLoaded.current = provider.id;
    void load(false);
    // `load` reads the current draft; the guard above keys the request by provider.
  }, [adding, provider, autoLoads, discovery, discovering]);

  /** The routing entry of one configured model, for a pending removal. */
  const endpoint = endpointFor(draftBase, provider?.api_base);
  const deploymentId = (model: string) =>
    settings?.deployments.find((item) => item.model === model && item.routing_group === group.trim() && sameEndpoint(item.api_base, endpoint))?.deployment_id ?? null;

  const addSelected = async () => {
    if (!provider || (!selected.size && !dropped.size) || revision === null || addBusy) return;
    let credentials: Partial<NewDeployment>;
    try {
      credentials = deploymentCredentials(provider, draftKey, draftBase);
    } catch (error) {
      await fail(error);
      return;
    }
    setAddBusy(true);
    const deployments: NewDeployment[] = [...selected].map((model) => ({
      model_name: group.trim() || "default",
      model,
      weight,
      ...credentials,
    }));
    try {
      // Removals first, one write at a time: each returns the revision the next
      // call has to quote.
      let current = revision;
      let removed = 0;
      for (const model of dropped) {
        const id = deploymentId(model);
        if (id === null) continue;
        const view = await deleteDeployment(id, current);
        accept(view);
        current = view.revision;
        removed += 1;
      }
      if (deployments.length) accept(await addDeployments(current, deployments));
      setAdding(false);
      setProvider(null);
      setDiscovery(null);
      setSelected(new Set());
      setDropped(new Set());
      void refreshProviders(true);
      announce([deployments.length ? t.modelsAdded(deployments.length) : "", removed ? t.modelsRemoved(removed) : ""].filter(Boolean).join(" · "));
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
              class={`row-icon test-btn${test ? ` is-${test}` : ""}`}
              disabled={test === "busy"}
              aria-label={test === "busy" ? t.testing : test === "ok" ? t.modelTestPassed : test === "fail" ? t.modelTestFailed : t.testModel(deployment.model)}
              title={test === "busy" ? t.testing : test === "ok" ? t.modelTestPassed : test === "fail" ? t.modelTestFailed : t.test}
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
                  <Icon name="PlugsConnected" size={15} />
                )}
              </span>
            </button>
            {detected ? (
              <button type="button" class="text-btn" onClick={() => void saveDetected(deployment)}>
                {t.saveToConfig}
              </button>
            ) : (
              <>
                <button
                  type="button"
                  class="row-icon edit-btn"
                  aria-label={t.editModel(deployment.model)}
                  title={t.edit}
                  onClick={() => openEdit(deployment)}
                >
                  <Icon name="PencilSimple" size={15} />
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

  const label = provider ? providerLabel(t, provider) : "";

  const status = (listError || listNote) && (
    <>
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
    </>
  );

  const picking = adding && provider !== null && (discovery !== null || manualOnly);
  const footer =
    settings === null || !adding ? null : provider === null ? (
      <div class="form-actions">
        <button type="button" class="btn btn-ghost" onClick={resetAdd}>
          {t.cancel}
        </button>
      </div>
    ) : (
      <div class="form-actions">
        {picking && (selected.size > 0 || dropped.size > 0) && (
          <p class="picker-count">
            {selected.size > 0 && <span>{t.modelsToAdd(selected.size)}</span>}
            {selected.size > 0 && dropped.size > 0 && " · "}
            {dropped.size > 0 && <span>{t.modelsToRemove(dropped.size)}</span>}
          </p>
        )}
        <button type="button" class="btn btn-ghost" onClick={backToProviders}>
          {t.cancel}
        </button>
        {picking && (
          <button type="button" class="btn btn-primary" disabled={addBusy || (!selected.size && !dropped.size)} onClick={() => void addSelected()}>
            {addBusy ? t.saving : t.save}
          </button>
        )}
      </div>
    );

  const keyText = (draft: ProviderDraft) =>
    draft.key.state === "saved"
      ? t.keySaved(draft.key.ending ?? "")
      : draft.key.state === "environment"
        ? t.fromEnvironment(draft.key.variable)
        : t.notSet;
  const baseText = (draft: ProviderDraft) =>
    draft.baseVariable !== null
      ? t.fromEnvironment(draft.baseVariable)
      : draft.base || (draft.placeholder ? t.defaultAddress(draft.placeholder) : t.notSet);

  const connection = saved && (
    <section class="connection" aria-labelledby="connection-title" aria-busy={connBusy || undefined}>
      <h3 id="connection-title" class="group-title" tabIndex={-1}>
        {t.connectionTitle}
      </h3>
      {conn === null ? (
        <p class={connFailed ? "line-error" : "dialog-dim"}>{connFailed ? t.providersLoadFailed : t.loading}</p>
      ) : (
        <div class="conn-rows">
          <div class="conn-row">
            <span class="conn-label">{t.setApiKey}</span>
            <span class="conn-value">{keyText(conn)}</span>
            <span class="text-actions">
              <button
                type="button"
                class="text-btn"
                aria-expanded={connMode === "key"}
                aria-controls={connMode === "key" ? "conn-key-form" : undefined}
                disabled={connBusy}
                onClick={() => (connMode === "key" ? closeConnEdit() : openConnEdit("key"))}
              >
                {conn.key.state === "none" || (conn.key.state === "environment" && !conn.key.stored) ? t.addKey : t.replaceKey}
              </button>
              {(conn.key.state === "saved" || (conn.key.state === "environment" && conn.key.stored)) && (
                <ConfirmPopover
                  triggerLabel={t.removeKeyLabel(label)}
                  triggerText={t.removeKey}
                  title={t.removeKeyTitle}
                  description={t.removeKeyDescription}
                  confirmLabel={t.removeKey}
                  cancelLabel={t.cancel}
                  busyLabel={t.removing}
                  disabled={connBusy}
                  onConfirm={() => saveConnection({ kind: "removeKey" }, t.keyRemoved)}
                />
              )}
            </span>
          </div>
          {connMode === "key" && (
            <form
              id="conn-key-form"
              class="conn-edit is-row"
              onSubmit={(event) => {
                event.preventDefault();
                void saveConnection({ kind: "key", key: connKey }, t.providerSaved);
              }}
            >
              <div class="field-grid is-single">
                <Secret t={t} id="conn-key" label={t.newApiKey} value={connKey} placeholder={t.setKeyPh} disabled={connBusy} onInput={setConnKey} />
              </div>
              <div class="form-actions">
                <button type="submit" class="btn btn-primary" disabled={connBusy || !connKey.trim()}>
                  {connBusy ? t.saving : t.save}
                </button>
                <button type="button" class="btn btn-ghost" disabled={connBusy} onClick={closeConnEdit}>
                  {t.cancel}
                </button>
              </div>
            </form>
          )}
          <div class="conn-row">
            <span class="conn-label">{t.setApiBase}</span>
            <span class="conn-value">{baseText(conn)}</span>
            <span class="text-actions">
              <button
                type="button"
                class="text-btn"
                aria-expanded={connMode === "base"}
                aria-controls={connMode === "base" ? "conn-base-form" : undefined}
                disabled={connBusy}
                onClick={() => (connMode === "base" ? closeConnEdit() : openConnEdit("base", conn.base))}
              >
                {t.editAddress}
              </button>
              {(conn.base !== "" || conn.baseRetained) && conn.placeholder !== "" && (
                <button type="button" class="text-btn" disabled={connBusy} onClick={() => openConnEdit("base", "")}>
                  {t.resetAddress}
                </button>
              )}
            </span>
          </div>
          {connMode === "base" && (
            <form
              id="conn-base-form"
              class="conn-edit"
              onSubmit={(event) => {
                event.preventDefault();
                void saveConnection({ kind: "base", base: connBase, key: connKey }, t.providerSaved);
              }}
            >
              <div class="field-grid">
                <div class="field">
                  <label class="field-label" for="conn-base">
                    {t.setApiBase}
                  </label>
                  <input
                    id="conn-base"
                    type="url"
                    inputMode="url"
                    value={connBase}
                    placeholder={conn.placeholder || "https://example.com/v1"}
                    disabled={connBusy}
                    autoComplete="off"
                    spellcheck={false}
                    onInput={(event) => setConnBase(event.currentTarget.value)}
                  />
                </div>
                <Secret
                  t={t}
                  id="conn-base-key"
                  label={t.addressKeyLabel}
                  value={connKey}
                  placeholder={conn.configured ? t.setKeyPh : t.setBasePh}
                  required={conn.configured}
                  disabled={connBusy}
                  onInput={setConnKey}
                />
              </div>
              <p class="conn-note">{conn.configured ? t.addressKeyNote : t.addressNoKeyNote}</p>
              <div class="form-actions">
                <button
                  type="submit"
                  class="btn btn-primary"
                  disabled={connBusy || (conn.configured && !connKey.trim()) || (!connBase.trim() && !conn.placeholder)}
                >
                  {connBusy ? t.saving : t.save}
                </button>
                <button type="button" class="btn btn-ghost" disabled={connBusy} onClick={closeConnEdit}>
                  {t.cancel}
                </button>
              </div>
            </form>
          )}
        </div>
      )}
      {connError && (
        <p class="line-error" role="alert">
          {connError}
        </p>
      )}
    </section>
  );


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
                  <ProviderPicker
                    t={t}
                    providers={providers}
                    onSelect={(item) => choose(item)}
                    onEdit={(item) => choose(item, "#connection-title")}
                    onDelete={removeProvider}
                  />
                )}
              </>
            ) : (
              <>
                {saved && connection}
                <section class="provider-detail">
                  <div class="provider-detail-head">
                    <div class="provider-detail-copy">
                      <h3 class="group-title">{autoLoads ? t.modelCatalogTitle : t.connectProviderTitle(label)}</h3>
                      <p>{t.providerDetailHint(provider.kind, provider.provider, provider.source, { local, manualOnly, needsBase: needsBase && provider.provider !== "azure" && provider.provider !== "custom" })}</p>
                      {(provider.kind === "common" || provider.kind === "compatible") && (provider.default_base || provider.key_variable) && (
                        <p class="provider-facts">
                          {provider.default_base && <span class="provider-fact">{provider.default_base}</span>}
                          {provider.key_variable && <span class="provider-fact">{local ? t.optionalKeyVariable(provider.key_variable) : provider.key_variable}</span>}
                        </p>
                      )}
                    </div>
                    {!manualOnly &&
                      (discovery === null && !autoLoads ? (
                        <button type="button" class="btn btn-primary" disabled={discovering || !canLoad} onClick={() => void load(false)}>
                          {discovering ? t.loading : t.loadModels}
                        </button>
                      ) : (
                        <button
                          type="button"
                          class="icon-btn"
                          aria-label={t.refreshModels}
                          title={t.refreshModels}
                          disabled={discovering || !canLoad}
                          aria-busy={discovering || undefined}
                          onClick={() => void load(discovery !== null)}
                        >
                          {discovering ? <span class="spinner" aria-hidden="true" /> : <Icon name="ArrowCounterClockwise" size={15} />}
                        </button>
                      ))}
                  </div>
                  {showsKey && (
                    <div class={needsBase ? "field-grid" : "field-grid is-single"}>
                      <label class="field">
                        <span class="field-label">
                          {t.setApiKey}
                          {keyRequired && <span class="field-required">{t.requiredField}</span>}
                        </span>
                        <input type="password" value={draftKey} placeholder={t.setKeyPh} autoComplete="off" onInput={(event) => setDraftKey(event.currentTarget.value)} />
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
                    // A saved provider's page reads its own address: what this
                    // connection already routes is what the catalogue marks.
                    apiBase={endpoint}
                    group={group}
                    weight={weight}
                    selected={selected}
                    dropped={dropped}
                    onGroup={setGroup}
                    onWeight={setWeight}
                    onSelected={setSelected}
                    onDropped={setDropped}
                  />
                )}
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
              {source}
            </>
          )}
          {!footer && status}
        </div>
        {footer && (
          <div class="dialog-foot">
            {status}
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

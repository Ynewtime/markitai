import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  addLLMDeployments,
  deleteLLMDeployment,
  deleteLLMProvider,
  discoverLLMModels,
  fetchLLMProviderCredentials,
  fetchLLMSettings,
  fetchProviderConnections,
  testLLMSettings,
  updateLLMDeployment,
  updateLLMProvider,
} from "../api/client";
import type {
  LLMDeployment,
  LLMModelCreate,
  LLMSettingsPayload,
  ModelDiscoveryResult,
  ProviderConnection,
} from "../api/types";
import type { Dict, Locale } from "../i18n";
import {
  ConfirmDeletePopover,
  openDeletePopoverCard,
} from "./ConfirmDeletePopover";
import { useMediaQuery } from "../hooks/useMediaQuery";
import { LangToggle } from "./LangToggle";
import { ThemeToggle } from "./ThemeToggle";
import { AppNotification } from "./WarningNotification";
import { ModelPicker } from "./settings/ModelPicker";
import { ProviderPicker } from "./settings/ProviderPicker";
import {
  CheckIcon,
  EyeIcon,
  EyeSlashIcon,
  WarningIcon,
  XIcon,
} from "./icons";

const EXIT_MS = 120;
type RowTest = "busy" | "ok" | "fail";
type TestNotice = {
  tone: "success" | "error";
  title: string;
  message: string;
};

function SecretInput({
  t,
  id,
  label,
  value,
  placeholder,
  disabled,
  onChange,
}: {
  t: Dict;
  id: string;
  label: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const [revealed, setRevealed] = useState(false);
  const toggleLabel = revealed ? t.concealField(label) : t.revealField(label);
  return (
    <div className="fld">
      <label className="lbl" htmlFor={id}>
        {label}
      </label>
      <div className="secret-input">
        <input
          id={id}
          type={revealed ? "text" : "password"}
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => onChange(event.target.value)}
        />
        <button
          type="button"
          className="secret-reveal"
          aria-label={toggleLabel}
          title={toggleLabel}
          disabled={disabled}
          onClick={() => setRevealed((current) => !current)}
        >
          {revealed ? <EyeSlashIcon /> : <EyeIcon />}
        </button>
      </div>
    </div>
  );
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function settingsError(error: unknown, t: Dict): string {
  if (error instanceof ApiError && error.code === "stale_revision") {
    return t.settingsConflict;
  }
  const text = errorText(error);
  return text.includes("saved provider connection no longer exists")
    ? t.providerConnectionMissing
    : text;
}

function isLocalProvider(provider: string): boolean {
  return provider === "claude-agent" || provider === "copilot" || provider === "chatgpt";
}

export function SettingsModal({
  t,
  locale,
  onLocale,
  onClose,
  onSaved,
  announce,
}: {
  t: Dict;
  locale: Locale;
  onLocale: (l: Locale) => void;
  onClose: () => void;
  onSaved: () => void;
  announce: (message: string) => void;
}) {
  const [settings, setSettings] = useState<LLMSettingsPayload | null>(null);
  // phone-width model rows squeeze the weight pill's grid column below its
  // content width, wrapping text inside the pill — swap to the short label
  const narrowBadges = useMediaQuery("(max-width: 780px)");
  const [providers, setProviders] = useState<ProviderConnection[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  const [closing, setClosing] = useState(false);
  const closingRef = useRef(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const addModelsButtonRef = useRef<HTMLButtonElement | null>(null);
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

  const [providersFailed, setProvidersFailed] = useState(false);
  const refreshProviders = useCallback(async (refresh = false) => {
    try {
      setProviders(await fetchProviderConnections(refresh));
      setProvidersFailed(false);
    } catch {
      // The configured-model list stays usable when provider detection fails;
      // the add-models flow surfaces the failure with a retry instead.
      setProvidersFailed(true);
    }
  }, []);

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
    void refreshProviders();
    return () => {
      stale = true;
    };
  }, [refreshProviders]);

  const [rowTests, setRowTests] = useState<Record<string, RowTest>>({});
  const [testNotice, setTestNotice] = useState<TestNotice | null>(null);
  const testTimersRef = useRef<Record<string, number>>({});
  const testNoticeTimerRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      for (const timer of Object.values(testTimersRef.current)) {
        window.clearTimeout(timer);
      }
      if (testNoticeTimerRef.current !== null) {
        window.clearTimeout(testNoticeTimerRef.current);
      }
    },
    [],
  );
  const settleTestButton = (id: string, state: "ok" | "fail") => {
    window.clearTimeout(testTimersRef.current[id]);
    setRowTests((previous) => ({ ...previous, [id]: state }));
    testTimersRef.current[id] = window.setTimeout(() => {
      setRowTests((previous) => {
        const next = { ...previous };
        delete next[id];
        return next;
      });
      delete testTimersRef.current[id];
    }, state === "ok" ? 1400 : 1800);
  };
  const runTest = async (deployment: LLMDeployment) => {
    const id = deployment.deployment_id;
    window.clearTimeout(testTimersRef.current[id]);
    if (testNoticeTimerRef.current !== null) {
      window.clearTimeout(testNoticeTimerRef.current);
      testNoticeTimerRef.current = null;
    }
    setTestNotice(null);
    setRowTests((previous) => ({ ...previous, [id]: "busy" }));
    try {
      const result = await testLLMSettings({ deployment_id: id });
      settleTestButton(id, result.ok ? "ok" : "fail");
      if (result.ok) {
        setTestNotice({
          tone: "success",
          title: t.modelTestPassed,
          message: t.modelTestReady(deployment.model),
        });
        testNoticeTimerRef.current = window.setTimeout(() => {
          setTestNotice(null);
          testNoticeTimerRef.current = null;
        }, 3000);
      } else {
        setTestNotice({
          tone: "error",
          title: t.modelTestFailed,
          message: result.detail,
        });
      }
    } catch (error) {
      settleTestButton(id, "fail");
      setTestNotice({
        tone: "error",
        title: t.modelTestFailed,
        message: errorText(error),
      });
    }
  };

  const removeDeployment = async (deployment: LLMDeployment) => {
    if (settings === null) return false;
    setListError(null);
    // Pick the focus successor before the row unmounts: the confirm popover
    // closes without returning focus (its trigger vanishes with the deleted
    // deployment), which would otherwise drop focus to <body> mid-dialog.
    const removedIndex = settings.deployments.findIndex(
      (item) => item.deployment_id === deployment.deployment_id,
    );
    const remaining = settings.deployments.filter(
      (item) => item.deployment_id !== deployment.deployment_id,
    );
    const successorId =
      removedIndex < 0
        ? null
        : (remaining[removedIndex] ?? remaining[removedIndex - 1])?.deployment_id ??
          null;
    try {
      const next = await deleteLLMDeployment(
        deployment.deployment_id,
        settings.revision,
      );
      setSettings(next);
      await refreshProviders(true);
      onSaved();
      window.requestAnimationFrame(() => {
        const rows = Array.from(
          cardRef.current?.querySelectorAll<HTMLElement>(".prov") ?? [],
        );
        const successorRow =
          successorId === null
            ? null
            : (rows.find((row) => row.dataset.deploymentId === successorId) ?? null);
        (successorRow?.querySelector<HTMLElement>("button") ??
          addModelsButtonRef.current)?.focus();
      });
      return true;
    } catch (error) {
      setListError(settingsError(error, t));
      return false;
    }
  };

  const [editing, setEditing] = useState<LLMDeployment | null>(null);
  const [editGroup, setEditGroup] = useState("");
  const [editModel, setEditModel] = useState("");
  const [editWeight, setEditWeight] = useState(1);
  // Raw text of the weight field while it has focus, so clearing it does not
  // snap to 0 mid-edit; blur resolves back to the parsed weight.
  const [editWeightDraft, setEditWeightDraft] = useState<string | null>(null);
  const [editBusy, setEditBusy] = useState(false);
  const openEdit = (deployment: LLMDeployment) => {
    setEditing(deployment);
    setEditGroup(deployment.routing_group);
    setEditModel(deployment.model);
    setEditWeight(deployment.weight);
    setEditWeightDraft(null);
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
        weight: editWeight,
        expected_revision: settings.revision,
      });
      setSettings(next);
      setEditing(null);
      onSaved();
      announce(t.saved);
    } catch (error) {
      setListError(settingsError(error, t));
    } finally {
      setEditBusy(false);
    }
  };

  const [editingProvider, setEditingProvider] = useState<ProviderConnection | null>(null);
  const [editProviderKey, setEditProviderKey] = useState("");
  const [editProviderBase, setEditProviderBase] = useState("");
  // provider default shown as the base field's placeholder; an empty field
  // saves null (keep using the default) rather than pinning the default URL
  const [editBasePlaceholder, setEditBasePlaceholder] = useState("");
  const [providerBusy, setProviderBusy] = useState(false);
  const [providerCredentialsBusy, setProviderCredentialsBusy] = useState(false);
  const providerCredentialsRequestRef = useRef(0);
  const openProviderEdit = async (connection: ProviderConnection) => {
    if (connection.provider_id === undefined) return;
    const requestId = ++providerCredentialsRequestRef.current;
    setEditingProvider(connection);
    setEditProviderKey("");
    setEditProviderBase("");
    setEditBasePlaceholder("");
    setProviderCredentialsBusy(true);
    setEditing(null);
    setListError(null);
    try {
      const credentials = await fetchLLMProviderCredentials(connection.provider_id);
      if (requestId !== providerCredentialsRequestRef.current) return;
      setEditProviderKey(credentials.api_key ?? "");
      setEditProviderBase(credentials.api_base ?? "");
      setEditBasePlaceholder(credentials.api_base_placeholder ?? "");
    } catch (error) {
      if (requestId === providerCredentialsRequestRef.current) {
        setEditingProvider(null);
        setListError(settingsError(error, t));
      }
    } finally {
      if (requestId === providerCredentialsRequestRef.current) {
        setProviderCredentialsBusy(false);
      }
    }
  };
  const closeProviderEdit = () => {
    providerCredentialsRequestRef.current += 1;
    setEditingProvider(null);
    setProviderCredentialsBusy(false);
  };
  const saveProviderEdit = async () => {
    if (
      settings === null ||
      editingProvider?.provider_id === undefined ||
      providerBusy ||
      providerCredentialsBusy
    ) {
      return;
    }
    setProviderBusy(true);
    setListError(null);
    try {
      const next = await updateLLMProvider(editingProvider.provider_id, {
        api_key: editProviderKey.trim() === "" ? null : editProviderKey.trim(),
        api_base: editProviderBase.trim() === "" ? null : editProviderBase.trim(),
        expected_revision: settings.revision,
      });
      setSettings(next);
      setEditingProvider(null);
      providerCredentialsRequestRef.current += 1;
      await refreshProviders(true);
      onSaved();
      announce(t.providerSaved);
    } catch (error) {
      setListError(settingsError(error, t));
    } finally {
      setProviderBusy(false);
    }
  };
  const removeProvider = async (connection: ProviderConnection) => {
    if (settings === null || connection.provider_id === undefined) return false;
    setListError(null);
    try {
      const next = await deleteLLMProvider(
        connection.provider_id,
        settings.revision,
      );
      setSettings(next);
      if (editingProvider?.provider_id === connection.provider_id) {
        setEditingProvider(null);
        providerCredentialsRequestRef.current += 1;
      }
      await refreshProviders(true);
      onSaved();
      return true;
    } catch (error) {
      setListError(settingsError(error, t));
      return false;
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
      provider.provider_id !== undefined ||
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
    setEditingProvider(null);
    providerCredentialsRequestRef.current += 1;
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

  // After an Escape-driven breadcrumb unwind the focused subtree unmounts;
  // move focus to the surviving level heading so it does not fall to <body>.
  const focusBreadcrumbHeading = useCallback(() => {
    window.requestAnimationFrame(() => {
      cardRef.current?.querySelector<HTMLElement>("#mdl-title")?.focus();
    });
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // An open delete confirmation is the topmost layer: it owns Escape and
      // bounds the Tab cycle. Its listener shares this document node, so
      // stopPropagation cannot arbitrate - the modal has to stand down itself.
      const popover = openDeletePopoverCard();
      if (event.key === "Escape") {
        if (popover !== null) return;
        event.preventDefault();
        event.stopPropagation();
        // Escape unwinds the add-models flow one breadcrumb level at a time
        // before it closes the whole modal.
        if (adding && provider !== null) {
          backToProviders();
          focusBreadcrumbHeading();
        } else if (adding) {
          resetAddFlow();
          focusBreadcrumbHeading();
        } else requestClose();
        return;
      }
      if (event.key !== "Tab") return;
      const root = popover ?? cardRef.current;
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
    // backToProviders/resetAddFlow only close over stable setters and refs,
    // so re-registering on the state they read is enough to stay current.
  }, [requestClose, adding, provider, focusBreadcrumbHeading]);

  const chooseProvider = (next: ProviderConnection) => {
    setProvider(next);
    closeProviderEdit();
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
        ...(provider.provider_id === undefined ||
        provider.provider_id.startsWith("legacy:")
          ? {}
          : { provider_id: provider.provider_id }),
        ...(provider.deployment_id === undefined
          ? {}
          : { deployment_id: provider.deployment_id }),
        ...(draftKey.trim() === "" ? {} : { api_key: draftKey.trim() }),
        ...(draftBase.trim() === "" ? {} : { api_base: draftBase.trim() }),
        refresh,
      });
      setDiscovery(result);
      const available = new Set(result.models.map((candidate) => candidate.model));
      setSelectedModels(
        (previous) => new Set([...previous].filter((model) => available.has(model))),
      );
    } catch (error) {
      setListError(settingsError(error, t));
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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- loadModels is recreated every render; autoDiscoveryRef already keys the fetch by provider.
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
      provider: provider.provider,
      ...(provider.provider_id === undefined ||
      provider.provider_id.startsWith("legacy:")
        ? {}
        : { credential_provider_id: provider.provider_id }),
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
      await refreshProviders(true);
      onSaved();
      announce(t.modelsAdded(deployments.length));
    } catch (error) {
      setListError(settingsError(error, t));
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
      setListError(settingsError(error, t));
    }
  };

  const renderDeployment = (deployment: LLMDeployment, detected = false) => {
    const test = rowTests[deployment.deployment_id];
    return (
      <div
        className="prov"
        data-deployment-id={deployment.deployment_id}
        key={`${detected ? "detected" : "configured"}:${deployment.deployment_id}`}
      >
        <div className="prov-line">
          <span className="prov-name mono">{deployment.routing_group}</span>
          <span className="prov-model mono" title={deployment.model}>{deployment.model}</span>
          <span className="minibadge" title={t.weightHint}>
            {detected
              ? t.sessionBadge
              : narrowBadges
                ? t.modelWeightShort(deployment.weight)
                : t.modelWeight(deployment.weight)}
          </span>
          <span className="prov-acts">
            <button
              type="button"
              className={`rowact model-test${test === undefined ? "" : ` ${test}`}`}
              disabled={test === "busy"}
              aria-label={
                test === "busy"
                  ? t.testing
                  : test === "ok"
                    ? t.modelTestPassed
                    : test === "fail"
                      ? t.modelTestFailed
                      : t.test
              }
              onClick={() => void runTest(deployment)}
            >
              <span className="model-test-swap" key={test ?? "idle"}>
                {test === "busy" ? (
                  <span className="spin" aria-hidden="true" />
                ) : test === "ok" ? (
                  <CheckIcon size={16} />
                ) : test === "fail" ? (
                  <WarningIcon size={15} />
                ) : (
                  t.test
                )}
              </span>
            </button>
            {detected ? (
              <button type="button" className="rowact" onClick={() => void saveDetected(deployment)}>{t.saveToConfig}</button>
            ) : (
              <>
                <button type="button" className="rowact" onClick={() => openEdit(deployment)}>{t.edit}</button>
                <ConfirmDeletePopover
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

  const providerEditor =
    editingProvider === null ? null : (
      <form
        key={editingProvider.provider_id}
        className="mdl-form provider-editor"
        onSubmit={(event) => {
          event.preventDefault();
          void saveProviderEdit();
        }}
      >
        <h3 className="picker-group mono">
          {t.editProvider(editingProvider.label)}
        </h3>
        <div className="connection-fields">
          <SecretInput
            t={t}
            id="provider-api-key"
            label={t.setApiKey}
            value={editProviderKey}
            placeholder={providerCredentialsBusy ? t.loading : t.setKeyPh}
            disabled={providerCredentialsBusy}
            onChange={setEditProviderKey}
          />
          <div className="fld">
            <label className="lbl" htmlFor="provider-api-base">
              {t.setApiBase}
            </label>
            <input
              id="provider-api-base"
              type="url"
              inputMode="url"
              value={editProviderBase}
              placeholder={
                providerCredentialsBusy
                  ? t.loading
                  : editBasePlaceholder || t.setBasePh
              }
              disabled={providerCredentialsBusy}
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => setEditProviderBase(event.target.value)}
            />
          </div>
        </div>
        <div className="set-actions">
          <button
            type="submit"
            className="btn primary"
            disabled={providerBusy || providerCredentialsBusy}
          >
            {providerBusy ? t.saving : t.save}
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={closeProviderEdit}
          >
            {t.cancel}
          </button>
        </div>
      </form>
    );

  return (
    <div
      className={closing ? "mdl-veil settings-veil out" : "mdl-veil settings-veil"}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) requestClose();
      }}
    >
      {testNotice !== null && (
        <AppNotification
          tone={testNotice.tone}
          title={testNotice.title}
          message={testNotice.message}
          closeLabel={t.close}
          onClose={() => {
            if (testNoticeTimerRef.current !== null) {
              window.clearTimeout(testNoticeTimerRef.current);
              testNoticeTimerRef.current = null;
            }
            setTestNotice(null);
          }}
        />
      )}
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
                  <h2 id="mdl-title" tabIndex={-1}>{t.addModels}</h2>
                ) : (
                  <>
                    <button type="button" onClick={backToProviders}>
                      {t.addModels}
                    </button>
                    <span aria-hidden="true">/</span>
                    <h2 id="mdl-title" tabIndex={-1}>{provider.label}</h2>
                  </>
                )}
              </>
            ) : (
              <h2 id="mdl-title" tabIndex={-1}>{t.settingsTitle}</h2>
            )}
          </nav>
          <button type="button" className="gearbtn" aria-label={t.close} title={t.close} onClick={requestClose}>
            <XIcon size={16} />
          </button>
        </div>

        <div className="mdl-body">
          {/* Phone-only appearance controls: at ≤780px the header hides
              .hdr-toggles and this section takes over (CSS gates both on the
              same breakpoint, so exactly one control location exists at any
              width). Desktop keeps it display: none, which also removes it
              from the focus trap's offsetParent-based focusable scan. Only on
              the root breadcrumb level — the add-models flow replaces the
              body wholesale. */}
          {!adding && (
            <section className="appearance-section" aria-labelledby="appearance-title">
              <h3 id="appearance-title" className="picker-group mono">
                {t.appearanceTitle}
              </h3>
              <div className="appearance-controls">
                <div className="appearance-opt">
                  <span className="lbl">{t.langAria}</span>
                  <LangToggle label={t.langAria} locale={locale} onLocale={onLocale} />
                </div>
                <div className="appearance-opt">
                  <span className="lbl">{t.themeAria}</span>
                  <ThemeToggle t={t} label={t.themeAria} />
                </div>
              </div>
            </section>
          )}
          {settings === null ? (
            <p className={loadError === null ? "mdl-dim mono" : "errline"}>{loadError ?? t.loading}</p>
          ) : adding ? (
            provider === null ? (
              <>
                {providers.length === 0 ? (
                  providersFailed ? (
                    <>
                      <p className="errline" role="alert">{t.providersLoadFailed}</p>
                      <button
                        type="button"
                        className="btn ghost"
                        onClick={() => void refreshProviders()}
                      >
                        {t.retryLoad}
                      </button>
                    </>
                  ) : (
                    <p className="mdl-dim mono">{t.loading}</p>
                  )
                ) : (
                  <ProviderPicker
                    t={t}
                    providers={providers}
                    selectedId={null}
                    onSelect={chooseProvider}
                    onEditProvider={openProviderEdit}
                    onDeleteProvider={removeProvider}
                  />
                )}
                {providerEditor}
                {editingProvider === null && (
                  <button type="button" className="btn ghost" onClick={resetAddFlow}>
                    {t.cancel}
                  </button>
                )}
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
                <span className="mono">{t.modelsConfigured(settings.deployments.length)}</span>
                <button ref={addModelsButtonRef} type="button" className="btn primary" onClick={() => setAdding(true)}>{t.addModels}</button>
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
                    <label className="fld"><span className="lbl">{t.weight}</span><input type="number" min={0} value={editWeightDraft ?? editWeight} onChange={(event) => { setEditWeightDraft(event.target.value); setEditWeight(Math.max(0, Number(event.target.value) || 0)); }} onBlur={() => setEditWeightDraft(null)} /></label>
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

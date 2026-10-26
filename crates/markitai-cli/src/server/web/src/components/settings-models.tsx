// Model settings, as a page with a breadcrumb: the configured models, then
// Add models → provider → model catalogue. A model row's Edit opens the
// provider's page the same way. Escape steps back one level.
// Every write carries the revision this draft started from; a conflict keeps
// the draft and reloads the lists, and the person decides to use the new revision.
import type { TargetedInputEvent } from "preact";
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { isRevisionConflict, modelsApi, openConfig, type ModelsScope } from "../api/client.ts";
import type { Deployment, DiscoveryResult, ModelShare, NewDeployment, ProviderCard, SettingsView, SharedModel, UserModelsView } from "../api/types.ts";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { serviceNote } from "../i18n/errors.ts";
import { addingIsBlocked, listFailure, providerWords, regionBlocked, runtimeFailure, testFailure, type TestFailure } from "../lib/model-test.ts";
import { endpointFor, modelName, pickerRows, refusedHosts, removals } from "../lib/models.ts";
import { isKimiCard, kimiOrder, kimiServiceAt, kimiServices, kimiTypedBase, type KimiService } from "../lib/kimi.ts";
import { foundFromKey } from "../lib/regions.ts";
import { cardEndpoints, type EndpointChoice, kimiEndpoints, matchingChoice } from "../lib/endpoints.ts";
import { AddressField } from "./address-field.tsx";
import { isSubscription, listsModels, ownEndpointLogin, usesCloudCredentials } from "../lib/provider-catalog.ts";
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
import { InfoTip } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";
import { ModelPicker } from "./model-picker.tsx";
import { RuntimeNotice } from "./runtime-notice.tsx";
import { loadBrandMarks } from "./brand-icon.tsx";
import { manageable, providerLabel, ProviderPicker } from "./provider-picker.tsx";
import { StateBlock } from "./state-block.tsx";
import { WeightPill } from "./weight-pill.tsx";
import { dismissToast, toast } from "../lib/toast.ts";

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
  const field = {
    id,
    required,
    value,
    placeholder,
    disabled,
    autoComplete: "off",
    spellcheck: false,
    onInput: (event: TargetedInputEvent<HTMLInputElement>) => onInput(event.currentTarget.value),
  };
  return (
    <div class="field">
      <label class="field-label" for={id}>
        {label}
        {required && <span class="field-required">{t.requiredField}</span>}
      </label>
      <div class="secret">
        {/* One input either way (Preact patches its type in place, so focus and
            caret stay); a branch per type, as Preact types an input's roles by it. */}
        {shown ? <input type="text" {...field} /> : <input type="password" {...field} />}
        <button type="button" class="secret-toggle" aria-label={toggle} title={toggle} disabled={disabled} onClick={() => setShown((value) => !value)}>
          <Icon name={shown ? "EyeSlash" : "Eye"} size={16} />
        </button>
      </div>
    </div>
  );
}

/** One model test notice at a time: a new test replaces the last one's. */
const TEST_TOAST = "model-test";

/** v1.15: whether an admin's saved model is private or shared with users' conversions, as a two-way toggle. */
function ShareToggle({ t, model, value, busy, onChange }: { t: Dict; model: string; value: ModelShare; busy: boolean; onChange: (next: ModelShare) => void }) {
  const options: { value: ModelShare; label: string }[] = [
    { value: "private", label: t.sharePrivate },
    { value: "shared", label: t.shareShared },
  ];
  return (
    <span class="seg share-seg" role="radiogroup" aria-label={t.shareAria(model)} aria-busy={busy || undefined}>
      {options.map((option) => (
        <button
          type="button"
          role="radio"
          key={option.value}
          aria-checked={value === option.value}
          class={value === option.value ? "is-on" : undefined}
          tabIndex={value === option.value ? 0 : -1}
          disabled={busy}
          title={option.value === "shared" ? t.shareSharedHint : t.sharePrivateHint}
          onClick={() => value !== option.value && onChange(option.value)}
          onKeyDown={(event) => {
            if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
            event.preventDefault();
            const next = value === "private" ? "shared" : "private";
            onChange(next);
            const group = event.currentTarget.parentElement;
            requestAnimationFrame(() => group?.querySelector<HTMLElement>('[aria-checked="true"]')?.focus());
          }}
        >
          {option.label}
        </button>
      ))}
    </span>
  );
}

/** The Test button of a model row: idle, testing, passed or failed for a moment. */
function TestButton({ t, model, state, onTest }: { t: Dict; model: string; state: "busy" | "ok" | "fail" | undefined; onTest: () => void }) {
  return (
    <button
      type="button"
      class={`row-icon test-btn${state ? ` is-${state}` : ""}`}
      disabled={state === "busy"}
      aria-label={state === "busy" ? t.testing : state === "ok" ? t.modelTestPassed : state === "fail" ? t.modelTestFailed : t.testModel(model)}
      title={state === "busy" ? t.testing : state === "ok" ? t.modelTestPassed : state === "fail" ? t.modelTestFailed : t.test}
      onClick={onTest}
    >
      <span class="test-swap" key={state ?? "idle"}>
        {state === "busy" ? (
          <span class="spinner" aria-hidden="true" />
        ) : state === "ok" ? (
          <Icon name="CheckBold" size={16} />
        ) : state === "fail" ? (
          <Icon name="WarningFill" size={15} />
        ) : (
          <Icon name="PlugsConnected" size={15} />
        )}
      </span>
    </button>
  );
}

/** A model's last failed test, under its row until it is tested again. */
function TestFailureLine({ t, failure }: { t: Dict; failure: TestFailure | undefined }) {
  if (!failure) return null;
  return (
    <p class="model-test-fail">
      <Icon name="WarningCircle" size={14} />
      <span>
        {t.testFailTitle(failure.reason)}
        {failure.variable && <span class="model-test-var"> · {failure.variable}</span>}
      </span>
    </p>
  );
}

/** A weight being edited in place that cannot be saved, under its row (round 18). */
function WeightErrorLine({ t, shown }: { t: Dict; shown: boolean }) {
  if (!shown) return null;
  return (
    <p class="config-check is-error weight-error" role="alert">
      <Icon name="WarningCircle" size={14} />
      <span>{t.weightInvalid}</span>
    </p>
  );
}

/** The models the admin shares, as a user sees them: what each is and, v1.16, this user's
 * own weight for it (0 = off for this user only) and a Test that runs on the server. */
function SharedModels({
  t,
  models,
  narrow,
  labelOf,
  tests,
  failures,
  onTest,
  onWeight,
  weightErrors,
  onWeightInvalid,
}: {
  t: Dict;
  weightErrors: Record<string, boolean>;
  onWeightInvalid: (key: string, invalid: boolean) => void;
  models: SharedModel[];
  narrow: boolean;
  labelOf: (provider: string) => string;
  tests: Record<string, "busy" | "ok" | "fail">;
  failures: Record<string, TestFailure>;
  onTest: (model: SharedModel) => void;
  onWeight: (model: SharedModel, weight: number | null) => Promise<boolean>;
}) {
  return (
    <section class="shared-models" aria-labelledby="shared-models-title">
      <div class="group-head-row">
        <h2 id="shared-models-title" class="group-title">
          {t.sharedByAdmin}
        </h2>
        <InfoTip label={t.sharedInfo} text={models.some((model) => model.default_weight !== undefined) ? t.sharedByAdminWeights : t.sharedByAdminText} />
      </div>
      <ul class="model-rows" aria-labelledby="shared-models-title">
        {models.map((model) => {
          // An older backend sends no `default_weight`: its shared models can only be read.
          const tunable = model.default_weight !== undefined;
          const own = tunable && model.weight !== model.default_weight;
          return (
            <li class={`model-row is-shared${model.weight === 0 ? " is-off" : ""}`} key={model.id} data-shared={model.id}>
              <div class="model-line">
                <span class="model-group">{modelName(model.model)}</span>
                <span class="model-id" title={model.model}>
                  {model.model}
                </span>
                {tunable ? (
                  <WeightPill
                    t={t}
                    model={modelName(model.model)}
                    value={model.weight}
                    narrow={narrow}
                    title={t.sharedWeightHint}
                    onSave={(weight) => onWeight(model, weight)}
                    onInvalid={(invalid) => onWeightInvalid(`shared:${model.id}`, invalid)}
                  />
                ) : (
                  <span class="mini-pill" title={t.weightHint}>
                    {narrow ? t.modelWeightShort(model.weight) : t.modelWeight(model.weight)}
                  </span>
                )}
                {own && (
                  <button
                    type="button"
                    class="text-btn weight-reset"
                    title={t.sharedWeightResetHint}
                    aria-label={t.sharedWeightResetAria(t.sharedWeightReset(model.default_weight ?? 1), modelName(model.model))}
                    onClick={() => void onWeight(model, null)}
                  >
                    {t.sharedWeightReset(model.default_weight ?? 1)}
                  </button>
                )}
                <span class="shared-tag">{labelOf(model.provider)}</span>
                {tunable && (
                  <span class="text-actions">
                    <TestButton t={t} model={model.model} state={tests[`shared:${model.id}`]} onTest={() => onTest(model)} />
                  </span>
                )}
              </div>
              <WeightErrorLine t={t} shown={weightErrors[`shared:${model.id}`] === true} />
              {model.weight === 0 && <p class="shared-off">{t.sharedOffForYou}</p>}
              <TestFailureLine t={t} failure={failures[`shared:${model.id}`]} />
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export function ModelsSettings({
  t,
  locale,
  scope = "admin",
  canShare = false,
  onSaved,
  announce,
  describe,
}: {
  t: Dict;
  locale: Locale;
  /** Whose models: the admin's (the config file), or v1.15 a user token's own plus the shared ones. */
  scope?: ModelsScope;
  /** v1.15: the admin's saved models carry a Private/Shared toggle. */
  canShare?: boolean;
  onSaved: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => Described;
}) {
  const api = useMemo(() => modelsApi(scope), [scope]);
  const { fetchSettings, fetchProviders, fetchCredentials, discoverModels, probeModel, addDeployments, deleteDeployment, updateDeployment, deleteProvider, updateProvider, updateShared } = api;
  const user = scope === "user";
  const card = useRef<HTMLElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const narrow = useMedia(NARROW);
  const [settings, setSettings] = useState<SettingsView | UserModelsView | null>(null);
  const shared: SharedModel[] = settings !== null && "shared" in settings ? (settings.shared ?? []) : [];
  const [loadError, setLoadError] = useState<unknown>(null);
  const [providers, setProviders] = useState<ProviderCard[]>([]);
  const [providersFailed, setProvidersFailed] = useState(false);
  // The revision every write of this draft is checked against.
  const [revision, setRevision] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [listError, setListError] = useState<Described | null>(null);

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
    // The provider marks load with this page, before Add models needs them.
    void loadBrandMarks();
    return () => {
      stale = true;
    };
  }, [refreshProviders, fetchSettings]);

  /** After a successful write: the new view and revision; the conflict is over.
   * Round 18: the provider cards come again too, since a write can give
   * hand-written models their ids (and a card names the connection by one). */
  const accept = (view: SettingsView | UserModelsView, cards = true) => {
    setSettings(view);
    setRevision(view.revision);
    setConflict(false);
    setListError(null);
    onSaved();
    if (cards) void refreshProviders(true);
  };

  const credentialText = (error: CredentialInputError) =>
    ({ environment_reference: t.literalCredentialsOnly, endpoint_key_required: t.endpointKeyRequired, key_required: t.keyRequired, masked_address: t.maskedAddress })[error.reason];

  /** A failed write. A revision conflict reloads the lists but keeps the draft and its revision. */
  const fail = async (error: unknown) => {
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
  /** Rows whose weight field holds a value that cannot be saved (round 18). */
  const [weightErrors, setWeightErrors] = useState<Record<string, boolean>>({});
  const weightInvalid = useCallback(
    (key: string, invalid: boolean) => setWeightErrors((previous) => (Boolean(previous[key]) === invalid ? previous : { ...previous, [key]: invalid })),
    [],
  );
  /** The last failed test of each model, shown under it until it is tested again. */
  const [failures, setFailures] = useState<Record<string, TestFailure>>({});
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
  /** Test one model: a row's own (`deployment_id`) or, v1.16, a shared one (`shared_id`, keyed `shared:<id>`),
   * which runs on the server with the admin's key and answers only whether it passed and why not. */
  const runTest = async (key: string, model: string, body: Record<string, unknown>, shared = false) => {
    dismissToast(TEST_TOAST);
    setFailures(({ [key]: _, ...rest }) => rest);
    setTests((previous) => ({ ...previous, [key]: "busy" }));
    try {
      const result = await probeModel(body);
      settle(key, result.ok ? "ok" : "fail");
      if (result.ok) {
        toast({ id: TEST_TOAST, tone: "success", title: t.modelTestPassed, text: t.modelTestReady(model) });
      } else {
        const failure = testFailure(result);
        if (shared) failure.variable = null;
        setFailures((previous) => ({ ...previous, [key]: failure }));
        // Only the provider's or the network's own words open under Details, never markitai's sentence.
        const words = shared ? null : providerWords(failure.detail, failure.reason, t.testFailTitle(failure.reason));
        toast({
          id: TEST_TOAST,
          tone: "error",
          title: `${t.testFailTitle(failure.reason)} · ${t.testFailFor(model)}`,
          text: shared
            ? t.sharedTestFailHint(failure.reason)
            : regionBlocked(failure.detail)
              ? t.modelRegionBlocked(model)
              : t.testFailHint(failure.reason, failure.variable, model, words !== null),
          ...(words ? { detail: words.text } : {}),
        });
      }
    } catch (error) {
      settle(key, "fail");
      const text = describe(error);
      toast({ id: TEST_TOAST, tone: "error", title: t.modelTestFailed, text: text.text, ...(text.detail ? { detail: text.detail } : {}) });
    }
  };

  /** v1.16: a weight changed in place, on a row's own model or (user) a shared one; null = the admin's again. */
  const saveWeight = async (id: string, model: string, weight: number | null, shared = false): Promise<boolean> => {
    if (revision === null) return false;
    try {
      const view = shared ? await updateShared(id, revision, weight) : await updateDeployment(id, { expected_revision: revision, weight });
      accept(view);
      const now = shared ? ((view as UserModelsView).shared ?? []).find((item) => item.id === id)?.weight : view.deployments.find((item) => item.deployment_id === id)?.weight;
      announce(t.weightSaved(modelName(model), now ?? weight ?? 0));
      return true;
    } catch (error) {
      await fail(error);
      return false;
    }
  };

  const removeDeployment = async (deployment: Deployment) => {
    if (!settings || revision === null) return false;
    const index = settings.deployments.findIndex((item) => item.deployment_id === deployment.deployment_id);
    const rest = settings.deployments.filter((item) => item.deployment_id !== deployment.deployment_id);
    const successor = index < 0 ? null : ((rest[index] ?? rest[index - 1])?.deployment_id ?? null);
    try {
      accept(await deleteDeployment(deployment.deployment_id, revision));
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

  const [sharing, setSharing] = useState<string | null>(null);
  const changeShare = async (deployment: Deployment, share: ModelShare) => {
    if (revision === null || sharing !== null) return;
    setSharing(deployment.deployment_id);
    try {
      accept(await updateDeployment(deployment.deployment_id, { expected_revision: revision, share }));
      announce(share === "shared" ? t.shareNowShared(deployment.model) : t.shareNowPrivate(deployment.model));
    } catch (error) {
      await fail(error);
    } finally {
      setSharing(null);
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
  /** The provider's other region that listed the key's models (v1.21): a new
   * connection keeps that address, a saved one is told to move. */
  const [region, setRegion] = useState<{ base: string; kept: boolean } | null>(null);
  /** On the Kimi card (lib/kimi.ts): the service that took the key and the address that listed its
   * models; `refused` once every service refused it. */
  const [kimiFound, setKimiFound] = useState<{ service: KimiService; base: string; typed: boolean } | null>(null);
  const [kimiRefused, setKimiRefused] = useState(false);
  const [selected, setSelected] = useState(new Set<string>());
  /** Models this group and endpoint already route that the reader unticked. */
  const [dropped, setDropped] = useState(new Set<string>());
  const [group, setGroup] = useState("default");
  const [weight, setWeight] = useState(1);
  /** The form's weight field holds a weight Save may write (round 18: a whole number from 0 to 100). */
  const [weightValid, setWeightValid] = useState(true);
  const [addBusy, setAddBusy] = useState(false);
  const autoLoaded = useRef<string | null>(null);
  const existing =
    provider !== null && (provider.provider_id !== undefined || provider.deployment_id !== undefined || provider.kind === "environment");
  // The Kimi card asks for a key once; the models go under the service that takes it.
  const services = useMemo(() => kimiServices(providers), [providers]);
  const kimi = isKimiCard(provider) && services !== null;
  /** The provider this page's models are saved under: on the Kimi card, the service found for the key. */
  const target = kimi && kimiFound ? services![kimiFound.service] : provider;
  const ollama = provider?.provider === "ollama";
  // A local server with a known address (Ollama, LM Studio) needs no key.
  const local = provider?.key_optional === true;
  // A subscription runtime signs in by itself; Bedrock and Vertex AI read the
  // server's cloud credentials (a Bedrock API key is optional).
  const subscription = provider !== null && isSubscription(provider);
  // A user's own models never read the server's cloud credentials: Bedrock takes their own key.
  const cloud = provider !== null && !user && usesCloudCredentials(provider);
  // Vertex AI (and watsonx, GigaChat) sign in with the server's own login at
  // their own endpoint: the server refuses them an address, so none is offered.
  const login = provider !== null ? ownEndpointLogin(provider) : null;
  // The provider lists no models (a saved connection too), or the server says so
  // when asked: IDs are entered by hand.
  const manualOnly = provider !== null && (!listsModels(provider) || discovery?.source === "manual");
  const autoLoads = !manualOnly && (existing || subscription || (local && Boolean(provider?.default_base)));
  // Azure, a custom endpoint and servers without a default address (vLLM) need one.
  const needsBase =
    provider !== null &&
    !subscription &&
    !cloud &&
    login === null &&
    (provider.provider === "azure" || provider.provider === "custom" || (provider.default_base === null && !existing));
  const showsKey = provider !== null && !existing && !ollama && !subscription && login === null && (!cloud || Boolean(provider.key_variable));
  // Every other new connection may move the address; its default shows as the placeholder.
  const baseOptional = provider !== null && !needsBase && !existing && !subscription && login === null;
  const keyRequired = provider !== null && provider.provider !== "custom" && !local;
  const canLoad =
    provider !== null &&
    !manualOnly &&
    (autoLoads ||
      (needsBase ? draftBase.trim() !== "" && (!keyRequired || draftKey.trim() !== "") : !keyRequired || draftKey.trim() !== ""));

  // The addresses the API address field lists: the Kimi card's two services, else the card's own.
  const endpointChoices = useMemo(
    () => (provider === null ? [] : kimi ? kimiEndpoints(services!.moonshot, services!.kimi_code) : cardEndpoints(provider)),
    [provider, kimi, services],
  );
  const endpointName = (choice: EndpointChoice) => t.endpointLabel(choice.region, choice.plan, choice.service);

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
  // A login provider or a subscription signs in without a key: its saved
  // connection offers none (a key a config entry still holds can be removed).
  const takesKey = login === null && !subscription;

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
      if (request === connRequest.current) setConn(providerDraft(credentials, connection.credential));
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
      // The list comes below, to pick this connection's card again.
      accept(await updateProvider(current.provider_id, body), false);
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
      if (listsModels(next)) void load(discovery !== null, next);
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
    setRegion(null);
    setKimiFound(null);
    setKimiRefused(false);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const backToProviders = () => {
    setProvider(null);
    resetConnection();
    setDiscovery(null);
    setRegion(null);
    setKimiFound(null);
    setKimiRefused(false);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };
  const focusHeading = () => requestAnimationFrame(() => card.current?.querySelector<HTMLElement>("#settings-title")?.focus());

  // Escape steps back one level: an open form, then the provider, then Add models.
  const stepBack = () => {
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
    } else return false;
    return true;
  };

  const choose = (next: ProviderCard, focus = "#settings-title") => {
    setProvider(next);
    resetConnection();
    if (manageable(next)) void loadConnection(next);
    requestAnimationFrame(() => card.current?.querySelector<HTMLElement>(focus)?.focus());
    setDraftKey("");
    setDraftBase("");
    setDiscovery(null);
    setRegion(null);
    setKimiFound(null);
    setKimiRefused(false);
    setSelected(new Set());
    setDropped(new Set());
    setListError(null);
    autoLoaded.current = null;
  };

  /** Load models on the Kimi card: each service in turn while it refuses the key (each tries its own
   * regions), or only the address typed by hand. The address that listed the models fills the field. */
  const loadKimi = async (refresh: boolean) => {
    if (!services) return;
    // a typed address keeps getting the key on Refresh; only a found one lets the page search again
    const typed = kimiTypedBase(draftBase, kimiFound);
    const order = typed ? [kimiServiceAt(typed)] : kimiOrder(draftKey);
    let result: DiscoveryResult | null = null;
    let service = order[0]!;
    for (const next of order) {
      service = next;
      result = await discoverModels(discoveryRequest(services[next], draftKey, typed, refresh));
      if (listFailure(result)?.reason !== "key_rejected") break;
    }
    if (!result) return;
    setDiscovery(result);
    const available = new Set(result.models.map((candidate) => candidate.model));
    setSelected((previous) => new Set([...previous].filter((model) => available.has(model))));
    const refused = listFailure(result)?.reason === "key_rejected";
    setKimiRefused(refused && !typed);
    if (result.status === "unavailable") {
      setKimiFound(null);
      if (!typed) setDraftBase("");
      return;
    }
    const base = result.api_base ?? (typed || services[service].default_base || "");
    setKimiFound({ service, base, typed: typed !== "" });
    setDraftBase(base);
  };

  const load = async (refresh = false, from = provider) => {
    if (!from || discovering) return;
    setDiscovering(true);
    setListError(null);
    if (isKimiCard(from) && services) {
      try {
        await loadKimi(refresh);
      } catch (error) {
        await fail(error);
      } finally {
        setDiscovering(false);
      }
      return;
    }
    try {
      const result = await discoverModels(discoveryRequest(from, draftKey, login === null ? draftBase : "", refresh));
      setDiscovery(result);
      const elsewhere = result.status !== "unavailable" ? result.api_base : undefined;
      if (elsewhere) {
        const kept = from.provider_id === undefined && from.deployment_id === undefined;
        if (kept) setDraftBase(elsewhere);
        setRegion({ base: elsewhere, kept });
      } else setRegion((current) => (current?.kept && draftBase.trim() === current.base ? current : null));
      const available = new Set(result.models.map((candidate) => candidate.model));
      setSelected((previous) => new Set([...previous].filter((model) => available.has(model))));
      // Unticked saved models stay unticked: a saved row the list lacks is still on the page.
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

  /** What a failed model list adds from the provider or the network (never markitai's own sentence). */
  const listed = discovery ? listFailure(discovery) : null;
  const listWords = discovery && listed ? providerWords(discovery.detail, listed.reason, t.testFailTitle(listed.reason)) : null;
  /** The address this page's models are at; the rows of its model list, as the picker shows them. */
  const endpoint = endpointFor(draftBase, target?.api_base);
  const pageRows = () =>
    target === null || settings === null
      ? []
      : pickerRows(target.provider, discovery?.models ?? [], settings.deployments, group, endpoint, target.default_base);

  const addSelected = async () => {
    if (!provider || !target || (!selected.size && !dropped.size) || revision === null || addBusy || !weightValid) return;
    let credentials: Partial<NewDeployment>;
    try {
      const found = region?.kept && draftBase.trim() === region.base ? region.base : undefined;
      // A Kimi service found at its default address saves none, as a key pasted on its own card would.
      const atDefault = kimi && draftBase.trim().replace(/\/+$/, "") === (target.default_base ?? "").replace(/\/+$/, "");
      credentials = deploymentCredentials(target, draftKey, login === null && !atDefault ? draftBase : "", found);
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
    // The saved models of every unticked row (also one saved under an alias, or
    // one the list lacks), by the ids of the revision this draft started from.
    const rows = pageRows();
    const remove = removals(rows, dropped);
    const removed = remove.length;
    if (!deployments.length && !removed) {
      // The unticked rows no longer stand for a saved model: nothing to save.
      setAddBusy(false);
      backToProviders();
      return;
    }
    try {
      if (settings?.batch_remove === true || !remove.length) {
        // One change, all or nothing: additions and removals under one revision.
        accept(await addDeployments(revision, deployments, remove));
      } else {
        // An older backend takes additions only: add first (the provider card may
        // stand for a model about to go), then remove by the ids the new view has.
        let view = deployments.length ? await addDeployments(revision, deployments) : null;
        if (view) accept(view);
        const gone = rows.filter((row) => dropped.has(row.model)).flatMap((row) => row.saved);
        for (const old of gone) {
          const current = view ?? settings!;
          const same = current.deployments.find(
            (item) => item.model === old.model && item.routing_group === old.routing_group && (item.api_base ?? "") === (old.api_base ?? ""),
          );
          if (!same) continue;
          view = await deleteDeployment(same.deployment_id, view?.revision ?? revision);
          accept(view);
        }
      }
      setAdding(false);
      setProvider(null);
      setDiscovery(null);
      setSelected(new Set());
      setDropped(new Set());
      announce([deployments.length ? t.modelsAdded(deployments.length) : "", removed ? t.modelsRemoved(removed) : ""].filter(Boolean).join(" · "));
    } catch (error) {
      await fail(error);
    } finally {
      setAddBusy(false);
    }
  };

  const source = settings && !user && (
    <p class="dialog-source">
      {t.setSourceLbl} {t.originLabel((settings as SettingsView).config_origin)} ·{" "}
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
        {(settings as SettingsView).config_path}
      </button>
    </p>
  );

  const modelRow = (deployment: Deployment, detected = false) => {
    const test = tests[deployment.deployment_id];
    return (
      <div class="model-row" data-deployment={deployment.deployment_id} key={`${detected ? "d" : "c"}:${deployment.deployment_id}`}>
        <div class="model-line">
          {/* A user's models are one pool: its rows lead with the model's name, as the shared ones do. */}
          <span class="model-group">{user ? modelName(deployment.model) : deployment.routing_group}</span>
          <span class="model-id" title={deployment.model}>
            {deployment.model}
          </span>
          {detected || !deployment.persisted ? (
            <span class="mini-pill" title={t.weightHint}>
              {detected ? t.sessionBadge : narrow ? t.modelWeightShort(deployment.weight) : t.modelWeight(deployment.weight)}
            </span>
          ) : (
            <WeightPill
              t={t}
              model={user ? modelName(deployment.model) : deployment.model}
              value={deployment.weight}
              narrow={narrow}
              title={user ? t.userWeightHint : t.weightHint}
              disabled={revision === null}
              onSave={(weight) => saveWeight(deployment.deployment_id, deployment.model, weight)}
              onInvalid={(invalid) => weightInvalid(deployment.deployment_id, invalid)}
            />
          )}
          {canShare && !detected && deployment.persisted && (
            <ShareToggle
              t={t}
              model={deployment.model}
              value={deployment.share === "shared" ? "shared" : "private"}
              busy={sharing === deployment.deployment_id}
              onChange={(next) => void changeShare(deployment, next)}
            />
          )}
          <span class="text-actions">
            <TestButton t={t} model={deployment.model} state={test} onTest={() => void runTest(deployment.deployment_id, deployment.model, { deployment_id: deployment.deployment_id })} />
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
        {!detected && <WeightErrorLine t={t} shown={weightErrors[deployment.deployment_id] === true} />}
        <TestFailureLine t={t} failure={failures[deployment.deployment_id]} />
      </div>
    );
  };

  const label = provider ? providerLabel(t, provider) : "";
  const detailHint = provider
    ? t.providerDetailHint(provider.kind, provider.provider, provider.source, {
        local,
        manualOnly,
        needsBase: needsBase && provider.provider !== "azure" && provider.provider !== "custom",
        subscription,
        // The login note below says how it connects; a saved connection's key is in its rows above.
        cloud: cloud && login === null && !saved,
      })
    : "";
  /** The server's key variables this page's key can come from instead (both Kimi services' on the Kimi card). */
  const keyVariables = provider === null ? [] : kimi ? [services!.moonshot.key_variable, services!.kimi_code.key_variable].filter((name): name is string => Boolean(name)) : provider.key_variable ? [provider.key_variable] : [];

  const status = listError && (
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
                  toast({ tone: "success", text: t.draftKept });
                }}
              >
                {t.useCurrentRevision}
              </button>
            </>
          )}
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
        {picking ? (
          <button type="button" class="btn btn-primary" disabled={addBusy || !weightValid || (!selected.size && !dropped.size)} onClick={() => void addSelected()}>
            {addBusy ? t.saving : t.save}
          </button>
        ) : (
          !manualOnly &&
          discovery === null &&
          !autoLoads && (
            <button type="button" class="btn btn-primary" disabled={discovering || !canLoad} onClick={() => void load(false)}>
              {discovering ? t.loading : t.loadModels}
            </button>
          )
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
      <h2 id="connection-title" class="group-title" tabIndex={-1}>
        {t.connectionTitle}
      </h2>
      {conn === null ? (
        <p class={connFailed ? "line-error" : "dialog-dim"}>{connFailed ? t.providersLoadFailed : t.loading}</p>
      ) : (
        <div class="conn-rows">
          {(takesKey || conn.key.state === "saved" || (conn.key.state === "environment" && conn.key.stored)) && (
            <div class="conn-row">
              <span class="conn-label">{t.setApiKey}</span>
              <span class="conn-value">{keyText(conn)}</span>
              <span class="text-actions">
                {takesKey && (
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
                )}
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
          )}
          {connMode === "key" && takesKey && (
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
            {/* A login provider or a subscription takes no address here (the config file may set one). */}
            {login === null && !subscription && (
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
            )}
          </div>
          {connMode === "base" && login === null && !subscription && (
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
                  <AddressField
                    id="conn-base"
                    value={connBase}
                    choices={endpointChoices}
                    label={endpointName}
                    listLabel={t.endpointList}
                    placeholder={conn.placeholder || "https://example.com/v1"}
                    disabled={connBusy}
                    describedBy={matchingChoice(endpointChoices, connBase)?.plan ? "conn-base-plan" : undefined}
                    onInput={setConnBase}
                  />
                  {matchingChoice(endpointChoices, connBase)?.plan && (
                    <p id="conn-base-plan" class="field-hint">
                      {t.planTerms}
                    </p>
                  )}
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
              {conn.baseMasked && <p class="conn-note">{t.maskedAddressNote}</p>}
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
    <section
      ref={card}
      class="settings-page"
      aria-labelledby="settings-title"
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented || document.querySelector(".confirm-card")) return;
        if (stepBack()) event.preventDefault();
      }}
    >
        <div class="page-head settings-head">
          <nav class="crumbs" aria-label={t.breadcrumbAria}>
            {adding ? (
              <>
                <button type="button" onClick={resetAdd}>
                  {t.navModels}
                </button>
                <span aria-hidden="true">/</span>
                {provider === null ? (
                  <h1 id="settings-title" tabIndex={-1}>
                    {t.addModels}
                  </h1>
                ) : (
                  <>
                    <button type="button" onClick={backToProviders}>
                      {t.addModels}
                    </button>
                    <span aria-hidden="true">/</span>
                    <h1 id="settings-title" tabIndex={-1}>
                      {label}
                    </h1>
                  </>
                )}
              </>
            ) : (
              <h1 id="settings-title" tabIndex={-1}>
                {t.navModels}
              </h1>
            )}
          </nav>
        </div>
        {/* A provider's page is a few sections: they take more room between them. */}
        <div class={adding && provider !== null ? "settings-body is-sections" : "settings-body"}>
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
                      <h2 class="group-title">{autoLoads ? t.modelCatalogTitle : t.connectProviderTitle(label)}</h2>
                      {detailHint && <p>{detailHint}</p>}
                      {(provider.kind === "common" || provider.kind === "compatible") && ((provider.default_base && !baseOptional) || (provider.key_variable && !showsKey)) && (
                        <p class="provider-facts">
                          {/* The address field shows the default itself; the key field names its variable. */}
                          {provider.default_base && !baseOptional && <span class="provider-fact">{provider.default_base}</span>}
                          {provider.key_variable && !showsKey && <span class="provider-fact">{local ? t.optionalKeyVariable(provider.key_variable) : provider.key_variable}</span>}
                        </p>
                      )}
                    </div>
                    {/* Load models is the page's primary action, in the pinned bar with Save; once a
                        catalogue is there, reloading it is this icon. */}
                    {!manualOnly &&
                      (discovery !== null || autoLoads) && (
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
                      )}
                  </div>
                  {(showsKey || baseOptional) && (
                    // The key and the address side by side, on the same grid as the model ID and routing below.
                    <div class={showsKey && (needsBase || baseOptional) ? "field-grid" : "field-grid is-single"}>
                      {showsKey && (
                        <div class="field">
                          <div class="field-label-row">
                            <label class="field-label" for="connect-key">
                              {t.setApiKey}
                              <span class="field-required">{keyRequired ? t.requiredField : t.optionalField}</span>
                            </label>
                            {/* Where else the key can come from: the server's variable, one (i) on every provider's page. */}
                            {keyVariables.length > 0 && !existing && <InfoTip label={t.keyVariableInfo} text={t.keyVariableHint(keyVariables)} />}
                          </div>
                          <input
                            id="connect-key"
                            type="password"
                            value={draftKey}
                            placeholder={t.setKeyPh}
                            autoComplete="off"
                            aria-describedby={keyVariables.length > 0 && !existing ? "connect-key-hint" : undefined}
                            onInput={(event) => setDraftKey(event.currentTarget.value)}
                          />
                          {keyVariables.length > 0 && !existing && (
                            <span id="connect-key-hint" class="sr-only">
                              {t.keyVariableHint(keyVariables)}
                            </span>
                          )}
                        </div>
                      )}
                      {((showsKey && needsBase) || baseOptional) && (
                        <div class="field">
                          <label class="field-label" for="connect-base">
                            {t.setApiBase}
                            <span class="field-required">{needsBase ? t.requiredField : t.optionalField}</span>
                          </label>
                          <AddressField
                            id="connect-base"
                            value={draftBase}
                            choices={endpointChoices}
                            label={endpointName}
                            listLabel={t.endpointList}
                            describedBy={matchingChoice(endpointChoices, draftBase)?.plan ? "connect-base-plan" : undefined}
                            placeholder={
                              kimi || foundFromKey(provider)
                                ? t.baseAuto
                                : needsBase
                                ? provider.provider === "hosted_vllm"
                                  ? "http://127.0.0.1:8000/v1"
                                  : "https://example.com/v1"
                                : ollama
                                  ? "http://127.0.0.1:11434"
                                  : (provider.default_base ?? "https://example.com/v1")
                            }
                            onInput={setDraftBase}
                          />
                          {/* A Coding Plan address: its terms, in one line. */}
                          {matchingChoice(endpointChoices, draftBase)?.plan && (
                            <p id="connect-base-plan" class="field-hint">
                              {t.planTerms}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                  {login !== null && <p class="conn-note login-note">{t.ownEndpointNote(label, login === "google" ? t.googleLogin : login)}</p>}
                  {/* Kimi's terms, on a saved Kimi Code connection: one line. */}
                  {saved && provider.provider === "kimi_code" && <p class="conn-note">{t.kimiMemberTerms}</p>}
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
                {discovery && runtimeFailure(discovery) ? (
                  <RuntimeNotice t={t} failure={runtimeFailure(discovery)!} provider={provider.provider} locale={locale} />
                ) : discovery && listFailure(discovery) ? (
                  <div class="test-failure" role="alert">
                    <Icon name="WarningCircle" size={16} />
                    <div class="test-failure-copy">
                      <p class="test-failure-title">
                        <strong>{t.testFailTitle(listFailure(discovery)!.reason)}</strong>
                        <span class="test-failure-model">{t.modelsUnavailable}</span>
                      </p>
                      <p>
                        {kimiRefused
                          ? t.kimiRefused
                          : t.discoveryFailHint(
                              listFailure(discovery)!.reason,
                              listFailure(discovery)!.variable,
                              refusedHosts(discovery.endpoints, endpoint || target?.default_base || ""),
                              target?.provider ?? provider.provider,
                            )}
                      </p>
                      {listWords && (
                        <details class="test-failure-detail">
                          <summary>{listWords.kind === "connection" ? t.connectionDetail : t.testFailDetail}</summary>
                          <p>{listWords.text}</p>
                        </details>
                      )}
                    </div>
                  </div>
                ) : discovery && discovery.status !== "ok" && (
                  <div class={discovery.status === "unavailable" ? "discovery-note is-error" : "discovery-note"} role={discovery.status === "unavailable" ? "alert" : "status"}>
                    <strong>{discovery.status === "unavailable" ? t.modelsUnavailable : t.modelsPartial}</strong>
                    {discovery.detail && <span title={serviceNote(locale, discovery.detail).detail || undefined}>{serviceNote(locale, discovery.detail).text}</span>}
                  </div>
                )}
                {discovery && region && (!region.kept || draftBase.trim() === region.base) && (
                  <div class="discovery-note" role="status">
                    {/* The address field shows where the key belongs; a saved connection is told to move. */}
                    <strong>{region.kept ? t.regionMovedTitle : t.regionElsewhereTitle}</strong>
                    {!region.kept && <span>{t.regionElsewhere(region.base)}</span>}
                  </div>
                )}
                {/* The Kimi card names the service that took the key; Kimi's terms only for a membership key. */}
                {discovery && kimi && kimiFound && (
                  <div class="discovery-note" role="status">
                    <strong>{t.kimiFound(kimiFound.service)}</strong>
                    {kimiFound.service === "kimi_code" && <span>{t.kimiMemberTerms}</span>}
                  </div>
                )}
                {/* An unavailable catalogue still leaves manual model entry. */}
                {(discovery || manualOnly) && (
                  <ModelPicker
                    t={t}
                    manualOnly={manualOnly}
                    provider={target?.provider ?? provider.provider}
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
                    groups={!user}
                    defaultBase={target?.default_base ?? null}
                    onWeightValid={setWeightValid}
                    {...(discovery && runtimeFailure(discovery)
                      ? addingIsBlocked(runtimeFailure(discovery))
                        ? { addBlocked: t.addBlockedByRuntime(runtimeFailure(discovery)!.name) }
                        : { addWarning: t.addWarnByRuntime(runtimeFailure(discovery)!.name) }
                      : {})}
                    {...(user ? { weightHint: t.userWeightHint } : {})}
                  />
                )}
              </>
            )
          ) : (
            <>
              {settings.deployments.length === 0 && (!user || shared.length === 0) ? (
                <StateBlock
                  icon="Cpu"
                  title={user ? t.userModelsNone : t.setStatusNone}
                  text={user ? t.userModelsEmptyText : t.modelsEmptyText}
                  primary={{ label: t.addModels, run: () => setAdding(true), ref: addButton }}
                />
              ) : (
                <>
                  <div class="settings-summary">
                    <span class="summary-label">
                      {settings.deployments.length === 0 ? t.userModelsOwnNone : user ? t.userModelsOwn(settings.deployments.length) : t.modelsConfigured(settings.deployments.length)}
                      {user && <InfoTip label={t.routingInfo} text={t.userModelsRouting} />}
                    </span>
                    <button ref={addButton} type="button" class="btn btn-primary" onClick={() => setAdding(true)}>
                      {t.addModels}
                    </button>
                  </div>
                  {settings.deployments.length > 0 && <div class="model-rows">{settings.deployments.map((deployment) => modelRow(deployment))}</div>}
                </>
              )}
              {user && shared.length > 0 && (
                <SharedModels
                  t={t}
                  models={shared}
                  narrow={narrow}
                  tests={tests}
                  failures={failures}
                  onTest={(model) => void runTest(`shared:${model.id}`, model.model, { shared_id: model.id }, true)}
                  onWeight={(model, weight) => saveWeight(model.id, model.model, weight, true)}
                  weightErrors={weightErrors}
                  onWeightInvalid={weightInvalid}
                  labelOf={(name) => {
                    const card = providers.find((item) => item.provider === name);
                    return card ? providerLabel(t, card) : name === "custom" ? t.providerCustom : name;
                  }}
                />
              )}
              {settings.detected.length > 0 && (
                <section class="detected">
                  <h2 class="group-title">{t.detectedSession}</h2>
                  <div class="model-rows">{settings.detected.map((deployment) => modelRow(deployment, true))}</div>
                </section>
              )}
              {source}
            </>
          )}
          {!footer && status}
        </div>
        {footer && (
          <div class="settings-foot">
            {status}
            {footer}
          </div>
        )}
    </section>
  );
}

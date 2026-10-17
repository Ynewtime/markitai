// The workbench: a sidebar shell around three views — Convert (`/`),
// Conversions (`/jobs`) and Settings (`/settings/…`). The Conversions view
// keeps the selected item and the open preview in its address, so reloads and
// Back return to them. Notices sit in the page, never over its buttons.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ApiError, documentPath, fetchBlob, fetchCapabilities, fetchOriginal, filePath, itemArchivePath, NetworkError, setItemAssets, setTicketDownloads } from "./api/client.ts";
import { setToken } from "./api/token.ts";
import type { Capabilities, ConvertDefaultsView, JobOptions } from "./api/types.ts";
import { CloudflareDialog } from "./components/cloudflare-dialog.tsx";
import { ConversionsPage } from "./components/conversions-page.tsx";
import { ConvertPage } from "./components/convert-page.tsx";
import { DropOverlay } from "./components/drop-overlay.tsx";
import { MAX_NOTES, NoticeStack, type Note } from "./components/notes.tsx";
import { onToast, toast, withLead } from "./lib/toast.ts";
import { OptionsPanel } from "./components/options-panel.tsx";
import { AccessSettings, AppearanceSettings, SettingsIndex } from "./components/settings-page.tsx";
import { ConfigSettings } from "./components/settings-config.tsx";
import { ModelsSettings } from "./components/settings-models.tsx";
import { Shell, useSidebar } from "./components/shell.tsx";
import type { PanelState } from "./components/conversions-page.tsx";
import { StateBlock } from "./components/state-block.tsx";
import { TokenDialog } from "./components/token-dialog.tsx";
import { UrlInput } from "./components/url-input.tsx";
import type { InspectorContext, ItemActionsApi, JobActionsApi } from "./components/inspector.tsx";
import { useCloudflareConsent } from "./hooks/use-cloudflare-consent.ts";
import { useConnectivity } from "./hooks/use-connectivity.ts";
import { useConversions } from "./hooks/use-conversions.ts";
import { useMedia } from "./hooks/use-media.ts";
import { detectLocale, dicts, loadLocale, storeLocale, type Locale } from "./i18n/index.ts";
import { apiErrorText } from "./i18n/errors.ts";
import { LANG_TAGS } from "./i18n/locale.ts";
import { versionLine } from "./lib/build.ts";
import { isFailed, isJobKey, jobKey, submissions, type Filter, type LabelWords } from "./lib/conversions.ts";
import { download } from "./lib/download.ts";
import { addStaged, oversized, removeStaged, type StagedSubmission, type Walked } from "./lib/files.ts";
import { basename, displayName, fmtBytes } from "./lib/format.ts";
import { useConvertDefaults } from "./hooks/use-convert-defaults.ts";
import { initialComposer, publicOptions, readRemembered, remember, resolveOptions, withOcrFor, type Composer } from "./lib/options.ts";
import { prepareRetryBatch, retryBatchAnnouncement, submitRetryBatch } from "./lib/retry-batch.ts";
import { ADMIN_SECTIONS, DIALOG_ENTRY, entryOnly, listEntry, listQuery, ONLY_ENTRY, OPEN_MODE, parseListState, parseRoute, routePath, SUMMARY_ENTRY, type PaneMode, type Route, type Section } from "./lib/route.ts";
import { canRetry, waitingJobs, type SessionItem } from "./lib/session.ts";
import { parseUrls } from "./lib/urls.ts";

const DEFAULT_DOCS = "https://markitai.dev";
const DEFAULT_SOURCE = "https://github.com/Ynewtime/markitai-rust";
/** Below this width the inspector is a bottom sheet and the preview a full-screen view. */
const PHONE = "(max-width: 960px)";

interface Upload {
  loaded: number;
  total: number;
}

let noteSequence = 0;

export function App() {
  const [locale, setLocale] = useState<Locale>(() => detectLocale());
  const t = dicts[locale];
  const localeRef = useRef(locale);
  localeRef.current = locale;
  useEffect(() => {
    document.documentElement.lang = LANG_TAGS[locale];
  }, [locale]);
  const chooseLocale = useCallback((next: Locale) => {
    storeLocale(next);
    // A pack that cannot load leaves the choice stored and the page in its current language.
    void loadLocale(next).then(
      () => setLocale(next),
      () => undefined,
    );
  }, []);

  // ---- live region: settled items, copies, retries
  const [live, setLive] = useState("");
  const announce = useCallback((message: string) => setLive((previous) => (previous === message ? `${message} ` : message)), []);
  useEffect(() => setLive(""), [locale]);

  // ---- transient notices: oldest first, newest last; a fourth removes the oldest
  const [notes, setNotes] = useState<Note[]>([]);
  const note = useCallback((next: Omit<Note, "id"> & { id?: string }) => {
    const id = next.id ?? `note-${++noteSequence}`;
    setNotes((previous) => [...previous.filter((entry) => entry.id !== id), { ...next, id }].slice(-MAX_NOTES));
    return id;
  }, []);
  const dismiss = useCallback((id: string) => setNotes((previous) => previous.filter((entry) => entry.id !== id)), []);
  // Notices raised by any page (a model test, a saved file) join the same stack.
  useEffect(
    () =>
      onToast((next) => {
        if ((next as { dismiss?: boolean }).dismiss) {
          if (next.id) dismiss(next.id);
        } else note(next);
      }),
    [note, dismiss],
  );

  // ---- service capabilities
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const versionInfo = useMemo(() => versionLine(caps), [caps]);
  const [unauthorized, setUnauthorized] = useState(false);
  const refreshCaps = useCallback(() => {
    fetchCapabilities().then(
      (value) => {
        setItemAssets(value.features?.item_assets === true);
        setTicketDownloads(value.features?.download_tickets === true);
        setCaps(value);
        setUnauthorized(false);
      },
      () => undefined,
    );
  }, []);
  useEffect(refreshCaps, [refreshCaps]);
  useEffect(() => {
    const onUnauthorized = () => setUnauthorized(true);
    window.addEventListener("markitai:unauthorized", onUnauthorized);
    return () => window.removeEventListener("markitai:unauthorized", onUnauthorized);
  }, []);
  const llmReady = caps?.llm.routable === true;
  const maxItems = caps?.limits.max_job_items ?? 1000;
  const presets = caps?.preset_options;
  const features = caps?.features ?? null;
  // v1.13: a user token converts and sees its own conversions; settings are the admin's.
  const role = caps?.access?.role ?? "admin";
  // v1.15: a user token also manages models of its own (and sees the ones the admin shares).
  const userModels = features?.user_models === true;
  const sections: Section[] =
    role === "admin"
      ? ["appearance", "models", "access", ...(features?.config_edit === true ? (["config"] as const) : [])]
      : userModels
        ? ["appearance", "models", ...(features?.user_config === true ? (["config"] as const) : [])]
        : ["appearance"];
  const docsUrl = caps?.docs_url && /^https:\/\//.test(caps.docs_url) ? caps.docs_url : DEFAULT_DOCS;
  const sourceUrl = caps?.source_url && /^https:\/\//.test(caps.source_url) ? caps.source_url : DEFAULT_SOURCE;

  // ---- composer options: v1.17 backends give their defaults (below); older ones, the last choice in this browser
  const [composer, setComposer] = useState<Composer>(() => initialComposer(readRemembered()));
  const serverDefaults = features?.convert_defaults === true;
  useEffect(() => {
    if (caps !== null && !serverDefaults) remember(composer);
  }, [composer, caps, serverDefaults]);
  useEffect(() => {
    // Also after the server's defaults arrive: a default that asks for a model waits for one.
    if (caps !== null && !caps.llm.routable) setComposer((state) => (state.llm || state.preset !== "minimal" ? { ...state, preset: "minimal", llm: false } : state));
  }, [caps, composer.llm, composer.preset]);
  const options = useMemo(() => resolveOptions(composer, presets), [composer, presets]);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  /** Text for any failure, in the interface language; the service's wording stays as detail. */
  const describe = useCallback((error: unknown): { text: string; detail: string } => {
    const words = dicts[localeRef.current];
    if (error instanceof NetworkError) return { text: words.submitNetworkFailed, detail: "" };
    if (error instanceof ApiError) {
      const known = apiErrorText(localeRef.current, error.body);
      if (known) return known;
      if (error.status === 413) return { text: words.submitTooLarge, detail: error.detail };
      if (error.status === 422) return { text: words.submitInvalid, detail: error.detail };
      if (error.status >= 500) return { text: words.submitServerFailed, detail: error.detail };
      return { text: words.submitHttpFailed(error.status), detail: error.detail };
    }
    if (error instanceof RangeError) return { text: words.diffTooLarge, detail: "" };
    return { text: error instanceof Error ? error.message : String(error), detail: "" };
  }, []);
  const onDefaultsSaved = useCallback(
    (view: ConvertDefaultsView, removed: boolean) =>
      toast({ id: "defaults-saved", tone: "success", text: removed ? dicts[localeRef.current].defaultsFollowServer : view.scope === "user" ? dicts[localeRef.current].defaultsSavedOwn : dicts[localeRef.current].defaultsSavedServer }),
    [],
  );
  const defaults = useConvertDefaults({ enabled: serverDefaults, composer, setComposer, presets, describe, onSaved: onDefaultsSaved });

  const conversions = useConversions(
    features,
    caps !== null,
    (done, failed, retained, skipped) => (retained > 0 ? dicts[localeRef.current].notifyRetained(retained) : dicts[localeRef.current].notifyBody(done, failed, skipped)),
    () => dicts[localeRef.current].connLost,
  );
  const conversionsRef = useRef(conversions);
  conversionsRef.current = conversions;

  const cloudflare = useCloudflareConsent(caps?.remote_services?.cloudflare);
  const authorize = cloudflare.authorize;

  // ---- connectivity
  const offline = useConnectivity(() => {
    refreshCaps();
    void conversionsRef.current.refresh();
    void conversionsRef.current.retryRestore();
    dismiss("offline");
    note({ tone: "success", text: dicts[localeRef.current].reconnected });
  });
  // An unreachable server is a lasting condition, not a notice: the list's panel
  // says it when nothing could be loaded, and a failed action says it in its
  // conversion's inspector. The connectivity hook asks again every few seconds.

  // ---- routes
  const [route, setRoute] = useState<Route>(() => parseRoute(location.pathname));
  const [listState, setListState] = useState(() => parseListState(location.search));
  const navigate = useCallback((next: Route) => {
    const path = routePath(next);
    const narrowed = next.view === "jobs" ? onlyRef.current : null;
    if (location.pathname !== path || location.search) history.pushState(narrowed ? { wb: ONLY_ENTRY, only: narrowed } : null, "", path);
    setRoute(next);
    if (next.view !== "jobs") setListState({ selected: null, mode: OPEN_MODE });
    requestAnimationFrame(() => document.getElementById("main")?.focus({ preventScroll: true }));
  }, []);
  useEffect(() => {
    const onPop = () => {
      setRoute(parseRoute(location.pathname));
      setListState(parseListState(location.search));
      setOnly(location.pathname === "/jobs" ? entryOnly(listEntry()) : null);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  /** Opening a conversion's window from the list adds a history entry (marked, so the page knows it is its own):
   * Back, the trackpad swipe and Esc then all close the window the same way. An item opened from the summary or
   * from a submission's window (a notice's name, Preview) adds one more level: Back returns to that window, while
   * ✕ and Esc go back over every level to the list. Moving between conversions and tabs replaces the address. */
  const updateList = useCallback((next: { selected: string | null; mode: PaneMode }, push = false) => {
    setListState(next);
    const url = `/jobs${listQuery(next)}`;
    if (location.pathname + location.search === url) return;
    const onList = location.pathname === "/jobs";
    const wasOpen = onList && parseListState(location.search).selected !== null;
    const entry = listEntry();
    const narrowed = onList ? entryOnly(entry) : null;
    if (next.selected === null) {
      if (wasOpen && entry?.wb === DIALOG_ENTRY) history.go(-Math.max(1, entry.depth ?? 1));
      else history.replaceState(narrowed ? { wb: ONLY_ENTRY, only: narrowed } : null, "", url);
      return;
    }
    if (onList && !wasOpen) history.pushState({ wb: DIALOG_ENTRY, depth: entry?.wb === SUMMARY_ENTRY ? 2 : 1, only: narrowed ?? undefined }, "", url);
    else if (onList && push && entry?.wb === DIALOG_ENTRY) history.pushState({ wb: DIALOG_ENTRY, depth: Math.max(1, entry.depth ?? 1) + 1, only: narrowed ?? undefined }, "", url);
    else if (push) history.pushState(narrowed ? { wb: ONLY_ENTRY, only: narrowed } : null, "", url);
    else history.replaceState(history.state, "", url);
  }, []);
  // A wide screen never shows the phone's settings index.
  const phone = useMedia(PHONE);
  useEffect(() => {
    if (route.view === "settings" && route.section === null && !phone) {
      history.replaceState(null, "", "/settings/appearance");
      setRoute({ view: "settings", section: "appearance" });
    }
  }, [route, phone]);

  const [collapsedSidebar, toggleSidebar] = useSidebar();
  const routeRef = useRef(route);
  routeRef.current = route;
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  // Exactly these items are listed (a notice's "Show items"): item keys, or none.
  const [only, setOnly] = useState<string[] | null>(() => (location.pathname === "/jobs" ? entryOnly(listEntry()) : null));
  const onlyRef = useRef(only);
  onlyRef.current = only;
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set());
  const [busy, setBusy] = useState<Set<string>>(() => new Set());
  const mark = (key: string, on: boolean) =>
    setBusy((previous) => {
      const next = new Set(previous);
      if (on) next.add(key);
      else next.delete(key);
      return next;
    });

  const titles: Record<Route["view"], string> = { convert: t.titleHome, jobs: t.titleWorkspace, settings: t.titleSettings, missing: t.titleNotFound };
  useEffect(() => {
    const base = titles[route.view];
    document.title = conversions.activeCount > 0 ? `${conversions.activeCount} · ${base}` : base;
  }, [route.view, conversions.activeCount, t]);

  // ---- submissions
  const [urlText, setUrlText] = useState("");
  const urls = useMemo(() => parseUrls(urlText).urls, [urlText]);
  const [staged, setStaged] = useState<StagedSubmission | null>(null);
  const controllers = useRef(new Set<AbortController>());
  const [upload, setUpload] = useState<Upload | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const cancelSubmit = useCallback(() => {
    for (const controller of controllers.current) controller.abort();
    controllers.current.clear();
    setSubmitting(false);
    setUpload(null);
    announce(t.submitCancelled);
  }, [announce, t.submitCancelled]);
  useEffect(() => () => controllers.current.forEach((controller) => controller.abort()), []);

  const sending = useRef(false);
  const send = useCallback(
    async (files: File[], list: string[], label: string | null = null) => {
      if (sending.current) return false;
      sending.current = true;
      try {
        const authorized = await authorize(optionsRef.current, [
          ...files.map((file) => ({ name: file.name, kind: "file" as const })),
          ...list.map((name) => ({ name, kind: "url" as const })),
        ]);
        if (authorized === null) return false;
        const controller = new AbortController();
        controllers.current.add(controller);
        setSubmitting(true);
        setUpload(files.length ? { loaded: 0, total: 0 } : null);
        let last = 0;
        const jobId = await conversionsRef.current.submit(
          files,
          list,
          authorized,
          {
            signal: controller.signal,
            onProgress: (loaded, total) => {
              const now = Date.now();
              if (now - last < 100 && !(total > 0 && loaded >= total)) return;
              last = now;
              setUpload({ loaded, total });
            },
          },
          label,
        );
        controllers.current.delete(controller);
        setSubmitting(controllers.current.size > 0);
        if (!controllers.current.size) setUpload(null);
        if (jobId !== null) {
          // The new submission is selected in the list: one item, or its group.
          const count = files.length + list.length;
          setRoute({ view: "jobs" });
          // One item opens on Preview once it is converted (Details while it runs or when it fails).
          updateList({ selected: count > 1 ? jobKey(jobId) : `${jobId}/i1`, mode: OPEN_MODE }, location.pathname !== "/jobs");
          setFilter("all");
          setQuery("");
          setOnly(null);
        }
        return jobId !== null;
      } finally {
        sending.current = false;
      }
    },
    [authorize, updateList],
  );

  // The model went away since the page last asked: show the real state.
  useEffect(() => {
    if (conversions.submitError instanceof ApiError && conversions.submitError.reason === "llm_unavailable") refreshCaps();
  }, [conversions.submitError, refreshCaps]);

  // Chosen, picked or dropped files wait for Convert, beside any URLs typed; the
  // staged list shows them, each with its own remove button.
  const stagedRef = useRef(staged);
  stagedRef.current = staged;
  const submitFiles = useCallback(
    (files: File[], folder?: { hidden: number; unreadable: number; truncated: boolean; folder?: boolean; name?: string }) => {
      dismiss("input");
      conversionsRef.current.clearSubmitError();
      const notes: string[] = [];
      if (folder && files.length === 0) {
        note({ id: "input", tone: "warning", text: t.dropEmptyFolder });
        return;
      }
      const large = oversized(files);
      if (large.length) {
        note({ id: "input", tone: "error", text: t.filesTooLarge(large.map((file) => `${file.name} (${fmtBytes(file.size)})`).join(", ")) });
        return;
      }
      const { staged: next, dropped } = addStaged(stagedRef.current, files, folder?.folder === true, maxItems);
      if (dropped) notes.push(t.dropTruncated(maxItems, (next?.files.length ?? 0) + dropped));
      else if (folder?.truncated) notes.push(t.dropLimit(files.length));
      if (folder?.hidden) notes.push(t.folderHidden(folder.hidden));
      if (folder?.unreadable) notes.push(t.folderUnreadable(folder.unreadable));
      if (notes.length) note({ id: "drop", tone: "warning", text: notes.join(" · ") });
      stagedRef.current = next;
      setStaged(next);
      // The staged list and its Convert button live on the Convert page.
      if (routeRef.current.view !== "convert") navigate({ view: "convert" });
    },
    [maxItems, t, note, dismiss, navigate],
  );
  const submitFolder = useCallback((files: File[], hidden: number) => submitFiles(files, { hidden, unreadable: 0, truncated: false, folder: true }), [submitFiles]);
  const submitWalked = useCallback((walked: Walked) => submitFiles(walked.files, walked), [submitFiles]);

  const submitUrls = useCallback(
    async (text: string) => {
      dismiss("input");
      conversionsRef.current.clearSubmitError();
      const parsed = parseUrls(text);
      if (parsed.invalid) {
        note({ id: "input", tone: "error", text: parsed.invalid.reason === "badScheme" ? t.badScheme(parsed.invalid.value) : t.badUrl(parsed.invalid.value) });
        return false;
      }
      let list = parsed.urls;
      if (list.length > maxItems) {
        note({ id: "drop", tone: "warning", text: t.dropTruncated(maxItems, list.length) });
        list = list.slice(0, maxItems);
      }
      const files = staged?.files ?? [];
      const label = staged?.label.trim() ? staged.label : null;
      if (!list.length && !files.length) return false;
      const ok = await send(files, list, label);
      if (ok) setStaged(null);
      return ok;
    },
    [maxItems, send, staged, t, note, dismiss],
  );

  // ---- a failed action on one conversion: an error toast (it stays until closed),
  // replaced when the action is tried again
  const failAction = useCallback(
    (key: string, failure: string, error?: unknown) => {
      const words = error === undefined ? { text: "", detail: "" } : describe(error);
      const [first = "", ...rest] = words.text.split(/(?<=[.!?。！？])\s*/);
      const text = withLead(failure, first, dicts[localeRef.current].toastLead);
      const detail = [rest.join(" ").trim(), words.detail].filter(Boolean).join("\n");
      note({ id: `action:${key}`, tone: "error", text, ...(detail ? { detail } : {}) });
    },
    [describe, note],
  );
  const clearFailure = useCallback((key: string) => dismiss(`action:${key}`), [dismiss]);

  // ---- item actions
  const requestItem = useCallback(
    async (item: SessionItem, request: () => Promise<unknown>, failure: string) => {
      clearFailure(item.key);
      mark(item.key, true);
      let error: unknown = null;
      try {
        error = await request();
      } catch (reason) {
        error = reason;
      }
      mark(item.key, false);
      if (error !== null && error !== undefined) failAction(item.key, failure, error);
      return error;
    },
    [clearFailure, failAction],
  );
  const savedOptions = (item: SessionItem): JobOptions => publicOptions(item.options ?? conversionsRef.current.jobs[item.jobId]?.options ?? optionsRef.current);
  const retry = useCallback(
    async (item: SessionItem, override?: JobOptions) => {
      const base = override ?? savedOptions(item);
      const next = item.skipped && item.skipReason === "image_only" ? withOcrFor(base) : base;
      const authorized = await authorize(next, [{ name: item.name, kind: item.kind }], true);
      if (authorized === null) return;
      const error = await requestItem(item, () => conversionsRef.current.retry(item, authorized), dicts[localeRef.current].retryFailed);
      if (error === null) announce(dicts[localeRef.current].retryAria(item.name));
    },
    [authorize, requestItem, announce],
  );
  const itemActions: ItemActionsApi = {
    preview: (item) => {
      setRoute({ view: "jobs" });
      updateList({ selected: item.key, mode: "preview" }, true);
    },
    compare: (item) => {
      setRoute({ view: "jobs" });
      updateList({ selected: item.key, mode: "compare" }, true);
    },
    download: (item, path) => {
      clearFailure(item.key);
      void download(filePath(item.jobId, path), basename(path)).catch((error: unknown) => failAction(item.key, dicts[localeRef.current].downloadFailedFile, error));
    },
    downloadJson: (item) => {
      clearFailure(item.key);
      const stem = basename(item.output ?? item.name).replace(/\.[^.]*$/, "").replace(/\.(llm|v\d+)$/, "");
      void download(documentPath(item.jobId, item.itemId), `${stem}.json`).catch((error: unknown) => failAction(item.key, dicts[localeRef.current].downloadFailedFile, error));
    },
    copyFailed: (item) => failAction(item.key, dicts[localeRef.current].copyFailed),
    retry: (item) => void retry(item),
    retryPlain: (item) => void retry(item, { ...savedOptions(item), llm: false, alt: null, desc: null }),
    retryWithOcr: (item) => {
      setComposer((state) => ({ ...state, ocr: true }));
      void retry(item);
    },
    reconvert: (item) => void reconvert([item]),
    enhance: (item) => {
      void (async () => {
        const authorized = await authorize({ ...savedOptions(item), llm: true }, [{ name: item.name, kind: item.kind }], true);
        if (authorized === null) return;
        const error = await requestItem(item, () => conversionsRef.current.enhance(item, authorized), dicts[localeRef.current].llmEnhanceFailed);
        if (error === null) announce(dicts[localeRef.current].enhanceWithLlm(item.name));
      })();
    },
    remove: async (item) => {
      const error = await requestItem(item, () => conversionsRef.current.removeItem(item), dicts[localeRef.current].deleteFailed);
      if (error !== null) return false;
      announce(dicts[localeRef.current].histDeleted(item.name));
      updateList({ selected: null, mode: OPEN_MODE });
      return true;
    },
    openModels: () => navigate({ view: "settings", section: "models" }),
  };

  const retryItems = useCallback(
    async (items: SessionItem[], key: string) => {
      const batch = prepareRetryBatch(items, conversionsRef.current.jobs, optionsRef.current);
      if (!batch.length) return;
      mark(key, true);
      try {
        const authorized = await cloudflare.authorizeBatch(batch.map(({ item, options }) => ({ options, sources: [{ name: item.name, kind: item.kind }] })), true);
        if (key.startsWith("job:")) clearFailure(key);
        const result = await submitRetryBatch(batch, authorized, (item, options) => conversionsRef.current.retry(item, options));
        if (result) announce(retryBatchAnnouncement(batch, result, dicts[localeRef.current]));
        // One submission's retries fail in its inspector; all of them, in the page.
        if (result?.failed && key.startsWith("job:")) failAction(key, dicts[localeRef.current].announceRetryAllFailed(result.attempted, result.failed));
        else if (result?.failed)
          note({
            tone: "error",
            text: dicts[localeRef.current].announceRetryAllFailed(result.attempted, result.failed),
            action: { label: dicts[localeRef.current].noteView, run: () => (setRoute({ view: "jobs" }), setFilter("failed")) },
          });
      } finally {
        mark(key, false);
      }
    },
    [cloudflare.authorizeBatch, announce, note, clearFailure, failAction],
  );
  // ---- Re-convert: the originals of finished items go back to the Convert page, options open
  const [optionsRequest, setOptionsRequest] = useState(0);
  const reconverting = useRef(false);
  const reconvert = useCallback(
    async (items: SessionItem[]) => {
      if (reconverting.current || !items.length) return;
      reconverting.current = true;
      const words = dicts[localeRef.current];
      try {
        // Kept uploads are read back from the server (a few at a time); a page is converted again from its address.
        const found: ({ file: File } | { url: string } | null)[] = new Array(items.length).fill(null);
        let next = 0;
        const worker = async () => {
          for (let at = next++; at < items.length; at = next++) {
            const item = items[at];
            if (!item) continue;
            if (item.kind === "url") {
              if (/^https?:\/\//i.test(item.name)) found[at] = { url: item.name };
              continue;
            }
            try {
              const original = await fetchOriginal(item.jobId, item.itemId);
              if (!original.download) continue;
              const { blob } = await fetchBlob(original.download);
              found[at] = { file: new File([blob], item.name, { type: blob.type }) };
            } catch {
              // Not kept any more (or not readable): reported with the others below.
            }
          }
        };
        await Promise.all([worker(), worker(), worker(), worker()]);
        const files = found.flatMap((entry) => (entry && "file" in entry ? [entry.file] : []));
        const addresses = found.flatMap((entry) => (entry && "url" in entry ? [entry.url] : []));
        const gone = items.filter((_, at) => found[at] === null).map((item) => displayName(item.name));
        if (!files.length && !addresses.length) {
          note({ id: "reconvert", tone: "error", text: words.reconvertNone, ...(gone.length ? { detail: gone.join("\n") } : {}) });
          return;
        }
        // They join whatever already waits on the Convert page.
        const { staged: joined } = addStaged(stagedRef.current, files, false, maxItems);
        stagedRef.current = joined;
        setStaged(joined);
        setUrlText((text) => {
          const lines = text.split(/\r?\n/).filter((line) => line.trim());
          return [...lines, ...addresses.filter((address) => !lines.includes(address))].join("\n");
        });
        navigate({ view: "convert" });
        setOptionsRequest((value) => value + 1);
        const count = files.length + addresses.length;
        announce(words.reconvertReady(count));
        if (gone.length) note({ id: "reconvert", tone: "warning", text: words.reconvertGone(gone.length), detail: gone.join("\n") });
        // LLM enhancement is one of the options the run may change: say at once when no model can serve it.
        if (!llmReady)
          note({
            id: "reconvert-model",
            tone: "warning",
            text: words.reconvertNoModel,
            ...(role === "admin" || userModels ? { action: { label: words.addModelAction, run: () => navigate({ view: "settings", section: "models" }) } } : {}),
          });
      } finally {
        reconverting.current = false;
      }
    },
    [maxItems, navigate, announce, note, llmReady, role, userModels],
  );
  const downloadArchive = useCallback(
    (path: string, name: string, key?: string) => {
      if (key) clearFailure(key);
      void download(path, name, features?.download_tickets === true).catch((error: unknown) => {
        if (key) return failAction(key, dicts[localeRef.current].downloadFailed, error);
        const [first = ""] = describe(error).text.split(/(?<=[.!?。！？])\s*/);
        note({ tone: "error", text: withLead(dicts[localeRef.current].downloadFailed, first, dicts[localeRef.current].toastLead) });
      });
    },
    [features, describe, note, clearFailure, failAction],
  );
  const stopJobs = useCallback(
    async (jobIds: string[], key: string) => {
      if (key.startsWith("job:")) clearFailure(key);
      mark(key, true);
      const result = await conversionsRef.current.stop(jobIds);
      mark(key, false);
      if (result.error && key.startsWith("job:")) failAction(key, dicts[localeRef.current].stopRemaining, result.error);
      else if (result.error) {
        const [first = ""] = describe(result.error).text.split(/(?<=[.!?。！？])\s*/);
        note({ tone: "error", text: withLead(dicts[localeRef.current].stopFailed, first, dicts[localeRef.current].toastLead) });
      } else if (result.stopping > 0) announce(dicts[localeRef.current].stopRequested(result.stopping));
    },
    [describe, note, announce, clearFailure, failAction],
  );
  const jobActions: JobActionsApi = {
    zip: (jobId) => downloadArchive(`/api/jobs/${encodeURIComponent(jobId)}/archive`, `markitai-${jobId}.zip`, `job:${jobId}`),
    retryFailed: (jobId) => void retryItems(conversionsRef.current.items.filter((item) => item.jobId === jobId), `job:${jobId}`),
    stop: (jobId) => void stopJobs([jobId], `job:${jobId}`),
    remove: async (jobId) => {
      clearFailure(`job:${jobId}`);
      const error = await conversionsRef.current.removeJob(jobId);
      if (error !== null) {
        failAction(`job:${jobId}`, dicts[localeRef.current].deleteFailed, error);
        return false;
      }
      updateList({ selected: null, mode: OPEN_MODE });
      return true;
    },
    select: (key) => updateList({ selected: key, mode: OPEN_MODE }),
  };

  // Announce a newly settled item once (not on reloads or restored jobs).
  const settledBefore = useRef(new Map<string, string>());
  useEffect(() => {
    const next = new Map<string, string>();
    for (const item of conversions.items) {
      const state = `${item.status}/${item.finishedAt ?? ""}`;
      next.set(item.key, state);
      const before = settledBefore.current.get(item.key);
      if (before !== undefined && before !== state && (item.status === "done" || item.status === "error") && !conversions.restoredQuiet.has(item.jobId)) {
        announce(t.announceSettled(item.name, isFailed(item) ? t.statusFailed : t.statusDone));
      }
    }
    settledBefore.current = next;
  }, [conversions.items, t, announce]);

  // ---- notices from the store
  const submitError = conversions.submitError;
  const [tokenOpen, setTokenOpen] = useState(false);
  // A refused submission is a strip with one sentence; the composer keeps its input.
  useEffect(() => {
    if (!submitError) return;
    const words = dicts[localeRef.current];
    // One short sentence; the service's wording is not repeated in a strip.
    if (submitError instanceof ApiError && submitError.status === 401) {
      note({ id: "submit", tone: "error", text: words.submitNeedsToken, action: { label: words.enterToken, run: () => setTokenOpen(true) } });
    } else {
      note({ id: "submit", tone: "error", text: submitError instanceof NetworkError ? words.submitNoServer : words.submitFailedShort });
    }
    conversions.clearSubmitError();
  }, [submitError]);
  const restoreFailed = conversions.restoreFailed.size > 0;
  useEffect(() => {
    if (restoreFailed) note({ id: "restore", tone: "warning", text: dicts[localeRef.current].restoreFailed, action: { label: dicts[localeRef.current].restoreRetry, run: () => void conversionsRef.current.retryRestore() } });
  }, [restoreFailed]);
  const uploadText = (() => {
    if (!upload) return t.submitting;
    if (!(upload.total > 0)) return upload.loaded > 0 ? t.uploadUnknown(fmtBytes(upload.loaded)) : t.submitting;
    if (upload.loaded >= upload.total) return t.uploadDone;
    return t.uploadProgress(Math.min(99, Math.floor((upload.loaded / upload.total) * 100)), fmtBytes(upload.loaded), fmtBytes(upload.total));
  })();
  // An upload in progress is the newest strip until it ends (Cancel stops it).
  const stack: Note[] = [...notes, ...(submitting ? [{ id: "upload", tone: "success" as const, text: uploadText, busy: true, action: { label: t.cancelSubmit, run: cancelSubmit } }] : [])].slice(-MAX_NOTES);

  const composerPanel = (
    <OptionsPanel
      t={t}
      state={composer}
      presets={presets}
      llmReady={llmReady}
      needsModel={role === "user" && !userModels ? t.advNeedsModelAdmin : t.advNeedsModel}
      cloudflare={caps?.remote_services?.cloudflare}
      urls={urls}
      staged={staged?.files ?? []}
      announce={announce}
      busy={submitting || cloudflare.pending !== null}
      openRequest={optionsRequest}
      textOnly={features?.text_only === true}
      defaults={defaults.control}
      onChange={(next) => {
        defaults.touch();
        setComposer(next);
      }}
      onFiles={(files) => submitFiles(files)}
      onFolder={submitFolder}
      source={(tools) => (
        <UrlInput
          t={t}
          text={urlText}
          onText={setUrlText}
          onConvert={submitUrls}
          busy={submitting || cloudflare.pending !== null}
          staged={staged}
          onClearStaged={() => setStaged(null)}
          onRemoveStaged={(file) => setStaged((current) => (current ? removeStaged(current, file) : null))}
          tools={tools}
        />
      )}
    />
  );

  const labelWords: LabelWords = { files: t.nFiles, pages: t.nPages, filesAndPages: t.filesAndPages };
  const ctx: InspectorContext = {
    t,
    locale,
    llmReady,
    excerpts: features?.result_excerpt === true,
    documentJson: features?.document_json === true,
    compare: features?.original === true,
    cancel: features?.cancel === true,
    retentionDays: caps?.limits.history_retention_days ?? null,
    // v1.15: a user token opens a Models page of its own.
    admin: role === "admin" || userModels,
    docsUrl,
    busy,
    labelWords,
    describe,
    failure: null,
  };
  const waiting = waitingJobs(conversions.items, conversions.jobs);
  const retryable = conversions.items.filter((item) => (isFailed(item) || item.skipReason === "user_stopped") && canRetry(item));

  // A selection that no longer exists (deleted, expired) falls back to the
  // overview; when it came with the address (a bookmark, a shared link), a
  // notice says why the page shows something else.
  const selected = listState.selected;
  const linked = useRef(listState.selected);
  const [gone, setGone] = useState(false);
  useEffect(() => {
    if (selected !== null) setGone(false);
    if (selected === null || !conversions.loaded) return;
    const exists = selected.startsWith("job:") ? selected.slice(4) in conversions.jobs : conversions.items.some((item) => item.key === selected);
    if (exists || conversions.restoreFailed.size > 0) return;
    updateList({ selected: null, mode: OPEN_MODE });
    if (selected === linked.current) setGone(true);
    linked.current = null;
  }, [selected, conversions.items, conversions.jobs, conversions.loaded]);
  const retentionDays = caps?.limits.history_retention_days;
  const gonePanel: PanelState | null = gone
    ? { icon: "LinkBreak", title: t.goneTitle, text: retentionDays ? t.goneTextDays(retentionDays) : t.goneText, secondary: { label: t.showAllConversions, run: () => setGone(false) } }
    : null;
  // The saved conversions cannot be listed: why, and the way on.
  const blocked: PanelState | null = unauthorized
    ? { icon: "Key", title: t.tokenNeededTitle, text: t.listNeedsToken, primary: { label: t.enterToken, run: () => setTokenOpen(true) } }
    : offline
      ? { icon: "Plugs", title: t.serverDownTitle, text: t.serverDownText, secondary: { label: t.tryNow, run: refreshCaps } }
      : conversions.loadError
        ? { icon: "WarningCircle", title: t.listBlockedTitle, text: t.listLoadFailed, secondary: { label: t.retryLoad, run: () => void conversions.refresh() }, tone: "error" }
        : null;

  const recent = useMemo(() => submissions(conversions.jobs, conversions.items).slice(0, 3), [conversions.jobs, conversions.items]);

  let page;
  if (route.view === "jobs") {
    page = (
      <ConversionsPage
        t={t}
        locale={locale}
        phone={phone}
        jobs={conversions.jobs}
        items={conversions.items}
        loaded={conversions.loaded}
        blocked={blocked}
        gone={gonePanel}
        selected={selected}
        mode={listState.mode}
        filter={filter}
        query={query}
        collapsed={collapsedGroups}
        ctx={ctx}
        itemActions={itemActions}
        jobActions={jobActions}
        labelWords={labelWords}
        compareAvailable={features?.original === true}
        head={{
          stopping: busy.has("all:stop"),
          retrying: busy.has("all:retry"),
          waiting: waiting.length > 0,
          retryable: retryable.length,
          canZip: conversions.items.some((item) => item.status === "done") && conversions.activeCount === 0,
          onStop: () => void stopJobs(waiting, "all:stop"),
          onRetry: () => void retryItems(conversions.items, "all:retry"),
          onZip: () => downloadArchive("/api/history/archive", "markitai-all.zip"),
        }}
        onSelect={(key) => updateList({ selected: key, mode: key === null ? OPEN_MODE : listState.mode })}
        onMode={(mode) => updateList({ selected, mode })}
        onNewConversion={() => navigate({ view: "convert" })}
        onCopied={() => note({ tone: "success", text: dicts[localeRef.current].copied })}
        onFilter={setFilter}
        onQuery={setQuery}
        onToggleGroup={(jobId, open) =>
          setCollapsedGroups((previous) => {
            const next = new Set(previous);
            const shouldOpen = open ?? next.has(jobId);
            if (shouldOpen) next.delete(jobId);
            else next.add(jobId);
            return next;
          })
        }
        only={only}
        onOnly={(keys) => {
          setOnly(keys);
          // Showing everything again keeps the entry: Back still returns to the window "Show items" came from.
          const entry = listEntry();
          if (keys === null && entryOnly(entry)) history.replaceState(entry?.wb === ONLY_ENTRY ? null : { ...entry, only: undefined }, "");
        }}
        // One item opens at once, one level deeper; several are the list, narrowed to exactly them, as an entry
        // of its own, so Back returns to the summary or submission it came from.
        onShowItems={(keys) => {
          const first = keys[0];
          if (keys.length === 1 && first) updateList({ selected: first, mode: "inspector" }, true);
          else if (keys.length > 1) {
            history.pushState({ wb: ONLY_ENTRY, only: keys }, "", "/jobs");
            setListState({ selected: null, mode: OPEN_MODE });
            setOnly(keys);
            setFilter("all");
            setQuery("");
          }
        }}
        downloadAll={(item) => {
          const name = `${basename(item.output ?? item.name).replace(/\.llm\.md$/i, "").replace(/\.md$/i, "") || "markitai"}.zip`;
          if (features?.item_archive === true) return () => downloadArchive(itemArchivePath(item.jobId, item.itemId), name, item.key);
          // Without the item route, a submission of this one item zips the same files (not the original).
          if (conversions.items.filter((other) => other.jobId === item.jobId).length === 1)
            return () => downloadArchive(`/api/jobs/${encodeURIComponent(item.jobId)}/archive`, name, item.key);
          return null;
        }}
        announce={announce}
        describe={describe}
        onReconvert={(chosen) => void reconvert(chosen)}
        onDeleteMany={async (jobIds, chosen) => {
          const words = dicts[localeRef.current];
          const { removed, failed } = await conversionsRef.current.removeMany(jobIds, chosen);
          if (selected !== null && !isJobKey(selected) && !conversionsRef.current.items.some((item) => item.key === selected)) updateList({ selected: null, mode: OPEN_MODE });
          if (!failed.length) {
            announce(words.deletedMany(removed));
            note({ tone: "success", text: words.deletedMany(removed) });
            return;
          }
          const [first] = failed;
          const why = first ? describe(first.error).text.split(/(?<=[.!?。！？])\s*/)[0] ?? "" : "";
          note({ tone: "error", text: words.deletedSome(removed, failed.length, first ? displayName(first.item.name) : "", why) });
        }}
      />
    );
  } else if (route.view === "missing") {
    page = (
      <div class="page page-missing">
        <StateBlock
          heading="h1"
          icon="Compass"
          title={t.notFoundTitle}
          text={t.notFoundText(route.path)}
          primary={{ label: t.navConvert, run: () => navigate({ view: "convert" }) }}
          secondary={{ label: t.conversions, run: () => navigate({ view: "jobs" }) }}
        />
      </div>
    );
  } else if (route.view === "settings") {
    const section = route.section ?? (phone ? null : "appearance");
    page =
      section === null ? (
        <div class="page">
          <SettingsIndex t={t} version={versionInfo} docsUrl={docsUrl} sourceUrl={sourceUrl} sections={sections} user={role === "user"} onNavigate={navigate} />
        </div>
      ) : caps === null && !unauthorized && ADMIN_SECTIONS.has(section) ? (
        // Who is asking is not known yet: an admin page waits rather than ask as the wrong one.
        <div class="page">
          <p class="pane-note">{t.loading}</p>
        </div>
      ) : !sections.includes(section) ? (
        <div class="page">
          <StateBlock
            heading="h1"
            icon="Key"
            title={role === "user" ? t.adminOnlyTitle : t.notFoundTitle}
            text={role === "user" ? t.adminOnlyText : t.notFoundText(routePath(route))}
            primary={{ label: t.navAppearanceShort, run: () => navigate({ view: "settings", section: "appearance" }) }}
          />
        </div>
      ) : section === "config" ? (
        <div class="page">
          <ConfigSettings key={role} t={t} scope={role === "user" ? "user" : "admin"} describe={describe} onSaved={refreshCaps} />
        </div>
      ) : section === "models" ? (
        <div class="page">
          <ModelsSettings
            key={role}
            t={t}
            locale={locale}
            scope={role === "user" ? "user" : "admin"}
            canShare={role === "admin" && userModels}
            docsUrl={docsUrl}
            onSaved={refreshCaps}
            announce={announce}
            describe={describe}
          />
        </div>
      ) : section === "appearance" ? (
        <div class="page">
          <AppearanceSettings
            t={t}
            locale={locale}
            onLocale={chooseLocale}
            userAccess={role === "user" ? { label: caps?.access?.label ?? null } : null}
          />
        </div>
      ) : (
        <div class="page">
          <AccessSettings
            t={t}
            locale={locale}
            unauthorized={unauthorized}
            canRegenerate={features?.token_regenerate === true}
            userTokens={features?.user_tokens === true && role === "admin" && !unauthorized}
            describe={describe}
            onToken={() => {
              setUnauthorized(false);
              conversions.clearSubmitError();
              refreshCaps();
              void conversions.refresh();
              void conversions.retryRestore();
            }}
          />
        </div>
      );
  } else {
    page = (
      <ConvertPage
        t={t}
        locale={locale}
        composer={composerPanel}
        recent={recent}
        blocked={conversions.loaded ? null : blocked}
        labelWords={labelWords}
        onOpen={(key) => {
          setRoute({ view: "jobs" });
          updateList({ selected: key, mode: OPEN_MODE }, true);
        }}
      />
    );
  }

  return (
    <>
      <Shell
        t={t}
        route={route}
        version={versionInfo}
        docsUrl={docsUrl}
        sourceUrl={sourceUrl}
        active={conversions.activeCount}
        total={conversions.items.length}
        collapsed={collapsedSidebar}
        onToggle={toggleSidebar}
        onNavigate={navigate}
        sections={sections}
      >
        {page}
      </Shell>
      {cloudflare.pending && cloudflare.scope && (
        <CloudflareDialog
          t={t}
          scope={cloudflare.scope}
          reason={cloudflare.reason}
          retry={cloudflare.pending.retry}
          requestCount={cloudflare.pending.requests.length}
          onClose={() => cloudflare.finish(false)}
          onConfirm={() => cloudflare.finish(true)}
        />
      )}
      {tokenOpen && (
        <TokenDialog
          t={t}
          onClose={() => setTokenOpen(false)}
          onSubmit={(value) => {
            setToken(value);
            setTokenOpen(false);
            setUnauthorized(false);
            conversions.clearSubmitError();
            refreshCaps();
            void conversions.refresh();
            void conversions.retryRestore();
          }}
        />
      )}
      <NoticeStack label={t.notificationsLabel} closeLabel={t.dismiss} detailsLabel={t.toastDetails} lead={t.toastLead} notes={stack} onDismiss={dismiss} />
      <DropOverlay
        label={t.dropToConvert}
        suspended={tokenOpen || cloudflare.pending !== null || route.view === "settings" || (route.view === "jobs" && listState.selected !== null)}
        limit={maxItems}
        onFiles={(files) => submitFiles(files)}
        onFolder={submitWalked}
      />
      <div class="sr-only" role="status" aria-live="polite">
        {live}
      </div>
    </>
  );
}


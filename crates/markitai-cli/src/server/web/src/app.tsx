// The workbench: a home view (`/`) and a workspace view (`/jobs`); settings,
// previews and the token prompt are dialogs over either.
import type { ComponentChildren } from "preact";
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ApiError, fetchCapabilities, NetworkError } from "./api/client.ts";
import { setToken } from "./api/token.ts";
import type { Capabilities, HistoryEntry, JobOptions } from "./api/types.ts";
import { AppFooter, AppHeader } from "./components/app-header.tsx";
import { CloudflareDialog } from "./components/cloudflare-dialog.tsx";
import { useCloudflareConsent } from "./hooks/use-cloudflare-consent.ts";
import { prepareRetryBatch, submitRetryBatch } from "./lib/retry-batch.ts";
import { hasCloudflareRequest } from "./lib/cloudflare.ts";
import { DropOverlay } from "./components/drop-overlay.tsx";
import { CapabilityHint, ErrorLine, InlineAction, NoticeLine } from "./components/feedback.tsx";
import { ClearButton, JobStats, ZipButton } from "./components/job-actions.tsx";
import { Ledger } from "./components/ledger.tsx";
import { domKey } from "./components/ledger-row.tsx";
import { Notification, type NotificationModel } from "./components/notification.tsx";
import { OptionsPanel } from "./components/options-panel.tsx";
import { PreviewModal } from "./components/preview-modal.tsx";
import type { PreviewTarget } from "./components/markdown-view.tsx";
import { SettingsModal } from "./components/settings-modal.tsx";
import { TokenDialog } from "./components/token-dialog.tsx";
import { UrlInput } from "./components/url-input.tsx";
import { useArchive } from "./hooks/use-archive.ts";
import { useConnectivity } from "./hooks/use-connectivity.ts";
import { useJobs } from "./hooks/use-jobs.ts";
import { detectLocale, dicts, storeLocale, type Locale } from "./i18n/index.ts";
import { apiErrorText, persistenceText } from "./i18n/errors.ts";
import { oversized, type Walked } from "./lib/files.ts";
import { fmtBytes } from "./lib/format.ts";
import { initialComposer, publicOptions, readRemembered, remember, resolveOptions, withOcrFor, type Composer } from "./lib/options.ts";
import { canRetry, failedToRetry, isPreviewable, isSettled, settledIdentity, itemFromPayload, waitingJobs, type SessionItem } from "./lib/session.ts";
import { itemNotification, publishNotice, terminalNotices, type ItemRequestFailure, type NotificationState } from "./lib/pricing.ts";
import { parseUrls } from "./lib/urls.ts";

type View = "home" | "workspace";
const WORKSPACE = "/jobs";
const viewOf = (): View => ((location.pathname.replace(/\/+$/, "") || "/") === WORKSPACE ? "workspace" : "home");

interface Upload {
  loaded: number;
  total: number;
}

export function App() {
  const [locale, setLocale] = useState<Locale>(() => detectLocale());
  const t = dicts[locale];
  const localeRef = useRef(locale);
  localeRef.current = locale;
  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  }, [locale]);
  const chooseLocale = useCallback((next: Locale) => {
    setLocale(next);
    storeLocale(next);
  }, []);

  // ---- live region: settled items, copies, retries
  const [live, setLive] = useState("");
  const announce = useCallback((message: string) => setLive((previous) => (previous === message ? `${message} ` : message)), []);
  // A past announcement belongs to the language in which it was emitted.
  // Clear it on a language change instead of replaying an invented completion.
  useEffect(() => setLive(""), [locale]);

  // ---- service capabilities
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [unauthorized, setUnauthorized] = useState(false);
  const refreshCaps = useCallback(() => {
    fetchCapabilities().then(
      (value) => {
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

  // ---- composer options, remembered in this browser
  const [composer, setComposer] = useState<Composer>(() => initialComposer(readRemembered()));
  useEffect(() => remember(composer), [composer]);
  useEffect(() => {
    if (caps !== null && !caps.llm.routable) setComposer((state) => (state.llm || state.preset !== "minimal" ? { ...state, preset: "minimal", llm: false } : state));
  }, [caps]);
  const options = useMemo(() => resolveOptions(composer, presets), [composer, presets]);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  /** Text for any failure, in the interface language; the service's wording stays as detail. */
  const describe = useCallback((error: unknown): { text: string; detail: string } => {
    const lang = localeRef.current;
    const words = dicts[lang];
    if (error instanceof NetworkError) return { text: words.submitNetworkFailed, detail: "" };
    if (error instanceof ApiError) {
      const known = apiErrorText(lang, error.body);
      if (known) return known;
      if (error.status === 413) return { text: words.submitTooLarge, detail: error.detail };
      if (error.status === 422) return { text: words.submitInvalid, detail: error.detail };
      if (error.status >= 500) return { text: words.submitServerFailed, detail: error.detail };
      return { text: words.submitHttpFailed(error.status), detail: error.detail };
    }
    if (error instanceof RangeError) return { text: words.diffTooLarge, detail: "" };
    return { text: error instanceof Error ? error.message : String(error), detail: "" };
  }, []);

  const jobs = useJobs(
    (done, failed, retained) => retained > 0 ? dicts[localeRef.current].notifyRetained(retained) : dicts[localeRef.current].notifyBody(done, failed),
    () => dicts[localeRef.current].connLost,
  );
  const archive = useArchive(jobs.jobs);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  // ---- notifications (one at a time; the newest wins)
  const [notification, setNotification] = useState<NotificationState>({ sequence: 0, note: null });
  const note = notification.note;
  const noticeOpener = useRef<HTMLElement | null>(null);
  const setNote = useCallback((next: NotificationModel | null, opener?: HTMLElement) => {
    if (next === null) noticeOpener.current = null;
    else {
      const active = opener ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
      // An automatic replacement keeps the original trigger while the user reads the card.
      if (active && active !== document.body && !active.closest(".notice-card")) noticeOpener.current = active;
    }
    setNotification((previous) => publishNotice(previous, next));
  }, []);
  const closeNote = useCallback(() => {
    const restore = document.activeElement instanceof HTMLElement && document.activeElement.closest(".notice-card") ? noticeOpener.current : null;
    setNote(null);
    if (restore?.isConnected) restore.focus({ preventScroll: true });
  }, [setNote]);
  // Request refusal belongs to the UI action, not the successful retained artifact.
  const [requestFailures, setRequestFailures] = useState<Record<string, ItemRequestFailure>>({});
  const requestItem = useCallback(async (item: SessionItem, operation: ItemRequestFailure["operation"], request: () => Promise<unknown>) => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : undefined;
    setRequestFailures((previous) => {
      if (!(item.key in previous)) return previous;
      const next = { ...previous };
      delete next[item.key];
      return next;
    });
    let error: unknown;
    try { error = await request(); } catch (reason) { error = reason; }
    if (error !== null && error !== undefined) {
      const current = jobsRef.current.items.find((row) => row.key === item.key) ?? item;
      setRequestFailures((previous) => ({ ...previous, [item.key]: { operation, error, identity: settledIdentity(current) } }));
      const model = itemNotification(current, dicts[localeRef.current], localeRef.current, { operation, ...describe(error) });
      if (model) setNote(model, opener);
    }
    return error;
  }, [describe, setNote]);

  const cloudflare = useCloudflareConsent(caps?.remote_services?.cloudflare);
  const authorizeCloudflare = cloudflare.authorize;

  const retryItem = useCallback(
    async (item: SessionItem, override?: JobOptions) => {
      const jobOptions = item.options ?? jobsRef.current.jobs[item.jobId]?.options ?? optionsRef.current;
      const base = publicOptions(override ?? jobOptions);
      const next = item.skipped && item.skipReason === "image_only" ? withOcrFor(base) : base;
      const authorized = await authorizeCloudflare(next, [{ name: item.name, kind: item.kind }], true);
      if (authorized === null) return undefined;
      return requestItem(item, "retry", () => jobsRef.current.retry(item, authorized));
    },
    [requestItem, authorizeCloudflare],
  );
  const showItemNotice = useCallback((item: SessionItem, opener?: HTMLElement) => {
    const failure = requestFailures[item.key];
    const actionProblem = failure && failure.identity === settledIdentity(item) ? { operation: failure.operation, ...describe(failure.error) } : undefined;
    const model = itemNotification(item, dicts[localeRef.current], localeRef.current, actionProblem);
    if (!model) return;
    const words = dicts[localeRef.current];
    if (!actionProblem && item.skipped && item.skipReason === "image_only") {
      model.action = {
        label: words.enableOcr,
        run: () => {
          setNote(null);
          setComposer((state) => ({ ...state, ocr: true }));
          void retryItem(item).then((error) => announce(error ? `${dicts[localeRef.current].retryFailed}: ${describe(error).text}` : dicts[localeRef.current].retryAria(item.name)));
        },
      };
    } else if (!actionProblem && item.status === "error" && item.errorCode === "no_model_configured" && canRetry(item)) {
      model.action = {
        label: words.retryPlain,
        run: () => {
          setNote(null);
          const base = publicOptions(item.options ?? jobsRef.current.jobs[item.jobId]?.options ?? optionsRef.current);
          void retryItem(item, { ...base, llm: false, alt: null, desc: null }).then((error) => announce(error ? `${dicts[localeRef.current].retryFailed}: ${describe(error).text}` : dicts[localeRef.current].retryAria(item.name)));
        },
      };
    }
    setNote(model, opener);
  }, [requestFailures, describe, setNote, retryItem, announce]);

  // Announce/notify only a new terminal identity already observed by this tab.
  // Locale changes, reconnect duplicates and first adopted history are quiet.
  const settledBefore = useRef(new Map<string, string | null>());
  useEffect(() => {
    const { next, changed } = terminalNotices(settledBefore.current, jobs.items);
    settledBefore.current = next;
    const total = jobs.items.length;
    const settled = jobs.items.filter(isSettled).length;
    for (const item of changed) {
      const word = item.rerunFailure ? t.rerunRetained(item.rerunFailure.operation) : item.status === "error" ? t.statusFailed : item.skipped ? t.statusSkipped : t.statusDone;
      announce(t.announceItem(item.name, word, settled, total));
      showItemNotice(item);
    }
  }, [jobs.items, t, announce, showItemNotice]);

  // ---- connectivity
  const offline = useConnectivity(() => {
    refreshCaps();
    void archive.refresh();
    void jobsRef.current.retryRestore();
    setNote({ tone: "success", title: dicts[localeRef.current].onlineTitle, message: dicts[localeRef.current].reconnected });
  });

  useEffect(() => {
    if (offline) setNote({ tone: "error", title: dicts[localeRef.current].offlineTitle, message: dicts[localeRef.current].submitNetworkFailed });
  }, [offline]);

  // ---- views
  const [view, setView] = useState<View>(viewOf);
  const navigate = useCallback((next: View) => {
    const path = next === "workspace" ? WORKSPACE : "/";
    if (location.pathname !== path) history.pushState(null, "", path);
    setView(next);
  }, []);
  useEffect(() => {
    const onPop = () => setView(viewOf());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const gear = useRef<HTMLButtonElement>(null);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    gear.current?.focus();
  }, []);
  const [tokenOpen, setTokenOpen] = useState(false);

  const [focusKey, setFocusKey] = useState<string | null>(null);
  const focusHandled = useCallback(() => setFocusKey(null), []);
  const openWorkspace = useCallback(() => {
    const already = view === "workspace";
    navigate("workspace");
    void archive.refresh();
    requestAnimationFrame(() => {
      const head = document.querySelector<HTMLElement>(".workspace .work-head");
      head?.scrollIntoView?.({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
      (document.querySelector<HTMLElement>('.ledger [role="listbox"] [role="option"]') ?? document.querySelector<HTMLElement>(".workspace .url-box"))?.focus();
      if (already) announce(t.historyCurrent);
    });
  }, [view, navigate, archive.refresh, announce, t.historyCurrent]);
  const goHome = useCallback(() => navigate("home"), [navigate]);

  // ---- submissions
  const [urlText, setUrlText] = useState("");
  const urls = useMemo(() => parseUrls(urlText).urls, [urlText]);
  const [dropNotice, setDropNotice] = useState<string | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
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
    async (files: File[], list: string[]) => {
      if (sending.current) return false;
      sending.current = true;
      try {
        const authorized = await authorizeCloudflare(optionsRef.current, [
          ...files.map((file) => ({ name: file.name, kind: "file" as const })),
          ...list.map((name) => ({ name, kind: "url" as const })),
        ]);
        if (authorized === null) return false;
        const controller = new AbortController();
        controllers.current.add(controller);
        setSubmitting(true);
        setUpload(files.length ? { loaded: 0, total: 0 } : null);
        let last = 0;
        const ok = await jobsRef.current.submit(files, list, authorized, {
          signal: controller.signal,
          onProgress: (loaded, total) => {
            // Progress fires very often; redraw ten times a second, never dropping the end.
            const now = Date.now();
            if (now - last < 100 && !(total > 0 && loaded >= total)) return;
            last = now;
            setUpload({ loaded, total });
          },
        });
        controllers.current.delete(controller);
        setSubmitting(controllers.current.size > 0);
        if (!controllers.current.size) setUpload(null);
        if (ok) navigate("workspace");
        return ok;
      } finally { sending.current = false; }
    },
    [navigate, authorizeCloudflare],
  );

  // The model went away since the page last asked: show the real state.
  useEffect(() => {
    if (jobs.submitError instanceof ApiError && jobs.submitError.reason === "llm_unavailable") refreshCaps();
  }, [jobs.submitError, refreshCaps]);

  const submitFiles = useCallback(
    (files: File[], folder?: { hidden: number; unreadable: number; truncated: boolean }) => {
      setInputError(null);
      jobsRef.current.clearSubmitError();
      const notes: string[] = [];
      let chosen = files;
      if (folder && files.length === 0) {
        setDropNotice(t.dropEmptyFolder);
        return;
      }
      if (chosen.length > maxItems) {
        notes.push(t.dropTruncated(maxItems, chosen.length));
        chosen = chosen.slice(0, maxItems);
      } else if (folder?.truncated) notes.push(t.dropLimit(chosen.length));
      if (folder?.hidden) notes.push(t.folderHidden(folder.hidden));
      if (folder?.unreadable) notes.push(t.folderUnreadable(folder.unreadable));
      setDropNotice(notes.length ? notes.join(" · ") : null);
      const large = oversized(chosen);
      if (large.length) {
        setInputError(t.filesTooLarge(large.map((file) => `${file.name} (${fmtBytes(file.size)})`).join(", ")));
        return;
      }
      void send(chosen, []);
    },
    [maxItems, send, t],
  );
  const submitFolder = useCallback((files: File[], hidden: number) => submitFiles(files, { hidden, unreadable: 0, truncated: false }), [submitFiles]);
  const submitWalked = useCallback((walked: Walked) => submitFiles(walked.files, walked), [submitFiles]);

  const submitUrls = useCallback(
    async (text: string) => {
      setInputError(null);
      jobsRef.current.clearSubmitError();
      const parsed = parseUrls(text);
      if (parsed.invalid) {
        setInputError(parsed.invalid.reason === "badScheme" ? t.badScheme(parsed.invalid.value) : t.badUrl(parsed.invalid.value));
        return false;
      }
      let list = parsed.urls;
      setDropNotice(null);
      if (list.length > maxItems) {
        setDropNotice(t.dropTruncated(maxItems, list.length));
        list = list.slice(0, maxItems);
      }
      return list.length ? send([], list) : false;
    },
    [maxItems, send, t],
  );

  // ---- ledger actions
  const enhanceItem = useCallback(
    async (item: SessionItem) => {
      const authorized = await authorizeCloudflare({ ...publicOptions(item.options ?? jobsRef.current.jobs[item.jobId]?.options ?? optionsRef.current), llm: true }, [{ name: item.name, kind: item.kind }], true);
      if (authorized === null) return undefined;
      return requestItem(item, "enhance", () => jobsRef.current.enhance(item, authorized));
    },
    [requestItem, authorizeCloudflare],
  );
  const removeItem = useCallback(
    async (item: SessionItem) => {
      const error = await requestItem(item, "delete", () => jobsRef.current.remove(item));
      if (error === null) void archive.refresh();
      return error;
    },
    [archive.refresh, requestItem],
  );
  const [retryingAll, setRetryingAll] = useState(false);
  const failed = useMemo(() => failedToRetry(jobs.items), [jobs.items]);
  const retryingAllRef = useRef(false);
  const retryAll = async () => {
    if (retryingAllRef.current) return;
    // Freeze identities and options before opening the shared confirmation.
    const batch = prepareRetryBatch(jobsRef.current.items, jobsRef.current.jobs, optionsRef.current);
    if (!batch.length) return;
    retryingAllRef.current = true;
    setRetryingAll(true);
    try {
      const authorized = await cloudflare.authorizeBatch(batch.map(({ item, options }) => ({ options, sources: [{ name: item.name, kind: item.kind }] })), true);
      const result = await submitRetryBatch(batch, authorized, (item, options) => requestItem(item, "retry", () => jobsRef.current.retry(item, options)));
      if (result) announce(result.failed ? t.announceRetryAllFailed(result.attempted, result.failed) : t.announceRetryAll(result.attempted));
    } finally {
      retryingAllRef.current = false;
      setRetryingAll(false);
    }
  };
  const waiting = useMemo(() => waitingJobs(jobs.items, jobs.jobs), [jobs.items, jobs.jobs]);
  const [stopping, setStopping] = useState(false);
  const stopRemaining = async () => {
    if (stopping) return;
    setStopping(true);
    const result = await jobs.stop(waiting);
    setStopping(false);
    if (result.error) {
      const text = describe(result.error);
      setNote({ tone: "error", title: t.stopRemaining, message: text.text, detail: text.detail });
    } else if (result.stopping > 0) announce(t.stopRequested(result.stopping));
  };

  // ---- preview
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ target: PreviewTarget; warnings: string[]; kind: "file" | "url"; createdAt: string | null } | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const returnKey = useRef<string | null>(null);
  const toTarget = (item: SessionItem): PreviewTarget => ({
    key: item.key,
    jobId: item.jobId,
    itemId: item.itemId,
    name: item.name,
    output: item.output,
    finishedAt: item.finishedAt,
    llmEnhanced: item.llmEnhanced,
    operation: item.operation,
  });
  const openPreview = useCallback((key: string, from: HTMLElement) => {
    const item = jobsRef.current.items.find((candidate) => candidate.key === key);
    if (!item || !isPreviewable(item)) return;
    setSelectedKey(key);
    opener.current = from;
    returnKey.current = key;
    setPreview({ target: toTarget(item), warnings: item.warnings, kind: item.kind, createdAt: jobsRef.current.jobs[item.jobId]?.createdAt ?? null });
  }, []);
  const closePreview = useCallback(() => {
    setPreview(null);
    const from = opener.current;
    const key = returnKey.current;
    opener.current = null;
    returnKey.current = null;
    requestAnimationFrame(() => {
      const fallback = key === null ? null : document.getElementById(domKey(key));
      (from?.isConnected ? from : fallback)?.focus();
    });
  }, []);

  // ---- saved jobs
  const openArchived = useCallback(
    async (jobId: string, from: HTMLElement) => {
      const snapshot = await archive.open(jobId);
      if (snapshot === null) return;
      // A saved job of several items joins the session ledger, so each of its
      // rows can be previewed, retried or deleted (an extension of the reference,
      // which previews only the first result).
      if (snapshot.items.length > 1 || snapshot.items.some((item) => item.status === "error" || item.rerun_failure != null || (item.warnings?.length ?? 0) > 0)) {
        jobsRef.current.adopt(snapshot);
        const first = snapshot.items.find((candidate) => candidate.status === "error" || candidate.rerun_failure != null || (candidate.warnings?.length ?? 0) > 0) ?? snapshot.items.find((candidate) => candidate.status === "done" && candidate.output !== null && !candidate.skipped) ?? snapshot.items[0];
        if (first) {
          setFocusKey(`${snapshot.job_id}/${first.item_id}`);
          showItemNotice(itemFromPayload(snapshot.job_id, first, Date.now()), from);
        }
        announce(t.adoptedJob(snapshot.items.length));
        return;
      }
      const item = snapshot.items.find((candidate) => candidate.status === "done" && candidate.output !== null && !candidate.skipped);
      if (!item) {
        announce(t.nothingToPreview);
        setNote({ tone: "warning", title: snapshot.items[0]?.name ?? snapshot.job_id, message: t.nothingToPreview });
        return;
      }
      opener.current = from;
      returnKey.current = null;
      setPreview({
        target: {
          key: `${snapshot.job_id}/${item.item_id}`,
          jobId: snapshot.job_id,
          itemId: item.item_id,
          name: item.name,
          output: item.output,
          finishedAt: item.finished_at,
          llmEnhanced: item.llm_enhanced,
          operation: item.operation,
        },
        warnings: item.warnings ?? [],
        kind: item.kind,
        createdAt: snapshot.created_at,
      });
    },
    [archive.open, announce, t, showItemNotice],
  );
  const retryArchived = useCallback(
    async (jobId: string) => {
      const snapshot = await archive.open(jobId);
      if (snapshot === null) return t.jobLoadFailed;
      const target = snapshot.items.find((item) => item.retryable && (item.status === "error" || (item.status === "done" && item.skipped)));
      if (!target) return t.noFailedItem;
      const base = publicOptions(target.options ?? snapshot.options);
      const authorized = await authorizeCloudflare(target.skip_reason === "image_only" ? withOcrFor(base) : base, [{ name: target.name, kind: target.kind }], true);
      if (authorized === null) return undefined;
      const error = await jobsRef.current.retryArchived(snapshot, target.item_id, authorized);
      if (error !== null) return error;
      setFocusKey(`${snapshot.job_id}/${target.item_id}`);
      announce(t.retryAria(target.name));
      return null;
    },
    [archive.open, announce, describe, t, authorizeCloudflare],
  );
  const enhanceArchived = useCallback(
    async (jobId: string) => {
      const snapshot = await archive.open(jobId);
      if (snapshot === null) return t.jobLoadFailed;
      const target = snapshot.items.find((item) => item.retryable && item.status === "done" && item.output !== null && !item.skipped && !item.llm_enhanced);
      if (!target) return t.noEnhanceableItem;
      const authorized = await authorizeCloudflare({ ...publicOptions(target.options ?? snapshot.options), llm: true }, [{ name: target.name, kind: target.kind }], true);
      if (authorized === null) return undefined;
      const error = await jobsRef.current.retryArchived(snapshot, target.item_id, authorized, "enhance");
      if (error !== null) return error;
      setFocusKey(`${snapshot.job_id}/${target.item_id}`);
      announce(t.enhanceWithLlm(target.name));
      return null;
    },
    [archive.open, announce, describe, t, authorizeCloudflare],
  );
  const deleteArchived = useCallback(
    async (entry: HistoryEntry) => {
      const removed = await archive.remove(entry.job_id);
      if (removed) announce(t.histDeleted(entry.names_preview[0] ?? entry.job_id));
      return removed;
    },
    [archive.remove, announce, t],
  );

  const clearAll = () => {
    if (jobs.running) {
      if (!jobs.terminalJobCount) return;
      jobs.clearSettled();
    } else {
      jobs.clear();
      navigate("home");
    }
    setSelectedKey(null);
    setPreview(null);
    setDropNotice(null);
    setInputError(null);
    setNote(null);
    void archive.refresh();
  };

  // A pointer elsewhere clears the ledger selection (Safari keeps focus on rows).
  useEffect(() => {
    if (view !== "workspace") return;
    const away = (event: PointerEvent) => {
      if (!(event.target instanceof Element) || event.target.closest('.ledger [role="option"], .dialog-preview')) return;
      setSelectedKey(null);
      const active = document.activeElement;
      if (active instanceof HTMLElement && active.closest('.ledger [role="option"]')) active.blur();
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [view]);

  // Moving between views moves focus to the new view's anchor.
  const previousView = useRef(view);
  useEffect(() => {
    if (previousView.current === view) return;
    previousView.current = view;
    const frame = requestAnimationFrame(() => {
      if (view === "workspace") {
        if (focusKey === null) document.querySelector<HTMLElement>('.ledger [role="option"][tabindex="0"]')?.focus();
      } else document.querySelector<HTMLElement>(".landing .url-box")?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [view, focusKey]);

  useEffect(() => {
    const base = view === "workspace" ? t.titleWorkspace : t.titleHome;
    document.title = view === "workspace" && jobs.activeCount > 0 ? `${jobs.activeCount} · ${base}` : base;
  }, [view, jobs.activeCount, t]);

  // ---- feedback under the composer, the same in both views
  const uploadText = (() => {
    if (!upload) return t.submitting;
    if (!(upload.total > 0)) return upload.loaded > 0 ? t.uploadUnknown(fmtBytes(upload.loaded)) : t.submitting;
    if (upload.loaded >= upload.total) return t.uploadDone;
    return t.uploadProgress(Math.min(99, Math.floor((upload.loaded / upload.total) * 100)), fmtBytes(upload.loaded), fmtBytes(upload.total));
  })();
  const submitError = jobs.submitError;
  const submitText = submitError && !(submitError instanceof ApiError && submitError.status === 401) ? describe(submitError) : null;
  const persistence = Object.values(jobs.jobs)
    .filter((job) => job.persistenceError)
    .map((job) => ({ id: job.jobId, ...persistenceText(locale, job.persistenceError ?? "") }));
  const feedback = (
    <>
      {submitting && (
        <NoticeLine busy>
          <span>{uploadText}</span>
          <InlineAction label={t.cancelSubmit} onClick={cancelSubmit} />
        </NoticeLine>
      )}
      {jobs.restoreFailed.size > 0 && (
        <NoticeLine>
          <span>{t.restoreFailed}</span>
          <InlineAction
            label={t.restoreRetry}
            onClick={() => void jobs.retryRestore().then((left) => left === 0 && announce(t.sessResults(jobs.items.length)))}
          />
        </NoticeLine>
      )}
      {(unauthorized || (submitError instanceof ApiError && submitError.status === 401)) && (
        <ErrorLine text={tokenRefusal(locale)}>
          {" · "}
          <button type="button" class="text-link" onClick={() => setTokenOpen(true)}>
            {t.enterToken}
          </button>
        </ErrorLine>
      )}
      {offline && !submitText && <ErrorLine text={t.submitNetworkFailed} />}
      {submitText && <ErrorLine text={submitText.text} detail={submitText.detail} />}
      {inputError && <ErrorLine text={inputError} />}
      {persistence.map((entry) => (
        <ErrorLine key={entry.id} text={entry.text} detail={entry.detail} />
      ))}
      {dropNotice && <NoticeLine>{dropNotice}</NoticeLine>}
      {caps !== null && !caps.llm.routable && <CapabilityHint t={t} onOpen={openSettings} />}
    </>
  );

  const composerFor = (compact: boolean, heading?: ComponentChildren, actions?: ComponentChildren) => (
    <OptionsPanel
      t={t}
      state={composer}
      presets={presets}
      llmReady={llmReady}
      cloudflare={caps?.remote_services?.cloudflare}
      urls={urls}
      announce={announce}
      busy={submitting || cloudflare.pending !== null}
      heading={heading}
      actions={actions}
      onChange={setComposer}
      onFiles={(files) => submitFiles(files)}
      onFolder={submitFolder}
      source={<UrlInput t={t} text={urlText} onText={setUrlText} onConvert={submitUrls} busy={submitting || cloudflare.pending !== null} compact={compact} />}
    />
  );

  const completedJobs = jobs.terminalJobCount + (archive.entries?.length ?? 0);
  const llmAvailable = llmReady && composer.llm;
  const llmDisabledReason = llmReady ? t.llmEnhanceTurnOn : t.llmEnhanceUnavailable;
  const downloadError = useCallback(
    (error: unknown) => {
      const text = describe(error);
      setNote({ tone: "error", title: dicts[localeRef.current].downloadFailed, message: text.text, detail: text.detail });
    },
    [describe],
  );

  return (
    <>
      <AppHeader
        t={t}
        version={caps?.version ?? null}
        locale={locale}
        onLocale={chooseLocale}
        onHome={goHome}
        onWorkspace={openWorkspace}
        workspaceActive={view === "workspace"}
        settingsOpen={settingsOpen}
        onSettings={() => (settingsOpen ? closeSettings() : openSettings())}
        gearRef={gear}
      />
      {cloudflare.pending && cloudflare.scope && <CloudflareDialog t={t} scope={cloudflare.scope} reason={cloudflare.reason} retry={cloudflare.pending.retry} requestCount={cloudflare.pending.requests.length} onClose={() => cloudflare.finish(false)} onConfirm={() => cloudflare.finish(true)} />}
      {note && <Notification replay={notification.sequence} note={note} closeLabel={t.close} detailsLabel={t.notificationDetails} warningsLabel={t.itemWarningsTitle} onClose={closeNote} />}
      {settingsOpen && <SettingsModal t={t} locale={locale} onClose={closeSettings} onSaved={refreshCaps} announce={announce} describe={describe} />}
      {preview && (
        <PreviewModal
          t={t}
          locale={locale}
          item={preview.target}
          warnings={preview.warnings}
          kind={preview.kind}
          createdAt={preview.createdAt}
          onClose={closePreview}
          announce={announce}
          describe={describe}
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
            jobs.clearSubmitError();
            refreshCaps();
            void archive.refresh();
            void jobs.retryRestore();
          }}
        />
      )}

      {view === "home" ? (
        <main class="landing page">
          <h1 class="landing-title">{t.heroTitle}</h1>
          <p class="landing-lede">{t.heroSub}</p>
          {jobs.items.length > 0 && (
            <button type="button" class="session-link" onClick={openWorkspace}>
              {jobs.activeCount > 0 ? t.sessProgress(jobs.activeCount) : t.sessResults(jobs.items.length)}
            </button>
          )}
          <div class="landing-composer">
            {composerFor(false)}
            {feedback}
          </div>
        </main>
      ) : (
        <main class="workspace page">
          <h1 class="sr-only">{t.titleWorkspace}</h1>
          <div class="work-stack">
            <div class="work-composer">
              {composerFor(
                true,
                <JobStats t={t} running={jobs.running} stats={jobs.stats} externalCharges={jobs.items.some((item) => hasCloudflareRequest(item.remoteProcessing))} />,
                jobs.items.length > 0 && (
                  <>
                    {waiting.length > 0 && (
                      <button type="button" class="btn btn-ghost" title={t.stopTitle} disabled={stopping} aria-busy={stopping || undefined} onClick={() => void stopRemaining()}>
                        {t.stopRemaining}
                      </button>
                    )}
                    {failed.length > 0 && (
                      <button type="button" class="btn btn-ghost" disabled={retryingAll} aria-busy={retryingAll || undefined} onClick={() => void retryAll()}>
                        {t.retryAllFailed(failed.length)}
                      </button>
                    )}
                    <ClearButton t={t} activeCount={jobs.activeCount} finishedJobs={jobs.terminalJobCount} onClear={clearAll} />
                  </>
                ),
              )}
              {feedback}
            </div>
            <Ledger
              t={t}
              locale={locale}
              items={jobs.items}
              jobs={jobs.jobs}
              archive={{
                entries: archive.entries,
                // While the service is unreachable the offline line already says so.
                error: archive.error && !(archive.error instanceof NetworkError) ? describe(archive.error).text : null,
                busy: archive.actions,
                rowErrors: archive.rowErrors,
                onRefresh: () => void archive.refresh(),
                onOpen: (jobId, from) => void openArchived(jobId, from),
                onRetry: retryArchived,
                onEnhance: enhanceArchived,
                onDelete: deleteArchived,
              }}
              stats={jobs.stats}
              settled={!jobs.running}
              selectedKey={selectedKey}
              focusKey={focusKey}
              llmAvailable={llmAvailable}
              llmDisabledReason={llmDisabledReason}
              onSelect={setSelectedKey}
              onFocusHandled={focusHandled}
              onPreview={openPreview}
              onRetry={retryItem}
              onEnhance={enhanceItem}
              onDelete={removeItem}
              describe={describe}
              onDownloadError={downloadError}
              requestFailures={requestFailures}
              onItemNotice={showItemNotice}
              onNotice={setNote}
            />
            <div class="zip-row">
              <ZipButton t={t} available={completedJobs > 0 && jobs.activeCount === 0} activeCount={jobs.activeCount} onError={downloadError} />
            </div>
          </div>
        </main>
      )}

      <AppFooter t={t} />
      <DropOverlay
        label={t.dropToConvert}
        suspended={settingsOpen || preview !== null || tokenOpen || cloudflare.pending !== null}
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

/** The refusal shown while this tab has no valid token. */
function tokenRefusal(locale: Locale): string {
  return apiErrorText(locale, { reason: "token_required" })?.text ?? "";
}

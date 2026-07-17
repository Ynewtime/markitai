import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCapabilities, historyArchiveUrl } from "./api/client";
import { MAX_JOB_ITEMS, type Capabilities, type JobOptions, type Preset } from "./api/types";
import type { SessionItem } from "./hooks/useJobs";
import { AppFooter, AppHeader } from "./components/AppHeader";
import { CapabilityHint } from "./components/CapabilityHint";
import { ClearJobsButton, DownloadArchiveButton } from "./components/DownloadActions";
import { DropOverlay } from "./components/DropZone";
import { ErrorInline } from "./components/ErrorInline";
import { ItemList } from "./components/ItemList";
import { JobStats } from "./components/JobStats";
import { LogoMark } from "./components/icons";
import { PreviewModal } from "./components/PreviewModal";
import { OptionsBar } from "./components/OptionsBar";
import { SettingsModal } from "./components/SettingsModal";
import { UrlInput } from "./components/UrlInput";
import {
  AppNotification,
  WarningNotification,
  type NotificationTone,
} from "./components/WarningNotification";
import { useArchivedJobs } from "./hooks/useArchivedJobs";
import { useJobs } from "./hooks/useJobs";
import { detectLocale, dicts, storeLocale, type Locale } from "./i18n";

/** Terminal = the item will not change again. */
function isSettled(i: SessionItem): boolean {
  return i.status === "done" || i.status === "error";
}

function settledWord(i: SessionItem): string {
  if (i.status === "error") return "failed";
  return i.skipped ? "skipped" : "done";
}

function optionDomId(key: string): string {
  return `opt-${key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

/** Explicit navigation state — settings and previews are modal overlays. */
type View = "home" | "workspace";

const WORKSPACE_PATH = "/jobs";
const OPTIONS_KEY = "markitai.options";
const OPTIONS_VERSION = 3;

function viewFromLocation(): View {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  return path === WORKSPACE_PATH || path.startsWith(`${WORKSPACE_PATH}/`)
    ? "workspace"
    : "home";
}

function readStoredOptions(): {
  preset: Preset | null;
  llm: boolean | null;
  ocr: boolean | null;
} {
  try {
    const raw = localStorage.getItem(OPTIONS_KEY);
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        const stored = parsed as {
          version?: unknown;
          preset?: unknown;
          llm?: unknown;
          ocr?: unknown;
        };
        const preset =
          stored.preset === "minimal" ||
          stored.preset === "standard" ||
          stored.preset === "rich"
            ? stored.preset
            : null;
        const llm = typeof stored.llm === "boolean" ? stored.llm : null;
        const ocr = typeof stored.ocr === "boolean" ? stored.ocr : false;

        // Before v2, standard + no-LLM was the implicit default. It leaves
        // image-analysis flags enabled and made URL jobs download every image
        // for no useful work. Migrate that old default to the CLI-like minimum.
        if (
          stored.version !== OPTIONS_VERSION &&
          llm === false &&
          preset === "standard"
        ) {
          return { preset: "minimal", llm: false, ocr };
        }
        return { preset, llm, ocr };
      }
    }
  } catch {
    /* localStorage unavailable / corrupt */
  }
  return { preset: null, llm: null, ocr: null };
}

export default function App() {
  const [locale, setLocale] = useState<Locale>(detectLocale);
  const t = dicts[locale];
  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  }, [locale]);
  const handleLocale = useCallback((l: Locale) => {
    setLocale(l);
    storeLocale(l);
  }, []);

  const [caps, setCaps] = useState<Capabilities | null>(null);
  const refreshCaps = useCallback(() => {
    fetchCapabilities().then(
      (c) => setCaps(c),
      () => undefined, // header just shows no version; POST errors surface inline
    );
  }, []);
  useEffect(() => {
    refreshCaps();
  }, [refreshCaps]);
  const llmConfigured = caps?.llm.routable === true;

  // Options remembered across visits (restored before caps arrive; the
  // downgrade effect below still applies when llm turns out unconfigured).
  const [preset, setPreset] = useState<Preset>(() => readStoredOptions().preset ?? "minimal");
  const [llm, setLlm] = useState<boolean>(() => readStoredOptions().llm ?? false); // opt-in, never the default
  const [ocr, setOcr] = useState<boolean>(() => readStoredOptions().ocr ?? false);
  useEffect(() => {
    try {
      localStorage.setItem(
        OPTIONS_KEY,
        JSON.stringify({ version: OPTIONS_VERSION, preset, llm, ocr }),
      );
    } catch {
      /* localStorage unavailable */
    }
  }, [preset, llm, ocr]);
  useEffect(() => {
    if (caps !== null && !caps.llm.routable) {
      setPreset("minimal");
      setLlm(false);
    }
  }, [caps]);

  // ---- sr-only polite live region: item settles, job completion, copied.
  const [liveMsg, setLiveMsg] = useState("");
  const [imageWarningKey, setImageWarningKey] = useState<string | null>(null);
  const [jobNotice, setJobNotice] = useState<{
    tone: NotificationTone;
    title: string;
    message: string;
  } | null>(null);
  const announce = useCallback((msg: string) => {
    // Alternate a trailing NBSP so repeating the same text re-announces.
    setLiveMsg((prev) => (prev === msg ? `${msg}\u00A0` : msg));
  }, []);

  const {
    items,
    jobs,
    stats,
    running,
    activeCount,
    now,
    submit,
    retry,
    retryArchived,
    enhance,
    enhanceArchived,
    deleteItem,
    submitError,
    clear,
    clearSettled,
    terminalJobCount,
    suppressedHistoryIds,
    historyRevision,
  } = useJobs();

  const archived = useArchivedJobs(jobs, suppressedHistoryIds);
  useEffect(() => {
    if (historyRevision > 0) void archived.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- archived is a fresh object each render; its stable refresh ref is the real input.
  }, [archived.refresh, historyRevision]);
  const deleteSessionItem = useCallback(
    async (item: SessionItem) => {
      const error = await deleteItem(item);
      if (error === null) void archived.refresh();
      return error;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- archived is a fresh object each render; its stable refresh ref is the real input.
    [archived.refresh, deleteItem],
  );

  const prevSettledRef = useRef<Map<string, boolean>>(new Map());
  const warnedImageRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const prev = prevSettledRef.current;
    const next = new Map<string, boolean>();
    let msg: string | null = null;
    let skippedImageKey: string | null = null;
    let settledCount = 0;
    for (const i of items) if (isSettled(i)) settledCount += 1;
    for (const i of items) {
      const settled = isSettled(i);
      next.set(i.key, settled);
      if (settled && prev.get(i.key) === false) {
        msg = t.announceItem(i.name, settledWord(i), settledCount, items.length);
      }
      if (i.skipped && i.skipReason === "image_only") {
        if (!warnedImageRef.current.has(i.key)) {
          warnedImageRef.current.add(i.key);
          skippedImageKey = i.key;
        }
      } else {
        warnedImageRef.current.delete(i.key);
      }
    }
    prevSettledRef.current = next;
    if (msg !== null) announce(msg);
    if (skippedImageKey !== null) setImageWarningKey(skippedImageKey);
  }, [items, t, announce]);

  // ---- settings modal (gear toggles; Esc/overlay close; focus returns to gear)
  const [settingsOpen, setSettingsOpen] = useState(false);
  const gearRef = useRef<HTMLButtonElement | null>(null);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    gearRef.current?.focus();
  }, []);

  // ---- URL-backed view state. /jobs is a real SPA route so refresh and
  // browser back/forward preserve the task-list view.
  const [view, setViewState] = useState<View>(viewFromLocation);
  const navigateView = useCallback((next: View) => {
    const path = next === "workspace" ? WORKSPACE_PATH : "/";
    if (window.location.pathname !== path) {
      window.history.pushState(null, "", path);
    }
    setViewState(next);
  }, []);
  useEffect(() => {
    const onPopState = () => setViewState(viewFromLocation());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  const effectiveView = view;

  const [focusItemKey, setFocusItemKey] = useState<string | null>(null);
  const handleFocusItem = useCallback(() => setFocusItemKey(null), []);
  const goHome = useCallback(() => {
    setFocusItemKey(null);
    navigateView("home");
  }, [navigateView]);
  const openTaskList = useCallback(() => {
    const alreadyOpen = effectiveView === "workspace";
    navigateView("workspace");
    void archived.refresh();
    window.requestAnimationFrame(() => {
      const anchor = document.querySelector<HTMLElement>(".workspace .jobhead");
      if (typeof anchor?.scrollIntoView === "function") {
        anchor.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth",
          block: "start",
        });
      }
      const firstRow = document.querySelector<HTMLElement>(
        '.flist [role="listbox"] [role="option"]',
      );
      (firstRow ?? document.querySelector<HTMLElement>(".workspace .urlin"))?.focus();
      if (alreadyOpen) announce(t.historyCurrent);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- archived is a fresh object each render; its stable refresh ref is the real input.
  }, [announce, archived.refresh, effectiveView, navigateView, t.historyCurrent]);
  // The URL draft lives here so the CLI-command line can mirror it live
  // (the home and composer inputs are never mounted at the same time).
  const [urlText, setUrlText] = useState("");
  const urlList = useMemo(
    () =>
      urlText
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    [urlText],
  );

  // Folder drops explain themselves on one neutral mono line (truncation
  // against the job limit, or nothing left after junk filtering).
  const [dropNotice, setDropNotice] = useState<string | null>(null);

  // Drops always use the options as currently set; any new conversion
  // brings the workspace forward.
  const optionsRef = useRef<JobOptions>({ preset, llm, ocr });
  useEffect(() => {
    optionsRef.current = { preset, llm, ocr };
  }, [preset, llm, ocr]);
  const submitFiles = useCallback(
    (files: File[], fromFolder = false) => {
      let notice: string | null = null;
      let send = files;
      if (fromFolder) {
        if (files.length === 0) notice = t.dropEmptyFolder;
        else if (files.length > MAX_JOB_ITEMS) {
          send = files.slice(0, MAX_JOB_ITEMS);
          notice = t.dropTruncated(MAX_JOB_ITEMS, files.length);
        }
      }
      setDropNotice(notice);
      if (send.length === 0) return;
      void submit(send, [], optionsRef.current).then((ok) => {
        if (ok) navigateView("workspace");
      });
    },
    [navigateView, submit, t],
  );
  const submitUrls = useCallback(
    async (urls: string[]) => {
      // Pasted URL batches obey the same job cap as folder drops, with the
      // same neutral truncation notice.
      let notice: string | null = null;
      let send = urls;
      if (urls.length > MAX_JOB_ITEMS) {
        send = urls.slice(0, MAX_JOB_ITEMS);
        notice = t.dropTruncated(MAX_JOB_ITEMS, urls.length);
      }
      setDropNotice(notice);
      const ok = await submit([], send, optionsRef.current);
      if (ok) navigateView("workspace");
      return ok;
    },
    [navigateView, submit, t],
  );
  // The OCR override applies to the retried request only; the persisted
  // toggle changes solely through the notification action that says so.
  const retryOptions = useCallback((skipReason: string | null): JobOptions => {
    const options = { ...optionsRef.current };
    if (skipReason === "image_only" && !options.llm && !options.ocr) {
      options.ocr = true;
    }
    return options;
  }, []);
  const retryItem = useCallback(
    (item: SessionItem) =>
      item.skipped ? retry(item, retryOptions(item.skipReason)) : retry(item),
    [retry, retryOptions],
  );
  // Enhancement failures surface exactly once, on the affected row: immediate
  // API rejections through the row's inline action error, and async SSE
  // failures through the row settling as failed. No app-level toast doubles them.
  const enhanceItem = useCallback(
    async (item: SessionItem) => {
      const sourceOptions = jobs[item.jobId]?.options ?? optionsRef.current;
      return enhance(item, { ...sourceOptions, llm: true });
    },
    [enhance, jobs],
  );
  const imageWarningItem =
    imageWarningKey === null
      ? null
      : (items.find((item) => item.key === imageWarningKey) ?? null);

  // Previewable = done with an output; skips complete as "done" but carry
  // no fresh result and stay non-selectable.
  const previewable = useCallback(
    (i: SessionItem) => i.status === "done" && i.output !== null && !i.skipped,
    [],
  );
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  // Selection is explicit. A direct /jobs reload starts neutral rather than
  // pinning Safari focus/selection to the first row.
  const selected =
    (selectedKey !== null ? items.find((i) => i.key === selectedKey) : undefined) ??
    null;

  const [previewOpen, setPreviewOpen] = useState(false);
  const [archivedPreview, setArchivedPreview] = useState<{
    item: SessionItem;
    createdAt: string | null;
  } | null>(null);
  const previewItem =
    archivedPreview?.item ?? (selected !== null && previewable(selected) ? selected : null);
  const previewOpenerRef = useRef<HTMLElement | null>(null);
  const previewReturnKeyRef = useRef<string | null>(null);
  const handleSelect = useCallback(
    (key: string | null) => setSelectedKey(key),
    [],
  );
  const openPreview = useCallback((key: string, opener: HTMLElement) => {
    setArchivedPreview(null);
    setSelectedKey(key);
    previewOpenerRef.current = opener;
    previewReturnKeyRef.current = key;
    setPreviewOpen(true);
  }, []);
  const closePreview = useCallback(() => {
    setPreviewOpen(false);
    setArchivedPreview(null);
    const opener = previewOpenerRef.current;
    const returnKey = previewReturnKeyRef.current;
    previewOpenerRef.current = null;
    previewReturnKeyRef.current = null;
    window.requestAnimationFrame(() => {
      const fallback =
        returnKey === null ? null : document.getElementById(optionDomId(returnKey));
      (opener?.isConnected ? opener : fallback)?.focus();
    });
  }, []);

  const handleClear = useCallback(() => {
    if (running) {
      if (terminalJobCount === 0) return;
      clearSettled();
    } else {
      clear();
      navigateView("home");
    }
    setSelectedKey(null);
    setPreviewOpen(false);
    previewOpenerRef.current = null;
    previewReturnKeyRef.current = null;
    setFocusItemKey(null);
    setDropNotice(null);
    setImageWarningKey(null);
    void archived.refresh();
  }, [archived, clear, clearSettled, navigateView, running, terminalJobCount]);

  // Persisted rows open in place. They never enter the current-session store,
  // so previewing one cannot move it between list regions or rewrite sessionStorage.
  const openArchivedJob = useCallback(
    async (jobId: string, opener: HTMLElement) => {
      const snapshot = await archived.openJob(jobId);
      if (snapshot === null) return null;
      const item = snapshot.items.find(
        (candidate) =>
          candidate.status === "done" &&
          candidate.output !== null &&
          !candidate.skipped,
      );
      if (item === undefined) {
        // All-failed/all-skipped jobs carry no output; say so instead of
        // letting the click silently do nothing.
        announce(t.nothingToPreview);
        setJobNotice({
          tone: "warning",
          title: snapshot.items[0]?.name ?? snapshot.job_id,
          message: t.nothingToPreview,
        });
        return snapshot;
      }
      setArchivedPreview({
        createdAt: snapshot.created_at,
        item: {
          key: `${snapshot.job_id}/${item.item_id}`,
          jobId: snapshot.job_id,
          itemId: item.item_id,
          name: item.name,
          kind: item.kind,
          status: item.status,
          error: item.error,
          output: item.output,
          durationMs: item.duration_ms,
          finishedAt: item.finished_at,
          costUsd: item.cost_usd,
          llmEnhanced: item.llm_enhanced,
          operation: item.operation,
          skipped: item.skipped,
          skipReason: item.skip_reason,
          sizeBytes: null,
          startedAt: null,
        },
      });
      previewOpenerRef.current = opener;
      previewReturnKeyRef.current = null;
      setPreviewOpen(true);
      return snapshot;
    },
    [announce, archived, t.nothingToPreview],
  );

  const retryArchivedJob = useCallback(
    async (jobId: string) => {
      const snapshot = await archived.openJob(jobId);
      if (snapshot === null) return t.jobLoadFailed;
      const retryable = snapshot.items.find(
        (item) => item.status === "error" || (item.status === "done" && item.skipped),
      );
      if (retryable === undefined) return t.noFailedItem;
      const error = retryable.skipped
        ? await retryArchived(
            snapshot,
            retryable.item_id,
            retryOptions(retryable.skip_reason),
          )
        : await retryArchived(snapshot, retryable.item_id);
      if (error === null) {
        setFocusItemKey(`${snapshot.job_id}/${retryable.item_id}`);
        announce(t.retryAria(retryable.name));
      }
      return error;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- archived is a fresh object each render; its stable openJob ref is the real input.
    [announce, archived.openJob, retryArchived, retryOptions, t],
  );

  const enhanceArchivedJob = useCallback(
    async (jobId: string) => {
      const snapshot = await archived.openJob(jobId);
      if (snapshot === null) return t.jobLoadFailed;
      const candidate = snapshot.items.find(
        (item) =>
          item.status === "done" &&
          item.output !== null &&
          !item.skipped &&
          !item.llm_enhanced,
      );
      if (candidate === undefined) return t.noEnhanceableItem;
      // The returned error renders inline on the archived row — its single
      // visible surface, matching enhanceItem above.
      const error = await enhanceArchived(snapshot, candidate.item_id, {
        ...snapshot.options,
        llm: true,
      });
      if (error === null) {
        setFocusItemKey(`${snapshot.job_id}/${candidate.item_id}`);
        announce(t.enhanceWithLlm(candidate.name));
      }
      return error;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- archived is a fresh object each render; its stable openJob ref is the real input.
    [
      announce,
      archived.openJob,
      enhanceArchived,
      t.enhanceWithLlm,
      t.jobLoadFailed,
      t.noEnhanceableItem,
    ],
  );

  // Keep the Base/LLM distinction visible even when every current result has
  // zero cost; hiding this column made unenhanced rows ambiguous.
  const showCost = true;
  const llmEnhanceAvailable = llmConfigured && llm;
  const llmDisabledReason = llmConfigured
    ? t.llmEnhanceTurnOn
    : t.llmEnhanceUnavailable;

  // Safari can retain focus on a generic tabindex row after a page click.
  // Clear ledger selection from a capture listener, while preserving the row
  // behind an open preview so closing it can restore focus to its opener.
  useEffect(() => {
    if (effectiveView !== "workspace") return;
    const clearLedgerFocus = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('.flist [role="option"], .preview-modal') !== null) return;
      setSelectedKey(null);
      const active = document.activeElement;
      if (
        active instanceof HTMLElement &&
        active.closest('.flist [role="option"]') !== null
      ) {
        active.blur();
      }
    };
    document.addEventListener("pointerdown", clearLedgerFocus, true);
    return () => document.removeEventListener("pointerdown", clearLedgerFocus, true);
  }, [effectiveView]);

  // Focus anchors on view transitions: the content under the reader is
  // replaced wholesale, so move focus onto the new view's anchor.
  const prevViewRef = useRef(effectiveView);
  useEffect(() => {
    if (prevViewRef.current === effectiveView) return;
    prevViewRef.current = effectiveView;
    const id = window.requestAnimationFrame(() => {
      if (effectiveView === "workspace") {
        if (focusItemKey === null) {
          document
            .querySelector<HTMLElement>('.work-list [role="option"][tabindex="0"]')
            ?.focus();
        }
      } else {
        document.querySelector<HTMLElement>(".drop-main .urlin")?.focus();
      }
    });
    return () => window.cancelAnimationFrame(id);
  }, [effectiveView, focusItemKey]);

  const archivedJobCount = archived.entries?.length ?? 0;
  const completedJobCount = terminalJobCount + archivedJobCount;

  const capHint =
    caps !== null && !caps.llm.routable ? (
      <CapabilityHint t={t} onOpenSettings={openSettings} />
    ) : null;

  // One element description feeds both archive-download slots (options row on
  // desktop, below the ledger on phones); CSS shows exactly one per
  // breakpoint, so gating and aria semantics can never drift apart.
  const archiveDownload = (
    <DownloadArchiveButton
      t={t}
      zipHref={
        completedJobCount > 0 && activeCount === 0 ? historyArchiveUrl() : null
      }
      activeCount={activeCount}
      onDownloadError={(message) =>
        setJobNotice({
          tone: "error",
          title: t.downloadFailed,
          message,
        })
      }
    />
  );

  return (
    <>
      <AppHeader
        t={t}
        version={caps?.version ?? null}
        locale={locale}
        onLocale={handleLocale}
        onHome={goHome}
        onHistory={openTaskList}
        historyActive={effectiveView === "workspace"}
        settingsOpen={settingsOpen}
        onToggleSettings={() => (settingsOpen ? closeSettings() : openSettings())}
        gearRef={gearRef}
      />
      {imageWarningItem !== null && (
        <WarningNotification
          title={t.imageSkippedTitle}
          message={t.imageSkipped(imageWarningItem.name)}
          actionLabel={t.enableOcr}
          closeLabel={t.close}
          onAction={() => {
            setImageWarningKey(null);
            // This action's label promises a persistent change, so it is the
            // one place a retry may flip the stored OCR toggle.
            setOcr(true);
            void retryItem(imageWarningItem).then((error) => {
              announce(
                error === null
                  ? t.retryAria(imageWarningItem.name)
                  : `${t.retryFailed}: ${error}`,
              );
            });
          }}
          onClose={() => setImageWarningKey(null)}
        />
      )}
      {jobNotice !== null && (
        <AppNotification
          tone={jobNotice.tone}
          title={jobNotice.title}
          message={jobNotice.message}
          closeLabel={t.close}
          onClose={() => setJobNotice(null)}
        />
      )}
      {settingsOpen && (
        <SettingsModal
          t={t}
          locale={locale}
          onLocale={handleLocale}
          onClose={closeSettings}
          onSaved={refreshCaps}
          announce={announce}
        />
      )}
      {previewOpen && previewItem !== null && (
        <PreviewModal
          t={t}
          item={previewItem}
          createdAt={
            archivedPreview?.createdAt ?? jobs[previewItem.jobId]?.createdAt ?? null
          }
          onClose={closePreview}
          announce={announce}
        />
      )}

      {effectiveView === "home" && (
        <main className="drop-main shell">
          <LogoMark size={56} className="logo-lg" />
          <h1 className="hero">{t.heroTitle}</h1>
          <p className="hero-sub">{t.heroSub}</p>
          {items.length > 0 && (
            <button type="button" className="sesslink mono" onClick={openTaskList}>
              {activeCount > 0 ? t.sessProgress(activeCount) : t.sessResults(items.length)}
            </button>
          )}
          <div className="convert-source">
            <UrlInput
              t={t}
              text={urlText}
              onText={setUrlText}
              onConvert={submitUrls}
              onFiles={submitFiles}
            />
          </div>
          {submitError !== null && (
            <ErrorInline text={`${t.createJobFailed}: ${submitError}`} />
          )}
          {dropNotice !== null && (
            <p className="notice" role="status">
              {dropNotice}
            </p>
          )}
          <OptionsBar
            t={t}
            preset={preset}
            llm={llm}
            ocr={ocr}
            llmConfigured={llmConfigured}
            urls={urlList}
            announce={announce}
            onPreset={setPreset}
            onLlm={setLlm}
            onOcr={setOcr}
          />
          {capHint}
        </main>
      )}

      {effectiveView === "workspace" && (
        <main className="shell workspace">
          <div className="jobhead">
            <JobStats t={t} running={running} stats={stats} />
            {items.length > 0 && (
              <div className="jobhead-r">
                <ClearJobsButton
                  t={t}
                  activeCount={activeCount}
                  clearableJobCount={terminalJobCount}
                  onClear={handleClear}
                />
              </div>
            )}
          </div>

          <div className="conversion-workspace">
            <div className="work-grid">
              <div className="work-list">
                <div className="composer">
                  <div className="convert-source">
                    <UrlInput
                      t={t}
                      text={urlText}
                      onText={setUrlText}
                      onConvert={submitUrls}
                      onFiles={submitFiles}
                      compact
                    />
                  </div>
                  {submitError !== null && (
                    <ErrorInline text={`${t.createJobFailed}: ${submitError}`} />
                  )}
                  {dropNotice !== null && (
                    <p className="notice" role="status">
                      {dropNotice}
                    </p>
                  )}
                  <OptionsBar
                    t={t}
                    preset={preset}
                    llm={llm}
                    ocr={ocr}
                    llmConfigured={llmConfigured}
                    urls={urlList}
                    announce={announce}
                    onPreset={setPreset}
                    onLlm={setLlm}
                    onOcr={setOcr}
                    trailing={archiveDownload}
                  />
                  {capHint}
                </div>
                <ItemList
                  t={t}
                  items={items}
                  jobs={jobs}
                  archive={{
                    entries: archived.entries,
                    error: archived.error,
                    rowProps: {
                      actions: archived.actions,
                      rowErrors: archived.rowErrors,
                      onRefresh: archived.refresh,
                      onOpen: openArchivedJob,
                      onRetry: retryArchivedJob,
                      onEnhance: enhanceArchivedJob,
                      onDelete: archived.deleteJob,
                      announce,
                    },
                  }}
                  showCost={showCost}
                  now={now}
                  stats={stats}
                  settled={!running}
                  selectedKey={selected?.key ?? null}
                  onSelect={handleSelect}
                  onPreview={openPreview}
                  focusKey={focusItemKey}
                  onFocusKeyHandled={handleFocusItem}
                  onRetry={retryItem}
                  onEnhance={enhanceItem}
                  onDelete={deleteSessionItem}
                  canDelete={(item) => jobs[item.jobId]?.status === "done"}
                  llmAvailable={llmEnhanceAvailable}
                  llmDisabledReason={llmDisabledReason}
                />
                {/* phone slot: the zip CTA reads better under the ledger it
                    archives than squeezed into the options row */}
                <div className="list-zip">{archiveDownload}</div>
              </div>
            </div>
          </div>
        </main>
      )}

      <AppFooter t={t} />
      <DropOverlay label={t.dropToConvert} onFiles={submitFiles} />
      <div className="sr-only" role="status" aria-live="polite">
        {liveMsg}
      </div>
    </>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCapabilities, jobArchiveUrl } from "./api/client";
import { MAX_JOB_ITEMS, type Capabilities, type JobOptions, type Preset } from "./api/types";
import type { SessionItem } from "./hooks/useJobs";
import { AppHeader } from "./components/AppHeader";
import { ArchivedJobRows } from "./components/ArchivedJobsSection";
import { CapabilityHint } from "./components/CapabilityHint";
import { DownloadActions } from "./components/DownloadActions";
import { DropOverlay, FilePicker } from "./components/DropZone";
import { ErrorInline } from "./components/ErrorInline";
import { ItemList } from "./components/ItemList";
import { JobStats } from "./components/JobStats";
import { LogoMark } from "./components/icons";
import { PreviewModal } from "./components/PreviewModal";
import { OptionsBar } from "./components/OptionsBar";
import { SettingsModal } from "./components/SettingsModal";
import { UrlInput } from "./components/UrlInput";
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

const OPTIONS_KEY = "markitai.options";

function readStoredOptions(): { preset: Preset | null; llm: boolean | null } {
  try {
    const raw = localStorage.getItem(OPTIONS_KEY);
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        const p = (parsed as { preset?: unknown }).preset;
        const l = (parsed as { llm?: unknown }).llm;
        return {
          preset: p === "minimal" || p === "standard" || p === "rich" ? p : null,
          llm: typeof l === "boolean" ? l : null,
        };
      }
    }
  } catch {
    /* localStorage unavailable / corrupt */
  }
  return { preset: null, llm: null };
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
  const llmConfigured = caps === null ? true : caps.llm.routable;

  // Options remembered across visits (restored before caps arrive; the
  // downgrade effect below still applies when llm turns out unconfigured).
  const [preset, setPreset] = useState<Preset>(() => readStoredOptions().preset ?? "standard");
  const [llm, setLlm] = useState<boolean>(() => readStoredOptions().llm ?? false); // opt-in, never the default
  useEffect(() => {
    try {
      localStorage.setItem(OPTIONS_KEY, JSON.stringify({ preset, llm }));
    } catch {
      /* localStorage unavailable */
    }
  }, [preset, llm]);
  useEffect(() => {
    if (caps !== null && !caps.llm.routable) {
      setPreset("minimal");
      setLlm(false);
    }
  }, [caps]);

  // ---- sr-only polite live region: item settles, job completion, copied.
  const [liveMsg, setLiveMsg] = useState("");
  const announce = useCallback((msg: string) => {
    // Alternate a trailing NBSP so repeating the same text re-announces.
    setLiveMsg((prev) => (prev === msg ? `${msg} ` : msg));
  }, []);

  const {
    items,
    jobs,
    jobCount,
    stats,
    running,
    activeCount,
    now,
    submit,
    retry,
    submitError,
    clear,
    clearSettled,
    terminalJobCount,
  } = useJobs();

  const archived = useArchivedJobs(jobs);

  const prevSettledRef = useRef<Map<string, boolean>>(new Map());
  useEffect(() => {
    const prev = prevSettledRef.current;
    const next = new Map<string, boolean>();
    let msg: string | null = null;
    let settledCount = 0;
    for (const i of items) if (isSettled(i)) settledCount += 1;
    for (const i of items) {
      const settled = isSettled(i);
      next.set(i.key, settled);
      if (settled && prev.get(i.key) === false) {
        msg = t.announceItem(i.name, settledWord(i), settledCount, items.length);
      }
    }
    prevSettledRef.current = next;
    if (msg !== null) announce(msg);
  }, [items, t, announce]);

  // ---- settings modal (gear toggles; Esc/overlay close; focus returns to gear)
  const [settingsOpen, setSettingsOpen] = useState(false);
  const gearRef = useRef<HTMLButtonElement | null>(null);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => {
    setSettingsOpen(false);
    gearRef.current?.focus();
  }, []);

  // ---- view state machine: a reload always starts at home. Restored jobs
  // remain reachable through the session link/history button without forcing
  // a navigation during hydration.
  const [view, setView] = useState<View>("home");
  const hasTaskRows = items.length > 0 || (archived.entries?.length ?? 0) > 0;
  const effectiveView: View = view === "workspace" && !hasTaskRows ? "home" : view;

  const [focusItemKey, setFocusItemKey] = useState<string | null>(null);
  const handleFocusItem = useCallback(() => setFocusItemKey(null), []);
  const goHome = useCallback(() => {
    setFocusItemKey(null);
    setView("home");
  }, []);
  const openTaskList = useCallback(() => {
    if (!hasTaskRows) return;
    setView("workspace");
    window.requestAnimationFrame(() => {
      document.querySelector<HTMLElement>('.flist [role="listbox"] [role="option"]')?.focus();
    });
  }, [hasTaskRows]);
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
  const optionsRef = useRef<JobOptions>({ preset, llm });
  useEffect(() => {
    optionsRef.current = { preset, llm };
  }, [preset, llm]);
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
        if (ok) setView("workspace");
      });
    },
    [submit, t],
  );
  const submitUrls = useCallback(
    async (urls: string[]) => {
      const ok = await submit([], urls, optionsRef.current);
      if (ok) setView("workspace");
      return ok;
    },
    [submit],
  );

  // Previewable = done with an output; skips complete as "done" but carry
  // no fresh result and stay non-selectable.
  const previewable = useCallback(
    (i: SessionItem) => i.status === "done" && i.output !== null && !i.skipped,
    [],
  );
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const selected =
    (selectedKey !== null
      ? items.find((i) => i.key === selectedKey && previewable(i))
      : undefined) ??
    items.find(previewable) ??
    null;
  useEffect(() => {
    const key = selected?.key ?? null;
    if (key !== selectedKey) setSelectedKey(key);
  }, [selected, selectedKey]);

  const [previewOpen, setPreviewOpen] = useState(false);
  const [archivedPreview, setArchivedPreview] = useState<{
    item: SessionItem;
    createdAt: string | null;
  } | null>(null);
  const previewItem = archivedPreview?.item ?? selected;
  const previewOpenerRef = useRef<HTMLElement | null>(null);
  const previewReturnKeyRef = useRef<string | null>(null);
  const handleSelect = useCallback((key: string) => setSelectedKey(key), []);
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
      setView("home");
    }
    setSelectedKey(null);
    setPreviewOpen(false);
    previewOpenerRef.current = null;
    previewReturnKeyRef.current = null;
    setFocusItemKey(null);
    setDropNotice(null);
    void archived.refresh();
  }, [archived, clear, clearSettled, running, terminalJobCount]);

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
      if (item === undefined) return snapshot;
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
          costUsd: item.cost_usd,
          skipped: item.skipped,
          skipReason: item.skip_reason,
          sizeBytes: null,
          startedAt: null,
          archived: true,
          retried: false,
        },
      });
      previewOpenerRef.current = opener;
      previewReturnKeyRef.current = null;
      setPreviewOpen(true);
      return snapshot;
    },
    [archived],
  );

  const showCost = stats.hasCost;

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

  const selectedJobRunning =
    selected !== null && jobs[selected.jobId]?.status === "running";

  const capHint =
    caps !== null && !caps.llm.routable ? (
      <CapabilityHint t={t} onOpenSettings={openSettings} />
    ) : null;

  return (
    <>
      <AppHeader
        t={t}
        version={caps?.version ?? null}
        locale={locale}
        onLocale={handleLocale}
        onHome={goHome}
        onHistory={openTaskList}
        settingsOpen={settingsOpen}
        onToggleSettings={() => (settingsOpen ? closeSettings() : openSettings())}
        gearRef={gearRef}
      />
      {settingsOpen && (
        <SettingsModal t={t} onClose={closeSettings} onSaved={refreshCaps} announce={announce} />
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
            <button type="button" className="sesslink mono" onClick={() => setView("workspace")}>
              {activeCount > 0 ? t.sessProgress(activeCount) : t.sessResults(items.length)}
            </button>
          )}
          <UrlInput t={t} text={urlText} onText={setUrlText} onConvert={submitUrls} />
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
            llmConfigured={llmConfigured}
            urls={urlList}
            announce={announce}
            onPreset={setPreset}
            onLlm={setLlm}
          />
          {capHint}
          <p className="drop-hint">
            {t.dropAnywhere} · <FilePicker label={t.browse} onFiles={submitFiles} />
          </p>
        </main>
      )}

      {effectiveView === "workspace" && (
        <main className="shell workspace">
          <div className="jobhead">
            <JobStats t={t} running={running} stats={stats} />
            {items.length > 0 && (
              <div className="jobhead-r">
                <DownloadActions
                  t={t}
                  multiJob={jobCount > 1}
                  zipHref={selected !== null ? jobArchiveUrl(selected.jobId) : null}
                  jobRunning={selectedJobRunning}
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
                  <UrlInput t={t} text={urlText} onText={setUrlText} onConvert={submitUrls} compact />
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
                    llmConfigured={llmConfigured}
                    urls={urlList}
                    announce={announce}
                    onPreset={setPreset}
                    onLlm={setLlm}
                  />
                  {capHint}
                  <p className="drop-hint">
                    {t.dropMore} · <FilePicker label={t.browse} onFiles={submitFiles} />
                  </p>
                </div>
                <ItemList
                  t={t}
                  items={items}
                  showCost={showCost}
                  now={now}
                  stats={stats}
                  settled={!running}
                  llmConfigured={llmConfigured}
                  selectedKey={selected?.key ?? null}
                  onSelect={handleSelect}
                  onPreview={openPreview}
                  focusKey={focusItemKey}
                  onFocusKeyHandled={handleFocusItem}
                  onOpenSettings={openSettings}
                  onRetry={retry}
                  hasArchivedRows={(archived.entries?.length ?? 0) > 0 || archived.error !== null}
                  archivedRows={
                    <ArchivedJobRows
                      t={t}
                      entries={archived.entries}
                      error={archived.error}
                      actions={archived.actions}
                      rowErrors={archived.rowErrors}
                      showCost={showCost}
                      startIndex={items.length}
                      onRefresh={archived.refresh}
                      onOpen={openArchivedJob}
                      onDelete={archived.deleteJob}
                      announce={announce}
                    />
                  }
                />
              </div>
            </div>
          </div>
        </main>
      )}

      <DropOverlay label={t.dropToConvert} onFiles={submitFiles} />
      <div className="sr-only" role="status" aria-live="polite">
        {liveMsg}
      </div>
    </>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchCapabilities, fetchJobSnapshot, jobArchiveUrl } from "./api/client";
import type { Capabilities, JobOptions, Preset } from "./api/types";
import type { SessionItem } from "./hooks/useJobs";
import { AppHeader } from "./components/AppHeader";
import { CapabilityHint } from "./components/CapabilityHint";
import { DownloadActions } from "./components/DownloadActions";
import { DropOverlay, FilePicker } from "./components/DropZone";
import { ErrorInline } from "./components/ErrorInline";
import { HistoryView } from "./components/HistoryView";
import { ItemList } from "./components/ItemList";
import { JobStats } from "./components/JobStats";
import { EyeIcon, LogoMark } from "./components/icons";
import { MarkdownPreview } from "./components/MarkdownPreview";
import { OptionsBar } from "./components/OptionsBar";
import { SettingsModal } from "./components/SettingsModal";
import { UrlInput } from "./components/UrlInput";
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

/** Explicit navigation state — settings is a modal and owns no view. */
type View = "home" | "workspace" | "history";
type PanelMode = "normal" | "expanded" | "hidden";

const PANEL_KEY = "markitai.panelMode";
const OPTIONS_KEY = "markitai.options";

function readPanelMode(): PanelMode {
  try {
    const v = localStorage.getItem(PANEL_KEY);
    if (v === "normal" || v === "expanded" || v === "hidden") return v;
  } catch {
    /* localStorage unavailable */
  }
  return "normal";
}

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
  const llmConfigured = caps === null ? true : caps.llm.configured;

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
    if (caps !== null && !caps.llm.configured) {
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
    submitError,
    clear,
    openArchived,
    removeJob,
  } = useJobs();

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

  // ---- view state machine: home | workspace | history. Switches are pure
  // UI state — jobs, event streams, selection and panel mode all survive.
  const [view, setView] = useState<View>("home");
  // A workspace with nothing in it renders as home (e.g. session restore
  // dropped every job) — the state machine never shows an empty ledger.
  const effectiveView: View = view === "workspace" && items.length === 0 ? "home" : view;

  // Session restore lands mid-conversion: jump from home into the workspace
  // exactly when the ledger goes from empty to populated.
  const prevCountRef = useRef(0);
  useEffect(() => {
    const prev = prevCountRef.current;
    prevCountRef.current = items.length;
    if (prev === 0 && items.length > 0 && view === "home") setView("workspace");
  }, [items.length, view]);

  const goHome = useCallback(() => setView("home"), []);
  const toggleHistory = useCallback(() => {
    setView((v) => (v === "history" ? (prevCountRef.current > 0 ? "workspace" : "home") : "history"));
  }, []);

  // Drops always use the options as currently set; any new conversion
  // brings the workspace forward.
  const optionsRef = useRef<JobOptions>({ preset, llm });
  useEffect(() => {
    optionsRef.current = { preset, llm };
  }, [preset, llm]);
  const submitFiles = useCallback(
    (files: File[]) => {
      void submit(files, [], optionsRef.current).then((ok) => {
        if (ok) setView("workspace");
      });
    },
    [submit],
  );
  const submitUrls = useCallback(
    async (urls: string[]) => {
      const ok = await submit([], urls, optionsRef.current);
      if (ok) setView("workspace");
      return ok;
    },
    [submit],
  );

  // ---- preview panel mode (normal | expanded | hidden), persisted.
  const [panelMode, setPanelModeRaw] = useState<PanelMode>(readPanelMode);
  const setPanelMode = useCallback((m: PanelMode) => {
    setPanelModeRaw(m);
    try {
      localStorage.setItem(PANEL_KEY, m);
    } catch {
      /* localStorage unavailable */
    }
  }, []);

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

  // Picking an item while the panel is hidden brings it back.
  const handleSelect = useCallback(
    (key: string) => {
      setSelectedKey(key);
      if (panelMode === "hidden") setPanelMode("normal");
    },
    [panelMode, setPanelMode],
  );

  // Hiding removes the focused toggle; showing removes the show button.
  // Hand focus to the counterpart after the commit (effect, not rAF — the
  // DOM must already hold the target).
  const panelFocusRef = useRef<"show" | "panel" | null>(null);
  const hidePreview = useCallback(() => {
    panelFocusRef.current = "show";
    setPanelMode("hidden");
  }, [setPanelMode]);
  const showPreview = useCallback(() => {
    panelFocusRef.current = "panel";
    setPanelMode("normal");
  }, [setPanelMode]);
  useEffect(() => {
    const target = panelFocusRef.current;
    if (target === null) return;
    panelFocusRef.current = null;
    if (target === "show") {
      document.querySelector<HTMLElement>(".showpv")?.focus();
    } else {
      const el =
        document.querySelector<HTMLElement>(".work-preview .tab.on") ??
        document.querySelector<HTMLElement>('.work-list [role="option"][tabindex="0"]');
      el?.focus();
    }
  }, [panelMode]);

  const handleClear = useCallback(() => {
    clear();
    setSelectedKey(null);
    setView("home");
  }, [clear]);

  // ---- history → session: merge the archived job, then show the workspace.
  const openHistoryJob = useCallback(
    async (jobId: string): Promise<string | null> => {
      if (jobs[jobId] === undefined) {
        try {
          const snap = await fetchJobSnapshot(jobId);
          if (snap === null) return t.histGone;
          openArchived(snap);
        } catch (e) {
          return e instanceof Error ? e.message : String(e);
        }
      }
      setView("workspace");
      return null;
    },
    [jobs, openArchived, t],
  );

  const showCost = stats.hasCost;
  const gridClass =
    panelMode === "hidden"
      ? "work-grid nopanel"
      : panelMode === "expanded" && selected !== null
        ? "work-grid expanded"
        : "work-grid";

  // Focus anchors on view transitions: the content under the reader is
  // replaced wholesale, so move focus onto the new view's anchor.
  const prevViewRef = useRef(effectiveView);
  useEffect(() => {
    if (prevViewRef.current === effectiveView) return;
    prevViewRef.current = effectiveView;
    const id = window.requestAnimationFrame(() => {
      if (effectiveView === "workspace") {
        document
          .querySelector<HTMLElement>('.work-list [role="option"][tabindex="0"]')
          ?.focus();
      } else if (effectiveView === "home") {
        document.querySelector<HTMLElement>(".drop-main .urlin")?.focus();
      } else {
        document.querySelector<HTMLElement>(".histmain .eyebrow")?.focus();
      }
    });
    return () => window.cancelAnimationFrame(id);
  }, [effectiveView]);

  const selectedJobRunning =
    selected !== null && jobs[selected.jobId]?.status === "running";

  const capHint =
    caps !== null && !caps.llm.configured ? (
      <CapabilityHint t={t} onOpenSettings={openSettings} />
    ) : null;

  const previewPlaceholder = running
    ? t.previewWaiting
    : stats.failed > 0
      ? t.previewAllFailed
      : t.previewEmptySelect;

  return (
    <>
      <AppHeader
        t={t}
        version={caps?.version ?? null}
        locale={locale}
        onLocale={handleLocale}
        historyActive={effectiveView === "history"}
        onHome={goHome}
        onHistory={toggleHistory}
        settingsOpen={settingsOpen}
        onToggleSettings={() => (settingsOpen ? closeSettings() : openSettings())}
        gearRef={gearRef}
      />
      {settingsOpen && (
        <SettingsModal t={t} onClose={closeSettings} onSaved={refreshCaps} announce={announce} />
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
          <UrlInput t={t} onConvert={submitUrls} />
          {submitError !== null && (
            <ErrorInline text={`${t.createJobFailed}: ${submitError}`} />
          )}
          <OptionsBar
            t={t}
            preset={preset}
            llm={llm}
            llmConfigured={llmConfigured}
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
            <div className="jobhead-r">
              {panelMode === "hidden" && (
                <button type="button" className="btn ghost showpv" onClick={showPreview}>
                  <EyeIcon size={14} />
                  {t.showPreview}
                </button>
              )}
              <DownloadActions
                t={t}
                multiJob={jobCount > 1}
                zipHref={selected !== null ? jobArchiveUrl(selected.jobId) : null}
                jobRunning={selectedJobRunning}
                activeCount={activeCount}
                onClear={handleClear}
              />
            </div>
          </div>

          <div className={gridClass}>
            <div className="work-list">
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
                onOpenSettings={openSettings}
              />
              {submitError !== null && (
                <ErrorInline text={`${t.createJobFailed}: ${submitError}`} />
              )}
              <div className="composer">
                <UrlInput t={t} onConvert={submitUrls} compact />
                <OptionsBar
                  t={t}
                  preset={preset}
                  llm={llm}
                  llmConfigured={llmConfigured}
                  onPreset={setPreset}
                  onLlm={setLlm}
                />
                {capHint}
                <p className="drop-hint">
                  {t.dropMore} · <FilePicker label={t.browse} onFiles={submitFiles} />
                </p>
              </div>
            </div>
            <div className="work-preview">
              {selected !== null ? (
                <MarkdownPreview
                  t={t}
                  item={selected}
                  createdAt={jobs[selected.jobId]?.createdAt ?? null}
                  expanded={panelMode === "expanded"}
                  onToggleExpand={() =>
                    setPanelMode(panelMode === "expanded" ? "normal" : "expanded")
                  }
                  onHide={hidePreview}
                  announce={announce}
                />
              ) : (
                <div className="panel panel-empty">
                  <p className="pane-note">{previewPlaceholder}</p>
                </div>
              )}
            </div>
          </div>
        </main>
      )}

      {effectiveView === "history" && (
        <main className="shell histmain">
          <div className="jobhead">
            <div>
              <h2 className="eyebrow" tabIndex={-1}>
                {t.histTitle}
              </h2>
            </div>
          </div>
          <HistoryView t={t} onOpen={openHistoryJob} onDeleted={removeJob} />
        </main>
      )}

      <DropOverlay label={t.dropToConvert} onFiles={submitFiles} />
      <div className="sr-only" role="status" aria-live="polite">
        {liveMsg}
      </div>
    </>
  );
}

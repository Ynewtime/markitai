// The peek window: a click on a conversion opens one window over the dimmed
// list, with Details, Preview and Compare as tabs, previous/next with its
// position, the one or two actions that matter and "⋯" for the rest. The
// expand button (or F) grows it to the whole viewport and back, remembered per
// browser; on a phone it always fills the screen. Up/Down move between items
// (outside the scrolling document), C switches Compare, Escape closes.
import type { ComponentChildren } from "preact";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import { useResultDocument } from "../hooks/use-result-document.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { headerActs } from "../lib/acts.ts";
import { readFull, storeFull, takePreviewView } from "../lib/peek-pref.ts";
import { copyText } from "../lib/clipboard.ts";
import { basename, readable } from "../lib/format.ts";
import { printDocument, type PrintJob } from "../lib/print.ts";
import { dismissToast, toast } from "../lib/toast.ts";
import type { SessionItem } from "../lib/session.ts";
import { ActionMenu, type MenuAction } from "./action-menu.tsx";
import { CompareView } from "./compare-view.tsx";
import { FileName } from "./file-name.tsx";
import { Icon } from "./icons.tsx";
import { ActionError, type ActionFailure, type ActSet } from "./inspector.tsx";
import { MarkdownView, type RenderedState, type Tab } from "./markdown-view.tsx";
import { PdfExportCard, readFurniture, storeFurniture } from "./pdf-settings.tsx";
import { SplitDownload, type DownloadEntry } from "./pick-menu.tsx";

export type PeekTab = "details" | "preview" | "compare";

const DOC_TOAST = "document";

const typing = (target: EventTarget | null) =>
  target instanceof HTMLElement && (target.isContentEditable || /^(?:INPUT|TEXTAREA|SELECT)$/.test(target.tagName));

/** What the open document lets the header's "⋯" do with it. */
interface DocTools {
  ready: boolean;
  copy: () => void;
  exportPdf: () => void;
  furniture: boolean;
  toggleFurniture: () => void;
}

export interface PeekProps {
  t: Dict;
  locale: Locale;
  /** The item shown, or null for a submission or the overview (Details only). */
  item: SessionItem | null;
  title: string;
  createdAt: string | null;
  position: { index: number; total: number } | null;
  tab: PeekTab;
  phone: boolean;
  compareAvailable: boolean;
  /** The server keeps originals (`features.original`): Files lists the item's own. */
  originals: boolean;
  /** Download all (.zip) of the item, or null when the server offers none. */
  onDownloadAll: (() => void) | null;
  acts: ActSet | null;
  /** The source page of a URL item, offered as "Open the page". */
  url: string | null;
  failure: ActionFailure | null;
  details: ComponentChildren;
  onTab: (tab: PeekTab) => void;
  onPrevious: (() => void) | null;
  onNext: (() => void) | null;
  onClose: () => void;
  onCopied: () => void;
  onCopyFailed: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}

export function Peek(props: PeekProps) {
  const { t, item, title, position, phone, compareAvailable, acts, url, failure, onTab, onPrevious, onNext, onClose } = props;
  const frame = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Record<PeekTab, HTMLButtonElement | null>>({ details: null, preview: null, compare: null });
  const [tools, setTools] = useState<DocTools | null>(null);
  const [pdfOpen, setPdfOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement | null>(null);
  const menuAnchor = useCallback(() => menuButton.current, []);
  const closePdf = useCallback(() => setPdfOpen(false), []);
  useEffect(() => setPdfOpen(false), [item?.key, tools === null]);
  const previewable = item !== null && item.status === "done" && item.output !== null && !item.skipped;
  // Preview first: a conversion opens on its document; Details is last.
  const tabs: PeekTab[] = previewable ? (compareAvailable ? ["preview", "compare", "details"] : ["preview", "details"]) : ["details"];
  const tab = tabs.includes(props.tab) ? props.tab : "details";
  const labels: Record<PeekTab, string> = { details: t.peekDetails, preview: t.preview, compare: t.compare };
  const [full, setFull] = useState(readFull);
  const wide = full && !phone;
  const toggleFull = useCallback(() => {
    setFull((value) => {
      storeFull(!value);
      return !value;
    });
  }, []);
  // The header's Download opens Preview's Files, where each file and the whole
  // item can be downloaded; the document switches to it and clears the request.
  const [wantFiles, setWantFiles] = useState(false);
  // Preview's own views share this row, at its end: as tabs when they fit beside
  // the window's tabs, else as one compact switch. `needed` is the width the tabs took.
  const row = useRef<HTMLDivElement>(null);
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  const [compactBar, setCompactBar] = useState(false);
  const needed = useRef(0);
  useEffect(() => setCompactBar(false), [item?.key, t]);
  useLayoutEffect(() => {
    const element = row.current;
    if (!element || typeof ResizeObserver !== "function") return;
    const check = () => {
      if (!compactBar) {
        if (element.scrollWidth > element.clientWidth + 1) {
          needed.current = element.scrollWidth;
          setCompactBar(true);
        }
      } else if (element.clientWidth >= needed.current) setCompactBar(false);
    };
    check();
    // Switching between tabs and the compact switch resizes what is observed
    // (and renews this observer), so a size change is judged in the next
    // frame, never inside the observer's own pass: WebKit reports that as a
    // loop with undelivered notifications.
    let frame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(check);
    });
    observer.observe(element);
    if (slot) observer.observe(slot);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [compactBar, slot, tab]);

  // Keys that work anywhere in the window.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || typing(event.target)) return;
      if (document.querySelector(".confirm-card, .menu, .pdf-card, .toc-sheet")) return;
      const target = event.target as Element | null;
      if (target?.closest?.(".conv-list")) return;
      if ((event.key === "f" || event.key === "F") && !phone) {
        event.preventDefault();
        toggleFull();
      } else if ((event.key === "c" || event.key === "C") && tabs.includes("compare")) {
        event.preventDefault();
        onTab(tab === "compare" ? "preview" : "compare");
      } else if ((event.key === "ArrowDown" || event.key === "ArrowUp") && !target?.closest?.(".pane, .orig-scroll, [role=tablist]")) {
        const go = event.key === "ArrowDown" ? onNext : onPrevious;
        if (!go) return;
        event.preventDefault();
        go();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [phone, tab, tabs.join(), onTab, onNext, onPrevious, toggleFull]);

  const onTabKey = (event: KeyboardEvent) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight" && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const index = tabs.indexOf(tab);
    const next =
      event.key === "Home" ? tabs[0] : event.key === "End" ? tabs[tabs.length - 1] : tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    if (!next) return;
    onTab(next);
    tabRefs.current[next]?.focus();
  };

  // "⋯": the document's own actions on Preview and Compare, the page, then the
  // conversion's secondary actions; on a phone the header's buttons too, except
  // the ones that must stay in reach (Stop while items wait).
  const { inline, folded } = headerActs(acts?.shown ?? [], phone);
  const openFiles = () => {
    setWantFiles(true);
    onTab("preview");
  };
  // Download's formats, and the result's file list (Preview's Files) to choose from.
  const formats = (act: ActSet["shown"][number]): DownloadEntry[] => [
    ...(act.menu ?? []),
    ...(previewable ? [{ key: "dl-files", label: t.downloadChooseFiles, icon: "Files" as const, run: openFiles }] : []),
  ];
  const moreKeys = new Set((acts?.more ?? []).map((act) => act.key));
  const menu: MenuAction[] = [
    // Folded on a phone, Download lists its formats in "⋯" (JSON once: its own entry stays below).
    ...folded.flatMap((act) =>
      act.menu
        ? formats(act)
            .filter((entry) => !(entry.key === "dl-json" && moreKeys.has("json")))
            .map((entry) => ({ key: entry.key, label: entry.ext ? t.downloadAs(`${entry.label} ${entry.ext}`) : entry.label, icon: entry.icon, run: entry.run, disabled: act.disabled }))
        : [{ key: act.key, label: act.label, icon: act.icon, run: act.run, disabled: act.disabled }],
    ),
    ...(tools
      ? [
          { key: "copy", label: t.copy, icon: "Copy" as const, run: tools.copy, disabled: !tools.ready },
          // Export PDF asks about the header and footer first, then prints.
          { key: "pdf", label: t.exportPdfAsk, icon: "FilePdf" as const, run: () => setPdfOpen(true), disabled: !tools.ready },
        ]
      : []),
    // Compare's header opens the page itself; elsewhere "⋯" does.
    ...(url && tab !== "compare" ? [{ key: "page", label: t.openPage, icon: "ArrowSquareOut" as const, run: () => undefined, href: url, hint: t.opensNewTab }] : []),
    ...(acts?.more ?? []).map((act) => ({ key: act.key, label: act.label, icon: act.icon, run: act.run, disabled: act.disabled })),
  ];

  const header = (
    <header class="peek-head">
      {phone && (
        <button type="button" class="icon-btn" aria-label={t.close} onClick={onClose}>
          <Icon name="ArrowLeft" size={18} />
        </button>
      )}
      {position && !phone && (
        <span class="peek-nav">
          <button type="button" class="icon-btn" aria-label={t.previousItem} title={`${t.previousItem} (↑)`} disabled={onPrevious === null} onClick={() => onPrevious?.()}>
            <Icon name="CaretUp" size={15} />
          </button>
          <button type="button" class="icon-btn" aria-label={t.nextItem} title={`${t.nextItem} (↓)`} disabled={onNext === null} onClick={() => onNext?.()}>
            <Icon name="CaretDown" size={15} />
          </button>
          <span class="peek-pos">{t.positionOf(position.index + 1, position.total)}</span>
        </span>
      )}
      {/* An item's name gives up its middle, as file names do; a phone puts it on its own row. */}
      <h2 id="peek-title" class="peek-title" title={item ? undefined : title}>
        {item ? <FileName name={title} url={url !== null} /> : title}
      </h2>
      <span class="peek-tools">
        {position && phone && (
          <>
            <button type="button" class="icon-btn" aria-label={t.previousItem} disabled={onPrevious === null} onClick={() => onPrevious?.()}>
              <Icon name="CaretUp" size={16} />
            </button>
            <button type="button" class="icon-btn" aria-label={t.nextItem} disabled={onNext === null} onClick={() => onNext?.()}>
              <Icon name="CaretDown" size={16} />
            </button>
          </>
        )}
        {inline.map((act) =>
          act.menu ? (
            <SplitDownload key={act.key} label={act.label} menuLabel={t.downloadFormats} title={act.title} entries={formats(act)} disabled={act.disabled} />
          ) : (
          <button
            key={act.key}
            type="button"
            class={act.primary ? "btn btn-primary" : "btn"}
            title={act.title}
            disabled={act.disabled}
            aria-busy={act.busy || undefined}
            onClick={act.run}
          >
            {act.icon && <Icon name={act.icon} size={14} />}
            {act.label}
          </button>
          ),
        )}
        <ActionMenu label={t.moreActions} actions={menu} danger={acts?.danger ?? undefined} plain triggerRef={menuButton} />
        {pdfOpen && tools && (
          <PdfExportCard t={t} anchor={menuAnchor} furniture={tools.furniture} onToggle={tools.toggleFurniture} onPrint={tools.exportPdf} onClose={closePdf} />
        )}
        {!phone && (
          <button type="button" class="icon-btn" aria-pressed={full} aria-label={t.peekFullScreen} title={`${full ? t.peekExitFullScreen : t.peekFullScreen} (F)`} onClick={toggleFull}>
            <Icon name={full ? "ArrowsInSimple" : "ArrowsOutSimple"} size={15} />
          </button>
        )}
        {!phone && (
          <button type="button" class="icon-btn" aria-label={t.close} title={`${t.close} (Esc)`} onClick={onClose}>
            <Icon name="X" size={15} />
          </button>
        )}
      </span>
    </header>
  );

  const body = (
    <div
      ref={frame}
      class={`peek${wide ? " is-full" : ""}${phone ? " is-phone" : ""}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby="peek-title"
      tabIndex={-1}
    >
      {header}
      <ActionError t={t} failure={failure} />
      {tabs.length > 1 && (
        <div ref={row} class="peek-tabs">
          <div class="peek-tablist" role="tablist" aria-label={t.peekViews}>
          {tabs.map((value) => (
            <button
              key={value}
              ref={(node) => {
                tabRefs.current[value] = node;
              }}
              type="button"
              role="tab"
              id={`peek-tab-${value}`}
              aria-selected={tab === value}
              aria-controls="peek-panel"
              tabIndex={tab === value ? 0 : -1}
              class={tab === value ? "tab is-on" : "tab"}
              onClick={() => onTab(value)}
              onKeyDown={onTabKey}
            >
              {labels[value]}
            </button>
          ))}
          </div>
          {tab === "preview" && <div ref={setSlot} class="peek-tabs-side" />}
        </div>
      )}
      <div id="peek-panel" class="peek-body" role={tabs.length > 1 ? "tabpanel" : undefined} aria-labelledby={tabs.length > 1 ? `peek-tab-${tab}` : undefined}>
        {tab === "details" || item === null ? (
          <div class="peek-details">{props.details}</div>
        ) : (
          <PeekDocument {...props} item={item} mode={tab === "compare" ? "compare" : "preview"} onTools={setTools} barHost={tab === "preview" ? slot : null} compactBar={compactBar} wantFiles={wantFiles} onFilesShown={() => setWantFiles(false)} />
        )}
      </div>
      {!phone && (
        <footer class="peek-keys" aria-hidden="true">
          {position && (
            <span>
              <kbd>↑</kbd>
              <kbd>↓</kbd> {t.keysMove}
            </span>
          )}
          <span>
            <kbd>Esc</kbd> {t.keysClose}
          </span>
          {tabs.includes("compare") && (
            <span>
              <kbd>C</kbd> {t.compare}
            </span>
          )}
          <span>
            <kbd>F</kbd> {full ? t.peekExitFullScreen : t.peekFullScreen}
          </span>
        </footer>
      )}
    </div>
  );
  return (
    <PeekVeil frame={frame} phone={phone} full={wide} onClose={onClose}>
      {body}
    </PeekVeil>
  );
}

/** The window over the dimmed list; a click on the dimmed part closes it. */
function PeekVeil({ frame, phone, full, onClose, children }: { frame: { current: HTMLDivElement | null }; phone: boolean; full: boolean; onClose: () => void; children: ComponentChildren }) {
  useModal(frame, onClose);
  // Notices sit at the bottom edge while the phone window covers the tab bar.
  useEffect(() => {
    if (!phone) return;
    const root = document.documentElement;
    root.style.setProperty("--strips-bottom", "env(safe-area-inset-bottom)");
    return () => root.style.removeProperty("--strips-bottom");
  }, [phone]);
  return (
    <div class={`veil peek-veil${phone ? " is-phone" : full ? " is-full" : ""}`} onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      {children}
    </div>
  );
}

/** Preview and Compare: the whole result, read once and kept while the tabs switch. */
function PeekDocument({
  t,
  locale,
  item,
  createdAt,
  mode,
  phone,
  onTab,
  onTools,
  onCopied,
  onCopyFailed,
  announce,
  describe,
  barHost,
  compactBar,
  wantFiles,
  onFilesShown,
  originals,
  onDownloadAll,
}: PeekProps & { item: SessionItem; mode: "preview" | "compare"; onTools: (tools: DocTools | null) => void; barHost: HTMLElement | null; compactBar: boolean; wantFiles: boolean; onFilesShown: () => void }) {
  const doc = useResultDocument(item);
  // Rendered, or Source when Details' result start asked for it
  const [tab, setTab] = useState<Tab>(takePreviewView);
  const shownKey = useRef(item.key);
  // A failed export or download: an error toast, replaced by the next one.
  const setError = useCallback((message: string | null) => (message === null ? dismissToast(DOC_TOAST) : toast({ id: DOC_TOAST, tone: "error", text: message })), []);
  const [furniture, setFurniture] = useState(readFurniture);
  const rendered = useRef<RenderedState>({ printable: null, images: null, complete: null });
  const printing = useRef<PrintJob | null>(null);
  useEffect(() => {
    if (shownKey.current !== item.key) setTab("rendered");
    shownKey.current = item.key;
    setError(null);
  }, [item.key]);
  useEffect(() => () => printing.current?.cancel(), []);
  useEffect(() => {
    if (!wantFiles || mode !== "preview") return;
    setTab("files");
    onFilesShown();
  }, [wantFiles, mode]);

  const exportPdf = useCallback(async () => {
    if (doc.markdown === null) return;
    // The document must be on screen (Rendered, not Compare) to be printed.
    if (tab !== "rendered" || mode !== "preview") {
      onTab("preview");
      setTab("rendered");
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    const source = rendered.current.printable;
    if (!source) return;
    printing.current?.cancel();
    setError(null);
    const path = doc.path;
    // A long document is drawn in steps and its pictures load as they come into view: print all of it.
    await rendered.current.complete;
    await rendered.current.images?.all();
    const job = printDocument({
      source,
      title: readable(basename(path || item.name)).replace(/\.llm\.md$/i, "").replace(/\.md$/i, ""),
      furniture,
      isCurrent: () => path === doc.path,
      messages: { loading: t.printImagesLoading, broken: t.printImageBroken },
    });
    printing.current = job;
    announce(t.exportPdf);
    job.done.then(
      () => undefined,
      (failure: Error) => {
        if (failure.name !== "AbortError") setError(failure.message);
      },
    );
  }, [doc.markdown, doc.path, tab, mode, furniture, item.name, t, announce, onTab]);

  useEffect(() => {
    onTools({
      ready: doc.markdown !== null,
      copy: () =>
        void copyText(doc.markdown ?? "").then((ok) => {
          if (ok) {
            announce(t.copied);
            onCopied();
          } else onCopyFailed();
        }),
      exportPdf: () => void exportPdf(),
      furniture,
      toggleFurniture: () => {
        const next = !furniture;
        setFurniture(next);
        storeFurniture(next);
      },
    });
  }, [doc.path, doc.markdown, exportPdf, furniture]);
  useEffect(() => () => onTools(null), []);

  return (
    <>
      {mode === "compare" ? (
        <CompareView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} phone={phone} />
      ) : (
        <MarkdownView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} tab={tab} onTab={setTab} rendered={rendered} onError={setError} barHost={barHost} compactBar={compactBar} originals={originals} onDownloadAll={onDownloadAll} />
      )}
    </>
  );
}

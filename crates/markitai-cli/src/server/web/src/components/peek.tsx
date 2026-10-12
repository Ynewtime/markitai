// The peek window: a click on a conversion opens one window over the dimmed
// list, with Details, Preview and Compare as tabs, previous/next with its
// position, the one or two actions that matter and "⋯" for the rest. ◫ docks
// the same window beside the list instead (remembered per browser); on a
// phone it fills the screen. Up/Down move between items (outside the scrolling
// document), C switches Compare, Escape closes.
import type { ComponentChildren } from "preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import { useResultDocument } from "../hooks/use-result-document.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { headerActs } from "../lib/acts.ts";
import { copyText } from "../lib/clipboard.ts";
import { basename } from "../lib/format.ts";
import { printDocument, type PrintJob } from "../lib/print.ts";
import type { SessionItem } from "../lib/session.ts";
import { ActionMenu, type MenuAction } from "./action-menu.tsx";
import { CompareView } from "./compare-view.tsx";
import { Icon } from "./icons.tsx";
import { ActionError, type ActionFailure, type ActSet } from "./inspector.tsx";
import { MarkdownView, type RenderedState, type Tab } from "./markdown-view.tsx";
import { readFurniture, storeFurniture } from "./pdf-settings.tsx";

export type PeekTab = "details" | "preview" | "compare";

const DOCK_KEY = "markitai.peek";
/** Whether this browser opens conversions docked beside the list instead of as a window. */
export function readDocked(): boolean {
  try {
    return localStorage.getItem(DOCK_KEY) === "docked";
  } catch {
    return false;
  }
}
export function storeDocked(docked: boolean): void {
  try {
    localStorage.setItem(DOCK_KEY, docked ? "docked" : "window");
  } catch {
    // A browser without storage keeps the choice for this page only.
  }
}

const typing = (target: EventTarget | null) =>
  target instanceof HTMLElement && (target.isContentEditable || /^(?:INPUT|TEXTAREA|SELECT)$/.test(target.tagName));

/** What the open document lets the header's "⋯" do with it. */
interface DocTools {
  path: string;
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
  docked: boolean;
  compareAvailable: boolean;
  acts: ActSet | null;
  /** The source page of a URL item, offered as "Open the page". */
  url: string | null;
  failure: ActionFailure | null;
  details: ComponentChildren;
  onTab: (tab: PeekTab) => void;
  onPrevious: (() => void) | null;
  onNext: (() => void) | null;
  onDock: () => void;
  onClose: () => void;
  onDownloadPath: (path: string) => void;
  onCopied: () => void;
  onCopyFailed: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}

export function Peek(props: PeekProps) {
  const { t, item, title, position, phone, docked, compareAvailable, acts, url, failure, onTab, onPrevious, onNext, onClose } = props;
  const frame = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Record<PeekTab, HTMLButtonElement | null>>({ details: null, preview: null, compare: null });
  const [tools, setTools] = useState<DocTools | null>(null);
  const previewable = item !== null && item.status === "done" && item.output !== null && !item.skipped;
  const tabs: PeekTab[] = previewable ? (compareAvailable ? ["details", "preview", "compare"] : ["details", "preview"]) : ["details"];
  const tab = tabs.includes(props.tab) ? props.tab : "details";
  const labels: Record<PeekTab, string> = { details: t.peekDetails, preview: t.preview, compare: t.compare };
  const windowed = !docked || phone;

  // Keys that work anywhere in the window (and, docked, outside the list).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || typing(event.target)) return;
      if (document.querySelector(".confirm-card, .menu, .pdf-card")) return;
      const target = event.target as Element | null;
      if (target?.closest?.(".conv-list")) return;
      if (event.key === "Escape" && !windowed) {
        event.preventDefault();
        onClose();
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
  }, [windowed, tab, tabs.join(), onTab, onNext, onPrevious, onClose]);

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
  const menu: MenuAction[] = [
    ...folded.map((act) => ({ key: act.key, label: act.label, icon: act.icon, run: act.run, disabled: act.disabled })),
    ...(tools
      ? [
          { key: "copy", label: t.copy, icon: "Copy" as const, run: tools.copy, disabled: !tools.ready },
          { key: "pdf", label: t.exportPdf, icon: "FilePdf" as const, run: tools.exportPdf, disabled: !tools.ready },
          { key: "furniture", label: t.pdfCustomHeaderFooter, run: tools.toggleFurniture, checked: tools.furniture, note: t.pdfPrintDialogHint },
        ]
      : []),
    ...(url ? [{ key: "page", label: t.openPage, icon: "ArrowSquareOut" as const, run: () => undefined, href: url, hint: t.opensNewTab }] : []),
    ...(acts?.more ?? []).map((act) => ({ key: act.key, label: act.label, icon: act.icon, run: act.run, disabled: act.disabled })),
  ];
  const run = (act: ActSet["shown"][number]) => (act.key === "download" && tools ? () => props.onDownloadPath(tools.path) : act.run);

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
      <h2 id="peek-title" class="peek-title" title={title}>
        {title}
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
        {inline.map((act) => (
          <button
            key={act.key}
            type="button"
            class={act.primary ? "btn btn-primary" : "btn"}
            title={act.title}
            disabled={act.disabled}
            aria-busy={act.busy || undefined}
            onClick={run(act)}
          >
            {act.icon && <Icon name={act.icon} size={14} />}
            {act.label}
          </button>
        ))}
        <ActionMenu label={t.moreActions} actions={menu} danger={acts?.danger ?? undefined} />
        {!phone && (
          <button type="button" class="icon-btn" aria-pressed={docked} aria-label={docked ? t.peekUndock : t.peekDock} title={docked ? t.peekUndock : t.peekDock} onClick={props.onDock}>
            <Icon name={docked ? "ArrowsOutSimple" : "SidebarSimple"} size={15} />
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
      class={`peek${docked && !phone ? " is-docked" : ""}${phone ? " is-phone" : ""}`}
      role={windowed ? "dialog" : "region"}
      aria-modal={windowed ? "true" : undefined}
      aria-labelledby="peek-title"
      tabIndex={-1}
    >
      {header}
      <ActionError t={t} failure={failure} />
      {tabs.length > 1 && (
        <div class="peek-tabs" role="tablist" aria-label={t.peekViews}>
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
      )}
      <div id="peek-panel" class="peek-body" role={tabs.length > 1 ? "tabpanel" : undefined} aria-labelledby={tabs.length > 1 ? `peek-tab-${tab}` : undefined}>
        {tab === "details" || item === null ? (
          <div class="peek-details">{props.details}</div>
        ) : (
          <PeekDocument {...props} item={item} mode={tab === "compare" ? "compare" : "preview"} onTools={setTools} />
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
        </footer>
      )}
    </div>
  );
  if (!windowed) return body;
  return (
    <PeekVeil frame={frame} phone={phone} onClose={onClose}>
      {body}
    </PeekVeil>
  );
}

/** The window over the dimmed list; a click on the dimmed part closes it. */
function PeekVeil({ frame, phone, onClose, children }: { frame: { current: HTMLDivElement | null }; phone: boolean; onClose: () => void; children: ComponentChildren }) {
  useModal(frame, onClose);
  // Notices sit at the bottom edge while the phone window covers the tab bar.
  useEffect(() => {
    if (!phone) return;
    const root = document.documentElement;
    root.style.setProperty("--strips-bottom", "env(safe-area-inset-bottom)");
    return () => root.style.removeProperty("--strips-bottom");
  }, [phone]);
  return (
    <div class={phone ? "veil peek-veil is-phone" : "veil peek-veil"} onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
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
}: PeekProps & { item: SessionItem; mode: "preview" | "compare"; onTools: (tools: DocTools | null) => void }) {
  const doc = useResultDocument(item);
  const [tab, setTab] = useState<Tab>("rendered");
  const [error, setError] = useState<string | null>(null);
  const [furniture, setFurniture] = useState(readFurniture);
  const rendered = useRef<RenderedState>({ printable: null, images: null });
  const printing = useRef<PrintJob | null>(null);
  useEffect(() => {
    setTab("rendered");
    setError(null);
  }, [item.key]);
  useEffect(() => () => printing.current?.cancel(), []);

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
    await rendered.current.images?.done;
    const job = printDocument({
      source,
      title: basename(path || item.name).replace(/\.llm\.md$/i, "").replace(/\.md$/i, ""),
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
      path: doc.path,
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
      {error && (
        <p class="viewer-alert line-error" role="alert">
          {error}
        </p>
      )}
      {mode === "compare" ? (
        <CompareView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} phone={phone} />
      ) : (
        <MarkdownView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} tab={tab} onTab={setTab} rendered={rendered} onError={setError} />
      )}
    </>
  );
}

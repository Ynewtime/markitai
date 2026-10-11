// The preview (B3), docked beside the list so the list stays usable: Up/Down
// move between items (outside the scrolling document), Escape closes, F goes
// full width, C toggles Compare. The header carries previous/next with the
// position, the name, and Copy / Download / "⋯" (PDF export). Compare turns
// the body into the original beside the Markdown (B2). On phones the preview
// is a full-screen view with previous/next.
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import { useResultDocument } from "../hooks/use-result-document.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { copyText } from "../lib/clipboard.ts";
import { basename } from "../lib/format.ts";
import { printDocument, type PrintJob } from "../lib/print.ts";
import type { SessionItem } from "../lib/session.ts";
import { CompareView } from "./compare-view.tsx";
import { Icon } from "./icons.tsx";
import { MarkdownView, type RenderedState, type Tab } from "./markdown-view.tsx";
import { readFurniture, storeFurniture } from "./pdf-settings.tsx";

const typing = (target: EventTarget | null) =>
  target instanceof HTMLElement && (target.isContentEditable || /^(?:INPUT|TEXTAREA|SELECT)$/.test(target.tagName));

export interface PreviewProps {
  t: Dict;
  locale: Locale;
  item: SessionItem;
  createdAt: string | null;
  position: { index: number; total: number };
  mode: "preview" | "compare";
  wide: boolean;
  phone: boolean;
  compareAvailable: boolean;
  onPrevious: (() => void) | null;
  onNext: (() => void) | null;
  onClose: () => void;
  onMode: (mode: "preview" | "compare") => void;
  onWide: () => void;
  onDownload: (item: SessionItem, path: string) => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}

function MoreMenu({ t, disabled, furniture, url, onExport, onFurniture }: { t: Dict; disabled: boolean; furniture: boolean; url: string | null; onExport: () => void; onFurniture: () => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = () => [...(root.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? [])];
  useEffect(() => {
    if (!open) return;
    requestAnimationFrame(() => items()[0]?.focus());
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [open]);
  const onKeyDown = (event: KeyboardEvent) => {
    const list = items();
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      trigger.current?.focus();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      list[(index + (event.key === "ArrowDown" ? 1 : list.length - 1)) % list.length]?.focus();
    } else if (event.key === "Tab") setOpen(false);
  };
  return (
    <span ref={root} class="menu-wrap" onKeyDown={onKeyDown}>
      <button
        ref={trigger}
        type="button"
        class="icon-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t.moreActions}
        title={t.moreActions}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="DotsThree" size={18} />
      </button>
      {open && (
        <div class="menu" role="menu" aria-label={t.moreActions}>
          <button
            type="button"
            role="menuitem"
            tabIndex={-1}
            onClick={() => {
              setOpen(false);
              onExport();
            }}
          >
            <Icon name="FilePdf" size={15} />
            {t.exportPdf}
          </button>
          <button type="button" role="menuitemcheckbox" aria-checked={furniture} tabIndex={-1} onClick={onFurniture}>
            <span class="menu-check" aria-hidden="true">
              {furniture && <Icon name="Check" size={13} />}
            </span>
            {t.pdfCustomHeaderFooter}
          </button>
          <p class="menu-note">{t.pdfPrintDialogHint}</p>
          {url && (
            <a role="menuitem" tabIndex={-1} href={url} target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
              <Icon name="ArrowSquareOut" size={15} />
              {t.openPage}
              <span class="sr-only"> {t.opensNewTab}</span>
            </a>
          )}
        </div>
      )}
    </span>
  );
}

export function PreviewPane(props: PreviewProps) {
  const { t, locale, item, createdAt, position, mode, wide, phone, compareAvailable, onPrevious, onNext, onClose, onMode, onWide, onDownload, announce, describe } = props;
  const doc = useResultDocument(item);
  const [tab, setTab] = useState<Tab>("rendered");
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const [error, setError] = useState<string | null>(null);
  const [furniture, setFurniture] = useState(readFurniture);
  const rendered = useRef<RenderedState>({ printable: null, images: null });
  const printing = useRef<PrintJob | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const url = item.kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;

  useEffect(() => {
    setTab("rendered");
    setError(null);
  }, [item.key]);
  useEffect(() => {
    if (copied === "idle") return;
    const timer = setTimeout(() => setCopied("idle"), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  useEffect(() => () => printing.current?.cancel(), []);

  const exportPdf = useCallback(async () => {
    if (doc.markdown === null) return;
    // The document must be on screen (Rendered tab) to be printed.
    if (tab !== "rendered" || mode !== "preview") {
      onMode("preview");
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
  }, [doc.markdown, doc.path, tab, mode, furniture, item.name, t, announce, onMode]);

  // Keys that work anywhere in the preview (and the list beside it).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || typing(event.target)) return;
      if (document.querySelector(".confirm-card, .menu, .veil:not(.preview-veil)") && !phone) return;
      const target = event.target as Element | null;
      const inList = target?.closest?.(".conv-list") != null;
      const inDocument = target?.closest?.(".pane, .orig-scroll") != null;
      if (event.key === "Escape") {
        if (phone) return; // the modal hook owns Escape there
        event.preventDefault();
        onClose();
      } else if ((event.key === "f" || event.key === "F") && !phone) {
        event.preventDefault();
        onWide();
      } else if ((event.key === "c" || event.key === "C") && compareAvailable) {
        event.preventDefault();
        onMode(mode === "compare" ? "preview" : "compare");
      } else if ((event.key === "ArrowDown" || event.key === "ArrowUp") && !inList && !inDocument) {
        const go = event.key === "ArrowDown" ? onNext : onPrevious;
        if (!go) return;
        event.preventDefault();
        go();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [phone, onClose, onWide, onMode, mode, compareAvailable, onNext, onPrevious]);

  if (phone) return <PhonePreview {...props} doc={doc} tab={tab} setTab={setTab} frame={frame} rendered={rendered} error={error} setError={setError} />;

  const heading = `preview-title`;
  return (
    <section ref={frame} class={`preview${wide ? " is-wide" : ""}${mode === "compare" ? " is-compare" : ""}`} aria-labelledby={heading}>
      <header class="preview-head">
        <div class="preview-nav">
          <button type="button" class="icon-btn" aria-label={t.previousItem} title={`${t.previousItem} (↑)`} disabled={onPrevious === null} onClick={() => onPrevious?.()}>
            <Icon name="CaretUp" size={16} />
          </button>
          <button type="button" class="icon-btn" aria-label={t.nextItem} title={`${t.nextItem} (↓)`} disabled={onNext === null} onClick={() => onNext?.()}>
            <Icon name="CaretDown" size={16} />
          </button>
          <span class="preview-pos">{t.positionOf(position.index + 1, position.total)}</span>
        </div>
        <h2 id={heading} class="preview-title" title={item.name}>
          {item.name}
        </h2>
        <div class="preview-tools">
          {compareAvailable && (
            <button type="button" class={mode === "compare" ? "btn btn-sm is-pressed" : "btn btn-sm"} aria-pressed={mode === "compare"} title={`${t.compare} (C)`} onClick={() => onMode(mode === "compare" ? "preview" : "compare")}>
              <Icon name="Columns" size={14} />
              {t.compare}
            </button>
          )}
          <button
            type="button"
            class="btn btn-sm"
            disabled={doc.markdown === null}
            onClick={() =>
              void copyText(doc.markdown ?? "").then((ok) => {
                setCopied(ok ? "copied" : "failed");
                announce(ok ? t.copied : t.copyFailed);
              })
            }
          >
            <Icon name={copied === "copied" ? "Check" : "Copy"} size={14} />
            {copied === "copied" ? t.copied : copied === "failed" ? t.copyFailed : t.copy}
          </button>
          <button type="button" class="btn btn-sm" disabled={!doc.path} onClick={() => onDownload(item, doc.path)}>
            <Icon name="DownloadSimple" size={14} />
            {t.downloadMd}
          </button>
          <MoreMenu
            t={t}
            disabled={doc.markdown === null}
            furniture={furniture}
            url={url}
            onExport={() => void exportPdf()}
            onFurniture={() => {
              const next = !furniture;
              setFurniture(next);
              storeFurniture(next);
            }}
          />
          <button type="button" class="icon-btn" aria-pressed={wide} aria-label={wide ? t.dockPreview : t.fullWidth} title={`${wide ? t.dockPreview : t.fullWidth} (F)`} onClick={onWide}>
            <Icon name={wide ? "ArrowsInSimple" : "ArrowsOutSimple"} size={16} />
          </button>
          <button type="button" class="icon-btn" aria-label={t.closePreview} title={`${t.closePreview} (Esc)`} onClick={onClose}>
            <Icon name="X" size={16} />
          </button>
        </div>
      </header>
      {error && (
        <p class="viewer-alert line-error" role="alert">
          {error}
        </p>
      )}
      <div class="preview-body">
        {mode === "compare" ? (
          <CompareView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} phone={false} />
        ) : (
          <MarkdownView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} tab={tab} onTab={setTab} rendered={rendered} onError={setError} />
        )}
      </div>
      <footer class="preview-keys" aria-hidden="true">
        <span>
          <kbd>↑</kbd>
          <kbd>↓</kbd> {t.keysMove}
        </span>
        <span>
          <kbd>Esc</kbd> {t.keysClose}
        </span>
        <span>
          <kbd>F</kbd> {wide ? t.dockPreview : t.fullWidth}
        </span>
        {compareAvailable && (
          <span>
            <kbd>C</kbd> {t.compare}
          </span>
        )}
      </footer>
    </section>
  );
}

function PhonePreview({
  t,
  locale,
  item,
  createdAt,
  position,
  mode,
  compareAvailable,
  onPrevious,
  onNext,
  onClose,
  onMode,
  onDownload,
  describe,
  doc,
  tab,
  setTab,
  frame,
  rendered,
  error,
  setError,
}: PreviewProps & {
  doc: ReturnType<typeof useResultDocument>;
  tab: Tab;
  setTab: (tab: Tab) => void;
  frame: { current: HTMLDivElement | null };
  rendered: { current: RenderedState };
  error: string | null;
  setError: (error: string | null) => void;
}) {
  useModal(frame, onClose);
  return (
    <div class="veil preview-veil">
      <div ref={frame} class={mode === "compare" ? "preview is-phone is-compare" : "preview is-phone"} role="dialog" aria-modal="true" aria-labelledby="preview-title" tabIndex={-1}>
        <header class="preview-head">
          <button type="button" class="icon-btn" aria-label={t.closePreview} onClick={onClose}>
            <Icon name="ArrowLeft" size={18} />
          </button>
          <h2 id="preview-title" class="preview-title">
            {item.name}
          </h2>
          <button type="button" class="icon-btn" aria-label={t.previousItem} disabled={onPrevious === null} onClick={() => onPrevious?.()}>
            <Icon name="CaretUp" size={16} />
          </button>
          <span class="preview-pos">{t.positionOf(position.index + 1, position.total)}</span>
          <button type="button" class="icon-btn" aria-label={t.nextItem} disabled={onNext === null} onClick={() => onNext?.()}>
            <Icon name="CaretDown" size={16} />
          </button>
        </header>
        <div class="preview-phone-tools">
          {compareAvailable && (
            <button type="button" class={mode === "compare" ? "btn btn-sm is-pressed" : "btn btn-sm"} aria-pressed={mode === "compare"} onClick={() => onMode(mode === "compare" ? "preview" : "compare")}>
              <Icon name="Columns" size={14} />
              {t.compare}
            </button>
          )}
          <button type="button" class="btn btn-sm" disabled={!doc.path} onClick={() => onDownload(item, doc.path)}>
            <Icon name="DownloadSimple" size={14} />
            {t.downloadShort}
          </button>
        </div>
        {error && (
          <p class="viewer-alert line-error" role="alert">
            {error}
          </p>
        )}
        <div class="preview-body">
          {mode === "compare" ? (
            <CompareView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} phone />
          ) : (
            <MarkdownView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} tab={tab} onTab={setTab} rendered={rendered} onError={setError} />
          )}
        </div>
      </div>
    </div>
  );
}

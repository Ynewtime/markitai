// The rendered document's contents: its headings, from the Markdown, as a
// column beside the document on a wide window or as a sheet from the bottom
// on a phone or a narrow window. The entry of the section being read is
// marked as the document scrolls; choosing one scrolls the document to it.
import { useEffect, useRef } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import type { TocEntry } from "../lib/toc.ts";
import { Icon } from "./icons.tsx";

function TocList({ entries, active, onPick, listRef }: { entries: TocEntry[]; active: number; onPick: (index: number) => void; listRef?: { current: HTMLOListElement | null } }) {
  return (
    <ol class="toc-list" ref={listRef}>
      {entries.map((entry, position) => (
        <li key={entry.index} class={`toc-item is-l${entry.level}`}>
          <button
            type="button"
            class={position === active ? "toc-link is-on" : "toc-link"}
            aria-current={position === active ? "location" : undefined}
            title={entry.text}
            onClick={() => onPick(entry.index)}
          >
            {entry.text}
          </button>
        </li>
      ))}
    </ol>
  );
}

/** Keep the marked entry in view inside the list's own scroll, never the page's. */
function useFollow(list: { current: HTMLElement | null }, active: number) {
  useEffect(() => {
    const root = list.current;
    const link = root?.querySelectorAll<HTMLElement>(".toc-link")[active];
    if (!root || !link) return;
    // The list is positioned, so a link's offset is measured from it.
    const top = link.offsetTop;
    const bottom = top + link.offsetHeight;
    if (top < root.scrollTop + 8) root.scrollTop = Math.max(0, top - 24);
    else if (bottom > root.scrollTop + root.clientHeight - 8) root.scrollTop = bottom - root.clientHeight + 24;
  }, [active]);
}

/** The column beside the document. */
export function TocAside({ t, entries, active, onPick }: { t: Dict; entries: TocEntry[]; active: number; onPick: (index: number) => void }) {
  const list = useRef<HTMLOListElement>(null);
  useFollow(list, active);
  return (
    <nav class="toc" aria-label={t.tocTitle}>
      <p class="toc-title" aria-hidden="true">
        {t.tocTitle}
      </p>
      <TocList entries={entries} active={active} onPick={onPick} listRef={list} />
    </nav>
  );
}

/** The sheet over the document (phones and narrow windows). Escape, the dimmed part or ✕ close it. */
export function TocSheet({ t, entries, active, onPick, onClose }: { t: Dict; entries: TocEntry[]; active: number; onPick: (index: number) => void; onClose: () => void }) {
  const card = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    (card.current?.querySelector<HTMLElement>(".toc-link.is-on") ?? card.current)?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !card.current) return;
      const nodes = [...card.current.querySelectorAll<HTMLElement>("button")];
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      opener?.focus?.();
    };
  }, []);
  useFollow(list, active);
  return (
    <div class="toc-veil" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div ref={card} class="toc-sheet" role="dialog" aria-modal="true" aria-labelledby="toc-sheet-title" tabIndex={-1}>
        <div class="toc-sheet-head">
          <h2 id="toc-sheet-title">{t.tocTitle}</h2>
          <button type="button" class="icon-btn" aria-label={t.close} onClick={onClose}>
            <Icon name="X" size={16} />
          </button>
        </div>
        <TocList entries={entries} active={active} onPick={onPick} listRef={list} />
      </div>
    </div>
  );
}

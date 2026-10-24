// A document's front matter above its body. Folded (the default): one light line,
// "Properties · N", the tags inline and a "Summary" mark at its end, no title (the body's
// own heading is the title). Open: a two-column properties table whose header folds it
// again. Either way the whole header row is one disclosure button with aria-expanded;
// the choice lasts for the tab's session. The table's key column has one fixed width: a key
// longer than it ends in an ellipsis (its whole name as a tooltip), and a value stays on one
// line that scrolls sideways inside its own cell.
import { useEffect, useId, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { frontOpen, onFrontOpen, setFrontOpen } from "../lib/front-pref.ts";
import { frontLine, frontRows, frontValue, type FrontRow } from "../lib/front-rows.ts";
import type { FrontMatter } from "../lib/frontmatter.ts";
import { Icon } from "./icons.tsx";

function useFrontOpen(): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(frontOpen);
  useEffect(() => onFrontOpen(setOpen), []);
  return [open, setFrontOpen];
}

function when(value: string): string {
  const date = new Date(value);
  try {
    return date.toLocaleString(document.documentElement.lang || undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return value;
  }
}

function Value({ t, row }: { t: Dict; row: FrontRow }) {
  if (row.kind === "tags")
    return (
      <ul class="front-tags">
        {row.value.map((tag) => (
          <li key={tag}>{tag}</li>
        ))}
      </ul>
    );
  if (row.kind === "link")
    return (
      <a class="text-link" href={row.value} target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">
        {row.value}
      </a>
    );
  if (row.kind === "time")
    return (
      <time dateTime={row.value} title={row.value}>
        {when(row.value)}
      </time>
    );
  const shown = frontValue(t, row.key, row.value);
  // a named value keeps the file's own word in its tooltip
  return shown === row.value ? <>{row.value}</> : <span title={row.value}>{shown}</span>;
}

/** A value that goes on past its cell's right edge fades there until it is scrolled to its end. */
function markMore(cell: HTMLElement): void {
  cell.classList.toggle("is-more", cell.scrollLeft + cell.clientWidth < cell.scrollWidth - 1);
}

/** A cut key gets its whole name as a tooltip; a cell whose value scrolls is reachable by the
 * keyboard when nothing in it is (a link already is). */
function fitTable(table: HTMLElement): void {
  for (const key of table.querySelectorAll<HTMLElement>("dt")) {
    if (key.scrollWidth > key.clientWidth + 1) key.title = key.textContent ?? "";
    else key.removeAttribute("title");
  }
  for (const cell of table.querySelectorAll<HTMLElement>(".front-cell")) {
    if (cell.scrollWidth > cell.clientWidth + 1 && !cell.querySelector("a")) cell.tabIndex = 0;
    else cell.removeAttribute("tabindex");
    markMore(cell);
  }
}

export function FrontCard({ t, front, compact = false }: { t: Dict; front: FrontMatter; compact?: boolean }) {
  const [open, setOpen] = useFrontOpen();
  const id = useId();
  const table = useRef<HTMLDListElement>(null);
  useLayoutEffect(() => {
    const target = table.current;
    if (!open || !target) return;
    const fit = () => fitTable(target);
    fit();
    // the web font and the width the table gets can both arrive later
    void document.fonts?.ready.then(fit);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fit);
    observer?.observe(target);
    // scroll does not bubble: one capturing listener follows every cell
    const scrolled = (event: Event) => event.target instanceof HTMLElement && event.target.classList.contains("front-cell") && markMore(event.target);
    target.addEventListener("scroll", scrolled, { capture: true, passive: true });
    return () => {
      observer?.disconnect();
      target.removeEventListener("scroll", scrolled, { capture: true });
    };
  }, [open, front, t]);
  const rows = frontRows(front);
  const line = frontLine(front);
  const classes = ["front-meta", compact ? "is-compact" : "", open ? "is-open" : ""].filter(Boolean).join(" ");
  if (open)
    return (
      <section class={classes} aria-label={t.frontLabel}>
        <button type="button" class="front-head" aria-expanded="true" aria-controls={id} onClick={() => setOpen(false)}>
          <Icon name="Tag" size={13} />
          <span class="front-head-label">{t.frontProperties(rows.length)}</span>
          <Icon name="CaretUp" size={12} />
        </button>
        <dl id={id} ref={table} class="front-table">
          {rows.map((row) => (
            <div key={row.key} class="front-row">
              <dt>{t.frontKey(row.key)}</dt>
              <dd>
                <div class="front-cell">
                  <Value t={t} row={row} />
                </div>
              </dd>
            </div>
          ))}
        </dl>
      </section>
    );
  return (
    <section class={classes} aria-label={t.frontLabel}>
      <button type="button" class="front-fold" aria-expanded="false" onClick={() => setOpen(true)}>
        <Icon name="Tag" size={13} />
        <span class="front-count">{t.frontProperties(rows.length)}</span>
        {line && (
          <span class="front-line">
            {line.kind === "tags"
              ? line.tags.map((tag, index) => (
                  <span key={tag} class="front-inline-tag">
                    {index > 0 && <span class="front-sep" aria-hidden="true"> · </span>}
                    {tag}
                  </span>
                ))
              : line.text}
          </span>
        )}
        <span class="front-toggle">
          {front.description && <span>{t.frontSummary}</span>}
          <Icon name="CaretDown" size={12} />
        </span>
      </button>
    </section>
  );
}

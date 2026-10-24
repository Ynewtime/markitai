// A document's front matter above its body. Folded (the default): one light line, the tag
// icon, "Properties · N" and a chevron, no title (the body's own heading is the title). Open:
// the same line, its chevron turned up, over a two-column properties table. The line is one
// disclosure button with aria-expanded that stays in place (owner, 2026-10-25): opening or
// folding changes only what is under it. The choice lasts for the tab's session. The table's key
// column has one fixed width; a key or value stays on one line that scrolls sideways inside its own
// cell. A row whose value does not
// fit gets a chevron at its end: it (or a tap on the value) opens the row so the value wraps in
// full, and folds it again.
import { useEffect, useId, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { frontOpen, onFrontOpen, setFrontOpen } from "../lib/front-pref.ts";
import { frontRows, frontValue, type FrontRow } from "../lib/front-rows.ts";
import type { FrontMatter } from "../lib/frontmatter.ts";
import { overflows, watchScrollCells } from "../lib/scroll-cells.ts";
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

/** The keys of the rows whose folded value is wider than its cell. */
function longRows(table: HTMLElement): string {
  return [...table.querySelectorAll<HTMLElement>(".front-row")]
    .filter((row) => {
      const cell = row.querySelector<HTMLElement>(".front-cell");
      return cell !== null && !cell.classList.contains("is-open") && overflows(cell);
    })
    .map((row) => row.dataset.key ?? "")
    .join("\n");
}

export function FrontCard({ t, front, compact = false }: { t: Dict; front: FrontMatter; compact?: boolean }) {
  const [open, setOpen] = useFrontOpen();
  const id = useId();
  const table = useRef<HTMLDListElement>(null);
  // rows whose value overflows (measured again on every resize), and the rows opened to wrap in full
  const [long, setLong] = useState("");
  const [wide, setWide] = useState<ReadonlySet<string>>(new Set());
  useEffect(() => setWide(new Set()), [front]);
  useLayoutEffect(() => {
    const target = table.current;
    if (!open || !target) return;
    return watchScrollCells(target, ".front-key, .front-cell", () => setLong(longRows(target)));
  }, [open, front, t]);
  const toggle = (key: string) =>
    setWide((current) => {
      const next = new Set(current);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  const longKeys = new Set(long.split("\n").filter(Boolean));
  const rows = frontRows(front);
  const classes = ["front-meta", compact ? "is-compact" : "", open ? "is-open" : ""].filter(Boolean).join(" ");
  return (
    <section class={classes} aria-label={t.frontLabel}>
      <button type="button" class="front-head" aria-expanded={open} aria-controls={open ? id : undefined} onClick={() => setOpen(!open)}>
        <Icon name="Tag" size={13} />
        <span class="front-head-label">{t.frontProperties(rows.length)}</span>
        <Icon name="CaretDown" size={12} />
      </button>
      {open && (
        <dl id={id} ref={table} class="front-table">
          {rows.map((row) => {
            const label = t.frontKey(row.key);
            const opened = wide.has(row.key);
            // an opened row keeps its chevron to fold again
            const more = opened || longKeys.has(row.key);
            return (
              <div key={row.key} class={["front-row", more ? "has-more" : "", opened ? "is-open" : ""].filter(Boolean).join(" ")} data-key={row.key}>
                <dt>
                  <span class="front-key" title={label === row.key ? label : `${label} (${row.key})`}>
                    {label}
                  </span>
                </dt>
                <dd>
                  <div
                    class={opened ? "front-cell is-open" : "front-cell"}
                    onClick={(event) => {
                      // a link opens; selecting text or scrolling is not a tap
                      if (!more || (event.target as Element).closest("a") || getSelection()?.toString()) return;
                      toggle(row.key);
                    }}
                  >
                    <Value t={t} row={row} />
                  </div>
                  {more && (
                    <button type="button" class="front-more" aria-expanded={opened} aria-label={opened ? t.frontCollapseRow(label) : t.frontExpandRow(label)} title={opened ? t.frontCollapseRow(label) : t.frontExpandRow(label)} onClick={() => toggle(row.key)}>
                      <Icon name="CaretDown" size={12} />
                    </button>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
    </section>
  );
}

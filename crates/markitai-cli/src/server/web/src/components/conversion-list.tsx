// The conversions list (A3): a status glyph, the name and the time per row,
// nothing else. Actions live in the peek window, so rows hold no controls.
// It is an ARIA tree in its flat form: every row is a sibling, and a
// submission of several items is a level-1 row (what was submitted, preset,
// time) followed by its items at level 2; a one-item submission is a single
// level-1 row. Each row is named by its own text. Arrows walk the rows and
// Right/Left open and close a submission; a click, Enter or Space opens the
// window. Docked beside the list, the window follows the arrows instead.
import { memo } from "preact/compat";
import { useEffect, useRef, useState } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { isGroup, jobKey, navigationOrder, presetOf, submissionLabel, type Submission } from "../lib/conversions.ts";
import { fmtClock } from "../lib/format.ts";
import { rowWarnings } from "../lib/notices.ts";
import type { SessionItem } from "../lib/session.ts";
import { Duration, glyphOf, NameText, StatusGlyph, statusWord, useElapsed } from "./item-bits.tsx";
import { Icon } from "./icons.tsx";

export const rowId = (key: string): string => `row-${key.replace(/[^\w-]/g, "-")}`;

const Row = memo(function Row({
  t,
  item,
  selected,
  focusable,
  level,
  position,
  size,
}: {
  t: Dict;
  item: SessionItem;
  selected: boolean;
  focusable: boolean;
  level: number;
  position: number;
  size: number;
}) {
  const glyph = glyphOf(item);
  const warnings = rowWarnings(item.warnings).length;
  const elapsed = useElapsed(item.status === "running" ? item.startedAt : null);
  const right =
    glyph === "running" ? (
      elapsed === null ? null : <Duration t={t} ms={elapsed} />
    ) : glyph === "done" || glyph === "retained" ? (
      item.durationMs === null ? null : <Duration t={t} ms={item.durationMs} />
    ) : (
      statusWord(t, glyph)
    );
  return (
    <li
      id={rowId(item.key)}
      role="treeitem"
      aria-level={level}
      aria-posinset={position}
      aria-setsize={size}
      aria-selected={selected}
      tabIndex={focusable ? 0 : -1}
      data-key={item.key}
      class={`row is-${glyph}${level > 1 ? " is-child" : ""}${selected ? " is-selected" : ""}`}
    >
      <StatusGlyph glyph={glyph} label={statusWord(t, glyph)} />
      <span class="row-name">
        <NameText name={item.name} kind={item.kind} />
        {warnings > 0 && (
          <span class="warn-mark" title={t.itemWarnings(warnings)}>
            <Icon name="Warning" size={12} />
            <span class="sr-only">{t.itemWarnings(warnings)}</span>
            <span aria-hidden="true">{warnings}</span>
          </span>
        )}
      </span>
      <span class="row-side">{right}</span>
    </li>
  );
});

export function ConversionList({
  t,
  locale,
  list,
  collapsed,
  selected,
  labelWords,
  onSelect,
  onToggle,
  onOpen,
  followFocus = true,
}: {
  t: Dict;
  locale: Locale;
  list: Submission[];
  collapsed: ReadonlySet<string>;
  selected: string | null;
  labelWords: Parameters<typeof submissionLabel>[2];
  onSelect: (key: string, from: "pointer" | "keyboard") => void;
  onToggle: (jobId: string, open?: boolean) => void;
  onOpen: (key: string) => void;
  /** Selection follows the keyboard (a docked window beside the list). Off when
   * selecting opens a window: the arrows then only move, Enter or Space opens. */
  followFocus?: boolean;
}) {
  const root = useRef<HTMLUListElement>(null);
  const order = navigationOrder(list, collapsed);
  const [cursor, setCursor] = useState<string | null>(null);
  const focusKey =
    !followFocus && cursor !== null && order.includes(cursor)
      ? cursor
      : selected !== null && order.includes(selected)
        ? selected
        : (order[0] ?? null);
  useEffect(() => {
    if (selected !== null) setCursor(selected);
  }, [selected]);

  // A selection made elsewhere (the preview's previous/next) keeps its row in view.
  useEffect(() => {
    if (selected === null) return;
    document.getElementById(rowId(selected))?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);

  const move = (key: string | undefined) => {
    if (key === undefined) return;
    if (followFocus) onSelect(key, "keyboard");
    else setCursor(key);
    requestAnimationFrame(() => document.getElementById(rowId(key))?.focus({ preventScroll: false }));
  };

  const onKeyDown = (event: KeyboardEvent) => {
    const row = (event.target as Element | null)?.closest?.("[data-key]");
    const key = row?.getAttribute("data-key");
    if (!key || event.altKey || event.ctrlKey || event.metaKey) return;
    const index = order.indexOf(key);
    const group = key.startsWith("job:") ? key.slice(4) : null;
    const parent = list.find((submission) => isGroup(submission) && submission.shown.some((item) => item.key === key));
    switch (event.key) {
      case "ArrowDown":
        move(order[index + 1]);
        break;
      case "ArrowUp":
        move(order[index - 1]);
        break;
      case "Home":
        move(order[0]);
        break;
      case "End":
        move(order[order.length - 1]);
        break;
      case "ArrowRight":
        if (group === null) return;
        if (collapsed.has(group)) onToggle(group, true);
        else move(order[index + 1]);
        break;
      case "ArrowLeft":
        if (group !== null && !collapsed.has(group)) onToggle(group, false);
        else if (parent) move(jobKey(parent.job.jobId));
        else return;
        break;
      case "Enter":
        if (group !== null) onToggle(group);
        else if (followFocus) onOpen(key);
        else onSelect(key, "keyboard");
        break;
      case " ":
        onSelect(key, "keyboard");
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  return (
    <ul
      ref={root}
      class="conv-list"
      role="tree"
      aria-label={t.itemsAria}
      onKeyDown={onKeyDown}
      onClick={(event) => {
        const row = (event.target as Element | null)?.closest?.("[data-key]");
        const key = row?.getAttribute("data-key");
        if (!key) return;
        // A click on a submission's header row selects it; its items handle their own.
        if (key.startsWith("job:")) {
          if ((event.target as Element).closest(".group-caret")) {
            onToggle(key.slice(4));
            return;
          }
        }
        onSelect(key, "pointer");
      }}
      onDblClick={(event) => {
        const key = (event.target as Element | null)?.closest?.("[data-key]")?.getAttribute("data-key");
        if (key && !key.startsWith("job:")) onOpen(key);
      }}
    >
      {list.map((submission, index) => {
        const position = index + 1;
        if (!isGroup(submission)) {
          return submission.shown.map((item) => (
            <Row key={item.key} t={t} item={item} selected={selected === item.key} focusable={focusKey === item.key} level={1} position={position} size={list.length} />
          ));
        }
        const job = submission.job;
        const key = jobKey(job.jobId);
        const open = !collapsed.has(job.jobId);
        const label = submissionLabel(job, submission.items, labelWords);
        const preset = presetOf(job.options);
        const running = submission.items.filter((item) => item.status === "running" || item.status === "queued").length;
        const settled = submission.items.length - running;
        return [
          <li
            key={key}
            id={rowId(key)}
            role="treeitem"
            aria-level={1}
            aria-posinset={position}
            aria-setsize={list.length}
            aria-expanded={open}
            aria-selected={selected === key}
            tabIndex={focusKey === key ? 0 : -1}
            data-key={key}
            class={`group-head${selected === key ? " is-selected" : ""}`}
          >
            <span class="group-caret" aria-hidden="true">
              <Icon name={open ? "CaretDown" : "CaretRight"} size={11} />
            </span>
            <span class="group-label">
              <b>{label}</b>
              {preset && <span class="group-meta"> · {t.presetName(preset)}</span>}
              {running > 0 && <span class="sr-only"> · {t.groupProgress(settled, submission.items.length)}</span>}
            </span>
            <span class="group-side">{running > 0 ? `${settled}/${submission.items.length}` : fmtClock(job.finishedAt ?? job.createdAt, locale)}</span>
          </li>,
          ...(open
            ? submission.shown.map((item, child) => (
                <Row key={item.key} t={t} item={item} selected={selected === item.key} focusable={focusKey === item.key} level={2} position={child + 1} size={submission.shown.length} />
              ))
            : []),
        ];
      })}
    </ul>
  );
}

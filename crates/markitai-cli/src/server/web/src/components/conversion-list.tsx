// The conversions list (A3): a status glyph, the name and the time per row,
// nothing else. Actions live in the peek window, so rows hold no controls.
// It is an ARIA tree in its flat form: every row is a sibling, and a
// submission of several items is a level-1 row (what was submitted, preset,
// time) followed by its items at level 2; a one-item submission is a single
// level-1 row. Each row is named by its own text. Arrows walk the rows and
// Right/Left open and close a submission; a click, Enter or Space opens the
// window.
// Selection for deleting several (v1.13): every row and submission header has a
// box at its left edge (on phones only in Select mode). Each row then
// carries `aria-checked` (a header is "mixed" while part of its
// items are checked); Space checks the focused row, Shift extends from the last
// one checked, Ctrl/⌘+A checks everything listed, Escape clears and Delete asks
// to delete. Enter still opens. Items still converting cannot be checked.
// The filter row starts with a box over the boxes' column (SelectAll): it
// checks everything listed, or clears once everything is.
import { memo } from "preact/compat";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { checkState, deletable, isGroup, jobKey, navigationOrder, presetOf, submissionLabel, type CheckState, type Submission } from "../lib/conversions.ts";
import { fmtClock } from "../lib/format.ts";
import { groupNotices, rowWarnings, rowEntries } from "../lib/notices.ts";
import type { SessionItem } from "../lib/session.ts";
import { NameText } from "./file-name.tsx";
import { Duration, glyphOf, StatusGlyph, statusWord, useElapsed } from "./item-bits.tsx";
import { HelpTooltip } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";

export const rowId = (key: string): string => `row-${key.replace(/[^\w-]/g, "-")}`;

/** A row's box: null when nothing can be selected (a phone outside Select mode). */
export type Check = CheckState | null;

function CheckBox({ check, label }: { check: Check; label: string }) {
  if (check === null) return null;
  return (
    <span class={`row-check is-${check}`} aria-hidden="true" title={check === "locked" ? label : undefined}>
      {check === "on" && <Icon name="Check" size={11} />}
      {check === "mixed" && <span class="check-dash" />}
    </span>
  );
}

/** The box over the rows' boxes: a checkbox for everything listed, "mixed" while part of it is checked. */
export function SelectAll({ t, state, onToggle }: { t: Dict; state: CheckState; onToggle: () => void }) {
  const locked = state === "locked";
  return (
    <button
      type="button"
      role="checkbox"
      class={`row-check select-all is-${locked ? "off" : state}`}
      aria-checked={state === "mixed" ? "mixed" : state === "on"}
      aria-label={t.selectAllListed}
      title={t.selectAllListed}
      disabled={locked}
      onClick={onToggle}
    >
      {state === "on" && <Icon name="Check" size={11} />}
      {state === "mixed" && <span class="check-dash" />}
    </button>
  );
}

/** What a row's warning mark says on hover, focus or tap: its notices, in the words of the details view. */
function warningLines(t: Dict, item: SessionItem): string[] {
  const entries = rowEntries(item);
  return groupNotices(entries).map((group) => t.noticeTitle(group.kind, group.count, group.models, group.sample, group.pages));
}

const Row = memo(function Row({
  t,
  item,
  selected,
  focusable,
  level,
  position,
  size,
  check,
}: {
  t: Dict;
  item: SessionItem;
  selected: boolean;
  focusable: boolean;
  level: number;
  position: number;
  size: number;
  check: Check;
}) {
  const glyph = glyphOf(item);
  const shown = rowWarnings(item.warnings);
  const warnings = useMemo(() => warningLines(t, item), [t, item.warnings]);
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
      aria-checked={check === null || check === "locked" ? undefined : check === "on"}
      tabIndex={focusable ? 0 : -1}
      data-key={item.key}
      class={`row is-${glyph}${level > 1 ? " is-child" : ""}${selected ? " is-selected" : ""}${check === "on" ? " is-checked" : ""}${check !== null ? " has-check" : ""}`}
    >
      <CheckBox check={check} label={t.selectBusy} />
      <StatusGlyph glyph={glyph} label={statusWord(t, glyph)} />
      <span class="row-name">
        <NameText name={item.name} kind={item.kind} />
        {shown.length > 0 && (
          // A press here shows the warnings instead of opening the row.
          <span class="warn-wrap" onClick={(event) => event.stopPropagation()} onDblClick={(event) => event.stopPropagation()}>
            <HelpTooltip text={`${t.itemWarnings(shown.length)}: ${warnings.join(" ")}`} lines={warnings} disabled>
              {() => (
                <span class="warn-mark">
                  <Icon name="Warning" size={12} />
                  <span aria-hidden="true">{shown.length}</span>
                </span>
              )}
            </HelpTooltip>
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
  checked,
  selecting,
  tapChecks,
  onCheck,
  onCheckAll,
  onClearChecks,
  onDeleteChecked,
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
  /** Item keys checked for deleting. */
  checked: ReadonlySet<string>;
  /** Rows show their boxes. */
  selecting: boolean;
  /** A tap anywhere on a row checks it (a phone's Select mode) instead of opening it. */
  tapChecks: boolean;
  onCheck: (keys: string[], on: boolean) => void;
  onCheckAll: () => void;
  onClearChecks: () => void;
  onDeleteChecked: () => void;
}) {
  const root = useRef<HTMLUListElement>(null);
  const order = navigationOrder(list, collapsed);
  const [cursor, setCursor] = useState<string | null>(null);
  const focusKey =
    cursor !== null && order.includes(cursor)
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

  // The keys a row's box stands for: an item itself, or a submission's items that can be deleted.
  const itemsOf = (key: string): SessionItem[] => {
    if (key.startsWith("job:")) return list.find((submission) => isGroup(submission) && jobKey(submission.job.jobId) === key)?.shown ?? [];
    for (const submission of list) {
      const item = submission.shown.find((candidate) => candidate.key === key);
      if (item) return [item];
    }
    return [];
  };
  const checkOf = (items: SessionItem[]): Check => (selecting ? checkState(items, checked) : null);
  // Shift extends from the last row checked, over the rows in view order.
  const [anchor, setAnchor] = useState<string | null>(null);
  const toggle = (key: string, extend: boolean) => {
    const items = itemsOf(key).filter(deletable);
    if (!items.length) return;
    if (extend && anchor !== null && order.includes(anchor)) {
      const [from, to] = [order.indexOf(anchor), order.indexOf(key)].sort((a, b) => a - b);
      const span = order.slice(from, (to ?? from) + 1).flatMap((member) => itemsOf(member).filter(deletable));
      onCheck(
        span.map((item) => item.key),
        true,
      );
    } else {
      const state = checkOf(items);
      onCheck(
        items.map((item) => item.key),
        state !== "on",
      );
    }
    setAnchor(key);
  };

  const move = (key: string | undefined) => {
    if (key === undefined) return;
    setCursor(key);
    requestAnimationFrame(() => document.getElementById(rowId(key))?.focus({ preventScroll: false }));
  };

  const onKeyDown = (event: KeyboardEvent) => {
    const row = (event.target as Element | null)?.closest?.("[data-key]");
    const key = row?.getAttribute("data-key");
    if (!key || event.altKey) return;
    if (event.ctrlKey || event.metaKey) {
      if (selecting && event.key.toLowerCase() === "a" && !event.shiftKey) {
        event.preventDefault();
        onCheckAll();
      }
      return;
    }
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
        else onSelect(key, "keyboard");
        break;
      case " ":
        if (selecting) toggle(key, event.shiftKey);
        else onSelect(key, "keyboard");
        break;
      case "Escape":
        if (!checked.size) return;
        onClearChecks();
        break;
      case "Delete":
      case "Backspace":
        if (!checked.size) return;
        onDeleteChecked();
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  return (
    <ul
      ref={root}
      class={selecting ? "conv-list is-selecting" : "conv-list"}
      role="tree"
      aria-label={t.itemsAria}
      onKeyDown={onKeyDown}
      onClick={(event) => {
        const row = (event.target as Element | null)?.closest?.("[data-key]");
        const key = row?.getAttribute("data-key");
        if (!key) return;
        if (selecting && (tapChecks || (event.target as Element).closest(".row-check"))) {
          if (tapChecks && key.startsWith("job:") && (event.target as Element).closest(".group-caret")) {
            onToggle(key.slice(4));
            return;
          }
          toggle(key, event.shiftKey);
          return;
        }
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
        if (tapChecks || (event.target as Element).closest(".row-check")) return;
        const key = (event.target as Element | null)?.closest?.("[data-key]")?.getAttribute("data-key");
        if (key && !key.startsWith("job:")) onOpen(key);
      }}
    >
      {list.map((submission, index) => {
        const position = index + 1;
        if (!isGroup(submission)) {
          return submission.shown.map((item) => (
            <Row key={item.key} t={t} item={item} selected={selected === item.key} focusable={focusKey === item.key} level={1} position={position} size={list.length} check={checkOf([item])} />
          ));
        }
        const job = submission.job;
        const key = jobKey(job.jobId);
        const open = !collapsed.has(job.jobId);
        const label = submissionLabel(job, submission.items, labelWords);
        const preset = presetOf(job.options);
        const running = submission.items.filter((item) => item.status === "running" || item.status === "queued").length;
        const settled = submission.items.length - running;
        const check = checkOf(submission.shown);
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
            aria-checked={check === null || check === "locked" ? undefined : check === "mixed" ? "mixed" : check === "on"}
            tabIndex={focusKey === key ? 0 : -1}
            data-key={key}
            class={`group-head${selected === key ? " is-selected" : ""}${check !== null ? " has-check" : ""}`}
          >
            <CheckBox check={check} label={t.selectBusy} />
            <span class="group-caret" aria-hidden="true">
              <Icon name={open ? "CaretDown" : "CaretRight"} size={14} />
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
                <Row key={item.key} t={t} item={item} selected={selected === item.key} focusable={focusKey === item.key} level={2} position={child + 1} size={submission.shown.length} check={checkOf([item])} />
              ))
            : []),
        ];
      })}
    </ul>
  );
}

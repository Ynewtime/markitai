// The Conversions view: a summary, New conversion and "⋯" for the batch
// actions, filters (All / Failed / Notices and a name filter), and the list at
// the page's width. A click on a conversion opens the peek window (Details,
// Preview and Compare as tabs) over the list, or over the whole page at full
// screen; on phones the window fills the screen. Dropping files
// anywhere still converts; typing URLs is the Convert page's job.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { counts, FILTERS, isJobKey, itemOrder, submissionLabel, submissions, type ConvJob, type Filter, type LabelWords } from "../lib/conversions.ts";
import { groupNotices } from "../lib/notices.ts";
import type { PaneMode } from "../lib/route.ts";
import { waitingJobs, type SessionItem } from "../lib/session.ts";
import { ActionMenu } from "./action-menu.tsx";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { ConversionList, deletable } from "./conversion-list.tsx";
import { Icon } from "./icons.tsx";
import { ItemInspector, itemActs, JobInspector, jobActs, Overview, stopAct, type ActSet, type InspectorContext, type ItemActionsApi, type JobActionsApi } from "./inspector.tsx";
import type { IconName } from "./icon-paths.ts";
import { Peek, type PeekTab } from "./peek.tsx";
import { StateBlock, type StateAction } from "./state-block.tsx";

/** A panel's content: why the list or the inspector has nothing to show, and the way on. */
export interface PanelState {
  icon: IconName;
  title: string;
  text: string;
  primary?: StateAction;
  secondary?: StateAction;
  tone?: "error";
}

export interface ConversionsPageProps {
  t: Dict;
  locale: Locale;
  phone: boolean;
  jobs: Record<string, ConvJob>;
  items: SessionItem[];
  loaded: boolean;
  /** Why the saved conversions cannot be shown (no token, no server, an
   * error), with the way out; null while they load or once they have. */
  blocked: PanelState | null;
  /** The conversion a link named no longer exists. */
  gone: PanelState | null;
  selected: string | null;
  mode: PaneMode;
  filter: Filter;
  query: string;
  collapsed: ReadonlySet<string>;
  ctx: InspectorContext;
  itemActions: ItemActionsApi;
  jobActions: JobActionsApi;
  labelWords: LabelWords;
  head: { stopping: boolean; retrying: boolean; waiting: boolean; retryable: number; canZip: boolean; onStop: () => void; onRetry: () => void; onZip: () => void };
  compareAvailable: boolean;
  onSelect: (key: string | null, from?: "pointer" | "keyboard") => void;
  onMode: (mode: PaneMode) => void;
  onNewConversion: () => void;
  onFilter: (filter: Filter) => void;
  onQuery: (query: string) => void;
  onToggleGroup: (jobId: string, open?: boolean) => void;
  onShowItems: (keys: string[]) => void;
  /** Download all (.zip) for an item's Files, or null when this server cannot zip it. */
  downloadAll: (item: SessionItem) => (() => void) | null;
  onCopied: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
  /** Delete whole submissions and single items; resolves once every request has answered. */
  onDeleteMany: (jobIds: string[], items: SessionItem[]) => Promise<void>;
}

function summaryText(t: Dict, items: SessionItem[]): string {
  const tally = counts(items);
  if (!tally.total) return t.summaryEmpty;
  if (tally.running || tally.queued) return t.summaryRunning(tally.done, tally.total, tally.running, tally.queued);
  return t.summaryDone(tally.done, tally.total, tally.failed, tally.stopped, tally.skipped);
}

const TAB_MODE: Record<PeekTab, PaneMode> = { details: "inspector", preview: "preview", compare: "compare" };

export function ConversionsPage(props: ConversionsPageProps) {
  const { t, locale, phone, jobs, items, loaded, selected, mode, filter, query, collapsed, ctx } = props;
  const list = useMemo(() => submissions(jobs, items, filter, query), [jobs, items, filter, query]);
  const order = useMemo(() => itemOrder(list), [list]);
  const tally = useMemo(() => counts(items), [items]);
  const kinds = useMemo(() => groupNotices(items.flatMap((item) => item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text })))).length, [items]);
  const selectedItem = selected !== null && !isJobKey(selected) ? (items.find((item) => item.key === selected) ?? null) : null;
  const selectedJobId = selected !== null && isJobKey(selected) ? selected.slice(4) : null;
  const selectedJob = selectedJobId !== null ? (jobs[selectedJobId] ?? null) : null;
  const jobItems = selectedJob ? items.filter((item) => item.jobId === selectedJob.jobId) : [];
  const position = selectedItem ? order.findIndex((item) => item.key === selectedItem.key) : -1;
  const step = (delta: number) => {
    const next = order[position + delta];
    return next ? () => props.onSelect(next.key, "keyboard") : null;
  };
  // The summary of everything listed opens in the same window as a conversion.
  const [overview, setOverview] = useState(false);
  useEffect(() => {
    const leave = () => setOverview(false);
    window.addEventListener("popstate", leave);
    return () => window.removeEventListener("popstate", leave);
  }, []);

  /** Close the window and give the keyboard back to the row it was opened from. */
  const close = () => {
    const key = selected;
    // The summary's window has a history entry of its own (Back closes it): closing by hand goes back over it.
    if (overview && (history.state as { wb?: string } | null)?.wb === "summary") history.back();
    setOverview(false);
    props.onSelect(null);
    if (key) requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-key="${CSS.escape(key)}"]`)?.focus({ preventScroll: true }));
  };

  // ---- selection for deleting several: item keys; only items listed and settled count
  const [checked, setChecked] = useState<ReadonlySet<string>>(() => new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const deleteButton = useRef<HTMLButtonElement>(null);
  const selecting = !phone || selectMode;
  const listed = useMemo(() => list.flatMap((submission) => submission.shown).filter(deletable), [list]);
  const chosen = useMemo(() => listed.filter((item) => checked.has(item.key)), [listed, checked]);
  // Keys of items that are gone (deleted, or retried into running) drop out.
  useEffect(() => {
    if (!checked.size) return;
    const live = new Set(items.filter(deletable).map((item) => item.key));
    if ([...checked].some((key) => !live.has(key))) setChecked(new Set([...checked].filter((key) => live.has(key))));
  }, [items]);
  useEffect(() => {
    if (!phone) setSelectMode(false);
  }, [phone]);
  const check = (keys: string[], on: boolean) =>
    setChecked((previous) => {
      const next = new Set(previous);
      for (const key of keys) {
        if (on) next.add(key);
        else next.delete(key);
      }
      return next;
    });
  const clearChecks = () => setChecked(new Set());
  const checkAll = () => setChecked(new Set(listed.map((item) => item.key)));
  const leaveSelect = () => {
    setSelectMode(false);
    clearChecks();
  };
  /** Whole submissions go as one request each; the rest item by item. */
  const deleteChosen = async () => {
    const byJob = new Map<string, SessionItem[]>();
    for (const item of chosen) byJob.set(item.jobId, [...(byJob.get(item.jobId) ?? []), item]);
    const jobIds: string[] = [];
    const singles: SessionItem[] = [];
    for (const [jobId, members] of byJob) {
      if (items.filter((item) => item.jobId === jobId).length === members.length) jobIds.push(jobId);
      else singles.push(...members);
    }
    await props.onDeleteMany(jobIds, singles);
    clearChecks();
    if (phone) setSelectMode(false);
    return true;
  };

  const counters: Record<Filter, number> = { all: tally.total, failed: tally.failed, notices: tally.notices };
  const filterLabels: Record<Filter, string> = { all: t.filterAll, failed: t.filterFailed, notices: t.filterNotices };
  const tab: PeekTab = mode === "compare" ? "compare" : mode === "preview" ? "preview" : "details";
  const url = selectedItem && selectedItem.kind === "url" && /^https?:\/\//i.test(selectedItem.name) ? selectedItem.name : null;

  // The full-screen window on a phone covers the list's Stop: an item's window
  // offers its submission's, and the overview the list's.
  const acts = (): ActSet | null => {
    if (selectedItem) {
      const own = itemActs(ctx, selectedItem, props.itemActions);
      const waiting = ctx.cancel && waitingJobs(items, jobs).includes(selectedItem.jobId);
      return waiting ? { ...own, shown: [...own.shown, stopAct(ctx, selectedItem.jobId, props.jobActions)] } : own;
    }
    if (selectedJob) return jobActs(ctx, selectedJob, jobItems, props.jobActions);
    if (!props.head.waiting || !ctx.cancel) return null;
    const stopping = props.head.stopping;
    return { shown: [{ key: "stop", label: t.stopRemaining, title: t.stopTitle, icon: "Stop", keep: true, disabled: stopping, busy: stopping, run: props.head.onStop }], more: [], danger: null };
  };
  const peekOpen = selectedItem !== null || selectedJob !== null || (overview && items.length > 0);
  const peek = peekOpen && (
    <Peek
      t={t}
      locale={locale}
      item={selectedItem}
      title={selectedItem ? selectedItem.name : selectedJob ? submissionLabel(selectedJob, jobItems, props.labelWords) : t.overviewTitle}
      createdAt={selectedItem ? (jobs[selectedItem.jobId]?.createdAt ?? null) : null}
      position={selectedItem ? { index: Math.max(0, position), total: order.length } : null}
      tab={tab}
      phone={phone}
      compareAvailable={props.compareAvailable}
      originals={props.compareAvailable}
      onDownloadAll={selectedItem ? props.downloadAll(selectedItem) : null}
      acts={acts()}
      url={url}
      failure={
        ctx.failure && [selectedItem?.key, `job:${selectedJob?.jobId ?? selectedItem?.jobId ?? ""}`].includes(ctx.failure.key) ? ctx.failure : null
      }
      details={
        selectedItem ? (
          <ItemInspector ctx={ctx} item={selectedItem} job={jobs[selectedItem.jobId]} actions={props.itemActions} peek headingId="peek-title" />
        ) : selectedJob ? (
          <JobInspector ctx={ctx} job={selectedJob} items={jobItems} actions={props.jobActions} peek headingId="peek-title" onShowItems={props.onShowItems} />
        ) : (
          <Overview ctx={ctx} items={items} peek onShowItems={props.onShowItems} />
        )
      }
      onTab={(next) => props.onMode(TAB_MODE[next])}
      onPrevious={selectedItem ? step(-1) : null}
      onNext={selectedItem ? step(1) : null}
      onClose={close}
      onCopied={props.onCopied}
      onCopyFailed={() => selectedItem && props.itemActions.copyFailed(selectedItem)}
      announce={props.announce}
      describe={props.describe}
    />
  );

  const empty = loaded && items.length === 0;
  const unloaded = !loaded && items.length === 0;
  const clearFilter = { label: t.clearFilter, run: () => (props.onQuery(""), props.onFilter("all")) };
  const noMatch = query.trim() ? t.filterNoMatchQuery(query.trim()) : filter === "failed" ? t.filterNoFailed : filter === "notices" ? t.filterNoNotices : t.filterNoMatch;
  const gone = props.gone && selected === null ? <StateBlock {...props.gone} /> : null;

  return (
    <div class="page conversions">
      <header class="page-head conv-head">
        <div class="conv-title">
          <h1>{t.conversions}</h1>
          {!unloaded && !empty && (
            <p class="conv-summary" aria-live="polite">
              {summaryText(t, items)}
              {(kinds > 0 || tally.failed > 0) && (
                <>
                  {" · "}
                  <button type="button" class="text-link" onClick={() => (props.onSelect(null), history.pushState({ wb: "summary" }, "", location.href), setOverview(true))}>
                    {t.overviewLink}
                  </button>
                </>
              )}
            </p>
          )}
        </div>
        {/* Beside New conversion; on a phone, a row of its own under the status. */}
        {props.head.waiting && ctx.cancel && (
          <button type="button" class="btn head-stop" title={t.stopTitle} disabled={props.head.stopping} aria-busy={props.head.stopping || undefined} onClick={props.head.onStop}>
            <Icon name="Stop" size={14} />
            {t.stopRemaining}
          </button>
        )}
        <div class="head-actions">
          <button type="button" class="btn" onClick={props.onNewConversion}>
            <Icon name="Plus" size={14} />
            {t.newConversion}
          </button>
          {items.length > 0 && (
            <ActionMenu
              label={t.moreActions}
              actions={[
                ...(props.head.retryable > 0
                  ? [{ key: "retry", label: t.retryFailedN(props.head.retryable), icon: "ArrowCounterClockwise" as const, run: props.head.onRetry, disabled: props.head.retrying }]
                  : []),
                { key: "zip", label: t.downloadAll, icon: "DownloadSimple" as const, run: props.head.onZip, disabled: !props.head.canZip },
              ]}
            />
          )}
        </div>
      </header>
      {items.length > 0 && (
        <div class="toolbar">
          <label class="search">
            <Icon name="MagnifyingGlass" size={14} />
            <input type="search" value={query} placeholder={t.filterByName} aria-label={t.filterAria} onInput={(event) => props.onQuery(event.currentTarget.value)} />
          </label>
          <div class="chips" role="group" aria-label={t.filterStatusAria}>
            {FILTERS.map((value) => (
              <button key={value} type="button" class={filter === value ? "filter-chip is-on" : "filter-chip"} aria-pressed={filter === value} onClick={() => props.onFilter(value)}>
                {filterLabels[value]}
                <span class="chip-count">{counters[value]}</span>
              </button>
            ))}
          </div>
          {/* Phones check rows in Select mode; wide screens always show the boxes. */}
          {phone && listed.length > 0 && (
            <button type="button" class="btn select-mode-btn" aria-pressed={selectMode} onClick={() => (selectMode ? leaveSelect() : setSelectMode(true))}>
              {selectMode ? t.done : t.selectMode}
            </button>
          )}
        </div>
      )}
      <div class="split is-single">
        <div class="list-col">
          {gone}
          {empty ? (
            <StateBlock icon="Tray" title={t.emptyTitle} text={ctx.retentionDays ? t.emptyTextKept(ctx.retentionDays) : t.emptyText} primary={{ label: t.newConversion, run: props.onNewConversion }} />
          ) : unloaded && props.blocked ? (
            <StateBlock {...props.blocked} />
          ) : unloaded ? (
            <p class="pane-note">{t.loading}</p>
          ) : list.length === 0 ? (
            <StateBlock icon="MagnifyingGlass" title={t.filterNoneTitle} text={noMatch} secondary={clearFilter} />
          ) : (
            <ConversionList
              t={t}
              locale={locale}
              list={list}
              collapsed={collapsed}
              selected={selected}
              labelWords={props.labelWords}
              onSelect={(key, from) => props.onSelect(key, from)}
              onToggle={props.onToggleGroup}
              onOpen={(key) => {
                const item = items.find((candidate) => candidate.key === key);
                if (item) props.itemActions.preview(item);
              }}
              checked={checked}
              selecting={selecting}
              tapChecks={phone && selectMode}
              onCheck={check}
              onCheckAll={checkAll}
              onClearChecks={clearChecks}
              onDeleteChecked={() => chosen.length > 0 && setConfirming(true)}
            />
          )}
          {selecting && list.length > 0 && (chosen.length > 0 || selectMode) && (
            <div class="select-bar" role="region" aria-label={t.selectionAria}>
              <span class="select-count" role="status">
                {t.selectedCount(chosen.length)}
              </span>
              <button type="button" class="select-link" disabled={chosen.length === listed.length} onClick={checkAll}>
                {t.selectAllN(listed.length)}
              </button>
              <button type="button" class="select-link" disabled={chosen.length === 0} onClick={clearChecks}>
                {t.clearSelection}
              </button>
              <button
                ref={deleteButton}
                type="button"
                class="btn btn-danger select-delete"
                disabled={chosen.length === 0}
                aria-haspopup="dialog"
                aria-expanded={confirming}
                onClick={() => setConfirming(true)}
              >
                <Icon name="Trash" size={14} />
                {t.deleteSelected}
              </button>
              <ConfirmPopover
                triggerLabel={t.deleteSelected}
                title={t.deleteManyTitle(chosen.length)}
                description={t.deleteManyText(chosen.length)}
                confirmLabel={t.deleteSelected}
                cancelLabel={t.cancel}
                busyLabel={t.deleting}
                anchor={() => deleteButton.current}
                open={confirming && chosen.length > 0}
                onClose={() => setConfirming(false)}
                onConfirm={deleteChosen}
              />
            </div>
          )}
        </div>
      </div>
      {peek}
    </div>
  );
}

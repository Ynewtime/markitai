// The Conversions view: a summary, New conversion and "⋯" for the batch
// actions, filters (All / Failed / Notices and a name filter), and the list at
// the page's width. A click on a conversion opens the peek window (Details,
// Preview and Compare as tabs) over the list, or docked beside it when this
// browser prefers that; on phones the window fills the screen. Dropping files
// anywhere still converts; typing URLs is the Convert page's job.
import { useMemo, useState } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { counts, FILTERS, isJobKey, itemOrder, submissionLabel, submissions, type ConvJob, type Filter, type LabelWords } from "../lib/conversions.ts";
import { groupNotices } from "../lib/notices.ts";
import type { PaneMode } from "../lib/route.ts";
import type { SessionItem } from "../lib/session.ts";
import { ActionMenu } from "./action-menu.tsx";
import { ConversionList } from "./conversion-list.tsx";
import { Icon } from "./icons.tsx";
import { ItemInspector, itemActs, JobInspector, jobActs, Overview, type InspectorContext, type ItemActionsApi, type JobActionsApi } from "./inspector.tsx";
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
  /** This browser docks the window beside the list instead of over it. */
  docked: boolean;
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
  onDock: () => void;
  onNewConversion: () => void;
  onFilter: (filter: Filter) => void;
  onQuery: (query: string) => void;
  onToggleGroup: (jobId: string, open?: boolean) => void;
  onShowItems: (keys: string[]) => void;
  onDownload: (item: SessionItem, path: string) => void;
  onCopied: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}

/** What a click may land on without being a click on the page's empty space. */
const NOT_EMPTY = "button, a, input, textarea, select, label, summary, [role=tree], [role=treeitem], .conv-list, .peek, .page-head, .toolbar, .state, .menu, .confirm-card";

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
  const docked = props.docked && !phone;

  /** Close the window and give the keyboard back to the row it was opened from. */
  const close = () => {
    const key = selected;
    setOverview(false);
    props.onSelect(null);
    if (key) requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-key="${CSS.escape(key)}"]`)?.focus({ preventScroll: true }));
  };

  // Docked beside the list, a click on the page's empty space (outside the list
  // and the window) lets go of the selection, as Escape does in the list.
  // Ending a text selection is not such a click.
  const canRelease = docked && (selected !== null || overview);
  const releaseOnEmpty = (event: MouseEvent) => {
    if (!canRelease || event.button !== 0) return;
    if ((event.target as Element | null)?.closest?.(NOT_EMPTY)) return;
    if (String(window.getSelection?.() ?? "").trim()) return;
    close();
  };
  const releaseOnEscape = (event: KeyboardEvent) => {
    if (!canRelease || event.key !== "Escape" || event.defaultPrevented) return;
    if (!(event.target as Element | null)?.closest?.(".conv-list")) return;
    if (document.querySelector(".menu, .confirm-card, .veil")) return;
    event.preventDefault();
    close();
  };

  const counters: Record<Filter, number> = { all: tally.total, failed: tally.failed, notices: tally.notices };
  const filterLabels: Record<Filter, string> = { all: t.filterAll, failed: t.filterFailed, notices: t.filterNotices };
  const tab: PeekTab = mode === "compare" ? "compare" : mode === "preview" ? "preview" : "details";
  const url = selectedItem && selectedItem.kind === "url" && /^https?:\/\//i.test(selectedItem.name) ? selectedItem.name : null;

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
      docked={docked}
      compareAvailable={props.compareAvailable}
      acts={selectedItem ? itemActs(ctx, selectedItem, props.itemActions) : selectedJob ? jobActs(ctx, selectedJob, jobItems, props.jobActions) : null}
      url={url}
      failure={ctx.failure && (ctx.failure.key === selectedItem?.key || ctx.failure.key === `job:${selectedJob?.jobId ?? ""}`) ? ctx.failure : null}
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
      onDock={props.onDock}
      onClose={close}
      onDownloadPath={(path) => selectedItem && props.onDownload(selectedItem, path)}
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
  const side = docked && peekOpen;

  return (
    <div class={side ? "page conversions is-docked" : "page conversions"} onClick={releaseOnEmpty} onKeyDown={releaseOnEscape}>
      <header class="page-head conv-head">
        <div class="conv-title">
          <h1>{t.conversions}</h1>
          {!unloaded && !empty && (
            <p class="conv-summary" aria-live="polite">
              {summaryText(t, items)}
              {(kinds > 0 || tally.failed > 0) && (
                <>
                  {" · "}
                  <button type="button" class="text-link" onClick={() => (props.onSelect(null), setOverview(true))}>
                    {t.overviewLink}
                  </button>
                </>
              )}
            </p>
          )}
        </div>
        <div class="head-actions">
          {props.head.waiting && ctx.cancel && (
            <button type="button" class="btn" title={t.stopTitle} disabled={props.head.stopping} aria-busy={props.head.stopping || undefined} onClick={props.head.onStop}>
              <Icon name="Stop" size={14} />
              {t.stopRemaining}
            </button>
          )}
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
        </div>
      )}
      <div class={side ? "split has-peek" : "split is-single"}>
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
              followFocus={side}
              onSelect={(key, from) => props.onSelect(key, from)}
              onToggle={props.onToggleGroup}
              onOpen={(key) => {
                const item = items.find((candidate) => candidate.key === key);
                if (item) props.itemActions.preview(item);
              }}
            />
          )}
        </div>
        {side && <div class="side-col">{peek}</div>}
      </div>
      {!side && peek}
    </div>
  );
}

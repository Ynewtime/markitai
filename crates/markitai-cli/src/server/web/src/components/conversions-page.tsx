// The Conversions view: a summary with the batch actions, the compact
// composer, filters (All / Failed / Notices and a name filter), and the list
// beside the inspector (A3). Opening a preview docks it beside the list (B3);
// Compare turns it into the original beside the Markdown (B2). On phones the
// inspector is a bottom sheet and the preview a full-screen view.
import type { ComponentChildren } from "preact";
import { useEffect, useMemo } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { counts, FILTERS, isJobKey, itemOrder, submissionLabel, submissions, type ConvJob, type Filter, type LabelWords } from "../lib/conversions.ts";
import type { PaneMode } from "../lib/route.ts";
import type { SessionItem } from "../lib/session.ts";
import { ConversionList } from "./conversion-list.tsx";
import { Icon } from "./icons.tsx";
import { ItemActions, ItemInspector, JobActions, JobInspector, Overview, type InspectorContext, type ItemActionsApi, type JobActionsApi } from "./inspector.tsx";
import { PreviewPane } from "./preview-pane.tsx";
import { Sheet } from "./sheet.tsx";
import type { IconName } from "./icon-paths.ts";
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
  composer: ComponentChildren;
  selected: string | null;
  mode: PaneMode;
  wide: boolean;
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
  onWide: () => void;
  onFilter: (filter: Filter) => void;
  onQuery: (query: string) => void;
  onToggleGroup: (jobId: string, open?: boolean) => void;
  onShowItems: (keys: string[]) => void;
  onDownload: (item: SessionItem, path: string) => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}

function summaryText(t: Dict, items: SessionItem[]): string {
  const tally = counts(items);
  if (!tally.total) return t.summaryEmpty;
  if (tally.running || tally.queued) return t.summaryRunning(tally.done, tally.total, tally.running, tally.queued);
  return t.summaryDone(tally.done, tally.total, tally.failed, tally.stopped, tally.skipped);
}

export function ConversionsPage(props: ConversionsPageProps) {
  const { t, locale, phone, jobs, items, loaded, selected, mode, wide, filter, query, collapsed, ctx } = props;
  const list = useMemo(() => submissions(jobs, items, filter, query), [jobs, items, filter, query]);
  const order = useMemo(() => itemOrder(list), [list]);
  const tally = useMemo(() => counts(items), [items]);
  const selectedItem = selected !== null && !isJobKey(selected) ? (items.find((item) => item.key === selected) ?? null) : null;
  const selectedJobId = selected !== null && isJobKey(selected) ? selected.slice(4) : null;
  const selectedJob = selectedJobId !== null ? (jobs[selectedJobId] ?? null) : null;
  const jobItems = selectedJob ? items.filter((item) => item.jobId === selectedJob.jobId) : [];
  const position = selectedItem ? order.findIndex((item) => item.key === selectedItem.key) : -1;
  const step = (delta: number) => {
    const next = order[position + delta];
    return next ? () => props.onSelect(next.key, "keyboard") : null;
  };
  const previewing = selectedItem !== null && mode !== "inspector" && selectedItem.status === "done" && selectedItem.output !== null && !selectedItem.skipped;
  // An opened preview fills the window beside the list: bring the split to the top.
  useEffect(() => {
    if (!previewing || phone) return;
    const split = document.querySelector<HTMLElement>(".split");
    if (split && split.getBoundingClientRect().top > 24) split.scrollIntoView({ block: "start" });
  }, [previewing, phone]);

  const counters: Record<Filter, number> = { all: tally.total, failed: tally.failed, notices: tally.notices };
  const filterLabels: Record<Filter, string> = { all: t.filterAll, failed: t.filterFailed, notices: t.filterNotices };

  const preview = previewing && selectedItem && (
    <PreviewPane
      t={t}
      locale={locale}
      item={selectedItem}
      createdAt={jobs[selectedItem.jobId]?.createdAt ?? null}
      position={{ index: Math.max(0, position), total: order.length }}
      mode={mode === "compare" ? "compare" : "preview"}
      wide={wide}
      phone={phone}
      compareAvailable={props.compareAvailable}
      onPrevious={step(-1)}
      onNext={step(1)}
      onClose={() => props.onMode("inspector")}
      onMode={props.onMode}
      onWide={props.onWide}
      onDownload={props.onDownload}
      onCopyFailed={props.itemActions.copyFailed}
      failure={ctx.failure}
      announce={props.announce}
      describe={props.describe}
    />
  );

  const inspector = selectedItem ? (
    <ItemInspector ctx={ctx} item={selectedItem} job={jobs[selectedItem.jobId]} actions={props.itemActions} sheet={phone} headingId={phone ? "sheet-title" : "insp-title"} />
  ) : selectedJob ? (
    <JobInspector ctx={ctx} job={selectedJob} items={jobItems} actions={props.jobActions} sheet={phone} headingId={phone ? "sheet-title" : "insp-title"} onShowItems={props.onShowItems} />
  ) : (
    <Overview ctx={ctx} items={items} onShowItems={props.onShowItems} />
  );

  const empty = loaded && items.length === 0;
  const unloaded = !loaded && items.length === 0;
  const clearFilter = { label: t.clearFilter, run: () => (props.onQuery(""), props.onFilter("all")) };
  const noMatch = query.trim() ? t.filterNoMatchQuery(query.trim()) : filter === "failed" ? t.filterNoFailed : filter === "notices" ? t.filterNoNotices : t.filterNoMatch;
  const gone = props.gone && selected === null ? <StateBlock {...props.gone} /> : null;
  const body = (
    <div class={`split${previewing && !phone ? " has-preview" : ""}${previewing && !phone && (wide || mode === "compare") ? " is-wide" : ""}${empty || unloaded ? " is-single" : ""}`}>
      <div class="list-col">
        {phone && gone}
        {empty ? (
          <StateBlock icon="Tray" title={t.emptyTitle} text={ctx.retentionDays ? t.emptyTextKept(ctx.retentionDays) : t.emptyText} />
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
          />
        )}
      </div>
      {/* With nothing to inspect, the list column says it once. */}
      {!phone && !empty && !unloaded && <div class="side-col">{previewing ? preview : (gone ?? inspector)}</div>}
    </div>
  );

  return (
    <div class="page conversions">
      <header class="page-head conv-head">
        <div class="conv-title">
          <h1>{t.conversions}</h1>
          <p class="conv-summary" aria-live="polite">
            {unloaded || empty ? "" : summaryText(t, items)}
          </p>
        </div>
        <div class="head-actions">
          {props.head.waiting && ctx.cancel && (
            <button type="button" class="btn" title={t.stopTitle} disabled={props.head.stopping} aria-busy={props.head.stopping || undefined} onClick={props.head.onStop}>
              <Icon name="Stop" size={14} />
              {t.stopRemaining}
            </button>
          )}
          {props.head.retryable > 0 && (
            <button type="button" class="btn" disabled={props.head.retrying} aria-busy={props.head.retrying || undefined} onClick={props.head.onRetry}>
              <Icon name="ArrowCounterClockwise" size={14} />
              {t.retryFailedN(props.head.retryable)}
            </button>
          )}
          {items.length > 0 && (
            <button type="button" class="btn" disabled={!props.head.canZip} title={props.head.canZip ? undefined : t.zipWhileRunning} onClick={props.head.onZip}>
              <Icon name="DownloadSimple" size={14} />
              {t.downloadAll}
            </button>
          )}
        </div>
      </header>
      <div class="conv-composer">{props.composer}</div>
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
      {body}
      {phone && previewing && preview}
      {phone && !previewing && selected !== null && (selectedItem || selectedJob) && (
        <Sheet
          t={t}
          titleId="sheet-title"
          title={
            <h2 id="sheet-title" class="sheet-heading">
              {selectedItem ? selectedItem.name : selectedJob ? submissionLabel(selectedJob, jobItems, props.labelWords) : ""}
            </h2>
          }
          onClose={() => props.onSelect(null)}
          onPrevious={selectedItem ? step(-1) : null}
          onNext={selectedItem ? step(1) : null}
          actions={
            selectedItem ? (
              <ItemActions ctx={ctx} item={selectedItem} actions={props.itemActions} sheet />
            ) : selectedJob ? (
              <JobActions ctx={ctx} job={selectedJob} items={jobItems} actions={props.jobActions} sheet />
            ) : null
          }
        >
          {inspector}
        </Sheet>
      )}
    </div>
  );
}

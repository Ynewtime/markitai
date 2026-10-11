// The inspector (A3): what one conversion is, in plain words. For an item:
// its state, actions, facts (source, output, models, cost, when), notices
// grouped by kind, and the start of the result. For a submission: its
// progress, batch actions, facts and the notices of all its items (A4). With
// nothing selected: an overview of everything listed. The actions render as a
// row here, or in the bottom sheet's pinned bar on phones.
import type { ComponentChildren } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { fetchBlob, filePath, imagePath } from "../api/client.ts";
import { hasToken } from "../api/token.ts";
import type { ItemResult } from "../api/types.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { itemErrorText, persistenceText } from "../i18n/errors.ts";
import { counts, isFailed, isStopped, presetOf, submissionLabel, type ConvJob, type LabelWords } from "../lib/conversions.ts";
import { costState, fmtUsd, itemCost, totalCost, type CostState } from "../lib/cost.ts";
import { basename, fmtBytes, fmtDateTime, fmtDay, localeTag, timestampMs } from "../lib/format.ts";
import { parseFrontMatter } from "../lib/frontmatter.ts";
import { loadArtifactImages, renderMarkdown, splitFrontmatter, type ImageLoads } from "../lib/markdown.ts";
import { groupNotices, type NoticeGroup } from "../lib/notices.ts";
import { cachedExcerpt, loadExcerpt, resultKey } from "../lib/results.ts";
import { canRetry, isPreviewable, type SessionItem } from "../lib/session.ts";
import { ActionMenu, type MenuDanger } from "./action-menu.tsx";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { FileName } from "./file-name.tsx";
import { Duration, glyphOf, StatusGlyph, statusWord, useElapsed } from "./item-bits.tsx";
import type { IconName } from "./icon-paths.ts";
import { Icon } from "./icons.tsx";
import { Skeleton } from "./skeleton.tsx";

type Described = { text: string; detail: string };

export interface ItemActionsApi {
  preview(item: SessionItem): void;
  compare(item: SessionItem): void;
  download(item: SessionItem, path: string): void;
  copyFailed(item: SessionItem): void;
  retry(item: SessionItem): void;
  retryPlain(item: SessionItem): void;
  retryWithOcr(item: SessionItem): void;
  enhance(item: SessionItem): void;
  remove(item: SessionItem): Promise<boolean>;
  openModels(): void;
}

export interface JobActionsApi {
  zip(jobId: string): void;
  retryFailed(jobId: string): void;
  stop(jobId: string): void;
  remove(jobId: string): Promise<boolean>;
  select(key: string): void;
}

/** An action on this conversion that failed (Retry, Delete, Download, Enhance,
 * Copy): one sentence, and the service's full wording behind Full details. */
export interface ActionFailure {
  key: string;
  text: string;
  detail: string;
}

/** The failed action, in the style of the conversion's own failure reason: a
 * red ✕, one sentence, Full details. No box: it reads as part of the inspector. */
export function ActionError({ t, failure }: { t: Dict; failure: ActionFailure | null }) {
  if (!failure) return null;
  return (
    <div class="action-error">
      <p class="action-error-line">
        <StatusGlyph glyph="failed" />
        <span>{failure.text}</span>
      </p>
      {failure.detail && (
        <details class="notice-details">
          <summary>{t.notificationDetails}</summary>
          <p class="raw">{failure.detail}</p>
        </details>
      )}
    </div>
  );
}

/** A lasting condition of this conversion (it ran without a model; its history
 * could not be saved): one line in the style of the failure reason, with at
 * most one link. */
function ConditionLine({ tone, text, action }: { tone: "info" | "error"; text: string; action?: { label: string; run: () => void } }) {
  return (
    <p class={`condition-line is-${tone}`}>
      {tone === "error" ? (
        <StatusGlyph glyph="failed" />
      ) : (
        <span class="condition-mark" aria-hidden="true">
          <Icon name="Info" size={14} />
        </span>
      )}
      <span>
        {text}
        {action && (
          <>
            {" "}
            <button type="button" class="inline-action" onClick={action.run}>
              {action.label}
            </button>
          </>
        )}
      </span>
    </p>
  );
}

export interface InspectorContext {
  t: Dict;
  locale: Locale;
  llmReady: boolean;
  excerpts: boolean;
  compare: boolean;
  cancel: boolean;
  retentionDays: number | null;
  docsUrl: string;
  busy: ReadonlySet<string>;
  labelWords: LabelWords;
  describe: (error: unknown) => Described;
  /** The selected conversion's failed action, if any. */
  failure: ActionFailure | null;
}

const docsPage = (docsUrl: string, page: string) => (/\/index\.md$/.test(docsUrl) ? docsUrl.replace(/index\.md$/, `${page}.md`) : docsUrl);

export function costText(t: Dict, state: CostState): string {
  switch (state.kind) {
    case "none":
      return t.costNone;
    case "complete":
      return fmtUsd(state.usd);
    case "partial":
      return t.costPartial(fmtUsd(state.usd), state.unpriced);
    default:
      return state.usd !== null && state.usd > 0 ? t.costUnknownKnown(fmtUsd(state.usd)) : t.costUnknown;
  }
}

function keptUntil(ctx: InspectorContext, finishedAt: string | null): string | null {
  const ms = timestampMs(finishedAt);
  if (ctx.retentionDays === null || ms === null) return null;
  return fmtDay(ms + ctx.retentionDays * 86_400_000, ctx.locale);
}

function optionsText(t: Dict, item: { options?: SessionItem["options"] | null }): string {
  const options = item.options;
  if (!options) return "";
  const parts: string[] = [];
  const preset = presetOf(options);
  if (preset) parts.push(t.presetName(preset));
  if (options.llm === true) parts.push("LLM");
  if (options.ocr === true) parts.push("OCR");
  if (options.screenshot === true) parts.push(t.advScreenshot);
  return parts.join(" · ");
}

/** The status sentence under an item's name. */
function itemLine(ctx: InspectorContext, item: SessionItem, elapsed: number | null, images: number): ComponentChildren {
  const { t, locale } = ctx;
  const glyph = glyphOf(item);
  if (glyph === "running") return elapsed === null ? t.statusRunning : <>{t.convertingFor} <Duration t={t} ms={elapsed} /></>;
  if (glyph === "queued") return t.waitingSlot;
  if (glyph === "stopped") return t.stoppedLine;
  if (glyph === "skipped") return item.skipReason === "image_only" ? t.skipImageOnly : t.skippedLine(item.skipReason ?? "");
  if (glyph === "failed") return itemErrorText(locale, { error: item.error, error_code: item.errorCode, kind: item.kind }).text;
  const how = item.llmEnhanced ? t.convertedWithLlm : t.convertedPlain;
  return (
    <>
      {item.durationMs === null ? how : <>{how} {t.inDuration} <Duration t={t} ms={item.durationMs} /></>}
      {images > 0 && ` · ${t.imagesSaved(images)}`}
    </>
  );
}

function NoticeList({ t, groups, docsUrl, where, onShow }: { t: Dict; groups: NoticeGroup[]; docsUrl: string; where: boolean; onShow?: (group: NoticeGroup) => void }) {
  if (!groups.length) return null;
  return (
    <ul class="notices">
      {groups.map((group) => (
        <li key={group.key} class="notice">
          <span class="notice-mark" aria-hidden="true">
            <Icon name="Warning" size={14} />
          </span>
          <div class="notice-body">
            <p class="notice-text">
              {t.noticeTitle(group.kind, group.count, group.models, group.sample)}
              {group.kind === "other" && group.count > 1 && <span class="notice-count"> ×{group.count}</span>}
            </p>
            {where && <p class="notice-where">{t.noticeWhere(group.names)}</p>}
            {(group.samples.length > 1 || group.kind !== "other") && (
              <details class="notice-details">
                <summary>{t.notificationDetails}</summary>
                <ul>{group.samples.map((sample) => <li key={sample}>{sample}</li>)}</ul>
              </details>
            )}
          </div>
          <span class="notice-action">
            {group.kind === "cost_incomplete" && (
              <a class="text-link" href={docsPage(docsUrl, "pricing")} target="_blank" rel="noopener noreferrer">
                {t.addPrice}
                <span class="sr-only"> {t.opensNewTab}</span>
              </a>
            )}
            {where && onShow && group.items.length > 0 && (
              <button type="button" class="text-link" onClick={() => onShow(group)}>
                {group.items.length === 1 ? t.showItem : t.showItems}
              </button>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** The first blocks of a result, with the item's own pictures as small figures. */
function Excerpt({ ctx, item, onOpen }: { ctx: InspectorContext; item: SessionItem; onOpen: () => void }) {
  const { t } = ctx;
  const key = resultKey(item);
  const [result, setResult] = useState<ItemResult | null>(() => cachedExcerpt(key));
  const [error, setError] = useState<unknown>(null);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const known = cachedExcerpt(key);
    setResult(known);
    setError(null);
    if (known) return;
    let stale = false;
    loadExcerpt(item, ctx.excerpts).then(
      (value) => !stale && setResult(value),
      (reason: unknown) => !stale && setError(reason),
    );
    return () => {
      stale = true;
    };
  }, [key]);
  const split = useMemo(() => (result ? splitFrontmatter(result.markdown) : null), [result]);
  const front = useMemo(() => parseFrontMatter(split?.frontmatter ?? null), [split]);
  useEffect(() => {
    const target = body.current;
    if (!target || !split || !result) return;
    let stale = false;
    let loads: ImageLoads | null = null;
    const placeholder = (alt: string) => t.imagePlaceholder(alt || t.imageUnavailable);
    // About a screenful: cut at a blank line near 1,800 characters.
    const text = split.body.trimStart();
    const cut = text.length > 1800 ? text.slice(0, Math.max(text.lastIndexOf("\n\n", 1800), 600)) : text;
    renderMarkdown(cut, {
      documentPath: item.output ?? "",
      artifacts: result.artifacts,
      imageURL: (path) => (hasToken() ? null : imagePath(item.jobId, item.itemId, path)),
      fileURL: (path) => filePath(item.jobId, path),
      placeholder,
      fallbackAlt: t.figureFrom(item.name),
      tableLabel: t.tableAria,
      codeLabel: t.codeAria,
    }).then(
      (fragment) => {
        if (stale) return;
        // Links inside an excerpt would compete with the excerpt's own action.
        for (const link of [...fragment.querySelectorAll("a")]) link.replaceWith(...link.childNodes);
        // The excerpt clips its own overflow: its few pictures load at once.
        for (const image of fragment.querySelectorAll("img")) image.setAttribute("loading", "eager");
        target.replaceChildren(fragment);
        loads = loadArtifactImages(
          target,
          (path) => fetchBlob(imagePath(item.jobId, item.itemId, path)).then((file) => file.blob),
          (image) => {
            const span = document.createElement("span");
            span.className = "img-blocked";
            span.textContent = placeholder(image.getAttribute("alt") ?? "");
            image.replaceWith(span);
          },
          { limit: 2 },
        );
      },
      () => undefined,
    );
    return () => {
      stale = true;
      loads?.dispose();
    };
  }, [split, result, item.jobId, item.output, item.name, t]);
  if (error) return <p class="pane-note line-error">{ctx.describe(error).text}</p>;
  if (!result || !split) return <Skeleton label={t.loading} kind="list" />;
  return (
    <div class="excerpt">
      {front && (front.title || front.description || front.tags.length > 0) && <FrontCard front={front} compact />}
      {/* A picture of the result's start: nothing in it is read or reached by the keyboard. */}
      <div ref={body} class="doc-body excerpt-body" aria-hidden="true" inert />
      <button type="button" class="excerpt-open" onClick={onOpen}>
        {t.openPreview}
        <Icon name="ArrowRight" size={13} />
      </button>
    </div>
  );
}

export function FrontCard({ front, compact = false }: { front: NonNullable<ReturnType<typeof parseFrontMatter>>; compact?: boolean }) {
  return (
    <div class={compact ? "front-card is-compact" : "front-card"}>
      {front.title && <p class="front-title">{front.title}</p>}
      {front.description && <p class="front-description">{front.description}</p>}
      {front.tags.length > 0 && (
        <ul class="front-tags">
          {front.tags.slice(0, 12).map((tag) => (
            <li key={tag}>{tag}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** One action of a conversion, as a button in the row or an entry of the "⋯" menu. */
interface Act {
  key: string;
  label: string;
  icon?: IconName;
  run: () => void;
  primary?: boolean;
  disabled?: boolean;
  busy?: boolean;
}

/** In the phone sheet's bar, which secondary actions stay beside the primary one first. */
const SHEET_ORDER = ["download", "plain", "retry-failed", "stop", "enhance", "compare", "again"];

/** The actions as a row: every one as a button on a wide screen; in the phone
 * sheet's pinned bar at most two, the rest (and Delete) in a "⋯" menu. */
function ActionRow({ t, acts, sheet, danger }: { t: Dict; acts: Act[]; sheet: boolean; danger: Danger | null }) {
  const button = (act: Act) => (
    <button
      key={act.key}
      type="button"
      class={act.primary ? "btn btn-primary" : "btn"}
      disabled={act.disabled}
      aria-busy={act.busy || undefined}
      onClick={act.run}
    >
      {act.icon && <Icon name={act.icon} size={14} />}
      {act.label}
    </button>
  );
  if (!sheet) {
    return (
      <div class="insp-actions">
        {acts.map(button)}
        {danger && (
          <ConfirmPopover
            triggerLabel={danger.aria}
            title={danger.title}
            description={danger.description}
            confirmLabel={danger.confirmLabel}
            cancelLabel={danger.cancelLabel}
            busyLabel={danger.busyLabel}
            onConfirm={danger.onConfirm}
          />
        )}
      </div>
    );
  }
  const rank = (act: Act) => (act.primary ? -1 : SHEET_ORDER.indexOf(act.key) < 0 ? SHEET_ORDER.length : SHEET_ORDER.indexOf(act.key));
  const ordered = [...acts].sort((a, b) => rank(a) - rank(b));
  const shown = ordered.slice(0, 2);
  const rest = ordered.slice(2);
  return (
    <div class="insp-actions is-sheet">
      {shown.map(button)}
      <ActionMenu
        label={t.moreActions}
        up
        actions={rest.map((act) => ({ key: act.key, label: act.label, icon: act.icon, run: act.run, disabled: act.disabled }))}
        danger={danger ?? undefined}
      />
    </div>
  );
}

type Danger = MenuDanger & { aria: string };

/** The item's actions; `sheet` lays them out for the bottom sheet's pinned bar. */
export function ItemActions({ ctx, item, actions, sheet = false }: { ctx: InspectorContext; item: SessionItem; actions: ItemActionsApi; sheet?: boolean }) {
  const { t } = ctx;
  const busy = ctx.busy.has(item.key);
  const previewable = isPreviewable(item);
  const retryable = canRetry(item);
  const noModel = item.status === "error" && item.errorCode === "no_model_configured";
  const imageOnly = item.skipped && item.skipReason === "image_only";
  const enhanceable = previewable && !item.llmEnhanced && item.retryable && ctx.llmReady;
  const acts: Act[] = [];
  if (previewable) acts.push({ key: "preview", label: t.preview, icon: "Eye", primary: true, run: () => actions.preview(item) });
  else if (retryable && !noModel && !imageOnly)
    acts.push({ key: "retry", label: t.retryPlainVerb, icon: "ArrowCounterClockwise", primary: true, disabled: busy, busy, run: () => actions.retry(item) });
  if (noModel && retryable) {
    acts.push({ key: "models", label: t.addModelAction, primary: true, run: actions.openModels });
    acts.push({ key: "plain", label: t.retryPlain, disabled: busy, run: () => actions.retryPlain(item) });
  }
  if (imageOnly && retryable) acts.push({ key: "ocr", label: t.enableOcr, primary: true, disabled: busy, run: () => actions.retryWithOcr(item) });
  if (previewable && ctx.compare) acts.push({ key: "compare", label: t.compare, icon: "Columns", run: () => actions.compare(item) });
  if (previewable && item.output)
    acts.push({ key: "download", label: sheet ? t.downloadShort : t.downloadMd, icon: "DownloadSimple", run: () => actions.download(item, item.output as string) });
  if (previewable && retryable === false && item.retryable)
    acts.push({ key: "again", label: t.runAgain, icon: "ArrowCounterClockwise", disabled: busy, busy, run: () => actions.retry(item) });
  if (enhanceable) acts.push({ key: "enhance", label: t.enhance, icon: "MagicWand", disabled: busy, run: () => actions.enhance(item) });
  const danger: Danger | null =
    item.status !== "running" && item.status !== "queued"
      ? {
          aria: t.histDeleteAria(item.name),
          label: t.deleteAction,
          title: t.deleteItemTitle(item.name),
          description: t.deleteItemDescription,
          confirmLabel: t.deletePermanently,
          cancelLabel: t.cancel,
          busyLabel: t.deleting,
          onConfirm: () => actions.remove(item),
        }
      : null;
  return <ActionRow t={t} acts={acts} sheet={sheet} danger={danger} />;
}

export function ItemInspector({
  ctx,
  item,
  job,
  actions,
  sheet = false,
  headingId = "insp-title",
}: {
  ctx: InspectorContext;
  item: SessionItem;
  job: ConvJob | undefined;
  actions: ItemActionsApi;
  sheet?: boolean;
  headingId?: string;
}) {
  const { t, locale } = ctx;
  const glyph = glyphOf(item);
  const elapsed = useElapsed(item.status === "running" ? item.startedAt : null);
  const result = cachedExcerpt(resultKey(item));
  const [, rerender] = useState(0);
  // The facts that come from the result (output size, images) appear once it is read.
  useEffect(() => {
    if (!isPreviewable(item) || result) return;
    let stale = false;
    loadExcerpt(item, ctx.excerpts).then(
      () => !stale && rerender((value) => value + 1),
      () => undefined,
    );
    return () => {
      stale = true;
    };
  }, [resultKey(item)]);
  const artifacts = result?.artifacts ?? [];
  const output = item.output ? artifacts.find((artifact) => artifact.relpath === item.output) : undefined;
  const images = artifacts.filter((artifact) => /\.(png|jpe?g|gif|webp|avif|svg)$/i.test(artifact.relpath) && !artifact.relpath.includes("/screenshots/")).length;
  const shots = artifacts.filter((artifact) => artifact.relpath.includes("/screenshots/")).length;
  const base = item.llmEnhanced ? artifacts.find((artifact) => artifact.relpath.endsWith(".md") && !artifact.relpath.endsWith(".llm.md")) : undefined;
  const failed = isFailed(item);
  const problem = failed || item.rerunFailure ? itemErrorText(locale, { error: item.rerunFailure?.error ?? item.error, error_code: item.rerunFailure?.error_code ?? item.errorCode, kind: item.kind }) : null;
  const notices = useMemo(() => groupNotices(item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text }))), [item.warnings, item.key, item.name]);
  const cost = itemCost(item);
  const kept = keptUntil(ctx, item.finishedAt);
  const also = [
    base ? t.alsoBase(basename(base.relpath)) : null,
    images > 0 ? t.imagesSaved(images) : null,
    shots > 0 ? t.screenshotsSaved(shots) : null,
  ].filter(Boolean);
  const url = item.kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;
  return (
    <section class={sheet ? "insp is-sheet" : "insp"} aria-labelledby={headingId}>
      <ActionError t={t} failure={ctx.failure?.key === item.key ? ctx.failure : null} />
      <header class="insp-head">
        {/* In the sheet the name is the sheet's own heading. */}
        {!sheet && (
          <h2 id={headingId} class="insp-title">
            <StatusGlyph glyph={glyph} label={statusWord(t, glyph)} />
            <span class="insp-name">{item.name}</span>
          </h2>
        )}
        <p class={failed ? "insp-line is-error" : "insp-line"}>
          {sheet && <StatusGlyph glyph={glyph} label={statusWord(t, glyph)} />}
          {itemLine(ctx, item, elapsed, images)}
        </p>
      </header>
      {!sheet && <ItemActions ctx={ctx} item={item} actions={actions} />}
      {job?.persistenceError && <ConditionLine tone="error" text={persistenceText(locale, job.persistenceError).text} />}
      {!ctx.llmReady && item.status === "done" && !item.llmEnhanced && !item.skipped && (
        <ConditionLine tone="info" text={t.noModelLine} action={{ label: t.configureModels, run: actions.openModels }} />
      )}
      {problem && (
        <div class={failed ? "problem is-error" : "problem is-retained"}>
          {item.rerunFailure && !failed && <p class="problem-title">{t.rerunRetained(item.rerunFailure.operation)}</p>}
          {item.rerunFailure && !failed && <p>{problem.text}</p>}
          {problem.formats && (
            <details class="notice-details">
              <summary>{t.supportedFormats}</summary>
              <p class="formats">{problem.formats}</p>
            </details>
          )}
          {problem.detail && (
            <details class="notice-details">
              <summary>{t.notificationDetails}</summary>
              <p class="raw">{problem.detail}</p>
            </details>
          )}
        </div>
      )}
      <dl class="facts">
        <dt>{t.factSource}</dt>
        <dd>
          {url ? (
            <a class="text-link fact-link" href={url} target="_blank" rel="noopener noreferrer">
              {url}
              <span class="sr-only"> {t.opensNewTab}</span>
            </a>
          ) : (
            <span class="fact-name">{item.name}</span>
          )}
          {item.sizeBytes !== null && <span class="fact-dim"> {fmtBytes(item.sizeBytes)}</span>}
        </dd>
        {item.output && (
          <>
            <dt>{t.factOutput}</dt>
            <dd class="mono fact-file">
              <FileName name={basename(item.output)} />
              {output && <span class="fact-dim fact-size">· {fmtBytes(output.size)}</span>}
            </dd>
          </>
        )}
        {also.length > 0 && (
          <>
            <dt>{t.factAlso}</dt>
            <dd>{also.join(", ")}</dd>
          </>
        )}
        {(item.models.length > 0 || (item.llmEnhanced && cost.kind !== "none")) && (
          <>
            <dt>{t.factModels}</dt>
            <dd>{item.models.length ? item.models.join(", ") : t.modelsNotReported}</dd>
          </>
        )}
        <dt>{t.factCost}</dt>
        <dd class={cost.kind === "unknown" ? "is-unknown" : undefined}>{costText(t, cost)}</dd>
        {optionsText(t, item) && (
          <>
            <dt>{t.factOptions}</dt>
            <dd>{optionsText(t, item)}</dd>
          </>
        )}
        {item.finishedAt && (
          <>
            <dt>{t.factFinished}</dt>
            <dd>{fmtWhen(item.finishedAt, locale)}</dd>
          </>
        )}
        {kept && job?.status !== "running" && (
          <>
            <dt>{t.factKept}</dt>
            <dd>{kept}</dd>
          </>
        )}
      </dl>
      {notices.length > 0 && (
        <section class="insp-section" aria-labelledby={`${headingId}-notices`}>
          <h3 id={`${headingId}-notices`}>{t.noticesHeading(notices.reduce((sum, group) => sum + group.count, 0))}</h3>
          <NoticeList t={t} groups={notices} docsUrl={ctx.docsUrl} where={false} />
        </section>
      )}
      {isPreviewable(item) && (
        <section class="insp-section" aria-labelledby={`${headingId}-start`}>
          <h3 id={`${headingId}-start`}>{t.resultStart}</h3>
          <Excerpt ctx={ctx} item={item} onOpen={() => actions.preview(item)} />
        </section>
      )}
    </section>
  );
}

export function fmtWhen(value: string, locale: Locale): string {
  const ms = timestampMs(value);
  if (ms === null) return fmtDateTime(value);
  return new Intl.DateTimeFormat(localeTag(locale), { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(ms));
}

/** Done, failed, converting and waiting, as one bar (A2): the submission's one bold element. */
function ProgressStrip({ t, items }: { t: Dict; items: SessionItem[] }) {
  const tally = counts(items);
  const parts: [string, number, string][] = [
    ["done", tally.done + tally.skipped, t.statusDone],
    ["failed", tally.failed, t.statusFailed],
    ["stopped", tally.stopped, t.statusStopped],
    ["running", tally.running, t.statusRunning],
    ["queued", tally.queued, t.statusQueued],
  ];
  return (
    <div class="strip-wrap">
      <div class="strip" role="img" aria-label={parts.filter(([, n]) => n > 0).map(([, n, word]) => `${n} ${word}`).join(", ")}>
        {parts.map(([kind, n]) => (n > 0 ? <span key={kind} class={`strip-${kind}`} style={{ flexGrow: n }} /> : null))}
      </div>
      <p class="strip-legend" aria-hidden="true">
        {parts
          .filter(([, n]) => n > 0)
          .map(([kind, n, word]) => (
            <span key={kind} class={`legend-${kind}`}>
              <i />
              {n} {word}
            </span>
          ))}
      </p>
    </div>
  );
}

export function JobInspector({
  ctx,
  job,
  items,
  actions,
  sheet = false,
  headingId = "insp-title",
  onShowItems,
}: {
  ctx: InspectorContext;
  job: ConvJob;
  items: SessionItem[];
  actions: JobActionsApi;
  sheet?: boolean;
  headingId?: string;
  onShowItems: (keys: string[]) => void;
}) {
  const { t, locale } = ctx;
  const tally = counts(items);
  const label = submissionLabel(job, items, ctx.labelWords);
  const notices = useMemo(() => groupNotices(items.flatMap((item) => item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text })))), [items]);
  const cost = totalCost(items);
  const running = job.status === "running";
  const kept = keptUntil(ctx, job.finishedAt);
  const preset = presetOf(job.options);
  return (
    <section class={sheet ? "insp is-sheet" : "insp"} aria-labelledby={headingId}>
      <ActionError t={t} failure={ctx.failure?.key === `job:${job.jobId}` ? ctx.failure : null} />
      <header class="insp-head">
        {!sheet && (
          <h2 id={headingId} class="insp-title">
            <span class="insp-name">{label}</span>
          </h2>
        )}
        <p class="insp-line">
          {t.itemsCount(items.length)}
          {preset && ` · ${t.presetName(preset)}`}
          {job.createdAt && ` · ${fmtWhen(job.createdAt, locale)}`}
        </p>
      </header>
      <ProgressStrip t={t} items={items} />
      {!sheet && <JobActions ctx={ctx} job={job} items={items} actions={actions} />}
      {job.persistenceError && <ConditionLine tone="error" text={persistenceText(locale, job.persistenceError).text} />}
      <dl class="facts">
        <dt>{t.factCost}</dt>
        <dd class={cost.kind === "unknown" ? "is-unknown" : undefined}>
          {costText(t, cost)}
          {cost.kind === "unknown" && cost.unknownItems > 0 && <span class="fact-dim"> · {t.costUnknownItems(cost.unknownItems)}</span>}
        </dd>
        {optionsText(t, job) && (
          <>
            <dt>{t.factOptions}</dt>
            <dd>{optionsText(t, job)}</dd>
          </>
        )}
        {!running && job.durationMs !== null && (
          <>
            <dt>{t.factTook}</dt>
            <dd>
              <Duration t={t} ms={job.durationMs} />
            </dd>
          </>
        )}
        {job.finishedAt && !running && (
          <>
            <dt>{t.factFinished}</dt>
            <dd>{fmtWhen(job.finishedAt, locale)}</dd>
          </>
        )}
        {kept && !running && (
          <>
            <dt>{t.factKept}</dt>
            <dd>{kept}</dd>
          </>
        )}
        {job.origin === "cli" && (
          <>
            <dt>{t.factOrigin}</dt>
            <dd>{t.originCliLong}</dd>
          </>
        )}
      </dl>
      {tally.failed > 0 && (
        <section class="insp-section" aria-labelledby={`${headingId}-failed`}>
          <h3 id={`${headingId}-failed`}>{t.failedHeading(tally.failed)}</h3>
          <ul class="failures">
            {items.filter(isFailed).slice(0, 8).map((item) => (
              <li key={item.key}>
                <button type="button" class="failure-link" onClick={() => actions.select(item.key)}>
                  <span class="failure-name">{item.name}</span>
                  <span class="failure-why">{itemErrorText(locale, { error: item.error, error_code: item.errorCode, kind: item.kind }).text}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      {notices.length > 0 && (
        <section class="insp-section" aria-labelledby={`${headingId}-notices`}>
          <h3 id={`${headingId}-notices`}>{t.noticeKinds(notices.length)}</h3>
          <NoticeList t={t} groups={notices} docsUrl={ctx.docsUrl} where onShow={(group) => onShowItems(group.items)} />
        </section>
      )}
      {tally.stopped > 0 && <p class="insp-note">{t.stoppedCount(tally.stopped)}</p>}
    </section>
  );
}

export function JobActions({ ctx, job, items, actions, sheet = false }: { ctx: InspectorContext; job: ConvJob; items: SessionItem[]; actions: JobActionsApi; sheet?: boolean }) {
  const { t } = ctx;
  const retryable = items.filter((item) => (isFailed(item) || isStopped(item)) && canRetry(item)).length;
  const waiting = items.some((item) => item.status === "queued" && item.operation === "convert");
  const running = job.status === "running";
  const busy = ctx.busy.has(`job:${job.jobId}`);
  const acts: Act[] = [];
  if (!running) acts.push({ key: "zip", label: t.downloadZip, icon: "DownloadSimple", primary: true, run: () => actions.zip(job.jobId) });
  if (retryable > 0)
    acts.push({ key: "retry-failed", label: t.retryFailedN(retryable), icon: "ArrowCounterClockwise", disabled: busy, busy, run: () => actions.retryFailed(job.jobId) });
  if (waiting && ctx.cancel) acts.push({ key: "stop", label: t.stopRemaining, icon: "Stop", disabled: busy, run: () => actions.stop(job.jobId) });
  const label = submissionLabel(job, items, ctx.labelWords);
  const danger: Danger | null = running
    ? null
    : {
        aria: t.deleteSubmissionAria(label),
        label: t.deleteAction,
        title: t.deleteSubmissionTitle(items.length),
        description: t.deleteItemDescription,
        confirmLabel: t.deletePermanently,
        cancelLabel: t.cancel,
        busyLabel: t.deleting,
        onConfirm: () => actions.remove(job.jobId),
      };
  return <ActionRow t={t} acts={acts} sheet={sheet} danger={danger} />;
}

/** Nothing selected: what the whole list adds up to, and what needs a look (A4). */
export function Overview({ ctx, items, onShowItems }: { ctx: InspectorContext; items: SessionItem[]; onShowItems: (keys: string[]) => void }) {
  const { t } = ctx;
  const tally = counts(items);
  const notices = useMemo(() => groupNotices(items.flatMap((item) => item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text })))), [items]);
  const cost = totalCost(items);
  if (!items.length) return <div class="insp is-empty"><p>{t.overviewEmpty}</p></div>;
  return (
    <section class="insp is-overview" aria-labelledby="overview-title">
      <header class="insp-head">
        <h2 id="overview-title" class="insp-title">
          <span class="insp-name">{t.overviewTitle}</span>
        </h2>
        <p class="insp-line">{t.selectHint}</p>
      </header>
      <ProgressStrip t={t} items={items} />
      <dl class="facts">
        <dt>{t.factCost}</dt>
        <dd class={cost.kind === "unknown" ? "is-unknown" : undefined}>
          {costText(t, cost)}
          {cost.kind === "unknown" && cost.unknownItems > 0 && <span class="fact-dim"> · {t.costUnknownItems(cost.unknownItems)}</span>}
        </dd>
      </dl>
      {tally.failed > 0 && (
        <p class="insp-note">
          <button type="button" class="text-link" onClick={() => onShowItems(items.filter(isFailed).map((item) => item.key))}>
            {t.failedHeading(tally.failed)}
          </button>
        </p>
      )}
      {notices.length > 0 && (
        <section class="insp-section" aria-labelledby="overview-notices">
          <h3 id="overview-notices">{t.noticeKinds(notices.length)}</h3>
          <NoticeList t={t} groups={notices} docsUrl={ctx.docsUrl} where onShow={(group) => onShowItems(group.items)} />
        </section>
      )}
    </section>
  );
}

export { costState };

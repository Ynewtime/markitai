// The conversion options in a dialog (owner, 2026-10-24): a sheet from the bottom edge on a phone, a
// centred dialog near the top on a wider screen, the same content in both. The first page has the
// preset and one row per group with its current value; a row opens its group in place (Back returns)
// and a choice applies at once. The foot keeps the CLI line and the defaults' two buttons.
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { CloudflareCapability, ConversionBackend, FetchStrategy, OutputProfile, Preset } from "../api/types.ts";
import type { DefaultsControl } from "../hooks/use-convert-defaults.ts";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict } from "../i18n/index.ts";
import { cloudflareReason } from "../lib/cloudflare.ts";
import { copyText } from "../lib/clipboard.ts";
import {
  applyPreset,
  BUILTIN_PRESETS,
  changeAdvanced,
  defaultsOf,
  featuresOf,
  groupValue,
  matchingPreset,
  offeredStrategies,
  OPTION_GROUPS,
  PRESETS,
  resolveOptions,
  sameAsDefaults,
  type Advanced,
  type Composer,
  type OptionGroup,
  type OptionKey,
  type PresetTable,
} from "../lib/options.ts";
import { markMore } from "../lib/scroll-cells.ts";
import { HelpTooltip } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";
import { languageName, languageTag } from "./pick-menu.tsx";

const BACKENDS: ConversionBackend[] = ["native", "cloudflare"];
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];
/** How far the sheet is pulled down by its handle before letting go closes it. */
const SWIPE_CLOSE = 80;
/** The stylesheet's phone tier, where the dialog is a sheet from the bottom edge. */
const SHEET = "(max-width: 760px)";

type Page = OptionGroup | "ocrLang";

/** A reason in a second line: no closing full stop. */
const short = (text: string) => text.replace(/[.。]$/, "");

function SwitchRow({
  id,
  name,
  sub,
  on,
  disabled = false,
  onToggle,
}: {
  id: string;
  name: string;
  sub?: ComponentChildren;
  on: boolean;
  disabled?: boolean;
  onToggle: (value: boolean) => void;
}) {
  return (
    // The whole row toggles for a pointer; the switch is the control a keyboard and a screen reader use.
    <div
      class={disabled ? "sheet-opt is-disabled" : "sheet-opt"}
      onClick={(event) => {
        if (!disabled && !(event.target as Element).closest("button, a")) onToggle(!on);
      }}
    >
      <span class="sheet-opt-text">
        <span id={`${id}-name`} class="sheet-opt-name">
          {name}
        </span>
        {sub && (
          <span id={`${id}-sub`} class="sheet-opt-sub">
            {sub}
          </span>
        )}
      </span>
      <button
        type="button"
        role="switch"
        class={on ? "switch is-on" : "switch"}
        aria-checked={on}
        aria-labelledby={`${id}-name`}
        aria-describedby={sub ? `${id}-sub` : undefined}
        disabled={disabled}
        onClick={() => onToggle(!on)}
      />
    </div>
  );
}

/** One choice of several: rows with a check on the current one, the arrow keys move and choose. */
function RadioList<T extends string | null>({
  id,
  labelledBy,
  value,
  choices,
  label,
  sub,
  lang,
  disabled,
  onPick,
}: {
  id: string;
  labelledBy: string;
  value: T | null;
  choices: readonly T[];
  label: (choice: T) => string;
  sub?: (choice: T) => string | undefined;
  lang?: (choice: T) => string | undefined;
  disabled?: (choice: T) => boolean;
  onPick: (choice: T) => void;
}) {
  const key = (choice: T) => `${id}-${choice ?? "default"}`;
  const enabled = choices.filter((choice) => !disabled?.(choice));
  const focusable = value !== null && enabled.includes(value) ? value : enabled[0];
  const move = (from: T, step: number) => {
    const at = enabled.indexOf(from);
    const next = enabled[(at + step + enabled.length) % enabled.length];
    if (next === undefined) return;
    onPick(next);
    document.getElementById(key(next))?.focus();
  };
  return (
    <div class="sheet-card" role="radiogroup" aria-labelledby={labelledBy}>
      {choices.map((choice) => {
        const off = disabled?.(choice) ?? false;
        const on = choice === value;
        const line = sub?.(choice);
        return (
          <button
            key={choice ?? "default"}
            id={key(choice)}
            type="button"
            role="radio"
            class={off ? "sheet-opt is-disabled" : "sheet-opt"}
            aria-checked={on}
            aria-disabled={off || undefined}
            aria-labelledby={`${key(choice)}-name`}
            aria-describedby={line ? `${key(choice)}-sub` : undefined}
            tabIndex={choice === focusable ? 0 : -1}
            onClick={() => {
              if (!off) onPick(choice);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown" || event.key === "ArrowRight") move(choice, 1);
              else if (event.key === "ArrowUp" || event.key === "ArrowLeft") move(choice, -1);
              else return;
              event.preventDefault();
            }}
          >
            <span class="sheet-opt-text">
              <span id={`${key(choice)}-name`} class="sheet-opt-name" lang={lang?.(choice)}>
                {label(choice)}
              </span>
              {line && (
                <span id={`${key(choice)}-sub`} class="sheet-opt-sub">
                  {line}
                </span>
              )}
            </span>
            <span class="sheet-check" aria-hidden="true">
              {on && <Icon name="Check" size={16} />}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function OptionsDialog({
  t,
  state,
  presets = BUILTIN_PRESETS,
  llmReady,
  needsModel,
  onAddModel,
  cloudflare,
  command,
  pathsNote,
  announce,
  defaults,
  changed,
  onChange,
  onClose,
  textOnly,
  localOnly,
  documentJson,
  ocrLanguages,
  serverOcrLang,
  ocrAllPages,
  pageMarkers,
}: {
  t: Dict;
  state: Composer;
  presets?: PresetTable;
  llmReady: boolean;
  needsModel: string;
  /** Where models are added (Settings → Models), when this caller may add one. */
  onAddModel?: () => void;
  cloudflare?: CloudflareCapability;
  command: string;
  /** Files are named relative to their folder: say where to run the command. */
  pathsNote: boolean;
  announce: (message: string) => void;
  defaults: DefaultsControl | null;
  /** The choices that differ from the defaults (the server's, or the user's own): shown bright. */
  changed: Set<OptionKey>;
  onChange: (next: Composer) => void;
  onClose: () => void;
  textOnly: boolean;
  localOnly: boolean;
  documentJson: boolean;
  ocrLanguages: string[] | null;
  serverOcrLang: string;
  ocrAllPages: boolean;
  pageMarkers: boolean;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const [stack, setStack] = useState<Page[]>([]);
  const page = stack[stack.length - 1] ?? null;
  const left = useRef<Page | null>(null);
  const [pull, setPull] = useState(0);
  const drag = useRef<number | null>(null);
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const cli = useRef<HTMLElement>(null);
  useModal(dialog, onClose);
  // The command stays on one line and scrolls sideways in its box; its right edge fades while
  // there is more (measured again when the box resizes, the web font arrives or the command changes).
  useLayoutEffect(() => {
    const line = cli.current;
    if (!line) return;
    const fit = () => markMore(line);
    fit();
    void document.fonts?.ready.then(fit);
    const resized = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fit);
    resized?.observe(line);
    return () => resized?.disconnect();
  }, [command]);
  // A group opened: focus its Back; back again: the row it was opened from.
  useEffect(() => {
    body.current?.scrollTo(0, 0);
    const root = dialog.current;
    if (!root) return;
    if (page !== null) root.querySelector<HTMLElement>(".sheet-back")?.focus();
    else if (left.current !== null) root.querySelector<HTMLElement>(`[data-page="${left.current}"]`)?.focus();
  }, [page]);
  useEffect(() => {
    if (copied === "idle") return;
    const timer = setTimeout(() => setCopied("idle"), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  const open = (next: Page) => setStack((current) => [...current, next]);
  const back = () => {
    left.current = page;
    setStack((current) => current.slice(0, -1));
  };

  const a = state.advanced;
  const effective = resolveOptions(state, presets);
  const reason = cloudflareReason(cloudflare);
  const matched = matchingPreset(state, presets);
  const analysisOff = !state.llm || a.pure;
  const set = <K extends keyof Advanced>(key: K, value: Advanced[K]) => onChange({ ...state, advanced: changeAdvanced(a, key, value) });
  const presetLabel: Record<Preset, string> = { minimal: t.presetMinimal, standard: t.presetStandard, rich: t.presetRich };
  const presetOff = (preset: Preset) => !llmReady && featuresOf(preset, presets).llm;
  const presetHelp = (preset: Preset) => {
    const features = featuresOf(preset, presets);
    const builtin = BUILTIN_PRESETS[preset];
    const custom = (Object.keys(builtin) as (keyof typeof builtin)[]).some((key) => features[key] !== builtin[key]);
    const text = custom ? t.helpServerPreset : { minimal: t.helpMinimal, standard: t.helpStandard, rich: t.helpRich }[preset];
    return presetOff(preset) ? `${text} ${needsModel}` : text;
  };
  const profileLabel = (profile: OutputProfile | null) =>
    ({ default: t.profileNone, rag: t.profileRag, obsidian: t.profileObsidian, okf: t.profileOkf })[profile ?? "default"];
  const profileSub = (profile: OutputProfile | null) =>
    ({ default: t.helpDefault, rag: t.helpRag, obsidian: t.helpObsidian, okf: t.helpOkf })[profile ?? "default"];
  const strategies = offeredStrategies(localOnly);
  const strategyLabel: Record<FetchStrategy, string> = {
    auto: t.strategyAuto,
    static: t.strategyStatic,
    playwright: t.strategyPlaywright,
    defuddle: t.strategyDefuddle,
    jina: t.strategyJina,
    cloudflare: t.strategyCloudflare,
  };
  const strategyOff = (value: FetchStrategy) => value === "cloudflare" && (reason !== null || !cloudflare?.browser_rendering);
  const strategySub = (value: FetchStrategy) =>
    value === "cloudflare" && reason
      ? short(t.cloudflareReason(reason))
      : { auto: t.helpAuto, static: t.helpStatic, playwright: t.helpPlaywright, defuddle: t.helpDefuddle, jina: t.helpJina, cloudflare: t.helpCloudflareUrl }[value];
  const backendLabel = (value: ConversionBackend) => (value === "native" ? t.backendNative : t.backendCloudflare);
  const backendOff = (value: ConversionBackend) => value === "cloudflare" && (reason !== null || !cloudflare?.file_conversion);
  const backendSub = (value: ConversionBackend) =>
    value === "native" ? t.helpNative : reason ? short(t.cloudflareReason(reason)) : t.helpCloudflareFile;
  const picker = ocrLanguages !== null && ocrLanguages.length > 0 && state.ocr && !state.llm;
  const ocrLang = state.ocrLang ?? serverOcrLang;
  const markersOn = state.profile === "rag" || a.pageMarkers === true;
  const pagesRow = pageMarkers || ocrAllPages;
  const on = (parts: [boolean, string][]) => parts.filter(([value]) => value).map(([, label]) => label);

  const groups: { page: OptionGroup; name: string; value: string; shown: boolean }[] = [
    { page: "enhance", name: t.advEnhance, value: groupValue(on([[state.llm, t.llmEnhance], [state.ocr, t.ocr]]), t.optionOff), shown: true },
    {
      page: "images",
      name: t.advImages,
      value: groupValue(on([[effective.alt === true, t.advAlt], [effective.desc === true, t.advDesc], [effective.screenshot === true, t.advScreenshot]]), t.optionOff),
      shown: true,
    },
    { page: "output", name: t.advOutput, value: [profileLabel(state.profile), ...on([[state.format === "json", t.advJson]])].join(" · "), shown: true },
    { page: "url", name: t.advStrategy, value: strategyLabel[a.strategy], shown: true },
    { page: "files", name: t.advBackend, value: backendLabel(a.backend), shown: !localOnly },
    {
      page: "pages",
      name: t.advPages,
      value: groupValue(on([[pageMarkers && markersOn, t.advPageMarkers], [ocrAllPages && effective.ocr_all_pages === true, t.advOcrAllPages]]), t.optionOff),
      shown: pagesRow,
    },
    {
      page: "other",
      name: t.advOther,
      value: groupValue(
        on([[a.screenshotOnly, t.advScreenshotOnly], [a.pure, t.advPure], [textOnly && a.textOnly, t.advTextOnly], [a.noCache, t.advNoCache], [a.noCompress, t.advNoCompress]]),
        t.optionOff,
      ),
      shown: true,
    },
  ];
  const differs = (group: OptionGroup) => (OPTION_GROUPS[group] as readonly OptionKey[]).some((key) => changed.has(key));
  const row = (group: (typeof groups)[number]) => (
    <button
      key={group.page}
      type="button"
      class="sheet-row"
      data-page={group.page}
      aria-label={t.optionRow(group.name, group.value)}
      onClick={() => open(group.page)}
    >
      <span class="sheet-row-name">{group.name}</span>
      <span class={differs(group.page) ? "sheet-row-value is-changed" : "sheet-row-value"} title={group.value}>
        {group.value}
      </span>
      <Icon name="CaretRight" size={14} />
    </button>
  );
  const shown = groups.filter((group) => group.shown);
  const title = page === "ocrLang" ? t.ocrLangLabel : (groups.find((group) => group.page === page)?.name ?? t.optionsTitle);

  const noModel = (
    <>
      {onAddModel ? t.noModel : short(needsModel)}
      {onAddModel && (
        <>
          {" · "}
          <button
            type="button"
            class="text-link"
            onClick={() => {
              onClose();
              onAddModel();
            }}
          >
            {t.addModelLink}
          </button>
        </>
      )}
    </>
  );

  const content = (() => {
    switch (page) {
      case null:
        return (
          <>
            <div class="sheet-preset">
              <span class="sheet-preset-name">
                <span id="opts-preset">{t.preset}</span>
                {matched === null && (
                  <span class="sheet-custom" role="status">
                    {t.presetCustomized}
                  </span>
                )}
              </span>
              {/* Only a request that still equals a preset marks it; a customized set leaves all three plain. */}
              <span class="seg" role="group" aria-labelledby="opts-preset">
                {PRESETS.map((preset) => {
                  const off = presetOff(preset);
                  return (
                    <button
                      key={preset}
                      type="button"
                      class={preset === matched ? "is-on" : undefined}
                      aria-pressed={preset === matched}
                      aria-describedby={`opts-preset-${preset}`}
                      aria-disabled={off || undefined}
                      onClick={() => {
                        if (!off) onChange(applyPreset(state, preset, presets));
                      }}
                    >
                      {presetLabel[preset]}
                    </button>
                  );
                })}
              </span>
              {PRESETS.map((preset) => (
                <span key={preset} id={`opts-preset-${preset}`} hidden>
                  {presetHelp(preset)}
                </span>
              ))}
            </div>
            <div class="sheet-card">{shown.slice(0, 3).map(row)}</div>
            <p class="sheet-section">{t.advanced}</p>
            <div class="sheet-card">{shown.slice(3).map(row)}</div>
          </>
        );
      case "enhance":
        return (
          <div class="sheet-card" role="group" aria-labelledby="opts-page">
            <SwitchRow
              id="opts-llm"
              name={t.llmEnhance}
              sub={llmReady ? t.helpLlm : noModel}
              on={state.llm}
              disabled={!llmReady}
              onToggle={(llm) => onChange({ ...state, llm })}
            />
            <SwitchRow
              id="opts-ocr"
              name={t.ocr}
              sub={state.ocr && state.llm ? t.advVlmOcr : t.helpOcr}
              on={state.ocr}
              onToggle={(ocr) => onChange({ ...state, ocr })}
            />
            {picker && (
              <button type="button" class="sheet-row" data-page="ocrLang" aria-label={t.optionRow(t.ocrLangLabel, languageName(ocrLang, t.ocrLangAutoShort))} onClick={() => open("ocrLang")}>
                <span class="sheet-row-name">{t.ocrLangLabel}</span>
                <span class={changed.has("ocr_lang") ? "sheet-row-value is-changed" : "sheet-row-value"}>
                  {languageName(ocrLang, t.ocrLangAutoShort)}
                </span>
                <Icon name="CaretRight" size={14} />
              </button>
            )}
          </div>
        );
      case "ocrLang":
        return (
          <RadioList
            id="opts-lang"
            labelledBy="opts-page"
            value={ocrLang}
            choices={ocrLanguages ?? []}
            label={(code) => languageName(code, t.ocrLangAutoShort)}
            sub={(code) => (code === "en" ? t.ocrLangAutoHint : undefined)}
            lang={languageTag}
            onPick={(code) => onChange({ ...state, ocrLang: code })}
          />
        );
      case "images": {
        const analysisNote = a.pure ? t.advPlainImages : t.advNeedsLlm;
        return (
          <div class="sheet-card" role="group" aria-labelledby="opts-page">
            <SwitchRow id="opts-alt" name={t.advAlt} sub={analysisOff ? analysisNote : t.helpAlt} on={effective.alt === true} disabled={analysisOff} onToggle={(value) => set("alt", value)} />
            <SwitchRow id="opts-desc" name={t.advDesc} sub={analysisOff ? analysisNote : t.helpDesc} on={effective.desc === true} disabled={analysisOff} onToggle={(value) => set("desc", value)} />
            <SwitchRow
              id="opts-screenshot"
              name={t.advScreenshot}
              sub={a.screenshotOnly ? t.advImpliedBySource : t.helpScreenshot}
              on={effective.screenshot === true}
              disabled={a.screenshotOnly}
              onToggle={(value) => set("screenshot", value)}
            />
          </div>
        );
      }
      case "output":
        return (
          <>
            <RadioList id="opts-profile" labelledBy="opts-page" value={state.profile} choices={PROFILES} label={profileLabel} sub={profileSub} onPick={(profile) => onChange({ ...state, profile })} />
            {documentJson && (
              <div class="sheet-card">
                <SwitchRow id="opts-json" name={t.advJson} sub={t.helpJson} on={state.format === "json"} onToggle={(json) => onChange({ ...state, format: json ? "json" : "markdown" })} />
              </div>
            )}
          </>
        );
      case "url":
        return (
          <RadioList
            id="opts-strategy"
            labelledBy="opts-page"
            value={a.strategy}
            choices={strategies}
            label={(value) => strategyLabel[value]}
            sub={strategySub}
            disabled={strategyOff}
            onPick={(value) => set("strategy", value)}
          />
        );
      case "files":
        return (
          <RadioList
            id="opts-backend"
            labelledBy="opts-page"
            value={a.backend}
            choices={BACKENDS}
            label={backendLabel}
            sub={backendSub}
            disabled={backendOff}
            onPick={(value) => set("backend", value)}
          />
        );
      case "pages":
        return (
          <div class="sheet-card" role="group" aria-labelledby="opts-page">
            {pageMarkers && (
              <SwitchRow
                id="opts-page-markers"
                name={t.advPageMarkers}
                // RAG output keeps its own page provenance either way.
                sub={state.profile === "rag" ? t.advRagPageMarkers : t.helpPageMarkers}
                on={markersOn}
                disabled={state.profile === "rag"}
                onToggle={(value) => set("pageMarkers", value)}
              />
            )}
            {ocrAllPages && (
              <SwitchRow
                id="opts-ocr-all-pages"
                name={t.advOcrAllPages}
                sub={state.ocr ? t.helpOcrAllPages : t.advNeedsOcr}
                on={effective.ocr_all_pages === true}
                disabled={!state.ocr}
                onToggle={(value) => set("ocrAllPages", value)}
              />
            )}
          </div>
        );
      case "other":
        return (
          <div class="sheet-card" role="group" aria-labelledby="opts-page">
            <SwitchRow
              id="opts-screenshot-only"
              name={t.advScreenshotOnly}
              sub={state.llm ? t.helpSource : t.advCaptureOnly}
              on={a.screenshotOnly}
              onToggle={(value) => set("screenshotOnly", value)}
            />
            <SwitchRow id="opts-pure" name={t.advPure} sub={t.helpPure} on={a.pure} onToggle={(value) => set("pure", value)} />
            {textOnly && <SwitchRow id="opts-text-only" name={t.advTextOnly} sub={t.helpTextOnly} on={a.textOnly} onToggle={(value) => set("textOnly", value)} />}
            <SwitchRow id="opts-no-cache" name={t.advNoCache} sub={t.helpCache} on={a.noCache} onToggle={(value) => set("noCache", value)} />
            <SwitchRow id="opts-no-compress" name={t.advNoCompress} sub={t.helpCompress} on={a.noCompress} onToggle={(value) => set("noCompress", value)} />
          </div>
        );
    }
  })();

  const copy = () => {
    void copyText(command).then((ok) => {
      setCopied(ok ? "copied" : "failed");
      announce(ok ? t.copied : t.copyFailed);
    });
  };
  // The handle (and the head around it) pulls the phone sheet down; far enough closes it.
  const grab = {
    onPointerDown: (event: PointerEvent) => {
      if ((event.target as Element).closest("button") || event.button !== 0 || !window.matchMedia(SHEET).matches) return;
      drag.current = event.clientY;
      (event.currentTarget as Element).setPointerCapture(event.pointerId);
    },
    onPointerMove: (event: PointerEvent) => {
      if (drag.current !== null) setPull(Math.max(0, event.clientY - drag.current));
    },
    onPointerUp: (event: PointerEvent) => {
      if (drag.current === null) return;
      const distance = event.clientY - drag.current;
      drag.current = null;
      if (distance > SWIPE_CLOSE) onClose();
      else setPull(0);
    },
    onPointerCancel: () => {
      drag.current = null;
      setPull(0);
    },
  };

  return (
    <div
      class="veil opts-veil"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialog}
        class={pull > 0 ? "dialog opts-dialog is-pulled" : "dialog opts-dialog"}
        role="dialog"
        aria-modal="true"
        aria-labelledby="opts-page"
        tabIndex={-1}
        style={pull > 0 ? { transform: `translateY(${pull}px)` } : undefined}
      >
        <div class="sheet-grab" {...grab}>
          <span class="sheet-handle" aria-hidden="true" />
          <div class="dialog-head sheet-head">
            {page !== null && (
              <button type="button" class="icon-btn sheet-back" aria-label={t.back} title={t.back} onClick={back}>
                <Icon name="CaretLeft" size={16} />
              </button>
            )}
            <h2 id="opts-page" class="dialog-title">
              {title}
            </h2>
            <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={onClose}>
              <Icon name="X" size={16} />
            </button>
          </div>
        </div>
        <div ref={body} class="sheet-body">
          {content}
        </div>
        <div class="sheet-foot">
          <div class="sheet-cli">
            <code ref={cli} class="cli-text" role="group" tabIndex={0} aria-label={t.cliAria} title={command} onScroll={(event) => markMore(event.currentTarget)}>
              <span class="cli-prompt" aria-hidden="true">
                ${" "}
              </span>
              {command}
            </code>
            {pathsNote && (
              <HelpTooltip text={t.cliPathsNote}>
                {(describedBy) => (
                  <button type="button" class="icon-btn cli-info" aria-label={t.cliPathsInfo} aria-describedby={describedBy}>
                    <Icon name="Info" size={15} />
                  </button>
                )}
              </HelpTooltip>
            )}
            <button
              type="button"
              class="icon-btn sheet-copy"
              aria-label={copied === "copied" ? t.copied : copied === "failed" ? t.copyFailed : t.copyCommand}
              title={t.copyCommand}
              onClick={copy}
            >
              <Icon name={copied === "copied" ? "Check" : "Copy"} size={15} />
            </button>
          </div>
          {defaults && <DefaultsBar t={t} state={state} presets={presets} control={defaults} llmReady={llmReady} changed={changed.size > 0} />}
        </div>
      </div>
    </div>
  );
}

/** Reset to defaults and Save as default, with the same meaning as before the dialog. */
function DefaultsBar({
  t,
  state,
  presets,
  control,
  llmReady,
  changed,
}: {
  t: Dict;
  state: Composer;
  presets: PresetTable;
  control: DefaultsControl;
  llmReady: boolean;
  changed: boolean;
}) {
  const { view, saving, failure } = control;
  // Without a model the defaults apply without LLM (the composer turns it off), so they still count as met.
  const same = sameAsDefaults(state, llmReady ? view.defaults : { ...view.defaults, llm: false }, presets);
  const remote = defaultsOf(state, presets) === null;
  const own = view.origin === "user";
  return (
    <div class="sheet-defaults" role="group" aria-label={t.defaultsGroup}>
      <div class="sheet-defaults-actions">
        {own && view.can_save && (
          <button type="button" class="text-link" disabled={saving} onClick={control.useServer}>
            {t.defaultsUseServer}
          </button>
        )}
        {/* aria-disabled, not disabled: the pressed button turns unavailable and must keep focus */}
        <button type="button" class="btn btn-sm" aria-disabled={!changed || undefined} onClick={() => changed && control.reset()}>
          {t.defaultsReset}
        </button>
        {view.can_save && (
          <button
            type="button"
            class="btn btn-sm"
            aria-disabled={same || remote || saving || undefined}
            aria-busy={saving || undefined}
            aria-describedby="opts-defaults-scope"
            onClick={() => !(same || remote || saving) && control.save()}
          >
            {t.defaultsSave}
          </button>
        )}
        <span id="opts-defaults-scope" hidden>
          {view.scope === "user" ? t.defaultsScopeUser : t.defaultsScopeServer}
        </span>
      </div>
      {view.can_save && remote && <p class="sheet-note">{t.defaultsNoCloudflare}</p>}
      {failure && (
        <p class="sheet-note line-error" role="alert">
          {failure.stale ? t.defaultsStale : failure.text}
        </p>
      )}
    </div>
  );
}

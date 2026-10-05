// The composer: the source card holds the URL line, whose bottom row carries
// the tools (Options, Upload) on the left and Convert on the right, and the
// options drawer below it; the drawer ends with the CLI command line.
import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import type { CloudflareCapability, ConversionBackend, FetchStrategy, OutputProfile, Preset } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { cloudflareReason } from "../lib/cloudflare.ts";
import { cliCommand } from "../lib/cli.ts";
import { copyText } from "../lib/clipboard.ts";
import {
  applyPreset,
  BUILTIN_PRESETS,
  changeAdvanced,
  featuresOf,
  hasAdvancedChoices,
  matchingPreset,
  PRESETS,
  resolveOptions,
  type Advanced,
  type Composer,
  type PresetTable,
} from "../lib/options.ts";
import { UploadPicker } from "./file-picker.tsx";
import { HelpTooltip, helpItemText, type HelpItem } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";

const STRATEGIES: FetchStrategy[] = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
const BACKENDS: ConversionBackend[] = ["native", "cloudflare"];
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];

let panels = 0;

/** A choice's help is read with the control, not shown on hover: the row's help lists it. */
function Described({ id, item }: { id: string; item: HelpItem }) {
  return (
    <span id={id} hidden>
      {helpItemText(item)}
    </span>
  );
}

function Chip({
  id,
  label,
  on,
  disabled = false,
  help,
  onToggle,
}: {
  id: string;
  label: string;
  on: boolean;
  disabled?: boolean;
  help: HelpItem;
  onToggle: (value: boolean) => void;
}) {
  return (
    <>
      <button
        type="button"
        role="switch"
        class="chip"
        aria-checked={on}
        aria-label={label}
        aria-describedby={id}
        disabled={disabled}
        onClick={() => onToggle(!on)}
      >
        {label}
      </button>
      <Described id={id} item={help} />
    </>
  );
}

function Segments<T extends string | null>({
  labelledBy,
  value,
  choices,
  label,
  help,
  disabled,
  focusableDisabled = false,
  onPick,
}: {
  labelledBy: string;
  /** null selects no choice: the row is in a state none of them describe. */
  value: T | null;
  choices: readonly T[];
  label: (choice: T) => string;
  help: (choice: T) => HelpItem;
  disabled?: (choice: T) => boolean;
  focusableDisabled?: boolean;
  onPick: (choice: T) => void;
}) {
  const describedBy = (choice: T) => `${labelledBy}-${choice ?? "default"}`;
  return (
    <>
      <span class="seg" role="group" aria-labelledby={labelledBy}>
        {choices.map((choice) => {
          const off = disabled?.(choice) ?? false;
          return (
            <button
              key={choice ?? "default"}
              type="button"
              class={choice === value ? "is-on" : undefined}
              aria-pressed={choice === value}
              aria-describedby={describedBy(choice)}
              disabled={off && !focusableDisabled}
              aria-disabled={off || undefined}
              onClick={() => {
                if (!off) onPick(choice);
              }}
            >
              {label(choice)}
            </button>
          );
        })}
      </span>
      {choices.map((choice) => (
        <Described key={choice ?? "default"} id={describedBy(choice)} item={help(choice)} />
      ))}
    </>
  );
}

function RowLabel({ id, text, hint, items, helpLabel }: { id: string; text: string; hint: string; items: HelpItem[]; helpLabel: string }) {
  return (
    <span class="opt-label">
      <span id={id}>{text}</span>
      <HelpTooltip text={hint} items={items}>
        {(describedBy) => (
          <button type="button" class="opt-help" aria-label={helpLabel} aria-describedby={describedBy}>
            <Icon name="Info" size={13} />
          </button>
        )}
      </HelpTooltip>
    </span>
  );
}

export function OptionsPanel({
  t,
  state,
  presets = BUILTIN_PRESETS,
  llmReady,
  cloudflare,
  urls,
  announce,
  source,
  heading,
  actions,
  busy,
  onChange,
  onFiles,
  onFolder,
}: {
  t: Dict;
  state: Composer;
  presets?: PresetTable;
  llmReady: boolean;
  cloudflare?: CloudflareCapability;
  urls: string[];
  announce: (message: string) => void;
  /** The URL line, given the tools for its bottom row. */
  source: (tools: ComponentChildren) => ComponentChildren;
  heading?: ComponentChildren;
  actions?: ComponentChildren;
  busy: boolean;
  onChange: (next: Composer) => void;
  onFiles: (files: File[]) => void;
  onFolder: (files: File[], hidden: number) => void;
}) {
  const [id] = useState(() => `opts-${++panels}`);
  const [open, setOpen] = useState(false);
  // The advanced fold starts open only when something inside already differs.
  const [advOpen, setAdvOpen] = useState(() => hasAdvancedChoices(state.advanced));
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => {
    if (copied === "idle") return;
    const timer = setTimeout(() => setCopied("idle"), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const reason = cloudflareReason(cloudflare);
  const a = state.advanced;
  const effective = resolveOptions(state, presets);
  const selectionReason = cloudflareReason(cloudflare, effective);
  const matched = matchingPreset(state, presets);
  const shownPreset = matched ?? state.preset;
  const analysisOff = !state.llm || a.pure;
  const set = <K extends keyof Advanced>(key: K, value: Advanced[K]) =>
    onChange({ ...state, advanced: changeAdvanced(a, key, value) });
  const presetLabel: Record<Preset, string> = { minimal: t.presetMinimal, standard: t.presetStandard, rich: t.presetRich };
  const presetOff = (preset: Preset) => !llmReady && featuresOf(preset, presets).llm;
  const presetHelp = (preset: Preset): HelpItem => {
    const features = featuresOf(preset, presets);
    const builtin = BUILTIN_PRESETS[preset];
    const changed = (Object.keys(builtin) as (keyof typeof builtin)[]).some((key) => features[key] !== builtin[key]);
    return {
      label: presetLabel[preset],
      text: changed ? t.helpServerPreset : { minimal: t.helpMinimal, standard: t.helpStandard, rich: t.helpRich }[preset],
      note: presetOff(preset) ? t.advNeedsModel : undefined,
    };
  };
  const profileLabel = (profile: OutputProfile | null) =>
    ({ default: t.profileNone, rag: t.profileRag, obsidian: t.profileObsidian, okf: t.profileOkf })[profile ?? "default"];
  const profileHelp = (profile: OutputProfile | null): HelpItem => ({
    label: profileLabel(profile),
    text: ({ default: t.helpDefault, rag: t.helpRag, obsidian: t.helpObsidian, okf: t.helpOkf })[profile ?? "default"],
  });
  const strategyLabel: Record<FetchStrategy, string> = {
    auto: t.strategyAuto,
    static: t.strategyStatic,
    playwright: t.strategyPlaywright,
    defuddle: t.strategyDefuddle,
    jina: t.strategyJina,
    cloudflare: t.strategyCloudflare,
  };
  const strategyHint: Record<FetchStrategy, string> = {
    auto: t.helpAuto,
    static: t.helpStatic,
    playwright: t.helpPlaywright,
    defuddle: t.helpDefuddle,
    jina: t.helpJina,
    cloudflare: t.helpCloudflareUrl,
  };
  const strategyHelp = (value: FetchStrategy): HelpItem => ({
    label: strategyLabel[value],
    text: strategyHint[value],
    note: value === "cloudflare" && reason ? t.cloudflareReason(reason) : undefined,
  });
  const backendLabel = (value: ConversionBackend) => (value === "native" ? t.backendNative : t.backendCloudflare);
  const backendHelp = (value: ConversionBackend): HelpItem => ({
    label: backendLabel(value),
    text: value === "native" ? t.helpNative : t.helpCloudflareFile,
    note: value === "cloudflare" && reason ? t.cloudflareReason(reason) : undefined,
  });
  const analysisNote = analysisOff ? (a.pure ? t.advPlainImages : t.advNeedsLlm) : undefined;
  const help = {
    llm: { label: t.llmEnhance, text: t.helpLlm, note: llmReady ? undefined : t.advNeedsModel },
    ocr: { label: t.ocr, text: t.helpOcr },
    alt: { label: t.advAlt, text: t.helpAlt, note: analysisNote },
    desc: { label: t.advDesc, text: t.helpDesc, note: analysisNote },
    screenshot: { label: t.advScreenshot, text: t.helpScreenshot, note: a.screenshotOnly ? t.advImpliedBySource : undefined },
    screenshotOnly: { label: t.advScreenshotOnly, text: t.helpSource },
    pure: { label: t.advPure, text: t.helpPure },
    noCache: { label: t.advNoCache, text: t.helpCache },
    noCompress: { label: t.advNoCompress, text: t.helpCompress },
  } satisfies Record<string, HelpItem>;
  const remoteNotice = effective.strategy === "defuddle" ? t.noticeDefuddle : effective.strategy === "jina" ? t.noticeJina : null;
  const command = cliCommand(urls, { ...effective, preset: shownPreset }, presets);
  const copy = () => {
    void copyText(command).then((ok) => {
      setCopied(ok ? "copied" : "failed");
      announce(ok ? t.copied : t.copyFailed);
    });
  };

  return (
    <div class="composer">
      {heading && (
        <div class="work-head">
          {heading}
          {actions && <div class="work-head-tools">{actions}</div>}
        </div>
      )}
      <div class="source-card">
        <div class="source-row">
          {source(
            <div class="tool-group" role="group" aria-label={t.sourceActions}>
              <button
                type="button"
                class={open ? "tool is-on" : "tool"}
                aria-label={t.options}
                title={t.options}
                aria-expanded={open}
                aria-controls={open ? id : undefined}
                onClick={() => setOpen((value) => !value)}
              >
                <Icon name="Sliders" size={14} />
                <span class="tool-text">{t.options}</span>
              </button>
              <UploadPicker t={t} disabled={busy} onFiles={onFiles} onFolder={onFolder} />
            </div>,
          )}
        </div>
        {open && (
          <div class="opts" id={id}>
            <div class="opts-grid">
              <div class="opt-row">
                <RowLabel id={`${id}-preset`} text={t.preset} hint={t.presetHint} items={PRESETS.map(presetHelp)} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  {/* Only a request that still equals a preset marks it: a
                      customized set leaves all three plain and says so itself. */}
                  <Segments
                    labelledBy={`${id}-preset`}
                    value={matched}
                    choices={PRESETS}
                    label={(preset) => presetLabel[preset]}
                    help={presetHelp}
                    disabled={presetOff}
                    onPick={(preset) => onChange(applyPreset(state, preset, presets))}
                  />
                  {matched === null && (
                    <span class="opt-custom" role="status">
                      {t.presetCustomized}
                    </span>
                  )}
                </div>
              </div>
              <div class="opt-row" role="group" aria-labelledby={`${id}-enhance`}>
                <RowLabel id={`${id}-enhance`} text={t.advEnhance} hint={t.helpEnhance} items={[help.llm, help.ocr]} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  <Chip
                    id={`${id}-llm`}
                    label={t.llmEnhance}
                    on={state.llm}
                    disabled={!llmReady}
                    help={help.llm}
                    onToggle={(llm) => onChange({ ...state, llm })}
                  />
                  <Chip id={`${id}-ocr`} label={t.ocr} on={state.ocr} help={help.ocr} onToggle={(ocr) => onChange({ ...state, ocr })} />
                  {state.ocr && <p class="opt-note">{state.llm ? t.advVlmOcr : t.advLocalOcr}</p>}
                </div>
              </div>
              <div class="opt-row" role="group" aria-labelledby={`${id}-images`}>
                <RowLabel
                  id={`${id}-images`}
                  text={t.advImages}
                  hint={t.helpImages}
                  items={[help.alt, help.desc, help.screenshot]}
                  helpLabel={t.helpLabel}
                />
                <div class="opt-controls">
                  <Chip
                    id={`${id}-alt`}
                    label={t.advAlt}
                    on={effective.alt === true}
                    disabled={analysisOff}
                    help={help.alt}
                    onToggle={(value) => set("alt", value)}
                  />
                  <Chip
                    id={`${id}-desc`}
                    label={t.advDesc}
                    on={effective.desc === true}
                    disabled={analysisOff}
                    help={help.desc}
                    onToggle={(value) => set("desc", value)}
                  />
                  <Chip
                    id={`${id}-screenshot`}
                    label={t.advScreenshot}
                    on={effective.screenshot === true}
                    disabled={a.screenshotOnly}
                    help={help.screenshot}
                    onToggle={(value) => set("screenshot", value)}
                  />
                </div>
              </div>
              <div class="opt-row">
                <RowLabel id={`${id}-output`} text={t.advOutput} hint={t.helpOutput} items={PROFILES.map(profileHelp)} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  <Segments
                    labelledBy={`${id}-output`}
                    value={state.profile}
                    choices={PROFILES}
                    label={profileLabel}
                    help={profileHelp}
                    onPick={(profile) => onChange({ ...state, profile })}
                  />
                </div>
              </div>
              <div class={advOpen ? "opt-row opt-fold is-open" : "opt-row opt-fold"}>
                <HelpTooltip text={t.helpAdvanced}>
                  {(describedBy) => (
                    <button
                      type="button"
                      class="opt-label opt-fold-btn"
                      aria-expanded={advOpen}
                      aria-controls={advOpen ? `${id}-adv` : undefined}
                      aria-describedby={describedBy}
                      onClick={() => setAdvOpen((value) => !value)}
                    >
                      <Icon name="CaretRight" size={11} />
                      <span>{t.advanced}</span>
                    </button>
                  )}
                </HelpTooltip>
              </div>
              {advOpen && (
                <div class="opt-fold-body" id={`${id}-adv`}>
                  <div class="opt-row">
                    <RowLabel
                      id={`${id}-strategy`}
                      text={t.advStrategy}
                      hint={t.helpStrategy}
                      items={STRATEGIES.map(strategyHelp)}
                      helpLabel={t.helpLabel}
                    />
                    <div class="opt-controls">
                      <Segments
                        labelledBy={`${id}-strategy`}
                        value={a.strategy}
                        choices={STRATEGIES}
                        label={(value) => strategyLabel[value]}
                        help={strategyHelp}
                        disabled={(value) => value === "cloudflare" && (reason !== null || !cloudflare?.browser_rendering)}
                        focusableDisabled
                        onPick={(value) => set("strategy", value)}
                      />
                      {remoteNotice && <p class="opt-note">{remoteNotice}</p>}
                    </div>
                  </div>
                  <div class="opt-row">
                    <RowLabel id={`${id}-backend`} text={t.advBackend} hint={t.helpBackend} items={BACKENDS.map(backendHelp)} helpLabel={t.helpLabel} />
                    <div class="opt-controls">
                      <Segments
                        labelledBy={`${id}-backend`}
                        value={a.backend}
                        choices={BACKENDS}
                        label={backendLabel}
                        help={backendHelp}
                        disabled={(value) => value === "cloudflare" && (reason !== null || !cloudflare?.file_conversion)}
                        focusableDisabled
                        onPick={(value) => set("backend", value)}
                      />
                    </div>
                  </div>
                  {reason && a.strategy !== "cloudflare" && a.backend !== "cloudflare" && <div class="opt-row">
                    <span class="opt-label">Cloudflare</span>
                    <div class="opt-controls"><p class="opt-note">{t.cloudflareReason(reason)}</p><details><summary>{t.helpLabel}</summary><p class="opt-note">{t.cloudflareSetup}</p></details></div>
                  </div>}
                  <div class="opt-row" role="group" aria-labelledby={`${id}-other`}>
                    <RowLabel
                      id={`${id}-other`}
                      text={t.advOther}
                      hint={t.helpOther}
                      items={[help.screenshotOnly, help.pure, help.noCache, help.noCompress]}
                      helpLabel={t.helpLabel}
                    />
                    <div class="opt-controls">
                      <Chip
                        id={`${id}-screenshot-only`}
                        label={t.advScreenshotOnly}
                        on={a.screenshotOnly}
                        help={help.screenshotOnly}
                        onToggle={(value) => set("screenshotOnly", value)}
                      />
                      <Chip id={`${id}-pure`} label={t.advPure} on={a.pure} help={help.pure} onToggle={(value) => set("pure", value)} />
                      <Chip id={`${id}-no-cache`} label={t.advNoCache} on={a.noCache} help={help.noCache} onToggle={(value) => set("noCache", value)} />
                      <Chip
                        id={`${id}-no-compress`}
                        label={t.advNoCompress}
                        on={a.noCompress}
                        help={help.noCompress}
                        onToggle={(value) => set("noCompress", value)}
                      />
                      {(a.pure || a.screenshotOnly) && <p class="opt-note">{t.advSourceExclusive}</p>}
                      {a.screenshotOnly && <p class="opt-note">{state.llm ? t.advScreenshotOnlyHint : t.advCaptureOnly}</p>}
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div class="cli-bar">
              <div class="cli-body">
                <code class="cli-text" role="group" tabIndex={0} aria-label={t.cliAria}>
                  <span class="cli-prompt" aria-hidden="true">
                    ${" "}
                  </span>
                  {command}
                </code>
              </div>
              <button type="button" class="pill-btn" onClick={copy}>
                {copied === "copied" ? t.copied : copied === "failed" ? t.copyFailed : t.copy}
              </button>
            </div>
          </div>
        )}
        {(a.strategy === "cloudflare" || a.backend === "cloudflare") && <div class="cloudflare-note">
          {selectionReason && <p role="alert">{t.cloudflareReason(selectionReason)}</p>}
          <p>{t.cloudflareScope}</p><p>{t.cloudflareCharges}</p>
          {selectionReason && selectionReason !== "incompatible_strategy" && <details><summary>{t.helpLabel}</summary><p>{t.cloudflareSetup}</p></details>}
        </div>}
      </div>
    </div>
  );
}

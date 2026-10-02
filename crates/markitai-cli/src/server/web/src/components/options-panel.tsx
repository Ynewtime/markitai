// The composer: a tool row (Options, CLI, Upload, Folder) above the source card,
// which holds the URL line, the options drawer and the CLI command line.
import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import type { ConversionBackend, FetchStrategy, OutputProfile, Preset } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
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
import { FilePicker, FolderPicker } from "./file-picker.tsx";
import { HelpTooltip } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";

const STRATEGIES: FetchStrategy[] = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
const BACKENDS: ConversionBackend[] = ["native", "cloudflare"];
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];
/** Accepted by the request schema but not implemented by this server. */
const UNSUPPORTED_STRATEGIES = new Set<FetchStrategy>(["cloudflare"]);
const UNSUPPORTED_BACKENDS = new Set<ConversionBackend>(["cloudflare"]);

let panels = 0;

function Chip({
  label,
  on,
  disabled = false,
  hint,
  onToggle,
}: {
  label: string;
  on: boolean;
  disabled?: boolean;
  hint: string;
  onToggle: (value: boolean) => void;
}) {
  return (
    <HelpTooltip text={hint} disabled={disabled}>
      {(describedBy) => (
        <button
          type="button"
          role="switch"
          class="chip"
          aria-checked={on}
          aria-label={label}
          aria-describedby={describedBy}
          disabled={disabled}
          onClick={() => onToggle(!on)}
        >
          {label}
        </button>
      )}
    </HelpTooltip>
  );
}

function Segments<T extends string | null>({
  labelledBy,
  value,
  choices,
  label,
  hint,
  disabled,
  onPick,
}: {
  labelledBy: string;
  value: T;
  choices: readonly T[];
  label: (choice: T) => string;
  hint: (choice: T) => string;
  disabled?: (choice: T) => boolean;
  onPick: (choice: T) => void;
}) {
  return (
    <span class="seg" role="group" aria-labelledby={labelledBy}>
      {choices.map((choice) => {
        const off = disabled?.(choice) ?? false;
        return (
          <HelpTooltip key={choice ?? "default"} text={hint(choice)} disabled={off}>
            {(describedBy) => (
              <button
                type="button"
                class={choice === value ? "is-on" : undefined}
                aria-pressed={choice === value}
                aria-describedby={describedBy}
                disabled={off}
                onClick={() => onPick(choice)}
              >
                {label(choice)}
              </button>
            )}
          </HelpTooltip>
        );
      })}
    </span>
  );
}

function RowLabel({ id, text, hint, helpLabel }: { id: string; text: string; hint: string; helpLabel: string }) {
  return (
    <span class="opt-label">
      <span id={id}>{text}</span>
      <HelpTooltip text={hint}>
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
  urls: string[];
  announce: (message: string) => void;
  source: ComponentChildren;
  heading?: ComponentChildren;
  actions?: ComponentChildren;
  busy: boolean;
  onChange: (next: Composer) => void;
  onFiles: (files: File[]) => void;
  onFolder: (files: File[], hidden: number) => void;
}) {
  const [id] = useState(() => `opts-${++panels}`);
  const [open, setOpen] = useState(false);
  const [cliOpen, setCliOpen] = useState(false);
  // The advanced fold starts open only when something inside already differs.
  const [advOpen, setAdvOpen] = useState(() => hasAdvancedChoices(state.advanced));
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => {
    if (copied === "idle") return;
    const timer = setTimeout(() => setCopied("idle"), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const a = state.advanced;
  const effective = resolveOptions(state, presets);
  const matched = matchingPreset(state, presets);
  const shownPreset = matched ?? state.preset;
  const analysisOff = !state.llm || a.pure;
  const set = <K extends keyof Advanced>(key: K, value: Advanced[K]) =>
    onChange({ ...state, advanced: changeAdvanced(a, key, value) });
  const presetLabel: Record<Preset, string> = { minimal: t.presetMinimal, standard: t.presetStandard, rich: t.presetRich };
  const presetHint = (preset: Preset) => {
    const features = featuresOf(preset, presets);
    if (!llmReady && features.llm) return t.advNeedsModel;
    const builtin = BUILTIN_PRESETS[preset];
    const changed = (Object.keys(builtin) as (keyof typeof builtin)[]).some((key) => features[key] !== builtin[key]);
    return changed ? t.helpServerPreset : { minimal: t.helpMinimal, standard: t.helpStandard, rich: t.helpRich }[preset];
  };
  const profileLabel = (profile: OutputProfile | null) =>
    ({ default: t.profileNone, rag: t.profileRag, obsidian: t.profileObsidian, okf: t.profileOkf })[profile ?? "default"];
  const profileHint = (profile: OutputProfile | null) =>
    ({ default: t.helpDefault, rag: t.helpRag, obsidian: t.helpObsidian, okf: t.helpOkf })[profile ?? "default"];
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
      <div class={heading ? "work-head" : "tool-bar"}>
        {heading}
        <div class="work-head-tools">
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
            <button
              type="button"
              class={cliOpen ? "tool tool-cli is-on" : "tool tool-cli"}
              aria-label={t.cliToggleAria}
              aria-expanded={cliOpen}
              aria-controls={cliOpen ? `${id}-cli` : undefined}
              onClick={() => setCliOpen((value) => !value)}
            >
              <Icon name="TerminalWindow" size={14} />
              <span>{t.cliToggle}</span>
            </button>
            <FilePicker t={t} disabled={busy} onFiles={onFiles} />
            <FolderPicker t={t} disabled={busy} onFolder={onFolder} />
          </div>
          {actions}
        </div>
      </div>
      <div class="source-card">
        <div class="source-row">{source}</div>
        {open && (
          <div class="opts" id={id}>
            <div class="opts-grid">
              <div class="opt-row">
                <RowLabel id={`${id}-preset`} text={t.preset} hint={t.presetHint} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  <Segments
                    labelledBy={`${id}-preset`}
                    value={shownPreset}
                    choices={PRESETS}
                    label={(preset) => presetLabel[preset]}
                    hint={presetHint}
                    disabled={(preset) => !llmReady && featuresOf(preset, presets).llm}
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
                <RowLabel
                  id={`${id}-enhance`}
                  text={t.advEnhance}
                  hint={llmReady ? t.helpEnhance : t.advNeedsModel}
                  helpLabel={t.helpLabel}
                />
                <div class="opt-controls">
                  <Chip
                    label={t.llmEnhance}
                    on={state.llm}
                    disabled={!llmReady}
                    hint={llmReady ? t.helpLlm : t.advNeedsModel}
                    onToggle={(llm) => onChange({ ...state, llm })}
                  />
                  <Chip label={t.ocr} on={state.ocr} hint={t.helpOcr} onToggle={(ocr) => onChange({ ...state, ocr })} />
                  {state.ocr && <p class="opt-note">{state.llm ? t.advVlmOcr : t.advLocalOcr}</p>}
                </div>
              </div>
              <div class="opt-row" role="group" aria-labelledby={`${id}-images`}>
                <RowLabel id={`${id}-images`} text={t.advImages} hint={t.helpImages} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  <Chip
                    label={t.advAlt}
                    on={effective.alt === true}
                    disabled={analysisOff}
                    hint={analysisOff ? (a.pure ? t.advPlainImages : t.advNeedsLlm) : t.helpAlt}
                    onToggle={(value) => set("alt", value)}
                  />
                  <Chip
                    label={t.advDesc}
                    on={effective.desc === true}
                    disabled={analysisOff}
                    hint={analysisOff ? (a.pure ? t.advPlainImages : t.advNeedsLlm) : t.helpDesc}
                    onToggle={(value) => set("desc", value)}
                  />
                  <Chip
                    label={t.advScreenshot}
                    on={effective.screenshot === true}
                    disabled={a.screenshotOnly}
                    hint={a.screenshotOnly ? t.advImpliedBySource : t.helpScreenshot}
                    onToggle={(value) => set("screenshot", value)}
                  />
                </div>
              </div>
              <div class="opt-row">
                <RowLabel id={`${id}-output`} text={t.advOutput} hint={t.helpOutput} helpLabel={t.helpLabel} />
                <div class="opt-controls">
                  <Segments
                    labelledBy={`${id}-output`}
                    value={state.profile}
                    choices={PROFILES}
                    label={profileLabel}
                    hint={profileHint}
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
                    <RowLabel id={`${id}-strategy`} text={t.advStrategy} hint={t.helpStrategy} helpLabel={t.helpLabel} />
                    <div class="opt-controls">
                      <Segments
                        labelledBy={`${id}-strategy`}
                        value={a.strategy}
                        choices={STRATEGIES}
                        label={(value) => strategyLabel[value]}
                        hint={(value) => strategyHint[value]}
                        disabled={(value) => UNSUPPORTED_STRATEGIES.has(value)}
                        onPick={(value) => set("strategy", value)}
                      />
                      {remoteNotice && <p class="opt-note">{remoteNotice}</p>}
                    </div>
                  </div>
                  <div class="opt-row">
                    <RowLabel id={`${id}-backend`} text={t.advBackend} hint={t.helpBackend} helpLabel={t.helpLabel} />
                    <div class="opt-controls">
                      <Segments
                        labelledBy={`${id}-backend`}
                        value={a.backend}
                        choices={BACKENDS}
                        label={(value) => (value === "native" ? t.backendNative : t.backendCloudflare)}
                        hint={(value) => (value === "native" ? t.helpNative : t.helpCloudflareFile)}
                        disabled={(value) => UNSUPPORTED_BACKENDS.has(value)}
                        onPick={(value) => set("backend", value)}
                      />
                    </div>
                  </div>
                  <div class="opt-row" role="group" aria-labelledby={`${id}-other`}>
                    <RowLabel id={`${id}-other`} text={t.advOther} hint={t.helpOther} helpLabel={t.helpLabel} />
                    <div class="opt-controls">
                      <Chip label={t.advScreenshotOnly} on={a.screenshotOnly} hint={t.helpSource} onToggle={(value) => set("screenshotOnly", value)} />
                      <Chip label={t.advPure} on={a.pure} hint={t.helpPure} onToggle={(value) => set("pure", value)} />
                      <Chip label={t.advNoCache} on={a.noCache} hint={t.helpCache} onToggle={(value) => set("noCache", value)} />
                      <Chip label={t.advNoCompress} on={a.noCompress} hint={t.helpCompress} onToggle={(value) => set("noCompress", value)} />
                      {(a.pure || a.screenshotOnly) && <p class="opt-note">{t.advSourceExclusive}</p>}
                      {a.screenshotOnly && <p class="opt-note">{state.llm ? t.advScreenshotOnlyHint : t.advCaptureOnly}</p>}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        {cliOpen && (
          <div class="cli-bar" id={`${id}-cli`}>
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
        )}
      </div>
    </div>
  );
}

import { useEffect, useId, useState, type ReactNode } from "react";
import type { ConversionBackend, FetchStrategy, OutputProfile, Preset } from "../api/types";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import {
  BUILTIN_PRESET_OPTIONS,
  changeAdvanced,
  hasExtraOptions,
  presetFeatures,
  matchingPreset,
  resolveOptions,
  type PresetOptions,
} from "../lib/conversionOptions";
import { buildCliCommand } from "../lib/cli";
import { copyTextToClipboard, type CopyState } from "../lib/clipboard";
import { FilePicker } from "./DropZone";
import { HelpTooltip } from "./HelpTooltip";
import { CaretRightIcon, InfoIcon, SlidersIcon, UploadIcon } from "./icons";

const STRATEGIES: FetchStrategy[] = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
const BACKENDS: ConversionBackend[] = ["native", "cloudflare"];
const PRESETS: Preset[] = ["minimal", "standard", "rich"];
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];

function Chip({ label, checked, disabled, hint, onChange }: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  hint?: string;
  onChange: (v: boolean) => void;
}) {
  return (
    <HelpTooltip text={hint ?? label}><button type="button" role="switch" aria-checked={checked} aria-label={label}
      disabled={disabled} className="chip" onClick={() => onChange(!checked)}>
      {label}
    </button></HelpTooltip>
  );
}

function Segments<T extends string | null>({ labelledBy, value, options, render, hint, disabled, onChange }: {
  labelledBy: string;
  value: T | null;
  options: readonly T[];
  render: (v: T) => string;
  hint: (v: T) => string;
  disabled?: (v: T) => boolean;
  onChange: (v: T) => void;
}) {
  return (
    <span className="seg" role="group" aria-labelledby={labelledBy}>
      {options.map((option) => (
        <HelpTooltip key={option ?? "default"} text={hint(option)}><button type="button"
          className={option === value ? "on" : undefined}
          aria-pressed={option === value} disabled={disabled?.(option)}
          onClick={() => onChange(option)}>
          {render(option)}
        </button></HelpTooltip>
      ))}
    </span>
  );
}

function RowLabel({ id, text, hint, helpLabel }: { id: string; text: string; hint?: string; helpLabel: string }) {
  return (
    <span className="optlbl">
      <span id={id}>{text}</span>
      {hint !== undefined && (
        // The tooltip wires the hint in through aria-describedby; the name stays short.
        <HelpTooltip text={hint}><button type="button" className="opthelp" aria-label={helpLabel}>
          <InfoIcon size={13} />
        </button></HelpTooltip>
      )}
    </span>
  );
}

export function OptionsPanel({
  t, value, preset, presetOptions = BUILTIN_PRESET_OPTIONS, llm, ocr, profile,
  llmConfigured, urls, announce, source, heading, headingActions, onFiles, onChange, onPreset, onLlm, onOcr, onProfile,
}: {
  t: Dict;
  value: Advanced;
  preset: Preset;
  presetOptions?: PresetOptions;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  llmConfigured: boolean;
  urls: string[];
  announce: (msg: string) => void;
  source?: ReactNode;
  heading?: ReactNode;
  headingActions?: ReactNode;
  onFiles?: (files: File[]) => void;
  onChange: (next: Advanced) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
  onProfile: (p: OutputProfile | null) => void;
}) {
  const [open, setOpen] = useState(false);
  // The rarely-touched fetch/source/cache settings sit behind one disclosure;
  // it starts open only when something in it already differs from default,
  // so a non-default choice is never hidden behind a closed fold.
  const [advOpen, setAdvOpen] = useState(() => hasExtraOptions(value));
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const id = useId();
  const state = { preset, llm, ocr, profile, advanced: value };
  const effective = resolveOptions(state, presetOptions);
  // An exact five-feature match highlights that bundle; otherwise the chosen
  // name stays pressed and the Custom badge says the bundle was adjusted.
  const matchedPreset = matchingPreset(state, presetOptions);
  const customized = matchedPreset === null;
  const shownPreset = matchedPreset ?? preset;
  const analysisDisabled = !llm || value.pure;
  const analysisHint = value.pure ? t.advPlainImages : t.advNeedsLlm;
  const forcedBackend = value.strategy === "cloudflare";
  const strategyHints = { auto: t.helpAuto, static: t.helpStatic, playwright: t.helpPlaywright,
    defuddle: t.helpDefuddle, jina: t.helpJina, cloudflare: t.helpCloudflareUrl };
  const backendHints = { native: t.helpNative, cloudflare: t.helpCloudflareFile };
  const remoteNotice = { auto: t.noticeAuto, static: null, playwright: null,
    defuddle: t.noticeDefuddle, jina: t.noticeJina, cloudflare: t.noticeCloudflareUrl }[effective.strategy];
  const bundleHint = (p: Preset) => {
    const features = presetFeatures(p, presetOptions);
    if (!llmConfigured && features.llm) return t.advNeedsModel;
    const builtin = BUILTIN_PRESET_OPTIONS[p];
    const overridden = (Object.keys(builtin) as (keyof typeof builtin)[])
      .some((key) => features[key] !== builtin[key]);
    return overridden ? t.helpServerPreset
      : { minimal: t.helpMinimal, standard: t.helpStandard, rich: t.helpRich }[p];
  };
  const set = <K extends keyof Advanced>(key: K, v: Advanced[K]) =>
    onChange(changeAdvanced(value, key, v));
  const command = buildCliCommand(urls, { ...effective, preset: shownPreset }, presetOptions);
  useEffect(() => {
    if (copyState === "idle") return;
    const handle = window.setTimeout(() => setCopyState("idle"), 1500);
    return () => window.clearTimeout(handle);
  }, [copyState]);

  const copy = () => {
    void copyTextToClipboard(command).then((ok) => {
      setCopyState(ok ? "copied" : "failed");
      announce(ok ? t.copied : t.copyFailed);
    });
  };

  const optionsToggle = (
    <button type="button" className={open ? "srcact optbtn on" : "srcact optbtn"}
      aria-label={t.options} title={t.options} aria-expanded={open}
      aria-controls={open ? id : undefined} onClick={() => setOpen((v) => !v)}>
      <SlidersIcon size={14} />
      <span>{t.options}</span>
    </button>
  );

  return (
    <div className="options">
      <div className={heading ? "jobhead" : "composer-toolbar"}>
        {heading}
        <div className="jobhead-r">
          <div className="source-tools" role="group" aria-label={t.sourceActions}>
            {optionsToggle}
            {onFiles && <FilePicker label={t.browse} onFiles={onFiles}
              icon={<UploadIcon />} className="srcact file-picker" />}
          </div>
          {headingActions}
        </div>
      </div>
      <div className="convert-source">
        {source && <div className="srcrow">{source}</div>}
        {open && (
          <div className="optpanel" id={id}>
            <div className="optrows">
              <div className="optrow">
                <RowLabel helpLabel={t.helpLabel} id={`${id}-preset`} text={t.preset} hint={t.presetHint} />
                <div className="optfield">
                  <Segments labelledBy={`${id}-preset`} value={shownPreset} options={PRESETS} render={(p) => p} hint={bundleHint}
                    disabled={(p) => !llmConfigured && presetFeatures(p, presetOptions).llm}
                    onChange={onPreset} />
                  {customized && <span className="optcustom" role="status">{t.presetCustomized}</span>}
                </div>
              </div>
              <div className="optrow" role="group" aria-labelledby={`${id}-enhance`}>
                <RowLabel helpLabel={t.helpLabel} id={`${id}-enhance`} text={t.advEnhance}
                  hint={llmConfigured ? t.helpEnhance : t.advNeedsModel} />
                <div className="optfield">
                  <Chip label={t.llmEnhance} checked={llm} disabled={!llmConfigured}
                    hint={llmConfigured ? t.helpLlm : t.advNeedsModel} onChange={onLlm} />
                  <Chip label={t.ocr} checked={ocr} hint={t.helpOcr} onChange={onOcr} />
                  {ocr && <p className="opthint">{llm ? t.advVlmOcr : t.advLocalOcr}</p>}
                </div>
              </div>
              <div className="optrow" role="group" aria-labelledby={`${id}-images`}>
                <RowLabel helpLabel={t.helpLabel} id={`${id}-images`} text={t.advImages}
                  hint={t.helpImages} />
                <div className="optfield">
                  <Chip label={t.advAlt} checked={effective.alt} disabled={analysisDisabled}
                    hint={analysisDisabled ? analysisHint : t.helpAlt} onChange={(v) => set("alt", v)} />
                  <Chip label={t.advDesc} checked={effective.desc} disabled={analysisDisabled}
                    hint={analysisDisabled ? analysisHint : t.helpDesc} onChange={(v) => set("desc", v)} />
                  <Chip label={t.advScreenshot} checked={effective.screenshot} disabled={value.screenshotOnly}
                    hint={value.screenshotOnly ? t.advImpliedBySource : t.helpScreenshot}
                    onChange={(v) => set("screenshot", v)} />
                </div>
              </div>
              <div className="optrow">
                <RowLabel helpLabel={t.helpLabel} id={`${id}-output`} text={t.advOutput} hint={t.helpOutput} />
                <div className="optfield">
                  <Segments labelledBy={`${id}-output`} value={profile} options={PROFILES}
                    hint={(p) => ({ default: t.helpDefault, rag: t.helpRag, obsidian: t.helpObsidian, okf: t.helpOkf })[p ?? "default"]}
                    render={(p) => p ?? t.profileNone} onChange={onProfile} />
                </div>
              </div>
              <div className={advOpen ? "optrow optadv on" : "optrow optadv"}>
                <HelpTooltip text={t.helpAdvanced}><button type="button" className="optlbl optadv-toggle"
                  aria-expanded={advOpen} aria-controls={advOpen ? `${id}-adv` : undefined}
                  onClick={() => setAdvOpen((v) => !v)}>
                  <CaretRightIcon size={11} />
                  <span>{t.advanced}</span>
                </button></HelpTooltip>
              </div>
              {advOpen && (
                <div className="optadv-body" id={`${id}-adv`}>
                  <div className="optrow">
                    <RowLabel helpLabel={t.helpLabel} id={`${id}-strategy`} text={t.advStrategy} hint={t.helpStrategy} />
                    <div className="optfield">
                      <Segments labelledBy={`${id}-strategy`} value={value.strategy} options={STRATEGIES}
                        render={(v) => v} hint={(v) => strategyHints[v]} onChange={(v) => set("strategy", v)} />
                      {remoteNotice && <p className="opthint">{remoteNotice}</p>}
                    </div>
                  </div>
                  <div className="optrow">
                    <RowLabel helpLabel={t.helpLabel} id={`${id}-backend`} text={t.advBackend}
                      hint={forcedBackend ? t.advCloudflareBackend : t.helpBackend} />
                    <div className="optfield">
                      <Segments labelledBy={`${id}-backend`} value={effective.backend} options={BACKENDS}
                        render={(v) => v} hint={(v) => forcedBackend ? t.advCloudflareBackend : backendHints[v]} disabled={() => forcedBackend}
                        onChange={(v) => set("backend", v)} />
                      {effective.backend === "cloudflare" && <p className="opthint">{t.noticeCloudflareFile}</p>}
                    </div>
                  </div>
                  <div className="optrow" role="group" aria-labelledby={`${id}-other`}>
                    <RowLabel helpLabel={t.helpLabel} id={`${id}-other`} text={t.advOther} hint={t.helpOther} />
                    <div className="optfield">
                      <Chip label={t.advScreenshotOnly} checked={effective.screenshot_only}
                        hint={t.helpSource} onChange={(v) => set("screenshotOnly", v)} />
                      <Chip label={t.advPure} checked={value.pure} hint={t.helpPure}
                        onChange={(v) => set("pure", v)} />
                      <Chip label={t.advNoCache} checked={value.noCache} hint={t.helpCache}
                        onChange={(v) => set("noCache", v)} />
                      <Chip label={t.advNoCompress} checked={value.noCompress} hint={t.helpCompress}
                        onChange={(v) => set("noCompress", v)} />
                      {(value.pure || value.screenshotOnly) && <p className="opthint">{t.advSourceExclusive}</p>}
                      {value.screenshotOnly && <p className="opthint">{llm ? t.advScreenshotOnlyHint : t.advCaptureOnly}</p>}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        <div className="optcli">
          <div className="clibody">
            <HelpTooltip text={t.helpCli}><code className="clitext" role="group" tabIndex={0} aria-label={t.cliAria}>
              <span className="clidollar" aria-hidden="true">${" "}</span>{command}
            </code></HelpTooltip>
          </div>
          <button type="button" className="badge" onClick={copy}>
            {copyState === "copied" ? t.copied : copyState === "failed" ? t.copyFailed : t.copy}
          </button>
        </div>
      </div>
    </div>
  );
}

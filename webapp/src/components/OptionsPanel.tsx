import { useId, useState } from "react";
import type {
  ConversionBackend,
  FetchStrategy,
  OutputProfile,
  Preset,
} from "../api/types";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import { advancedIsCustom } from "../lib/advanced";
import { SlidersIcon } from "./icons";

const STRATEGIES: FetchStrategy[] = [
  "auto",
  "static",
  "playwright",
  "defuddle",
  "jina",
  "cloudflare",
];
const BACKENDS: ConversionBackend[] = ["native", "kreuzberg", "cloudflare"];
const PRESETS: Preset[] = ["minimal", "standard", "rich"];
/** null first: no profile is the default and keeps output byte-identical. */
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];

function Check({
  label,
  checked,
  disabled,
  hint,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  hint?: string;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className={disabled ? "advcheck off" : "advcheck"} title={hint}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

function Switch({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <span className="advcheck">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        className={checked ? "switch on" : "switch"}
        onClick={() => onChange(!checked)}
      />
      <span>{label}</span>
    </span>
  );
}

function Segments<T extends string | null>({
  label,
  value,
  options,
  render,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  render: (v: T) => string;
  onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <span className="advselect">
      <span id={id}>{label}</span>
      <span className="seg" role="group" aria-labelledby={id}>
        {options.map((option) => (
          <button
            key={option ?? "default"}
            type="button"
            className={option === value ? "on" : undefined}
            aria-pressed={option === value}
            onClick={() => onChange(option)}
          >
            {render(option)}
          </button>
        ))}
      </span>
    </span>
  );
}

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <span className="advselect">
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </span>
  );
}

/** What the collapsed button says beyond its own name.
 *
 * Every conversion option now lives behind this disclosure, so a summary is
 * not decoration: without it, turning LLM on and collapsing the panel would
 * leave nothing on screen saying the next conversion costs money. */
function summary(t: Dict, llm: boolean, ocr: boolean, profile: OutputProfile | null, custom: boolean): string[] {
  const parts: string[] = [];
  if (llm) parts.push(t.llmEnhanceShort);
  if (ocr) parts.push(t.ocr);
  if (profile) parts.push(profile);
  if (custom) parts.push("+");
  return parts;
}

export function OptionsPanel({
  t,
  value,
  preset,
  llm,
  ocr,
  profile,
  llmConfigured,
  onChange,
  onPreset,
  onLlm,
  onOcr,
  onProfile,
}: {
  t: Dict;
  value: Advanced;
  preset: Preset;
  /** Image analysis is LLM work; the CLI rejects `--alt`/`--desc` without
   * `--llm`, so those rows are shown disabled rather than hidden — the
   * capability stays visible, with the reason. */
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  /** No routable deployment means the LLM rows would promise what the server
   * cannot do, so they are left out entirely rather than shown broken. */
  llmConfigured: boolean;
  onChange: (next: Advanced) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
  onProfile: (p: OutputProfile | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const set = <K extends keyof Advanced>(key: K, v: Advanced[K]) =>
    onChange({ ...value, [key]: v });
  const chips = summary(t, llm, ocr, profile, advancedIsCustom(value));

  return (
    <>
      <button
        type="button"
        className={open ? "clibtn on" : "clibtn"}
        aria-label={t.options}
        title={t.options}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        <SlidersIcon size={14} />
        <span>{t.options}</span>
        {chips.length > 0 && (
          <span className="advchips" aria-label={t.optionsChanged}>
            {chips.join(" · ")}
          </span>
        )}
      </button>
      {open && (
        <div className="advwrap" id={id}>
          {llmConfigured && (
            <div className="advrow">
              <span className="advlbl">{t.advEnhance}</span>
              <Switch label={t.llmEnhance} checked={llm} onChange={onLlm} />
              <Switch label={t.ocr} checked={ocr} onChange={onOcr} />
              {llm && (
                <Segments
                  label={t.preset}
                  value={preset}
                  options={PRESETS}
                  render={(p) => p}
                  onChange={onPreset}
                />
              )}
            </div>
          )}
          {!llmConfigured && (
            <div className="advrow">
              <span className="advlbl">{t.advEnhance}</span>
              <Switch label={t.ocr} checked={ocr} onChange={onOcr} />
            </div>
          )}
          <div className="advrow">
            <span className="advlbl">{t.advOutput}</span>
            <Segments
              label={t.profile}
              value={profile}
              options={PROFILES}
              render={(p) => p ?? t.profileNone}
              onChange={onProfile}
            />
          </div>
          <div className="advrow">
            <span className="advlbl">{t.advImages}</span>
            <Check
              label={t.advAlt}
              checked={value.alt}
              disabled={!llm}
              hint={llm ? undefined : t.advNeedsLlm}
              onChange={(v) => set("alt", v)}
            />
            <Check
              label={t.advDesc}
              checked={value.desc}
              disabled={!llm}
              hint={llm ? undefined : t.advNeedsLlm}
              onChange={(v) => set("desc", v)}
            />
            <Check
              label={t.advScreenshot}
              // --screenshot-only implies --screenshot, so the box shows the
              // state that will actually be sent rather than going stale.
              checked={value.screenshot || value.screenshotOnly}
              disabled={value.screenshotOnly}
              hint={value.screenshotOnly ? t.advImpliedBySource : undefined}
              onChange={(v) => set("screenshot", v)}
            />
          </div>
          <div className="advrow">
            <span className="advlbl">{t.advSource}</span>
            <Check
              label={t.advScreenshotOnly}
              checked={value.screenshotOnly}
              hint={t.advScreenshotOnlyHint}
              onChange={(v) => set("screenshotOnly", v)}
            />
            <Check
              label={t.advPure}
              checked={value.pure}
              hint={t.advPureHint}
              onChange={(v) => set("pure", v)}
            />
          </div>
          <div className="advrow">
            <span className="advlbl">{t.advFetch}</span>
            <Select
              label={t.advStrategy}
              value={value.strategy}
              options={STRATEGIES}
              onChange={(v) => set("strategy", v)}
            />
            <Select
              label={t.advBackend}
              value={value.backend}
              options={BACKENDS}
              onChange={(v) => set("backend", v)}
            />
          </div>
          <div className="advrow">
            <span className="advlbl">{t.advOther}</span>
            <Check
              label={t.advNoCache}
              checked={value.noCache}
              hint={t.advNoCacheHint}
              onChange={(v) => set("noCache", v)}
            />
            <Check
              label={t.advNoCompress}
              checked={value.noCompress}
              onChange={(v) => set("noCompress", v)}
            />
          </div>
        </div>
      )}
    </>
  );
}

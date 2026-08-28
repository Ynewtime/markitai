import { useId, useState } from "react";
import type { ConversionBackend, FetchStrategy } from "../api/types";
import type { Advanced } from "../lib/advanced";
import { advancedIsCustom } from "../lib/advanced";
import type { Dict } from "../i18n";
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

export function AdvancedOptions({
  t,
  value,
  llm,
  onChange,
}: {
  t: Dict;
  value: Advanced;
  /** Image analysis is LLM work; the CLI rejects `--alt`/`--desc` without
   * `--llm`, so those rows are shown disabled rather than hidden — the
   * capability stays visible, with the reason. */
  llm: boolean;
  onChange: (next: Advanced) => void;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const set = <K extends keyof Advanced>(key: K, v: Advanced[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <>
      <button
        type="button"
        className={open ? "clibtn on" : "clibtn"}
        aria-label={t.advanced}
        title={t.advanced}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        <SlidersIcon size={14} />
        <span>{t.advanced}</span>
        {advancedIsCustom(value) && (
          <span className="advdot" aria-label={t.advancedChanged} />
        )}
      </button>
      {open && (
        <div className="advwrap" id={id}>
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

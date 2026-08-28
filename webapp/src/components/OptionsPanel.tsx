import { useEffect, useId, useState } from "react";
import type {
  ConversionBackend,
  FetchStrategy,
  OutputProfile,
  Preset,
} from "../api/types";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import { advancedIsCustom } from "../lib/advanced";
import { buildCliCommand } from "../lib/cli";
import { copyTextToClipboard } from "../lib/clipboard";
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

type CopyState = "idle" | "copied" | "failed";

/** One shape for every boolean, so a row reads as a set of things you can
 * turn on rather than a mix of switches, checkboxes and buttons. */
function Chip({
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
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      title={hint}
      disabled={disabled}
      className="chip"
      onClick={() => onChange(!checked)}
    >
      {label}
    </button>
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
    <>
      <span className="seg" role="group" aria-label={label} id={id}>
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
    </>
  );
}

function Pick<T extends string>({
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
    <span className="advpick">
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
 * Every conversion option lives behind this disclosure, so a summary is not
 * decoration: without it, turning LLM on and collapsing would leave nothing
 * on screen saying the next conversion costs money. */
function summary(
  t: Dict,
  llm: boolean,
  ocr: boolean,
  profile: OutputProfile | null,
  custom: boolean,
): string[] {
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
  urls,
  announce,
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
   * `--llm`, so those chips are shown disabled rather than hidden — the
   * capability stays visible, with the reason. */
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  /** No routable deployment means the LLM chips would promise what the server
   * cannot do, so they are left out entirely rather than shown broken. */
  llmConfigured: boolean;
  urls: string[];
  announce: (msg: string) => void;
  onChange: (next: Advanced) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
  onProfile: (p: OutputProfile | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const id = useId();
  const set = <K extends keyof Advanced>(key: K, v: Advanced[K]) =>
    onChange({ ...value, [key]: v });
  const chips = summary(t, llm, ocr, profile, advancedIsCustom(value));
  const command = buildCliCommand(urls, preset, llm, ocr, profile);

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
          <div className="advcard">
            <div className="advrows">
              <div className="advrow">
                <span className="advlbl">{t.advEnhance}</span>
                <span className="advfield">
                  {llmConfigured && (
                    <Chip label={t.llmEnhance} checked={llm} onChange={onLlm} />
                  )}
                  <Chip label={t.ocr} checked={ocr} onChange={onOcr} />
                  {llmConfigured && llm && (
                    <Segments
                      label={t.preset}
                      value={preset}
                      options={PRESETS}
                      render={(p) => p}
                      onChange={onPreset}
                    />
                  )}
                </span>
              </div>
              <div className="advrow">
                <span className="advlbl">{t.advOutput}</span>
                <span className="advfield">
                  <Segments
                    label={t.profile}
                    value={profile}
                    options={PROFILES}
                    render={(p) => p ?? t.profileNone}
                    onChange={onProfile}
                  />
                </span>
              </div>
              <div className="advrow">
                <span className="advlbl">{t.advImages}</span>
                <span className="advfield">
                  <Chip
                    label={t.advAlt}
                    checked={value.alt}
                    disabled={!llm}
                    hint={llm ? undefined : t.advNeedsLlm}
                    onChange={(v) => set("alt", v)}
                  />
                  <Chip
                    label={t.advDesc}
                    checked={value.desc}
                    disabled={!llm}
                    hint={llm ? undefined : t.advNeedsLlm}
                    onChange={(v) => set("desc", v)}
                  />
                  <Chip
                    label={t.advScreenshot}
                    // --screenshot-only implies --screenshot, so the chip shows
                    // the state that will actually be sent rather than a stale
                    // one.
                    checked={value.screenshot || value.screenshotOnly}
                    disabled={value.screenshotOnly}
                    hint={
                      value.screenshotOnly ? t.advImpliedBySource : undefined
                    }
                    onChange={(v) => set("screenshot", v)}
                  />
                </span>
              </div>
              <div className="advrow">
                <span className="advlbl">{t.advSource}</span>
                <span className="advfield">
                  <Chip
                    label={t.advScreenshotOnly}
                    checked={value.screenshotOnly}
                    hint={t.advScreenshotOnlyHint}
                    onChange={(v) => set("screenshotOnly", v)}
                  />
                  <Chip
                    label={t.advPure}
                    checked={value.pure}
                    hint={t.advPureHint}
                    onChange={(v) => set("pure", v)}
                  />
                </span>
              </div>
              <div className="advrow">
                <span className="advlbl">{t.advFetch}</span>
                <span className="advfield">
                  <Pick
                    label={t.advStrategy}
                    value={value.strategy}
                    options={STRATEGIES}
                    onChange={(v) => set("strategy", v)}
                  />
                  <Pick
                    label={t.advBackend}
                    value={value.backend}
                    options={BACKENDS}
                    onChange={(v) => set("backend", v)}
                  />
                </span>
              </div>
              <div className="advrow">
                <span className="advlbl">{t.advOther}</span>
                <span className="advfield">
                  <Chip
                    label={t.advNoCache}
                    checked={value.noCache}
                    hint={t.advNoCacheHint}
                    onChange={(v) => set("noCache", v)}
                  />
                  <Chip
                    label={t.advNoCompress}
                    checked={value.noCompress}
                    onChange={(v) => set("noCompress", v)}
                  />
                </span>
              </div>
            </div>
            <div className="advcli">
              <code className="clitext" tabIndex={0} aria-label={t.cliAria}>
                <span className="clidollar" aria-hidden="true">
                  ${" "}
                </span>
                {command}
              </code>
              <button type="button" className="badge" onClick={copy}>
                {copyState === "copied"
                  ? t.copied
                  : copyState === "failed"
                    ? t.copyFailed
                    : t.copy}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

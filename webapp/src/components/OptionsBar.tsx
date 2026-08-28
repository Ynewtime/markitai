import type { ReactNode } from "react";
import type { OutputProfile, Preset } from "../api/types";
import { useMediaQuery } from "../hooks/useMediaQuery";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import { AdvancedOptions } from "./AdvancedOptions";
import { CliCommand } from "./CliCommand";

const PRESETS: Preset[] = ["minimal", "standard", "rich"];
/** null first: no profile is the default and keeps output byte-identical. */
const PROFILES: (OutputProfile | null)[] = [null, "rag", "obsidian", "okf"];

/** App's mobile breakpoint (app.css ≤780px tier) — there the options row must
 * hold every toggle on one line at 360px, so the LLM label drops to its short
 * form. The switch keeps the full name in aria-label. */
const PHONE_Q = "(max-width: 780px)";

/** OCR is always available as a conversion option. LLM controls require a
 * routable deployment, and Preset remains a refinement of enabled LLM work. */
export function OptionsBar({
  t,
  preset,
  llm,
  ocr,
  profile,
  advanced,
  llmConfigured,
  urls,
  announce,
  onPreset,
  onLlm,
  onOcr,
  onProfile,
  onAdvanced,
  trailing,
}: {
  t: Dict;
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  advanced: Advanced;
  llmConfigured: boolean;
  urls: string[];
  announce: (msg: string) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
  onProfile: (p: OutputProfile | null) => void;
  onAdvanced: (a: Advanced) => void;
  /** Extra row member after the CLI disclosure — the workspace composer parks
   * its archive download at the row's right edge; home passes nothing. */
  trailing?: ReactNode;
}) {
  const phone = useMediaQuery(PHONE_Q);
  return (
    <div className="options">
      {llmConfigured && (
        <div className="opt">
          <span className="lbl">{phone ? t.llmEnhanceShort : t.llmEnhance}</span>
          <button
            type="button"
            role="switch"
            aria-checked={llm}
            aria-label={t.llmEnhance}
            className={llm ? "switch on" : "switch"}
            onClick={() => onLlm(!llm)}
          />
        </div>
      )}
      <div className="opt">
        <span className="lbl">{t.ocr}</span>
        <button
          type="button"
          role="switch"
          aria-checked={ocr}
          aria-label={t.ocr}
          className={ocr ? "switch on" : "switch"}
          onClick={() => onOcr(!ocr)}
        />
      </div>
      {llmConfigured && llm && (
        <div className="opt">
          <span className="lbl" id="preset-lbl">
            {t.preset}
          </span>
          <div className="seg" role="group" aria-labelledby="preset-lbl">
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                className={p === preset ? "on" : undefined}
                aria-pressed={p === preset}
                onClick={() => onPreset(p)}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="opt">
        <span className="lbl" id="profile-lbl">
          {t.profile}
        </span>
        <div className="seg" role="group" aria-labelledby="profile-lbl">
          {PROFILES.map((p) => (
            <button
              key={p ?? "default"}
              type="button"
              className={p === profile ? "on" : undefined}
              aria-pressed={p === profile}
              onClick={() => onProfile(p)}
            >
              {p ?? t.profileNone}
            </button>
          ))}
        </div>
      </div>
      <AdvancedOptions t={t} value={advanced} llm={llm} onChange={onAdvanced} />
      <CliCommand
        t={t}
        urls={urls}
        preset={preset}
        llm={llm}
        ocr={ocr}
        profile={profile}
        announce={announce}
      />
      {trailing}
    </div>
  );
}

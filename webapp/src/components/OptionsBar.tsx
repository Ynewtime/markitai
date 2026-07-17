import type { ReactNode } from "react";
import type { Preset } from "../api/types";
import { useMediaQuery } from "../hooks/useMediaQuery";
import type { Dict } from "../i18n";
import { CliCommand } from "./CliCommand";

const PRESETS: Preset[] = ["minimal", "standard", "rich"];

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
  llmConfigured,
  urls,
  announce,
  onPreset,
  onLlm,
  onOcr,
  trailing,
}: {
  t: Dict;
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  llmConfigured: boolean;
  urls: string[];
  announce: (msg: string) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
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
      <CliCommand
        t={t}
        urls={urls}
        preset={preset}
        llm={llm}
        ocr={ocr}
        announce={announce}
      />
      {trailing}
    </div>
  );
}

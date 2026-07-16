import type { Preset } from "../api/types";
import type { Dict } from "../i18n";
import { CliCommand } from "./CliCommand";

const PRESETS: Preset[] = ["minimal", "standard", "rich"];

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
}) {
  return (
    <div className="options">
      {llmConfigured && (
        <div className="opt">
          <span className="lbl">{t.llmEnhance}</span>
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
    </div>
  );
}

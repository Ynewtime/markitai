import type { Preset } from "../api/types";
import type { Dict } from "../i18n";

const PRESETS: Preset[] = ["minimal", "standard", "rich"];

/** Preset segmented control + LLM switch. When llm is not configured the
 * switch and the standard/rich presets are disabled (CapabilityHint explains). */
export function OptionsBar({
  t,
  preset,
  llm,
  llmConfigured,
  onPreset,
  onLlm,
}: {
  t: Dict;
  preset: Preset;
  llm: boolean;
  llmConfigured: boolean;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
}) {
  // Attach the "why is this disabled" explanation to the control itself.
  const disabledTitle = !llmConfigured
    ? `${t.capHintPre}${t.capHintLink}${t.capHintPost}`
    : undefined;
  return (
    <div className="options">
      <div className="opt">
        <span className="lbl" id="preset-lbl">
          {t.preset}
        </span>
        <div className="seg" role="group" aria-labelledby="preset-lbl">
          {PRESETS.map((p) => {
            const disabled = !llmConfigured && p !== "minimal";
            return (
              <button
                key={p}
                type="button"
                className={p === preset ? "on" : undefined}
                aria-pressed={p === preset}
                disabled={disabled}
                title={disabled ? disabledTitle : undefined}
                onClick={() => onPreset(p)}
              >
                {p}
              </button>
            );
          })}
        </div>
      </div>
      <div className="opt">
        <span className="lbl">{t.llmEnhance}</span>
        <button
          type="button"
          role="switch"
          aria-checked={llm}
          aria-label={t.llmEnhance}
          className={llm ? "switch on" : "switch"}
          disabled={!llmConfigured}
          title={!llmConfigured ? disabledTitle : undefined}
          onClick={() => onLlm(!llm)}
        />
      </div>
    </div>
  );
}

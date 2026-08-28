import type { ReactNode } from "react";
import type { OutputProfile, Preset } from "../api/types";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import { OptionsPanel } from "./OptionsPanel";

/** The row under the composer: one button, and the panel it opens.
 *
 * The product's promise is that you drop a file in and get markdown, so the
 * default screen asks nothing. The button carries a summary of what is on,
 * so a collapsed panel never hides a setting that costs money. The
 * equivalent CLI command lives in the panel's footer rather than behind a
 * second disclosure — it is those settings spelled out, not a sibling of
 * them. */
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
  return (
    <div className="options">
      <OptionsPanel
        t={t}
        value={advanced}
        preset={preset}
        llm={llm}
        ocr={ocr}
        profile={profile}
        llmConfigured={llmConfigured}
        onChange={onAdvanced}
        onPreset={onPreset}
        onLlm={onLlm}
        onOcr={onOcr}
        onProfile={onProfile}
        urls={urls}
        announce={announce}
      />
      {trailing}
    </div>
  );
}

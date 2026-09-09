import type { ReactNode } from "react";
import type { OutputProfile, Preset } from "../api/types";
import type { Dict } from "../i18n";
import type { Advanced } from "../lib/advanced";
import type { PresetOptions } from "../lib/conversionOptions";
import { OptionsPanel } from "./OptionsPanel";

export function OptionsBar({
  t,
  preset,
  presetOptions,
  llm,
  ocr,
  profile,
  advanced,
  llmConfigured,
  urls,
  announce,
  source,
  heading,
  headingActions,
  onFiles,
  onPreset,
  onLlm,
  onOcr,
  onProfile,
  onAdvanced,
}: {
  t: Dict;
  preset: Preset;
  presetOptions?: PresetOptions;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  advanced: Advanced;
  llmConfigured: boolean;
  urls: string[];
  announce: (msg: string) => void;
  /** Input row and optional workspace heading; tools stay above the card. */
  source?: ReactNode;
  heading?: ReactNode;
  headingActions?: ReactNode;
  onFiles?: (files: File[]) => void;
  onPreset: (p: Preset) => void;
  onLlm: (v: boolean) => void;
  onOcr: (v: boolean) => void;
  onProfile: (p: OutputProfile | null) => void;
  onAdvanced: (a: Advanced) => void;
}) {
  return (
    <OptionsPanel
      t={t}
      value={advanced}
      preset={preset}
      presetOptions={presetOptions}
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
      source={source}
      heading={heading}
      headingActions={headingActions}
      onFiles={onFiles}
    />
  );
}

// The composer: the source card holds the URL line, whose bottom row carries
// the tools (Options, Upload) on the left and Convert on the right. Options
// opens the conversion options in a dialog (options-dialog.tsx); a dot on it
// says the current choices differ from the defaults.
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type { CloudflareCapability } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { cloudflareReason } from "../lib/cloudflare.ts";
import { listEntry, OPTIONS_ENTRY } from "../lib/route.ts";
import { cliCommand, stagedPaths } from "../lib/cli.ts";
import { BUILTIN_DEFAULTS, BUILTIN_PRESETS, changedOptions, matchingPreset, resolveOptions, type Composer, type PresetTable } from "../lib/options.ts";
import { UploadPicker } from "./file-picker.tsx";
import { Icon } from "./icons.tsx";
import { OptionsDialog } from "./options-dialog.tsx";
import type { DefaultsControl } from "../hooks/use-convert-defaults.ts";

export function OptionsPanel({
  t,
  state,
  presets = BUILTIN_PRESETS,
  llmReady,
  needsModel,
  onAddModel,
  cloudflare,
  urls,
  staged = [],
  announce,
  source,
  heading,
  actions,
  busy,
  defaults = null,
  onChange,
  onFiles,
  onFolder,
  openRequest = 0,
  textOnly = false,
  localOnly = false,
  documentJson = false,
  ocrLanguages = null,
  serverOcrLang = "en",
  ocrAllPages = false,
  pageMarkers = false,
}: {
  t: Dict;
  state: Composer;
  presets?: PresetTable;
  llmReady: boolean;
  /** v1.15: what to do while no model can serve this caller (where to add one). */
  needsModel?: string;
  /** Opens Settings → Models, when this caller may add a model there. */
  onAddModel?: () => void;
  cloudflare?: CloudflareCapability;
  urls: string[];
  /** Files waiting for Convert: the command names them relative to their folder. */
  staged?: File[];
  announce: (message: string) => void;
  /** The URL line, given the tools for its bottom row. */
  source: (tools: ComponentChildren) => ComponentChildren;
  heading?: ComponentChildren;
  actions?: ComponentChildren;
  busy: boolean;
  /** v1.17: the server's (or the user's own) defaults, saved and restored here. */
  defaults?: DefaultsControl | null;
  onChange: (next: Composer) => void;
  onFiles: (files: File[]) => void;
  onFolder: (files: File[], hidden: number) => void;
  /** Counts up each time something outside asks for the options to be shown (Re-convert). */
  openRequest?: number;
  /** The backend offers Text only (`features.text_only`). */
  textOnly?: boolean;
  /** A user token: its conversions never use a remote service, so the remote
   * strategies and the Cloudflare backend are not offered (both backends
   * refuse them with 403 `remote_processing_forbidden`). */
  localOnly?: boolean;
  /** v1.20 `features.document_json`: the JSON switch (`format`). */
  documentJson?: boolean;
  /** v1.20 `features.ocr_lang`: the capabilities' `ocr_languages`; null hides the picker. */
  ocrLanguages?: string[] | null;
  /** v1.20: the server's `ocr.lang`, what the picker shows until one is chosen. */
  serverOcrLang?: string;
  /** v1.20 `features.ocr_all_pages`. */
  ocrAllPages?: boolean;
  /** v1.20 `features.page_markers`. */
  pageMarkers?: boolean;
}) {
  const modelNote = needsModel ?? t.advNeedsModel;
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (openRequest > 0) setOpen(true);
  }, [openRequest]);
  // The dialog has a history entry of its own, as the conversion window has: Back (or the swipe)
  // closes it and stays on Convert, and closing it by hand goes back over that entry. An entry
  // left from before a reload is dropped, so it never takes a Back of its own.
  const pushed = useRef(false);
  // Leaving for Settings from the dialog waits for Back over its entry, so the two never race.
  const backing = useRef(false);
  const afterClose = useRef<(() => void) | null>(null);
  useEffect(() => {
    if (listEntry()?.wb === OPTIONS_ENTRY) history.replaceState(null, "", location.href);
  }, []);
  useEffect(() => {
    if (!open) return;
    history.pushState({ wb: OPTIONS_ENTRY }, "", location.href);
    pushed.current = true;
    const onPop = () => {
      if (listEntry()?.wb === OPTIONS_ENTRY) return;
      pushed.current = false;
      backing.current = false;
      setOpen(false);
      const next = afterClose.current;
      afterClose.current = null;
      if (next) next();
      else requestAnimationFrame(() => trigger.current?.focus());
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [open]);
  const close = () => {
    if (pushed.current && listEntry()?.wb === OPTIONS_ENTRY) {
      backing.current = true;
      history.back();
      return;
    }
    setOpen(false);
    requestAnimationFrame(() => trigger.current?.focus());
  };

  const a = state.advanced;
  const effective = resolveOptions(state, presets);
  const selectionReason = cloudflareReason(cloudflare, effective);
  const shownPreset = matchingPreset(state, presets) ?? state.preset;
  // Without a model the defaults apply without LLM (the composer turns it off), so they still count as met.
  const reference = defaults ? (llmReady ? defaults.view.defaults : { ...defaults.view.defaults, llm: false }) : BUILTIN_DEFAULTS;
  const changed = changedOptions(state, reference, presets);
  const paths = stagedPaths(staged);
  const command = cliCommand([...paths, ...urls], { ...effective, preset: shownPreset }, presets);

  return (
    <div class="composer">
      {heading && (
        <div class="work-head">
          {heading}
          {actions && <div class="work-head-tools">{actions}</div>}
        </div>
      )}
      <div class="source-card">
        <div class="source-row">
          {source(
            <div class="tool-group" role="group" aria-label={t.sourceActions}>
              <button
                ref={trigger}
                type="button"
                class={open ? "tool is-on" : "tool"}
                aria-label={changed.size > 0 ? t.optionRow(t.options, t.defaultsChanged) : t.options}
                title={t.options}
                aria-haspopup="dialog"
                aria-expanded={open}
                onClick={() => setOpen(true)}
              >
                <Icon name="Sliders" size={14} />
                <span class="tool-text">{t.options}</span>
                {changed.size > 0 && <span class="tool-dot" aria-hidden="true" />}
              </button>
              <UploadPicker t={t} disabled={busy} onFiles={onFiles} onFolder={onFolder} />
            </div>,
          )}
        </div>
        {(a.strategy === "cloudflare" || a.backend === "cloudflare") && <div class="cloudflare-note">
          {selectionReason && <p role="alert">{t.cloudflareReason(selectionReason)}</p>}
          <p>{t.cloudflareScope}</p><p>{t.cloudflareCharges}</p>
          {selectionReason && selectionReason !== "incompatible_strategy" && <details><summary>{t.helpLabel}</summary><p>{t.cloudflareSetup}</p></details>}
        </div>}
      </div>
      {open && (
        <OptionsDialog
          t={t}
          state={state}
          presets={presets}
          llmReady={llmReady}
          needsModel={modelNote}
          onAddModel={
            onAddModel &&
            (() => {
              if (backing.current) afterClose.current = onAddModel;
              else onAddModel();
            })
          }
          cloudflare={cloudflare}
          command={command}
          pathsNote={paths.length > 0}
          announce={announce}
          defaults={defaults}
          changed={changed}
          onChange={onChange}
          onClose={close}
          textOnly={textOnly}
          localOnly={localOnly}
          documentJson={documentJson}
          ocrLanguages={ocrLanguages}
          serverOcrLang={serverOcrLang}
          ocrAllPages={ocrAllPages}
          pageMarkers={pageMarkers}
        />
      )}
    </div>
  );
}

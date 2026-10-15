// v1.17: the composer's defaults from the server (`/api/convert-defaults`): the server's
// configuration for the admin, a user's own over it. Until the person changes something,
// the composer follows them; Save as default writes the current choices back.
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { ApiError, fetchConvertDefaults, saveConvertDefaults } from "../api/client.ts";
import type { ConvertDefaultsView } from "../api/types.ts";
import { composerFromDefaults, defaultsOf, type Composer, type PresetTable } from "../lib/options.ts";

export interface DefaultsControl {
  view: ConvertDefaultsView;
  saving: boolean;
  /** A failed save, already in words; `stale` when the defaults changed meanwhile (reloaded). */
  failure: { text: string; stale: boolean } | null;
  save: () => void;
  reset: () => void;
  /** A user's own defaults removed: the server's apply again. */
  useServer: () => void;
}

export function useConvertDefaults({
  enabled,
  composer,
  setComposer,
  presets,
  describe,
  onSaved,
}: {
  enabled: boolean;
  composer: Composer;
  setComposer: (next: Composer) => void;
  presets: PresetTable | undefined;
  describe: (error: unknown) => { text: string };
  onSaved: (view: ConvertDefaultsView, removed: boolean) => void;
}): { control: DefaultsControl | null; touch: () => void } {
  const [view, setView] = useState<ConvertDefaultsView | null>(null);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<DefaultsControl["failure"]>(null);
  const touched = useRef(false);
  const composerRef = useRef(composer);
  composerRef.current = composer;
  useEffect(() => {
    if (!enabled) return;
    let stale = false;
    fetchConvertDefaults().then(
      (value) => {
        if (stale) return;
        setView(value);
        if (!touched.current) setComposer(composerFromDefaults(value.defaults, presets));
      },
      () => undefined,
    );
    return () => {
      stale = true;
    };
  }, [enabled, presets]);
  const write = useCallback(
    (removing: boolean) => {
      if (!view || saving) return;
      const wanted = removing ? null : defaultsOf(composerRef.current, presets);
      if (!removing && wanted === null) return;
      setSaving(true);
      setFailure(null);
      saveConvertDefaults(view.revision, wanted).then(
        (next) => {
          setView(next);
          setSaving(false);
          if (removing) setComposer(composerFromDefaults(next.defaults, presets));
          onSaved(next, removing);
        },
        (error: unknown) => {
          setSaving(false);
          const stale = error instanceof ApiError && error.status === 409;
          setFailure({ text: describe(error).text, stale });
          if (stale) fetchConvertDefaults().then(setView, () => undefined);
        },
      );
    },
    [view, saving, presets, describe, onSaved],
  );
  const control: DefaultsControl | null = view && {
    view,
    saving,
    failure,
    save: () => write(false),
    reset: () => {
      setFailure(null);
      setComposer(composerFromDefaults(view.defaults, presets));
    },
    useServer: () => write(true),
  };
  return { control, touch: () => (touched.current = true) };
}

// A model row's routing weight, edited in place (v1.16): the pill is a button;
// a click turns it into a number field with the value selected. Enter or
// leaving the field saves (with the page's revision, as every settings write),
// Escape puts the pill back unchanged. A value that is not a whole number of 0
// or more is not sent: the field says so and stays open (on blur it reverts).
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { parseWeight } from "../lib/models.ts";


export function WeightPill({
  t,
  model,
  value,
  narrow,
  title,
  disabled = false,
  onSave,
}: {
  t: Dict;
  /** The model's name, for the field's label and the pill's own. */
  model: string;
  value: number;
  narrow: boolean;
  /** What the weight does here. */
  title: string;
  disabled?: boolean;
  /** Save a new weight; whether it was saved (a failure is shown by the page). */
  onSave: (weight: number) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const pill = useRef<HTMLButtonElement>(null);
  /** Enter or Escape closed the field: the blur that follows does nothing. */
  const settled = useRef(false);
  /** The pill takes the focus back after a keyboard close, not after a click elsewhere. */
  const refocus = useRef(false);

  useLayoutEffect(() => {
    if (draft === null || busy) return;
    const input = field.current;
    if (input && document.activeElement !== input) {
      input.focus();
      input.select();
    }
  }, [draft === null, busy]);
  useEffect(() => {
    if (draft !== null || !refocus.current) return;
    refocus.current = false;
    pill.current?.focus();
  }, [draft]);

  const close = (keyboard: boolean) => {
    refocus.current = keyboard;
    setDraft(null);
    setInvalid(false);
  };
  const commit = async (keyboard: boolean) => {
    if (draft === null || busy) return;
    const next = parseWeight(draft);
    if (next === null) {
      if (keyboard) {
        setInvalid(true);
        return;
      }
      close(false);
      return;
    }
    settled.current = keyboard;
    if (next === value) {
      close(keyboard);
      return;
    }
    setBusy(true);
    try {
      await onSave(next);
    } finally {
      setBusy(false);
      close(keyboard);
    }
  };

  const text = narrow ? t.modelWeightShort(value) : t.modelWeight(value);
  if (draft === null) {
    return (
      <button
        ref={pill}
        type="button"
        class="mini-pill weight-pill"
        disabled={disabled}
        title={`${title}. ${t.weightEditHint}`}
        aria-label={t.weightEditAria(text, model)}
        onClick={() => {
          settled.current = false;
          setDraft(String(value));
        }}
      >
        {text}
      </button>
    );
  }
  return (
    <span class={`mini-pill weight-pill is-editing${invalid ? " is-invalid" : ""}`} aria-busy={busy || undefined}>
      <span aria-hidden="true">{narrow ? t.weightShortLabel : t.weightLabel}</span>
      <input
        ref={field}
        class="weight-input"
        type="text"
        inputMode="numeric"
        enterKeyHint="done"
        autoComplete="off"
        spellcheck={false}
        maxLength={6}
        size={Math.max(1, draft.length)}
        value={draft}
        disabled={busy}
        aria-label={t.weightFieldLabel(model)}
        aria-invalid={invalid || undefined}
        title={invalid ? t.weightInvalid : t.weightKeysHint}
        onInput={(event) => {
          setDraft(event.currentTarget.value);
          setInvalid(false);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void commit(true);
          } else if (event.key === "Escape") {
            // The page's own Escape (step back) must not run as well.
            event.preventDefault();
            event.stopPropagation();
            settled.current = true;
            close(true);
          } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
            event.preventDefault();
            const current = parseWeight(draft) ?? value;
            setDraft(String(Math.max(0, Math.min(999999, current + (event.key === "ArrowUp" ? 1 : -1)))));
            setInvalid(false);
          }
        }}
        onBlur={() => {
          if (settled.current) return;
          void commit(false);
        }}
      />
      {invalid && (
        <span class="sr-only" role="alert">
          {t.weightInvalid}
        </span>
      )}
    </span>
  );
}

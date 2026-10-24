// A model row's routing weight, edited in place (v1.16): the pill is a button;
// a click turns it into a number field with the value selected. Enter or
// leaving the field saves (with the page's revision, as every settings write),
// Escape puts the pill back unchanged. A value that is not a whole number from 0
// to 100 is not sent: the row says so under the pill (`onInvalid`), Enter keeps
// the field open, leaving it reverts; ArrowUp/ArrowDown step within 0–100.
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { parseWeight, stepWeight, WEIGHT_PLACEHOLDER } from "../lib/models.ts";


export function WeightPill({
  t,
  model,
  value,
  narrow,
  title,
  disabled = false,
  onSave,
  onInvalid,
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
  /** Whether the field holds a value that cannot be saved, for the row's message under the pill. */
  onInvalid?: (invalid: boolean) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [busy, setBusy] = useState(false);
  // A typed value out of range says so at once; an empty one when Enter refuses it.
  const shown = invalid || (draft !== null && draft.trim() !== "" && parseWeight(draft) === null);
  useEffect(() => onInvalid?.(shown), [shown]);
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
        title={`${title}${t.toastLead}${t.weightEditHint}`}
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
    <span class={`mini-pill weight-pill is-editing${shown ? " is-invalid" : ""}`} aria-busy={busy || undefined}>
      <span aria-hidden="true">{narrow ? t.weightShortLabel : t.weightLabel}</span>
      <input
        ref={field}
        class="weight-input"
        type="text"
        inputMode="numeric"
        enterKeyHint="done"
        autoComplete="off"
        spellcheck={false}
        maxLength={3}
        placeholder={WEIGHT_PLACEHOLDER}
        size={draft ? Math.max(1, draft.length) : WEIGHT_PLACEHOLDER.length}
        value={draft}
        disabled={busy}
        aria-label={t.weightFieldLabel(model)}
        aria-invalid={shown || undefined}
        title={shown ? t.weightInvalid : t.weightKeysHint}
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
            setDraft(String(stepWeight(draft, value, event.key === "ArrowUp" ? 1 : -1)));
            setInvalid(false);
          }
        }}
        onBlur={() => {
          if (settled.current) return;
          void commit(false);
        }}
      />
      {shown && !onInvalid && (
        <span class="sr-only" role="alert">
          {t.weightInvalid}
        </span>
      )}
    </span>
  );
}

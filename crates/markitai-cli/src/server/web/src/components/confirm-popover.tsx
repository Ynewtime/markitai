// An anchored confirmation for a destructive action. The card lives on <body>
// so no scrolling container clips it, and flips above its trigger when the
// space below is short. Escape or a click elsewhere cancels; focus returns.
import { createPortal } from "preact/compat";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { Icon } from "./icons.tsx";

let cards = 0;

/** Dialogs ask whether a confirmation is on top before handling Escape or Tab. */
export const openConfirmCard = (): HTMLElement | null => document.querySelector<HTMLElement>(".confirm-card");

export function ConfirmPopover({
  triggerLabel,
  title,
  description,
  confirmLabel,
  cancelLabel,
  busyLabel,
  disabled = false,
  onConfirm,
}: {
  triggerLabel: string;
  title: string;
  description: string;
  confirmLabel: string;
  cancelLabel: string;
  busyLabel: string;
  disabled?: boolean;
  onConfirm: () => Promise<boolean>;
}) {
  const [id] = useState(() => `confirm-${++cards}`);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const card = useRef<HTMLDivElement>(null);
  const arrow = useRef<HTMLSpanElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);

  const close = useCallback((returnFocus = true) => {
    setOpen(false);
    if (returnFocus) requestAnimationFrame(() => trigger.current?.isConnected && trigger.current.focus());
  }, []);

  const place = useCallback(() => {
    const anchor = trigger.current;
    const box = card.current;
    if (!anchor || !box) return;
    const rect = anchor.getBoundingClientRect();
    const margin = 12;
    const gap = 10;
    const width = Math.min(352, window.innerWidth - margin * 2);
    const height = box.offsetHeight || 154;
    const below = window.innerHeight - rect.bottom >= height + gap + margin;
    const top = below ? rect.bottom + gap : Math.max(margin, rect.top - height - gap);
    const left = Math.min(window.innerWidth - width - margin, Math.max(margin, rect.right - width));
    box.classList.toggle("is-above", !below);
    box.style.top = `${top}px`;
    box.style.left = `${left}px`;
    box.style.visibility = "visible";
    if (arrow.current) arrow.current.style.left = `${Math.min(width - 20, Math.max(20, rect.left + rect.width / 2 - left))}px`;
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    place();
    const frame = requestAnimationFrame(place);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    cancel.current?.focus();
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && (root.current?.contains(event.target) || card.current?.contains(event.target))) return;
      close(false);
    };
    const keys = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      }
      // Tab stays between the two buttons while the card is up.
      if (event.key === "Tab" && card.current) {
        const buttons = [...card.current.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];
        const first = buttons[0];
        const last = buttons[buttons.length - 1];
        if (!first || !last) return;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        } else if (!card.current.contains(document.activeElement)) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", keys, true);
    return () => {
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", keys, true);
    };
  }, [open, close]);

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    const done = await onConfirm();
    setBusy(false);
    if (done) close(false);
  };

  return (
    <span ref={root} class="confirm" onClick={(event) => event.stopPropagation()}>
      <button
        ref={trigger}
        type="button"
        class="row-icon is-danger"
        aria-label={triggerLabel}
        title={triggerLabel}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        disabled={disabled || busy}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="Trash" size={14} />
      </button>
      {open &&
        createPortal(
          <div
            ref={card}
            id={id}
            class="confirm-card"
            role="alertdialog"
            aria-modal="false"
            aria-labelledby={`${id}-title`}
            aria-describedby={`${id}-text`}
            aria-busy={busy || undefined}
            onClick={(event) => event.stopPropagation()}
          >
            <span ref={arrow} class="confirm-arrow" aria-hidden="true" />
            <div class="confirm-head">
              <span class="confirm-icon" aria-hidden="true">
                <Icon name="WarningFill" size={18} />
              </span>
              <span class="confirm-copy">
                <strong id={`${id}-title`}>{title}</strong>
                <span id={`${id}-text`}>{description}</span>
              </span>
            </div>
            <div class="confirm-actions">
              <button ref={cancel} type="button" class="btn btn-ghost btn-sm" disabled={busy} onClick={() => close()}>
                {cancelLabel}
              </button>
              <button type="button" class="btn btn-danger btn-sm" disabled={busy} onClick={() => void confirm()}>
                {busy ? busyLabel : confirmLabel}
              </button>
            </div>
          </div>,
          document.body,
        )}
    </span>
  );
}

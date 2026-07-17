import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TrashIcon, WarningIcon } from "./icons";

type Position = {
  top: number;
  left: number;
  arrowLeft: number;
  placement: "above" | "below";
};

/** The card is portalled to document.body, outside any ancestor dialog's DOM
 * subtree, so dialogs must ask document-wide whether a confirmation layer is
 * on top before handling Escape or trapping Tab focus themselves. */
// eslint-disable-next-line react-refresh/only-export-components -- shared with SettingsModal's Escape/focus handling; splitting the file would orphan the comment above.
export function openDeletePopoverCard(): HTMLElement | null {
  return document.querySelector<HTMLElement>(".delete-popover-card");
}

/** Anchored destructive confirmation. The card is portalled to the viewport so
 * modal/list overflow cannot clip it, and flips around the trigger as needed. */
export function ConfirmDeletePopover({
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
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [position, setPosition] = useState<Position | null>(null);
  const rootRef = useRef<HTMLSpanElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  const cardId = useId();

  const close = useCallback((returnFocus = true) => {
    setOpen(false);
    setPosition(null);
    if (returnFocus) {
      window.requestAnimationFrame(() => {
        if (triggerRef.current?.isConnected) triggerRef.current.focus();
      });
    }
  }, []);

  const placeCard = useCallback(() => {
    const trigger = triggerRef.current;
    if (trigger === null) return;
    const rect = trigger.getBoundingClientRect();
    const margin = 12;
    const gap = 10;
    const width = Math.min(352, window.innerWidth - margin * 2);
    const height = cardRef.current?.offsetHeight ?? 154;
    const roomBelow = window.innerHeight - rect.bottom;
    const placement = roomBelow >= height + gap + margin ? "below" : "above";
    const top =
      placement === "below"
        ? rect.bottom + gap
        : Math.max(margin, rect.top - height - gap);
    const left = Math.min(
      window.innerWidth - width - margin,
      Math.max(margin, rect.right - width),
    );
    const arrowLeft = Math.min(width - 20, Math.max(20, rect.left + rect.width / 2 - left));
    setPosition({ top, left, arrowLeft, placement });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    placeCard();
    const frame = window.requestAnimationFrame(placeCard);
    window.addEventListener("resize", placeCard);
    window.addEventListener("scroll", placeCard, true);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", placeCard);
      window.removeEventListener("scroll", placeCard, true);
    };
  }, [open, placeCard]);

  useEffect(() => {
    if (!open) return;
    cancelRef.current?.focus();
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      if (
        rootRef.current?.contains(event.target) ||
        cardRef.current?.contains(event.target)
      ) {
        return;
      }
      close(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [close, open]);

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    const removed = await onConfirm();
    if (removed) close(false);
    else setBusy(false);
  };

  const card = open
    ? createPortal(
        <div
          ref={cardRef}
          id={cardId}
          className={`delete-popover-card ${position?.placement ?? "below"}`}
          role="alertdialog"
          aria-modal="false"
          aria-labelledby={titleId}
          aria-describedby={descriptionId}
          aria-busy={busy || undefined}
          style={
            position === null
              ? { visibility: "hidden" }
              : { top: position.top, left: position.left }
          }
          onClick={(event) => event.stopPropagation()}
        >
          {position !== null && (
            <span
              className="delete-popover-arrow"
              style={{ left: position.arrowLeft }}
              aria-hidden="true"
            />
          )}
          <div className="delete-popover-head">
            <span className="delete-popover-symbol" aria-hidden="true">
              <WarningIcon size={18} />
            </span>
            <span className="delete-popover-copy">
              <strong id={titleId}>{title}</strong>
              <span id={descriptionId}>{description}</span>
            </span>
          </div>
          <div className="delete-popover-actions">
            <button
              ref={cancelRef}
              type="button"
              className="btn ghost sm"
              disabled={busy}
              onClick={() => close()}
            >
              {cancelLabel}
            </button>
            <button
              type="button"
              className="btn danger sm"
              disabled={busy}
              onClick={() => void confirm()}
            >
              {busy ? busyLabel : confirmLabel}
            </button>
          </div>
        </div>,
        document.body,
      )
    : null;

  return (
    <span
      ref={rootRef}
      className="delete-popover"
      onClick={(event) => event.stopPropagation()}
    >
      <button
        ref={triggerRef}
        type="button"
        className="rowicon danger"
        aria-label={triggerLabel}
        title={triggerLabel}
        aria-haspopup="dialog"
        aria-controls={open ? cardId : undefined}
        aria-expanded={open}
        disabled={disabled || busy}
        onClick={() => setOpen((value) => !value)}
      >
        <TrashIcon />
      </button>
      {card}
    </span>
  );
}

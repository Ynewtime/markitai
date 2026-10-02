// PDF export options behind one button of the preview bar. The card stays
// inside the preview dialog (fixed position, so nothing clips it) because
// assistive technology treats content outside an aria-modal dialog as inert.
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";

export const PDF_FURNITURE_KEY = "markitai.pdf.custom-header-footer";

/** On unless explicitly turned off. */
export function readFurniture(): boolean {
  try {
    return localStorage.getItem(PDF_FURNITURE_KEY) !== "false";
  } catch {
    return true;
  }
}

export function storeFurniture(value: boolean): void {
  try {
    localStorage.setItem(PDF_FURNITURE_KEY, String(value));
  } catch {
    /* The choice lasts for this page only. */
  }
}

export function PdfSettings({
  t,
  disabled,
  furniture,
  onToggle,
}: {
  t: Dict;
  disabled: boolean;
  furniture: boolean;
  onToggle: () => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const card = useRef<HTMLDivElement>(null);
  const arrow = useRef<HTMLSpanElement>(null);

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
    const width = Math.min(300, window.innerWidth - margin * 2);
    const height = box.offsetHeight || 170;
    const below = window.innerHeight - rect.bottom >= height + gap + margin;
    box.classList.toggle("is-above", !below);
    box.style.top = `${below ? rect.bottom + gap : Math.max(margin, rect.top - height - gap)}px`;
    const left = Math.min(window.innerWidth - width - margin, Math.max(margin, rect.right - width));
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
    card.current?.querySelector<HTMLButtonElement>('button[role="switch"]')?.focus();
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && (root.current?.contains(event.target) || card.current?.contains(event.target))) return;
      close(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", escape, true);
    return () => {
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", escape, true);
    };
  }, [open, close]);

  return (
    <span ref={root} class="pdf-settings">
      <button
        ref={trigger}
        type="button"
        class="btn btn-ghost btn-sm"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? "pdf-card" : undefined}
        disabled={disabled}
        title={t.pdfSettings}
        aria-label={t.pdfSettings}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="Gear" size={13} />
        <span class="btn-label">{t.pdfSettings}</span>
      </button>
      {open && (
        <div ref={card} id="pdf-card" class="pdf-card" role="dialog" aria-modal="false" aria-labelledby="pdf-card-title" onClick={(event) => event.stopPropagation()}>
          <span ref={arrow} class="confirm-arrow" aria-hidden="true" />
          <span class="pdf-card-head">
            <strong id="pdf-card-title">{t.pdfSettings}</strong>
            {/* Touch screen readers have no Escape key: a visible way out. */}
            <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={() => close()}>
              <Icon name="X" size={14} />
            </button>
          </span>
          <span class="pdf-card-row">
            <span>{t.pdfCustomHeaderFooter}</span>
            <button
              type="button"
              role="switch"
              aria-checked={furniture}
              aria-label={t.pdfCustomHeaderFooter}
              class={furniture ? "switch is-on" : "switch"}
              onClick={onToggle}
            />
          </span>
          <span class="pdf-card-hint">{t.pdfPrintDialogHint}</span>
        </div>
      )}
    </span>
  );
}

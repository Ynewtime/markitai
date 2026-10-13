// Export PDF asks one thing before the print dialog: whether the PDF carries
// Markitai's own header and footer. The card points at the "⋯" button it was
// opened from and stays inside the window's dialog (fixed position, so nothing
// clips it), because assistive technology treats content outside an aria-modal
// dialog as inert. Escape, ✕, Cancel or a click elsewhere close it; focus
// returns to "⋯".
import { useCallback, useEffect, useLayoutEffect, useRef } from "preact/hooks";
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

export function PdfExportCard({
  t,
  anchor,
  furniture,
  onToggle,
  onPrint,
  onClose,
}: {
  t: Dict;
  /** The button the card points at, and where focus returns. */
  anchor: () => HTMLElement | null;
  furniture: boolean;
  onToggle: () => void;
  onPrint: () => void;
  onClose: () => void;
}) {
  const card = useRef<HTMLDivElement>(null);
  const arrow = useRef<HTMLSpanElement>(null);

  const close = useCallback(
    (returnFocus = true) => {
      onClose();
      if (returnFocus)
        requestAnimationFrame(() => {
          const node = anchor();
          if (node?.isConnected) node.focus();
        });
    },
    [onClose, anchor],
  );

  const place = useCallback(() => {
    const node = anchor();
    const box = card.current;
    if (!node || !box) return;
    const rect = node.getBoundingClientRect();
    const margin = 12;
    const gap = 10;
    const width = Math.min(340, window.innerWidth - margin * 2);
    const height = box.offsetHeight || 200;
    const below = window.innerHeight - rect.bottom >= height + gap + margin;
    box.classList.toggle("is-above", !below);
    box.style.top = `${below ? rect.bottom + gap : Math.max(margin, rect.top - height - gap)}px`;
    const left = Math.min(window.innerWidth - width - margin, Math.max(margin, rect.right - width));
    box.style.left = `${left}px`;
    box.style.visibility = "visible";
    if (arrow.current) arrow.current.style.left = `${Math.min(width - 20, Math.max(20, rect.left + rect.width / 2 - left))}px`;
  }, [anchor]);

  useLayoutEffect(() => {
    place();
    const frame = requestAnimationFrame(place);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [place]);

  useEffect(() => {
    card.current?.querySelector<HTMLButtonElement>(".pdf-card-actions .btn-primary")?.focus();
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && (card.current?.contains(event.target) || anchor()?.contains(event.target))) return;
      close(false);
    };
    const keys = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      }
      // Focus stays in the card while it is open.
      if (event.key !== "Tab" || !card.current) return;
      const nodes = [...card.current.querySelectorAll<HTMLElement>("button:not(:disabled)")];
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", keys, true);
    return () => {
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", keys, true);
    };
  }, [close, anchor]);

  return (
    <div ref={card} class="pdf-card" role="dialog" aria-modal="false" aria-labelledby="pdf-card-title" aria-describedby="pdf-card-hint" onClick={(event) => event.stopPropagation()}>
      <span ref={arrow} class="confirm-arrow" aria-hidden="true" />
      <div class="pdf-card-head">
        <strong id="pdf-card-title">{t.exportPdf}</strong>
        {/* Touch screen readers have no Escape key: a visible way out. */}
        <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={() => close()}>
          <Icon name="X" size={14} />
        </button>
      </div>
      <div class="pdf-card-option">
        <label class="pdf-card-row" for="pdf-card-furniture">
          <span>{t.pdfCustomHeaderFooter}</span>
          <button
            id="pdf-card-furniture"
            type="button"
            role="switch"
            aria-checked={furniture}
            aria-describedby="pdf-card-hint"
            class={furniture ? "switch is-on" : "switch"}
            onClick={onToggle}
          />
        </label>
        <p id="pdf-card-hint" class="pdf-card-hint">
          {/* The words carry their own space between sentences (none in Chinese or Japanese). */}
          {t.pdfFurnitureWhat}
          {furniture && t.pdfPrintDialogHint}
        </p>
      </div>
      <div class="pdf-card-actions">
        <button type="button" class="btn" onClick={() => close()}>
          {t.cancel}
        </button>
        <button
          type="button"
          class="btn btn-primary"
          onClick={() => {
            onClose();
            onPrint();
          }}
        >
          {t.pdfPrint}
        </button>
      </div>
    </div>
  );
}

// Shared modal behaviour: the page behind stops scrolling, focus moves into
// the dialog, Tab cycles inside it, and Escape closes it, unless a smaller
// layer (a confirmation or the PDF settings card) is on top and owns both.
import { useEffect, useRef, type MutableRef } from "preact/hooks";

const FOCUSABLE = 'button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])';

export const topLayer = (): HTMLElement | null =>
  document.querySelector<HTMLElement>(".confirm-card") ?? document.querySelector<HTMLElement>(".pdf-card");

export function useModal(dialog: MutableRef<HTMLElement | null>, onEscape: () => void): void {
  const escape = useRef(onEscape);
  escape.current = onEscape;

  useEffect(() => {
    document.body.classList.add("has-modal");
    dialog.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      const layer = topLayer();
      if (event.key === "Escape") {
        if (layer !== null) return;
        event.preventDefault();
        event.stopPropagation();
        escape.current();
        return;
      }
      if (event.key !== "Tab") return;
      const root = layer ?? dialog.current;
      if (!root) return;
      const nodes = [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
        (node) => !node.hasAttribute("disabled") && node.offsetParent !== null,
      );
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      const active = document.activeElement;
      if (!first || !last) {
        event.preventDefault();
        root.focus();
      } else if (!(active instanceof HTMLElement) || !root.contains(active)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && (active === first || active === root)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      // The veil leaves the document after this cleanup; another may stay open.
      requestAnimationFrame(() => {
        if (!document.querySelector(".veil")) document.body.classList.remove("has-modal");
      });
    };
  }, [dialog]);
}

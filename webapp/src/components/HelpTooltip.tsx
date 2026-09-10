import { cloneElement, useEffect, useId, useLayoutEffect, useRef, useState, type ReactElement, type HTMLAttributes } from "react";
import { createPortal } from "react-dom";

type Trigger = HTMLAttributes<HTMLElement> & { disabled?: boolean };

/** How long a shown tooltip may stay up, even while hovered. */
const AUTO_HIDE_MS = 4000;

export function HelpTooltip({ text, children }: { text: string; children: ReactElement<Trigger> }) {
  const id = useId();
  const anchor = useRef<HTMLSpanElement>(null);
  const tooltip = useRef<HTMLSpanElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [open, setOpen] = useState(false);
  const cancelHide = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = null;
  };
  const hideSoon = () => {
    cancelHide();
    hideTimer.current = setTimeout(() => setOpen(false), 150);
  };
  const show = () => {
    cancelHide();
    window.dispatchEvent(new CustomEvent("markitai:option-help", { detail: id }));
    setOpen(true);
  };
  useLayoutEffect(() => {
    if (!open) return;
    const target = anchor.current;
    const bubble = tooltip.current;
    if (!target || !bubble) return;
    const update = () => {
      const margin = 12;
      const gap = 8;
      const rect = target.getBoundingClientRect();
      const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
      const viewportHeight = window.innerHeight;
      const above = Math.max(0, rect.top - gap - margin);
      const below = Math.max(0, viewportHeight - margin - rect.bottom - gap);
      // Measure the rendered border box, not a guessed 360 x 200 rectangle.
      // Reset the height cap so a previous cramped placement cannot stick.
      bubble.style.maxHeight = `${Math.max(0, Math.min(200, viewportHeight - margin * 2))}px`;
      const natural = bubble.getBoundingClientRect();
      // Above by default; below only when above cannot fit the bubble (and,
      // when neither side fits, whichever side has strictly more room).
      const fitsAbove = natural.height <= above;
      const fitsBelow = natural.height <= below;
      const placeBelow = !fitsAbove && (fitsBelow || below > above);
      const space = placeBelow ? below : above;
      bubble.style.maxHeight = `${Math.min(200, space)}px`;
      const measured = bubble.getBoundingClientRect();
      const left = Math.max(margin, Math.min(rect.left + (rect.width - measured.width) / 2,
        viewportWidth - margin - measured.width));
      const desiredTop = placeBelow ? rect.bottom + gap : rect.top - gap - measured.height;
      const top = Math.max(margin, Math.min(desiredTop, viewportHeight - margin - measured.height));
      bubble.style.left = `${left}px`;
      bubble.style.top = `${top}px`;
      bubble.style.visibility = "visible";
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    const observer = typeof ResizeObserver === "undefined" ? undefined : new ResizeObserver(update);
    observer?.observe(target);
    observer?.observe(bubble);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      observer?.disconnect();
    };
  }, [open, text]);
  useEffect(() => {
    if (!open) return;
    const dismiss = () => { cancelHide(); setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") dismiss(); };
    const otherHelp = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== id) dismiss();
    };
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && tooltip.current?.contains(event.target)) return;
      if (event.target instanceof Node && !anchor.current?.contains(event.target)) dismiss();
    };
    window.addEventListener("markitai:option-help", otherHelp);
    window.addEventListener("pointerdown", outside);
    window.addEventListener("keydown", escape);
    // Hover tooltips self-dismiss after a few seconds even while the pointer
    // rests on the trigger: they must never sit over the form. Re-hover
    // re-shows and restarts the clock.
    const autoHide = setTimeout(dismiss, AUTO_HIDE_MS);
    return () => {
      clearTimeout(autoHide);
      window.removeEventListener("markitai:option-help", otherHelp);
      window.removeEventListener("pointerdown", outside);
      window.removeEventListener("keydown", escape);
    };
  }, [open, id]);
  useEffect(() => () => { if (hideTimer.current) clearTimeout(hideTimer.current); }, []);
  return (
    <span ref={anchor} className="help-anchor" onMouseEnter={show} onMouseLeave={hideSoon}
      onFocus={show} onBlur={() => { cancelHide(); setOpen(false); }} onClick={show}
      tabIndex={children.props.disabled ? 0 : undefined}
      aria-describedby={children.props.disabled ? id : undefined}>
      {cloneElement(children, { "aria-describedby": [children.props["aria-describedby"], id].filter(Boolean).join(" ") })}
      {/* Keep the accessible description in the trigger's subtree, including inside aria-modal dialogs. */}
      <span id={id} className="sr-only">{text}</span>
      {open && createPortal(
        <span ref={tooltip} className="option-tooltip" role="tooltip"
          style={{ left: 0, top: 0, visibility: "hidden" }} onMouseEnter={cancelHide} onMouseLeave={hideSoon}>
          {text}
        </span>, document.body)}
    </span>
  );
}

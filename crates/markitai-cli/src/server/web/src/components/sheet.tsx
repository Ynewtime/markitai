// The phone's bottom sheet, after the blog's collection modal: it slides up
// over a dimmed page with rounded top corners, a header row with the title,
// previous/next and close, a body that scrolls inside the sheet, and a pinned
// action bar clear of the home indicator. Swiping down (on the header, or on
// the body when it is scrolled to the top), tapping the dimmed page and Escape
// close it; focus stays inside while it is open.
import type { ComponentChildren } from "preact";
import { createPortal } from "preact/compat";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";

const EXIT_MS = 200;
/** A drag past this many pixels (or a quick flick) closes the sheet. */
const CLOSE_DISTANCE = 96;

const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

export function Sheet({
  t,
  title,
  titleId,
  onClose,
  onPrevious,
  onNext,
  actions,
  children,
}: {
  t: Dict;
  title: ComponentChildren;
  titleId: string;
  onClose: () => void;
  onPrevious: (() => void) | null;
  onNext: (() => void) | null;
  actions?: ComponentChildren;
  children: ComponentChildren;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  // Notices sit directly above the action bar while the sheet is open.
  useEffect(() => {
    const root = document.documentElement;
    // From the bar's top edge to the bottom of the window (the sheet may end above the tab bar).
    const place = () => {
      const top = bar.current?.getBoundingClientRect().top;
      if (top !== undefined) root.style.setProperty("--strips-bottom", `${Math.max(0, window.innerHeight - top)}px`);
    };
    place();
    const observer = typeof ResizeObserver === "undefined" || !bar.current ? null : new ResizeObserver(place);
    if (observer && bar.current) observer.observe(bar.current);
    window.addEventListener("resize", place);
    // The sheet slides in: measure again once it has arrived.
    const settled = setTimeout(place, 200);
    return () => {
      clearTimeout(settled);
      window.removeEventListener("resize", place);
      observer?.disconnect();
      root.style.removeProperty("--strips-bottom");
    };
  }, []);
  const [leaving, setLeaving] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [drag, setDrag] = useState(0);
  const start = useRef<{ y: number; at: number } | null>(null);
  /** The distance of the drag in progress; state lags the last pointer move. */
  const distance = useRef(0);

  const close = useCallback(() => {
    if (leaving) return;
    if (reducedMotion()) {
      onClose();
      return;
    }
    setLeaving(true);
    setTimeout(onClose, EXIT_MS);
  }, [leaving, onClose]);
  useModal(dialog, close);

  // The body starts at its top for every item the sheet shows.
  useEffect(() => {
    body.current?.scrollTo?.({ top: 0 });
    setScrolled(false);
  }, [titleId, title]);

  const begin = (y: number) => {
    start.current = { y, at: performance.now() };
    distance.current = 0;
  };
  const moveTo = (y: number): boolean => {
    if (!start.current) return false;
    distance.current = Math.max(0, y - start.current.y);
    setDrag(distance.current);
    return distance.current > 0;
  };
  const end = () => {
    if (!start.current) return;
    const moved = distance.current;
    const elapsed = performance.now() - start.current.at;
    const flick = moved > 24 && moved / Math.max(elapsed, 1) > 0.6;
    start.current = null;
    distance.current = 0;
    if (moved > CLOSE_DISTANCE || flick) close();
    setDrag(0);
  };

  // Header and handle: pointer drags (mouse, pen and touch alike).
  const headHandlers = {
    onPointerDown: (event: PointerEvent) => {
      if ((event.target as Element).closest("button")) return;
      (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
      begin(event.clientY);
    },
    onPointerMove: (event: PointerEvent) => void moveTo(event.clientY),
    // The release carries the final position, even when no move preceded it in this frame.
    onPointerUp: (event: PointerEvent) => {
      moveTo(event.clientY);
      end();
    },
    onPointerCancel: end,
  };

  // Body: a downward swipe from the top of the content drags the sheet instead of scrolling.
  useEffect(() => {
    const node = body.current;
    if (!node) return;
    const onStart = (event: TouchEvent) => {
      if (node.scrollTop <= 0) begin(event.touches[0]?.clientY ?? 0);
    };
    const onMove = (event: TouchEvent) => {
      if (!start.current) return;
      if (moveTo(event.touches[0]?.clientY ?? 0) && event.cancelable) event.preventDefault();
      else if (node.scrollTop > 0) start.current = null;
    };
    node.addEventListener("touchstart", onStart, { passive: true });
    node.addEventListener("touchmove", onMove, { passive: false });
    node.addEventListener("touchend", end);
    node.addEventListener("touchcancel", end);
    return () => {
      node.removeEventListener("touchstart", onStart);
      node.removeEventListener("touchmove", onMove);
      node.removeEventListener("touchend", end);
      node.removeEventListener("touchcancel", end);
    };
  });

  return createPortal(
    <div
      class={leaving ? "veil sheet-veil is-leaving" : "veil sheet-veil"}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div
        ref={dialog}
        class={`sheet${scrolled ? " is-scrolled" : ""}${drag > 0 ? " is-dragging" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        style={drag > 0 ? { transform: `translateY(${drag}px)` } : undefined}
      >
        <div class="sheet-head" {...headHandlers}>
          <span class="sheet-handle" aria-hidden="true" />
          <div class="sheet-title">{title}</div>
          <div class="sheet-nav">
            <button type="button" class="icon-btn" aria-label={t.previousItem} title={t.previousItem} disabled={onPrevious === null} onClick={() => onPrevious?.()}>
              <Icon name="CaretUp" size={16} />
            </button>
            <button type="button" class="icon-btn" aria-label={t.nextItem} title={t.nextItem} disabled={onNext === null} onClick={() => onNext?.()}>
              <Icon name="CaretDown" size={16} />
            </button>
            <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={close}>
              <Icon name="X" size={16} />
            </button>
          </div>
        </div>
        <div ref={body} class="sheet-body" onScroll={(event) => setScrolled(event.currentTarget.scrollTop > 4)}>
          {children}
        </div>
        {actions && (
          <div ref={bar} class="sheet-bar">
            {actions}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

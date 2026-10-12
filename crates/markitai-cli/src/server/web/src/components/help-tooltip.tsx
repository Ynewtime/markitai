// A help bubble for one control: shown on hover, focus or tap, positioned in
// the viewport (above unless it does not fit), gone a few seconds after it
// opens unless the pointer rests on it, on Escape, on a click elsewhere, or
// when another help bubble opens. A row's bubble can also list its choices.
// `InfoTip` is the same bubble on a small (i) inside another control (a menu
// entry): it opens while that control has focus, on hover, or on a tap of the
// (i), which does not reach the control.
import type { ComponentChildren } from "preact";
import { createPortal } from "preact/compat";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { Icon } from "./icons.tsx";

const AUTO_HIDE_MS = 4000;
/** Reading time added per listed choice. */
const PER_ITEM_MS = 1500;
let serial = 0;

/** One choice of a row: what it does and, when it is unavailable, why. */
export interface HelpItem {
  label: string;
  text: string;
  note?: string;
}

/** The plain-text form of one choice, for its accessible description. */
export const helpItemText = (item: HelpItem): string => (item.note ? `${item.text} ${item.note}` : item.text);

export function HelpTooltip({
  text,
  items,
  disabled = false,
  children,
}: {
  text: string;
  items?: HelpItem[];
  /** A disabled control cannot take focus; the wrapper then takes it instead. */
  disabled?: boolean;
  /** Renders the control, given the id of its description. */
  children: (describedBy: string) => ComponentChildren;
}) {
  const [id] = useState(() => `help-${++serial}`);
  const anchor = useRef<HTMLSpanElement>(null);
  const bubble = useRef<HTMLDivElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hovered = useRef(false);
  const [open, setOpen] = useState(false);
  const cap = items?.length ? 320 : 200;
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
    window.dispatchEvent(new CustomEvent("markitai:help", { detail: id }));
    setOpen(true);
  };

  useBubble({ open, setOpen, id, anchor, bubble, hovered, cap, cancelHide, extraMs: (items?.length ?? 0) * PER_ITEM_MS, deps: [text, items] });

  useEffect(() => cancelHide, []);

  return (
    <span
      ref={anchor}
      class="help-anchor"
      tabIndex={disabled ? 0 : undefined}
      aria-describedby={disabled ? id : undefined}
      onMouseEnter={() => {
        hovered.current = true;
        show();
      }}
      onMouseLeave={() => {
        hovered.current = false;
        hideSoon();
      }}
      onFocusIn={show}
      onFocusOut={() => {
        cancelHide();
        setOpen(false);
      }}
      onClick={show}
    >
      {children(id)}
      <span id={id} class="sr-only">
        {[text, ...(items ?? []).map((item) => `${item.label}: ${helpItemText(item)}`)].join(" ")}
      </span>
      {open &&
        createPortal(
          <div
            ref={bubble}
            class={items?.length ? "help-bubble is-list" : "help-bubble"}
            role="tooltip"
            onMouseEnter={() => {
              hovered.current = true;
              cancelHide();
            }}
            onMouseLeave={() => {
              hovered.current = false;
              hideSoon();
            }}
          >
            {items?.length ? <p class="help-lede">{text}</p> : text}
            {items?.length ? (
              <dl class="help-items">
                {items.map((item) => (
                  <div key={item.label} class="help-item">
                    <dt>{item.label}</dt>
                    <dd>
                      {item.text}
                      {item.note && <span class="help-note">{item.note}</span>}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </div>,
          document.body,
        )}
    </span>
  );
}

/** Places an open bubble beside its anchor and closes it on Escape, a click
 * elsewhere, another bubble opening, or after a while unless the pointer rests on it. */
function useBubble({
  open,
  setOpen,
  id,
  anchor,
  bubble,
  hovered,
  cap,
  cancelHide,
  extraMs,
  deps,
  beside = false,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  id: string;
  anchor: { current: HTMLElement | null };
  bubble: { current: HTMLDivElement | null };
  hovered: { current: boolean };
  cap: number;
  cancelHide: () => void;
  extraMs: number;
  deps: unknown[];
  /** In a menu: beside it, or under the entry, instead of over the other entries. */
  beside?: boolean;
}) {
  useLayoutEffect(() => {
    if (!open) return;
    const target = anchor.current;
    const card = bubble.current;
    if (!target || !card) return;
    const place = () => {
      const margin = 12;
      const gap = 8;
      const width = document.documentElement.clientWidth || window.innerWidth;
      const height = window.innerHeight;
      // Inside a menu the bubble keeps the other entries readable: beside the
      // menu when there is room, else under the whole entry.
      const menu = beside ? target.closest(".menu") : null;
      const row = beside ? target.closest("button, a") : null;
      if (menu && row) {
        const frame = menu.getBoundingClientRect();
        const line = row.getBoundingClientRect();
        card.style.maxHeight = `${Math.max(0, Math.min(cap, height - margin * 2))}px`;
        const box = card.getBoundingClientRect();
        const room = frame.left - gap - margin;
        const left = room >= box.width ? frame.left - gap - box.width : Math.max(margin, Math.min(line.left, width - margin - box.width));
        const top = room >= box.width ? line.top + (line.height - box.height) / 2 : line.bottom + gap;
        card.style.left = `${left}px`;
        card.style.top = `${Math.max(margin, Math.min(top, height - margin - box.height))}px`;
        card.style.visibility = "visible";
        return;
      }
      const rect = target.getBoundingClientRect();
      const above = Math.max(0, rect.top - gap - margin);
      const below = Math.max(0, height - margin - rect.bottom - gap);
      card.style.maxHeight = `${Math.max(0, Math.min(cap, height - margin * 2))}px`;
      const natural = card.getBoundingClientRect();
      const under = natural.height > above && (natural.height <= below || below > above);
      card.style.maxHeight = `${Math.min(cap, under ? below : above)}px`;
      const box = card.getBoundingClientRect();
      const left = Math.max(margin, Math.min(rect.left + (rect.width - box.width) / 2, width - margin - box.width));
      const wanted = under ? rect.bottom + gap : rect.top - gap - box.height;
      const top = Math.max(margin, Math.min(wanted, height - margin - box.height));
      card.style.left = `${left}px`;
      card.style.top = `${top}px`;
      card.style.visibility = "visible";
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, cap, ...deps]);

  useEffect(() => {
    if (!open) return;
    const dismiss = () => {
      cancelHide();
      setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismiss();
    };
    const other = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== id) dismiss();
    };
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && (bubble.current?.contains(event.target) || anchor.current?.contains(event.target))) return;
      dismiss();
    };
    window.addEventListener("markitai:help", other);
    window.addEventListener("pointerdown", away);
    window.addEventListener("keydown", escape);
    // A pointer resting on the anchor or the bubble keeps it; leaving hides it.
    const timer = setTimeout(() => {
      if (!hovered.current) dismiss();
    }, AUTO_HIDE_MS + extraMs);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("markitai:help", other);
      window.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", escape);
    };
  }, [open, id, extraMs]);

}

/** A small (i) inside another control, with its explanation in a bubble. The
 * control names the explanation with `aria-describedby={id}`; the (i) itself is
 * hidden from assistive technology and takes no focus. */
export function InfoTip({ id, text }: { id: string; text: string }) {
  const anchor = useRef<HTMLSpanElement>(null);
  const bubble = useRef<HTMLDivElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hovered = useRef(false);
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
    window.dispatchEvent(new CustomEvent("markitai:help", { detail: id }));
    setOpen(true);
  };
  useBubble({ open, setOpen, id, anchor, bubble, hovered, cap: 200, cancelHide, extraMs: 0, deps: [text], beside: true });
  // The control it sits in shows the explanation while it has keyboard focus.
  useEffect(() => {
    const control = anchor.current?.parentElement?.closest<HTMLElement>("button, a, [tabindex]");
    if (!control) return;
    const focus = () => {
      if (control.matches(":focus-visible")) show();
    };
    const blur = () => {
      cancelHide();
      setOpen(false);
    };
    control.addEventListener("focus", focus);
    control.addEventListener("blur", blur);
    return () => {
      control.removeEventListener("focus", focus);
      control.removeEventListener("blur", blur);
    };
  }, []);
  useEffect(() => cancelHide, []);
  return (
    <>
      <span
        ref={anchor}
        class={open ? "info-tip is-open" : "info-tip"}
        aria-hidden="true"
        // Hover is a mouse's; a touch only taps.
        onPointerEnter={(event) => {
          if (event.pointerType !== "mouse") return;
          hovered.current = true;
          show();
        }}
        onPointerLeave={(event) => {
          if (event.pointerType !== "mouse") return;
          hovered.current = false;
          hideSoon();
        }}
        // A tap opens the explanation (a touch also sends the hover first, so it
        // never closes it) and does not reach the control; a tap elsewhere closes it.
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          show();
        }}
      >
        <Icon name="Info" size={14} />
      </span>
      <span id={id} hidden>
        {text}
      </span>
      {open &&
        createPortal(
          <div
            ref={bubble}
            class="help-bubble"
            role="tooltip"
            onMouseEnter={() => {
              hovered.current = true;
              cancelHide();
            }}
            onMouseLeave={() => {
              hovered.current = false;
              hideSoon();
            }}
          >
            {text}
          </div>,
          // Inside the window it explains (a dialog), so it is part of what assistive technology reads there.
          anchor.current?.closest("[role=dialog]") ?? document.body,
        )}
    </>
  );
}

// A help bubble for one control: shown on hover, focus or tap, positioned in
// the viewport (above unless it does not fit), gone a few seconds after it
// opens unless the pointer rests on it, on Escape, on a click elsewhere, or
// when another help bubble opens. A row's bubble can also list its choices.
import type { ComponentChildren } from "preact";
import { createPortal } from "preact/compat";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";

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

  useLayoutEffect(() => {
    if (!open) return;
    const target = anchor.current;
    const card = bubble.current;
    if (!target || !card) return;
    const place = () => {
      const margin = 12;
      const gap = 8;
      const rect = target.getBoundingClientRect();
      const width = document.documentElement.clientWidth || window.innerWidth;
      const height = window.innerHeight;
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
  }, [open, text, items, cap]);

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
    }, AUTO_HIDE_MS + (items?.length ?? 0) * PER_ITEM_MS);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("markitai:help", other);
      window.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", escape);
    };
  }, [open, id, items?.length]);

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

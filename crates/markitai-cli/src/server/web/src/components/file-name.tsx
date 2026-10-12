// A name on one line that, when it is too long, gives up its middle: a path
// or web address whole folders first ("slow/annual-reports/…/s5.html", the
// host kept), a file name its middle characters ("2025年度…披露全文.pdf.llm.md",
// the extension kept). The cut is measured against the width the name has, so
// the "…" sits right against what follows. The whole name is in the title
// (pointer hover) and in the text read aloud; a long press shows it in full on
// a touch screen until the next tap.
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { displayName } from "../lib/format.ts";
import { nameShape, shortenShape, type NameKind, type ShortName } from "../lib/middle.ts";

const LONG_PRESS_MS = 450;

let context: CanvasRenderingContext2D | null = null;
function textWidth(font: string, text: string): number {
  context ??= document.createElement("canvas").getContext("2d");
  if (!context) return text.length * 8;
  context.font = font;
  return context.measureText(text).width;
}

/** One observer for every name on the page; each name refits when its box changes size. */
const fitters = new Map<Element, () => void>();
let observer: ResizeObserver | null = null;
function watch(element: Element, fit: () => void): () => void {
  if (typeof ResizeObserver !== "function") return () => undefined;
  observer ??= new ResizeObserver((entries) => {
    for (const entry of entries) fitters.get(entry.target)?.();
  });
  fitters.set(element, fit);
  observer.observe(element);
  return () => {
    fitters.delete(element);
    observer?.unobserve(element);
  };
}

export function MiddleName({ text, kind, title, class: className }: { text: string; kind: NameKind; title?: string; class?: string }) {
  const root = useRef<HTMLSpanElement>(null);
  const hostRef = useRef<HTMLSpanElement>(null);
  const [short, setShort] = useState<ShortName | null>(null);
  const [full, setFull] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shape = nameShape(text, kind);

  useLayoutEffect(() => {
    const element = root.current;
    if (!element || full) return;
    const fit = () => {
      const width = element.clientWidth;
      if (!width) return;
      const style = getComputedStyle(element);
      const font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      // A host the layout hides (phone list rows) takes no room: the path is cut alone.
      const hostShown = Boolean(shape.host) && hostRef.current !== null && getComputedStyle(hostRef.current).display !== "none";
      const fits = (candidate: string) => textWidth(font, candidate) <= width + 0.5;
      const next = hostShown ? shortenShape(shape, fits) : { host: shape.host, body: shortenShape({ ...shape, host: "" }, fits).body };
      setShort((previous) => (previous && previous.host === next.host && previous.body === next.body ? previous : next));
    };
    fit();
    const stop = watch(element, fit);
    let live = true;
    document.fonts?.ready.then(() => live && fit());
    return () => {
      live = false;
      stop();
    };
  }, [text, kind, full]);

  useEffect(() => {
    if (!full) return;
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setFull(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [full]);
  useEffect(() => () => (timer.current ? clearTimeout(timer.current) : undefined), []);
  const stop = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };

  const shown = short ?? { host: shape.host, body: shape.path + shape.query };
  return (
    <span
      ref={root}
      class={`mid-name${full ? " is-full" : ""}${className ? ` ${className}` : ""}`}
      title={title ?? text}
      onPointerDown={(event) => {
        if (event.pointerType !== "touch") return;
        if (full) {
          setFull(false);
          return;
        }
        stop();
        timer.current = setTimeout(() => setFull(true), LONG_PRESS_MS);
      }}
      onPointerUp={stop}
      onPointerCancel={stop}
      onPointerLeave={stop}
    >
      {full ? (
        text
      ) : (
        <>
          <span class="mid-line" aria-hidden="true">
            {shape.host && (
              <span ref={hostRef} class="mid-host">
                {shown.host}
              </span>
            )}
            {shown.body}
          </span>
          {/* The whole name sizes the box (a row of no height after the drawn one, so the
              baseline stays the drawn name's); the cut one is drawn in it. */}
          <span class="mid-sizer" aria-hidden="true">
            {shape.host && <span class="mid-host">{shape.host}</span>}
            {shape.path + shape.query}
          </span>
          <span class="sr-only">{title ?? text}</span>
        </>
      )}
    </span>
  );
}

/** A file name, or a web address (`url`) as written. */
export function FileName({ name, url = false, class: className }: { name: string; url?: boolean; class?: string }) {
  return <MiddleName text={name} kind={url ? "url" : "file"} class={className} />;
}

/** An item's name in a list: a web address without its scheme, its host dimmed. */
export function NameText({ name, kind }: { name: string; kind: "file" | "url" }) {
  return <MiddleName text={kind === "url" ? displayName(name) : name} kind={kind} title={name} class={kind === "url" ? "nm has-host" : "nm"} />;
}

// A file name on one line that, when it is too long, gives up the middle and
// keeps its end: "2025年度可持续发展…披露全文.pdf.llm.md". The whole name is in the
// title (pointer hover) and in the text read aloud; a long press shows it in
// full on a touch screen until the next tap.
import { useEffect, useRef, useState } from "preact/hooks";
import { nameTail } from "../lib/format.ts";

const LONG_PRESS_MS = 450;

export function FileName({ name, class: className }: { name: string; class?: string }) {
  const [full, setFull] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const root = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!full) return;
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setFull(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [full]);
  useEffect(() => () => (timer.current ? clearTimeout(timer.current) : undefined), []);
  const cut = nameTail(name);
  const stop = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };
  return (
    <span
      ref={root}
      class={`file-name${full ? " is-full" : ""}${className ? ` ${className}` : ""}`}
      title={name}
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
        name
      ) : (
        <>
          <span class="file-name-head">{name.slice(0, cut)}</span>
          <span class="file-name-tail">{name.slice(cut)}</span>
        </>
      )}
    </span>
  );
}

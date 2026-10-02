import { useEffect, useState } from "preact/hooks";

/** The single phone tier of the stylesheet. */
export const NARROW = "(max-width: 780px)";

/** Whether a media query matches, following viewport changes. Copy that must
 * change with the layout is chosen here, never with CSS `content`. */
export function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);
  return matches;
}

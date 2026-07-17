import { useEffect, useState } from "react";

/** Reactive matchMedia — true while `query` matches, tracking viewport
 * changes. Components use this (never CSS `content`) when responsive copy
 * must swap, so the accessible name and the visible text stay in React's
 * hands. Guarded initial read keeps non-browser environments (tests) safe. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    // Sync in case the viewport changed between render and subscription.
    setMatches(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

// Which of a conversion window's actions stay buttons in its header.

/** Wider screens show every action as a button. On a phone the header keeps the
 * primary actions and those that must stay in reach (Stop while items wait,
 * as the full-screen window covers the list's own Stop); the rest fold into "⋯". */
export function headerActs<T extends { primary?: boolean; keep?: boolean }>(shown: T[], phone: boolean): { inline: T[]; folded: T[] } {
  if (!phone) return { inline: shown, folded: [] };
  return { inline: shown.filter((act) => act.primary || act.keep), folded: shown.filter((act) => !act.primary && !act.keep) };
}

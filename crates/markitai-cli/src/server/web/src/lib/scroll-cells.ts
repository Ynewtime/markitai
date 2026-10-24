// Table cells that keep their text on one line and scroll it sideways (the properties' keys and
// values, Details' fact labels). A cell with more to its right fades there until it is scrolled
// to its end; a cell that scrolls is reachable by the keyboard when nothing in it already is.

/** Whether the cell's content is wider than the cell. */
export const overflows = (cell: HTMLElement): boolean => cell.scrollWidth > cell.clientWidth + 1;

/** Fades the cell's right edge while it has more to its right. */
export function markMore(cell: HTMLElement): void {
  cell.classList.toggle("is-more", cell.scrollLeft + cell.clientWidth < cell.scrollWidth - 1);
}

/** Fits every `selector` cell under `root` now, when the web font arrives, when the root's width or
 * content changes and while a cell scrolls; `onFit` runs after each fit. Returns the disposer. */
export function watchScrollCells(root: HTMLElement, selector: string, onFit?: () => void): () => void {
  const fit = () => {
    for (const cell of root.querySelectorAll<HTMLElement>(selector)) {
      markMore(cell);
      if (overflows(cell) && !cell.querySelector("a, button")) cell.tabIndex = 0;
      else cell.removeAttribute("tabindex");
    }
    onFit?.();
  };
  let frame = 0;
  const later = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(fit);
  };
  fit();
  void document.fonts?.ready.then(later);
  const resized = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(later);
  resized?.observe(root);
  const changed = typeof MutationObserver === "undefined" ? null : new MutationObserver(later);
  changed?.observe(root, { childList: true, subtree: true, characterData: true });
  // scroll does not bubble: one capturing listener follows every cell
  const scrolled = (event: Event) => {
    if (event.target instanceof HTMLElement && event.target.matches(selector)) markMore(event.target);
  };
  root.addEventListener("scroll", scrolled, { capture: true, passive: true });
  return () => {
    cancelAnimationFrame(frame);
    resized?.disconnect();
    changed?.disconnect();
    root.removeEventListener("scroll", scrolled, { capture: true });
  };
}

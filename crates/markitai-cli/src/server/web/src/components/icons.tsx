// One icon family (Phosphor, regular weight) on a 256 grid, drawn in the
// current text colour; the brand mark is drawn here by hand.
import { ICON_PATHS, type IconName } from "./icon-paths.ts";

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 256 256" fill="currentColor" aria-hidden="true" focusable="false">
      {ICON_PATHS[name].map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}

/** Brand tile: a dark rounded square with a white zigzag M, identical in both themes. */
export function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg class="logo" width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true" focusable="false">
      <rect width="32" height="32" rx="8" fill="#18181b" />
      <rect class="logo-edge" x="0.5" y="0.5" width="31" height="31" rx="7.5" fill="none" stroke-width="1" />
      <path
        d="M8 23V9L13 17L16 11L19 17L24 9V23"
        stroke="#fff"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        fill="none"
      />
    </svg>
  );
}

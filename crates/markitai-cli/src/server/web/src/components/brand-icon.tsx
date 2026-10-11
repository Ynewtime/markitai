// Provider marks: the vendored lobehub monochrome marks (brands.js, loaded as
// the models settings open) drawn in the current text colour, or a monogram
// tile for a provider without a mark. Neither is announced: the provider's
// name always follows.
import { useEffect, useState } from "preact/hooks";
import { monogram } from "../lib/provider-catalog.ts";
import type { BrandMark } from "./brand-marks.ts";
import { Icon } from "./icons.tsx";

type Marks = Record<string, BrandMark>;
let marks: Marks | null = null;
let loading: Promise<Marks> | null = null;

/** Start loading the marks, once; a failed load leaves monograms and is retried next time. */
export function loadBrandMarks(): Promise<Marks> {
  loading ??= import("/ui/brands.js").then(
    (module) => (marks = module.BRAND_MARKS),
    () => {
      loading = null;
      return (marks = {});
    },
  );
  return loading;
}

/** The marks, or null while they load (tiles then keep their place, empty). */
export function useBrandMarks(): Marks | null {
  const [value, setValue] = useState(marks);
  useEffect(() => {
    let live = true;
    void loadBrandMarks().then((loaded) => {
      if (live) setValue(loaded);
    });
    return () => {
      live = false;
    };
  }, []);
  return value;
}

export function BrandIcon({
  marks: loaded,
  icon,
  label,
  size,
  generic = false,
}: {
  marks: Marks | null;
  icon: string | null;
  label: string;
  size: 22 | 32;
  /** Not one provider (any OpenAI-compatible endpoint): a plug instead of letters. */
  generic?: boolean;
}) {
  if (loaded === null) return <span class={`brand-icon is-${size}`} aria-hidden="true" />;
  const mark = icon === null ? undefined : loaded[icon];
  if (mark) {
    return (
      <svg
        class={`brand-icon is-${size}`}
        width={size}
        height={size}
        viewBox={mark.viewBox}
        fill="currentColor"
        fill-rule={mark.rule ?? "evenodd"}
        aria-hidden="true"
        focusable="false"
      >
        {mark.paths.map((path) =>
          typeof path === "string" ? <path key={path} d={path} /> : <path key={path[0]} d={path[0]} fill-opacity={path[1]} />,
        )}
      </svg>
    );
  }
  return (
    <span class={`brand-tile is-${size}`} aria-hidden="true">
      {generic ? <Icon name="Plugs" size={size === 32 ? 18 : 13} /> : monogram(label)}
    </span>
  );
}

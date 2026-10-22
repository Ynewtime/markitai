// What a page or list says when it has nothing to show, or cannot show it: a
// framed panel with an icon, a title, one line, and at most one primary and one
// secondary action. One layout serves every case (no conversions, no models, a
// filter without matches, a list that could not load, a conversion that no
// longer exists, a page that does not exist). The panel itself is not a control.
import type { Ref } from "preact";
import type { IconName } from "./icon-paths.ts";
import { Icon } from "./icons.tsx";

export interface StateAction {
  label: string;
  run: () => void;
  ref?: Ref<HTMLButtonElement>;
}

export function StateBlock({
  icon,
  title,
  text,
  primary,
  secondary,
  tone,
  heading = "p",
}: {
  icon: IconName;
  title: string;
  text?: string;
  primary?: StateAction;
  secondary?: StateAction;
  tone?: "error";
  /** The title's element: `h1` for a page of its own. */
  heading?: "p" | "h1" | "h2";
}) {
  const Title = heading;
  return (
    <div class={`state${tone ? ` is-${tone}` : ""}`}>
      <span class="state-icon" aria-hidden="true">
        <Icon name={icon} size={20} />
      </span>
      <Title class="state-title">{title}</Title>
      {text && <p class="state-text">{text}</p>}
      {(primary || secondary) && (
        <div class="state-actions">
          {primary && (
            <button ref={primary.ref} type="button" class="btn btn-primary" onClick={primary.run}>
              {primary.label}
            </button>
          )}
          {secondary && (
            <button ref={secondary.ref} type="button" class="btn" onClick={secondary.run}>
              {secondary.label}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

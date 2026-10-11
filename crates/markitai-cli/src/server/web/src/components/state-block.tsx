// What a page or list says when it has nothing to show, or cannot show it:
// one title, one line, and the action that changes it. The same block serves
// an empty list, a filter without matches, a list that could not load, and a
// page that does not exist.
import type { JSX } from "preact";

export interface StateAction {
  label: string;
  run: () => void;
  primary?: boolean;
}

export function StateBlock({
  title,
  text,
  actions = [],
  tone,
  heading = "p",
}: {
  title?: string;
  text?: string;
  actions?: StateAction[];
  tone?: "error";
  /** The title's element: `h1` for a page of its own. */
  heading?: "p" | "h1" | "h2";
}) {
  const Title = heading as keyof JSX.IntrinsicElements;
  return (
    <div class={`state${tone ? ` is-${tone}` : ""}`}>
      {title && <Title class="state-title">{title}</Title>}
      {text && <p class="state-text">{text}</p>}
      {actions.length > 0 && (
        <div class="state-actions">
          {actions.map((action) => (
            <button key={action.label} type="button" class={action.primary ? "btn btn-primary" : "btn"} onClick={action.run}>
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

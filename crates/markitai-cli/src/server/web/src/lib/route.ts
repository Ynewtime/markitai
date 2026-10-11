// The workbench's addresses. Each view is a real path the backend answers with
// the page, so reloads, Back and new tabs keep it; the selected conversion and
// the open preview ride in the query (`?sel=<job>/<item>&view=preview`).

export type Section = "models" | "appearance" | "access";
export const SECTIONS: Section[] = ["models", "appearance", "access"];

export type Route =
  | { view: "convert" }
  | { view: "jobs" }
  /** `section` null: the phone's settings index (wide screens show Models). */
  | { view: "settings"; section: Section | null };

export function parseRoute(pathname: string): Route {
  const path = pathname.replace(/\/+$/, "") || "/";
  if (path === "/jobs") return { view: "jobs" };
  if (path === "/settings") return { view: "settings", section: null };
  const section = /^\/settings\/(models|appearance|access)$/.exec(path)?.[1] as Section | undefined;
  if (section) return { view: "settings", section };
  return { view: "convert" };
}

export function routePath(route: Route): string {
  if (route.view === "jobs") return "/jobs";
  if (route.view === "settings") return route.section ? `/settings/${route.section}` : "/settings";
  return "/";
}

export type PaneMode = "inspector" | "preview" | "compare";

export interface ListState {
  selected: string | null;
  mode: PaneMode;
}

export function parseListState(search: string): ListState {
  const params = new URLSearchParams(search);
  const selected = params.get("sel");
  const view = params.get("view");
  return {
    selected: selected && /^(?:job:)?[\w-]{1,64}(?:\/[\w-]{1,64})?$/.test(selected) ? selected : null,
    mode: view === "preview" || view === "compare" ? view : "inspector",
  };
}

export function listQuery(state: ListState): string {
  const params = new URLSearchParams();
  if (state.selected) params.set("sel", state.selected);
  if (state.selected && state.mode !== "inspector") params.set("view", state.mode);
  const text = params.toString();
  return text ? `?${text}` : "";
}

/** A plain left click on a same-origin link: the page handles it instead of the browser. */
export function ownClick(event: MouseEvent): boolean {
  return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && !event.defaultPrevented;
}

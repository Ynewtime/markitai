// The workbench's addresses. Each view is a real path the backend answers with
// the page, so reloads, Back and new tabs keep it; the selected conversion and
// the open preview ride in the query (`?sel=<job>/<item>&view=preview`).

export type Section = "appearance" | "models" | "access" | "config";
export const SECTIONS: Section[] = ["appearance", "models", "access", "config"];
/** Sections only the admin token opens (v1.13); a user token sees Appearance only. */
export const ADMIN_SECTIONS: ReadonlySet<Section> = new Set(["models", "access", "config"]);

export type Route =
  | { view: "convert" }
  | { view: "jobs" }
  /** `section` null: the phone's settings index (wide screens show Models). */
  | { view: "settings"; section: Section | null }
  /** An address that is no page of the workbench (the server answers it with 404). */
  | { view: "missing"; path: string };

export function parseRoute(pathname: string): Route {
  const path = pathname.replace(/\/+$/, "") || "/";
  if (path === "/" || path === "/index.html") return { view: "convert" };
  if (path === "/jobs") return { view: "jobs" };
  if (path === "/settings") return { view: "settings", section: null };
  const section = /^\/settings\/(models|appearance|access|config)$/.exec(path)?.[1] as Section | undefined;
  if (section) return { view: "settings", section };
  return { view: "missing", path };
}

export function routePath(route: Route): string {
  if (route.view === "jobs") return "/jobs";
  if (route.view === "settings") return route.section ? `/settings/${route.section}` : "/settings";
  if (route.view === "missing") return route.path;
  return "/";
}

export type PaneMode = "inspector" | "preview" | "compare";
/** A conversion opens on its Preview (Details for a submission, or a conversion without a result). */
export const OPEN_MODE: PaneMode = "preview";

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
    mode: view === "details" ? "inspector" : view === "compare" ? "compare" : OPEN_MODE,
  };
}

export function listQuery(state: ListState): string {
  const params = new URLSearchParams();
  if (state.selected) params.set("sel", state.selected);
  if (state.selected && state.mode !== OPEN_MODE) params.set("view", state.mode === "inspector" ? "details" : state.mode);
  const text = params.toString();
  return text ? `?${text}` : "";
}

/** A plain left click on a same-origin link: the page handles it instead of the browser. */
export function ownClick(event: MouseEvent): boolean {
  return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && !event.defaultPrevented;
}

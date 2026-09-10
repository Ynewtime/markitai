import { Component, type ErrorInfo, type ReactNode } from "react";

/** Last-resort boundary: a render crash must still leave a usable page with the
 * reason and a reload, not a blank screen. Kept in its own module so it can be
 * tested without importing the entry point (which mounts the app). */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Markitai UI crashed", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error === null) return this.props.children;
    // The boundary runs before any locale provider is available, so read the
    // document language the boot script set.
    const zh =
      typeof document !== "undefined" &&
      document.documentElement.lang.startsWith("zh");
    return (
      <div className="crash" role="alert">
        <h1>{zh ? "界面出错了" : "Something went wrong"}</h1>
        <p>{this.state.error.message}</p>
        <button type="button" onClick={() => window.location.reload()}>
          {zh ? "重新加载" : "Reload"}
        </button>
      </div>
    );
  }
}

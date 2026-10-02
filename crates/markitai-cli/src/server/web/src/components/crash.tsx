// Last resort: a rendering failure leaves a readable page with the reason and
// a reload, in the language boot.js already set on <html>.
import { Component, type ComponentChildren } from "preact";

export class CrashBoundary extends Component<{ children: ComponentChildren }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static override getDerivedStateFromError(error: Error) {
    return { error };
  }

  override componentDidCatch(error: Error) {
    console.error("Markitai interface failed", error);
  }

  override render() {
    if (this.state.error === null) return this.props.children;
    const zh = document.documentElement.lang.startsWith("zh");
    return (
      <div class="crash" role="alert">
        <h1>{zh ? "界面出错了" : "Something went wrong"}</h1>
        <p>{this.state.error.message}</p>
        <button type="button" class="btn btn-ghost" onClick={() => location.reload()}>
          {zh ? "重新加载" : "Reload"}
        </button>
      </div>
    );
  }
}

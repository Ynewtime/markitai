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
    const lang = document.documentElement.lang;
    const [title, reload] = lang.startsWith("zh") ? ["界面出错了", "重新加载"] : lang.startsWith("ja") ? ["問題が発生しました", "再読み込み"] : ["Something went wrong", "Reload"];
    return (
      <div class="crash" role="alert">
        <h1>{title}</h1>
        <p>{this.state.error.message}</p>
        <button type="button" class="btn btn-ghost" onClick={() => location.reload()}>
          {reload}
        </button>
      </div>
    );
  }
}

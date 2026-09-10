/** Single inline printed-red mono line. No toasts, no modals.
 * `detail` keeps the raw server message reachable on hover/focus without
 * printing it into the reading line. */
export function ErrorInline({ text, detail }: { text: string; detail?: string }) {
  return (
    <p className="errline" role="alert" title={detail}>
      {text}
    </p>
  );
}

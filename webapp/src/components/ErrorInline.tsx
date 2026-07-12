/** Single inline printed-red mono line. No toasts, no modals. */
export function ErrorInline({ text }: { text: string }) {
  return (
    <p className="errline" role="alert">
      {text}
    </p>
  );
}

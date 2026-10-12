// The shape of a document while it loads: a few grey lines where its text
// will be, inside the same frame, so nothing moves when the text arrives. It
// fades in after a moment (a quick result never flashes it) and says what is
// happening to assistive technology.
export function Skeleton({ label, kind = "doc" }: { label: string; kind?: "doc" | "source" | "list" | "page" }) {
  const lines = kind === "list" ? 5 : kind === "page" ? 0 : 12;
  return (
    <div class={`skeleton-doc is-${kind}`} role="status">
      <p class="skeleton-label">{label}</p>
      <div class="skeleton-lines" aria-hidden="true">
        {Array.from({ length: lines }, (_, index) => (
          <span key={index} />
        ))}
      </div>
    </div>
  );
}

// The version line's parts, each kept whole when the line wraps (`10-14` never breaks at its hyphen).
export function VersionText({ text }: { text: string }) {
  const parts = text.split(" ");
  return (
    <>
      {parts.map((part, index) => (
        <span key={index}>
          {index > 0 && " "}
          <span class="version-part">{part}</span>
        </span>
      ))}
    </>
  );
}

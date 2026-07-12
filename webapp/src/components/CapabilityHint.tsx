import type { Dict } from "../i18n";

/** One mono line shown when llm.configured is false — the embedded link
 * opens the in-app LLM settings panel. */
export function CapabilityHint({ t, onOpenSettings }: { t: Dict; onOpenSettings: () => void }) {
  return (
    <p className="caphint">
      {t.capHintPre}
      <button type="button" className="linklike" onClick={onOpenSettings}>
        {t.capHintLink}
      </button>
      {t.capHintPost}
    </p>
  );
}

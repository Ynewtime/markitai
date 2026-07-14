import type { ProviderConnection } from "../../api/types";
import type { Dict } from "../../i18n";

export function ProviderPicker({
  t,
  providers,
  selectedId,
  onSelect,
}: {
  t: Dict;
  providers: ProviderConnection[];
  selectedId: string | null;
  onSelect: (provider: ProviderConnection) => void;
}) {
  const groups = ["local_cli", "oauth", "environment", "configured", "common"];
  return (
    <div className="provider-picker">
      {groups.map((kind) => {
        const cards = providers.filter((provider) => provider.kind === kind);
        if (cards.length === 0) return null;
        return (
          <section key={kind}>
            <h3 className="picker-group mono">{t.providerGroup(kind)}</h3>
            <div className="provider-grid">
              {cards.map((provider) => (
                <button
                  key={provider.id}
                  type="button"
                  className={selectedId === provider.id ? "provider-card on" : "provider-card"}
                  aria-pressed={selectedId === provider.id}
                  onClick={() => onSelect(provider)}
                >
                  <span className="provider-card-name">{provider.label}</span>
                  <span className="provider-card-meta mono">
                    {t.providerCardMeta(provider.kind, provider.status, provider.source)}
                  </span>
                </button>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

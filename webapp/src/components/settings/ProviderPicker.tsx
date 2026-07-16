import type { ProviderConnection } from "../../api/types";
import type { Dict } from "../../i18n";
import { ConfirmDeletePopover } from "../ConfirmDeletePopover";

export function ProviderPicker({
  t,
  providers,
  selectedId,
  onSelect,
  onEditProvider,
  onDeleteProvider,
}: {
  t: Dict;
  providers: ProviderConnection[];
  selectedId: string | null;
  onSelect: (provider: ProviderConnection) => void;
  onEditProvider?: (provider: ProviderConnection) => void;
  onDeleteProvider?: (provider: ProviderConnection) => Promise<boolean>;
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
              {cards.map((provider) => {
                const unavailable =
                  (provider.kind === "local_cli" || provider.kind === "oauth") &&
                  provider.status !== "ready";
                const manageable =
                  provider.kind === "configured" &&
                  provider.provider_id !== undefined &&
                  onEditProvider !== undefined &&
                  onDeleteProvider !== undefined;
                const modelCount = provider.model_count ?? 0;
                return (
                  <div
                    key={provider.id}
                    className={
                      selectedId === provider.id
                        ? "provider-card on"
                        : "provider-card"
                    }
                  >
                    <button
                      type="button"
                      className="provider-card-select"
                      aria-label={t.selectProvider(provider.label)}
                      aria-pressed={selectedId === provider.id}
                      disabled={unavailable}
                      onClick={() => onSelect(provider)}
                    >
                      <span className="provider-card-name">{provider.label}</span>
                      <span className="provider-card-meta mono">
                        {provider.api_base ??
                          t.providerCardMeta(
                            provider.kind,
                            provider.status,
                            provider.source,
                          )}
                      </span>
                    </button>
                    {manageable && (
                      <span className="provider-card-controls">
                        <span className="minibadge">
                          {t.providerModels(modelCount)}
                        </span>
                        <span className="prov-acts">
                          <button
                            type="button"
                            className="rowact"
                            aria-label={t.editProvider(provider.label)}
                            onClick={() => onEditProvider(provider)}
                          >
                            {t.edit}
                          </button>
                          <ConfirmDeletePopover
                            triggerLabel={t.deleteProvider(provider.label)}
                            title={t.deleteProviderTitle(provider.label)}
                            description={t.deleteProviderDescription(modelCount)}
                            confirmLabel={t.deletePermanently}
                            cancelLabel={t.cancel}
                            busyLabel={t.deleting}
                            onConfirm={() => onDeleteProvider(provider)}
                          />
                        </span>
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}

// Provider cards grouped by where their credentials come from. Saved providers
// show their model count and can be edited or deleted in place.
import type { ProviderCard } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";

// "compatible": the OpenAI-compatible providers beyond the reference's list.
const GROUPS = ["environment", "configured", "common", "compatible"];

export function providerLabel(t: Dict, card: { provider: string; label: string }): string {
  if (card.provider === "custom") return t.providerCustom;
  if (!card.label || card.label === "Unknown provider") return card.provider ? `${t.providerUnknown} (${card.provider})` : t.providerUnknown;
  return card.label;
}

/** The added OpenAI-compatible providers name their documented host on the card. */
function endpointHost(card: ProviderCard): string | null {
  if (card.kind !== "compatible" || !card.default_base) return null;
  try {
    return new URL(card.default_base).host;
  } catch {
    return null;
  }
}

export function ProviderPicker({
  t,
  providers,
  onSelect,
  onEdit,
  onDelete,
}: {
  t: Dict;
  providers: ProviderCard[];
  onSelect: (card: ProviderCard) => void;
  onEdit: (card: ProviderCard) => void;
  onDelete: (card: ProviderCard) => Promise<boolean>;
}) {
  const kinds = [...GROUPS, ...new Set(providers.map((card) => card.kind).filter((kind) => !GROUPS.includes(kind)))];
  return (
    <div class="provider-groups">
      {kinds.map((kind) => {
        const cards = providers.filter((card) => card.kind === kind);
        if (!cards.length) return null;
        return (
          <section key={kind}>
            <h3 class="group-title">{t.providerGroup(kind)}</h3>
            <div class="provider-grid">
              {cards.map((card) => {
                const label = providerLabel(t, card);
                const manageable = card.kind === "configured" && card.provider_id !== undefined;
                const models = card.model_count ?? 0;
                return (
                  <div key={card.id} class="provider-card">
                    <button
                      type="button"
                      class="provider-pick"
                      aria-label={t.selectProvider(label)}
                      disabled={card.status === "disabled"}
                      onClick={() => onSelect(card)}
                    >
                      <span class="provider-name">{label}</span>
                      <span class="provider-meta">
                        {card.api_base ??
                          endpointHost(card) ??
                          (card.kind === "compatible" && card.key_optional && card.default_base === null
                            ? t.serverAddressRequired
                            : t.providerCardMeta(card.kind, card.status, card.source))}
                      </span>
                    </button>
                    {manageable && (
                      <span class="provider-tools">
                        <span class="mini-pill">{t.providerModels(models)}</span>
                        <span class="text-actions">
                          <button type="button" class="text-btn" aria-label={t.editProvider(label)} onClick={() => onEdit(card)}>
                            {t.edit}
                          </button>
                          <ConfirmPopover
                            triggerLabel={t.deleteProvider(label)}
                            title={t.deleteProviderTitle(label)}
                            description={t.deleteProviderDescription(models)}
                            confirmLabel={t.deletePermanently}
                            cancelLabel={t.cancel}
                            busyLabel={t.deleting}
                            onConfirm={() => onDelete(card)}
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

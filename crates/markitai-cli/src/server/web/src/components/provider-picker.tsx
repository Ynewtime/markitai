// The provider picker of Add models: one search over everything, saved
// connections as rows (how each authenticates, its model count, Edit opens the
// provider's page where its key and address are managed, Delete removes it in
// place), the common eight as cards, then every other provider by name under a
// group filter. A provider is one saved row (the server's key folds into its
// connection) and is marked as added below; selecting it opens that row.
import { useState } from "preact/hooks";
import type { ProviderCard } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { mergeKimi } from "../lib/kimi.ts";
import { arrangeProviders, atOwnEndpoint, iconKey, providerLabel, savedAuth, type ProviderFilter } from "../lib/provider-catalog.ts";
import { BrandIcon, useBrandMarks } from "./brand-icon.tsx";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";

export { providerLabel };

/** A saved provider whose key and address this page can change. */
export const manageable = (card: ProviderCard): boolean => card.kind === "configured" && card.provider_id !== undefined;

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
  const marks = useBrandMarks();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ProviderFilter>("all");
  const label = (card: ProviderCard) => providerLabel(t, card);
  const view = arrangeProviders(mergeKimi(providers), { query, filter, label });
  const shown = view.filters.includes(filter) ? filter : "all";
  const searching = query.trim() !== "";
  const found = view.saved.length + view.common.length + view.all.length;
  /** How the row connects, and where when it is not the provider's own endpoint. */
  const meta = (card: ProviderCard) => {
    const auth = savedAuth(card, view.folded.get(card.id) ?? null, view.variables.get(card.provider) ?? null);
    const how = auth === null ? null : auth.kind === "environment" ? t.savedUsing(auth.variable) : auth.kind === "key" ? t.savedKey : t.savedNoKey;
    const where = card.api_base && (card.provider === "custom" || !atOwnEndpoint(card)) ? card.api_base : null;
    const parts = [how, where].filter((part): part is string => part !== null);
    return parts.length ? parts.join(" · ") : (card.api_base ?? t.providerCardMeta(card.kind, card.status, card.source));
  };
  const icon = (card: ProviderCard, name: string, size: 22 | 32) => (
    <BrandIcon marks={marks} icon={iconKey(card)} label={name} size={size} generic={card.provider === "custom"} />
  );
  return (
    <div class="provider-picker">
      <label class="search provider-search">
        <Icon name="MagnifyingGlass" size={14} />
        <input
          type="search"
          value={query}
          placeholder={t.searchProviders(view.total)}
          aria-label={t.searchProviders(view.total)}
          onInput={(event) => setQuery(event.currentTarget.value)}
        />
      </label>
      <div class="sr-only" role="status">
        {searching ? t.providersFound(found) : ""}
      </div>
      {view.saved.length > 0 && (
        <section aria-labelledby="providers-saved">
          <h2 id="providers-saved" class="group-title">
            {t.providersSaved}
          </h2>
          <ul class="provider-rows">
            {view.saved.map((card) => {
              const name = label(card);
              const models = card.model_count ?? 0;
              return (
                <li key={card.id} class="provider-row">
                  <button type="button" class="provider-pick" disabled={card.status === "disabled"} onClick={() => onSelect(card)}>
                    {icon(card, name, 22)}
                    <span class="provider-text">
                      <span class="provider-name">{name}</span>
                      <span class="provider-meta">{meta(card)}</span>
                    </span>
                  </button>
                  {manageable(card) && (
                    <span class="provider-tools">
                      <span class="mini-pill">{t.providerModels(models)}</span>
                      <span class="text-actions">
                        <button type="button" class="text-btn" aria-label={t.editProvider(name)} onClick={() => onEdit(card)}>
                          {t.edit}
                        </button>
                        <ConfirmPopover
                          triggerLabel={t.deleteProvider(name)}
                          title={t.deleteProviderTitle(name)}
                          description={t.deleteProviderDescription(models)}
                          confirmLabel={t.deletePermanently}
                          cancelLabel={t.cancel}
                          busyLabel={t.deleting}
                          onConfirm={() => onDelete(card)}
                        />
                      </span>
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {view.common.length > 0 && (
        <section aria-labelledby="providers-common">
          <h2 id="providers-common" class="group-title">
            {t.providersCommon}
          </h2>
          <ul class="provider-tiles">
            {view.common.map((choice) => (
              <li key={choice.provider}>
                <button type="button" class="provider-tile" disabled={choice.card.status === "disabled"} onClick={() => onSelect(choice.card)}>
                  {icon(choice.card, choice.label, 32)}
                  <span class="provider-tile-text">
                    <span class="provider-name" title={choice.label}>
                      {choice.label}
                    </span>
                    {choice.added && <Added t={t} />}
                  </span>
                </button>
              </li>
            ))}
            {/* A search can leave a row part-filled: empty cells finish it,
                four columns wide, two on a phone (the rest hide there). */}
            {Array.from({ length: (4 - (view.common.length % 4)) % 4 }, (_, index) => (
              <li key={`fill-${index}`} class={index < view.common.length % 2 ? "provider-fill" : "provider-fill is-wide"} aria-hidden="true" />
            ))}
          </ul>
        </section>
      )}
      <section aria-labelledby="providers-all">
        <h2 id="providers-all" class="group-title">
          {t.providersAll(view.all.length)}
        </h2>
        {view.filters.length > 1 && (
          <div class="provider-filter">
            <div class="chips" role="group" aria-label={t.providerFilterAria}>
              {view.filters.map((value) => (
                <button
                  key={value}
                  type="button"
                  class={shown === value ? "filter-chip is-on" : "filter-chip"}
                  aria-pressed={shown === value}
                  onClick={() => setFilter(value)}
                >
                  {t.providerFilter(value)}
                </button>
              ))}
            </div>
          </div>
        )}
        {view.all.length > 0 ? (
          <ul class="provider-list">
            {view.all.map((choice) => (
              <li key={choice.provider}>
                <button type="button" class="provider-item" disabled={choice.card.status === "disabled"} onClick={() => onSelect(choice.card)}>
                  {icon(choice.card, choice.label, 22)}
                  <span class="provider-item-name" title={choice.label}>
                    {choice.label}
                  </span>
                  {choice.added && <Added t={t} />}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p class="dialog-dim">{t.providersNoMatch}</p>
        )}
      </section>
    </div>
  );
}

/** A provider that already has a saved row: choosing it opens that connection. */
function Added({ t }: { t: Dict }) {
  return (
    <span class="provider-added">
      <Icon name="Check" size={12} />
      {t.providerAdded}
    </span>
  );
}

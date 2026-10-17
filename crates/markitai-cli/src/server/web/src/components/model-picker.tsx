// Choose models from a provider's catalogue, in an inset panel whose head holds
// the search and Select all: search, select what is visible, add an identifier
// by hand, and set the routing group and weight (all in plain sight, on the grid
// of the connection's key and address). Models
// already configured with the same group and endpoint cannot be added twice:
// a saved model shows ticked on its listed row (also when it was saved under an
// alias the provider defines), or as a row of its own when the list lacks it.
// The selection count sits with the dialog's pinned actions.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Deployment, ModelCandidate } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { manualModelId, parseWeight, pickerRows, stepWeight, WEIGHT_MAX, WEIGHT_PLACEHOLDER } from "../lib/models.ts";
import { Icon } from "./icons.tsx";

export function ModelPicker({
  t,
  provider,
  candidates,
  deployments,
  apiBase,
  group,
  weight,
  selected,
  dropped,
  onGroup,
  onWeight,
  onSelected,
  onDropped,
  manualOnly = false,
  groups = true,
  weightHint,
  defaultBase,
  onWeightValid,
  addBlocked,
  addWarning,
}: {
  t: Dict;
  /** The provider lists no models: manual entry is open and explained. */
  manualOnly?: boolean;
  /** v1.15: a user's models form one pool, so its picker sets no routing group. */
  groups?: boolean;
  /** What the weight does, when it differs from the admin's groups. */
  weightHint?: string;
  /** The provider's documented address: a saved model without one of its own is there. */
  defaultBase?: string | null;
  /** Whether the weight field holds a weight Save may write (a whole number from 0 to 100). */
  onWeightValid?: (valid: boolean) => void;
  /** Why a model cannot be added now (the provider's runtime cannot run): entry by hand is off, with this said. */
  addBlocked?: string;
  /** A caution for entry by hand that stays open: it may not work until the provider answers. */
  addWarning?: string;
  provider: string;
  candidates: ModelCandidate[];
  deployments: Deployment[];
  apiBase: string;
  group: string;
  weight: number;
  selected: Set<string>;
  /** Configured models the reader unticked; Save removes them. */
  dropped: Set<string>;
  onGroup: (value: string) => void;
  onWeight: (value: number) => void;
  onSelected: (value: Set<string>) => void;
  onDropped: (value: Set<string>) => void;
}) {
  const [query, setQuery] = useState("");
  const [manual, setManual] = useState("");
  const [added, setAdded] = useState<ModelCandidate[]>([]);
  const [weightDraft, setWeightDraft] = useState<string | null>(null);
  const selectAll = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setQuery("");
    setManual("");
    setAdded([]);
    setWeightDraft(null);
  }, [provider]);

  const all = useMemo(() => {
    const byModel = new Map<string, ModelCandidate>();
    for (const candidate of [...candidates, ...added]) byModel.set(candidate.model, candidate);
    return [...byModel.values()];
  }, [candidates, added]);
  // Every listed model with the saved ones it stands for, then saved models the
  // list does not show; the configured ones head the list.
  const rows = useMemo(() => pickerRows(provider, all, deployments, group, apiBase, defaultBase), [provider, all, deployments, group, apiBase, defaultBase]);
  const configuredModels = useMemo(() => new Set(rows.filter((row) => row.saved.length > 0).map((row) => row.model)), [rows]);
  const configured = (model: string) => configuredModels.has(model);
  const needle = query.trim().toLowerCase();
  const visible = rows.filter(
    (row) => !needle || [row.model, row.label, ...row.aliases].some((text) => text.toLowerCase().includes(needle)),
  );
  // A provider that lists models but not this saved one says so; one without a list has nothing to compare.
  const listed = candidates.length > 0;
  const weightError = weightDraft !== null && weightDraft.trim() !== "" && parseWeight(weightDraft) === null;
  const weightEmpty = weightDraft !== null && weightDraft.trim() === "";
  useEffect(() => onWeightValid?.(!weightError && !weightEmpty), [weightError, weightEmpty]);
  // The tick is the state this connection will keep: a configured model is on
  // unless it was unticked, another one is on once it was ticked.
  const chosen = (model: string) => (configured(model) ? !dropped.has(model) : selected.has(model));
  const chosenVisible = visible.filter((candidate) => chosen(candidate.model));
  const allChosen = visible.length > 0 && chosenVisible.length === visible.length;

  useEffect(() => {
    if (selectAll.current) selectAll.current.indeterminate = chosenVisible.length > 0 && !allChosen;
  }, [allChosen, chosenVisible.length]);

  const toggle = (model: string) => {
    if (configured(model)) {
      const next = new Set(dropped);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      onDropped(next);
      return;
    }
    const next = new Set(selected);
    if (next.has(model)) next.delete(model);
    else next.add(model);
    onSelected(next);
  };
  const toggleVisible = () => {
    if (allChosen) {
      // Clearing the row leaves every configured model out and nothing added.
      onDropped(new Set(visible.filter((candidate) => configured(candidate.model)).map((candidate) => candidate.model)));
      onSelected(new Set());
      return;
    }
    onDropped(new Set());
    onSelected(new Set(visible.filter((candidate) => !configured(candidate.model)).map((candidate) => candidate.model)));
  };
  const addManual = () => {
    const model = manualModelId(provider, manual);
    if (!model) return;
    setAdded((previous) => [...previous.filter((candidate) => candidate.model !== model), { model, label: model.split("/").slice(1).join("/") || model, supports_vision: false }]);
    const next = new Set(selected);
    next.add(model);
    onSelected(next);
    setManual("");
  };

  return (
    <div class="picker">
      <div class="picker-panel">
        {/* Nothing to search or select in bulk when every ID is typed by hand. */}
        <div class="picker-tools" hidden={manualOnly}>
          <span class="picker-search">
            <Icon name="MagnifyingGlass" size={14} />
            <input type="search" value={query} placeholder={t.searchModels} aria-label={t.searchModels} onInput={(event) => setQuery(event.currentTarget.value)} />
          </span>
          <div class="picker-bar">
            <label class="check-line">
              <input ref={selectAll} type="checkbox" checked={allChosen} disabled={!visible.length} onChange={toggleVisible} />
              {t.selectAll}
            </label>
            <span class="picker-total">{t.pickerShown(visible.length, rows.length)}</span>
          </div>
        </div>
        <div class="picker-list" role="group" aria-label={t.modelsAvailable}>
          {visible.map((candidate) => {
            const done = configured(candidate.model);
            const removed = done && dropped.has(candidate.model);
            const note = candidate.unlisted && listed ? t.savedNotListed : candidate.aliases.length ? candidate.aliases.map((alias) => t.savedAsAlias(alias)).join(" · ") : null;
            return (
              <label key={candidate.model} class={removed ? "picker-option is-removed" : "picker-option"} data-unlisted={candidate.unlisted || undefined}>
                {/* The tick states what this connection keeps: a model it already
                    routes comes in ticked, and unticking it schedules a removal. */}
                <input type="checkbox" checked={chosen(candidate.model)} onChange={() => toggle(candidate.model)} />
                <span class="picker-option-main">
                  <span>{candidate.label}</span>
                  <span class="picker-id">{candidate.model}</span>
                  {note && <span class="picker-id picker-saved-note">{note}</span>}
                </span>
                {candidate.supports_vision && <span class="mini-pill">{t.visionTag}</span>}
                {done && <span class={removed ? "mini-pill is-removed" : "mini-pill"}>{removed ? t.willRemove : t.alreadyConfigured}</span>}
              </label>
            );
          })}
          {visible.length === 0 && <p class="picker-note">{addBlocked ? t.noListUntilRuntime : manualOnly && !needle ? t.manualModelsOnly : t.noModelsFound}</p>}
        </div>
      </div>
      {/* Model IDs by hand and the routing settings, in plain sight on the same grid as the key and address. */}
      <section class="picker-extra" aria-labelledby="picker-extra-title">
        <h3 id="picker-extra-title" class="group-title">
          {t.manualAndRouting}
        </h3>
        <div class="field-grid">
          <div class="field">
            <label class="field-label" for="manual-model">
              {t.manualModelLabel}
            </label>
            <div class="input-group">
              <input
                id="manual-model"
                type="text"
                value={manual}
                disabled={Boolean(addBlocked)}
                aria-describedby={addBlocked || addWarning ? "manual-model-note" : undefined}
                placeholder={provider === "azure" ? t.azureDeploymentPh : t.manualModelPh}
                onInput={(event) => setManual(event.currentTarget.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addManual();
                  }
                }}
              />
              <button type="button" class="btn" disabled={!manual.trim() || Boolean(addBlocked)} onClick={addManual}>
                {t.addModelId}
              </button>
            </div>
            {(addBlocked || addWarning) && (
              <p id="manual-model-note" class="field-hint manual-note">
                {addBlocked ?? addWarning}
              </p>
            )}
          </div>
          <span class="field-grid-gap" aria-hidden="true" />
          {groups && (
            <label class="field">
              <span class="field-label">{t.routingGroup}</span>
              <input type="text" value={group} onInput={(event) => onGroup(event.currentTarget.value)} />
            </label>
          )}
          <label class="field">
            <span class="field-label">{t.weight}</span>
            <input
              type="number"
              inputMode="numeric"
              min={0}
              max={WEIGHT_MAX}
              step={1}
              placeholder={WEIGHT_PLACEHOLDER}
              value={weightDraft ?? weight}
              aria-invalid={weightError || weightEmpty || undefined}
              aria-describedby={weightError || weightEmpty ? "weight-error weight-hint" : "weight-hint"}
              onInput={(event) => {
                const text = event.currentTarget.value;
                setWeightDraft(text);
                const next = parseWeight(text);
                if (next !== null) onWeight(next);
              }}
              onKeyDown={(event) => {
                // The browser's own stepping stops at min and max as well; this keeps a typed value in range too.
                if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
                event.preventDefault();
                const next = stepWeight(weightDraft ?? String(weight), weight, event.key === "ArrowUp" ? 1 : -1);
                setWeightDraft(String(next));
                onWeight(next);
              }}
              // A valid value settles; an invalid or empty one stays to be corrected.
              onBlur={() => weightDraft !== null && parseWeight(weightDraft) !== null && setWeightDraft(null)}
            />
          </label>
        </div>
        {(weightError || weightEmpty) && (
          <p id="weight-error" class="config-check is-error weight-error" role="alert">
            <Icon name="WarningCircle" size={14} />
            <span>{t.weightInvalid}</span>
          </p>
        )}
        <p id="weight-hint" class="field-hint">
          {weightHint ?? t.weightHint}
          <br />
          {t.weightRangeHint}
        </p>
      </section>
    </div>
  );
}

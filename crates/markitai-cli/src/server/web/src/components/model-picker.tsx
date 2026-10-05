// Choose models from a provider's catalogue: search, select what is visible,
// add an identifier by hand, and set the routing group and weight. Models
// already configured with the same group and endpoint cannot be added twice.
// The selection count sits with the dialog's pinned actions.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Deployment, ModelCandidate } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { manualModelId, sameEndpoint } from "../lib/models.ts";

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
}: {
  t: Dict;
  /** The provider lists no models: manual entry is open and explained. */
  manualOnly?: boolean;
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
  const configured = (model: string) =>
    deployments.some(
      (deployment) => deployment.model === model && deployment.routing_group === group.trim() && sameEndpoint(deployment.api_base, apiBase),
    );
  const needle = query.trim().toLowerCase();
  const matching = all.filter((candidate) => !needle || candidate.model.toLowerCase().includes(needle) || candidate.label.toLowerCase().includes(needle));
  // Models already enabled for this group and endpoint head the list.
  const visible = [...matching.filter((candidate) => configured(candidate.model)), ...matching.filter((candidate) => !configured(candidate.model))];
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
      {/* Nothing to search or select in bulk when every ID is typed by hand. */}
      <div class="picker-tools" hidden={manualOnly}>
        <input type="search" value={query} placeholder={t.searchModels} aria-label={t.searchModels} onInput={(event) => setQuery(event.currentTarget.value)} />
        <label class="check-line">
          <input ref={selectAll} type="checkbox" checked={allChosen} disabled={!visible.length} onChange={toggleVisible} />
          {t.selectAll}
        </label>
      </div>
      <div class="picker-list" role="group" aria-label={t.modelsAvailable}>
        {visible.map((candidate) => {
          const done = configured(candidate.model);
          const removed = done && dropped.has(candidate.model);
          return (
            <label key={candidate.model} class={removed ? "picker-option is-removed" : "picker-option"}>
              {/* The tick states what this connection keeps: a model it already
                  routes comes in ticked, and unticking it schedules a removal. */}
              <input type="checkbox" checked={chosen(candidate.model)} onChange={() => toggle(candidate.model)} />
              <span class="picker-option-main">
                <span>{candidate.label}</span>
                <span class="picker-id">{candidate.model}</span>
              </span>
              {candidate.supports_vision && <span class="mini-pill">{t.visionTag}</span>}
              {done && <span class={removed ? "mini-pill is-removed" : "mini-pill"}>{removed ? t.willRemove : t.alreadyConfigured}</span>}
            </label>
          );
        })}
        {visible.length === 0 && <p class="picker-note">{manualOnly && !needle ? t.manualModelsOnly : t.noModelsFound}</p>}
      </div>
      {/* Both folds share one quiet line: an opened one takes the line below. */}
      <div class="picker-folds">
        <details class="disclosure" open={manualOnly}>
          <summary>{t.manualModelToggle}</summary>
          <div class="manual-row">
            <input
              type="text"
              value={manual}
              placeholder={provider === "azure" ? t.azureDeploymentPh : t.manualModelPh}
              onInput={(event) => setManual(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  addManual();
                }
              }}
            />
            <button type="button" class="btn btn-ghost btn-sm" onClick={addManual}>
              {t.addManual}
            </button>
          </div>
        </details>
        <div class="picker-foot">
          <details class="disclosure">
            <summary>{t.advanced}</summary>
            <div class="field-grid">
              <label class="field">
                <span class="field-label">{t.routingGroup}</span>
                <input type="text" value={group} onInput={(event) => onGroup(event.currentTarget.value)} />
              </label>
              <label class="field">
                <span class="field-label">{t.weight}</span>
                <input
                  type="number"
                  min={0}
                  value={weightDraft ?? weight}
                  onInput={(event) => {
                    setWeightDraft(event.currentTarget.value);
                    onWeight(Math.max(0, Math.floor(Number(event.currentTarget.value)) || 0));
                  }}
                  onBlur={() => setWeightDraft(null)}
                />
              </label>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}

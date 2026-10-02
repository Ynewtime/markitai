// Choose models from a provider's catalogue: search, select what is visible,
// add an identifier by hand, and set the routing group and weight. Models
// already configured with the same group and endpoint cannot be added twice.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Deployment, ModelCandidate } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";

export const MAX_MODELS = 50;

const normalizeBase = (value: string | null | undefined) => (value ?? "").trim().replace(/\/+$/, "").toLowerCase();

export function ModelPicker({
  t,
  provider,
  candidates,
  deployments,
  apiBase,
  group,
  weight,
  selected,
  onGroup,
  onWeight,
  onSelected,
}: {
  t: Dict;
  provider: string;
  candidates: ModelCandidate[];
  deployments: Deployment[];
  apiBase: string;
  group: string;
  weight: number;
  selected: Set<string>;
  onGroup: (value: string) => void;
  onWeight: (value: number) => void;
  onSelected: (value: Set<string>) => void;
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
      (deployment) =>
        deployment.model === model && deployment.routing_group === group.trim() && normalizeBase(deployment.api_base) === normalizeBase(apiBase),
    );
  const needle = query.trim().toLowerCase();
  const visible = all.filter((candidate) => !needle || candidate.model.toLowerCase().includes(needle) || candidate.label.toLowerCase().includes(needle));
  const selectable = visible.filter((candidate) => !configured(candidate.model));
  const chosenVisible = selectable.filter((candidate) => selected.has(candidate.model));
  const allChosen = selectable.length > 0 && chosenVisible.length === selectable.length;

  useEffect(() => {
    if (selectAll.current) selectAll.current.indeterminate = chosenVisible.length > 0 && !allChosen;
  }, [allChosen, chosenVisible.length]);

  const toggle = (model: string) => {
    const next = new Set(selected);
    if (next.has(model)) next.delete(model);
    else if (next.size < MAX_MODELS) next.add(model);
    onSelected(next);
  };
  const toggleVisible = () => {
    const next = new Set(selected);
    if (allChosen) for (const candidate of selectable) next.delete(candidate.model);
    else
      for (const candidate of selectable) {
        if (next.size >= MAX_MODELS) break;
        next.add(candidate.model);
      }
    onSelected(next);
  };
  const addManual = () => {
    let model = manual.trim();
    if (!model) return;
    if (!model.includes("/")) model = `${provider === "custom" ? "openai" : provider}/${model}`;
    setAdded((previous) => [...previous.filter((candidate) => candidate.model !== model), { model, label: model.split("/").slice(1).join("/") || model, supports_vision: false }]);
    const next = new Set(selected);
    if (next.size < MAX_MODELS) next.add(model);
    onSelected(next);
    setManual("");
  };

  return (
    <div class="picker">
      <div class="picker-tools">
        <input type="search" value={query} placeholder={t.searchModels} aria-label={t.searchModels} onInput={(event) => setQuery(event.currentTarget.value)} />
        <label class="check-line">
          <input ref={selectAll} type="checkbox" checked={allChosen} disabled={!selectable.length} onChange={toggleVisible} />
          {t.selectVisible}
        </label>
      </div>
      <div class="picker-list" role="group" aria-label={t.modelsAvailable}>
        {visible.map((candidate) => {
          const done = configured(candidate.model);
          return (
            <label key={candidate.model} class={done ? "picker-option is-disabled" : "picker-option"}>
              <input
                type="checkbox"
                checked={selected.has(candidate.model)}
                disabled={done || (!selected.has(candidate.model) && selected.size >= MAX_MODELS)}
                onChange={() => toggle(candidate.model)}
              />
              <span class="picker-option-main">
                <span>{candidate.label}</span>
                <span class="picker-id">{candidate.model}</span>
              </span>
              {candidate.supports_vision && <span class="mini-pill">{t.visionTag}</span>}
              {done && <span class="mini-pill">{t.alreadyConfigured}</span>}
            </label>
          );
        })}
        {visible.length === 0 && <p class="picker-note">{t.noModelsFound}</p>}
      </div>
      <details class="disclosure">
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
        <p class="picker-count">{t.modelsSelected(selected.size, MAX_MODELS)}</p>
      </div>
    </div>
  );
}

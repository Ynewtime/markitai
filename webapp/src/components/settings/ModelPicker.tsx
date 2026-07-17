import { useEffect, useMemo, useRef, useState } from "react";
import type { LLMDeployment, ModelCandidate } from "../../api/types";
import type { Dict } from "../../i18n";

const MAX_SELECTION = 50;

function normalizeBase(value: string | null | undefined): string {
  return (value ?? "").trim().replace(/\/+$/, "").toLowerCase();
}

export function ModelPicker({
  t,
  provider,
  candidates,
  deployments,
  apiBase,
  routingGroup,
  weight,
  selected,
  onRoutingGroup,
  onWeight,
  onSelected,
}: {
  t: Dict;
  provider: string;
  candidates: ModelCandidate[];
  deployments: LLMDeployment[];
  apiBase: string;
  routingGroup: string;
  weight: number;
  selected: Set<string>;
  onRoutingGroup: (value: string) => void;
  onWeight: (value: number) => void;
  onSelected: (selected: Set<string>) => void;
}) {
  const [query, setQuery] = useState("");
  const [manual, setManual] = useState("");
  const [manualModels, setManualModels] = useState<ModelCandidate[]>([]);
  // Raw text of the weight field while it has focus, so clearing it does not
  // snap to 0 mid-edit; blur resolves back to the parsed weight.
  const [weightDraft, setWeightDraft] = useState<string | null>(null);
  const selectAllRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setQuery("");
    setManual("");
    setManualModels([]);
    setWeightDraft(null);
  }, [provider]);

  const allCandidates = useMemo(() => {
    const byModel = new Map<string, ModelCandidate>();
    for (const candidate of [...candidates, ...manualModels]) {
      byModel.set(candidate.model, candidate);
    }
    return [...byModel.values()];
  }, [candidates, manualModels]);

  const isConfigured = (model: string) =>
    deployments.some(
      (deployment) =>
        deployment.model === model &&
        deployment.routing_group === routingGroup.trim() &&
        normalizeBase(deployment.api_base) === normalizeBase(apiBase),
    );

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return allCandidates.filter(
      (candidate) =>
        needle === "" ||
        candidate.model.toLowerCase().includes(needle) ||
        candidate.label.toLowerCase().includes(needle),
    );
  }, [allCandidates, query]);
  const selectable = visible.filter((candidate) => !isConfigured(candidate.model));
  const selectedVisible = selectable.filter((candidate) => selected.has(candidate.model));
  const allVisibleSelected = selectable.length > 0 && selectedVisible.length === selectable.length;

  useEffect(() => {
    if (selectAllRef.current !== null) {
      selectAllRef.current.indeterminate =
        selectedVisible.length > 0 && !allVisibleSelected;
    }
  }, [allVisibleSelected, selectedVisible.length]);

  const toggle = (model: string) => {
    const next = new Set(selected);
    if (next.has(model)) next.delete(model);
    else if (next.size < MAX_SELECTION) next.add(model);
    onSelected(next);
  };

  const toggleVisible = () => {
    const next = new Set(selected);
    if (allVisibleSelected) {
      for (const candidate of selectable) next.delete(candidate.model);
    } else {
      for (const candidate of selectable) {
        if (next.size >= MAX_SELECTION) break;
        next.add(candidate.model);
      }
    }
    onSelected(next);
  };

  const addManual = () => {
    let model = manual.trim();
    if (model === "") return;
    if (!model.includes("/")) {
      const prefix = provider === "custom" ? "openai" : provider;
      model = `${prefix}/${model}`;
    }
    setManualModels((previous) => [
      ...previous.filter((candidate) => candidate.model !== model),
      { model, label: model.split("/", 2)[1] ?? model, supports_vision: false },
    ]);
    const next = new Set(selected);
    if (next.size < MAX_SELECTION) next.add(model);
    onSelected(next);
    setManual("");
  };

  return (
    <div className="model-picker">
      <div className="model-tools">
        <input
          type="search"
          value={query}
          placeholder={t.searchModels}
          aria-label={t.searchModels}
          onChange={(event) => setQuery(event.target.value)}
        />
        <label className="checkline mono">
          <input
            ref={selectAllRef}
            type="checkbox"
            checked={allVisibleSelected}
            disabled={selectable.length === 0}
            onChange={toggleVisible}
          />
          {t.selectVisible}
        </label>
      </div>

      <div className="model-list" role="group" aria-label={t.modelsAvailable}>
        {visible.map((candidate) => {
          const configured = isConfigured(candidate.model);
          return (
            <label className={configured ? "model-option disabled" : "model-option"} key={candidate.model}>
              <input
                type="checkbox"
                checked={selected.has(candidate.model)}
                disabled={configured || (!selected.has(candidate.model) && selected.size >= MAX_SELECTION)}
                onChange={() => toggle(candidate.model)}
              />
              <span className="model-option-main">
                <span className="mono">{candidate.label}</span>
                <span className="model-id mono">{candidate.model}</span>
              </span>
              {candidate.supports_vision && <span className="minibadge">Vision</span>}
              {configured && <span className="minibadge">{t.alreadyConfigured}</span>}
            </label>
          );
        })}
        {visible.length === 0 && <p className="picker-note mono">{t.noModelsFound}</p>}
      </div>

      <details className="manual-disclosure">
        <summary className="mono">{t.manualModelToggle}</summary>
        <div className="manual-model">
          <input
            type="text"
            value={manual}
            placeholder={provider === "azure" ? t.azureDeploymentPh : t.manualModelPh}
            onChange={(event) => setManual(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addManual();
              }
            }}
          />
          <button type="button" className="btn ghost sm" onClick={addManual}>
            {t.addManual}
          </button>
        </div>
      </details>

      <div className="picker-footer">
        <details className="picker-advanced">
          <summary className="mono">{t.advanced}</summary>
          <div className="advanced-grid">
            <label className="fld">
              <span className="lbl">{t.routingGroup}</span>
              <input type="text" value={routingGroup} onChange={(event) => onRoutingGroup(event.target.value)} />
            </label>
            <label className="fld">
              <span className="lbl">{t.weight}</span>
              <input
                type="number"
                min={0}
                value={weightDraft ?? weight}
                onChange={(event) => {
                  setWeightDraft(event.target.value);
                  onWeight(Math.max(0, Number(event.target.value) || 0));
                }}
                onBlur={() => setWeightDraft(null)}
              />
            </label>
          </div>
        </details>
        <p className="picker-count mono">{t.modelsSelected(selected.size, MAX_SELECTION)}</p>
      </div>
    </div>
  );
}

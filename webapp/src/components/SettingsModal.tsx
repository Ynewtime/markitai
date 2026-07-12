import { useCallback, useEffect, useRef, useState } from "react";
import {
  addLLMModel,
  deleteLLMModel,
  fetchDetectedProviders,
  fetchLLMSettings,
  testLLMSettings,
  updateLLMModel,
} from "../api/client";
import type {
  DetectedProvider,
  LLMModelCreate,
  LLMModelUpdate,
  LLMSettingsModel,
  LLMSettingsPayload,
  LLMSettingsUpdate,
} from "../api/types";
import type { Dict } from "../i18n";
import { XIcon } from "./icons";

/** CLI-voice result words — identical in both locales (like status words). */
const TEST_WORDS = { ok: "ok", fail: "failed" } as const;

/** Local CLI providers: no API key / base — fields are meaningless. */
const LOCAL_PREFIXES = ["claude-agent/", "copilot/", "chatgpt/"];

function isLocalModel(model: string): boolean {
  return LOCAL_PREFIXES.some((p) => model.startsWith(p));
}

const CONFIRM_RESET_MS = 4000;
const EXIT_MS = 120;

type RowTest = { state: "busy" } | { state: "ok" | "fail"; detail: string };
type FormMode = { kind: "add" } | { kind: "edit"; name: string };

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Centered LLM settings modal (headless, no library): multi-provider list
 * with per-row test/edit/delete, a collapsible add/edit form with local
 * provider awareness, and one-click add for detected CLI providers.
 * Focus is trapped, Esc closes, body scroll locks; the opener (gear)
 * regains focus after the 120ms exit fade. */
export function SettingsModal({
  t,
  onClose,
  onSaved,
  announce,
}: {
  t: Dict;
  onClose: () => void;
  onSaved: () => void;
  announce: (msg: string) => void;
}) {
  const [settings, setSettings] = useState<LLMSettingsPayload | null>(null);
  const [detected, setDetected] = useState<DetectedProvider[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  // ---- modal shell: enter/exit, focus trap, scroll lock
  const [closing, setClosing] = useState(false);
  const closingRef = useRef(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);

  const requestClose = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setClosing(true);
    closeTimerRef.current = window.setTimeout(onClose, EXIT_MS);
  }, [onClose]);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current !== null) window.clearTimeout(closeTimerRef.current);
    };
  }, []);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  useEffect(() => {
    cardRef.current?.focus();
  }, []);

  // Esc + Tab trap live on the document (capture) — focus can land on body
  // when the focused row is deleted, and the dialog must still own the keys.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        requestClose();
        return;
      }
      if (e.key !== "Tab") return;
      const root = cardRef.current;
      if (root === null) return;
      const focusables = Array.from(
        root.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => !el.hasAttribute("disabled") && el.offsetParent !== null);
      if (focusables.length === 0) {
        e.preventDefault();
        root.focus();
        return;
      }
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      if (!(active instanceof HTMLElement) || !root.contains(active)) {
        // focus escaped (e.g. the focused element was removed) — recapture
        e.preventDefault();
        first?.focus();
        return;
      }
      if (e.shiftKey) {
        if (active === first || active === root) {
          e.preventDefault();
          last?.focus();
        }
      } else if (active === last) {
        e.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [requestClose]);

  // ---- data
  const refresh = useCallback(async () => {
    const s = await fetchLLMSettings();
    setSettings(s);
    return s;
  }, []);

  useEffect(() => {
    let stale = false;
    fetchLLMSettings().then(
      (s) => {
        if (!stale) setSettings(s);
      },
      (e: unknown) => {
        if (!stale) setLoadError(errText(e));
      },
    );
    fetchDetectedProviders().then(
      (d) => {
        if (!stale) setDetected(d);
      },
      () => undefined, // endpoint unavailable — just no quick-add rows
    );
    return () => {
      stale = true;
    };
  }, []);

  // ---- collapsible add / edit form
  const [form, setForm] = useState<FormMode | null>(null);
  const [fName, setFName] = useState("");
  const [fModel, setFModel] = useState("");
  const [fKey, setFKey] = useState("");
  const [fBase, setFBase] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [formBusy, setFormBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const resetForm = () => {
    setForm(null);
    setFName("");
    setFModel("");
    setFKey("");
    setFBase("");
    setShowKey(false);
    setFormError(null);
  };

  const openAdd = () => {
    resetForm();
    setForm({ kind: "add" });
  };

  const openEdit = (m: LLMSettingsModel) => {
    setForm({ kind: "edit", name: m.model_name });
    setFName(m.model_name);
    setFModel(m.model);
    setFKey(""); // never prefill (and never post back) masked values
    setFBase(m.api_base ?? "");
    setShowKey(false);
    setFormError(null);
  };

  // Focus the first editable field when the form opens or switches target.
  const formKey = form === null ? null : form.kind === "add" ? "add" : `edit:${form.name}`;
  useEffect(() => {
    if (formKey === null) return;
    cardRef.current
      ?.querySelector<HTMLElement>(".mdl-form input:not(:disabled)")
      ?.focus();
  }, [formKey]);

  const localForm = isLocalModel(fModel.trim());

  const submitForm = async () => {
    if (form === null || formBusy) return;
    const name = fName.trim();
    const model = fModel.trim();
    if (name === "" || model === "") return;
    setFormBusy(true);
    setFormError(null);
    try {
      const local = isLocalModel(model);
      if (form.kind === "add") {
        const body: LLMModelCreate = { model_name: name, model };
        if (!local && fKey.trim() !== "") body.api_key = fKey.trim();
        if (!local && fBase.trim() !== "") body.api_base = fBase.trim();
        await addLLMModel(body);
      } else {
        // PUT: api_key omitted = keep the stored key (blank field means keep).
        const body: LLMModelUpdate = { model };
        if (!local) {
          if (fKey.trim() !== "") body.api_key = fKey.trim();
          body.api_base = fBase.trim() === "" ? null : fBase.trim();
        }
        await updateLLMModel(form.name, body);
      }
      await refresh();
      onSaved(); // capabilities hot-refresh — llm controls unlock
      announce(t.saved);
      resetForm();
    } catch (e) {
      setFormError(errText(e));
    }
    setFormBusy(false);
  };

  // ---- per-row test / delete
  const [rowTest, setRowTest] = useState<Record<string, RowTest>>({});
  const [confirmDel, setConfirmDel] = useState<string | null>(null);
  const confirmTimerRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    };
  }, []);

  const runRowTest = async (m: LLMSettingsModel) => {
    setRowTest((prev) => ({ ...prev, [m.model_name]: { state: "busy" } }));
    try {
      // Stored keys only exist masked — never sent; the backend resolves
      // credentials for saved entries (env/config) itself.
      const body: LLMSettingsUpdate = { model: m.model };
      if (m.api_base !== null) body.api_base = m.api_base;
      const res = await testLLMSettings(body);
      setRowTest((prev) => ({
        ...prev,
        [m.model_name]: { state: res.ok ? "ok" : "fail", detail: res.detail },
      }));
    } catch (e) {
      setRowTest((prev) => ({
        ...prev,
        [m.model_name]: { state: "fail", detail: errText(e) },
      }));
    }
  };

  const runDelete = async (name: string) => {
    if (confirmDel !== name) {
      setConfirmDel(name);
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
      confirmTimerRef.current = window.setTimeout(() => setConfirmDel(null), CONFIRM_RESET_MS);
      return;
    }
    if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    setConfirmDel(null);
    setListError(null);
    try {
      await deleteLLMModel(name);
      if (form !== null && form.kind === "edit" && form.name === name) resetForm();
      await refresh();
      onSaved();
    } catch (e) {
      setListError(errText(e));
    }
  };

  // ---- detected quick-add (one click fills and submits)
  const [detBusy, setDetBusy] = useState<string | null>(null);
  const quickAdd = async (c: DetectedProvider) => {
    if (detBusy !== null) return;
    setDetBusy(c.provider);
    setListError(null);
    try {
      await addLLMModel({ model_name: c.provider, model: c.model });
      await refresh();
      onSaved();
      announce(t.saved);
    } catch (e) {
      setListError(errText(e));
    }
    setDetBusy(null);
  };

  const models = settings?.models ?? [];
  const candidates = detected.filter(
    (c) =>
      !models.some((m) => m.model === c.model || m.model_name === c.provider),
  );
  const formValid = fName.trim() !== "" && fModel.trim() !== "";

  return (
    <div
      className={closing ? "mdl-veil out" : "mdl-veil"}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) requestClose();
      }}
    >
      <div
        ref={cardRef}
        className="mdl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="mdl-title"
        tabIndex={-1}
      >
        <div className="mdl-head">
          <h2 id="mdl-title" className="eyebrow">
            {t.settingsTitle}
          </h2>
          <button
            type="button"
            className="gearbtn"
            aria-label={t.close}
            title={t.close}
            onClick={requestClose}
          >
            <XIcon size={16} />
          </button>
        </div>

        <div className="mdl-body">
          {settings === null ? (
            <p className={loadError === null ? "mdl-dim mono" : "errline"}>
              {loadError ?? t.loading}
            </p>
          ) : (
            <>
              {models.length === 0 ? (
                <p className="mdl-dim mono">{t.setStatusNone}</p>
              ) : (
                <div className="provlist">
                  {models.map((m) => {
                    const local = isLocalModel(m.model);
                    const tst = rowTest[m.model_name];
                    return (
                      <div className="prov" key={m.model_name}>
                        <div className="prov-line">
                          <span className="prov-name mono">{m.model_name}</span>
                          <span
                            className="prov-model mono"
                            title={
                              m.api_base !== null
                                ? `${m.model} · ${m.api_base}`
                                : m.model
                            }
                          >
                            {m.model}
                          </span>
                          {local ? (
                            <span className="minibadge">local</span>
                          ) : (
                            m.api_key_masked !== null && (
                              <span className="prov-key mono">{m.api_key_masked}</span>
                            )
                          )}
                          <span className="prov-acts">
                            <button
                              type="button"
                              className="rowact"
                              disabled={tst?.state === "busy"}
                              aria-label={`${t.test} ${m.model_name}`}
                              onClick={() => void runRowTest(m)}
                            >
                              {t.test}
                            </button>
                            <button
                              type="button"
                              className="rowact"
                              aria-label={`${t.edit} ${m.model_name}`}
                              onClick={() => openEdit(m)}
                            >
                              {t.edit}
                            </button>
                            <button
                              type="button"
                              className={confirmDel === m.model_name ? "rowact warn" : "rowact"}
                              aria-label={`${t.histDelete} ${m.model_name}`}
                              onClick={() => void runDelete(m.model_name)}
                            >
                              {confirmDel === m.model_name ? t.histConfirm : t.histDelete}
                            </button>
                          </span>
                        </div>
                        {tst !== undefined && (
                          <div
                            className={
                              tst.state === "fail" ? "prov-detail mono fail" : "prov-detail mono"
                            }
                            role="status"
                          >
                            {tst.state === "busy"
                              ? t.testing
                              : `${tst.state === "ok" ? TEST_WORDS.ok : TEST_WORDS.fail} · ${tst.detail}`}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {candidates.length > 0 && (
                <div className="detlist">
                  {candidates.map((c) => (
                    <div className="det-row mono" key={c.provider}>
                      <span>{t.detDetected(c.label)}</span>
                      <span aria-hidden="true">·</span>
                      <button
                        type="button"
                        className="rowact"
                        disabled={detBusy !== null}
                        aria-label={`${t.detAdd} ${c.label}`}
                        onClick={() => void quickAdd(c)}
                      >
                        {detBusy === c.provider ? t.saving : t.detAdd}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {listError !== null && <p className="errline mdl-err">{listError}</p>}

              {form === null ? (
                <div>
                  <button type="button" className="btn ghost sm" onClick={openAdd}>
                    {t.addModel}
                  </button>
                </div>
              ) : (
                <form
                  className="mdl-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void submitForm();
                  }}
                >
                  <label className="fld">
                    <span className="lbl">{t.setModelName}</span>
                    <input
                      type="text"
                      value={fName}
                      placeholder={t.setModelNamePh}
                      spellCheck={false}
                      autoComplete="off"
                      disabled={form.kind === "edit"}
                      onChange={(e) => setFName(e.target.value)}
                    />
                  </label>
                  <label className="fld">
                    <span className="lbl">{t.setModel}</span>
                    <input
                      type="text"
                      value={fModel}
                      placeholder={t.setModelPh}
                      spellCheck={false}
                      autoComplete="off"
                      onChange={(e) => setFModel(e.target.value)}
                    />
                  </label>
                  <label className="fld">
                    <span className="lbl">{t.setApiKey}</span>
                    <span className="keyrow">
                      <input
                        type={showKey ? "text" : "password"}
                        value={fKey}
                        placeholder={localForm ? "" : t.setKeyPh}
                        spellCheck={false}
                        autoComplete="off"
                        disabled={localForm}
                        onChange={(e) => setFKey(e.target.value)}
                      />
                      <button
                        type="button"
                        className="btn ghost sm"
                        aria-pressed={showKey}
                        disabled={localForm}
                        onClick={() => setShowKey((v) => !v)}
                      >
                        {showKey ? t.hide : t.show}
                      </button>
                    </span>
                    <span className="fld-hint">
                      {localForm
                        ? t.localNote
                        : form.kind === "edit"
                          ? t.keepKeyHint
                          : t.setKeyHint}
                    </span>
                  </label>
                  <label className="fld">
                    <span className="lbl">{t.setApiBase}</span>
                    <input
                      type="text"
                      value={fBase}
                      placeholder={localForm ? "" : t.setBasePh}
                      spellCheck={false}
                      autoComplete="off"
                      disabled={localForm}
                      onChange={(e) => setFBase(e.target.value)}
                    />
                  </label>

                  <div className="set-actions">
                    <button
                      type="submit"
                      className="btn primary"
                      disabled={formBusy || !formValid}
                    >
                      {formBusy ? t.saving : t.save}
                    </button>
                    <button type="button" className="btn ghost" onClick={resetForm}>
                      {t.cancel}
                    </button>
                    {formError !== null && (
                      <span className="set-note mono fail" role="status">
                        {formError}
                      </span>
                    )}
                  </div>
                </form>
              )}

              <p className="mdl-src mono">
                {t.setSourceLbl} {settings.source} · {settings.config_path}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

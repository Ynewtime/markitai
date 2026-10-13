// Edit config (admin only, v1.13): the configuration file the Models page names
// as its source, as text. The server sends it with every secret masked (`sk…7890`);
// a masked value left where it is keeps the stored secret. The text is checked
// as JSON while it is typed; Review shows the changes as a diff, and Save sends
// them with the revision they were made against (a file changed meanwhile is
// refused, as every settings write is). The server refuses a configuration it
// cannot use and says why; the model list applies at once, the rest at restart.
// The textarea knows only LF, so the page edits the text with LF line breaks and
// sends it back with the file's own (CRLF on Windows), keeping its BOM.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ApiError, fetchConfigDocument, saveConfigDocument } from "../api/client.ts";
import type { ConfigDocument } from "../api/types.ts";
import type { Dict } from "../i18n/index.ts";
import { compareLines, foldRows } from "../lib/diff.ts";
import { jsonProblem } from "../lib/config-text.ts";
import { editorText, fileText, textLayout } from "../lib/line-endings.ts";
import { Icon } from "./icons.tsx";
import { DiffRows } from "./markdown-view.tsx";
import { StateBlock } from "./state-block.tsx";

type Stage = "edit" | "review";

export function ConfigSettings({
  t,
  announce,
  describe,
  onSaved,
}: {
  t: Dict;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
  onSaved: () => void;
}) {
  const [doc, setDoc] = useState<ConfigDocument | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [text, setText] = useState("");
  const [stage, setStage] = useState<Stage>("edit");
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<{ text: string; detail: string; stale: boolean } | null>(null);
  const [saved, setSaved] = useState<{ restart: boolean } | null>(null);
  const [opened, setOpened] = useState<Set<number>>(() => new Set());
  const editor = useRef<HTMLTextAreaElement>(null);
  const reviewHeading = useRef<HTMLHeadingElement>(null);

  const load = () => {
    setLoadError(null);
    fetchConfigDocument().then(
      (value) => {
        setDoc(value);
        setText(editorText(value.text ?? ""));
        setStage("edit");
        setFailure(null);
      },
      (error: unknown) => setLoadError(error),
    );
  };
  useEffect(load, []);

  // The file's text as the server sent it; the editor's text has LF breaks only.
  const raw = doc?.text ?? "";
  const original = useMemo(() => editorText(raw), [raw]);
  const layout = useMemo(() => textLayout(raw), [raw]);
  const changed = doc !== null && text !== original;
  // Checked as it is typed, a moment after the last key.
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    const timer = setTimeout(() => setProblem(jsonProblem(text, t)), 250);
    return () => clearTimeout(timer);
  }, [text, t]);
  const diff = useMemo(() => (stage === "review" ? compareLines(original, text) : null), [stage, original, text]);
  useEffect(() => {
    if (stage === "review") requestAnimationFrame(() => reviewHeading.current?.focus());
  }, [stage]);

  const save = async () => {
    if (!doc || saving) return;
    setSaving(true);
    setFailure(null);
    try {
      const next = await saveConfigDocument(doc.revision, fileText(raw, text));
      setDoc(next);
      setText(editorText(next.text ?? ""));
      setStage("edit");
      setOpened(new Set());
      setSaved({ restart: next.restart_required === true });
      announce(t.configSaved);
      onSaved();
    } catch (error) {
      const stale = error instanceof ApiError && error.status === 409 && (error.reason === "stale_revision" || error.reason === "config_changed");
      const described = describe(error);
      setFailure({ ...described, stale });
    } finally {
      setSaving(false);
    }
  };

  const head = (
    <div class="page-head config-head">
      <div>
        <h1 id="config-title">{t.navConfig}</h1>
        {doc && (
          <p class="config-path">
            <code>{doc.path}</code>
            {!doc.exists && <span class="config-new"> · {t.configAbsent}</span>}
          </p>
        )}
      </div>
    </div>
  );

  if (loadError) {
    return (
      <section class="settings-page" aria-labelledby="config-title">
        {head}
        <StateBlock icon="WarningCircle" title={t.configLoadFailed} text={describe(loadError).text} secondary={{ label: t.retryLoad, run: load }} tone="error" />
      </section>
    );
  }
  if (!doc) {
    return (
      <section class="settings-page" aria-labelledby="config-title">
        {head}
        <p class="pane-note">{t.loading}</p>
      </section>
    );
  }
  if (doc.text === null) {
    return (
      <section class="settings-page" aria-labelledby="config-title">
        {head}
        <StateBlock icon="WarningCircle" title={t.configUnreadableTitle} text={t.configUnreadableText(doc.problem ?? "")} secondary={{ label: t.retryLoad, run: load }} tone="error" />
      </section>
    );
  }

  return (
    <section class="settings-page config-page" aria-labelledby="config-title">
      {head}
      <div class="settings-body">
        {doc.read_only && <p class="config-note is-warn">{t.configReadOnly}</p>}
        {doc.masked > 0 && <p class="config-note">{t.configMasked(doc.masked)}</p>}
        {(layout.kind === "crlf" || layout.kind === "cr") && <p class="config-note">{t.configEndings(layout.kind === "crlf" ? "CRLF" : "CR")}</p>}
        {layout.kind === "mixed" && <p class="config-note">{t.configEndingsMixed(layout.counts, layout.newline === "\r\n" ? "CRLF" : layout.newline === "\n" ? "LF" : "CR")}</p>}
        {layout.bom && <p class="config-note">{t.configBom}</p>}
        {saved && !changed && (
          <p class="config-note is-done" role="status">
            <Icon name="Check" size={14} />
            <span>{saved.restart ? t.configSavedRestart : t.configSaved}</span>
          </p>
        )}
        {stage === "edit" ? (
          <div class="config-editor">
            <label class="sr-only" for="config-text">
              {t.configEditorLabel}
            </label>
            <textarea
              ref={editor}
              id="config-text"
              value={text}
              spellcheck={false}
              autoComplete="off"
              autoCapitalize="off"
              wrap="off"
              readOnly={doc.read_only}
              aria-invalid={problem !== null || undefined}
              aria-describedby="config-check"
              onInput={(event) => {
                setText(event.currentTarget.value);
                setSaved(null);
                setFailure(null);
              }}
            />
            <p id="config-check" class={problem ? "config-check is-error" : "config-check"} aria-live="polite">
              <Icon name={problem ? "WarningCircle" : "Check"} size={14} />
              <span>{problem ?? (changed ? t.configValidChanged : t.configValid)}</span>
            </p>
          </div>
        ) : (
          <div class="config-review">
            <h2 ref={reviewHeading} class="group-title" tabIndex={-1}>
              {t.configReviewTitle}
            </h2>
            {diff?.kind === "rows" ? (
              <div class="diff config-diff" aria-label={t.configReviewTitle}>
                <p class="doc-meta diff-meta">
                  config.json · +{diff.added} -{diff.removed}
                </p>
                {foldRows(diff.rows).map((segment, index) =>
                  segment.kind === "rows" || opened.has(index) ? (
                    <DiffRows key={index} rows={segment.rows} marks={{ add: t.diffAdded, del: t.diffRemoved }} />
                  ) : (
                    <button key={index} type="button" class="diff-fold" onClick={() => setOpened((previous) => new Set(previous).add(index))}>
                      <Icon name="CaretDown" size={13} />
                      {t.configFold(segment.rows.length)}
                    </button>
                  ),
                )}
              </div>
            ) : (
              <p class="pane-note">{t.diffTooLarge}</p>
            )}
          </div>
        )}
        {failure && (
          <div class="config-failure" role="alert">
            <p class="line-error">{failure.stale ? t.configStale : failure.text}</p>
            {failure.detail && !failure.stale && <p class="config-detail">{failure.detail}</p>}
            {failure.stale && (
              <button type="button" class="btn btn-sm" onClick={load}>
                {t.configReloadFile}
              </button>
            )}
          </div>
        )}
        {!doc.read_only && (
          <div class="config-actions">
            {stage === "edit" ? (
              <>
                <button type="button" class="btn" disabled={!changed} onClick={() => (setText(original), setFailure(null))}>
                  {t.configDiscard}
                </button>
                <button type="button" class="btn btn-primary config-review-btn" disabled={!changed || problem !== null} onClick={() => setStage("review")}>
                  {t.configReview}
                </button>
              </>
            ) : (
              <>
                <button type="button" class="btn" onClick={() => (setStage("edit"), requestAnimationFrame(() => editor.current?.focus()))}>
                  {t.configBackToEdit}
                </button>
                <button type="button" class="btn btn-primary" disabled={saving} aria-busy={saving || undefined} onClick={() => void save()}>
                  {saving ? t.saving : t.configSave}
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

// Convert (`/`): one headline, one sentence and the composer with its options,
// then the latest submissions, each one click from its details.
import type { ComponentChildren } from "preact";
import type { Dict, Locale } from "../i18n/index.ts";
import { counts, isGroup, jobKey, submissionLabel, type LabelWords, type Submission } from "../lib/conversions.ts";
import { fmtClock } from "../lib/format.ts";
import { NameText } from "./file-name.tsx";
import { glyphOf, StatusGlyph, statusWord } from "./item-bits.tsx";
import { Icon } from "./icons.tsx";
import type { PanelState } from "./conversions-page.tsx";
import { StateBlock } from "./state-block.tsx";

export function ConvertPage({
  t,
  locale,
  composer,
  recent,
  blocked,
  labelWords,
  onOpen,
}: {
  t: Dict;
  locale: Locale;
  composer: ComponentChildren;
  recent: Submission[];
  /** Why the recent conversions cannot be shown (no token, no server). */
  blocked: PanelState | null;
  labelWords: LabelWords;
  onOpen: (key: string) => void;
}) {
  return (
    <div class="page convert">
      <header class="hero">
        <h1>{t.heroTitle}</h1>
        <p>{t.heroSub}</p>
      </header>
      <div class="convert-composer">{composer}</div>
      {blocked && recent.length === 0 && (
        <div class="recent">
          <StateBlock {...blocked} />
        </div>
      )}
      {recent.length > 0 && (
        <section class="recent" aria-labelledby="recent-title">
          <div class="recent-head">
            <h2 id="recent-title">{t.recentTitle}</h2>
          </div>
          <ul class="recent-list">
            {recent.map((submission) => {
              const group = isGroup(submission);
              const first = submission.items[0];
              const tally = counts(submission.items);
              const key = group ? jobKey(submission.job.jobId) : (first?.key ?? jobKey(submission.job.jobId));
              const glyph = group ? (tally.running || tally.queued ? "running" : tally.failed ? "failed" : "done") : first ? glyphOf(first) : "done";
              return (
                <li key={key}>
                  <button type="button" class="recent-row" onClick={() => onOpen(key)}>
                    <StatusGlyph glyph={glyph} label={statusWord(t, glyph)} />
                    <span class="recent-name">
                      {group || !first ? <span class="nm">{submissionLabel(submission.job, submission.items, labelWords)}</span> : <NameText name={first.name} kind={first.kind} />}
                    </span>
                    <span class="recent-side">
                      {group && tally.failed > 0 ? t.failedCount(tally.failed) : fmtClock(submission.job.finishedAt ?? submission.job.createdAt, locale)}
                    </span>
                    <Icon name="CaretRight" size={13} />
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}

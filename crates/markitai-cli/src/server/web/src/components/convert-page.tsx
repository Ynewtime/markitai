// Convert (`/`): one headline, one sentence and the composer with its options,
// then the latest submissions, each one click from its details.
import type { ComponentChildren } from "preact";
import type { Dict, Locale } from "../i18n/index.ts";
import { counts, isGroup, jobKey, submissionLabel, type LabelWords, type Submission } from "../lib/conversions.ts";
import { fmtClock } from "../lib/format.ts";
import { glyphOf, NameText, StatusGlyph, statusWord } from "./item-bits.tsx";
import { Icon } from "./icons.tsx";

export function ConvertPage({
  t,
  locale,
  composer,
  notes,
  recent,
  labelWords,
  onOpen,
  onAll,
}: {
  t: Dict;
  locale: Locale;
  composer: ComponentChildren;
  notes: ComponentChildren;
  recent: Submission[];
  labelWords: LabelWords;
  onOpen: (key: string) => void;
  onAll: () => void;
}) {
  return (
    <div class="page convert">
      <header class="hero">
        <h1>{t.heroTitle}</h1>
        <p>{t.heroSub}</p>
      </header>
      {notes}
      <div class="convert-composer">{composer}</div>
      {recent.length > 0 && (
        <section class="recent" aria-labelledby="recent-title">
          <div class="recent-head">
            <h2 id="recent-title">{t.recentTitle}</h2>
            <a
              href="/jobs"
              class="text-link"
              onClick={(event) => {
                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
                event.preventDefault();
                onAll();
              }}
            >
              {t.allConversions}
            </a>
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

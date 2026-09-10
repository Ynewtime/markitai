import type { SessionStats } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { fmtCost } from "../lib/format";

/** Left block of the job header: eyebrow (the page heading in workspace —
 * an h2, visually unchanged) + session-level counters. Skips are counted
 * apart from real completions. */
export function JobStats({
  t,
  running,
  stats,
}: {
  t: Dict;
  running: boolean;
  stats: SessionStats;
}) {
  return (
    <div>
      <h2 className="eyebrow">{t.conversions}</h2>
      {stats.total > 0 && (
        <div className="stats">
          <strong>
            {t.currentSession} · {running ? `${t.statusRunning} · ` : ""}
            {stats.done}/{stats.total} {t.statusDone}
          </strong>
          {stats.skipped > 0 && (
            <>
              {" · "}
              {stats.skipped} {t.statusSkipped}
            </>
          )}
          {stats.failed > 0 && (
            <>
              {" · "}
              {stats.failed} {t.statusFailed}
            </>
          )}
          {stats.hasCost && <> · {fmtCost(stats.costTotal)}</>}
        </div>
      )}
    </div>
  );
}

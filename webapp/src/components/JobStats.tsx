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
      <h2 className="eyebrow">{running ? t.converting : t.done}</h2>
      <div className="stats">
        <strong>
          {stats.done}/{stats.total} {t.statDone}
        </strong>
        {stats.skipped > 0 && (
          <>
            {" · "}
            {stats.skipped} {t.statSkipped}
          </>
        )}
        {stats.failed > 0 && (
          <>
            {" · "}
            {stats.failed} {t.statFailed}
          </>
        )}
        {stats.hasCost && <> · {fmtCost(stats.costTotal)}</>}
      </div>
    </div>
  );
}

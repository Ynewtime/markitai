// The conversions list: every saved and running submission with all of its
// items, newest activity first. A submission of one item is one row; a
// submission of several is a header (what was submitted, preset, time) with
// its items under it. Pure functions only.
import type { JobOptions, JobStatus, Pricing } from "../api/types.ts";
import { readable, timestampMs } from "./format.ts";
import { rowWarnings } from "./notices.ts";
import { isActive, type SessionItem } from "./session.ts";

export interface ConvJob {
  jobId: string;
  status: JobStatus;
  createdAt: string | null;
  finishedAt: string | null;
  options: JobOptions | null;
  /** What the submission was called by the backend or this tab; null when it named nothing. */
  label: string | null;
  origin: "web" | "cli";
  persistenceError: string | null;
  /** Wall time reported by history. */
  durationMs: number | null;
  costUsd: number | null;
  pricing: Pricing | null;
}

export type Filter = "all" | "failed" | "notices";
export const FILTERS: Filter[] = ["all", "failed", "notices"];

export interface Submission {
  job: ConvJob;
  /** Every item of the job, in input order. */
  items: SessionItem[];
  /** The items the filter keeps. */
  shown: SessionItem[];
}

export const isFailed = (item: SessionItem): boolean => item.status === "error" && item.skipReason !== "user_stopped";
export const isStopped = (item: SessionItem): boolean => item.skipReason === "user_stopped";

/** Something to read about besides the result: warnings, a kept failed rerun, or a skip. */
export const hasNotices = (item: SessionItem): boolean =>
  rowWarnings(item.warnings).length > 0 || item.rerunFailure !== null || (item.skipped && !isStopped(item));

export function itemMatches(item: SessionItem, filter: Filter, needle: string): boolean {
  if (filter === "failed" && !isFailed(item)) return false;
  if (filter === "notices" && !hasNotices(item)) return false;
  if (!needle) return true;
  return item.name.toLowerCase().includes(needle) || readable(item.name).toLowerCase().includes(needle);
}

function activity(job: ConvJob, items: SessionItem[]): number {
  if (job.status === "running" || items.some(isActive)) return Number.POSITIVE_INFINITY;
  let latest = timestampMs(job.finishedAt);
  for (const item of items) {
    const at = timestampMs(item.finishedAt);
    if (at !== null && (latest === null || at > latest)) latest = at;
  }
  return latest ?? timestampMs(job.createdAt) ?? 0;
}

/** Submissions newest activity first (running first); empty ones dropped when filtered. */
export function submissions(jobs: Record<string, ConvJob>, items: SessionItem[], filter: Filter = "all", query = ""): Submission[] {
  const byJob = new Map<string, SessionItem[]>();
  for (const item of items) {
    const list = byJob.get(item.jobId);
    if (list) list.push(item);
    else byJob.set(item.jobId, [item]);
  }
  const needle = query.trim().toLowerCase();
  const result: (Submission & { at: number; created: number })[] = [];
  for (const [jobId, list] of byJob) {
    const job = jobs[jobId] ?? placeholderJob(jobId);
    const shown = list.filter((item) => itemMatches(item, filter, needle));
    if (!shown.length && (filter !== "all" || needle)) continue;
    result.push({ job, items: list, shown, at: activity(job, list), created: timestampMs(job.createdAt) ?? Number.POSITIVE_INFINITY });
  }
  result.sort((left, right) => right.at - left.at || right.created - left.created || left.job.jobId.localeCompare(right.job.jobId));
  return result.map(({ job, items: all, shown }) => ({ job, items: all, shown }));
}

export function placeholderJob(jobId: string): ConvJob {
  return { jobId, status: "running", createdAt: null, finishedAt: null, options: null, label: null, origin: "web", persistenceError: null, durationMs: null, costUsd: null, pricing: null };
}

/** A submission of several items is shown as a group with a header. */
export const isGroup = (submission: Submission): boolean => submission.items.length > 1;

export const jobKey = (jobId: string): string => `job:${jobId}`;
export const isJobKey = (key: string): boolean => key.startsWith("job:");

/** The keys a keyboard walks through, top to bottom: headers and the items of open groups. */
export function navigationOrder(list: Submission[], collapsed: ReadonlySet<string>): string[] {
  const keys: string[] = [];
  for (const submission of list) {
    if (!isGroup(submission)) {
      for (const item of submission.shown) keys.push(item.key);
      continue;
    }
    keys.push(jobKey(submission.job.jobId));
    if (!collapsed.has(submission.job.jobId)) for (const item of submission.shown) keys.push(item.key);
  }
  return keys;
}

/** Item keys only, in list order: what Preview's previous/next walks. */
export function itemOrder(list: Submission[]): SessionItem[] {
  return list.flatMap((submission) => submission.shown);
}

export interface Counts {
  total: number;
  done: number;
  failed: number;
  stopped: number;
  skipped: number;
  running: number;
  queued: number;
  notices: number;
}

export function counts(items: SessionItem[]): Counts {
  const result: Counts = { total: items.length, done: 0, failed: 0, stopped: 0, skipped: 0, running: 0, queued: 0, notices: 0 };
  for (const item of items) {
    if (item.status === "running") result.running++;
    else if (item.status === "queued") result.queued++;
    else if (isStopped(item)) result.stopped++;
    else if (item.status === "error") result.failed++;
    else if (item.skipped) result.skipped++;
    else result.done++;
    if (hasNotices(item)) result.notices++;
  }
  return result;
}

/** What a submission without a name is called, from what it holds. */
export interface LabelWords {
  files: (n: number) => string;
  pages: (n: number) => string;
  filesAndPages: (files: number, pages: number) => string;
}

export function submissionLabel(job: ConvJob, items: SessionItem[], words: LabelWords): string {
  if (job.label?.trim()) return job.label.trim();
  const pages = items.filter((item) => item.kind === "url").length;
  const files = items.length - pages;
  if (pages && files) return words.filesAndPages(files, pages);
  return pages ? words.pages(pages) : words.files(files);
}

/** The preset a submission ran with, for its header; null when it named none. */
export function presetOf(options: JobOptions | null | undefined): string | null {
  return options?.preset ?? null;
}

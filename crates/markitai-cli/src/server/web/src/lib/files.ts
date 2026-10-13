// Files chosen or dropped for a job. A folder skips its hidden and system
// content (dot names, Thumbs.db, desktop.ini) but never the folder itself or
// files picked one by one; walking a dropped tree is bounded.

export const MAX_FILE_BYTES = 100 * 1024 * 1024;

/** A batch waiting for Convert: a folder, several files, or a `.urls` list.
 * One file converts as soon as it is chosen; a batch is worth a look first. */
export interface StagedSubmission {
  files: File[];
  /** What the batch is called: the folder name, the list name, or its files. */
  label: string;
  kind: "folder" | "files" | "urls";
}

const SYSTEM_FILE = /^(?:thumbs\.db|desktop\.ini)$/i;
export const isHiddenName = (name: string): boolean => name.startsWith(".") || SYSTEM_FILE.test(name);

/** The folder-relative path a File from a folder input carries (`dir/sub/a.pdf`). */
export function relativePath(file: File): string {
  return (file as File & { markitaiPath?: string }).markitaiPath || file.webkitRelativePath || file.name;
}

/** Split a folder input's files into those to keep and a count of hidden ones. */
export function selectFolderFiles(values: File[]): { files: File[]; hidden: number } {
  const files: File[] = [];
  let hidden = 0;
  for (const file of values) {
    const path = relativePath(file);
    const below = path.split("/").slice(1);
    if (path.includes("/") && below.some(isHiddenName)) hidden++;
    else files.push(file);
  }
  return { files, hidden };
}

export interface Walked {
  files: File[];
  hidden: number;
  truncated: boolean;
  unreadable: number;
  /** Whether the gesture reached into a directory, which makes it a batch. */
  folder: boolean;
}

interface EntryLike {
  name: string;
  isFile: boolean;
  isDirectory: boolean;
  file?(success: (file: File) => void, failure: (error: unknown) => void): void;
  createReader?(): { readEntries(success: (batch: EntryLike[]) => void, failure: (error: unknown) => void): void };
}

function readAll(entry: EntryLike): Promise<EntryLike[]> {
  const reader = entry.createReader?.();
  if (!reader) return Promise.resolve([]);
  return new Promise((resolve, reject) => {
    const all: EntryLike[] = [];
    // Readers answer in batches (Chromium: 100) until an empty one.
    const next = () =>
      reader.readEntries((batch) => {
        if (!batch.length) resolve(all);
        else {
          all.push(...batch);
          next();
        }
      }, reject);
    next();
  });
}

function entryFile(entry: EntryLike): Promise<File> {
  return new Promise((resolve, reject) => {
    if (!entry.file) reject(new Error("not a file"));
    else entry.file(resolve, reject);
  });
}

/** Walk dropped entries (read during the drop event itself). Stops at `limit`
 * files, and at structural bounds so an enormous tree cannot hold the page. */
export async function walkEntries(
  entries: EntryLike[],
  { limit = 1000, visits = 50_000, depth = 64 }: { limit?: number; visits?: number; depth?: number } = {},
): Promise<Walked> {
  const files: File[] = [];
  let hidden = 0;
  let truncated = false;
  let seen = 0;
  let unreadable = 0;
  let folder = false;
  async function visit(entry: EntryLike, path: string, level: number): Promise<void> {
    if (truncated) return;
    if (++seen > visits || level > depth) {
      truncated = true;
      return;
    }
    if (level > 0 && isHiddenName(entry.name)) {
      hidden++;
      return;
    }
    if (entry.isFile) {
      if (files.length >= limit) {
        truncated = true;
        return;
      }
      try {
        const file = await entryFile(entry);
        Object.defineProperty(file, "markitaiPath", { value: path, configurable: true });
        files.push(file);
      } catch {
        unreadable++;
      }
    } else if (entry.isDirectory) {
      folder = true;
      let children: EntryLike[];
      try {
        children = await readAll(entry);
      } catch {
        unreadable++;
        return;
      }
      children.sort((left, right) => left.name.localeCompare(right.name));
      for (const child of children) await visit(child, `${path}/${child.name}`, level + 1);
    }
  }
  for (const entry of entries) await visit(entry, entry.name, 0);
  return { files, hidden, truncated, unreadable, folder };
}

export const oversized = (files: File[]): File[] => files.filter((file) => file.size > MAX_FILE_BYTES);

/** What a chosen set of files is called while it waits for Convert: a folder, a
 * URL list, or several files (null for a single file, which needs no name). */
export function stagedBatch(files: File[], folder: boolean): StagedSubmission | null {
  // A folder is named for the folder, even when a list sits inside it: that list
  // is one of its files, not the submission.
  if (folder) {
    const path = files[0] ? relativePath(files[0]) : "";
    const name = path.includes("/") ? path.slice(0, path.indexOf("/")) : "";
    return { files, label: name, kind: "folder" };
  }
  const list = files.find((file) => /\.urls$/i.test(relativePath(file)));
  if (list) {
    const path = relativePath(list);
    return { files, label: path.slice(path.lastIndexOf("/") + 1), kind: "urls" };
  }
  if (files.length < 2) return null;
  return { files, label: "", kind: "files" };
}

const sameFile = (left: File, right: File) =>
  relativePath(left) === relativePath(right) && left.size === right.size && left.lastModified === right.lastModified;

/** Files chosen again join what already waits for Convert (the same file once), up to
 * `limit`; the batch keeps its name only while it is one folder or one list. */
export function addStaged(current: StagedSubmission | null, chosen: File[], folder: boolean, limit: number): { staged: StagedSubmission | null; dropped: number } {
  const files = current ? [...current.files] : [];
  let dropped = 0;
  for (const file of chosen) {
    if (files.some((kept) => sameFile(kept, file))) continue;
    if (files.length >= limit) dropped++;
    else files.push(file);
  }
  if (!files.length) return { staged: null, dropped };
  if (current && files.length === current.files.length) return { staged: current, dropped };
  const named = stagedBatch(chosen, folder);
  const kind = current && current.files.length ? (named && named.kind === current.kind && named.label === current.label ? current.kind : "files") : (named?.kind ?? "files");
  const label = kind === "files" ? "" : current && current.files.length ? current.label : (named?.label ?? "");
  return { staged: { files, label, kind }, dropped };
}

/** The batch without one of its files; null once nothing is left. */
export function removeStaged(current: StagedSubmission, file: File): StagedSubmission | null {
  const files = current.files.filter((kept) => kept !== file);
  if (!files.length) return null;
  // A URL list taken out leaves plain files.
  const kind = current.kind === "urls" && !files.some((kept) => /\.urls$/i.test(relativePath(kept))) ? "files" : current.kind;
  return { files, label: kind === "files" ? "" : current.label, kind };
}

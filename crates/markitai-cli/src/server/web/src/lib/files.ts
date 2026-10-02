// Files chosen or dropped for a job. A folder skips its hidden and system
// content (dot names, Thumbs.db, desktop.ini) but never the folder itself or
// files picked one by one; walking a dropped tree is bounded.

export const MAX_FILE_BYTES = 100 * 1024 * 1024;

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
  return { files, hidden, truncated, unreadable };
}

export const oversized = (files: File[]): File[] => files.filter((file) => file.size > MAX_FILE_BYTES);

/** Folder drops: recursive collection over webkitGetAsEntry() trees.
 * Hidden files/dirs (dot-prefixed) and .DS_Store-class junk are skipped;
 * results are plain File handles whose names are already basenames (the
 * backend dedupes same-named outputs). Depth is capped to stop runaway
 * nesting. */

const MAX_DEPTH = 8;

function isJunkName(name: string): boolean {
  return name.startsWith(".") || name === "Thumbs.db" || name === "desktop.ini";
}

function readAllEntries(dir: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  // readEntries returns batches (Chromium caps one call at 100) until empty.
  const reader = dir.createReader();
  return new Promise((resolve) => {
    const all: FileSystemEntry[] = [];
    const step = () => {
      reader.readEntries(
        (batch) => {
          if (batch.length === 0) resolve(all);
          else {
            all.push(...batch);
            step();
          }
        },
        () => resolve(all), // unreadable dir: keep what we have
      );
    };
    step();
  });
}

function entryFile(entry: FileSystemFileEntry): Promise<File | null> {
  return new Promise((resolve) => {
    entry.file(resolve, () => resolve(null));
  });
}

/** Flatten a dropped entry list (files and directories) into files. The
 * entries must have been read synchronously inside the drop handler. */
export async function collectFromEntries(
  entries: (FileSystemEntry | null)[],
): Promise<File[]> {
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry, depth: number): Promise<void> => {
    if (isJunkName(entry.name)) return;
    if (entry.isFile) {
      const f = await entryFile(entry as FileSystemFileEntry);
      if (f !== null) out.push(f);
    } else if (entry.isDirectory && depth < MAX_DEPTH) {
      const children = await readAllEntries(entry as FileSystemDirectoryEntry);
      for (const child of children) await walk(child, depth + 1);
    }
  };
  for (const entry of entries) {
    if (entry !== null) await walk(entry, 0);
  }
  return out;
}

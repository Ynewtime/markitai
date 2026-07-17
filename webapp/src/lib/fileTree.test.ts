import { describe, expect, it } from "vitest";

import { collectFromEntries } from "./fileTree";

function fileEntry(name: string): FileSystemEntry {
  return {
    isFile: true,
    isDirectory: false,
    name,
    file: (ok: (f: File) => void) => ok(new File(["x"], name)),
  } as unknown as FileSystemEntry;
}

/** One batch per readEntries call, mirroring Chromium's batching; an empty
 * batch ends the listing. `failAt` makes that call report a read error. */
function dirEntry(
  name: string,
  batches: FileSystemEntry[][],
  failAt: number | null = null,
): FileSystemEntry {
  return {
    isFile: false,
    isDirectory: true,
    name,
    createReader: () => {
      let call = 0;
      return {
        readEntries: (ok: (b: FileSystemEntry[]) => void, err: () => void) => {
          // Advance before calling back: the walker re-enters synchronously.
          const index = call;
          call += 1;
          if (index === failAt) err();
          else ok(batches[index] ?? []);
        },
      };
    },
  } as unknown as FileSystemEntry;
}

describe("collectFromEntries", () => {
  it("skips dotfiles and OS junk, including junk directories", async () => {
    const root = dirEntry("root", [
      [
        fileEntry(".hidden"),
        fileEntry("Thumbs.db"),
        fileEntry("desktop.ini"),
        fileEntry("good.txt"),
        dirEntry(".git", [[fileEntry("config")]]),
      ],
    ]);
    const files = await collectFromEntries([root, null]);
    expect(files.map((f) => f.name)).toEqual(["good.txt"]);
  });

  it("prunes directories nested deeper than the depth cap", async () => {
    let inner = dirEntry("d9", [[fileEntry("too-deep.txt")]]);
    for (let i = 8; i >= 2; i -= 1) inner = dirEntry(`d${i}`, [[inner]]);
    const root = dirEntry("d1", [[inner, fileEntry("shallow.txt")]]);
    const files = await collectFromEntries([root]);
    expect(files.map((f) => f.name)).toEqual(["shallow.txt"]);
  });

  it("accumulates entries across multiple readEntries batches", async () => {
    const root = dirEntry("root", [
      [fileEntry("a.txt"), fileEntry("b.txt")],
      [fileEntry("c.txt")],
    ]);
    const files = await collectFromEntries([root]);
    expect(files.map((f) => f.name)).toEqual(["a.txt", "b.txt", "c.txt"]);
  });

  it("keeps other entries when a directory is unreadable", async () => {
    const files = await collectFromEntries([
      dirEntry("locked", [], 0),
      fileEntry("ok.txt"),
    ]);
    expect(files.map((f) => f.name)).toEqual(["ok.txt"]);
  });

  it("keeps batches read before a directory becomes unreadable", async () => {
    const root = dirEntry("partial", [[fileEntry("kept.txt")]], 1);
    const files = await collectFromEntries([root]);
    expect(files.map((f) => f.name)).toEqual(["kept.txt"]);
  });
});

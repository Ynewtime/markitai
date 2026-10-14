import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { jsonProblem } from "./config-text.ts";
import { compareLines } from "./diff.ts";
import { editorText, fileText, textLayout } from "./line-endings.ts";

const CRLF_CONFIG = '{\r\n  "llm": {\r\n    "enabled": true,\r\n    "concurrency": 4\r\n  },\r\n  "output": {\r\n    "dir": "out"\r\n  }\r\n}\r\n';

/** What a textarea does to a value: CRLF and CR become LF. */
const textarea = (value: string) => value.replace(/\r\n|\r/g, "\n");

test("a CRLF file sent back unchanged is byte-identical", () => {
  const shown = editorText(CRLF_CONFIG);
  assert.equal(shown.includes("\r"), false);
  assert.equal(textarea(shown), shown);
  assert.equal(fileText(CRLF_CONFIG, textarea(shown)), CRLF_CONFIG);
  const layout = textLayout(CRLF_CONFIG);
  assert.equal(layout.kind, "crlf");
  assert.equal(layout.newline, "\r\n");
  assert.equal(layout.finalBreak, true);
});

test("Review shows only the line that changed, and only that line differs when saved", () => {
  const edited = editorText(CRLF_CONFIG).replace('"concurrency": 4', '"concurrency": 8');
  const diff = compareLines(editorText(CRLF_CONFIG), edited);
  assert.equal(diff.kind, "rows");
  if (diff.kind !== "rows") return;
  assert.deepEqual([diff.added, diff.removed], [1, 1]);
  const sent = fileText(CRLF_CONFIG, edited);
  assert.equal(sent, CRLF_CONFIG.replace('"concurrency": 4', '"concurrency": 8'));
  const before = CRLF_CONFIG.split("\r\n");
  const after = sent.split("\r\n");
  assert.equal(after.length, before.length);
  assert.deepEqual(
    after.flatMap((line, index) => (line === before[index] ? [] : [index])),
    [3],
  );
});

test("new lines get the file's ending, a removed trailing newline stays removed", () => {
  const added = editorText(CRLF_CONFIG).replace('"dir": "out"', '"dir": "out",\n    "allow_overwrite": true');
  assert.equal(fileText(CRLF_CONFIG, added), CRLF_CONFIG.replace('"dir": "out"', '"dir": "out",\r\n    "allow_overwrite": true'));
  assert.equal(fileText(CRLF_CONFIG, editorText(CRLF_CONFIG).replace(/\n$/, "")), CRLF_CONFIG.replace(/\r\n$/, ""));
  const bare = '{\r\n  "a": 1\r\n}';
  assert.equal(textLayout(bare).finalBreak, false);
  assert.equal(fileText(bare, editorText(bare)), bare);
  assert.equal(fileText(bare, editorText(bare) + "\n"), bare + "\r\n");
});

test("a mixed file keeps every untouched line's ending; new lines get the most common one", () => {
  const mixed = '{\r\n  "a": 1,\n  "b": 2,\r\n  "c": 3\r\n}\n';
  const layout = textLayout(mixed);
  assert.equal(layout.kind, "mixed");
  assert.deepEqual(layout.counts, { crlf: 3, lf: 2, cr: 0 });
  assert.equal(layout.newline, "\r\n");
  assert.equal(fileText(mixed, editorText(mixed)), mixed);
  // A changed line keeps the ending of the line it replaced.
  assert.equal(fileText(mixed, editorText(mixed).replace('"a": 1', '"a": 10')), mixed.replace('"a": 1', '"a": 10'));
  // A new line gets CRLF; the others are untouched.
  assert.equal(
    fileText(mixed, editorText(mixed).replace('"a": 1,\n', '"a": 1,\n  "z": 0,\n')),
    mixed.replace('"a": 1,\n', '"a": 1,\n  "z": 0,\r\n'),
  );
  // A change that adds lines pairs the changed line with its key.
  assert.equal(
    fileText(mixed, editorText(mixed).replace('  "b": 2,\n', '  "a2": 0,\n  "b": 20,\n')),
    mixed.replace('  "b": 2,\r\n', '  "a2": 0,\r\n  "b": 20,\r\n'),
  );
  const lfMiddle = '{\r\n  "x": {\r\n    "c": 3\n  }\r\n}\r\n';
  assert.equal(
    fileText(lfMiddle, editorText(lfMiddle).replace('    "c": 3\n', '    "u": 1,\n    "c": 4\n')),
    '{\r\n  "x": {\r\n    "u": 1,\r\n    "c": 4\n  }\r\n}\r\n',
  );
  // A removed line takes its ending with it.
  assert.equal(fileText(mixed, editorText(mixed).replace('  "b": 2,\n', "")), mixed.replace('  "b": 2,\r\n', ""));
  // On a tie, the first ending in the file.
  assert.equal(textLayout('{\n"a": 1\r\n}').newline, "\n");
  assert.equal(textLayout('{\r\n"a": 1\n}').newline, "\r\n");
});

test("LF files, single lines and old Mac CR files round trip too", () => {
  const lf = '{\n  "a": 1\n}\n';
  assert.equal(textLayout(lf).kind, "lf");
  assert.equal(fileText(lf, editorText(lf)), lf);
  assert.equal(fileText(lf, '{\n  "a": 2\n}\n'), '{\n  "a": 2\n}\n');
  assert.equal(textLayout("{}").kind, "none");
  assert.equal(fileText("{}", '{\n  "a": 1\n}'), '{\n  "a": 1\n}');
  const cr = '{\r  "a": 1\r}\r';
  assert.equal(textLayout(cr).kind, "cr");
  assert.equal(fileText(cr, editorText(cr)), cr);
  assert.equal(fileText(cr, editorText(cr).replace("1", "2")), cr.replace("1", "2"));
});

test("a byte order mark is hidden from the editor and the JSON check, and kept", () => {
  const bom = "\uFEFF" + CRLF_CONFIG;
  assert.equal(textLayout(bom).bom, true);
  assert.equal(editorText(bom).startsWith("\uFEFF"), false);
  const words = { configNotJson: () => "bad", configNotJsonAnywhere: "bad", configNotObject: "object" };
  assert.equal(jsonProblem(editorText(bom), words), null);
  assert.equal(fileText(bom, editorText(bom)), bom);
  assert.equal(fileText(bom, editorText(bom).replace("true", "false")), bom.replace("true", "false"));
});

test("a change too large to compare still keeps equal head and tail lines' endings", () => {
  const lines = Array.from({ length: 5000 }, (_, index) => `  "k${index}": ${index},`);
  const raw = `{\r\n${lines.join("\n")}\r\n  "end": 0\r\n}\r\n`;
  const edited = editorText(raw).replace(/: (\d+),/g, (_, n: string) => `: ${Number(n) + 1},`);
  const sent = fileText(raw, edited);
  assert.ok(sent.startsWith("{\r\n"));
  // The changed lines keep their own endings too, by their place in the change.
  assert.equal(sent, raw.replace(/: (\d+),/g, (_, n: string) => `: ${Number(n) + 1},`));
  assert.equal(editorText(sent), edited);
});

test("the Edit config page edits LF text and sends the file's endings", () => {
  const source = readFileSync(new URL("../components/settings-config.tsx", import.meta.url), "utf8");
  assert.match(source, /setText\(editorText\(value\.text/);
  assert.match(source, /saveConfigDocument\(doc\.revision, fileText\(raw, text\), scope\)/);
});

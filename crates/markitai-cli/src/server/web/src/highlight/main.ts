// Code highlighting for the preview, built as its own file (`/ui/highlight.js`)
// and loaded only when a rendered result contains code blocks. A small
// tokenizer per language family colours comments, strings, numbers, keywords,
// types and calls; the text itself is never reparsed as HTML (spans are built
// with textContent), so highlighting cannot change what the sanitizer allowed.

type Rule = [cls: string, pattern: RegExp];

const words = (list: string) => new RegExp(`\\b(?:${list.trim().split(/\s+/).join("|")})\\b`, "y");

const NUMBER: Rule = ["n", /\b(?:0x[\da-f_]+|0b[01_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:e[+-]?\d+)?)[a-z]*\b/iy];
const DQ: Rule = ["s", /"(?:[^"\\\n]|\\.)*"?/y];
const SQ: Rule = ["s", /'(?:[^'\\\n]|\\.)*'?/y];
const BT: Rule = ["s", /`(?:[^`\\]|\\[\s\S])*`?/y];
const SLASH_COMMENT: Rule = ["c", /\/\/[^\n]*/y];
const BLOCK_COMMENT: Rule = ["c", /\/\*[\s\S]*?(?:\*\/|$)/y];
const HASH_COMMENT: Rule = ["c", /#[^\n]*/y];
const CALL: Rule = ["f", /\b[a-z_$][\w$]*(?=\s*\()/iy];
const TYPE: Rule = ["t", /\b[A-Z][\w$]*\b/y];
const CONST: Rule = ["k", /\b(?:true|false|null|undefined|None|True|False|nil|NaN)\b/y];

const C_LIKE = words(`
  abstract as async await break case catch class const continue default defer delete do else enum export extends
  final finally fn for func function go if impl implements import in instanceof interface let loop match mod move
  mut new package private protected pub public readonly return static struct super switch this throw throws trait
  try type typeof unsafe use var void where while with yield chan select range map goto sizeof typedef union
  extern inline virtual override namespace template typename using auto char int long short float double bool
  unsigned signed string self Self crate dyn ref val fun object when is internal open lateinit`);
const PYTHON = words(`
  and as assert async await break class continue def del elif else except finally for from global if import in is
  lambda nonlocal not or pass raise return try while with yield match case self cls print`);
const SHELL = words(`
  if then else elif fi for in do done while until case esac function return local export readonly declare set unset
  source echo exit cd sudo apt npm npx pip uv cargo git curl wget cat grep sed awk mkdir rm cp mv ls chmod docker`);
const SQL = new RegExp(
  `\\b(?:${"select from where and or not insert into values update set delete create table drop alter add index primary key foreign references join left right inner outer on group by order having limit offset as distinct union all case when then else end null is in like between exists view returning with desc asc default unique"
    .split(" ")
    .join("|")})\\b`,
  "iy",
);

const LANGS: Record<string, Rule[]> = {
  clike: [SLASH_COMMENT, BLOCK_COMMENT, DQ, SQ, BT, NUMBER, CONST, ["k", C_LIKE], TYPE, CALL],
  python: [["s", /(?:[rbuf]{0,2})("""|''')[\s\S]*?(?:\1|$)/iy], HASH_COMMENT, ["s", /[rbuf]{0,2}"(?:[^"\\\n]|\\.)*"?/iy], ["s", /[rbuf]{0,2}'(?:[^'\\\n]|\\.)*'?/iy], ["t", /@[\w.]+/y], NUMBER, CONST, ["k", PYTHON], TYPE, CALL],
  shell: [HASH_COMMENT, DQ, SQ, ["t", /\$\{[^}\n]*\}|\$[\w@#?*!$-]+/y], ["k", SHELL], ["f", /(?<=\s|^)--?[\w-]+/y], NUMBER],
  sql: [["c", /--[^\n]*/y], BLOCK_COMMENT, SQ, DQ, NUMBER, ["k", SQL], CALL],
  json: [["p", /"(?:[^"\\\n]|\\.)*"(?=\s*:)/y], DQ, NUMBER, CONST],
  yaml: [HASH_COMMENT, ["p", /[\w.-]+(?=\s*:(?:\s|$))/y], DQ, SQ, NUMBER, CONST, ["k", /^\s*-(?=\s)/my]],
  toml: [HASH_COMMENT, ["t", /^\s*\[[^\]\n]*\]/my], ["p", /[\w.-]+(?=\s*=)/y], DQ, SQ, NUMBER, CONST],
  markup: [["c", /<!--[\s\S]*?(?:-->|$)/y], ["k", /<\/?[\w:-]+|\/?>/y], ["p", /\b[\w:-]+(?==)/y], DQ, SQ],
  css: [BLOCK_COMMENT, DQ, SQ, ["p", /[\w-]+(?=\s*:[^;{}]*[;}])/y], ["t", /[.#][\w-]+/y], ["k", /@[\w-]+/y], ["n", /#[\da-f]{3,8}\b|\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms|deg)?\b/iy]],
  diff: [["s", /^\+[^\n]*/my], ["d", /^-[^\n]*/my], ["c", /^@@[^\n]*/my]],
};

const ALIASES: Record<string, string> = {
  js: "clike", javascript: "clike", jsx: "clike", ts: "clike", typescript: "clike", tsx: "clike", mjs: "clike",
  java: "clike", c: "clike", h: "clike", cpp: "clike", "c++": "clike", cc: "clike", cs: "clike", csharp: "clike",
  go: "clike", golang: "clike", rust: "clike", rs: "clike", swift: "clike", kotlin: "clike", kt: "clike",
  scala: "clike", php: "clike", dart: "clike", zig: "clike", groovy: "clike", objc: "clike", "objective-c": "clike",
  py: "python", python: "python", python3: "python", ipython: "python",
  sh: "shell", bash: "shell", zsh: "shell", shell: "shell", console: "shell", powershell: "shell", ps1: "shell", fish: "shell",
  sql: "sql", mysql: "sql", postgresql: "sql", sqlite: "sql",
  json: "json", jsonc: "json", json5: "json", ipynb: "json",
  yaml: "yaml", yml: "yaml", toml: "toml", ini: "toml", cfg: "toml",
  html: "markup", xml: "markup", svg: "markup", xhtml: "markup", vue: "markup",
  css: "css", scss: "css", less: "css",
  diff: "diff", patch: "diff",
};

/** A language family from the code's class, else from its first lines. */
export function family(code: Element): string | null {
  const named = /\blanguage-([\w+#-]+)/.exec(code.className)?.[1]?.toLowerCase();
  if (named) return ALIASES[named] ?? (named === "text" || named === "plaintext" || named === "txt" ? null : "clike");
  const text = (code.textContent ?? "").slice(0, 400).trimStart();
  if (/^[[{]/.test(text) && /"\s*:/.test(text)) return "json";
  if (/^<[!?\w]/.test(text)) return "markup";
  if (/^(?:\$ |#!\/|(?:sudo|npm|pip|uv|cargo|git|curl|cd|export) )/m.test(text)) return "shell";
  if (/^(?:def |class \w+.*:$|import \w|from \w+ import)/m.test(text)) return "python";
  if (/^\s*(?:select|insert|update|create|with)\b/i.test(text)) return "sql";
  if (/^(?:diff |--- |\+\+\+ |@@ )/m.test(text)) return "diff";
  if (/[;{}]\s*$/m.test(text) || /^(?:fn|function|const|let|var|func|package|use|#include|public|import)\b/m.test(text)) return "clike";
  return null;
}

/** Split text into [class, text] runs; unmatched text has class "". */
export function tokenize(text: string, rules: Rule[]): [string, string][] {
  const out: [string, string][] = [];
  let plain = "";
  let index = 0;
  outer: while (index < text.length) {
    for (const [cls, pattern] of rules) {
      pattern.lastIndex = index;
      const match = pattern.exec(text);
      if (match && match[0].length > 0) {
        if (plain) out.push(["", plain]);
        plain = "";
        out.push([cls, match[0]]);
        index += match[0].length;
        continue outer;
      }
    }
    // Copy a whole word at once so a keyword never matches inside an identifier.
    const word = /[\w$]+|\s+|./y;
    word.lastIndex = index;
    const piece = word.exec(text)?.[0] ?? text[index] ?? "";
    plain += piece;
    index += piece.length;
  }
  if (plain) out.push(["", plain]);
  return out;
}

/** Larger blocks stay plain: tokenizing them would hold the page. */
const MAX_BLOCK = 120_000;
const MAX_TOTAL = 1_500_000;

/** Highlight every unhighlighted code block under `root`, a few per frame. */
export function highlightAll(root: ParentNode, done?: () => void): () => void {
  const blocks = [...root.querySelectorAll<HTMLElement>("pre > code:not([data-hl])")];
  let budget = MAX_TOTAL;
  let index = 0;
  let stopped = false;
  const step = () => {
    const until = performance.now() + 12;
    while (!stopped && index < blocks.length && performance.now() < until) {
      const code = blocks[index++] as HTMLElement;
      code.setAttribute("data-hl", "");
      const text = code.textContent ?? "";
      const lang = family(code);
      if (lang === null || text.length > MAX_BLOCK || text.length > budget) continue;
      budget -= text.length;
      const fragment = document.createDocumentFragment();
      for (const [cls, piece] of tokenize(text, LANGS[lang] ?? [])) {
        if (!cls) {
          fragment.append(piece);
          continue;
        }
        const span = document.createElement("span");
        span.className = `tok-${cls}`;
        span.textContent = piece;
        fragment.append(span);
      }
      code.replaceChildren(fragment);
    }
    if (stopped) return;
    if (index < blocks.length) setTimeout(step, 0);
    else done?.();
  };
  step();
  return () => {
    stopped = true;
  };
}

export { LANGS };

// Mermaid diagrams: Pele's drawing read back through the allowlist, what falls back to code, and what never
// reaches the page, against the pinned renderer the page loads as /ui/mermaid.js.
import assert from "node:assert/strict";
import test from "node:test";
// The vendored parser the page loads as /ui/marked.js.
import { marked } from "../../../../../../../vendor/web/marked.js";
import { en } from "../i18n/en.ts";
import { drawDiagram, LIMITS, type DrawOptions } from "../mermaid/main.ts";
import { buildDiagram, cleanDiagram, diagramNote, diagramStyles, parseSvg, readDiagram, type DiagramElement, type DiagramNode } from "./diagram.ts";

const prefix = "md-diagram-1-";
const options: DrawOptions = { fontFamily: "sans-serif", fontSize: 14, maxWidth: 760, idPrefix: prefix, links: true };
const draw = (source: string, extra: Partial<DrawOptions> = {}) => drawDiagram(source, { ...options, ...extra });

function drawn(source: string, extra: Partial<DrawOptions> = {}): { svg: string; tree: DiagramElement } {
  const result = draw(source, extra);
  assert.ok(result.ok, `not drawn: ${JSON.stringify(result)}`);
  const tree = readDiagram(result.svg, { prefix, links: extra.links ?? true });
  assert.ok(tree, "the drawing did not read back");
  return { svg: result.svg, tree };
}

const elements = (node: DiagramNode): DiagramElement[] => (typeof node === "string" ? [] : [node, ...node.children.flatMap(elements)]);
const text = (node: DiagramNode): string => (typeof node === "string" ? node : node.children.map(text).join(""));
const attribute = (element: DiagramElement, name: string) => element.attributes.find(([key]) => key === name)?.[1];

test("a mermaid fence reaches the page as a code block the diagram pass finds", () => {
  const html = marked.parse("```mermaid\nflowchart LR\n  A --> B\n```", { async: false, gfm: true });
  assert.equal(html.trim(), '<pre><code class="language-mermaid">flowchart LR\n  A --&gt; B\n</code></pre>');
});

test("a flowchart is drawn with its labels, Pele's classes and its colour variables", () => {
  const { tree } = drawn("flowchart LR\n  A[Start] --> B{Ready?}\n  B -->|yes| C[Done]\n  B -->|no| A");
  assert.equal(tree.tag, "svg");
  assert.equal(attribute(tree, "class"), "pele pele-flowchart");
  for (const label of ["Start", "Ready?", "Done", "yes", "no"]) assert.ok(text(tree).includes(label), label);
  const variables = new Map(tree.styles);
  assert.equal(variables.get("--_bg"), "var(--pele-bg,#fff)");
  assert.equal(variables.get("--_c"), "var(--pele-accent,#5b7bd5)");
  assert.equal(variables.get("max-width"), "100%");
  // Corner radii follow --pele-radius through the CSS object model, not a style attribute.
  assert.ok(elements(tree).some((element) => element.tag === "rect" && element.styles.some(([name, value]) => name === "rx" && value === "var(--_r)")));
  assert.ok(elements(tree).every((element) => !element.attributes.some(([name]) => name === "style")));
});

test("a sequence diagram is drawn with its participants and messages", () => {
  const { tree } = drawn("sequenceDiagram\n  participant Alice\n  participant Bob\n  Alice->>Bob: Hello Bob\n  Bob-->>Alice: Hi Alice");
  assert.equal(attribute(tree, "aria-roledescription"), "sequence");
  for (const label of ["Alice", "Bob", "Hello Bob", "Hi Alice"]) assert.ok(text(tree).includes(label), label);
});

test("every element and attribute Pele draws survives the allowlist", () => {
  const diagrams = [
    "flowchart TD\n  subgraph one [Group]\n    A([Round]) --> B[(Store)]\n  end\n  B -.-> C((Circle))\n  click C \"https://example.com/c\" \"Open C\"",
    "sequenceDiagram\n  autonumber\n  Alice->>+John: Hi\n  John-->>-Alice: Hi back\n  Note right of John: thinks\n  loop Every minute\n    John->>Alice: ping\n  end",
    "classDiagram\n  Animal <|-- Duck\n  Animal : +int age\n  Animal: +isMammal() bool",
    "stateDiagram-v2\n  [*] --> Still\n  Still --> Moving\n  Moving --> [*]",
    "erDiagram\n  CUSTOMER ||--o{ ORDER : places\n  ORDER ||--|{ LINE-ITEM : contains",
    "gantt\n  title Plan\n  dateFormat YYYY-MM-DD\n  section A\n  Task one :a1, 2026-01-01, 30d\n  Task two :after a1, 20d",
    "pie title Pets\n  \"Dogs\" : 386\n  \"Cats\" : 85\n  \"Rats\" : 15",
    "gitGraph\n  commit\n  branch dev\n  commit\n  checkout main\n  merge dev",
    "mindmap\n  root((Root))\n    One\n    Two\n      Three",
    "xychart\n  title Sales\n  x-axis [jan, feb, mar]\n  y-axis \"Units\" 0 --> 10\n  bar [3, 7, 5]\n  line [3, 7, 5]",
    "journey\n  title Day\n  section Work\n    Code: 5: Me\n    Review: 3: Me, You",
    "timeline\n  title History\n  2024 : One\n  2025 : Two : Three",
    "quadrantChart\n  x-axis Low --> High\n  y-axis Low --> High\n  A: [0.3, 0.6]\n  B: [0.7, 0.2]",
  ];
  // Dropped on purpose: Pele's data hooks, and the link attributes the workbench writes itself.
  const ignored = new Set(["data-id", "data-icon", "target", "rel", "xmlns"]);
  for (const source of diagrams) {
    const { svg, tree } = drawn(source);
    const tags = [...svg.matchAll(/<([A-Za-z]+)[\s/>]/g)].map((match) => match[1]);
    assert.equal(elements(tree).length, tags.length, `${source.split("\n")[0]}: elements lost`);
    const written = [...svg.matchAll(/\s([\w:-]+)="/g)].map((match) => match[1] as string).filter((name) => !ignored.has(name) && name !== "style" && name !== "class");
    const read = parseSvg(svg) as DiagramElement;
    const keptNames = new Set(elements(tree).flatMap((element) => element.attributes.map(([name]) => name)));
    for (const name of new Set(written)) assert.ok(keptNames.has(name), `${source.split("\n")[0]}: attribute ${name} lost`);
    const styled = elements(read).flatMap((element) => element.attributes.filter(([name]) => name === "style").map(([, value]) => value));
    const keptStyles = elements(tree).reduce((count, element) => count + element.styles.length, 0);
    const writtenStyles = styled.reduce((count, value) => count + value.split(";").filter((part) => part.includes(":")).length, 0);
    assert.equal(keptStyles, writtenStyles, `${source.split("\n")[0]}: style declarations lost`);
  }
});

test("a flowchart wider than the column takes the vertical layout instead of shrinking", () => {
  const chain = "flowchart LR\n  A[Pull request] --> B[Checks] --> C[Code review] --> D[Merge to main] --> E[Tag release] --> F[Publish]";
  const width = (svg: string) => Number(/viewBox="0 0 ([\d.]+)/.exec(svg)?.[1]);
  const wide = drawn(chain, { maxWidth: 2000 }).svg;
  const narrow = drawn(chain, { maxWidth: 700 }).svg;
  assert.ok(width(wide) > 700, `the chain is ${width(wide)} px wide laid out left to right`);
  assert.ok(width(narrow) <= 700, `upright it is ${width(narrow)} px wide`);
});

test("a label on a series colour takes that series' label colour; others keep Pele's", () => {
  const fillOf = (tree: DiagramElement, label: string) =>
    elements(tree).find((element) => element.tag === "text" && text(element) === label)?.attributes.find(([name]) => name === "fill")?.[1];
  const on = (series: number) => `var(--diagram-on-${series}, var(--_bg))`;
  const git = drawn("gitGraph\n  commit\n  branch dev\n  commit\n  branch feature\n  commit\n  branch fix\n  commit").tree;
  assert.deepEqual(["main", "dev", "feature", "fix"].map((name) => fillOf(git, name)), [on(1), on(2), on(3), on(4)]);
  // A pie's labels sit apart from its slices: each takes the slice its angle falls in; a label outside (2%) keeps the text colour.
  const pie = drawn('pie\n  "A" : 10\n  "B" : 1\n  "C" : 30\n  "D" : 1\n  "E" : 20').tree;
  assert.deepEqual(["16%", "48%", "32%", "2%"].map((label) => fillOf(pie, label)), [on(1), on(3), on(5), undefined]);
  const ring = drawn('---\nconfig:\n  pie:\n    donutHole: 0.2\n---\npie\n  "A" : 40\n  "B" : 46\n  "C" : 9').tree;
  assert.deepEqual(["42%", "48%", "9%"].map((label) => fillOf(ring, label)), [on(1), on(2), on(3)]);
  const timeline = drawn("timeline\n  section Early\n    2024 : One\n  section Late\n    2026 : Three").tree;
  assert.deepEqual(["Early", "2024", "Late", "2026", "One"].map((label) => fillOf(timeline, label)), [on(1), on(1), on(2), on(2), undefined]);
  const treemap = drawn('treemap-beta\n"Root"\n  "Leaf": 10\n  "Other": 20').tree;
  assert.equal(fillOf(treemap, "Leaf"), on(1));
  // The accent and critical bars of a Gantt chart are not series colours: their labels stay as Pele draws them.
  const gantt = drawn("gantt\n  dateFormat YYYY-MM-DD\n  section A\n  Task one :crit, a1, 2026-01-01, 30d\n  Task two :after a1, 20d").tree;
  assert.deepEqual(["Task one", "Task two"].map((label) => fillOf(gantt, label)), ["var(--_bg)", "var(--_bg)"]);
  // A colour the diagram sets itself, or a translucent fill, is left alone.
  const own = readDiagram(
    '<svg fill="var(--_fg)"><g><rect fill="var(--pele-series-4,#e45756)"/><text fill="var(--_bg)" style="fill:#ff0000">set</text></g><g><rect fill="var(--pele-series-2,#f58518)" fill-opacity="0.16"/><text fill="var(--_bg)">pale</text></g></svg>',
    { prefix, links: true },
  ) as DiagramElement;
  assert.deepEqual(["set", "pale"].map((label) => fillOf(own, label)), ["var(--_bg)", "var(--_bg)"]);
});

test("a diagram with an error falls back to its code, with the line in the note", () => {
  const result = draw("flowchart LR\n  A --> B\n  B -->");
  assert.equal(result.ok, false);
  assert.ok(!result.ok && result.code === "syntax" && result.line > 0);
  if (!result.ok) assert.equal(diagramNote(en, result.code, result.line), en.diagramSyntax(result.line));
  const unknown = draw("not a diagram at all");
  assert.deepEqual(unknown.ok ? null : unknown.code, "unsupported-diagram");
  assert.equal(diagramNote(en, "unsupported-diagram", 0), en.diagramUnsupported);
  assert.equal(diagramNote(en, "failed", 0), en.diagramFailed);
});

test("a huge diagram falls back to its code: too long, or too many edges", () => {
  const long = `flowchart TD\n${Array.from({ length: 6000 }, (_, i) => `  n${i} --> n${i + 1}`).join("\n")}`;
  assert.ok(long.length > LIMITS.source);
  const result = draw(long);
  assert.deepEqual(result.ok ? null : result.code, "limit");
  assert.equal(diagramNote(en, "limit", 0), en.diagramTooLarge);
  const dense = `flowchart TD\n${Array.from({ length: LIMITS.edges + 1 }, (_, i) => `n${i % 50}-->n${(i * 7) % 50}`).join("\n")}`;
  assert.ok(dense.length < LIMITS.source);
  const crowded = draw(dense);
  assert.deepEqual(crowded.ok ? null : crowded.code, "limit");
});

test("markup and script addresses in labels and links stay text or go", () => {
  const source = [
    "flowchart LR",
    '  A["#lt;script#gt;alert(1)#lt;/script#gt;"] --> B["<img src=x onerror=alert(1)>"]',
    '  B --> C["javascript:alert(1)"]',
    '  C --> D["safe"]',
    '  click A "javascript:alert(1)"',
    '  click B "data:text/html,<script>alert(1)</script>"',
    '  click D "https://example.com/d"',
  ].join("\n");
  const { svg, tree } = drawn(source);
  assert.ok(!/<script|<img|<foreignObject|onerror=|javascript:alert\(1\)"|data:text/i.test(svg.replace(/>[^<]*</g, "><")));
  const all = elements(tree);
  assert.ok(all.every((element) => !["script", "foreignObject", "image", "img", "style", "iframe"].includes(element.tag)));
  assert.ok(all.every((element) => element.attributes.every(([name, value]) => !/^on/i.test(name) && !/^\s*(?:javascript|data|vbscript):/i.test(value))));
  // The entity-coded tag is the label's text: drawn as characters, never read as markup.
  assert.ok(text(tree).includes("<script>alert(1)</script>"));
  assert.ok(text(tree).includes("javascript:alert(1)"));
  const links = all.filter((element) => element.tag === "a");
  assert.deepEqual(
    links.map((link) => link.attributes),
    [[["href", "https://example.com/d"], ["target", "_blank"], ["rel", "noopener noreferrer"], ["referrerpolicy", "no-referrer"]]],
  );
  // Without links (a Details excerpt) the link is a plain group.
  const plain = drawn(source, { links: false }).tree;
  assert.ok(elements(plain).every((element) => element.tag !== "a" && attribute(element, "href") === undefined));
});

test("markup Pele never writes is still refused: loads, scripts, foreign ids, the page's classes", () => {
  const hostile = [
    '<svg xmlns="http://www.w3.org/2000/svg" class="pele sr-only btn" onload="alert(1)" style="--_bg:url(https://evil.example/x);behavior:url(x);--x:red;fill:red">',
    "<script>alert(1)</script><style>*{display:none}</style>",
    '<foreignObject><div xmlns="http://www.w3.org/1999/xhtml">x</div></foreignObject>',
    '<image href="https://evil.example/p.png"/><feImage href="https://evil.example/f.svg"/>',
    '<a href="javascript:alert(1)"><text>bad</text></a><a xlink:href="https://example.com/ok"><text>ok</text></a>',
    `<linearGradient id="${prefix}g"><stop offset="0" stop-color="red"/></linearGradient>`,
    `<rect id="page-header" fill="url(#${prefix}g)" stroke="url(https://evil.example/s)" style="fill:a\\75rl(x);stroke-width:2"/>`,
    `<use href="https://evil.example/s.svg#a"/><use href="#${prefix}g"/>`,
    '<set attributeName="href" to="javascript:alert(1)"/><animate attributeName="x"/>',
    "</svg>",
  ].join("");
  const parsed = parseSvg(hostile);
  assert.ok(parsed);
  const tree = cleanDiagram(parsed, { prefix, links: true }) as DiagramElement;
  const tags = elements(tree).map((element) => element.tag);
  assert.deepEqual(tags, ["svg", "g", "text", "a", "text", "linearGradient", "stop", "rect", "use"]);
  assert.equal(attribute(tree, "class"), "pele");
  assert.equal(attribute(tree, "onload"), undefined);
  assert.deepEqual(tree.styles, [["fill", "red"]]);
  const rect = elements(tree).find((element) => element.tag === "rect") as DiagramElement;
  assert.deepEqual(rect.attributes, [["fill", `url(#${prefix}g)`]]);
  assert.deepEqual(rect.styles, [["stroke-width", "2"]]);
  assert.equal(attribute(elements(tree).find((element) => element.tag === "a") as DiagramElement, "href"), "https://example.com/ok");
  assert.equal(attribute(elements(tree).find((element) => element.tag === "use") as DiagramElement, "href"), `#${prefix}g`);
});

test("declarations: Pele's variables and drawing properties with plain values only", () => {
  assert.deepEqual(diagramStyles("--_bg:var(--pele-bg,#fff); rx:var(--_r); FILL:#f9f !important; stroke:expression(alert(1)); font-family:evil", new Map()), [
    ["--_bg", "var(--pele-bg,#fff)"],
    ["rx", "var(--_r)"],
    ["fill", "#f9f"],
    ["font-family", "evil"],
  ]);
  assert.deepEqual(diagramStyles("--pele-bg:red;--_x:attr(x);color:image(x);font:{x};width:9px;fill:url(#elsewhere)", new Map()), []);
});

test("markup that is not well formed is not read", () => {
  for (const bad of ["<svg><g></svg>", "<svg>&nbsp;</svg>", '<!DOCTYPE svg><svg/>', "<svg/><svg/>", "text<svg/>", "<svg><![CDATA[x]]></svg>", "<svg a=b/>"]) {
    assert.equal(parseSvg(bad), null, bad);
  }
  assert.equal(cleanDiagram(parseSvg("<g/>") as DiagramElement, { prefix, links: true }), null);
  assert.deepEqual(parseSvg('<svg a="&lt;&#x41;&#66;&quot;"><text>&amp;</text></svg>'), {
    tag: "svg",
    attributes: [["a", '<AB"']],
    styles: [],
    children: [{ tag: "text", attributes: [], styles: [], children: ["&"] }],
  });
});

test("the drawing is built with DOM calls: text nodes, namespaced xml:space, styles through the object model", () => {
  const calls: string[] = [];
  const fake = (tag: string) => {
    const node = {
      tag,
      children: [] as unknown[],
      setAttribute: (name: string, value: string) => calls.push(`${tag} @${name}=${value}`),
      setAttributeNS: (ns: string, name: string, value: string) => calls.push(`${tag} @${ns}|${name}=${value}`),
      style: { setProperty: (name: string, value: string) => calls.push(`${tag} style ${name}=${value}`) },
      append: (child: unknown) => node.children.push(child),
    };
    return node;
  };
  const doc = {
    createElementNS: (ns: string, tag: string) => (calls.push(`create ${ns} ${tag}`), fake(tag)),
    createTextNode: (value: string) => (calls.push(`text ${value}`), { value }),
  } as unknown as Document;
  const tree = readDiagram('<svg style="--_bg:var(--pele-bg,#fff)"><text xml:space="preserve">&lt;b&gt;</text></svg>', { prefix, links: true }) as DiagramElement;
  buildDiagram(doc, tree);
  assert.deepEqual(calls, [
    "create http://www.w3.org/2000/svg svg",
    "svg style --_bg=var(--pele-bg,#fff)",
    "create http://www.w3.org/2000/svg text",
    "text @http://www.w3.org/XML/1998/namespace|xml:space=preserve",
    "text <b>",
  ]);
});

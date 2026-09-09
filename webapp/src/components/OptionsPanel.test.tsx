import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { ADVANCED_DEFAULTS } from "../lib/advanced";
import { OptionsPanel } from "./OptionsPanel";
import { UrlInput } from "./UrlInput";

const t = dicts.en;
const base = {
  t,
  value: ADVANCED_DEFAULTS,
  preset: "minimal" as const,
  llm: true,
  ocr: false,
  profile: null,
  llmConfigured: true,
  urls: [] as string[],
  announce: vi.fn(),
  onChange: vi.fn(),
  onPreset: vi.fn(),
  onLlm: vi.fn(),
  onOcr: vi.fn(),
  onProfile: vi.fn(),
};

type Props = Parameters<typeof OptionsPanel>[0];

function panel(props: Partial<Props> = {}) {
  const handlers = {
    onChange: vi.fn(),
    onPreset: vi.fn(),
    onLlm: vi.fn(),
    onOcr: vi.fn(),
    onProfile: vi.fn(),
  };
  render(<OptionsPanel {...base} {...handlers} {...props} />);
  return handlers;
}

function open(props: Partial<Props> = {}) {
  const handlers = panel(props);
  fireEvent.click(screen.getByRole("button", { name: (props.t ?? t).options }));
  return handlers;
}

/** The fetch/source/cache switches live behind the Advanced fold, which
 * only starts open when one of them already differs from its default. */
function openAdvanced(props: Partial<Props> = {}) {
  const handlers = open(props);
  const fold = screen.getByRole("button", { name: (props.t ?? t).advanced });
  if (fold.getAttribute("aria-expanded") !== "true") fireEvent.click(fold);
  return handlers;
}

describe("OptionsPanel", () => {
  it("leads with the preset, then labelled rows, and folds the rare switches away", () => {
    open();
    expect(screen.getByRole("group", { name: t.preset })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: t.advEnhance }))
      .toContainElement(screen.getByLabelText(t.llmEnhance));
    expect(screen.getByRole("group", { name: t.advImages }))
      .toContainElement(screen.getByLabelText(t.advAlt));
    expect(screen.getByRole("group", { name: t.advOutput }))
      .toContainElement(screen.getByRole("button", { name: "rag" }));
    const fold = screen.getByRole("button", { name: t.advanced });
    expect(fold).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText(t.advStrategy)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(t.advPure)).not.toBeInTheDocument();
    fireEvent.click(fold);
    expect(fold).toHaveAttribute("aria-expanded", "true");
    const body = document.getElementById(fold.getAttribute("aria-controls")!);
    expect(body).toContainElement(screen.getByLabelText(t.advStrategy));
    expect(body).toContainElement(screen.getByLabelText(t.advPure));
    expect(body).toContainElement(screen.getByLabelText(t.advNoCache));
  });

  it("starts the Advanced fold open when one of its switches is already non-default", () => {
    // A closed fold must never hide a choice that changes the job.
    open({ value: { ...ADVANCED_DEFAULTS, noCache: true } });
    expect(screen.getByRole("button", { name: t.advanced })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText(t.advNoCache)).toBeChecked();
  });

  it("asks nothing until opened", () => {
    panel();
    expect(screen.queryByLabelText(t.advAlt)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(t.llmEnhance)).not.toBeInTheDocument();
  });

  it("shares one live command across disclosure states instead of a stale summary", () => {
    const { rerender } = render(<OptionsPanel {...base} ocr profile="rag" />);
    const command = screen.getByLabelText(t.cliAria);
    expect(command).toHaveTextContent("--llm --ocr --profile rag");
    for (const expanded of [true, false, true, false]) {
      fireEvent.click(screen.getByRole("button", { name: t.options }));
      expect(screen.getByRole("button", { name: t.options })).toHaveAttribute("aria-expanded", String(expanded));
      expect(screen.getAllByLabelText(t.cliAria)).toEqual([command]);
      expect(command).toBeVisible();
      expect(screen.queryByLabelText(t.optionsChanged)).not.toBeInTheDocument();
      expect(document.querySelector(".optsum")).toBeNull();
    }
    rerender(<OptionsPanel {...base} urls={["https://example.com/new"]}
      value={{ ...ADVANCED_DEFAULTS, strategy: "jina" }} />);
    expect(command).toHaveTextContent("markitai https://example.com/new");
    expect(command).toHaveTextContent("--strategy jina");
    expect(command).not.toHaveTextContent("--ocr");
  });

  it("carries the primary switches", () => {
    const { onLlm, onOcr, onProfile } = open();
    fireEvent.click(screen.getByLabelText(t.llmEnhance));
    expect(onLlm).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByLabelText(t.ocr));
    expect(onOcr).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "rag" }));
    expect(onProfile).toHaveBeenCalledWith("rag");
  });

  it("keeps OCR reachable when no LLM is configured", () => {
    open({ llmConfigured: false, llm: false });
    expect(screen.getByLabelText(t.ocr)).toBeInTheDocument();
    expect(screen.getByLabelText(t.llmEnhance)).toBeDisabled();
    // the reason rides the row label as a tooltip mark, not a hint paragraph
    expect(screen.getByRole("button", { name: t.advNeedsModel })).toBeVisible();
    expect(screen.getByRole("button", { name: "standard" })).toBeDisabled();
  });

  it("keeps presets reachable while LLM is off", () => {
    const { onPreset } = open({ llm: false });
    fireEvent.click(screen.getByRole("button", { name: "standard" }));
    expect(onPreset).toHaveBeenCalledWith("standard");
  });

  it("displays the inherited rich preset instead of false overrides", () => {
    open({ preset: "rich" });
    expect(screen.getByLabelText(t.advAlt)).toBeChecked();
    expect(screen.getByLabelText(t.advDesc)).toBeChecked();
    expect(screen.getByLabelText(t.advScreenshot)).toBeChecked();
    expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--preset rich");
    expect(screen.queryByText(t.presetCustomized)).not.toBeInTheDocument();
  });

  it("shows a customized preset and its explicit off switch in the command", () => {
    open({ preset: "rich", value: { ...ADVANCED_DEFAULTS, alt: false } });
    expect(screen.getByLabelText(t.advAlt)).not.toBeChecked();
    expect(screen.getByText(t.presetCustomized)).toBeVisible();
    expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--preset rich --no-alt");
  });

  it("explains and enforces source mode exclusivity", () => {
    const { onChange } = open({ value: { ...ADVANCED_DEFAULTS, pure: true } });
    expect(screen.getByLabelText(t.advAlt)).toBeDisabled();
    expect(screen.getByText(t.advSourceExclusive)).toBeVisible();
    fireEvent.click(screen.getByLabelText(t.advScreenshotOnly));
    expect(onChange).toHaveBeenCalledWith({ ...ADVANCED_DEFAULTS, pure: false, screenshotOnly: true });
  });

  it("displays the implied Cloudflare backend and external-service notice", () => {
    open({ value: { ...ADVANCED_DEFAULTS, strategy: "cloudflare", backend: "kreuzberg" } });
    const backend = screen.getByRole("group", { name: t.advBackend });
    const cloudflare = within(backend).getByRole("button", { name: "cloudflare" });
    expect(cloudflare).toHaveAttribute("aria-pressed", "true");
    for (const button of within(backend).getAllByRole("button")) expect(button).toBeDisabled();
    expect(screen.getByRole("button", { name: t.advCloudflareBackend })).toBeInTheDocument();
    expect(screen.getByText(t.noticeCloudflareUrl)).toBeVisible();
    expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--strategy cloudflare --backend cloudflare");
  });

  it("reports one changed option without touching the rest", () => {
    const { onChange } = openAdvanced();
    fireEvent.click(screen.getByLabelText(t.advPure));
    expect(onChange).toHaveBeenCalledWith({ ...ADVANCED_DEFAULTS, pure: true });
  });

  it("disables image analysis when LLM is off", () => {
    // The CLI rejects --alt/--desc without --llm. Showing them disabled keeps
    // the capability discoverable; hiding them would not say why.
    openAdvanced({ llm: false });
    expect(screen.getByLabelText(t.advAlt)).toBeDisabled();
    expect(screen.getByLabelText(t.advDesc)).toBeDisabled();
    expect(screen.getByLabelText(t.advPure)).not.toBeDisabled();
  });

  it("shows page screenshots as on once the source is screenshots", () => {
    // --screenshot-only implies --screenshot, so an unchecked box here would
    // contradict what the job actually runs with.
    open({ value: { ...ADVANCED_DEFAULTS, screenshotOnly: true } });
    const box = screen.getByLabelText(t.advScreenshot);
    expect(box).toBeChecked();
    expect(box).toBeDisabled();
  });

  it("shows the command those settings produce, in the panel", () => {
    // It is the settings spelled out, not a sibling of them — a second
    // disclosure made it look like an unrelated feature.
    open({ llm: true, ocr: false });
    expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--llm");
    expect(screen.getByRole("button", { name: t.copy })).toBeInTheDocument();
  });

  it("offers every fetch strategy and backend the CLI accepts as one labelled choice each", () => {
    // Segmented buttons, not selects: every value is visible at once and the
    // current one reads without opening anything.
    const { onChange } = openAdvanced();
    const strategy = screen.getByRole("group", { name: t.advStrategy });
    for (const name of ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"]) {
      expect(within(strategy).getByRole("button", { name })).toHaveAttribute(
        "aria-pressed", name === "auto" ? "true" : "false",
      );
    }
    const backend = screen.getByRole("group", { name: t.advBackend });
    expect(within(backend).getByRole("button", { name: "kreuzberg" })).toBeInTheDocument();
    fireEvent.click(within(strategy).getByRole("button", { name: "jina" }));
    expect(onChange).toHaveBeenCalledWith({ ...ADVANCED_DEFAULTS, strategy: "jina" });
  });

  it("explains rows with tooltips instead of hint paragraphs", () => {
    open();
    expect(screen.getByRole("button", { name: t.presetHint })).toBeInTheDocument();
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});

it.each([dicts.en, dicts.zh])("keeps tools above the input in either locale", (dict) => {
  panel({ t: dict, onFiles: vi.fn(), source: (
    <UrlInput t={dict} text="" onText={vi.fn()} onConvert={vi.fn()}
      />
  ) });
  const input = screen.getByRole("textbox");
  const actions = screen.getByRole("group", { name: dict.sourceActions });
  const toggle = screen.getByRole("button", { name: dict.options });
  const convert = screen.getByRole("button", { name: dict.convert });
  expect(actions).toContainElement(toggle);
  expect(actions).toContainElement(screen.getByLabelText(dict.browse, { selector: "input" }));
  expect(actions).not.toContainElement(convert);
  expect(convert).toBeDisabled();
  expect(actions.compareDocumentPosition(input) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute("aria-expanded", "true");
  expect(document.getElementById(toggle.getAttribute("aria-controls")!)).toContainElement(screen.getByLabelText(dict.advAlt));
  expect(screen.getAllByLabelText(dict.cliAria)).toHaveLength(1);
  expect(screen.getAllByRole("textbox")).toHaveLength(1);
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute("aria-expanded", "false");
});
it("prints the literal input placeholder without a CLI comment", () => {
  open({ llm: false });
  const command = screen.getByLabelText(t.cliAria);
  expect(command).toHaveTextContent("$ markitai <your-files-or-url-or-url_files> -o out/ --preset minimal");
  expect(command.textContent).not.toContain("--no-");
  expect(command).toHaveAccessibleDescription(t.helpCli);
  expect(document.querySelector(".clinote")).toBeNull();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
});
it("uses real inputs without adding a CLI comment", () => {
  open({ urls: ["https://example.com/a"] });
  expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("markitai https://example.com/a -o out/");
  expect(document.querySelector(".clinote")).toBeNull();
});
it("compacts against the server preset map rather than hardcoded rich defaults", () => {
  open({ preset: "rich", ocr: true, presetOptions: {
    rich: { llm: true, ocr: true, alt: false, desc: false, screenshot: false },
  } });
  expect(screen.getByLabelText(t.cliAria).textContent).toBe("$ markitai <your-files-or-url-or-url_files> -o out/ --preset rich");
});


it("shows Custom with no preset pressed for minimal plus LLM", () => {
  const { onChange, onPreset } = open();
  expect(screen.getByText(t.presetCustomized)).toBeVisible();
  for (const name of ["minimal", "standard", "rich"]) {
    expect(screen.getByRole("button", { name })).toHaveAttribute("aria-pressed", "false");
  }
  expect(screen.getByLabelText(t.advAlt)).not.toBeChecked();
  expect(screen.getByLabelText(t.advDesc)).not.toBeChecked();
  expect(onChange).not.toHaveBeenCalled();
  expect(onPreset).not.toHaveBeenCalled();
});

it("highlights a matching bundle without changing options or the command", () => {
  const { onPreset } = open({ value: { ...ADVANCED_DEFAULTS, alt: true, desc: true } });
  expect(screen.getByRole("button", { name: "standard" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByText(t.presetCustomized)).not.toBeInTheDocument();
  expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--preset minimal --llm --alt --desc");
  expect(onPreset).not.toHaveBeenCalled();
});

it.each([dicts.en, dicts.zh])("describes every choice and group in either locale with accessible help", (dict) => {
  openAdvanced({ t: dict });
  const fields = document.querySelectorAll(".optfield button");
  expect(fields.length).toBe(25);
  for (const button of fields) {
    const description = document.getElementById(button.getAttribute("aria-describedby")!);
    expect(description?.textContent?.length).toBeGreaterThan(5);
    expect(description?.textContent?.length).toBeLessThan(180);
    expect(description?.textContent?.match(/[。.!?]/g)).toHaveLength(1);
  }
  for (const label of document.querySelectorAll(".optrow > .optlbl")) {
    expect(label.querySelector("button[aria-describedby]")).not.toBeNull();
  }
  const help = screen.getByRole("button", { name: dict.presetHint });
  fireEvent.focus(help);
  const tooltip = screen.getByRole("tooltip");
  expect(tooltip).toHaveTextContent(dict.presetHint);
  expect(tooltip.parentElement).toBe(document.body);
  fireEvent.keyDown(window, { key: "Escape" });
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  fireEvent.click(help);
  expect(screen.getByRole("tooltip")).toBeVisible();
  fireEvent.blur(help);
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
});

it.each(["defuddle", "jina", "cloudflare", "static", "playwright", "auto"] as const)(
  "discloses the %s URL destination separately from a Cloudflare file backend", (strategy) => {
    openAdvanced({ value: { ...ADVANCED_DEFAULTS, strategy, backend: "cloudflare" } });
    expect(screen.getByText(t.noticeCloudflareFile)).toBeVisible();
    const notices = { defuddle: t.noticeDefuddle, jina: t.noticeJina, cloudflare: t.noticeCloudflareUrl,
      auto: t.noticeAuto, static: null, playwright: null };
    for (const [name, notice] of Object.entries(notices)) {
      if (notice) expect(screen.queryByText(notice) !== null).toBe(name === strategy);
    }
  },
);

it.each([dicts.en, dicts.zh])("explains extra vision OCR costs without enabling images", (dict) => {
  open({ t: dict, llm: true, ocr: true });
  expect(screen.getByText(dict.advVlmOcr)).toBeVisible();
  expect(dict.advVlmOcr).toMatch(/^LLM \+ OCR/);
  expect(dict.advVlmOcr).toMatch(/more model costs|更多模型费用/);
  expect(screen.getByLabelText(dict.advAlt)).not.toBeChecked();
});

it("uploads files from the top toolbar without submitting URLs", () => {
  const onFiles = vi.fn();
  const onConvert = vi.fn();
  panel({ onFiles, source: <UrlInput t={t} text="https://example.com" onText={vi.fn()} onConvert={onConvert} /> });
  const file = new File(["hello"], "hello.txt", { type: "text/plain" });
  fireEvent.change(screen.getByLabelText(t.browse, { selector: "input" }), { target: { files: [file] } });
  expect(onFiles).toHaveBeenCalledWith([file]);
  expect(onConvert).not.toHaveBeenCalled();
});

it.each([dicts.en, dicts.zh])("keeps all conversion help to one short sentence", (dict) => {
  for (const [key, text] of Object.entries(dict)) {
    if (!key.startsWith("help") || key === "helpLabel" || typeof text !== "string") continue;
    expect(text.length, key).toBeLessThan(180);
    expect(text.match(/[。.!?]/g), key).toHaveLength(1);
    expect(text, key).not.toMatch(/ · |LLM (on|off|开启|关闭)/);
  }
});

it.each([dicts.en, dicts.zh])("uses factual terse help for overridden and disabled presets", (dict) => {
  const props = { ...base, t: dict, presetOptions: {
    rich: { llm: false, ocr: true, alt: false, desc: false, screenshot: false },
  } };
  const { rerender } = render(<OptionsPanel {...props} />);
  fireEvent.click(screen.getByRole("button", { name: dict.options }));
  const rich = screen.getByRole("button", { name: "rich" });
  expect(rich).toHaveAccessibleDescription(dict.helpServerPreset);
  rerender(<OptionsPanel {...base} t={dict} />);
  expect(rich).toHaveAccessibleDescription(dict.helpRich);
  rerender(<OptionsPanel {...base} t={dict} llmConfigured={false} llm={false} />);
  expect(rich).toBeDisabled();
  expect(rich).toHaveAccessibleDescription(dict.advNeedsModel);
  expect(screen.getByRole("switch", { name: dict.advAlt })).toHaveAccessibleDescription(dict.advNeedsLlm);
  rerender(<OptionsPanel {...base} t={dict} value={{ ...ADVANCED_DEFAULTS, pure: true }} />);
  expect(screen.getByRole("switch", { name: dict.advAlt })).toHaveAccessibleDescription(dict.advPlainImages);
});

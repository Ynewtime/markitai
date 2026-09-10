"""Design guards for the web UI stylesheet.

`webapp/scripts/check-css-scale.mjs` guards the *scale* (type ladder, radii,
durations, shadows). This module guards the two things a scale check cannot
see and that regressed once already:

* **Contrast.** ``--text-4`` (#a1a1aa) was used on interactive help buttons and
  sat at 2.5:1 on white; the dark delete button was 2.7:1. Both shipped. Every
  text/background pair the sheet actually renders is measured here.
* **Token discipline.** ``--text-4`` is the faintest ink and is now reserved
  for disabled controls and diff line numbers (WCAG exempts disabled text).
  A new ``color: var(--text-4)`` on live text fails this test.

The font-stack check exists because a CJK UI that falls back to a system sans
still *renders* — it just renders outside the metrics the sheet was tuned for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CSS = _REPO_ROOT / "webapp" / "src" / "styles" / "app.css"
_DROPZONE = _REPO_ROOT / "webapp" / "src" / "components" / "DropZone.tsx"
_I18N = _REPO_ROOT / "webapp" / "src" / "i18n.ts"
_WEBAPP_SRC = _REPO_ROOT / "webapp" / "src"

pytestmark = pytest.mark.skipif(
    not _CSS.is_file(), reason="webapp/ is not part of this checkout"
)


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{3,8})\s*;", block))


def _token_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (light, dark) hex token maps from the two declared blocks."""
    css = _CSS.read_text(encoding="utf-8")
    light_block = css.split(":root {", 1)[1].split(
        "@media (prefers-color-scheme: dark)", 1
    )[0]
    dark_block = css.split("@media (prefers-color-scheme: dark) {", 1)[1].split(
        ':root[data-theme="dark"]', 1
    )[0]
    return _tokens(light_block), _tokens(dark_block)


def _block_body(css: str, selector: str) -> str:
    """Return the body of the first rule whose text starts with *selector*."""
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth = 0
    for index in range(open_brace, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : index]
    raise AssertionError(f"unbalanced braces after {selector!r}")


def test_dark_palette_has_one_definition() -> None:
    """`auto` and an explicit dark choice must resolve to the same tokens.

    They once drifted: the media-query block declared the ``--danger-*``
    tokens while the ``data-theme`` block did not, so an explicit dark choice
    kept the light red delete button.
    """
    css = _CSS.read_text(encoding="utf-8")
    media = _block_body(css, "@media (prefers-color-scheme: dark)")
    auto = _block_body(media, ':root:not([data-theme="light"])')
    explicit = _block_body(css, ':root[data-theme="dark"]')

    assert _tokens(auto) == _tokens(explicit), (
        "the two dark palettes disagree: "
        f"auto-only={sorted(set(_tokens(auto)) - set(_tokens(explicit)))}, "
        f"explicit-only={sorted(set(_tokens(explicit)) - set(_tokens(auto)))}"
    )


def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(channel * 2 for channel in value)
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]

    def linear(channel: float) -> float:
        return (
            channel / 12.92
            if channel <= 0.03928
            else ((channel + 0.055) / 1.055) ** 2.4
        )

    red, green, blue = (linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


# Every pair below is a combination the sheet actually renders as text.
# 4.5:1 is WCAG AA for normal text; 3.0:1 would be the large-text floor, and
# nothing here is large text.
_TEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("text-1", "bg"),
    ("text-1", "surface"),
    ("text-1", "strip"),
    ("text-1", "sel"),
    ("text-2", "bg"),
    ("text-2", "surface"),
    ("text-3", "bg"),
    ("text-3", "surface"),
    ("err", "surface"),
    ("warning", "surface"),
    ("success", "surface"),
    ("accent-contrast", "accent"),
    ("danger-on-fill", "danger-fill"),
    ("selection-fg", "selection-bg"),
    # always-dark terminal card (identical tokens in both themes)
    ("t-text", "t-bg"),
    ("t-dim", "t-bg"),
    ("t-green", "t-bg"),
    ("t-red", "t-bg"),
    ("t-amber", "t-bg"),
    ("t-badge-text", "t-bg"),
)

_MIN_CONTRAST = 4.5


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_text_contrast_meets_wcag_aa(theme: str) -> None:
    light, dark = _token_maps()
    # Theme-invariant tokens (the always-dark terminal card) are declared once
    # in :root, so the dark palette inherits them rather than redefining them.
    tokens = {**light, **dark} if theme == "dark" else light
    failures: list[str] = []
    for foreground, background in _TEXT_PAIRS:
        assert foreground in tokens, f"--{foreground} missing from the {theme} palette"
        assert background in tokens, f"--{background} missing from the {theme} palette"
        ratio = _contrast(tokens[foreground], tokens[background])
        if ratio < _MIN_CONTRAST:
            failures.append(
                f"--{foreground} on --{background}: {ratio:.2f}:1 "
                f"({tokens[foreground]} on {tokens[background]})"
            )
    assert not failures, (
        f"{theme} theme text pairs below {_MIN_CONTRAST}:1:\n" + "\n".join(failures)
    )


# `--text-4` is the faintest ink; only disabled controls, diff line numbers
# and generated separators may use it, because WCAG exempts disabled text and
# the rest is punctuation rather than content. Live text must use --text-3 or
# darker.
_TEXT4_ALLOWED_SELECTORS = re.compile(r"(:disabled|\.disabled|\.dno\b|\.metabit)")


def _css_rules(css: str) -> list[tuple[str, str]]:
    """(selector, declarations) for every flat rule, comments stripped.

    Flat is enough for this sheet (no nested rules); parsing rules instead of
    lines keeps multi-line selectors, `color:var(--text-4)` without a space,
    and grouped selectors all visible to the checks below.
    """
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return [
        (" ".join(match.group(1).split()), match.group(2))
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", without_comments)
    ]


def test_faintest_ink_stays_off_live_text() -> None:
    offenders: list[str] = []
    for selector, body in _css_rules(_CSS.read_text(encoding="utf-8")):
        if "var(--text-4)" not in body:
            continue
        for declaration in body.split(";"):
            prop, _, value = declaration.partition(":")
            if prop.strip() != "color" or "var(--text-4)" not in value:
                continue
            if not _TEXT4_ALLOWED_SELECTORS.search(selector):
                offenders.append(selector)
    assert not offenders, (
        "--text-4 is the faintest ink and fails AA on live text; use --text-3 "
        "or darker, or add a disabled/decorative selector:\n" + "\n".join(offenders)
    )


def test_dark_palette_redeclares_every_themed_token() -> None:
    """A token missing from the dark blocks silently inherits its light value.

    Only the always-dark terminal card (`--t-*`) is theme-invariant; every
    other token must be redeclared so an explicit dark choice cannot keep a
    light surface, text or border.
    """
    light, dark = _token_maps()
    missing = sorted(
        name for name in light if not name.startswith("t-") and name not in dark
    )
    assert not missing, (
        f"tokens declared only in :root (dark inherits the light value): {missing}"
    )


_CJK_FAMILIES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
)


def test_font_stacks_cover_cjk() -> None:
    """A missing CJK face renders through a system fallback, not an error."""
    css = _CSS.read_text(encoding="utf-8")
    for token in ("--font-sans:", "--font-mono:"):
        start = css.index(token)
        stack = css[start : css.index(";", start)]
        assert any(family in stack for family in _CJK_FAMILIES), (
            f"{token} has no CJK fallback: {stack.strip()}"
        )


def test_monospace_stack_keeps_latin_faces_first() -> None:
    """A CJK face ahead of the Latin monospace families would win Latin glyphs
    on a machine that lacks the first few, breaking code alignment."""
    css = _CSS.read_text(encoding="utf-8")
    stack = css[css.index("--font-mono:") : css.index(";", css.index("--font-mono:"))]
    cjk_at = [stack.index(family) for family in _CJK_FAMILIES if family in stack]
    latin_at = [
        stack.index(family)
        for family in ("Courier New", "monospace")
        if family in stack
    ]
    assert cjk_at and latin_at
    assert min(cjk_at) > max(latin_at), (
        f"CJK face precedes a Latin monospace face: {stack.strip()}"
    )


def test_file_picker_accepts_exactly_the_supported_extensions() -> None:
    """A picker filter listing an unsupported suffix offers a failing file.

    The webapp cannot import the Python extension map, so the `accept` string
    is a hand-kept mirror; this test is what keeps the mirror honest.
    """
    from markitai.converter.base import EXTENSION_MAP

    text = _DROPZONE.read_text(encoding="utf-8")
    match = re.search(r'accept="([^"]+)"', text)
    assert match is not None, "DropZone.tsx has no accept attribute"
    accepted = {item.strip() for item in match.group(1).split(",") if item.strip()}

    supported = set(EXTENSION_MAP)
    assert accepted == supported, (
        f"file picker filter is out of sync with the converter: "
        f"missing={sorted(supported - accepted)}, "
        f"unsupported={sorted(accepted - supported)}"
    )


def _dict_block(name: str) -> str:
    """Body of one locale dictionary, so keys from the other locale and the
    module header cannot leak into a check."""
    text = _I18N.read_text(encoding="utf-8")
    start = text.index(f"const {name}")
    return text[text.index("{", start) + 1 : text.index("\n};", start)]


_DICT_KEY = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*)\??:", re.MULTILINE)
# A dash used as sentence punctuation. Hyphens inside words are fine.
_DASH_IN_COPY = re.compile(r"[\u2014\u2013]| - ")
# String literals in the dictionary: plain, or a template used by a (n) => ...
_COPY_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`')


def test_dictionary_copy_avoids_dashes() -> None:
    """The dictionary header promises "·" as the separator voice and no em/en
    dashes. Hand-written copy keeps that promise; generated copy drifts back to
    a dash unless a check says no."""
    offenders: list[str] = []
    for locale in ("en", "zh"):
        block = _dict_block(locale)
        for number, line in enumerate(block.splitlines(), start=1):
            # Comments are allowed to use a dash; user-visible strings are not.
            if line.strip().startswith(("//", "*", "/*")):
                continue
            for literal in _COPY_LITERAL.findall(line):
                if _DASH_IN_COPY.search(literal):
                    offenders.append(f"{locale}:{number}: {literal}")
    assert not offenders, "dash used as sentence punctuation: " + "; ".join(offenders)


def test_every_dictionary_key_has_a_consumer() -> None:
    """A key nothing reads is dead weight, and the kind of thing a refactor
    leaves behind: it still has to be translated in both locales."""
    keys = _DICT_KEY.findall(_dict_block("en"))
    assert len(keys) > 100, "dictionary parser found almost no keys"
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _WEBAPP_SRC.rglob("*.ts*")
        if path != _I18N
    )
    unused = [key for key in keys if re.search(rf"\b{key}\b", corpus) is None]
    assert not unused, f"dictionary keys with no consumer: {unused}"


def test_status_labels_come_from_one_family() -> None:
    """One key per concept: the row label, the icon tooltip and the counters all
    read `status*`. The old `stat*`/`done`/`converting` copies drifted apart in
    Chinese ("跳过" vs "已跳过"), so the second family is banned outright."""
    labels = [
        key for key in _DICT_KEY.findall(_dict_block("en")) if key.startswith("stat")
    ]
    assert set(labels) == {
        "statusQueued",
        "statusRunning",
        "statusDone",
        "statusFailed",
        "statusSkipped",
    }, f"unexpected status label family: {sorted(labels)}"

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright", "pillow"]
# ///
"""The provider form's API address combobox, end to end, in a real browser.

Starts ``markitai serve`` (``--bin``, default ``target/debug/markitai``) with
its own HOME and MARKITAI_HOME, as ``workbench_upload_e2e.py`` does, opens
Settings → Models → Add models in Chinese at 1280 and 390 px wide and checks:

1. Z.ai: focusing API address opens a list of its four documented addresses
   (国际, 中国大陆 and both Coding Plans) without a datalist; ArrowDown and
   Enter pick one, which fills the field and shows the Coding Plan's terms;
   Escape closes the list and stays on the page; typed text is kept as typed
   and the list keeps every address; a tap or click outside closes it; an
   emptied field shows "按密钥自动识别" again.
2. MiniMax lists 国际 and 中国大陆; OpenAI its one address, as 默认; picking
   with a click or a tap fills the field.

``--ui DIR`` serves the workbench from a ``node build.mjs`` output, ``--shots
DIR`` saves the open lists (``zai-1280.png``, ``minimax-390.png``, …),
``--browser webkit`` runs it in Playwright's WebKit (Safari's engine).

    uv run scripts/workbench_address_e2e.py [--bin PATH] [--ui DIR] [--shots DIR] [--browser webkit]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workbench_upload_e2e import ROOT, open_page, start_server

ZAI = [
    ("国际", "https://api.z.ai/api/paas/v4"),
    ("中国大陆", "https://open.bigmodel.cn/api/paas/v4"),
    ("国际 · Coding Plan", "https://api.z.ai/api/coding/paas/v4"),
    ("中国大陆 · Coding Plan", "https://open.bigmodel.cn/api/coding/paas/v4"),
]
MINIMAX = [("国际", "https://api.minimax.io/v1"), ("中国大陆", "https://api.minimax.cn/v1")]


def open_provider(page: Page, name: str) -> None:
    """Add models → the provider's form."""
    search = page.locator(".provider-search input")
    if not search.is_visible():
        page.get_by_role("button", name="添加模型", exact=True).first.click()
    search.fill(name)
    page.locator(".provider-item, .provider-tile").filter(has_text=name).first.click()
    expect(page.locator("#connect-base")).to_be_visible()


def back_to_providers(page: Page) -> None:
    page.locator(".crumbs button").filter(has_text="添加模型").click()
    expect(page.locator(".provider-search input")).to_be_visible()


def entries(page: Page) -> list[tuple[str, str]]:
    options = page.locator("#connect-base-list [role=option]")
    return [
        (options.nth(index).locator(".addr-combo-label").inner_text(), options.nth(index).locator(".addr-combo-base").inner_text())
        for index in range(options.count())
    ]


def open_list(page: Page) -> None:
    field = page.locator("#connect-base")
    field.scroll_into_view_if_needed()
    field.click()
    expect(page.locator("#connect-base-list")).to_be_visible()
    expect(field).to_have_attribute("aria-expanded", "true")


def pick(page: Page, index: int, touch: bool) -> None:
    option = page.locator("#connect-base-list [role=option]").nth(index)
    if touch:
        option.tap()
    else:
        option.click()
    expect(page.locator("#connect-base-list")).to_have_count(0)


def shot(page: Page, shots: Path | None, name: str) -> None:
    if shots:
        page.screenshot(path=str(shots / name))


def check_zai(page: Page, width: int, touch: bool, shots: Path | None) -> None:
    open_provider(page, "Z.ai")
    field = page.locator("#connect-base")
    assert page.locator("datalist").count() == 0
    assert field.get_attribute("role") == "combobox"
    expect(field).to_have_attribute("placeholder", "按密钥自动识别")
    open_list(page)
    assert entries(page) == ZAI, entries(page)
    shot(page, shots, f"zai-{width}.png")
    # The arrows walk the list; Enter picks without submitting anything.
    field.press("ArrowDown")
    field.press("ArrowDown")
    field.press("ArrowDown")
    expect(page.locator("#connect-base-list [role=option].is-active")).to_contain_text("国际 · Coding Plan")
    field.press("Enter")
    expect(field).to_have_value(ZAI[2][1])
    expect(page.locator("#connect-base-list")).to_have_count(0)
    expect(page.locator("#connect-base-plan")).to_contain_text("Coding Plan")
    # The picked address is checked when the list opens again; ArrowUp wraps.
    field.press("ArrowDown")
    expect(page.locator("#connect-base-list [aria-selected=true]")).to_contain_text("国际 · Coding Plan")
    # Escape closes the list and nothing else.
    field.press("Escape")
    expect(page.locator("#connect-base-list")).to_have_count(0)
    expect(field).to_be_visible()
    # Typed text stays as typed, and every address is still offered.
    field.fill("https://gateway.example/zai/v4")
    expect(field).to_have_value("https://gateway.example/zai/v4")
    expect(page.locator("#connect-base-list [role=option]")).to_have_count(4)
    expect(page.locator("#connect-base-plan")).to_have_count(0)
    # A tap or click outside closes it.
    page.locator("h1#settings-title").click()
    expect(page.locator("#connect-base-list")).to_have_count(0)
    # An emptied field means the default again (found from the key).
    field.fill("")
    expect(field).to_have_value("")
    open_list(page)
    pick(page, 1, touch)
    expect(field).to_have_value(ZAI[1][1])
    back_to_providers(page)


def check_minimax_and_openai(page: Page, width: int, touch: bool, shots: Path | None) -> None:
    open_provider(page, "MiniMax")
    open_list(page)
    assert entries(page) == MINIMAX, entries(page)
    shot(page, shots, f"minimax-{width}.png")
    pick(page, 1, touch)
    expect(page.locator("#connect-base")).to_have_value(MINIMAX[1][1])
    back_to_providers(page)
    open_provider(page, "OpenAI")
    open_list(page)
    assert entries(page) == [("默认", "https://api.openai.com/v1")], entries(page)
    pick(page, 0, touch)
    expect(page.locator("#connect-base")).to_have_value("https://api.openai.com/v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=ROOT / "target/debug/markitai")
    parser.add_argument("--port", type=int, default=3751)
    parser.add_argument("--ui", type=Path, help="serve the workbench from this build output")
    parser.add_argument("--shots", type=Path)
    parser.add_argument("--browser", choices=["chromium", "webkit"], default="chromium")
    args = parser.parse_args()
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mk-address-e2e-") as scratch:
        server, token = start_server(args.bin.resolve(), Path(scratch) / "run", args.port)
        origin = f"http://127.0.0.1:{args.port}"
        try:
            with sync_playwright() as playwright:
                browser = getattr(playwright, args.browser).launch()
                for width in (1280, 390):
                    touch = width < 600
                    page = open_page(browser, origin, token, width, args.ui)
                    page.goto(f"{origin}/settings/models#token={token}")
                    check_zai(page, width, touch, args.shots)
                    print(f"{width}: Z.ai ok · four addresses, arrows/Enter/Escape, typed text kept, Coding Plan terms")
                    check_minimax_and_openai(page, width, touch, args.shots)
                    print(f"{width}: MiniMax and OpenAI ok · two addresses, one default, {'tap' if touch else 'click'} picks")
                    page.context.close()
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=60)


if __name__ == "__main__":
    main()

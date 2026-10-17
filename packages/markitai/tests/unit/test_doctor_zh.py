"""The doctor report under a Chinese terminal; check names and --json stay English."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import i18n
from markitai.cli.commands.doctor import doctor
from markitai.cli.commands.doctor_text import zh_hint, zh_hint_line, zh_message


@pytest.fixture
def zh() -> Iterator[None]:
    before = i18n._lang
    i18n.set_language("zh")
    yield
    i18n._lang = before


@pytest.mark.parametrize(
    ("english", "chinese"),
    [
        (
            "v3.9.2, reads Chinese, English, Japanese and Latin-script text "
            "(ocr.lang: en)",
            "v3.9.2，可识别中文、英文、日文和拉丁字母文字（ocr.lang：en）",
        ),
        (
            "v3.9.2, reads Korean text (ocr.lang: ko)",
            "v3.9.2，可识别韩文文字（ocr.lang：ko）",
        ),
        (
            "installed, but its Chromium browser is not downloaded",
            "已安装，但尚未下载其 Chromium 浏览器",
        ),
        (
            "Chromium is installed but cannot launch: exit code 1",
            "Chromium 已安装，但无法启动：exit code 1",
        ),
        ("installed", "已安装"),
        ("Authenticated (octocat)", "已认证 (octocat)"),
        (
            "not set: OPENAI_API_KEY, GEMINI_API_KEY; 2 models left out, 1 still used",
            "未设置：OPENAI_API_KEY、GEMINI_API_KEY；2 个模型被忽略，仍有 1 个在用",
        ),
        (
            "not set: OPENAI_API_KEY; no model can be used without it",
            "未设置：OPENAI_API_KEY；没有它，任何模型都无法使用",
        ),
        ("2 active model(s): OpenAI, Gemini", "2 个生效模型：OpenAI, Gemini"),
        (
            "3 detected: a/x, b/y (+1 more)",
            "检测到 3 个：a/x, b/y（另有 1 个）",
        ),
        (
            "--ocr --llm sends page images to 1 vision model "
            "(MARKITAI_NO_VLM_OCR=1 keeps RapidOCR)",
            "--ocr --llm 会把页面图片发给 1 个视觉模型（MARKITAI_NO_VLM_OCR=1 则仍用 RapidOCR）",
        ),
        ("No access token found", "没有找到访问令牌"),
        (
            "a sentence nobody wrote a translation for",
            "a sentence nobody wrote a translation for",
        ),
    ],
    ids=lambda value: str(value)[:30],
)
def test_check_messages(english: str, chinese: str) -> None:
    assert zh_message(english) == chinese


def test_remedies_keep_commands_and_translate_prose() -> None:
    assert (
        zh_hint_line("export OPENAI_API_KEY=your_key")
        == "export OPENAI_API_KEY=your_key"
    )
    assert (
        zh_hint_line("# or add them to ~/.markitai/.env")
        == "# 或把它们写入 ~/.markitai/.env"
    )
    assert (
        zh_hint_line(
            "Run 'markitai init' to configure models for OPENAI_API_KEY (it adds them to /c.json)"
        )
        == "运行 'markitai init'，为 OPENAI_API_KEY 配置模型（会写入 /c.json）"
    )
    hint = (
        "Run 'markitai auth copilot login' (or 'copilot login') to authenticate "
        "with GitHub Copilot.\nIf Copilot CLI is not installed, install it with:\n"
        "  winget install GitHub.Copilot"
    )
    assert zh_hint(hint).splitlines() == [
        "运行 'markitai auth copilot login'（或 'copilot login'）完成登录。",
        "未安装 Copilot 命令行时，用下面的命令安装：",
        "  winget install GitHub.Copilot",
    ]


def _run(*args: str) -> str:
    config = MagicMock()
    config.llm.model_list = []
    config.ocr = MagicMock()
    config.ocr.lang = "en"
    with (
        patch("markitai.cli.commands.doctor.ConfigManager") as manager,
        patch("markitai.fetch_playwright.is_playwright_available", return_value=True),
        patch(
            "markitai.fetch_playwright.is_playwright_browser_installed",
            return_value=False,
        ),
        patch("markitai.fetch_playwright.clear_browser_cache"),
        patch("markitai.cli.commands.doctor.shutil.which", return_value=None),
        patch("markitai.utils.office.find_libreoffice", return_value=None),
    ):
        manager.return_value.load.return_value = config
        return CliRunner().invoke(doctor, list(args)).output


def test_the_terminal_report_is_chinese_with_english_names(zh: None) -> None:
    output = _run()
    assert "系统检查" in output
    assert "Playwright：已安装，但尚未下载其 Chromium 浏览器" in output
    assert "Playwright: installed" not in output


def test_json_stays_english_under_a_chinese_terminal(zh: None) -> None:
    results = json.loads(_run("--json"))
    assert results["playwright"]["name"] == "Playwright"
    assert (
        results["playwright"]["message"]
        == "installed, but its Chromium browser is not downloaded"
    )

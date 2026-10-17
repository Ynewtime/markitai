"""Chinese wording of the doctor report's check messages and remedies.

The checks build English sentences: ``--json`` carries them as they are, and
so does the English terminal. Under a Chinese terminal the report is shown in
Chinese by matching those sentences here, so a check and its JSON have one
source of truth. Check names, commands, environment variables and file paths
stay as they are; a sentence with no entry is shown unchanged.
"""

from __future__ import annotations

import re

_LANGUAGES = {
    "Chinese": "中文",
    "Traditional Chinese": "繁体中文",
    "English": "英文",
    "Japanese": "日文",
    "Korean": "韩文",
    "Arabic": "阿拉伯文",
    "Thai": "泰文",
    "Latin": "拉丁字母",
    "Chinese, English, Japanese and Latin-script": "中文、英文、日文和拉丁字母",
}

_MODELS = re.compile(r"(\d+) (LLM model|vision model|model)s?")


def _models(text: str) -> str:
    """``2 vision models`` -> ``2 个视觉模型``."""
    nouns = {"LLM model": " LLM 模型", "vision model": "视觉模型", "model": "模型"}
    return _MODELS.sub(lambda m: f"{m.group(1)} 个{nouns[m.group(2)]}", text)


#: (English sentence as a pattern, Chinese template with the same groups)
_MESSAGES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.DOTALL), zh)
    for p, zh in [
        (r"Chromium installed", "Chromium 已安装"),
        (
            r"installed, but its Chromium browser is not downloaded",
            "已安装，但尚未下载其 Chromium 浏览器",
        ),
        (
            r"not installed \(needed for JavaScript-heavy pages\)",
            "未安装（渲染依赖 JavaScript 的页面需要它）",
        ),
        (
            r"Chromium is installed but slow to start \(no headless launch within "
            r"(\d+) s\); a first launch after installing can be that slow while "
            r"antivirus scans it, so run doctor again",
            "Chromium 已安装，但启动很慢（{0} 秒内未能无头启动）；安装后的首次启动"
            "可能因杀毒软件扫描而这么慢，请再运行一次 doctor",
        ),
        (
            r"Chromium is installed but cannot launch: headless launch failed: (.*)",
            "Chromium 已安装，但无法启动：无头启动失败：{0}",
        ),
        (
            r"Chromium is installed but cannot launch: (.*)",
            "Chromium 已安装，但无法启动：{0}",
        ),
        (r"fastapi \+ uvicorn installed", "已安装 fastapi + uvicorn"),
        (r"serve extra not installed", "未安装 serve 可选依赖"),
        (r"installed", "已安装"),
        (
            r"not found; PowerPoint fallback blocked: (.*)",
            "未找到；PowerPoint 备用方案被阻止：{0}",
        ),
        (
            r"not found; PowerPoint fallback available \(via AppleScript, needs "
            r"one-time Automation permission\)",
            "未找到；可改用 PowerPoint（通过 AppleScript，需要一次性的“自动化”授权）",
        ),
        (
            r"not found \(needed to render PPTX slides as images\)",
            "未找到（将 PPTX 幻灯片渲染为图片需要它）",
        ),
        (
            r"not installed \(needed for legacy \.doc and \.ppt files\)",
            "未安装（读取旧版 .doc 和 .ppt 文件需要它）",
        ),
        (
            r"v([^,]+), reads (.+) text \(ocr\.lang: (.+)\)",
            "v{0}，可识别{1}文字（ocr.lang：{2}）",
        ),
        (
            r"v([^,]+), Unsupported ocr\.lang (.+?): use en, zh, zh_tw, ja, ko, ar, "
            r"th, latin, or a code such as fr/de/es \(see the OCR configuration "
            r"docs\)",
            "v{0}，不支持的 ocr.lang {1}：请使用 en、zh、zh_tw、ja、ko、ar、th、"
            "latin，或 fr/de/es 这类代码（见 OCR 配置文档）",
        ),
        (r"v([^,]+), (.+)", "v{0}，{1}"),
        (
            r"not installed: --ocr unavailable \(everything else works\)",
            "未安装：--ocr 不可用（其余功能不受影响）",
        ),
        (r"SDK \+ CLI installed", "SDK 与命令行均已安装"),
        (
            r"SDK installed but '(\w+)' CLI not found",
            "SDK 已安装，但未找到 '{0}' 命令行",
        ),
        (r"([\w-]+-sdk) not installed", "未安装 {0}"),
        (r"Check failed: (.*)", "检查失败：{0}"),
        (r"Authenticated( \(.*\))?", "已认证{0}"),
        (r"Not authenticated", "未认证"),
        (r"Failed to check auth: (.*)", "检查认证状态失败：{0}"),
        (
            r"Config file not found \((.*)\)",
            "未找到配置文件（{0}）",
        ),
        (
            r"Cannot determine login state: could not parse (.*)",
            "无法判断登录状态：无法解析 {0}",
        ),
        (r"No logged in users found", "没有已登录的用户"),
        (
            r"Credentials file not found \((.*)\)",
            "未找到凭据文件（{0}）",
        ),
        (r"Auth file not found \((.*)\)", "未找到认证文件（{0}）"),
        (r"Token expired \(no refresh token\)", "令牌已过期（没有刷新令牌）"),
        (r"No access token found", "没有找到访问令牌"),
        (r"Failed to read (credentials|auth file): (.*)", "读取{0}失败：{1}"),
        (
            r"not signed in; Claude Code would use ANTHROPIC_API_KEY, billed per token",
            "未登录；Claude Code 会改用 ANTHROPIC_API_KEY，按 token 计费",
        ),
        (
            r"signed in with a Claude subscription( \(.*?\))?"
            r"(, used instead of ANTHROPIC_API_KEY)?",
            "已用 Claude 订阅登录{0}{1}",
        ),
        (r"signed in \((.*)\), billed per token", "已登录（{0}），按 token 计费"),
        (r"signed in \((.*)\)", "已登录（{0}）"),
        (
            r"Models are configured, but all have weight 0 \(disabled\)",
            "已配置模型，但权重全为 0（已禁用）",
        ),
        (
            r"not set: (.+?); (\d+ (?:LLM )?models?) left out, (\d+) still used",
            "未设置：{0}；{1}被忽略，仍有 {2} 个在用",
        ),
        (
            r"not set: (.+?); no model can be used without (it|them)",
            "未设置：{0}；没有{1}，任何模型都无法使用",
        ),
        (r"(\d+) active model\(s\): (.*)", "{0} 个生效模型：{1}"),
        (
            r"no models in llm\.model_list; --llm picks from the keys it finds "
            r"\((.*)\)",
            "llm.model_list 中没有模型；--llm 会从找到的密钥中选用（{0}）",
        ),
        (
            r"no models in llm\.model_list and no provider key set",
            "llm.model_list 中没有模型，也没有设置服务商密钥",
        ),
        (r"(\d+) detected: (.*)", "检测到 {0} 个：{1}"),
        (
            r"none of the active models reads images \(or set "
            r"model_info\.supports_vision\)",
            "生效的模型都不能读图（或设置 model_info.supports_vision）",
        ),
        (r"needs an LLM model \(see LLM API\)", "需要一个 LLM 模型（见 LLM API）"),
        (
            r"--ocr --llm sends page images to (\d+ vision models?) "
            r"\(MARKITAI_NO_VLM_OCR=1 keeps RapidOCR\)",
            "--ocr --llm 会把页面图片发给 {0}（MARKITAI_NO_VLM_OCR=1 则仍用 RapidOCR）",
        ),
        (
            r"the vision models are left out until (.+) is set \(see LLM API\)",
            "视觉模型被忽略，直到设置 {0}（见 LLM API）",
        ),
        (
            r"unavailable: needs a vision model \(see Vision Model\)",
            "不可用：需要视觉模型（见 Vision Model）",
        ),
    ]
]

_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.DOTALL), zh)
    for p, zh in [
        (
            r"Run 'markitai doctor' again in a minute",
            "一分钟后再运行 'markitai doctor'",
        ),
        (
            r"Set ocr\.lang to one of: (.*)",
            "把 ocr.lang 设为以下之一：{0}",
        ),
        (
            r"Set weight > 0 on at least one model to enable LLM use",
            "至少给一个模型设置 weight > 0，才能使用 LLM",
        ),
        (r"(.*) \(or add to ~/\.markitai/\.env\)", "{0}（或写入 ~/.markitai/.env）"),
        (
            r"Run 'markitai init' to configure models for (.+?)( \(it adds them to .*\))?",
            "运行 'markitai init'，为 {0} 配置模型{1}",
        ),
        (
            r"Set a provider API key \((.*?)\) or sign in to Claude Code, then run "
            r"'markitai init'( \(it adds them to .*?\))?; or configure "
            r"llm\.model_list by hand",
            "设置服务商 API 密钥（{0}）或登录 Claude Code，再运行 'markitai init'{1}；"
            "也可以手动配置 llm.model_list",
        ),
        (r"Install Claude Code CLI: (.*)", "安装 Claude Code 命令行：{0}"),
        (r"Install Copilot CLI: (.*)", "安装 Copilot 命令行：{0}"),
        (
            r"Use vision-capable models like (.*)",
            "请使用能读图的模型，例如 {0}",
        ),
        (
            r"Run 'claude login' to use a Claude subscription, or use "
            r"(\S+) \(the same key, directly\)",
            "运行 'claude login' 以使用 Claude 订阅，或改用 {0}（同一把密钥，直接调用）",
        ),
        (
            r"Run 'markitai auth (\w+) login' \(or '([\w ]+)'\) to authenticate "
            r"with (Claude Code CLI|GitHub Copilot)\.",
            "运行 'markitai auth {0} login'（或 '{1}'）完成登录。",
        ),
        (
            r"Alternatively, set a cloud provider env var:",
            "或者设置云服务商的环境变量：",
        ),
        (
            r"Alternatively, set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN "
            r"env var \(requires 'Copilot Requests' permission\)\.",
            "或者设置环境变量 COPILOT_GITHUB_TOKEN、GH_TOKEN 或 GITHUB_TOKEN"
            "（需要 'Copilot Requests' 权限）。",
        ),
        (
            r"If Claude Code CLI is not installed, install it with:",
            "未安装 Claude Code 命令行时，用下面的命令安装：",
        ),
        (
            r"If Copilot CLI is not installed, install it with:",
            "未安装 Copilot 命令行时，用下面的命令安装：",
        ),
        (
            r"Run 'markitai auth chatgpt login' to authenticate via the OAuth "
            r"Device Code Flow \(visit the URL, enter the code\)\.",
            "运行 'markitai auth chatgpt login'，通过 OAuth 设备码流程登录"
            "（打开网址，输入验证码）。",
        ),
        (
            r"Login is also triggered automatically on the first chatgpt/ model call\.",
            "第一次调用 chatgpt/ 模型时也会自动触发登录。",
        ),
        (r"# or add (.+) to ~/\.markitai/\.env", "# 或把{0}写入 ~/.markitai/.env"),
        (r"(.*)  # or: (.*)", "{0}  # 或：{1}"),
        (
            r"  CLAUDE_CODE_USE_BEDROCK=1, CLAUDE_CODE_USE_VERTEX=1, or "
            r"CLAUDE_CODE_USE_FOUNDRY=1",
            "  CLAUDE_CODE_USE_BEDROCK=1、CLAUDE_CODE_USE_VERTEX=1 或 "
            "CLAUDE_CODE_USE_FOUNDRY=1",
        ),
    ]
]


def _translate(table: list[tuple[re.Pattern[str], str]], text: str) -> str:
    for pattern, zh in table:
        match = pattern.fullmatch(text)
        if match is not None:
            groups = [group or "" for group in match.groups()]
            return _fill(zh, groups)
    return text


_TARGET = re.compile(r" \(it adds them to (.*)\)")

_WORDS = {
    ", used instead of ANTHROPIC_API_KEY": "，优先于 ANTHROPIC_API_KEY 使用",
    "it": "它",
    "them": "它们",
    "credentials": "凭据",
    "auth file": "认证文件",
}

_NAMES = re.compile(r"[A-Z][A-Z0-9_]+(?:, [A-Z][A-Z0-9_]+)+(?:, \.\.\.)?")
_MORE = re.compile(r"(.*) \(\+(\d+) more\)")


_PAREN_TAIL = re.compile(r" \((.*)\)")


def _group(group: str) -> str:
    """One captured piece: a language, a model count, a tail or a plain word."""
    target = _TARGET.fullmatch(group)
    if target is not None:
        return f"（会写入 {target.group(1)}）"
    parenthesized = _PAREN_TAIL.fullmatch(group)
    if parenthesized is not None:
        return f"（{_group(parenthesized.group(1))}）"
    if group in _WORDS:
        return _WORDS[group]
    if group in _LANGUAGES:
        return _LANGUAGES[group]
    if _NAMES.fullmatch(group):
        return group.replace(", ...", "等").replace(", ", "、")
    more = _MORE.fullmatch(group)
    if more is not None:
        return f"{more.group(1)}（另有 {more.group(2)} 个）"
    target = _TARGET.fullmatch(group)
    if target is not None:
        return f"（会写入 {target.group(1)}）"
    return _models(group)


def _fill(template: str, groups: list[str]) -> str:
    return template.format(*[_group(group) for group in groups])


_SUMMARY_NAMES = {"Playwright (slow to start)": "Playwright（启动慢）"}


def zh_summary_name(name: str) -> str:
    """A name in the "not available" summary: ``1 LLM model`` -> ``1 个 LLM 模型``."""
    return _SUMMARY_NAMES.get(name) or _models(name)


def zh_message(message: str) -> str:
    """The Chinese form of a check message (unchanged when there is none)."""
    return _translate(_MESSAGES, message)


def zh_hint_line(line: str) -> str:
    """The Chinese form of one remedy line (commands stay as they are)."""
    return _translate(_HINTS, line)


def zh_hint(hint: str) -> str:
    """A remedy, line by line."""
    return "\n".join(zh_hint_line(line) for line in hint.splitlines())

"""Internationalization (i18n) module for Markitai CLI.

This module provides language detection and translation functionality
for multi-language support in the CLI interface.

Usage:
    from markitai.cli.i18n import t, set_language

    # Get translated text
    print(t("success"))  # "completed" or "完成"

    # Override language
    set_language("zh")
"""

from __future__ import annotations

import os

# Module-level cache for language setting
_lang: str | None = None

# Translation texts
TEXTS: dict[str, dict[str, str]] = {
    # Common
    "success": {"en": "completed", "zh": "完成"},
    "failed": {"en": "failed", "zh": "失败"},
    "warning": {"en": "warning", "zh": "警告"},
    "error": {"en": "error", "zh": "错误"},
    "enabled": {"en": "Enabled", "zh": "已启用"},
    "disabled": {"en": "Disabled", "zh": "已禁用"},
    "not_found": {"en": "not found", "zh": "未找到"},
    "installed": {"en": "installed", "zh": "已安装"},
    "missing": {"en": "missing", "zh": "缺失"},
    "total": {"en": "Total", "zh": "总计"},
    # Doctor command
    "doctor.title": {"en": "System Check", "zh": "系统检查"},
    "doctor.required": {"en": "Required Dependencies", "zh": "必需依赖"},
    "doctor.optional": {"en": "Optional Capabilities", "zh": "可选能力"},
    "doctor.auth": {"en": "Authentication", "zh": "认证状态"},
    "doctor.summary": {
        "en": "Core check passed ({passed} required/configured checks passed, {degraded} non-blocking warnings)",
        "zh": "核心检查通过（{passed} 项必需或已配置检查通过，{degraded} 项非阻断警告）",
    },
    "doctor.summary_failed": {
        "en": "Health check failed ({failed} required/configured checks failed, {passed} passed, {degraded} non-blocking warnings)",
        "zh": "健康检查未通过（{failed} 项必需或已配置检查失败，{passed} 项通过，{degraded} 项非阻断警告）",
    },
    "doctor.summary_repair_failed": {
        "en": "Requested repair failed ({degraded} non-blocking warnings remain)",
        "zh": "请求的修复未成功（仍有 {degraded} 项非阻断警告）",
    },
    "doctor.all_good": {
        "en": "All checks passed; dependencies and capabilities configured correctly",
        "zh": "所有检查均通过，依赖与能力配置正确",
    },
    "doctor.fix_hint": {
        "en": "To resolve unavailable checks and capabilities:",
        "zh": "解决不可用的检查项与能力：",
    },
    "doctor.fix_attempting": {
        "en": "Attempting safe automatic repairs...",
        "zh": "正在尝试安全的自动修复……",
    },
    "doctor.fix_installing": {
        "en": "Installing {component}...",
        "zh": "正在安装 {component}……",
    },
    "doctor.fix_success": {
        "en": "{component} capability verified",
        "zh": "已验证 {component} 能力可用",
    },
    "doctor.fix_verification_failed": {
        "en": "{component} verification failed: {detail}",
        "zh": "{component} 验证失败：{detail}",
    },
    "doctor.fix_failed": {
        "en": "Repair failed: {detail}",
        "zh": "修复失败：{detail}",
    },
    "doctor.fix_error": {
        "en": "Repair error: {detail}",
        "zh": "修复出错：{detail}",
    },
    "doctor.playwright_package_manual": {
        "en": "Playwright is not installed in the Markitai environment; replace the isolated tool installation manually:",
        "zh": "Markitai 环境中未安装 Playwright；请手动替换隔离的工具安装：",
    },
    "doctor.manual_install": {
        "en": "Install this capability manually:",
        "zh": "请手动安装此能力：",
    },
    "doctor.config_source": {"en": "Config: {path}", "zh": "配置文件：{path}"},
    "doctor.config_defaults": {
        "en": "defaults (no config file found; create one with 'markitai init' at ~/.markitai/config.json or ./markitai.json)",
        "zh": "默认配置（未找到配置文件；可运行 'markitai init' 在 ~/.markitai/config.json 或 ./markitai.json 创建）",
    },
    # Cache command
    "cache.title": {"en": "Cache Statistics", "zh": "缓存统计"},
    "cache.llm": {"en": "LLM responses", "zh": "LLM 响应"},
    "cache.spa": {"en": "SPA domains", "zh": "SPA 域名"},
    "cache.proxy": {"en": "Proxy detection", "zh": "代理检测"},
    "cache.entries": {"en": "entries", "zh": "条"},
    "cache.cleared": {
        "en": "Cleared {count} cache entries",
        "zh": "已清理 {count} 条缓存",
    },
    "cache.no_entries": {"en": "No cache entries to clear", "zh": "无缓存可清理"},
    "cache.empty_hint": {
        "en": "No cache yet — it fills up as you convert with --llm",
        "zh": "暂无缓存——使用 --llm 转换时会自动累积",
    },
    # Config command
    "config.title": {"en": "Configuration Sources", "zh": "配置来源"},
    "config.cli_args": {"en": "CLI arguments", "zh": "命令行参数"},
    "config.env_vars": {"en": "Environment variables", "zh": "环境变量"},
    "config.local_file": {"en": "Local config file", "zh": "本地配置文件"},
    "config.user_file": {"en": "User config file", "zh": "用户配置文件"},
    "config.defaults": {"en": "Default values", "zh": "默认值"},
    "config.highest": {"en": "highest priority", "zh": "最高优先级"},
    "config.lowest": {"en": "lowest priority", "zh": "最低优先级"},
    "config.loaded": {"en": "loaded", "zh": "已加载"},
    "config.created": {"en": "Configuration file created", "zh": "配置文件已创建"},
    "config.valid": {"en": "Configuration is valid", "zh": "配置有效"},
}


def detect_language() -> str:
    """Detect user language preference from environment variables.

    Priority: MARKITAI_LANG > LANG/LC_ALL > default (en)

    Returns:
        Language code: "en" or "zh"
    """
    lang = os.environ.get("MARKITAI_LANG", "")
    if not lang:
        lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")

    if lang.lower().startswith("zh"):
        return "zh"
    return "en"


def get_language() -> str:
    """Get current language setting (cached).

    Returns:
        Language code: "en" or "zh"
    """
    global _lang
    if _lang is None:
        _lang = detect_language()
    return _lang


def set_language(lang: str) -> None:
    """Override language setting.

    Args:
        lang: Language code ("en" or "zh")
    """
    global _lang
    _lang = lang


def t(key: str, **kwargs: str | int) -> str:
    """Get translated text for a key.

    Args:
        key: Translation key (e.g., "doctor.summary")
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated text, or the key itself if not found

    Example:
        >>> t("doctor.summary", passed=3, degraded=1)
        "Check complete (3 required passed, 1 optional missing)"
    """
    lang = get_language()
    translations = TEXTS.get(key)

    if translations is None:
        return key

    text = translations.get(lang, translations.get("en", key))

    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            # Return unformatted text if format args don't match
            pass

    return text

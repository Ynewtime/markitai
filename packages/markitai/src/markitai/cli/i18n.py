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
    "doctor.optional": {"en": "Optional Dependencies", "zh": "可选依赖"},
    "doctor.auth": {"en": "Authentication", "zh": "认证状态"},
    "doctor.summary": {
        "en": "Check complete ({passed} required passed, {optional} optional missing)",
        "zh": "检查完成（{passed} 必需通过，{optional} 可选缺失）",
    },
    "doctor.all_good": {
        "en": "All dependencies configured correctly",
        "zh": "所有依赖配置正确",
    },
    "doctor.fix_hint": {"en": "To fix missing dependencies:", "zh": "修复缺失依赖："},
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
        >>> t("doctor.summary", passed=3, optional=1)
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

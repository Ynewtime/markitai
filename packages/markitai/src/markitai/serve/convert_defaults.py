"""The conversion options the workbench's composer starts from (contract v1.17).

The server's defaults are its configuration (``config.json`` over the built-in
defaults, the keys the CLI reads); a user token may keep its own over them.
The admin saves the server's into the configuration file, a user saves its own
into its models file. Cloudflare is never a default: it needs the caller's
confirmation in each request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from markitai.serve.errors import ApiError

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

Kind = Literal["flag", "inverted", "word", "language"]

#: The ``ocr.lang`` values the composer offers (contract v1.20), in its
#: order: ``en`` first (the default model: Latin, Chinese, Japanese), then
#: the languages that need their own RapidOCR recognizer. The same list as
#: markitai-rust's.
OCR_LANGUAGES = ("en", "zh", "zh_tw", "ja", "ko", "ru", "ar", "th", "hi", "el")


def ocr_languages(configured: str) -> list[str]:
    """The ``ocr_lang`` values a job may name: :data:`OCR_LANGUAGES`, and the
    configured ``ocr.lang`` as written when it is not one of them."""
    languages = list(OCR_LANGUAGES)
    if configured not in languages:
        languages.append(configured)
    return languages


#: The options that can be defaulted: name → (config path, kind, words, null
#: allowed). ``no_compress`` is ``image.compress`` turned around.
OPTIONS: dict[str, tuple[tuple[str, str], Kind, tuple[str, ...], bool]] = {
    "llm": (("llm", "enabled"), "flag", (), False),
    "ocr": (("ocr", "enabled"), "flag", (), False),
    "alt": (("image", "alt_enabled"), "flag", (), False),
    "desc": (("image", "desc_enabled"), "flag", (), False),
    "screenshot": (("screenshot", "enabled"), "flag", (), False),
    "screenshot_only": (("screenshot", "screenshot_only"), "flag", (), False),
    "pure": (("llm", "pure"), "flag", (), False),
    "no_cache": (("cache", "no_cache"), "flag", (), False),
    "no_compress": (("image", "compress"), "inverted", (), False),
    "profile": (("output", "profile"), "word", ("rag", "obsidian", "okf"), True),
    "strategy": (
        ("fetch", "strategy"),
        "word",
        ("auto", "static", "playwright", "defuddle", "jina"),
        False,
    ),
    # v1.20
    "format": (("output", "format"), "word", ("markdown", "json"), False),
    "ocr_lang": (("ocr", "lang"), "language", (), False),
    "ocr_all_pages": (("ocr", "per_page_routing"), "inverted", (), False),
}


def invalid(detail: str) -> ApiError:
    return ApiError(422, "invalid_convert_defaults", detail)


def of_config(config: MarkitaiConfig) -> dict[str, Any]:
    """The defaults a configuration gives; a Cloudflare strategy reads ``auto``."""
    shown: dict[str, Any] = {}
    for name, ((section, key), kind, words, nullable) in OPTIONS.items():
        value = getattr(getattr(config, section), key)
        if kind == "flag":
            shown[name] = bool(value)
        elif kind == "inverted":
            shown[name] = not value
        elif kind == "language" or value in words:
            shown[name] = value
        else:
            shown[name] = None if nullable else words[0]
    return shown


def checked(value: Any, languages: list[str] | None = None) -> dict[str, Any]:
    """A caller's ``defaults`` object: only known options, each of its kind;
    ``ocr_lang`` one of *languages* (:func:`ocr_languages`)."""
    if not isinstance(value, dict):
        raise invalid("defaults must be an object")
    for key, entry in value.items():
        option = OPTIONS.get(key) if isinstance(key, str) else None
        if option is None:
            shown = str(key)[:48]
            raise invalid(
                f"'{shown}' cannot be a default; these can: {', '.join(OPTIONS)}"
            )
        _path, kind, words, nullable = option
        if kind == "language":
            allowed = languages or list(OCR_LANGUAGES)
            if not isinstance(entry, str) or entry not in allowed:
                raise invalid(f"default '{key}' must be one of: {', '.join(allowed)}")
        elif kind == "word":
            fits = (entry is None and nullable) or (
                isinstance(entry, str) and entry in words
            )
            if not fits:
                raise invalid(f"default '{key}' must be one of: {', '.join(words)}")
        elif not isinstance(entry, bool):
            raise invalid(f"default '{key}' must be true or false")
    return dict(value)


def revision(value: Any) -> str:
    """``sha256:`` + the digest of the JSON of what a save replaces."""
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def stored(name: str, value: Any) -> Any:
    """The value an option has at its configuration key."""
    return (not value) if OPTIONS[name][1] == "inverted" else value


def write_into_file(raw: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    """Write *defaults* into the configuration file's JSON: a key the file
    already has is set, a value the built-in defaults already give is not
    added, and an ``output`` section left empty by removing the profile goes."""
    from markitai.config import MarkitaiConfig

    builtin = of_config(MarkitaiConfig())
    wanted: dict[str, Any] = {}
    for name, value in defaults.items():
        (section, key), *_ = OPTIONS[name]
        found = raw.get(section)
        has = isinstance(found, dict) and key in found
        if has or builtin[name] != value:
            wanted[name] = value
    removing = "profile" in wanted and wanted["profile"] is None
    for name, value in wanted.items():
        (section, key), *_ = OPTIONS[name]
        found = raw.get(section)
        if not isinstance(found, dict):
            if value is None:
                continue
            found = raw[section] = {}
        if value is None:
            found.pop(key, None)
        else:
            found[key] = stored(name, value)
    if removing and raw.get("output") == {}:
        del raw["output"]


def write_into_running(config: MarkitaiConfig, defaults: Mapping[str, Any]) -> None:
    """Apply *defaults* to the running configuration."""
    for name, value in defaults.items():
        (section, key), *_ = OPTIONS[name]
        setattr(getattr(config, section), key, stored(name, value))


def answer(
    server: dict[str, Any],
    own: dict[str, Any] | None,
    *,
    scope: Literal["server", "user"],
    can_save: bool,
) -> dict[str, Any]:
    """The ``ConvertDefaultsView`` a caller sees."""
    return {
        "defaults": {**server, **(own or {})},
        "server": server,
        "own": own,
        "origin": "user" if own is not None else "server",
        "scope": scope,
        "can_save": can_save,
        "revision": revision(own if scope == "user" else server),
    }

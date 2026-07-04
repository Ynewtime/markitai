"""Interactive config editor — schema introspection and edit loop.

Walks the Pydantic MarkitaiConfig model tree to extract a flat list of
editable scalar settings.  The main selector uses a prompt_toolkit
Application (search box, fuzzy filtering, scrollable list).  Sub-editors
for individual values use questionary prompts with injected Esc support.
"""

from __future__ import annotations

import types as _types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from markitai.config import MarkitaiConfig

# Sections to skip entirely (too complex or too advanced for inline editing)
_SKIP_SECTIONS = {"presets", "prompts", "domain_profiles"}

# Sentinel to distinguish "user cancelled" from "user set value to None"
_CANCEL = object()


def extract_editable_settings(cfg: MarkitaiConfig) -> list[dict[str, Any]]:
    """Walk the config model and return a flat list of editable settings.

    Each entry is a dict with keys:
        key: dot-separated path (e.g. "output.dir")
        value: current value
        field_type: "str", "int", "float", "bool", "literal"
        description: human-readable description or ""
        choices: list of valid values (only for "literal" type)

    Complex types (lists, dicts, nested models) are skipped.
    """
    settings: list[dict[str, Any]] = []
    _walk_model(cfg, "", type(cfg).model_fields, settings)
    return settings


def _walk_model(
    obj: BaseModel,
    prefix: str,
    fields: dict[str, FieldInfo],
    settings: list[dict[str, Any]],
) -> None:
    """Recursively walk model fields, collecting scalar settings."""
    for name, field_info in fields.items():
        if name in _SKIP_SECTIONS:
            continue

        key = f"{prefix}{name}" if prefix else name
        value = getattr(obj, name, None)
        annotation = field_info.annotation

        # Unwrap Optional[X] / X | None → X
        origin = get_origin(annotation)
        args = get_args(annotation)
        if (origin is typing.Union or origin is _types.UnionType) and type(
            None
        ) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                annotation = non_none[0]
                origin = get_origin(annotation)
                args = get_args(annotation)

        # Nested BaseModel → recurse
        if isinstance(value, BaseModel):
            _walk_model(value, f"{key}.", type(value).model_fields, settings)
            continue

        # Skip complex types (list, dict) — check both value AND annotation
        if isinstance(value, (list, dict)):
            continue
        if origin in (list, dict):
            continue
        # Skip None values where the unwrapped annotation is still a BaseModel
        if value is None and annotation is not None:
            try:
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    continue
            except TypeError:
                pass

        # Determine field_type
        description = ""
        if field_info.description:
            description = field_info.description
        elif field_info.title:
            description = field_info.title

        if origin is typing.Literal:
            choices = list(args)
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "literal",
                    "description": description,
                    "choices": choices,
                }
            )
        elif annotation is bool or (isinstance(value, bool)):
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "bool",
                    "description": description,
                }
            )
        elif annotation is int:
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "int",
                    "description": description,
                }
            )
        elif annotation is float:
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "float",
                    "description": description,
                }
            )
        elif annotation is str or isinstance(value, str):
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "str",
                    "description": description,
                }
            )
        # else: skip unknown types


def format_display_value(value: Any) -> str:
    """Format a config value for display in the choice list."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fuzzy_match(query: str, text: str) -> tuple[bool, int]:
    """Case-insensitive fuzzy match with scoring.

    Characters in *query* must appear in *text* in order but not
    necessarily consecutively.

    Returns (matched, score) — lower score is better.
    Empty query matches everything with score 0.
    """
    if not query:
        return True, 0
    ql, tl = query.lower(), text.lower()
    qi, score, prev = 0, 0, -2
    for ti, ch in enumerate(tl):
        if qi < len(ql) and ch == ql[qi]:
            score += ti
            if ti == prev + 1:
                score -= ti
            prev = ti
            qi += 1
    return (True, score) if qi == len(ql) else (False, 0)


def _select_setting(settings: list[dict[str, Any]], config_path: str) -> str | None:
    """Claude Code-style setting selector built on prompt_toolkit.

    Returns the selected setting key, or None if cancelled.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.widgets import Frame

    if not settings:
        return None

    VISIBLE = 15
    state: dict[str, Any] = {
        "cursor": 0,
        "scroll": 0,
        "rendered_lines": 0,
        "filtered": list(range(len(settings))),
        "last_query": "",
    }

    def _refresh_filtered() -> list[int]:
        """Recompute filtered indices when search query changes."""
        q = search_buf.text.strip()
        if q == state["last_query"]:
            return state["filtered"]
        state["last_query"] = q
        if not q:
            state["filtered"] = list(range(len(settings)))
        else:
            scored = []
            for i, s in enumerate(settings):
                text = f"{s['key']} {format_display_value(s['value'])}"
                ok, sc = fuzzy_match(q, text)
                if ok:
                    scored.append((sc, i))
            scored.sort(key=lambda x: x[0])
            state["filtered"] = [i for _, i in scored]
        return state["filtered"]

    def _get_list_text() -> FormattedText:
        filtered = _refresh_filtered()
        if not filtered:
            return FormattedText([("", "  (no matches)\n")])

        cursor = state["cursor"]
        scroll = state["scroll"]

        # Clamp cursor
        if cursor >= len(filtered):
            state["cursor"] = cursor = max(0, len(filtered) - 1)
        if cursor < scroll:
            state["scroll"] = scroll = cursor
        if cursor >= scroll + VISIBLE:
            state["scroll"] = scroll = cursor - VISIBLE + 1

        max_key = max(len(settings[i]["key"]) for i in filtered)
        max_val = min(
            max(len(format_display_value(settings[i]["value"])) for i in filtered),
            30,
        )

        parts: list[tuple[str, str]] = []

        # "N more above"
        if scroll > 0:
            parts.append(("class:muted", f"  ↑ {scroll} more above\n"))

        end = min(scroll + VISIBLE, len(filtered))
        for row in range(scroll, end):
            idx = filtered[row]
            s = settings[idx]
            key = s["key"].ljust(max_key)
            val = format_display_value(s["value"]).ljust(max_val)
            desc = s.get("description", "")
            prefix = " ❯ " if row == cursor else "   "
            line = f"{prefix}{key}  {val}"
            if desc:
                line += f"  [{desc}]"
            line += "\n"
            style = "reverse" if row == cursor else ""
            parts.append((style, line))

        # "N more below"
        remaining = len(filtered) - end
        if remaining > 0:
            parts.append(("class:muted", f"  ↓ {remaining} more below\n"))

        # Track lines for erasure: items + indicators
        state["rendered_lines"] = sum(1 for _, t in parts if t.endswith("\n"))
        return FormattedText(parts)

    def _get_footer() -> FormattedText:
        if search_buf.text:
            return FormattedText(
                [("class:muted", "  Type to filter · Enter to edit · Esc to clear")]
            )
        return FormattedText(
            [("class:muted", "  Type to filter · Enter to edit · Esc to exit")]
        )

    search_buf = Buffer(name="search")

    def _on_search_changed(_: Any) -> None:
        state["cursor"] = 0
        state["scroll"] = 0
        state["last_query"] = ""  # invalidate filter cache

    search_buf.on_text_changed += _on_search_changed

    kb = KeyBindings()

    @kb.add("up")
    def _up(_event: Any) -> None:
        if state["cursor"] > 0:
            state["cursor"] -= 1

    @kb.add("down")
    def _down(_event: Any) -> None:
        filtered = _refresh_filtered()
        if state["cursor"] < len(filtered) - 1:
            state["cursor"] += 1

    @kb.add("enter")
    def _enter(event: Any) -> None:
        filtered = _refresh_filtered()
        if filtered:
            event.app.exit(result=settings[filtered[state["cursor"]]]["key"])
        else:
            event.app.exit(result=None)

    @kb.add("escape")
    def _escape(event: Any) -> None:
        if search_buf.text:
            search_buf.text = ""
            state["cursor"] = 0
            state["scroll"] = 0
        else:
            event.app.exit(result=None)

    @kb.add("c-c")
    def _ctrl_c(event: Any) -> None:
        event.app.exit(result=None)

    list_control = FormattedTextControl(_get_list_text)

    layout = Layout(
        HSplit(
            [
                Window(
                    FormattedTextControl(
                        FormattedText(
                            [
                                ("bold", "  Markitai Configuration  "),
                                ("class:muted", f"({config_path})"),
                            ]
                        )
                    ),
                    height=2,
                ),
                Frame(Window(BufferControl(search_buf), height=1)),
                Window(list_control, height=Dimension(max=VISIBLE + 2)),
                Window(FormattedTextControl(_get_footer), height=1),
            ]
        )
    )

    app: Application[str | None] = Application(
        layout=layout, key_bindings=kb, full_screen=False
    )
    try:
        result = app.run()
    except (EOFError, KeyboardInterrupt):
        result = None

    # Erase rendered UI: header(2) + frame(3) + list lines + footer(1)
    total = 2 + 3 + state["rendered_lines"] + 1
    _erase_lines(total)
    return result


def _esc_key_bindings():
    """Create key bindings that map Esc to cancel (same as Ctrl-C)."""
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()

    @kb.add(Keys.Escape, eager=True)
    def _(event):
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    return kb


def _add_esc_to_question(question):
    """Inject Esc key binding into a questionary Question's application."""
    from prompt_toolkit.key_binding import merge_key_bindings

    app = question.application
    app.key_bindings = merge_key_bindings([app.key_bindings, _esc_key_bindings()])
    return question


def _prompt_new_value(setting: dict[str, Any]) -> Any:
    """Prompt the user for a new value based on field type.

    Returns the new value, or _CANCEL if the user cancelled (Esc / Ctrl-C).
    """
    import questionary
    from questionary import Choice

    key = setting["key"]
    current = setting["value"]
    field_type = setting["field_type"]

    if field_type == "bool":
        current_bool = current if current is not None else False
        choice_true = Choice(title="true", value=True)
        choice_false = Choice(title="false", value=False)
        result = _add_esc_to_question(
            questionary.select(
                f"  {key}",
                choices=[choice_true, choice_false],
                default=choice_true if current_bool else choice_false,
                use_jk_keys=False,
                instruction="(↑↓ move · Esc to cancel)",
            )
        ).ask()
        return _CANCEL if result is None else result

    if field_type == "literal":
        raw_choices = setting.get("choices", [])
        # Preserve original typed values via Choice(value=...)
        choice_objs = [Choice(title=str(c), value=c) for c in raw_choices]
        default_choice = None
        if current is not None:
            default_choice = next(
                (ch for ch in choice_objs if ch.value == current), None
            )
        result = _add_esc_to_question(
            questionary.select(
                f"  {key}",
                choices=choice_objs,
                default=default_choice,
                use_jk_keys=False,
                instruction="(↑↓ move · Esc to cancel)",
            )
        ).ask()
        return _CANCEL if result is None else result

    kb = _esc_key_bindings()

    if field_type == "int":
        result = questionary.text(
            f"  {key}",
            default=str(current) if current is not None else "",
            validate=lambda v: (
                v == ""
                or (v.lstrip("-").isdigit() and v != "-")
                or "Must be an integer"
            ),
            instruction="(Esc to cancel)",
            key_bindings=kb,
        ).ask()
        if result is None:
            return _CANCEL
        # Empty input = keep current value (all int fields have non-Optional defaults)
        return int(result) if result else current

    if field_type == "float":

        def _validate_float(v: str) -> bool | str:
            if v == "":
                return True
            try:
                float(v)
                return True
            except ValueError:
                return "Must be a number"

        result = questionary.text(
            f"  {key}",
            default=str(current) if current is not None else "",
            validate=_validate_float,
            instruction="(Esc to cancel)",
            key_bindings=kb,
        ).ask()
        if result is None:
            return _CANCEL
        # Empty input = keep current value (all float fields have non-Optional defaults)
        return float(result) if result else current

    # str fallback
    result = questionary.text(
        f"  {key}",
        default=str(current) if current is not None else "",
        instruction="(Esc to cancel)",
        key_bindings=kb,
    ).ask()
    return _CANCEL if result is None else result


def _erase_lines(n: int) -> None:
    """Erase the last *n* lines from the terminal (move up + clear to end)."""
    import sys

    if n > 0 and sys.stdout.isatty():
        sys.stdout.write(f"\033[{n}A\033[J")
        sys.stdout.flush()


def _get_cursor_row() -> int:
    """Query the terminal for the current cursor row (1-based). Returns 0 on failure."""
    import sys

    if not sys.stdin.isatty():
        return 0
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdout.write("\033[6n")
            sys.stdout.flush()
            buf = b""
            while True:
                ch = sys.stdin.buffer.read(1)
                buf += ch
                if ch == b"R":
                    break
            # Response: ESC [ row ; col R
            parts = buf.decode().lstrip("\033[").rstrip("R").split(";")
            return int(parts[0])
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return 0


def run_config_editor() -> None:
    """Run the interactive config editor loop."""
    from rich.console import Console

    from markitai.config import ConfigManager

    console = Console(stderr=True)
    manager = ConfigManager()
    manager.load()
    cfg = manager.config
    config_path = str(manager.config_path or "defaults")

    while True:
        settings = extract_editable_settings(cfg)
        selected_key = _select_setting(settings, config_path)
        # _select_setting erases its own UI on exit

        if selected_key is None:
            break

        setting = next((s for s in settings if s["key"] == selected_key), None)
        if setting is None:
            continue

        row_before = _get_cursor_row()
        new_value = _prompt_new_value(setting)
        row_after = _get_cursor_row()
        if row_before and row_after:
            _erase_lines(row_after - row_before)
        else:
            # Fallback: text prompts render 1 line, select prompts vary
            _erase_lines(2)

        if new_value is _CANCEL:
            continue

        old_config_dict = manager.config.model_dump()
        try:
            manager.set(selected_key, new_value)
            # Validate before saving so invalid values (e.g. image.quality=500)
            # never reach disk, where they would break every subsequent load.
            try:
                MarkitaiConfig.model_validate(manager.config.model_dump())
            except ValidationError as ve:
                manager.restore(MarkitaiConfig.model_validate(old_config_dict))
                errors = "; ".join(
                    f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                    for err in ve.errors()
                )
                console.print(f"  [red]✗[/] Invalid value: {errors}")
            else:
                manager.save()
                console.print(
                    f"  [green]✓[/] {selected_key} = {format_display_value(new_value)}"
                )
        except Exception as e:
            console.print(f"  [red]✗[/] Error: {e}")
        cfg = manager.config

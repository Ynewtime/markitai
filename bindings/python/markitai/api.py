"""Typed Python API shared by synchronous and asynchronous callers.

Importing this module is cheap: asyncio, dataclasses, json and pathlib are
imported by the first call that needs them, and the result records and type
names live in `markitai._records`, which `__getattr__` and `_load_records`
bring in on first use. Type checkers see the same names either way.
"""

from __future__ import annotations

import sys

from . import _native

# Not imported at run time: a type checker sees the real definitions, while
# the interpreter resolves these names on first use.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    from ._records import ConfigModel, ConversionOutput, ConversionUsage, OutputProfileName

__all__ = [
    "ConversionError", "ConversionOutput", "ConversionUsage", "FetchError",
    "NoModelConfiguredError", "OutputProfileName", "aconvert", "convert",
    "enable_worker_processes",
]

_RECORDS = ("ConfigModel", "ConversionOutput", "ConversionUsage", "OutputProfileName")


def _load_records() -> tuple[type[ConversionOutput], type[ConversionUsage]]:
    """Import `markitai._records` and bind its names here, with the ones they annotate with.

    The records keep `markitai.api` as their module, so a tool that resolves
    their string annotations (`typing.get_type_hints`) looks the names up here.
    """
    from . import _records

    namespace = globals()
    if "ConversionOutput" not in namespace:
        from collections.abc import Mapping
        from pathlib import Path
        from typing import Any

        bound = {name: getattr(_records, name) for name in _RECORDS}
        bound.update(Any=Any, Mapping=Mapping, Path=Path)
        # A single merge, so a thread that sees `ConversionOutput` sees every name.
        namespace.update(bound)
    return _records.ConversionOutput, _records.ConversionUsage


if not TYPE_CHECKING:
    def __getattr__(name: str):
        if name in _RECORDS:
            _load_records()
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ConversionError(RuntimeError):
    """A conversion failed; code retains the native error category."""

    def __init__(
        self, message: str, *, code: str = "conversion_error",
        usage: ConversionUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.usage = usage


class FetchError(Exception):
    """An HTTP source could not be fetched."""

    code = "fetch_error"
    usage: ConversionUsage | None = None


class NoModelConfiguredError(ValueError):
    """LLM processing was requested without a configured model."""

    code = "no_model_configured"
    usage: ConversionUsage | None = None


def enable_worker_processes() -> None:
    """Compatibility hook; Rust handles its own execution without subprocesses."""


def _result(response: str) -> ConversionOutput:
    import json

    envelope = json.loads(response)
    if not envelope["ok"]:
        error = envelope["error"]
        code, message = error["code"], error["message"]
        raw_usage = error.get("usage")
        usage = None
        if raw_usage is not None:
            _, usage_type = _load_records()
            usage = usage_type(**raw_usage)
        if code == "fetch_error":
            exception = FetchError(message)
        elif code == "no_model_configured":
            exception = NoModelConfiguredError(message)
        elif code in {"invalid_input", "invalid_json", "config_error"}:
            exception = ValueError(message)
        elif code == "not_found":
            exception = FileNotFoundError(message)
        elif code == "is_directory":
            exception = IsADirectoryError(message)
        elif code == "io_error":
            exception = OSError(message)
        else:
            exception = ConversionError(message, code=code)
        # Preserve built-in and public exception categories. None means the
        # producer supplied no accounting, not that the failed call was free.
        exception.usage = usage
        raise exception
    from pathlib import Path

    output_type, usage_type = _load_records()
    data = envelope["result"]
    data["usage"] = usage_type(**data.get("usage", {}))
    for key in ("output_path", "llm_output_path"):
        data[key] = Path(data[key]) if data.get(key) is not None else None
    for key in ("assets", "screenshots"):
        data[key] = [Path(value) for value in data.get(key, [])]
    return output_type(**data)


def _convert(
    source: str | Path,
    *,
    output_dir: str | Path | None,
    config: Mapping[str, Any] | ConfigModel | None,
    llm: bool | None,
    ocr: bool | None,
    screenshot: bool | None,
    alt: bool | None,
    desc: bool | None,
    profile: OutputProfileName | None,
) -> ConversionOutput:
    import json
    from collections.abc import Mapping
    from pathlib import Path

    if not isinstance(source, (str, Path)):
        raise TypeError("source must be a string or pathlib.Path")
    if config is not None:
        config = dict(config) if isinstance(config, Mapping) else config.model_dump(mode="json")
    options = {
        "output_dir": str(output_dir) if output_dir is not None else None,
        "config": config,
        "llm": llm,
        "ocr": ocr,
        "screenshot": screenshot,
        "alt": alt,
        "desc": desc,
        "profile": profile,
    }
    request = json.dumps({"source": str(source), "options": options}, ensure_ascii=False)
    return _result(_native.convert_json(request))


def convert(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: Mapping[str, Any] | ConfigModel | None = None,
    llm: bool | None = None,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
    profile: OutputProfileName | None = None,
) -> ConversionOutput:
    """Convert a file or URL, releasing the GIL while the Rust engine runs."""
    # An event loop can only be running once asyncio has been imported, so a
    # synchronous caller that never imported it does not pay for it here.
    get_running_loop = getattr(sys.modules.get("asyncio"), "get_running_loop", None)
    if get_running_loop is not None:
        try:
            get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "markitai.convert() cannot be called from a running event loop; "
                "use `await markitai.aconvert(...)` instead"
            )
    return _convert(
        source, output_dir=output_dir, config=config, llm=llm, ocr=ocr,
        screenshot=screenshot, alt=alt, desc=desc, profile=profile,
    )


async def aconvert(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: Mapping[str, Any] | ConfigModel | None = None,
    llm: bool | None = None,
    ocr: bool | None = None,
    screenshot: bool | None = None,
    alt: bool | None = None,
    desc: bool | None = None,
    profile: OutputProfileName | None = None,
) -> ConversionOutput:
    """Convert without blocking the event loop; work runs in a native thread.

    Cancelling the await abandons the result but does not interrupt a running
    conversion or roll back files it writes.
    """
    import asyncio

    return await asyncio.to_thread(
        _convert, source, output_dir=output_dir, config=config, llm=llm, ocr=ocr,
        screenshot=screenshot, alt=alt, desc=desc, profile=profile,
    )

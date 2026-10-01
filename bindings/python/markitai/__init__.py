"""Document conversion through the in-process Rust engine."""

from ._native import __version__
from .api import (
    ConversionError,
    FetchError,
    NoModelConfiguredError,
    aconvert,
    convert,
    enable_worker_processes,
)

# Not imported at run time: type checkers see these as ordinary re-exports,
# while the interpreter loads each on first access (see `__getattr__`), which
# keeps dataclasses, typing, pathlib and the configuration models out of
# `import markitai`.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from .api import ConversionOutput, ConversionUsage, OutputProfileName
    from .config import MarkitaiConfig

__all__ = [
    "ConversionError", "ConversionOutput", "ConversionUsage", "FetchError",
    "NoModelConfiguredError", "OutputProfileName", "__version__", "aconvert",
    "convert", "enable_worker_processes", "MarkitaiConfig",
]

_RECORDS = ("ConversionOutput", "ConversionUsage", "OutputProfileName")
_LAZY = (*_RECORDS, "MarkitaiConfig", "config")

if not TYPE_CHECKING:
    def __getattr__(name):
        namespace = globals()
        if name in _RECORDS:
            from . import api
            namespace[name] = getattr(api, name)
        elif name in ("MarkitaiConfig", "config"):
            # Importing the submodule also binds `config` on this package.
            from .config import MarkitaiConfig
            namespace["MarkitaiConfig"] = MarkitaiConfig
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return namespace[name]

    def __dir__():
        return sorted({*globals(), *_LAZY})

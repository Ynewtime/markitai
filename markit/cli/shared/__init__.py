"""Shared CLI utilities for convert and batch commands."""

from markit.cli.shared.context import ConversionContext
from markit.cli.shared.credentials import get_effective_api_key, get_unique_credentials
from markit.cli.shared.executor import SignalHandler, execute_phased_conversion, execute_single_file
from markit.cli.shared.options import ConversionOptions

__all__ = [
    "ConversionOptions",
    "ConversionContext",
    "SignalHandler",
    "execute_single_file",
    "execute_phased_conversion",
    "get_effective_api_key",
    "get_unique_credentials",
]

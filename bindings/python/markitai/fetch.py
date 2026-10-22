"""Compatibility with Python markitai's `markitai.fetch` module."""


async def close_shared_clients() -> None:
    """Compatibility no-op: the native engine shares no HTTP client between calls.

    Python markitai's long-lived hosts awaited this on shutdown to release its
    shared HTTP clients; the native engine has none left open, so awaiting it
    does nothing.
    """

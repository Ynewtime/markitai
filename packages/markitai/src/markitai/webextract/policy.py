"""Extraction policy: per-URL and per-extractor inclusion rules.

This module provides the policy layer that governs how structured extractors
integrate with the pipeline.  It is a placeholder for future thread-policy
support (Task 4) and other per-site extraction rules.

Currently exports:
- ``ExtractionPolicy``: A dataclass that will carry thread-inclusion rules and
  other site-specific extraction settings once Task 4 is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractionPolicy:
    """Per-URL extraction rules for the pipeline.

    This is a forward-compatible placeholder.  Fields will be populated in
    Task 4 when thread-policy support is introduced.

    Attributes:
        site: Identifier of the site this policy applies to (e.g. ``"x_tweet"``).
        settings: Arbitrary key/value settings for the extractor. Intended for
            future expansion (e.g. thread inclusion rules, reply depth limits).
    """

    site: str = "generic"
    settings: dict[str, Any] = field(default_factory=dict)

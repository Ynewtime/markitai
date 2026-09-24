"""Output path utilities for Markitai."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path


def resolve_name_conflict(
    path: Path,
    on_conflict: str,
    rename_fn: Callable[[int], Path],
    *,
    exists: Callable[[Path], bool] = Path.exists,
) -> Path | None:
    """Resolve a filename conflict using the given strategy.

    This is the single authoritative implementation of the
    skip / overwrite / rename-with-sequential-version-number pattern
    used throughout the codebase.

    Args:
        path: The target file path that already exists.
        on_conflict: Strategy -- "skip", "overwrite", or "rename".
        rename_fn: A callable that receives a sequence number (starting at 2)
            and returns a candidate ``Path``.  Called repeatedly until the
            returned path does not yet exist on disk.

    Returns:
        Resolved path, or ``None`` if the file should be skipped.
    """
    if not exists(path):
        return path

    if on_conflict == "skip":
        return None
    if on_conflict == "overwrite":
        return path

    # rename: find next available sequence number
    seq = 2
    while True:
        candidate = rename_fn(seq)
        if not exists(candidate):
            return candidate
        seq += 1


def resolve_output_path(
    base_path: Path,
    on_conflict: str,
) -> Path | None:
    """Resolve output path based on conflict strategy.

    Args:
        base_path: The original output file path
        on_conflict: Conflict resolution strategy ("skip", "overwrite", "rename")

    Returns:
        Resolved path, or None if file should be skipped.
        For rename strategy: file.pdf.md -> file.pdf.v2.md -> file.pdf.v3.md
        For rename with .llm.md: file.pdf.llm.md -> file.pdf.v2.llm.md
    """
    return resolve_name_conflict(
        base_path, on_conflict, _versioned_rename(base_path), exists=_occupied_on_disk
    )


def split_markdown_name(name: str) -> tuple[str, str]:
    """Split an output name into its stem and markitai suffix.

    ``file.pdf.llm.md`` -> ``("file.pdf", ".llm.md")``, ``file.pdf.md`` ->
    ``("file.pdf", ".md")``. A name without either suffix is all stem.
    """
    if name.endswith(".llm.md"):
        return name[: -len(".llm.md")], ".llm.md"
    if name.endswith(".md"):
        return name[: -len(".md")], ".md"
    return name, ""


def _versioned_rename(base_path: Path) -> Callable[[int], Path]:
    """Build the ``file.pdf.md`` -> ``file.pdf.v<N>.md`` rename function."""
    base_stem, markitai_suffix = split_markdown_name(base_path.name)

    def _rename(seq: int) -> Path:
        return base_path.parent / f"{base_stem}.v{seq}{markitai_suffix}"

    return _rename


def _occupied_on_disk(path: Path) -> bool:
    """Whether the output name is taken on disk by its ``.md`` or ``.llm.md``."""
    stem, _ = split_markdown_name(path.name)
    return (
        path.exists()
        or path.with_name(f"{stem}.md").exists()
        or path.with_name(f"{stem}.llm.md").exists()
    )


def _probe_case_insensitive(directory: Path) -> bool:
    """Whether names in *directory* compare case-insensitively on disk.

    Creates a mixed-case probe file and looks it up in lower case. When the
    directory cannot be probed (missing, read-only) the answer is True: a
    reservation that folds case only ever renames more, never clobbers.
    """
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".MarkitaiCaseProbe-", dir=directory
        ) as probe:
            name = os.path.basename(probe.name)
            return os.path.exists(os.path.join(directory, name.lower()))
    except OSError:
        return True


class OutputNameReservations:
    """Batch-scoped reservation table for output markdown names.

    Every item of a batch resolves its output name before its conversion
    (or LLM call) is awaited, but writes the file only afterwards, so a
    disk-only conflict check lets two items that derive the same name —
    ``report.pdf`` next to a ``.urls`` entry for ``http://host/report.pdf``,
    or ``page?a=1`` and ``page?a=2`` under ``--llm``, which writes no base
    ``.md`` to hold the name — both pick it and silently overwrite each
    other. A claim is atomic and blocks the whole name: its ``.md`` and
    ``.llm.md`` forms alike, compared case-folded where the file system
    does. Files and URLs of one batch share one table.
    """

    def __init__(self) -> None:
        self._claimed: set[tuple[str, str]] = set()
        self._case_insensitive: dict[str, bool] = {}
        self._lock = threading.Lock()

    def _key(self, path: Path) -> tuple[str, str]:
        parent = os.path.abspath(path.parent)
        stem, _ = split_markdown_name(path.name)
        folds = self._case_insensitive.get(parent)
        if folds is None:
            folds = _probe_case_insensitive(Path(parent))
            self._case_insensitive[parent] = folds
        if folds:
            return parent.casefold(), stem.casefold()
        return parent, stem

    def reserve(self, path: Path) -> None:
        """Hold a name without resolving it (e.g. a finished item's output)."""
        with self._lock:
            self._claimed.add(self._key(path))

    def is_reserved(self, path: Path) -> bool:
        """Whether *path*'s name is already held in this table."""
        with self._lock:
            return self._key(path) in self._claimed

    def claim(
        self,
        base_path: Path,
        on_conflict: str,
        *,
        reuse: Path | None = None,
    ) -> Path | None:
        """Resolve and reserve an output path in one step.

        ``on_conflict`` applies to names taken on disk (earlier runs); a
        name held by another item of this batch is always renamed around,
        whatever the strategy — skipping or overwriting there would lose
        one of the two results.

        Args:
            base_path: Default output path for the item.
            on_conflict: ``"skip"``, ``"overwrite"`` or ``"rename"``.
            reuse: The item's own earlier output (a resumed item redoing
                its work); claimed as-is, i.e. overwritten, unless another
                item of this batch already holds it.

        Returns:
            The reserved path, or None when the item should be skipped.
        """
        with self._lock:
            if reuse is not None:
                reuse_key = self._key(reuse)
                if reuse_key not in self._claimed:
                    self._claimed.add(reuse_key)
                    return reuse

            def taken(path: Path) -> bool:
                return self._key(path) in self._claimed

            if not taken(base_path):
                resolved = resolve_output_path(base_path, on_conflict)
                if resolved is None:
                    return None
                if not taken(resolved):
                    self._claimed.add(self._key(resolved))
                    return resolved

            rename = _versioned_rename(base_path)
            seq = 2
            while True:
                candidate = rename(seq)
                if not taken(candidate) and not _occupied_on_disk(candidate):
                    self._claimed.add(self._key(candidate))
                    return candidate
                seq += 1


@dataclass
class _ClaimScope:
    """One batch item's view of the reservation table (see output_claim_scope)."""

    reservations: OutputNameReservations
    reuse: Path | None
    on_conflict: str | None
    on_claimed: Callable[[Path], None] | None
    resolved: dict[str, Path | None] = field(default_factory=dict)


_active_claim_scope: ContextVar[_ClaimScope | None] = ContextVar(
    "markitai_output_claim_scope", default=None
)


@contextmanager
def output_claim_scope(
    reservations: OutputNameReservations,
    *,
    reuse: Path | None = None,
    on_conflict: str | None = None,
    on_claimed: Callable[[Path], None] | None = None,
) -> Iterator[None]:
    """Route this item's output-name resolution through a reservation table.

    Batch drivers wrap each item in a scope; the shared file and URL
    pipelines resolve their output through :func:`resolve_item_output_path`,
    which claims from the table while a scope is active and falls back to
    the plain disk check otherwise (single-file runs, serve, the API).
    Resolving the same base path twice inside one scope returns the first
    answer, so a caller and the cascade it delegates to agree on the name.

    Args:
        reservations: The batch's table.
        reuse: The item's own earlier output to overwrite (resume).
        on_conflict: Per-item strategy override (resume redoes an item
            over its previous output: ``"overwrite"``).
        on_claimed: Called with the reserved path, e.g. to record it in
            the batch state before the (interruptible) work starts.
    """
    token = _active_claim_scope.set(
        _ClaimScope(
            reservations=reservations,
            reuse=reuse,
            on_conflict=on_conflict,
            on_claimed=on_claimed,
        )
    )
    try:
        yield
    finally:
        _active_claim_scope.reset(token)


def resolve_item_output_path(base_path: Path, on_conflict: str) -> Path | None:
    """Resolve an output path, claiming it when a batch scope is active.

    Same contract as :func:`resolve_output_path`; see
    :func:`output_claim_scope` for the batch behavior.
    """
    scope = _active_claim_scope.get()
    if scope is None:
        return resolve_output_path(base_path, on_conflict)

    memo_key = os.path.abspath(base_path)
    if memo_key in scope.resolved:
        return scope.resolved[memo_key]
    resolved = scope.reservations.claim(
        base_path, scope.on_conflict or on_conflict, reuse=scope.reuse
    )
    scope.resolved[memo_key] = resolved
    if resolved is not None and scope.on_claimed is not None:
        scope.on_claimed(resolved)
    return resolved

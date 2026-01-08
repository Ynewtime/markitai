"""State management for batch processing with resume support."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from markit.exceptions import StateError
from markit.utils.logging import get_logger

log = get_logger(__name__)


FileStatus = Literal["pending", "processing", "completed", "failed", "skipped"]


@dataclass
class FileState:
    """State of a single file in the batch."""

    path: str
    status: FileStatus
    output_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    file_hash: str | None = None  # For detecting changes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BatchState:
    """State of a batch processing operation."""

    batch_id: str
    input_dir: str
    output_dir: str
    started_at: str
    updated_at: str
    completed_at: str | None = None
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    files: dict[str, FileState] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchState":
        """Create from dictionary."""
        files = {k: FileState.from_dict(v) for k, v in data.get("files", {}).items()}
        return cls(
            batch_id=data["batch_id"],
            input_dir=data["input_dir"],
            output_dir=data["output_dir"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            completed_at=data.get("completed_at"),
            total_files=data.get("total_files", 0),
            completed_files=data.get("completed_files", 0),
            failed_files=data.get("failed_files", 0),
            skipped_files=data.get("skipped_files", 0),
            files=files,
            options=data.get("options", {}),
        )

    @property
    def pending_files(self) -> list[str]:
        """Get list of pending file paths."""
        return [
            path for path, state in self.files.items() if state.status in ("pending", "processing")
        ]

    @property
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        return len(self.pending_files) == 0


class StateManager:
    """Manages batch processing state for resume support.

    State is persisted to a JSON file and updated incrementally.
    This allows resuming interrupted batch operations.
    """

    def __init__(self, state_file: Path | str) -> None:
        """Initialize the state manager.

        Args:
            state_file: Path to the state file
        """
        self.state_file = Path(state_file)
        self._state: BatchState | None = None

    def create_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        files: list[Path],
        options: dict[str, Any] | None = None,
    ) -> BatchState:
        """Create a new batch state.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            files: List of files to process
            options: Batch options

        Returns:
            New BatchState
        """
        now = datetime.now().isoformat()
        batch_id = self._generate_batch_id(input_dir, now)

        file_states = {}
        for file_path in files:
            rel_path = str(file_path.relative_to(input_dir))
            file_states[rel_path] = FileState(
                path=str(file_path),
                status="pending",
                file_hash=self._compute_file_hash(file_path),
            )

        self._state = BatchState(
            batch_id=batch_id,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            started_at=now,
            updated_at=now,
            total_files=len(files),
            files=file_states,
            options=options or {},
        )

        self._save()
        log.info(
            "Batch state created",
            batch_id=batch_id,
            total_files=len(files),
        )

        return self._state

    def load_batch(self) -> BatchState | None:
        """Load existing batch state.

        Returns:
            BatchState if exists, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
            self._state = BatchState.from_dict(data)
            log.info(
                "Batch state loaded",
                batch_id=self._state.batch_id,
                pending=len(self._state.pending_files),
            )
            return self._state
        except (json.JSONDecodeError, KeyError) as e:
            raise StateError(f"Invalid state file: {e}") from e

    def update_file_status(
        self,
        file_path: Path | str,
        status: FileStatus,
        output_path: Path | str | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status of a file.

        Args:
            file_path: Path to the file
            status: New status
            output_path: Output file path (if completed)
            error: Error message (if failed)
        """
        if self._state is None:
            raise StateError("No batch state loaded")

        # Find file by path or relative path
        file_key = self._find_file_key(file_path)
        if file_key is None:
            raise StateError(f"File not found in batch: {file_path}")

        file_state = self._state.files[file_key]
        file_state.status = status
        now = datetime.now().isoformat()

        if status == "processing":
            file_state.started_at = now
        elif status in ("completed", "failed", "skipped"):
            file_state.completed_at = now
            if status == "completed":
                self._state.completed_files += 1
                file_state.output_path = str(output_path) if output_path else None
            elif status == "failed":
                self._state.failed_files += 1
                file_state.error = error
            elif status == "skipped":
                self._state.skipped_files += 1

        self._state.updated_at = now
        self._save()

    def mark_batch_complete(self) -> None:
        """Mark the batch as complete."""
        if self._state is None:
            raise StateError("No batch state loaded")

        self._state.completed_at = datetime.now().isoformat()
        self._state.updated_at = self._state.completed_at
        self._save()

        log.info(
            "Batch complete",
            batch_id=self._state.batch_id,
            completed=self._state.completed_files,
            failed=self._state.failed_files,
            skipped=self._state.skipped_files,
        )

    def get_pending_files(self) -> list[Path]:
        """Get list of pending files to process.

        Returns:
            List of Path objects for pending files
        """
        if self._state is None:
            return []

        pending = []
        for _rel_path, file_state in self._state.files.items():
            if file_state.status in ("pending", "processing"):
                # Check if file has changed
                file_path = Path(file_state.path)
                if file_path.exists():
                    current_hash = self._compute_file_hash(file_path)
                    if current_hash != file_state.file_hash:
                        log.warning(
                            "File changed since batch started",
                            file=str(file_path),
                        )
                    pending.append(file_path)
                else:
                    log.warning(
                        "File no longer exists",
                        file=str(file_path),
                    )

        return pending

    def get_state(self) -> BatchState | None:
        """Get current batch state."""
        return self._state

    def clear(self) -> None:
        """Clear the state file."""
        if self.state_file.exists():
            self.state_file.unlink()
        self._state = None
        log.info("State file cleared")

    def _save(self) -> None:
        """Save state to file."""
        if self._state is None:
            return

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    def _find_file_key(self, file_path: Path | str) -> str | None:
        """Find file key in state by path."""
        if self._state is None:
            return None

        path_str = str(file_path)
        for key, state in self._state.files.items():
            if state.path == path_str or key == path_str:
                return key
        return None

    def _generate_batch_id(self, input_dir: Path, timestamp: str) -> str:
        """Generate a unique batch ID."""
        data = f"{input_dir}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file for change detection."""
        if not file_path.exists():
            return ""

        # Use file size and mtime for quick hash
        stat = file_path.stat()
        data = f"{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(data.encode()).hexdigest()

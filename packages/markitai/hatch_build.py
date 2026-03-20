from __future__ import annotations

from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface

_README_NAME = "README.md"


def _resolve_readme(project_root: Path) -> Path:
    candidates = (
        project_root / _README_NAME,
        project_root.parent.parent / _README_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise OSError(f"Readme file does not exist: {_README_NAME}")


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        readme_path = _resolve_readme(Path(self.root))
        metadata["readme"] = {
            "text": readme_path.read_text(encoding="utf-8"),
            "content-type": "text/markdown",
        }


def get_metadata_hook() -> type[CustomMetadataHook]:
    return CustomMetadataHook

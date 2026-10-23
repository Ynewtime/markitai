"""The workbench's installed npm packages, compared with its lockfile."""
import json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "crates/markitai-cli/src/server/web"


def stale_web_dependencies(web=WEB):
    """What `node_modules` lacks or holds at another version than package-lock.json.

    Optional packages may be absent: they are the other platforms' builds
    (esbuild ships one per system). An empty list means `npm ci` has nothing to do.
    """
    web = Path(web)
    lock = json.loads((web / "package-lock.json").read_text(encoding="utf-8"))
    problems = []
    for key, entry in lock["packages"].items():
        if not key or entry.get("link"):
            continue
        name = key.rsplit("node_modules/", 1)[-1]
        try:
            installed = json.loads((web / key / "package.json").read_text(encoding="utf-8")).get("version")
        except FileNotFoundError:
            if not entry.get("optional"):
                problems.append(f"{name} is not installed")
            continue
        except (OSError, ValueError):
            installed = None
        if installed != entry.get("version"):
            problems.append(f"{name} {installed or '(unreadable)'} is installed, the lockfile has {entry.get('version')}")
    return problems

"""Reject personal machine paths and common secret formats without echoing them."""
from pathlib import Path
import re
import subprocess


# Synthetic names are allowed for path/escaping tests. Never add real identities.
SYNTHETIC_USERS = {
    "a", "a%20b", "alice", "bob", "demo", "example", "fixture", "isolated",
    "markitai", "private", "public", "shared", "test", "test-user", "tester",
    "example-user", "models", "serve", "me", "runneradmin", "ci",
    "u", "user", "username", "...", "…",
}
HOME_PATHS = re.compile(
    rb"(?:/" + rb"Users/|/" + rb"home/|[A-Za-z]:[\\/]+Users[\\/]+)"
    rb"([^\s/\\\"'<>`]+)"
)
TEMP_PATHS = re.compile(rb"/(?:private/)?var/" + rb"folders/[a-zA-Z0-9_-]{2}/[a-zA-Z0-9_-]{8,}")
SECRETS = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE" + rb" KEY-----"
    rb"|(?<![a-zA-Z0-9])(?:gh[pousr]_[a-zA-Z0-9]{30,}"
    rb"|github_pat_[a-zA-Z0-9_]{30,}|sk-[a-zA-Z0-9_-]{24,}|AKIA[A-Z0-9]{16})"
)


def findings(data):
    """Return only category and line, never the matched private value."""
    result = []
    for match in HOME_PATHS.finditer(data):
        name = match[1].decode("utf-8", errors="replace").lower()
        if name not in SYNTHETIC_USERS:
            result.append(("personal home path", data[:match.start()].count(b"\n") + 1))
    for label, pattern in (("personal temporary path", TEMP_PATHS), ("secret-shaped value", SECRETS)):
        result.extend((label, data[:match.start()].count(b"\n") + 1) for match in pattern.finditer(data))
    return result


def check_repository(root):
    root = Path(root)
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    result = []
    for name in filter(None, names):
        path = root / name
        if path.is_file() and not path.is_symlink():
            result.extend((name, line, category) for category, line in findings(path.read_bytes()))
    return result


if __name__ == "__main__":
    problems = check_repository(Path(__file__).resolve().parents[1])
    for path, line, category in problems:
        print(f"{path}:{line}: {category}")
    raise SystemExit(bool(problems))
